# 4.57.0.dev qwen3_next

import time, os, math, json
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder, file_get_contents

#global vars
loader, stats = None, None

#======== rewriting core classes ==============
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextMLP, Qwen3NextSparseMoeBlock, Qwen3NextDecoderLayer, Qwen3NextConfig, Qwen3NextModel, Qwen3NextForCausalLM, Qwen3NextDynamicCache, Qwen3NextRMSNorm, create_causal_mask, repeat_kv, MoeModelOutputWithPast, MoeCausalLMOutputWithPast, TransformersKwargs, Cache

class loaderLayer:
	def _load_layer_weights(self):
		t1 = time.perf_counter()
		base = f"model.layers.{self.layer_idx}."
		#loader.preload_layer_safetensors(base)
		d = loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if stats: stats.set("layer_load", t1)
			
	def _unload_layer_weights(self): #v1 with separate files for each base
		base = f"model.layers.{self.layer_idx}."
		for attr_path in loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)
	
	def _unload_layer_weights2(self):
		base = f"model.layers.{self.layer_idx}."
		for base1 in list(loader.manifest.keys()):
			if base1.startswith(base):
				for attr_path, filename in loader.manifest[base1].items():
					parent, leaf = _walk_to_parent(self, base1[len(base):]+attr_path)
					_set_meta_placeholder(parent, leaf)

	def _load_expert_weights(self):
		t1 = time.perf_counter()
		base = f"model.layers.{self.layer_idx}.mlp.experts.{self.expert_idx}."
		d = loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if stats: stats.set("experts_load", t1)

	def _unload_expert_weights(self):
		base = f"model.layers.{self.layer_idx}.mlp.experts.{self.expert_idx}."
		for attr_path in loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)

	def _load_experts_weights(self, experts_idx):
		for expert_idx in experts_idx:
			self.experts[expert_idx]._load_expert_weights()


class MyQwen3NextMLP(Qwen3NextMLP, loaderLayer):
	def forward(self, x):
		#if hasattr(self, "expert_idx"): self._load_expert_weights()
		out = super().forward(x)
		if hasattr(self, "expert_idx"): self._unload_expert_weights()
		return out
		

class MyQwen3NextSparseMoeBlock(Qwen3NextSparseMoeBlock, loaderLayer):

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		batch_size, sequence_length, hidden_dim = hidden_states.shape
		hidden_states = hidden_states.view(-1, hidden_dim)
		# router_logits: (batch * sequence_length, n_experts)
		router_logits = self.gate(hidden_states)

		routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
		routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
		if self.norm_topk_prob:
			routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
		# we cast back to the input dtype
		routing_weights = routing_weights.to(hidden_states.dtype)

		final_hidden_states = torch.zeros(
			(batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
		)

		# One hot encode the selected experts to create an expert mask
		# this will be used to easily index which expert is going to be sollicitated
		expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

		# Loop over all available experts in the model and perform the computation on each expert
		expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()[:20]
		self._load_experts_weights(expert_hit.cpu().squeeze().tolist()) #meine
		
		#=========
		"""
		for expert_idx in expert_hit:
			expert_layer = self.experts[expert_idx]
			idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
			current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
			current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
			final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))	
		"""

		# assume experts all share dimensions:
		# in_dim -> hidden_dim -> out_dim
		token_pos_list, expert_id_list, local_rank_list = [], [], []

		for expert_idx in expert_hit:
			tok_pos, local_rank = torch.where(expert_mask[expert_idx].squeeze(0))
			if tok_pos.numel() == 0: continue
			if local_rank.numel() > 0: assert local_rank.max() < hidden_states.size(0), (f"Index {local_rank.max().item()} >= hidden_states.size(0)={hidden_states.size(0)}")
			assert tok_pos.max() < routing_weights.size(1), (f"Index {tok_pos.max().item()} >= routing_weights.size(1)={routing_weights.size(1)}")

			token_pos_list.append(tok_pos)
			expert_id_list.append(torch.full_like(tok_pos, int(expert_idx)))
			local_rank_list.append(local_rank)
		
		token_pos = torch.cat(token_pos_list)      # (M,)
		expert_ids = torch.cat(expert_id_list)     # (M,)
		local_rank = torch.cat(local_rank_list)    # (M,)

		x = hidden_states[token_pos]               # (M, in_dim)
		M, in_dim = x.shape
		device, dtype = x.device, x.dtype

		# ---- Gather expert params as stacked tensors ----
		def stack_params(attr):
			return torch.stack([getattr(self.experts[eid], attr).weight for eid in expert_ids], dim=0)

		W_gate = stack_params("gate_proj")     # (E, hidden, in), (E, hidden)
		W_up   = stack_params("up_proj")       # (E, hidden, in), (E, hidden)
		W_down = stack_params("down_proj")     # (E, out, hidden), (E, out)

		# ---- Layer by layer ----
		x_un = x.unsqueeze(-1)                                # (M, in, 1)
		
		# gate_proj
		h1 = torch.bmm(W_gate, x_un).squeeze(-1)     # (M, hidden)
		# up_proj
		h2 = torch.bmm(W_up,   x_un).squeeze(-1)     # (M, hidden)

		# elementwise gated activation
		h = self.experts[expert_ids[0]].act_fn(h1) * h2 # (M, hidden)

		# down_proj
		h_un = h.unsqueeze(-1)                       # (M, hidden, 1)
		out = torch.bmm(W_down, h_un).squeeze(-1)    # (M, out)

		# routing weights
		rw = routing_weights[token_pos, local_rank].unsqueeze(-1).to(out.dtype)
		out = out * rw

		# scatter back
		final_hidden_states.index_add_(0, token_pos.to(final_hidden_states.device), out.to(final_hidden_states.dtype))
		print(expert_hit.shape, "x:", x.shape, x_un.shape, "W_gate:", W_gate.shape, "W_up:", W_up.shape, out, out.shape)
		#==============================================

		shared_expert_output = self.shared_expert(hidden_states)
		shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
		final_hidden_states = final_hidden_states + shared_expert_output
		final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
		return final_hidden_states, router_logits



class MyQwen3NextDecoderLayer(Qwen3NextDecoderLayer, loaderLayer):
	def __init__(self, config, layer_idx):
		super().__init__(config, layer_idx)
		self.layer_idx = layer_idx
		self.mlp.layer_idx = layer_idx
		if hasattr(self.mlp, "experts"):
			for expert_idx, expert_layer in enumerate(self.mlp.experts):
				expert_layer.expert_idx = expert_idx
				expert_layer.layer_idx = layer_idx
				expert_layer._unload_expert_weights()

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out
	

class MyQwen3NextModel(Qwen3NextModel):
	def __init__(self, config: Qwen3NextConfig):
		super().__init__(config)
		self.config = config
		self.layers = nn.ModuleList()
		for layer_idx in range(config.num_hidden_layers):
			self.layers.append(MyQwen3NextDecoderLayer(config, layer_idx))
			self.layers[-1]._unload_layer_weights()
				
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		**kwargs: Unpack[TransformersKwargs],
	) -> MoeModelOutputWithPast:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		if use_cache and past_key_values is None:
			past_key_values = Qwen3NextDynamicCache(config=self.config)

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		causal_mask = create_causal_mask(
			config=self.config,
			input_embeds=inputs_embeds,
			attention_mask=attention_mask,
			cache_position=cache_position,
			past_key_values=past_key_values,
			position_ids=position_ids,
		)
		linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

		hidden_states = inputs_embeds

		# create position embeddings to be shared across the decoder layers
		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		#===============================================
		for decoder_layer in self.layers:
			#print(decoder_layer.layer_idx, "decoder_layer /", self.config.num_hidden_layers, stats.print_and_clean())
			layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
			hidden_states = decoder_layer(
				hidden_states,
				position_embeddings=position_embeddings,
				attention_mask=layer_mask,
				position_ids=position_ids,
				past_key_values=past_key_values,
				use_cache=use_cache,
				cache_position=cache_position,
				**kwargs,
			)

		print("./qwen3_next.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		hidden_states = self.norm(hidden_states)
		#================================================

		return MoeModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
		)


import transformers.models.qwen3_next.modeling_qwen3_next as modeling
modeling.Qwen3NextMLP = MyQwen3NextMLP
modeling.Qwen3NextSparseMoeBlock = MyQwen3NextSparseMoeBlock
modeling.Qwen3NextModel = MyQwen3NextModel
#===============================================


class MyQwen3NextForCausalLM(Qwen3NextForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():			
			return super().generate(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		self.offload_layers_to_gpu_cpu(cpu_layers_num=layers_num)

	def offload_layers_to_gpu_cpu(self, gpu_layers_num=0, cpu_layers_num=0):
		print("offloading layers to CPU/GPU...")
		layer_idx = 0
		while (gpu_layers_num>0 or cpu_layers_num>0) and layer_idx < self.num_hidden_layers:
			base = f"model.layers.{layer_idx}."
			#loader.preload_layer_safetensors(base)
			if gpu_layers_num>0:
				loader.offload_dict_to_gpu_cpu(base, gpu=True)
				gpu_layers_num-=1
			else:
				loader.offload_dict_to_gpu_cpu(base, gpu=False)
				cpu_layers_num-=1
			layer_idx+=1
		#import gc; gc.collect()
		print("finished offloading layers to CPU/GPU")


#=============================================================================================
class Qwen3NextMultiTokenPredictor(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.fc = nn.Linear(config.hidden_size*2, config.hidden_size, bias=False)
		self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.pre_fc_norm_embedding = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.pre_fc_norm_hidden = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.layers = nn.ModuleList([Qwen3NextDecoderLayer(config, layer_idx) for layer_idx in range(1)])

	def forward(self, x, inputs_embeds, position_ids, position_embeddings):
		inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
		hidden_states = self.pre_fc_norm_hidden(x)
		hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
		hidden_states = self.fc(hidden_states)
		
		hidden_states = self.layers[0](
			hidden_states,
			position_embeddings=position_embeddings,
			position_ids=position_ids
		)
		hidden_states = self.norm(hidden_states)		

		return MoeModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=None #past_key_values,
		)


class MyQwen3NextForCausalLM_MTP(MyQwen3NextForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.mtp = Qwen3NextMultiTokenPredictor(config)
		self.logits2 = None
		self.past_key_values = None
		self.input_ids, self.cache_position = None, None
	
	def forward(self, **args):
		if self.logits2 is None:
			if self.input_ids: 
				#args["input_ids"] = torch.cat([args["input_ids"], self.input_ids], dim=-1)
				#args["cache_position"] = torch.cat([self.cache_position, args["cache_position"]], dim=-1)
				pass

			outputs = self.model(**args)
			hidden_states, past_key_values = outputs.last_hidden_state, outputs.past_key_values
			logits = self.lm_head(hidden_states[:, -1:, :])
			self.past_key_values = past_key_values

			#MTP
			position_ids = args["cache_position"].unsqueeze(0)			
			position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
			inputs_embeds = self.model.embed_tokens(args["input_ids"]) #double computation, can be taken from main model
			outputs = self.mtp(hidden_states, inputs_embeds, position_ids, position_embeddings)
			hidden_states = outputs.last_hidden_state
			self.logits2 = self.lm_head(hidden_states[:, -1:, :])
			#print(past_key_values, past_key_values.key_cache[0].shape if past_key_values.key_cache[0] else None, "- k shape. hs:", hidden_states.shape)
		else:
			self.input_ids, self.cache_position = args["input_ids"], args["cache_position"]
			logits, past_key_values = self.logits2, self.past_key_values
			self.logits2 = None
		
		return MoeCausalLMOutputWithPast(
			logits=logits,
			past_key_values=past_key_values
		)