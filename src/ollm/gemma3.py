# gemma3-27B
import time, os, math, json
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder, file_get_contents
from .kvcache import oCache

#global vars
loader, stats = None, None

#======== rewriting core classes ==============
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3DecoderLayer, Gemma3Config, Gemma3Model, Gemma3TextModel, Gemma3ForCausalLM, Gemma3RMSNorm, create_sliding_window_causal_mask, create_causal_mask, repeat_kv, TransformersKwargs, Cache, BaseModelOutputWithPast, Gemma3ModelOutputWithPast

class loaderLayer: #gemma3 specific
	def _load_layer_weights(self):
		t1 = time.perf_counter()
		base = f"language_model.model.layers.{self.layer_idx}."
		loader.preload_layer_safetensors(base)
		d = loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if stats: stats.set("layer_load", t1)
			
	def _unload_layer_weights(self):
		base = f"language_model.model.layers.{self.layer_idx}."
		for attr_path in loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)


class MyGemma3MLP(Gemma3MLP, loaderLayer):
	def forward(self, x):		
		out = super().forward(x)		
		return out
		

class MyGemma3DecoderLayer(Gemma3DecoderLayer, loaderLayer):
	def __init__(self, config, layer_idx):
		super().__init__(config, layer_idx)
		self.layer_idx = layer_idx
		self.mlp.layer_idx = layer_idx

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out
	

class MyGemma3TextModel(Gemma3TextModel):
	def __init__(self, config: Gemma3Config):
		super().__init__(config)
		self.config = config
		self.layers = nn.ModuleList()
		for layer_idx in range(config.num_hidden_layers):
			self.layers.append(MyGemma3DecoderLayer(config, layer_idx))
			self.layers[-1]._unload_layer_weights()				

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		**kwargs: Unpack[TransformersKwargs],
	) -> BaseModelOutputWithPast:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if self.gradient_checkpointing and self.training and use_cache:
			logger.warning_once(
				"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
			)
			use_cache = False

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		if use_cache and past_key_values is None and not self.training:
			past_key_values = DynamicCache(config=self.config)

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens,
				past_seen_tokens + inputs_embeds.shape[1],
				device=inputs_embeds.device,
			)

		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		# It may already have been prepared by e.g. `generate`
		if not isinstance(causal_mask_mapping := attention_mask, dict):
			# Prepare mask arguments
			mask_kwargs = {
				"config": self.config,
				"input_embeds": inputs_embeds,
				"attention_mask": attention_mask,
				"cache_position": cache_position,
				"past_key_values": past_key_values,
				"position_ids": position_ids,
			}
			sliding_mask_kwargs = mask_kwargs.copy()

			if self.config.use_bidirectional_attention:
				mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
				sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(self.config.sliding_window)

			# Create the masks
			causal_mask_mapping = {
				"full_attention": create_causal_mask(**mask_kwargs),
				"sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
			}

		# embed positions
		hidden_states = inputs_embeds

		# create position embeddings to be shared across the decoder layers
		position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
		position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None

		#=======================================================			
		for decoder_layer in self.layers[: self.config.num_hidden_layers]:
			layer_outputs = decoder_layer(
				hidden_states,
				position_embeddings_global=position_embeddings_global,
				position_embeddings_local=position_embeddings_local,
				attention_mask=causal_mask_mapping[decoder_layer.attention_type],
				position_ids=position_ids,
				past_key_values=past_key_values,
				output_attentions=output_attentions,
				use_cache=use_cache,
				cache_position=cache_position,
				**kwargs,
			)

			hidden_states = layer_outputs[0]
					
		hidden_states = self.norm(hidden_states)
		#self.embed_tokens.to(hidden_states.device); self.parent_lm_head.to(hidden_states.device)
		if stats: print("./gemma3.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		#=======================================================

		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
		)


import transformers.models.gemma3.modeling_gemma3 as modeling
#modeling.Gemma3MLP = MyGemma3MLP
modeling.Gemma3TextModel = MyGemma3TextModel
#===============================================


class MyGemma3ForCausalLM(Gemma3ForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():			
			return super().generate(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		print("offloading layers to CPU...")
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			base = f"language_model.model.layers.{layer_idx}."
			loader.preload_layer_safetensors(base)
			loader.offload_dict_to_gpu_cpu(base, gpu=False)
		#import gc; gc.collect()
		print("finished offloading layers to CPU")