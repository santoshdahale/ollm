# efficiant gpt-oss-20B that runs on consumer PC with 8GB VRAM

import time, os
from datetime import datetime
import threading
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack

from transformers import GptOssForCausalLM, AutoTokenizer, AutoModelForCausalLM
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder
from .gds_loader import GDSWeights
#from .attention import online_chunked_grouped_attention_rope_no_mask as chunked_attention

#global vars
loader, stats = None, None

#======== rewriting core classes (tested on transformers==4.52.3) ==============
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention, GptOssModel, GptOssConfig, GptOssDecoderLayer, create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeModelOutputWithPast

class MyGptOssAttention(GptOssAttention):
	def forward(self, *args, **kwargs):
		out = super().forward(*args, **kwargs)
		#print(self.layer_idx, "attention:", out[0].shape)
		return out		

class MyGptOssDecoderLayer(GptOssDecoderLayer):
	def __init__(self, config: GptOssConfig, layer_idx: int):
		super().__init__(config, layer_idx)	
		self.layer_idx = layer_idx
	
	def _get_my_manifests(self):
		a = []
		for manifest_name in loader.manifest.keys():
			base = f"model.layers.{self.layer_idx}."
			if not manifest_name.startswith(base): continue
			attr_path = manifest_name.replace(base, "")
			a.append((manifest_name, attr_path))
		return a

	def _load_layer_weights(self):
		for manifest_name, attr_path in self._get_my_manifests():
			try:
				t1 = time.perf_counter()
				tensor = loader.load_param_to_cuda(manifest_name)
				parent, leaf = _walk_to_parent(self, attr_path)
				_assign_tensor_to_module(parent, leaf, tensor)
				if stats: stats.set("layer_load", t1)
			except Exception as e:
				raise RuntimeError(f"failed to load {manifest_name} into {attr_path}: {e}")

	def _unload_layer_weights(self):
		for manifest_name, attr_path in self._get_my_manifests():
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out


class MyGptOssModel(GptOssModel):
	def __init__(self, config: GptOssConfig):		
		super().__init__(config)
		self.config = config
		self.layers = nn.ModuleList([MyGptOssDecoderLayer(config, layer_idx) for layer_idx in range(2)])
		for decoder_layer in self.layers: decoder_layer._unload_layer_weights()
		print("./MyGptOssModel init")

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[list[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		**kwargs: Unpack,
	) -> MoeModelOutputWithPast:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if use_cache and past_key_values is None:
			past_key_values = DynamicCache()

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		# It may already have been prepared by e.g. `generate`
		if not isinstance(causal_mask_mapping := attention_mask, dict):
			mask_kwargs = {
				"config": self.config,
				"input_embeds": inputs_embeds,
				"attention_mask": attention_mask,
				"cache_position": cache_position,
				"past_key_values": past_key_values,
			}
			causal_mask_mapping = {
				"full_attention": create_causal_mask(**mask_kwargs),
				"sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
			}

		hidden_states = inputs_embeds
		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		# === meine ========================
		for layer_idx in range(self.config.num_hidden_layers):
			decoder_layer = self.layers[layer_idx % 2]
			decoder_layer.layer_idx = layer_idx
			decoder_layer.self_attn.layer_idx = layer_idx
			hidden_states = decoder_layer(
				hidden_states,
				attention_mask=causal_mask_mapping[decoder_layer.attention_type],
				position_ids=position_ids,
				past_key_value=past_key_values,
				use_cache=use_cache,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**kwargs,
			)

		print("./gpt_oss.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")			
		#./===================================

		hidden_states = self.norm(hidden_states)
		return MoeModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
		)
		


import transformers.models.gpt_oss.modeling_gpt_oss as modeling
modeling.GptOssAttention = MyGptOssAttention
modeling.GptOssModel = MyGptOssModel
#===============================================


class MyGptOssForCausalLM(GptOssForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link

	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)		