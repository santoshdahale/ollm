# Voxtral-12B
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
from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel, BaseModelOutputWithPast, TransformersKwargs, Cache

class loaderLayer: #Gemma3 copied
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


"""
class MyVoxtralTextModel(VoxtralTextModel):
	def __init__(self, config: VoxtralConfig):
		super().__init__(config)
		self.config = config
		self.layers = nn.ModuleList()
		for layer_idx in range(config.num_hidden_layers):
			self.layers.append(MyVoxtralDecoderLayer(config, layer_idx))
			self.layers[-1]._unload_layer_weights()

	def forward(self, **args):
		out = super().forward(**args)
		if stats: print("./Voxtral.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		return out
"""


class MyMistralModel(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList()
        print("MyMistralModel ini")
    	
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
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


import transformers.models.mistral.modeling_mistral as modeling
modeling.MistralModel = MyMistralModel

from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration
#===============================================

class oForGeneration:
	def generate(self, **args):
		with torch.no_grad():			
			return super().generate(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		print(f"offloading layers to CPU {layers_num}/{self.num_hidden_layers}...")
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			base = f"language_model.model.layers.{layer_idx}."
			loader.preload_layer_safetensors(base)
			loader.offload_dict_to_gpu_cpu(base, gpu=False)		
		print(f"./finished offloading layers to CPU {layers_num}/{self.num_hidden_layers}")


class MyVoxtralForConditionalGeneration(VoxtralForConditionalGeneration, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.num_hidden_layers = config.text_config.num_hidden_layers
		#self.language_model = None #AutoModelForCausalLM.from_config(config.text_config)


