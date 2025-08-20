# efficiant Llama that run on consumer PC
# venv: US1-asr3.12
import json, time, os, shutil
from datetime import datetime
import cupy as cp #need to be moved from here
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, LlamaConfig
from transformers import Cache, QuantizedCacheConfig, OffloadedCache, HQQQuantizedCache, QuantoQuantizedCache, QuantizedCache, DynamicCache

from utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder, remove_layers_weights, file_get_contents
from gds_loader import GDSWeights
from attention import online_chunked_grouped_attention_rope_no_mask as chunked_attention

#import torch.multiprocessing as mp
#mp.set_start_method("fork") 
import threading

#======== rewriting core classes tested on transformers==4.52.3 ============== 
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

class MyLlamaAttention(LlamaAttention):
	def forward(
		self,
		hidden_states: torch.Tensor,
		position_embeddings: Tuple[torch.Tensor, torch.Tensor],
		attention_mask: Optional[torch.Tensor],
		past_key_value: Optional = None, #[Cache]
		cache_position: Optional[torch.LongTensor] = None,
		**kwargs: Optional,  #Unpack[FlashAttentionKwargs], #kwargs={'position_ids': tensor([[44]], device='cuda:0'), 'output_attentions': False, 'use_cache': True}
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		input_shape = hidden_states.shape[:-1]
		hidden_shape = (*input_shape, -1, self.head_dim)

		query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
		key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
		value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)        
		#print(query_states.shape, key_states.shape, value_states.shape); exit()

		cos, sin = position_embeddings
		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

		if past_key_value is not None:
			# sin and cos are specific to RoPE models; cache_position needed for the static cache
			cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
			key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

		#===

		attn_output1 = chunked_attention(query_states, key_states, value_states, position_ids=kwargs["position_ids"]).transpose(1, 2) #transpose?
		attn_weights = None
		"""
		attention_interface: Callable = eager_attention_forward
		attn_output, attn_weights = attention_interface(
			self,
			query_states,
			key_states,
			value_states,
			attention_mask,
			dropout=0.0 if not self.training else self.attention_dropout,
			scaling=self.scaling,
			**kwargs,
		)
		"""
		#print(attn_output1.shape, attn_output.shape)
		#print("Error:", (attn_output1 - attn_output).abs().max().item())
		attn_output = attn_output1
		#===
		attn_output = attn_output.reshape(*input_shape, -1).contiguous()
		attn_output = self.o_proj(attn_output)
		return attn_output, attn_weights



class MyLlamaDecoderLayer(LlamaDecoderLayer):
	def __init__(self, config: LlamaConfig, layer_idx: int):
		self.layer_idx = layer_idx
		super().__init__(config, layer_idx)

	def _layer_param_manifest_names(self) -> dict:
		"""
		Return a mapping of local attribute paths -> manifest names in your storage.
		Adjust the keys/names for your model's param naming convention.
		Example keys are relative to `self` (not full HF names).
		"""
		base = f"model.layers.{self.layer_idx}"
		return {
			"self_attn.q_proj.weight": f"{base}.self_attn.q_proj.weight",
			"self_attn.k_proj.weight": f"{base}.self_attn.k_proj.weight",
			"self_attn.v_proj.weight": f"{base}.self_attn.v_proj.weight",
			"self_attn.o_proj.weight": f"{base}.self_attn.o_proj.weight",
			"mlp.gate_proj.weight":  f"{base}.mlp.gate_proj.weight",
			"mlp.up_proj.weight":    f"{base}.mlp.up_proj.weight",
			"mlp.down_proj.weight":  f"{base}.mlp.down_proj.weight",
			"input_layernorm.weight":  f"{base}.input_layernorm.weight",
			"post_attention_layernorm.weight": f"{base}.post_attention_layernorm.weight",
			# add any biases or additional params you need
		}    

	def _load_layer_weights(self): #0.026 seconds to join on 1B
		"""
		loader.load_tensor(manifest_name) -> torch.Tensor (on CUDA recommended)
		This function will iterate the manifest map and assign weights into submodules.
		"""
		manifest_map = self._layer_param_manifest_names()
		for attr_path, manifest_name in manifest_map.items():
			try:
				# 1) load tensor from loader
				#tensor = loader.load_tensor(manifest_name)  # MUST return a torch.Tensor ideally on CUDA
				tensor = loader.load_param_to_cuda(manifest_name)

				# 2) assign into local module
				parent, leaf = _walk_to_parent(self, attr_path)
				_assign_tensor_to_module(parent, leaf, tensor)
			except Exception as e:
				# Be explicit about failures so you can debug missing names
				raise RuntimeError(f"failed to load {manifest_name} into {attr_path}: {e}")

		# optionally synchronize if your loader uses async DMA
		#if torch.cuda.is_available(): torch.cuda.synchronize()

	
	def _load_layer_weights2(self, manifest): #0.038seconds to join on 1B
		#torch.cuda.set_device(device_id)		
		manifest_map = self._layer_param_manifest_names()
		for attr_path, manifest_name in manifest_map.items():
			try:
				attr = manifest[manifest_name]
				x = np.fromfile(attr["path"], dtype=cp.float16).reshape(attr["shape"])
				tensor = torch.from_numpy(x)
				tensor = tensor.to("cuda")
				parent, leaf = _walk_to_parent(self, attr_path)
				_assign_tensor_to_module(parent, leaf, tensor)
			except Exception as e:
				# Be explicit about failures so you can debug missing names
				raise RuntimeError(f"failed to load {manifest_name} into {attr_path}: {e}")

		# optionally synchronize if your loader uses async DMA
		#if torch.cuda.is_available(): torch.cuda.synchronize()


	def _unload_layer_weights(self):
		"""Replace each loaded attribute with a meta placeholder to free GPU memory."""
		manifest_map = self._layer_param_manifest_names()
		for attr_path in manifest_map.keys():
			parent, leaf = _walk_to_parent(self, attr_path)
			# replace with placeholder (keeps module graph intact)
			_set_meta_placeholder(parent, leaf)
	

	def forward(self, *args, **kwargs):
		#self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out


class MyLlamaModel(LlamaModel):
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		**flash_attn_kwargs: Optional #Unpack[FlashAttentionKwargs],
	) -> BaseModelOutputWithPast:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache

		if (input_ids is None) ^ (inputs_embeds is not None): raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if not isinstance(past_key_values, (type(None), Cache)): raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

		if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)

		if use_cache and past_key_values is None: past_key_values = DynamicCache()

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)

		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		causal_mask = self._update_causal_mask(
			attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
		)

		hidden_states = inputs_embeds

		# create position embeddings to be shared across the decoder layers
		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None

		#=== meine        
		for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
			p1 = None
			if layer_idx+1 < len(self.layers):
				#self.layers[layer_idx+1]._load_layer_weights2(loader.manifest) #primitive
				p1 = threading.Thread(target=self.layers[layer_idx+1]._load_layer_weights, args=()) #loader.manifest,
				p1.start()

			if layer_idx==0: decoder_layer._load_layer_weights()
			layer_outputs = decoder_layer(
				hidden_states,
				attention_mask=causal_mask,
				position_ids=position_ids,
				past_key_value=past_key_values,
				output_attentions=output_attentions,
				use_cache=use_cache,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**flash_attn_kwargs,
			)
			hidden_states = layer_outputs[0]
			
			t1 = time.perf_counter()
			if p1 is not None: p1.join()
			print(layer_idx, "p1. join wait s:", time.perf_counter() - t1)
		#================

		hidden_states = self.norm(hidden_states)

		print("Llama forward finished2", datetime.now()) #meine1
		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values if use_cache else None,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
		)

# Monkey-patch
import transformers.models.llama.modeling_llama as llama_modeling
llama_modeling.LlamaAttention = MyLlamaAttention
llama_modeling.LlamaDecoderLayer = MyLlamaDecoderLayer
llama_modeling.LlamaModel = MyLlamaModel
#===============================================

class MyKVCache(DynamicCache):	
	def __init__(self, layers_num):
		super().__init__()
		self.cache_folder = "./kv_cache"
		if os.path.exists(self.cache_folder): shutil.rmtree(self.cache_folder)
		os.makedirs(self.cache_folder)

	def update(
		self,
		key_states: torch.Tensor,
		value_states: torch.Tensor,
		layer_idx: int,
		cache_kwargs: Optional[Dict[str, Any]] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		#print("MyKVCache", layer_idx, key_states.shape, value_states.shape, cache_kwargs)		
		tensors = self.load_from_disk(layer_idx)
		if tensors is not None:
			self.key_cache[layer_idx], self.value_cache[layer_idx] = tensors
		out = super().update(key_states, value_states, layer_idx, cache_kwargs) #tuple of (self.key_cache[layer_idx], self.value_cache[layer_idx])
		self.save_to_disk(out, layer_idx)
		self.key_cache[layer_idx], self.value_cache[layer_idx] = torch.empty(0), torch.empty(0)
		return out

	def load_from_disk(self, layer_idx, device="cuda:0"):
		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		if not os.path.exists(path): return None
		tensors = torch.load(path, map_location=device)
		return tensors

	def save_to_disk(self, tensors, layer_idx):
		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		tensors = (tensors[0].cpu(), tensors[1].cpu())
		torch.save(tensors, path)


class MyLlamaForCausalLM(LlamaForCausalLM):
	def __init__(self, config):
		super().__init__(config)
	
	def clean_layers_weights(self, device="cpu"):
		manifest_map = loader.manifest
		for name, v in manifest_map.items():
			if name.startswith("model.layers."):				
				tensor = torch.empty([0], device="cpu")
				#tensor = torch.empty(v["shape"], dtype=torch.float8_e4m3fn, device="cpu") #[0]
				parent, leaf = _walk_to_parent(self, name)
				_assign_tensor_to_module(parent, leaf, tensor)

	def load_nonlayer_weights(self):
		manifest_map = loader.manifest
		for name, v in manifest_map.items():
			if name.startswith("model.layers."): continue
			tensor = loader.load_param_to_cuda(name)
			parent, leaf = _walk_to_parent(self, name)
			_assign_tensor_to_module(parent, leaf, tensor)
			print("setting 0:", name)
	if torch.cuda.is_available(): torch.cuda.synchronize()


def inference_chat():
	sm, um = "You are helpful AI assistant", "List planets starting from Mercury"
	#sm, um = file_get_contents("./temp/landing_gprompt.txt"), "What can you do?"
	messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, return_tensors="pt").to(device)
	with torch.no_grad():
		#cache_config = QuantizedCacheConfig(nbits=4, axis_key=1, axis_value=1)
		past_key_values = DynamicCache() #MyKVCache(len(model.model.layers)) #HQQQuantizedCache
		print("\n\nGenerate starting", datetime.now())
		outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, past_key_values=past_key_values, use_cache=True).detach().cpu()
		answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
		print(answer)


#==============================================================================================
if __name__ == "__main__":
	device = torch.device("cuda")
	loader = GDSWeights("./gds_export/manifest.json")
	if 1==0: #prepare model without layers weights
		model_id = "meta-llama/Llama-3.2-1B-Instruct" #"meta-llama/Meta-Llama-3-8B-Instruct"
		tokenizer = AutoTokenizer.from_pretrained(model_id)
		tokenizer.pad_token = tokenizer.eos_token		
		model = MyLlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True) #cpu | meta, dtype=should be bfloat16
		model.clean_layers_weights()
		model.save_pretrained("./models/llama3-1B") #saving model without layers weights
		tokenizer.save_pretrained("./models/llama3-1B"); exit()
	
	elif 2==2: #modeling_utils setting _initialize_weights do nothing maybe helpful		
		model_id = "./models/llama3-8B/"
		print("loading model:", model_id)
		tokenizer = AutoTokenizer.from_pretrained(model_id)
		model = MyLlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		model.clean_layers_weights()
		model.eval()
		#print("model -> cuda"); time.sleep(20)
		model.cuda()
	
	else:
		def load_with_ignore_mismatched(model, state_dict):
			model_state = model.state_dict()
			filtered_state = {}

			for k, v in state_dict.items():
				if k not in model_state:
					# key missing in model → skip
					continue
				if v.shape != model_state[k].shape:
					# shape mismatch → skip
					print(f"Skipping {k}, checkpoint shape {v.shape}, model shape {model_state[k].shape}")
					continue
				filtered_state[k] = v

			# load filtered dict
			missing, unexpected = model.load_state_dict(filtered_state, strict=False)
			print("Missing keys:", missing)
			print("Unexpected keys:", unexpected)

		from accelerate import init_empty_weights
		from safetensors.torch import load_file
		model_id = "./models/llama3-8B"
		tokenizer = AutoTokenizer.from_pretrained(model_id)
		with init_empty_weights():
			model = MyLlamaForCausalLM(AutoConfig.from_pretrained("./models/llama3-8B/"))
		state_dict = load_file("./models/llama3-8B/model.safetensors", device="cuda:0")  # or "cpu"		
		print("./stats loaded")
		#model.load_state_dict(state_dict, strict=False)
		load_with_ignore_mismatched(model, state_dict)
		#model.cuda()

	inference_chat()