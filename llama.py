# efficiant Llama that run on consumer PC
# venv: US1-asr3.12
import json, time
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union
from transformers import AutoTokenizer, LlamaModel, LlamaForCausalLM

from utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder, remove_layers_weights, file_get_contents
from gds_loader import GDSWeights
from attention import online_chunked_grouped_attention_rope as chunked_attention

#======== rewriting core classes tested on transformers==4.52.3 ============== 
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, LlamaAttention, LlamaDecoderLayer, LlamaConfig

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
		attn_output1 = chunked_attention(query_states, key_states, value_states).transpose(1, 2) #transpose?
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

	def _load_layer_weights(self):
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
		if torch.cuda.is_available():
			torch.cuda.synchronize()

	def _unload_layer_weights(self):
		"""Replace each loaded attribute with a meta placeholder to free GPU memory."""
		manifest_map = self._layer_param_manifest_names()
		for attr_path in manifest_map.keys():
			parent, leaf = _walk_to_parent(self, attr_path)
			# replace with placeholder (keeps module graph intact)
			_set_meta_placeholder(parent, leaf)
	

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out


# Monkey-patch
import transformers.models.llama.modeling_llama as llama_modeling
llama_modeling.LlamaAttention = MyLlamaAttention
llama_modeling.LlamaDecoderLayer = MyLlamaDecoderLayer
#===============================================


class MyLlamaForCausalLM(LlamaForCausalLM):
	def __init__(self, config):
		super().__init__(config)
	
	def load_weights(self): #emptying layers weights for now
		manifest_map = loader.manifest
		for name, v in manifest_map.items():
			if name.startswith("model.layers."):# continue
				tensor = torch.tensor(0, device=device)  #loader.load_param_to_cuda(name) #temp
				parent, leaf = _walk_to_parent(self, name)
				_assign_tensor_to_module(parent, leaf, tensor)
				#print("setting 0:", name)
		#if torch.cuda.is_available(): torch.cuda.synchronize()

def inference_chat():
	gp = file_get_contents("./temp/landing_gprompt.txt")# "You are helpful AI assistant" #
	um = "List planets"
	messages = [{"role":"system", "content":gp }, {"role":"user", "content":um}]	
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, truncation=True, max_length=4000, return_tensors="pt").to(device)
	outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False).detach().cpu()
	answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
	print(answer)


#==============================================================================================
if __name__ == "__main__":	
	loader = GDSWeights("./gds_export/manifest.json")
	device = torch.device("cuda")	
	model_id = "meta-llama/Llama-3.2-1B-Instruct" #Qwen/Qwen2-0.5B | Qwen/Qwen2-0.5B-Instruct | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | "meta-llama/Llama-3.2-1B-Instruct" | "meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	tokenizer.pad_token = tokenizer.eos_token
	model = MyLlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu") #cpu | meta
	#print(model, "\n\n", model.dtype)
	model.eval()
	model.load_weights() #ideally we need to load weights upen their need
	#remove_layers_weights(model) #not decreasing anything
	model.cuda()
	#print("model -> cuda"); time.sleep(20)	
	inference_chat()