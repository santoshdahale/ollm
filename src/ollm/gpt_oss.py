# efficiant gpt-oss-20B that runs on consumer PC with 8GB VRAM

import time, os, math
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from transformers import GptOssForCausalLM, AutoTokenizer, AutoModelForCausalLM
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder
from .gds_loader import GDSWeights
#from .attention import online_chunked_grouped_attention_rope_no_mask as chunked_attention

#global vars
loader, stats = None, None

#======== rewriting core classess ==============
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention, GptOssModel, GptOssConfig, GptOssDecoderLayer, create_causal_mask, create_sliding_window_causal_mask, repeat_kv, MoeModelOutputWithPast

class MyGptOssAttention(GptOssAttention):
	def forward(self, *args, **kwargs):
		out = super().forward(*args, **kwargs)
		#print(self.layer_idx, "attention:", out[0].shape)
		return out		


class oDecoderLayer:
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


class MyGptOssDecoderLayer(GptOssDecoderLayer, oDecoderLayer):
	def __init__(self, config: GptOssConfig, layer_idx: int):
		super().__init__(config, layer_idx)	
		self.layer_idx = layer_idx

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

		# ===== meine ========================
		self.embed_tokens.cpu(); self.parent_lm_head.cpu()

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
		hidden_states = self.norm(hidden_states)
		self.embed_tokens.to(hidden_states.device); self.parent_lm_head.to(hidden_states.device)
		#./===================================
		
		return MoeModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
		)
		

def my_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,          # (B, Hq, Tq, D)  -- RoPE already applied upstream
    key: torch.Tensor,            # (B, Hkv, Tk, D) -- RoPE already applied upstream
    value: torch.Tensor,          # (B, Hkv, Tk, D)
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    q_block_size: int = 64,
    k_block_size: int = 512,
    return_attn_weights: bool = False,  # if True, falls back to dense path to produce weights
    eps: float = 1e-12,
    **kwargs,
):
    """
    Memory-efficient attention with chunked GEMM + online softmax merging.
    Matches your original semantics:
      - repeat_kv() to expand KV heads to Hq
      - optional attention_mask added to logits
      - 'sinks' logits are concatenated to keys for normalization, then dropped
      - dropout applied to probabilities (not renormalized), exactly like F.dropout(scores)
    Notes:
      * For very long sequences, returning full attn_weights is impractical; default is None.
      * RoPE/position_ids should be handled before calling this (as in your snippet).
    """
    device = query.device
    dtype_in = query.dtype
    B, Hq, Tq, D = query.shape

    # Fast path (optional): if user explicitly wants attention weights, do the dense version.
    if return_attn_weights:
        key_states   = repeat_kv(key, module.num_key_value_groups)      # (B, Hq, Tk, D)
        value_states = repeat_kv(value, module.num_key_value_groups)    # (B, Hq, Tk, D)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # (B,Hq,Tq,Tk)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # sinks
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(B, Hq, Tq, -1)  # (B,Hq,Tq,S)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-sinks.shape[-1]]  # drop sink columns
        scores = F.dropout(scores, p=dropout, training=module.training)
        attn_output = torch.matmul(scores, value_states)                 # (B,Hq,Tq,D)
        return attn_output.transpose(1, 2).contiguous(), scores

    # --------- Streaming / Chunked path ----------
    # Repeat KV to match query heads (GQA)
    key_states   = repeat_kv(key,   module.num_key_value_groups).to(torch.float32)   # (B,Hq,Tk,D)
    value_states = repeat_kv(value, module.num_key_value_groups).to(torch.float32)   # (B,Hq,Tk,D)
    query_states = query.to(torch.float32)                                           # (B,Hq,Tq,D)

    B, Hq, Tk, _ = key_states.shape
    scale = float(scaling)

    # Accumulators for online softmax across key-blocks
    # m: running max per (B,Hq,Tq), s: running sum of exp in 'm'-frame
    m  = torch.full((B, Hq, Tq), float("-inf"), device=device, dtype=torch.float32)
    s  = torch.zeros((B, Hq, Tq), device=device, dtype=torch.float32)
    wv = torch.zeros((B, Hq, Tq, D), device=device, dtype=torch.float32)

    # Iterate over key blocks
    for k_start in range(0, Tk, k_block_size):
        k_end = min(Tk, k_start + k_block_size)
        k_blk = key_states[:, :, k_start:k_end, :]    # (B,Hq,Bk,D)
        v_blk = value_states[:, :, k_start:k_end, :]  # (B,Hq,Bk,D)

        # Optional mask slice, same semantics as your code
        if attention_mask is not None:
            # attention_mask shape is broadcastable to (B,Hq,Tq,Tk)
            mask_blk = attention_mask[:, :, :, k_start:k_end]           # (B,H?,Tq,Bk)
        else:
            mask_blk = None

        # Process query in blocks for cache-friendliness
        for q_start in range(0, Tq, q_block_size):
            q_end = min(Tq, q_start + q_block_size)
            q_blk = query_states[:, :, q_start:q_end, :]                # (B,Hq,Bq,D)

            # scores = (B,Hq,Bq,Bk)
            scores = torch.einsum("b h q d, b h k d -> b h q k", q_blk, k_blk) * scale

            if mask_blk is not None:
                # Add mask (typically 0 or -inf); broadcasting over Hq if mask is (B,1,Tq,Bk)
                scores = scores + mask_blk[:, :scores.shape[1], q_start:q_end, :]

            # Local max over k for numerical stability
            local_max = torch.amax(scores, dim=-1)                      # (B,Hq,Bq)

            # exp(scores - local_max)
            exp_scores = torch.exp(scores - local_max.unsqueeze(-1))    # (B,Hq,Bq,Bk)

            # Dropout on probabilities (exactly like your original): applied BEFORE V matmul,
            # not renormalized (F.dropout semantics).
            if module.training and dropout > 0.0:
                keep_p = 1.0 - dropout
                # same shape as exp_scores; generate mask on the same device/dtype float32
                drop_mask = (torch.rand_like(exp_scores) < keep_p).to(exp_scores.dtype) / keep_p
                exp_scores = exp_scores * drop_mask
                del drop_mask

            sum_exp = exp_scores.sum(dim=-1)                            # (B,Hq,Bq)
            weighted_v_chunk = torch.einsum("b h q k, b h k d -> b h q d", exp_scores, v_blk)  # (B,Hq,Bq,D)

            # Merge with global accumulators (log-sum-exp merge)
            first = torch.isinf(m[:, :, q_start:q_end])                 # (B,Hq,Bq)
            if first.any():
                idx = first
                m[:, :, q_start:q_end][idx]  = local_max[idx]
                s[:, :, q_start:q_end][idx]  = sum_exp[idx]
                wv[:, :, q_start:q_end][idx] = weighted_v_chunk[idx]

            merge = ~first
            if merge.any():
                m_old = m[:, :, q_start:q_end][merge]                   # (N,)
                s_old = s[:, :, q_start:q_end][merge]                   # (N,)
                wv_old = wv[:, :, q_start:q_end][merge]                 # (N,D)

                lm_new = local_max[merge]                               # (N,)
                se_new = sum_exp[merge]                                 # (N,)
                wv_new = weighted_v_chunk[merge]                        # (N,D)

                new_m  = torch.maximum(m_old, lm_new)
                alpha  = torch.exp(m_old - new_m)
                beta   = torch.exp(lm_new - new_m)

                s[:, :, q_start:q_end][merge]  = s_old * alpha + se_new * beta
                wv[:, :, q_start:q_end][merge] = wv_old * alpha.unsqueeze(-1) + wv_new * beta.unsqueeze(-1)
                m[:, :, q_start:q_end][merge]  = new_m

            # free temps
            del scores, local_max, exp_scores, sum_exp, weighted_v_chunk, q_blk

        del k_blk, v_blk, mask_blk

    # ---- Merge the SINK logits (no V contribution) ----
    # Your original concatenates sink logits to the last dim, subtracts row-max,
    # softmax over keys+sink, then drops sink columns before matmul with V.
    # Equivalent effect: merge sinks into the (m,s) normalizer, leave wv untouched.
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(B, Hq, Tq, -1).to(torch.float32)  # (B,Hq,Tq,S)
    # Row-wise max over sink columns
    sink_max = torch.amax(sinks, dim=-1)                     # (B,Hq,Tq)
    sink_sumexp = torch.exp(sinks - sink_max.unsqueeze(-1)).sum(dim=-1)  # (B,Hq,Tq)

    # Merge sinks into (m,s); wv has no sink contribution
    new_m  = torch.maximum(m, sink_max)
    alpha  = torch.exp(m - new_m)        # scales existing sums in new frame
    beta   = torch.exp(sink_max - new_m) # scales sink sums in new frame
    s = s * alpha + sink_sumexp * beta
    m = new_m
    del sinks, sink_max, sink_sumexp, new_m, alpha, beta

    # Final normalize: out = wv / s
    attn_out = (wv / (s.unsqueeze(-1) + eps)).to(dtype_in)   # (B,Hq,Tq,D)

    # Match your return layout: transpose to (B, Tq, Hq, D)
    attn_out = attn_out.transpose(1, 2).contiguous()

    # We do not materialize attn_weights in streaming mode.
    return attn_out, None



import transformers.models.gpt_oss.modeling_gpt_oss as modeling
modeling.GptOssAttention = MyGptOssAttention
modeling.GptOssModel = MyGptOssModel
#modeling.eager_attention_forward = my_eager_attention_forward
#===============================================


class MyGptOssForCausalLM(GptOssForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		layer = oDecoderLayer()
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			layer.layer_idx = layer_idx
			for manifest_name, attr_path in layer._get_my_manifests():
				loader.offload_param_to_cpu(manifest_name)
		print("./gpt_oss offloading layers to CPU. Done.")
