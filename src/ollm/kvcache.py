import os, time, shutil
import torch
from transformers import Cache, DynamicCache
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List

class KVCache(DynamicCache):
	def __init__(self, cache_dir="./kv_cache", stats=None):
		super().__init__()
		self.cache_folder = os.path.join(cache_dir, "kv_cache")
		self.key_cache2, self.value_cache2 = [], []
		if os.path.exists(self.cache_folder): shutil.rmtree(self.cache_folder)
		os.makedirs(self.cache_folder)
		self.stats = stats

	def update(
		self,
		key_states: torch.Tensor,
		value_states: torch.Tensor,
		layer_idx: int,
		cache_kwargs: Optional[Dict[str, Any]] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		tensors = self.load_from_disk(layer_idx)
		if tensors is not None:
			self.key_cache[layer_idx], self.value_cache[layer_idx] = tensors
			if layer_idx < len(self.key_cache2):
				self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], self.key_cache2[layer_idx]], dim=-2)
				self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], self.value_cache2[layer_idx]], dim=-2)
				self.key_cache2[layer_idx] = torch.cat([self.key_cache2[layer_idx], key_states], dim=-2)
				self.value_cache2[layer_idx] = torch.cat([self.value_cache2[layer_idx], value_states], dim=-2)				
			else:
				self.key_cache2.append(key_states)
				self.value_cache2.append(value_states)
		
		out = super().update(key_states, value_states, layer_idx, cache_kwargs) #tuple of (self.key_cache[layer_idx], self.value_cache[layer_idx])		
		if tensors is None: self.save_to_disk(out, layer_idx) #save only first time cause it's slow to save
		self.key_cache[layer_idx], self.value_cache[layer_idx] = torch.empty(0), torch.empty(0)
		return out

	def load_from_disk(self, layer_idx, device="cuda:0"):
		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		if not os.path.exists(path): return None
		t1 = time.perf_counter()
		tensors = torch.load(path, map_location=device)
		if self.stats: self.stats.set("kvload", t1)
		return tensors

	def save_to_disk(self, tensors, layer_idx):
		t1 = time.perf_counter()
		path = f"{self.cache_folder}/layer_{layer_idx}.pt"
		tensors = (tensors[0].cpu(), tensors[1].cpu())
		torch.save(tensors, path)
		if self.stats: self.stats.set("kvsave", t1)