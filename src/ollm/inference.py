import os
import requests
import zipfile
import torch
from transformers import AutoTokenizer, AutoProcessor
from .utils import Stats, file_get_contents
from .gds_loader import GDSWeights, MoEWeightsLoader2, Gemma3Loader
from .kvcache import KVCache
from .optimizations import (
    GPUMemoryPool, MemoryManager,
    CompressedKVCache, AdaptiveOptimizer,
    LayerPrefetcher, SpeculativeDecoder,
    StreamingInference, DynamicBatcher,
    AttentionOptimizer
)

class Inference:
	def __init__(self, model_id, device="cuda:0", logging=True, multimodality=False, enable_optimizations=True):
		self.model_id = model_id
		self.device = torch.device(device)
		self.multimodality = multimodality
		self.stats = Stats() if logging else None
		self.enable_optimizations = enable_optimizations
		
		# Initialize optimization components
		if enable_optimizations:
			self._init_optimizations()

	def download_and_unpack(self, models_dir: str):
		os.makedirs(models_dir, exist_ok=True)
		urls = {
			"llama3-1B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-1B-chat.zip",
			"llama3-3B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-3B-chat.zip",
			"llama3-8B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-8B-chat.zip",
			"gpt-oss-20B":    "https://ollm.s3.us-east-1.amazonaws.com/models/gpt-oss-20B.zip"
		}
		url = urls[self.model_id]
		
		# Extract filename from URL
		filename = url.split("/")[-1]
		zip_path = os.path.join(models_dir, filename)

		# Download the file
		print(f"Downloading {url} ...")
		response = requests.get(url, stream=True)
		response.raise_for_status()
		with open(zip_path, "wb") as f:
			for chunk in response.iter_content(chunk_size=8192):
				f.write(chunk)
		print(f"Downloaded to {zip_path}")

		# Unzip
		print(f"Unpacking {zip_path} ...")
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(models_dir)
		print(f"Unpacked to {models_dir}")

		os.remove(zip_path) # Optional: remove the zip file after extraction

	
	def hf_download(self, model_dir):
		from huggingface_hub import snapshot_download
		urls = {"qwen3-next-80B": "Qwen/Qwen3-Next-80B-A3B-Instruct", "gemma3-12B":"google/gemma-3-12b-it"}
		url = urls[self.model_id]
		print(f"Downloading {url} ...")
		snapshot_download(
		    repo_id=url,
		    local_dir=model_dir,
		    local_dir_use_symlinks=False
		)

	
	def ini_model(self, models_dir="./models/", force_download=False):
		models_list = ["llama3-1B-chat", "llama3-3B-chat", "llama3-8B-chat", "gpt-oss-20B", "qwen3-next-80B", "gemma3-12B"]
		if self.model_id not in models_list:
			raise ValueError("Incorrect model id. It must be one of", models_list)
		
		model_dir = os.path.join(models_dir, self.model_id)
		if os.path.exists(model_dir)==False or force_download==True:
			if self.model_id in ["qwen3-next-80B", "gemma3-12B"]:
				self.hf_download(model_dir)
			else:
				self.download_and_unpack(models_dir)
		
		print("loading model from", model_dir)
		if self.model_id=="qwen3-next-80B":
			from . import qwen3_next
			qwen3_next.loader = MoEWeightsLoader2(model_dir)
			qwen3_next.stats = self.stats
			self.model = qwen3_next.MyQwen3NextForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="gemma3-12B":
			from . import gemma3
			gemma3.loader = Gemma3Loader(model_dir)
			gemma3.stats = self.stats
			automodel = gemma3.MyGemma3ForConditionalGeneration if self.multimodality else gemma3.MyGemma3ForCausalLM
			self.model = automodel.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.processor = AutoProcessor.from_pretrained(model_dir)
		elif self.model_id=="gpt-oss-20B":
			from . import gpt_oss
			gpt_oss.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
			gpt_oss.stats = self.stats
			self.model = gpt_oss.MyGptOssForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			
			# Enable GPT-OSS specific optimizations if requested
			if self.enable_optimizations:
				try:
					if hasattr(self.model, 'enable_gpt_oss_optimizations'):
						self.model.enable_gpt_oss_optimizations()
						print("GPT-OSS enhanced optimizations enabled")
				except Exception as e:
					print(f"Warning: Could not enable GPT-OSS optimizations: {e}")
		else:
			from . import llama
			llama.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
			llama.stats = self.stats			
			self.model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="cpu", attn_implementation="flash_attention_2", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.model.clean_layers_weights()

		self.model.eval()
		self.model.to(self.device)
		self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
		
		# Setup optimizations after model is loaded
		if self.enable_optimizations:
			self.setup_optimizations()

	
	def offload_layers_to_cpu(self, **args):
		self.model.offload_layers_to_cpu(**args)
	
	def offload_layers_to_gpu_cpu(self, **args):
		self.model.offload_layers_to_gpu_cpu(**args)
	
	def _init_optimizations(self):
		"""Initialize optimization components"""
		self.memory_manager = MemoryManager(device=str(self.device))
		self.adaptive_optimizer = None  # Will be initialized after model loading
		self.prefetcher = None
		self.speculative_decoder = None
		self.streaming_processor = None
		self.batcher = None
		self.compressed_cache = None
	
	def setup_optimizations(self, config=None):
		"""Setup optimizations after model is loaded"""
		if not self.enable_optimizations or not hasattr(self, 'model'):
			return
		
		config = config or {}
		
		# Initialize adaptive optimizer
		self.adaptive_optimizer = AdaptiveOptimizer(
			model=self.model,
			device=str(self.device)
		)
		
		# Initialize layer prefetcher
		self.prefetcher = LayerPrefetcher(
			model=self.model,
			prefetch_distance=config.get('prefetch_distance', 2),
			device=str(self.device)
		)
		
		# Initialize compressed KV cache
		self.compressed_cache = CompressedKVCache(
			compression_method=config.get('kv_compression', 'quantization'),
			bits=config.get('compression_bits', 8)
		)
		
		# Initialize streaming processor
		self.streaming_processor = StreamingInference(
			model=self.model,
			tokenizer=self.tokenizer
		)
		
		# Initialize dynamic batcher
		self.batcher = DynamicBatcher(
			model=self.model,
			tokenizer=self.tokenizer,
			max_batch_size=config.get('max_batch_size', 8)
		)

	def DiskCache(self, cache_dir="./kvcache"):
		if self.model_id in ["gpt-oss-20B"]:
			print(f"{self.model_id} DiskCache is not supported at the moment. Using default DynamicCache instead")
			return None
		elif self.model_id=="qwen3-next-80B":
			from .qwen3_next import Qwen3NextDiskCache
			return Qwen3NextDiskCache(self.model.config, cache_dir=cache_dir, stats=self.stats)
		else:
			return KVCache(cache_dir=cache_dir, stats=self.stats) #config=?
	
	def generate_optimized(self, input_text, max_new_tokens=100, **kwargs):
		"""Generate text with all optimizations enabled"""
		if not self.enable_optimizations:
			return self._generate_standard(input_text, max_new_tokens, **kwargs)
		
		# Setup optimizations if not done yet
		if self.adaptive_optimizer is None:
			self.setup_optimizations(kwargs.get('optimization_config', {}))
		
		# Tokenize input
		input_ids = self.tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
		input_ids = input_ids.to(self.device)
		seq_len = input_ids.shape[1]
		
		# Determine optimal strategy
		strategy = kwargs.get('strategy', 'auto')
		if strategy == 'auto':
			strategy = self._choose_generation_strategy(seq_len, **kwargs)
		
		if strategy == 'speculative' and self.speculative_decoder:
			return self._generate_speculative(input_ids, max_new_tokens, **kwargs)
		elif strategy == 'streaming':
			return self._generate_streaming(input_text, max_new_tokens, **kwargs)
		else:
			return self._generate_with_optimizations(input_ids, max_new_tokens, **kwargs)
	
	def _choose_generation_strategy(self, seq_len, **kwargs):
		"""Choose optimal generation strategy based on sequence characteristics"""
		if seq_len > 8192:
			return 'streaming'
		elif seq_len < 2048 and self.speculative_decoder:
			return 'speculative' 
		else:
			return 'optimized'
	
	def _generate_standard(self, input_text, max_new_tokens, **kwargs):
		"""Standard generation without optimizations"""
		input_ids = self.tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
		input_ids = input_ids.to(self.device)
		
		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids,
				max_new_tokens=max_new_tokens,
				do_sample=kwargs.get('do_sample', True),
				temperature=kwargs.get('temperature', 0.7),
				pad_token_id=self.tokenizer.eos_token_id
			)
		
		generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
		return generated_text
	
	def _generate_with_optimizations(self, input_ids, max_new_tokens, **kwargs):
		"""Generate with memory and attention optimizations"""
		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids,
				max_new_tokens=max_new_tokens,
				past_key_values=self.compressed_cache,
				do_sample=kwargs.get('do_sample', True),
				temperature=kwargs.get('temperature', 0.7),
				pad_token_id=self.tokenizer.eos_token_id
			)
		
		generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
		return generated_text
	
	def _generate_speculative(self, input_ids, max_new_tokens, **kwargs):
		"""Generate using speculative decoding"""
		result = self.speculative_decoder.generate(
			input_ids=input_ids,
			max_new_tokens=max_new_tokens,
			temperature=kwargs.get('temperature', 0.7)
		)
		
		generated_text = self.tokenizer.decode(
			result['sequences'][0][input_ids.shape[1]:], 
			skip_special_tokens=True
		)
		return generated_text
	
	def _generate_streaming(self, input_text, max_new_tokens, **kwargs):
		"""Generate using streaming for very long inputs"""
		# For very long inputs, use chunked processing
		processor = self.streaming_processor
		result = processor.process_long_sequence(
			input_text=input_text,
			max_new_tokens=max_new_tokens
		)
		return result
	
	def get_optimization_stats(self):
		"""Get comprehensive optimization statistics"""
		if not self.enable_optimizations:
			return {"optimizations_enabled": False}
		
		stats = {"optimizations_enabled": True, "model_id": self.model_id}
		
		if self.memory_manager:
			stats["memory"] = self.memory_manager.get_memory_report()
		
		if self.adaptive_optimizer:
			stats["optimizer"] = self.adaptive_optimizer.get_performance_report()
		
		if self.prefetcher:
			stats["prefetcher"] = self.prefetcher.get_stats()
		
		if self.compressed_cache:
			stats["kv_cache"] = {
				"compression_method": self.compressed_cache.compression_method,
				"compressed_layers": len(self.compressed_cache.compressed_keys)
			}
		
		if self.batcher:
			stats["batcher"] = self.batcher.get_stats()
		
		# Model-specific optimization stats
		if self.model_id == "gpt-oss-20B" and hasattr(self.model, 'get_gpt_oss_optimization_stats'):
			try:
				stats["gpt_oss_specific"] = self.model.get_gpt_oss_optimization_stats()
			except Exception as e:
				stats["gpt_oss_specific"] = {"error": str(e)}
		
		return stats
	
	def set_draft_model(self, draft_model):
		"""Set draft model for speculative decoding"""
		if self.enable_optimizations and hasattr(self, 'model'):
			self.speculative_decoder = SpeculativeDecoder(
				main_model=self.model,
				draft_model=draft_model,
				num_candidates=4
			)
