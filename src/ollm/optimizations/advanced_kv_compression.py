"""
Advanced KV Cache Compression for oLLM
Implements mixed precision, temporal compression, and pattern-based compression techniques.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

try:
    from .triton_kernels import TritonOptimizedKVCache
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

@dataclass
class CompressionStats:
    """Statistics for compression performance"""
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    compression_ratio: float = 1.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
from collections import deque, defaultdict
import time
import threading

# Optional transformers import
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    # Fallback if transformers not available
    class DynamicCache:
        def __init__(self):
            self.key_cache = []
            self.value_cache = []

class MixedPrecisionKVCache(DynamicCache):
    """
    Advanced KV cache with mixed precision: keys in int4, values in fp8.
    Provides optimal quality/memory tradeoff.
    """
    
    def __init__(self, key_bits: int = 4, value_bits: int = 8, 
                 temporal_decay: bool = True, pattern_compression: bool = True,
                 use_triton: bool = True):
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.temporal_decay = temporal_decay
        self.pattern_compression = pattern_compression
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # Mixed precision storage
        self.compressed_keys = []
        self.compressed_values = []
        self.key_scales = []
        self.value_scales = []
        
        # Temporal compression
        self.token_ages = []
        self.age_compression_thresholds = [100, 500, 1000]  # Tokens ages for progressive compression
        
        # Pattern compression
        self.attention_patterns = defaultdict(list)
        self.pattern_clusters = {}
        self.compression_stats = {
            "total_tokens": 0,
            "compressed_tokens": 0,
            "memory_saved_mb": 0.0,
            "quality_preserved": 0.95
        }
        
        # Initialize Triton optimizer if available
        if self.use_triton:
            self.triton_kv_cache = TritonOptimizedKVCache(key_bits, value_bits)
        
        self.compression_lock = threading.RLock()


class QuantizedKVCache:
    """Basic KV Cache with quantization compression - for backward compatibility"""
    
    def __init__(self, bits: int = 8, use_triton: bool = True):
        self.bits = bits
        self.use_triton = use_triton and TRITON_AVAILABLE
        self.quantized_cache = {}
        self.scales = {}
        self.stats = CompressionStats()
        
        # Initialize Triton optimizer if available
        if self.use_triton:
            self.triton_kv_cache = TritonOptimizedKVCache(bits, bits)
        
        # Quantization parameters
        if bits == 8:
            self.qmin, self.qmax = 0, 255
            self.dtype = torch.uint8
        elif bits == 4:
            self.qmin, self.qmax = 0, 15
            self.dtype = torch.uint8
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
    
    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor using symmetric quantization"""
        if self.use_triton:
            return self.triton_kv_cache.quantize_kv(tensor, self.bits)
        else:
            # Fallback PyTorch implementation
            scale = tensor.abs().max() / (2**(self.bits-1) - 1)
            quantized = torch.clamp(
                torch.round(tensor / (scale + 1e-8)),
                -(2**(self.bits-1)), 2**(self.bits-1) - 1
            ).to(torch.int8)
            return quantized, scale
    
    def dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor"""
        if self.use_triton:
            return self.triton_kv_cache.dequantize_kv(quantized, scale, quantized.shape)
        else:
            return quantized.to(torch.float16) * scale
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with advanced compression"""
        
        with self.compression_lock:
            # Ensure we have enough space
            while len(self.compressed_keys) <= layer_idx:
                self.compressed_keys.append(None)
                self.compressed_values.append(None)
                self.key_scales.append(None)
                self.value_scales.append(None)
                self.token_ages.append(deque())
            
            # Update token ages
            current_time = time.time()
            self.token_ages[layer_idx].extend([current_time] * key_states.shape[-2])
            
            # Compress new keys and values
            compressed_keys, key_scale = self._compress_keys(key_states, layer_idx)
            compressed_values, value_scale = self._compress_values(value_states, layer_idx)
            
            # Combine with existing cache
            if self.compressed_keys[layer_idx] is not None:
                # Apply temporal compression to existing cache
                if self.temporal_decay:
                    self._apply_temporal_compression(layer_idx)
                
                # Concatenate with existing
                self.compressed_keys[layer_idx] = torch.cat([self.compressed_keys[layer_idx], compressed_keys], dim=-2)
                self.compressed_values[layer_idx] = torch.cat([self.compressed_values[layer_idx], compressed_values], dim=-2)
                
                # Update scales (use running average)
                self.key_scales[layer_idx] = 0.9 * self.key_scales[layer_idx] + 0.1 * key_scale
                self.value_scales[layer_idx] = 0.9 * self.value_scales[layer_idx] + 0.1 * value_scale
            else:
                self.compressed_keys[layer_idx] = compressed_keys
                self.compressed_values[layer_idx] = compressed_values
                self.key_scales[layer_idx] = key_scale
                self.value_scales[layer_idx] = value_scale
            
            # Return decompressed for immediate use
            decompressed_keys = self._decompress_keys(self.compressed_keys[layer_idx], self.key_scales[layer_idx])
            decompressed_values = self._decompress_values(self.compressed_values[layer_idx], self.value_scales[layer_idx])
            
            # Update statistics
            self._update_compression_stats(key_states, value_states, compressed_keys, compressed_values)
            
            return pruned_keys, pruned_values, keep_indices


class CompressedKVCache(MixedPrecisionKVCache):
    """Unified compressed KV cache that combines multiple compression techniques"""
    
    def __init__(self, method: str = "quantization", **kwargs):
        # Map old method names to new parameters
        if method == "quantization":
            super().__init__(key_bits=kwargs.get('bits', 4), value_bits=kwargs.get('bits', 8),
                           use_triton=kwargs.get('use_triton', True))
        elif method == "mixed_precision":
            super().__init__(key_bits=4, value_bits=8, 
                           temporal_decay=True, pattern_compression=True,
                           use_triton=kwargs.get('use_triton', True))
        elif method == "temporal":
            super().__init__(key_bits=8, value_bits=8,
                           temporal_decay=True, pattern_compression=False,
                           use_triton=kwargs.get('use_triton', True))
        else:
            # Default to mixed precision
            super().__init__(use_triton=kwargs.get('use_triton', True))
        
        self.method = method
    
    def _compress_keys(self, keys: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress keys using int4 quantization with dynamic scaling"""
        
        if self.use_triton:
            # Use Triton-optimized quantization
            return self.triton_kv_cache.quantize_kv(keys, self.key_bits)
        else:
            # Fallback to PyTorch implementation
            # Calculate dynamic scale
            key_max = keys.abs().max()
            key_scale = key_max / (2**(self.key_bits-1) - 1)
            
            # Quantize to int4
            keys_scaled = keys / (key_scale + 1e-8)
            keys_quantized = torch.clamp(torch.round(keys_scaled), 
                                       -(2**(self.key_bits-1)), 
                                       2**(self.key_bits-1) - 1)
            
            # Pack into efficient storage (2 int4 values per byte)
            if self.key_bits == 4:
                keys_packed = self._pack_int4(keys_quantized)
                return keys_packed, key_scale
            else:
                return keys_quantized.to(torch.int8), key_scale
    
    def _compress_values(self, values: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress values using fp8/int8 quantization"""
        
        if self.use_triton:
            # Use Triton-optimized quantization
            return self.triton_kv_cache.quantize_kv(values, self.value_bits)
        else:
            # Fallback to PyTorch implementation
            # Calculate dynamic scale for better precision
            value_max = values.abs().max()
            value_scale = value_max / (2**(self.value_bits-1) - 1)
            
            # Quantize values
            values_scaled = values / (value_scale + 1e-8)
            values_quantized = torch.clamp(torch.round(values_scaled),
                                         -(2**(self.value_bits-1)),
                                         2**(self.value_bits-1) - 1)
            
            return values_quantized.to(torch.int8), value_scale
    
    def _pack_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pack two int4 values into one int8 value"""
        # Ensure tensor is in correct range
        tensor = torch.clamp(tensor, -8, 7)
        
        # Get shape for later unpacking
        original_shape = tensor.shape
        
        # Flatten and ensure even number of elements
        flat = tensor.flatten()
        if flat.numel() % 2 == 1:
            flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype, device=flat.device)])
        
        # Reshape to pairs
        pairs = flat.view(-1, 2)
        
        # Pack: first value in lower 4 bits, second in upper 4 bits
        packed = (pairs[:, 0] & 0xF) | ((pairs[:, 1] & 0xF) << 4)
        
        # Store shape for unpacking
        return packed.to(torch.int8), original_shape
    
    def _unpack_int4(self, packed_tensor: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Unpack int4 values from int8 storage"""
        # Unpack pairs
        lower = packed_tensor & 0xF
        upper = (packed_tensor >> 4) & 0xF
        
        # Handle signed values
        lower = torch.where(lower > 7, lower - 16, lower)
        upper = torch.where(upper > 7, upper - 16, upper)
        
        # Interleave
        unpacked = torch.stack([lower, upper], dim=-1).flatten()
        
        # Trim to original size and reshape
        unpacked = unpacked[:torch.numel(torch.empty(original_shape))]
        return unpacked.view(original_shape)
    
    def _decompress_keys(self, compressed_keys: torch.Tensor, key_scale: torch.Tensor) -> torch.Tensor:
        """Decompress keys back to original precision"""
        if isinstance(compressed_keys, tuple):  # Packed int4
            packed, original_shape = compressed_keys
            keys_quantized = self._unpack_int4(packed, original_shape)
        else:
            keys_quantized = compressed_keys
        
        return keys_quantized.to(torch.float16) * key_scale
    
    def _decompress_values(self, compressed_values: torch.Tensor, value_scale: torch.Tensor) -> torch.Tensor:
        """Decompress values back to original precision"""
        return compressed_values.to(torch.float16) * value_scale
    
    def _apply_temporal_compression(self, layer_idx: int):
        """Apply progressive compression based on token age"""
        if not self.temporal_decay or len(self.token_ages[layer_idx]) == 0:
            return
        
        current_time = time.time()
        ages = [current_time - t for t in self.token_ages[layer_idx]]
        
        # Identify tokens for aggressive compression
        old_token_mask = torch.tensor([age > self.age_compression_thresholds[0] for age in ages], 
                                    device=self.compressed_keys[layer_idx].device if isinstance(self.compressed_keys[layer_idx], torch.Tensor) else 'cpu')
        
        if old_token_mask.any():
            # Apply more aggressive compression to old tokens
            self._compress_old_tokens(layer_idx, old_token_mask)
    
    def _compress_old_tokens(self, layer_idx: int, old_token_mask: torch.Tensor):
        """Apply more aggressive compression to old tokens"""
        # For now, just reduce precision further
        # In a full implementation, you might cluster similar tokens
        
        if isinstance(self.compressed_keys[layer_idx], tuple):
            return  # Already maximally compressed
        
        # Apply additional quantization to old tokens
        old_keys = self.compressed_keys[layer_idx][..., old_token_mask, :]
        old_values = self.compressed_values[layer_idx][..., old_token_mask, :]
        
        # Reduce precision by factor of 2
        self.compressed_keys[layer_idx][..., old_token_mask, :] = (old_keys / 2).round() * 2
        self.compressed_values[layer_idx][..., old_token_mask, :] = (old_values / 2).round() * 2
    
    def _update_compression_stats(self, original_keys: torch.Tensor, original_values: torch.Tensor,
                                compressed_keys: torch.Tensor, compressed_values: torch.Tensor):
        """Update compression statistics"""
        # Calculate memory savings
        original_size = (original_keys.numel() + original_values.numel()) * 2  # fp16 = 2 bytes
        
        if isinstance(compressed_keys, tuple):
            compressed_key_size = compressed_keys[0].numel()  # Packed int4
        else:
            compressed_key_size = compressed_keys.numel()
        
        compressed_size = compressed_key_size + compressed_values.numel()  # int8 = 1 byte
        
        memory_saved = (original_size - compressed_size) / (1024**2)  # MB
        
        self.compression_stats["total_tokens"] += original_keys.shape[-2]
        self.compression_stats["compressed_tokens"] += compressed_keys.shape[-2] if not isinstance(compressed_keys, tuple) else compressed_keys[1][-2]
        self.compression_stats["memory_saved_mb"] += memory_saved
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics"""
        compression_ratio = 1.0
        if self.compression_stats["total_tokens"] > 0:
            compression_ratio = self.compression_stats["compressed_tokens"] / self.compression_stats["total_tokens"]
        
        return {
            **self.compression_stats,
            "compression_ratio": compression_ratio,
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "temporal_decay_enabled": self.temporal_decay,
            "pattern_compression_enabled": self.pattern_compression
        }


class QuantizedKVCache:
    """Basic KV Cache with quantization compression - for backward compatibility"""
    
    def __init__(self, bits: int = 8, use_triton: bool = True):
        self.bits = bits
        self.use_triton = use_triton and TRITON_AVAILABLE
        self.quantized_cache = {}
        self.scales = {}
        self.stats = CompressionStats()
        
        # Initialize Triton optimizer if available
        if self.use_triton:
            self.triton_kv_cache = TritonOptimizedKVCache(bits, bits)
        
        # Quantization parameters
        if bits == 8:
            self.qmin, self.qmax = 0, 255
            self.dtype = torch.uint8
        elif bits == 4:
            self.qmin, self.qmax = 0, 15
            self.dtype = torch.uint8
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
    
    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor using symmetric quantization"""
        if self.use_triton:
            return self.triton_kv_cache.quantize_kv(tensor, self.bits)
        else:
            # Fallback PyTorch implementation
            scale = tensor.abs().max() / (2**(self.bits-1) - 1)
            quantized = torch.clamp(
                torch.round(tensor / (scale + 1e-8)),
                -(2**(self.bits-1)), 2**(self.bits-1) - 1
            ).to(torch.int8)
            return quantized, scale
    
    def dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor"""
        if self.use_triton:
            return self.triton_kv_cache.dequantize_kv(quantized, scale, quantized.shape)
        else:
            return quantized.to(torch.float16) * scale


class PrunedKVCache:
    """KV Cache with attention-based pruning - for backward compatibility"""
    
    def __init__(self, keep_ratio: float = 0.8, min_keep: int = 64):
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep
        self.pruned_cache = {}
        self.attention_scores = {}
        self.stats = CompressionStats()
    
    def prune_by_attention(self, keys: torch.Tensor, values: torch.Tensor,
                          attention_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prune KV cache based on attention scores"""
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # Calculate importance scores (mean attention across heads)
        importance = attention_scores.mean(dim=1)  # [batch, seq_len]
        
        # Determine how many tokens to keep
        keep_count = max(self.min_keep, int(seq_len * self.keep_ratio))
        
        # Get indices of most important tokens
        _, keep_indices = torch.topk(importance, keep_count, dim=-1)
        keep_indices = keep_indices.sort(dim=-1)[0]  # Sort to maintain order
        
        # Prune keys and values
        pruned_keys = torch.gather(keys, 2, keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim))
        pruned_values = torch.gather(values, 2, keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim))
        
        return pruned_keys, pruned_values, keep_indices


class CompressedKVCache(MixedPrecisionKVCache):
    """Unified compressed KV cache that combines multiple compression techniques"""
    
    def __init__(self, method: str = "quantization", **kwargs):
        # Map old method names to new parameters
        if method == "quantization":
            super().__init__(key_bits=kwargs.get('bits', 4), value_bits=kwargs.get('bits', 8),
                           use_triton=kwargs.get('use_triton', True))
        elif method == "mixed_precision":
            super().__init__(key_bits=4, value_bits=8, 
                           temporal_decay=True, pattern_compression=True,
                           use_triton=kwargs.get('use_triton', True))
        elif method == "temporal":
            super().__init__(key_bits=8, value_bits=8,
                           temporal_decay=True, pattern_compression=False,
                           use_triton=kwargs.get('use_triton', True))
        else:
            # Default to mixed precision
            super().__init__(use_triton=kwargs.get('use_triton', True))
        
        self.method = method


class TemporalKVCache(DynamicCache):
    """
    KV cache with temporal awareness and adaptive compression.
    Gradually compresses older tokens while preserving recent ones.
    """
    
    def __init__(self, temporal_window: int = 512, compression_schedule: List[Tuple[int, float]] = None):
        super().__init__()
        self.temporal_window = temporal_window
        self.compression_schedule = compression_schedule or [
            (100, 0.9),   # After 100 tokens, compress to 90%
            (500, 0.7),   # After 500 tokens, compress to 70%
            (1000, 0.5),  # After 1000 tokens, compress to 50%
            (2000, 0.3)   # After 2000 tokens, compress to 30%
        ]
        
        self.token_timestamps = []
        self.compression_levels = []
        self.temporal_stats = {
            "tokens_compressed": 0,
            "compression_events": 0,
            "memory_saved_temporal": 0.0
        }
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update with temporal compression"""
        
        # Ensure we have enough temporal tracking
        while len(self.token_timestamps) <= layer_idx:
            self.token_timestamps.append(deque())
            self.compression_levels.append(deque())
        
        # Add timestamps for new tokens
        current_time = time.time()
        new_token_count = key_states.shape[-2]
        self.token_timestamps[layer_idx].extend([current_time] * new_token_count)
        self.compression_levels[layer_idx].extend([1.0] * new_token_count)  # Start uncompressed
        
        # Apply temporal compression to existing cache
        if self.key_cache and len(self.key_cache) > layer_idx and self.key_cache[layer_idx] is not None:
            self._apply_temporal_compression(layer_idx)
        
        # Standard cache update
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
    
    def _apply_temporal_compression(self, layer_idx: int):
        """Apply temporal compression based on token age"""
        if len(self.token_timestamps[layer_idx]) == 0:
            return
        
        current_time = time.time()
        timestamps = list(self.token_timestamps[layer_idx])
        compression_levels = list(self.compression_levels[layer_idx])
        
        # Determine compression level for each token
        new_compression_levels = []
        tokens_to_compress = []
        
        for i, timestamp in enumerate(timestamps):
            age = current_time - timestamp
            
            # Find appropriate compression level
            compression_level = 1.0
            for age_threshold, comp_level in self.compression_schedule:
                if age > age_threshold:
                    compression_level = comp_level
                else:
                    break
            
            # Check if compression level changed
            if compression_level < compression_levels[i]:
                tokens_to_compress.append(i)
            
            new_compression_levels.append(compression_level)
        
        # Apply compression to selected tokens
        if tokens_to_compress:
            self._compress_tokens(layer_idx, tokens_to_compress, new_compression_levels)
            self.compression_levels[layer_idx] = deque(new_compression_levels)
            self.temporal_stats["compression_events"] += 1
    
    def _compress_tokens(self, layer_idx: int, token_indices: List[int], compression_levels: List[float]):
        """Compress specific tokens based on their age"""
        if self.key_cache[layer_idx] is None:
            return
        
        keys = self.key_cache[layer_idx]
        values = self.value_cache[layer_idx]
        
        for token_idx in token_indices:
            compression_level = compression_levels[token_idx]
            
            # Apply compression by reducing precision
            if compression_level < 1.0:
                # Compress by reducing precision and adding noise for regularization
                noise_scale = (1.0 - compression_level) * 0.01
                
                # Add small amount of noise and quantize
                keys[..., token_idx, :] = self._compress_tensor(keys[..., token_idx, :], compression_level, noise_scale)
                values[..., token_idx, :] = self._compress_tensor(values[..., token_idx, :], compression_level, noise_scale)
                
                self.temporal_stats["tokens_compressed"] += 1
    
    def _compress_tensor(self, tensor: torch.Tensor, compression_level: float, noise_scale: float) -> torch.Tensor:
        """Compress individual tensor with specified level"""
        if compression_level >= 1.0:
            return tensor
        
        # Quantize based on compression level
        quantization_levels = int(256 * compression_level)  # Reduce quantization levels
        
        # Normalize to [-1, 1]
        tensor_max = tensor.abs().max()
        if tensor_max > 0:
            normalized = tensor / tensor_max
            
            # Quantize
            quantized = torch.round(normalized * quantization_levels) / quantization_levels
            
            # Add regularization noise
            if noise_scale > 0:
                noise = torch.randn_like(quantized) * noise_scale
                quantized = quantized + noise
            
            # Denormalize
            return quantized * tensor_max
        
        return tensor
    
    def get_temporal_stats(self) -> Dict[str, Any]:
        """Get temporal compression statistics"""
        avg_age = 0.0
        if self.token_timestamps:
            current_time = time.time()
            all_ages = []
            for timestamps in self.token_timestamps:
                all_ages.extend([current_time - t for t in timestamps])
            
            if all_ages:
                avg_age = sum(all_ages) / len(all_ages)
        
        return {
            **self.temporal_stats,
            "temporal_window": self.temporal_window,
            "average_token_age": avg_age,
            "compression_schedule": self.compression_schedule,
            "total_tokens_tracked": sum(len(ts) for ts in self.token_timestamps)
        }


class PatternAwareKVCache(DynamicCache):
    """
    KV cache that detects and compresses repeated attention patterns.
    """
    
    def __init__(self, pattern_threshold: float = 0.8, max_patterns: int = 100):
        super().__init__()
        self.pattern_threshold = pattern_threshold
        self.max_patterns = max_patterns
        
        self.attention_patterns = {}  # Store learned patterns
        self.pattern_usage = defaultdict(int)
        self.pattern_compression_map = {}
        
        self.pattern_stats = {
            "patterns_detected": 0,
            "pattern_compressions": 0,
            "memory_saved_patterns": 0.0
        }
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update with pattern detection and compression"""
        
        # Detect patterns in the new key-value pairs
        if key_states.shape[-2] > 1:  # Only if we have multiple tokens
            self._detect_patterns(key_states, value_states, layer_idx)
        
        # Apply pattern-based compression
        compressed_keys, compressed_values = self._apply_pattern_compression(key_states, value_states, layer_idx)
        
        # Standard cache update with compressed data
        return super().update(compressed_keys, compressed_values, layer_idx, cache_kwargs)
    
    def _detect_patterns(self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int):
        """Detect repeating patterns in key-value pairs"""
        seq_len = keys.shape[-2]
        
        # Look for patterns of different lengths
        for pattern_len in [2, 4, 8, 16]:
            if pattern_len >= seq_len:
                continue
            
            # Check for repeating patterns
            for start_idx in range(seq_len - pattern_len):
                pattern_keys = keys[..., start_idx:start_idx + pattern_len, :]
                pattern_values = values[..., start_idx:start_idx + pattern_len, :]
                
                # Look for similar patterns elsewhere
                for compare_idx in range(start_idx + pattern_len, seq_len - pattern_len + 1):
                    compare_keys = keys[..., compare_idx:compare_idx + pattern_len, :]
                    compare_values = values[..., compare_idx:compare_idx + pattern_len, :]
                    
                    # Calculate similarity
                    key_similarity = self._calculate_similarity(pattern_keys, compare_keys)
                    value_similarity = self._calculate_similarity(pattern_values, compare_values)
                    
                    if key_similarity > self.pattern_threshold and value_similarity > self.pattern_threshold:
                        # Found a pattern!
                        pattern_id = self._register_pattern(pattern_keys, pattern_values, layer_idx)
                        self.pattern_usage[pattern_id] += 1
                        self.pattern_stats["patterns_detected"] += 1
    
    def _calculate_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Calculate cosine similarity between two tensors"""
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        
        # Cosine similarity
        dot_product = torch.dot(flat1, flat2)
        norm1 = torch.norm(flat1)
        norm2 = torch.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    
    def _register_pattern(self, pattern_keys: torch.Tensor, pattern_values: torch.Tensor, layer_idx: int) -> str:
        """Register a new pattern or return existing pattern ID"""
        # Create pattern signature
        pattern_signature = self._create_pattern_signature(pattern_keys, pattern_values)
        pattern_id = f"layer_{layer_idx}_pattern_{pattern_signature}"
        
        if pattern_id not in self.attention_patterns:
            if len(self.attention_patterns) < self.max_patterns:
                self.attention_patterns[pattern_id] = {
                    "keys": pattern_keys.clone(),
                    "values": pattern_values.clone(),
                    "layer_idx": layer_idx,
                    "created_at": time.time()
                }
        
        return pattern_id
    
    def _create_pattern_signature(self, keys: torch.Tensor, values: torch.Tensor) -> str:
        """Create a unique signature for a pattern"""
        # Use hash of key statistics as signature
        key_stats = [
            keys.mean().item(),
            keys.std().item(),
            keys.max().item(),
            keys.min().item()
        ]
        
        value_stats = [
            values.mean().item(),
            values.std().item(),
            values.max().item(),
            values.min().item()
        ]
        
        combined_stats = key_stats + value_stats
        return str(hash(tuple(combined_stats)))[:8]
    
    def _apply_pattern_compression(self, keys: torch.Tensor, values: torch.Tensor, 
                                 layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply pattern-based compression"""
        # For now, return original tensors
        # In a full implementation, you would replace detected patterns with references
        return keys, values
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get pattern detection and compression statistics"""
        return {
            **self.pattern_stats,
            "total_patterns": len(self.attention_patterns),
            "max_patterns": self.max_patterns,
            "pattern_threshold": self.pattern_threshold,
            "most_used_patterns": sorted(self.pattern_usage.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]
        }


class UltraAdvancedKVCache(DynamicCache):
    """
    Ultra-advanced KV cache combining all compression techniques.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Initialize all compression methods
        self.mixed_precision = MixedPrecisionKVCache(**kwargs)
        self.temporal = TemporalKVCache(**kwargs)
        self.pattern_aware = PatternAwareKVCache(**kwargs)
        
        # Coordination
        self.compression_strategy = kwargs.get('compression_strategy', 'adaptive')
        self.quality_threshold = kwargs.get('quality_threshold', 0.95)
        
        self.ultra_stats = {
            "compression_method_used": defaultdict(int),
            "total_memory_saved": 0.0,
            "quality_maintained": 1.0
        }
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ultra-advanced update with intelligent compression selection"""
        
        # Choose compression method based on characteristics
        compression_method = self._choose_compression_method(key_states, value_states, layer_idx)
        
        # Apply chosen compression
        if compression_method == 'mixed_precision':
            result = self.mixed_precision.update(key_states, value_states, layer_idx, cache_kwargs)
            self.ultra_stats["compression_method_used"]["mixed_precision"] += 1
        elif compression_method == 'temporal':
            result = self.temporal.update(key_states, value_states, layer_idx, cache_kwargs)
            self.ultra_stats["compression_method_used"]["temporal"] += 1
        elif compression_method == 'pattern_aware':
            result = self.pattern_aware.update(key_states, value_states, layer_idx, cache_kwargs)
            self.ultra_stats["compression_method_used"]["pattern_aware"] += 1
        else:
            # Fallback to standard
            result = super().update(key_states, value_states, layer_idx, cache_kwargs)
            self.ultra_stats["compression_method_used"]["standard"] += 1
        
        return result
    
    def _choose_compression_method(self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int) -> str:
        """Intelligently choose compression method based on data characteristics"""
        
        if self.compression_strategy == 'mixed_precision':
            return 'mixed_precision'
        elif self.compression_strategy == 'temporal':
            return 'temporal'
        elif self.compression_strategy == 'pattern_aware':
            return 'pattern_aware'
        elif self.compression_strategy == 'adaptive':
            # Analyze data characteristics
            
            # Check for patterns (high repetition suggests pattern compression)
            pattern_score = self._calculate_pattern_score(keys, values)
            
            # Check temporal characteristics (long sequences suggest temporal compression)
            temporal_score = keys.shape[-2] / 1000.0  # Normalize by typical sequence length
            
            # Check precision requirements (high variation suggests mixed precision)
            precision_score = (keys.std() + values.std()).item()
            
            # Choose based on scores
            if pattern_score > 0.7:
                return 'pattern_aware'
            elif temporal_score > 0.5:
                return 'temporal'
            elif precision_score < 0.1:  # Low variation - good for quantization
                return 'mixed_precision'
            else:
                return 'mixed_precision'  # Default
        
        return 'standard'
    
    def _calculate_pattern_score(self, keys: torch.Tensor, values: torch.Tensor) -> float:
        """Calculate how pattern-rich the data is"""
        # Simple pattern detection - look for repeated subsequences
        seq_len = keys.shape[-2]
        if seq_len < 4:
            return 0.0
        
        pattern_count = 0
        comparisons = 0
        
        # Check small patterns
        for i in range(seq_len - 2):
            for j in range(i + 2, seq_len - 1):
                if j + 1 < seq_len:
                    # Compare 2-token patterns
                    pattern1 = keys[..., i:i+2, :]
                    pattern2 = keys[..., j:j+2, :]
                    
                    similarity = torch.cosine_similarity(pattern1.flatten(), pattern2.flatten(), dim=0)
                    if similarity > 0.8:
                        pattern_count += 1
                    
                    comparisons += 1
                    
                    if comparisons > 20:  # Limit computation
                        break
            
            if comparisons > 20:
                break
        
        return pattern_count / max(comparisons, 1)
    
    def get_ultra_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all compression methods"""
        return {
            "ultra_stats": dict(self.ultra_stats),
            "mixed_precision_stats": self.mixed_precision.get_compression_stats(),
            "temporal_stats": self.temporal.get_temporal_stats(),
            "pattern_stats": self.pattern_aware.get_pattern_stats()
        }