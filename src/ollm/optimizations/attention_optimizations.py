"""
Advanced attention mechanisms including sliding window and sparse attention.
Reduces computational complexity from O(n²) to O(n*w) or O(n*log(n)).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

try:
    from .triton_kernels import TritonSparseAttention, TritonSlidingWindowAttention, create_strided_sparsity_mask
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
from ..attention import online_chunked_grouped_attention_rope_no_mask


@dataclass
class AttentionStats:
    """Statistics tracker for attention operations"""
    total_operations: int = 0
    total_time: float = 0.0
    memory_saved: int = 0
    sparsity_ratio: float = 0.0
    
    def update(self, time_taken: float, memory_saved: int = 0, sparsity_ratio: float = 0.0):
        """Update statistics"""
        self.total_operations += 1
        self.total_time += time_taken
        self.memory_saved += memory_saved
        self.sparsity_ratio = (self.sparsity_ratio * (self.total_operations - 1) + sparsity_ratio) / self.total_operations
    
    def get_average_time(self) -> float:
        """Get average operation time"""
        return self.total_time / self.total_operations if self.total_operations > 0 else 0.0
    
    def reset(self):
        """Reset all statistics"""
        self.total_operations = 0
        self.total_time = 0.0
        self.memory_saved = 0
        self.sparsity_ratio = 0.0

class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention implementation that only attends to nearby tokens.
    Memory efficient for long sequences.
    """
    
    def __init__(self, window_size: int = 512, use_triton: bool = True):
        super().__init__()
        self.window_size = window_size
        self.overlap = window_size // 4  # Default overlap for window processing
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # Initialize Triton sliding window attention if available
        if self.use_triton:
            self.triton_sliding_attention = TritonSlidingWindowAttention(window_size)
        
        self.stats = AttentionStats()
    
    def create_sliding_mask(self, seq_len: int, device: str = "cuda") -> torch.Tensor:
        """Create sliding window attention mask"""
        # Create a mask where each position can only attend to positions within its window
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0
        
        return mask
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with sliding window attention.
        
        Args:
            query: (B, H, L, D) query tensor
            key: (B, H, L, D) key tensor  
            value: (B, H, L, D) value tensor
            attention_mask: Optional mask tensor
            position_ids: Position indices for RoPE
            
        Returns:
            Output tensor (B, H, L, D)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        if self.use_triton and seq_len > self.window_size:
            # Use Triton-optimized sliding window attention
            output = self.triton_sliding_attention.forward(query, key, value)
            
            # Update statistics 
            window_coverage = min(1.0, self.window_size / seq_len)
            self.stats.window_efficiency = window_coverage
            self.stats.flops_saved = int(seq_len * seq_len * (1 - window_coverage)) * batch_size * num_heads * head_dim
            
            return output
        elif seq_len <= self.window_size:
            # Use full attention for short sequences
            return online_chunked_grouped_attention_rope_no_mask(
                query, key, value, position_ids
            )
        else:
            # Fallback to PyTorch implementation
            # Process sequence in overlapping windows
            output = torch.zeros_like(query)
            counts = torch.zeros(batch_size, num_heads, seq_len, 1, device=query.device)
        
        for start in range(0, seq_len, self.window_size):
            end = min(start + self.window_size, seq_len)
            window_len = end - start
            
            # Extract window
            q_window = query[:, :, start:end, :]
            k_window = key[:, :, start:end, :]
            v_window = value[:, :, start:end, :]
            
            # Adjust position IDs for window
            if position_ids is not None:
                pos_window = position_ids[:, start:end] if position_ids.dim() == 2 else position_ids[start:end]
            else:
                pos_window = None
            
            # Compute attention for window
            window_output = online_chunked_grouped_attention_rope_no_mask(
                q_window, k_window, v_window, pos_window
            )
            
            # Accumulate output with overlap handling
            if start == 0:
                # First window - use all positions
                output[:, :, start:end, :] += window_output
                counts[:, :, start:end, :] += 1
            elif end == seq_len:
                # Last window - use all positions
                output[:, :, start:end, :] += window_output
                counts[:, :, start:end, :] += 1
            else:
                # Middle window - blend overlap regions
                overlap_start = self.overlap // 2
                overlap_end = window_len - self.overlap // 2
                
                # Non-overlapping region
                output[:, :, start + overlap_start:start + overlap_end, :] += window_output[:, :, overlap_start:overlap_end, :]
                counts[:, :, start + overlap_start:start + overlap_end, :] += 1
                
                # Overlapping regions with fade in/out
                if overlap_start > 0:
                    fade_in = torch.linspace(0, 1, overlap_start, device=query.device).view(1, 1, -1, 1)
                    output[:, :, start:start + overlap_start, :] += window_output[:, :, :overlap_start, :] * fade_in
                    counts[:, :, start:start + overlap_start, :] += fade_in
                
                if overlap_end < window_len:
                    fade_out = torch.linspace(1, 0, window_len - overlap_end, device=query.device).view(1, 1, -1, 1)
                    output[:, :, start + overlap_end:end, :] += window_output[:, :, overlap_end:, :] * fade_out
                    counts[:, :, start + overlap_end:end, :] += fade_out
        
        # Normalize by counts (handle overlaps)
        output = output / (counts + 1e-8)
        
        return output


class SparseAttentionOptimizer(nn.Module):
    """
    Implements various sparse attention patterns to reduce computation.
    """
    
    def __init__(self, pattern: str = "strided", 
                 local_window: int = 64, stride: int = 256,
                 use_triton: bool = True):
        super().__init__()
        self.pattern = pattern
        self.local_window = local_window
        self.stride = stride
        self.sparsity_masks = {}
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # Initialize Triton sparse attention if available
        if self.use_triton:
            self.triton_sparse_attention = TritonSparseAttention(pattern)
        
        self.stats = AttentionStats()
    
    def create_strided_mask(self, seq_len: int, device: str = "cuda") -> torch.Tensor:
        """Create strided attention pattern"""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        for i in range(seq_len):
            # Local attention within window
            local_start = max(0, i - self.local_window // 2)
            local_end = min(seq_len, i + self.local_window // 2 + 1)
            mask[i, local_start:local_end] = 0
            
            # Strided attention to distant positions
            for j in range(i + self.stride, seq_len, self.stride):
                mask[i, j] = 0
            for j in range(i - self.stride, -1, -self.stride):
                mask[i, j] = 0
        
        return mask
    
    def create_random_mask(self, seq_len: int, device: str = "cuda") -> torch.Tensor:
        """Create random sparse attention pattern"""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        # Each position attends to a random subset of positions
        num_attend = int(seq_len * self.random_ratio)
        
        for i in range(seq_len):
            # Always attend to local window
            local_start = max(0, i - self.local_window // 2)
            local_end = min(seq_len, i + self.local_window // 2 + 1)
            mask[i, local_start:local_end] = 0
            
            # Add random positions
            remaining_positions = list(range(seq_len))
            for j in range(local_start, local_end):
                if j in remaining_positions:
                    remaining_positions.remove(j)
            
            if remaining_positions:
                random_indices = torch.randperm(len(remaining_positions))[:num_attend]
                for idx in random_indices:
                    mask[i, remaining_positions[idx]] = 0
        
        return mask
    
    def create_sparse_mask(self, seq_len: int, device: str = "cuda") -> torch.Tensor:
        """Create sparse attention mask based on pattern"""
        if self.pattern == "strided":
            return self.create_strided_mask(seq_len, device)
        elif self.pattern == "random":
            return self.create_random_mask(seq_len, device)
        elif self.pattern == "local":
            # Only local attention
            mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
            for i in range(seq_len):
                start = max(0, i - self.local_window // 2)
                end = min(seq_len, i + self.local_window // 2 + 1)
                mask[i, start:end] = 0
            return mask
        else:
            raise ValueError(f"Unknown sparsity pattern: {self.pattern}")
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with sparse attention.
        
        Args:
            query: (B, H, L, D) query tensor
            key: (B, H, L, D) key tensor
            value: (B, H, L, D) value tensor
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor (B, H, L, D)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        device = query.device
        
        if self.use_triton:
            # Create Triton-compatible sparsity mask
            sparsity_mask = create_strided_sparsity_mask(seq_len, self.stride, self.local_window).to(device)
            
            # Use Triton kernel for sparse attention  
            output = self.triton_sparse_attention.forward(query, key, value, sparsity_mask)
            
            # Update statistics
            total_elements = seq_len * seq_len
            sparse_elements = sparsity_mask.sum().item()
            self.stats.sparsity_ratio = 1.0 - (sparse_elements / total_elements)
            self.stats.flops_saved = int(total_elements - sparse_elements) * batch_size * num_heads * head_dim
            
            return output
        else:
            # Fallback to PyTorch implementation
            # Create sparse mask
            sparse_mask = self.create_sparse_mask(seq_len, device)
            
            # Combine with existing mask if provided
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                
                # Convert boolean mask to additive mask
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.float().masked_fill(attention_mask, float('-inf'))
                
                sparse_mask = sparse_mask + attention_mask
            
            # Compute attention scores
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            
            # Apply sparse mask
            scores = scores + sparse_mask.unsqueeze(0).unsqueeze(0)
            
            # Compute attention weights
            attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        
        return output


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention that processes sequences at different resolutions.
    Combines global context with local details efficiently.
    """
    
    def __init__(self, scales: list = [1, 2, 4], head_dim: int = 64):
        super().__init__()
        self.scales = scales
        self.head_dim = head_dim
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
    
    def downsample_sequence(self, tensor: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample sequence by averaging every `scale` positions"""
        if scale == 1:
            return tensor
        
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        
        # Pad sequence if necessary
        pad_len = (scale - seq_len % scale) % scale
        if pad_len > 0:
            tensor = F.pad(tensor, (0, 0, 0, pad_len))
            seq_len += pad_len
        
        # Reshape and average
        tensor = tensor.view(batch_size, num_heads, seq_len // scale, scale, head_dim)
        return tensor.mean(dim=3)
    
    def upsample_sequence(self, tensor: torch.Tensor, scale: int, target_len: int) -> torch.Tensor:
        """Upsample sequence by repeating values"""
        if scale == 1:
            return tensor[:, :, :target_len, :]
        
        batch_size, num_heads, downsampled_len, head_dim = tensor.shape
        
        # Repeat each position `scale` times
        tensor = tensor.unsqueeze(3).repeat(1, 1, 1, scale, 1)
        tensor = tensor.view(batch_size, num_heads, downsampled_len * scale, head_dim)
        
        # Trim to target length
        return tensor[:, :, :target_len, :]
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale attention.
        
        Args:
            query: (B, H, L, D) query tensor
            key: (B, H, L, D) key tensor
            value: (B, H, L, D) value tensor
            
        Returns:
            Output tensor (B, H, L, D)
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        outputs = []
        
        for i, scale in enumerate(self.scales):
            # Downsample inputs
            q_down = self.downsample_sequence(query, scale)
            k_down = self.downsample_sequence(key, scale)
            v_down = self.downsample_sequence(value, scale)
            
            # Compute attention at this scale
            scale_factor = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q_down, k_down.transpose(-2, -1)) * scale_factor
            attn_weights = F.softmax(scores, dim=-1)
            output_down = torch.matmul(attn_weights, v_down)
            
            # Upsample output back to original resolution
            output_up = self.upsample_sequence(output_down, scale, seq_len)
            outputs.append(output_up)
        
        # Combine outputs from different scales
        combined_output = torch.stack(outputs, dim=0)
        weights = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1, 1)
        output = (combined_output * weights).sum(dim=0)
        
        return output


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention that switches between different patterns based on sequence characteristics.
    Automatically selects the most appropriate attention mechanism.
    """
    
    def __init__(self, head_dim: int = 64, 
                 short_seq_threshold: int = 512,
                 long_seq_threshold: int = 4096):
        super().__init__()
        self.head_dim = head_dim
        self.short_threshold = short_seq_threshold
        self.long_threshold = long_seq_threshold
        
        # Different attention mechanisms
        self.sliding_window = SlidingWindowAttention(window_size=1024)
        self.sparse_attention = SparseAttention(sparsity_pattern="strided")
        self.multiscale_attention = MultiScaleAttention()
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adaptive forward pass that selects attention mechanism based on sequence length.
        
        Args:
            query: (B, H, L, D) query tensor
            key: (B, H, L, D) key tensor
            value: (B, H, L, D) value tensor
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE
            
        Returns:
            Output tensor (B, H, L, D)
        """
        seq_len = query.shape[2]
        
        if seq_len <= self.short_threshold:
            # Use full attention for short sequences
            return online_chunked_grouped_attention_rope_no_mask(
                query, key, value, position_ids
            )
        elif seq_len <= self.long_threshold:
            # Use sliding window for medium sequences
            return self.sliding_window(query, key, value, attention_mask, position_ids)
        else:
            # Use sparse attention for very long sequences
            return self.sparse_attention(query, key, value, attention_mask)


class AttentionOptimizer:
    """
    Utility class for optimizing attention computation based on available resources.
    """
    
    @staticmethod
    def estimate_memory_usage(batch_size: int, num_heads: int, seq_len: int, 
                            head_dim: int, dtype: torch.dtype = torch.float16) -> float:
        """Estimate memory usage for attention computation in GB"""
        element_size = torch.tensor(0, dtype=dtype).element_size()
        
        # QKV tensors
        qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim * element_size
        
        # Attention scores matrix
        scores_memory = batch_size * num_heads * seq_len * seq_len * element_size
        
        # Output tensor
        output_memory = batch_size * num_heads * seq_len * head_dim * element_size
        
        total_memory = qkv_memory + scores_memory + output_memory
        return total_memory / (1024**3)  # Convert to GB
    
    @staticmethod
    def choose_attention_mechanism(seq_len: int, available_memory_gb: float,
                                 batch_size: int = 1, num_heads: int = 32,
                                 head_dim: int = 64) -> str:
        """Choose optimal attention mechanism based on constraints"""
        
        # Estimate memory usage for full attention
        full_attention_memory = AttentionOptimizer.estimate_memory_usage(
            batch_size, num_heads, seq_len, head_dim
        )
        
        if full_attention_memory <= available_memory_gb * 0.8:  # 80% threshold
            return "full"
        elif seq_len <= 8192:
            return "sliding_window"
        elif seq_len <= 32768:
            return "sparse"
        else:
            return "multiscale"
    
    @staticmethod
    def get_optimal_chunk_sizes(seq_len: int, available_memory_gb: float) -> Tuple[int, int]:
        """Get optimal chunk sizes for chunked attention"""
        base_memory_per_token = 0.001  # Rough estimate in GB
        
        max_tokens_in_memory = int(available_memory_gb / base_memory_per_token)
        
        # Calculate chunk sizes
        q_chunk_size = min(seq_len, max_tokens_in_memory // 4)
        k_chunk_size = min(seq_len, max_tokens_in_memory // 8)
        
        return q_chunk_size, k_chunk_size