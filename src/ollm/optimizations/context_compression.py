"""
Context compression techniques to handle very long sequences efficiently.
Includes hierarchical attention and intelligent context summarization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from dataclasses import dataclass
import math

@dataclass
class CompressionConfig:
    """Configuration for context compression"""
    compression_ratio: float = 0.5
    importance_threshold: float = 0.1
    min_context_length: int = 512
    max_context_length: int = 8192
    preserve_recent_tokens: int = 256

class ContextCompressor:
    """
    Intelligent context compression that preserves important information
    while reducing sequence length for memory efficiency.
    """
    
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.importance_cache = {}
        
    def compute_token_importance(self, hidden_states: torch.Tensor, 
                                attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute importance scores for each token in the sequence.
        
        Args:
            hidden_states: (B, L, D) hidden states
            attention_weights: (B, H, L, L) attention weights
            
        Returns:
            Importance scores (B, L)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Method 1: Attention-based importance
        if attention_weights is not None:
            # Average attention received by each token
            attention_importance = attention_weights.mean(dim=1).sum(dim=1)  # (B, L)
        else:
            attention_importance = torch.ones(batch_size, seq_len, device=hidden_states.device)
        
        # Method 2: Magnitude-based importance
        magnitude_importance = torch.norm(hidden_states, dim=-1)  # (B, L)
        
        # Method 3: Gradient-based importance (if available)
        gradient_importance = torch.ones_like(magnitude_importance)
        if hidden_states.requires_grad and hidden_states.grad is not None:
            gradient_importance = torch.norm(hidden_states.grad, dim=-1)
        
        # Combine importance scores
        combined_importance = (
            0.4 * attention_importance + 
            0.4 * magnitude_importance + 
            0.2 * gradient_importance
        )
        
        # Normalize to [0, 1]
        combined_importance = (combined_importance - combined_importance.min(dim=1, keepdim=True)[0]) / \
                            (combined_importance.max(dim=1, keepdim=True)[0] - combined_importance.min(dim=1, keepdim=True)[0] + 1e-8)
        
        return combined_importance
    
    def select_important_tokens(self, hidden_states: torch.Tensor, 
                              importance_scores: torch.Tensor,
                              target_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select most important tokens to keep.
        
        Args:
            hidden_states: (B, L, D) hidden states
            importance_scores: (B, L) importance scores
            target_length: Target sequence length after compression
            
        Returns:
            Tuple of (compressed_hidden_states, selected_indices)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Always preserve recent tokens
        recent_tokens = min(self.config.preserve_recent_tokens, seq_len)
        preserve_indices = torch.arange(seq_len - recent_tokens, seq_len, device=hidden_states.device)
        
        # Select important tokens from the rest
        remaining_budget = target_length - recent_tokens
        if remaining_budget > 0:
            # Get importance scores for non-recent tokens
            early_importance = importance_scores[:, :seq_len - recent_tokens]
            
            # Select top important tokens
            _, top_indices = torch.topk(early_importance, min(remaining_budget, early_importance.shape[1]), dim=1)
            
            # Combine with recent tokens
            selected_indices = torch.cat([
                top_indices,
                preserve_indices.unsqueeze(0).expand(batch_size, -1)
            ], dim=1)
        else:
            # Only keep recent tokens
            selected_indices = preserve_indices.unsqueeze(0).expand(batch_size, -1)
        
        # Sort indices to maintain order
        selected_indices = torch.sort(selected_indices, dim=1)[0]
        
        # Gather selected hidden states
        compressed_states = torch.gather(
            hidden_states, 
            dim=1, 
            index=selected_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )
        
        return compressed_states, selected_indices
    
    def compress_context(self, hidden_states: torch.Tensor,
                        attention_weights: Optional[torch.Tensor] = None,
                        target_ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Compress context by removing less important tokens.
        
        Args:
            hidden_states: (B, L, D) hidden states
            attention_weights: Optional attention weights
            target_ratio: Target compression ratio (overrides config)
            
        Returns:
            Dictionary with compressed states and metadata
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Determine target length
        ratio = target_ratio or self.config.compression_ratio
        target_length = max(int(seq_len * ratio), self.config.min_context_length)
        target_length = min(target_length, self.config.max_context_length)
        
        if target_length >= seq_len:
            # No compression needed
            return {
                'compressed_states': hidden_states,
                'selected_indices': torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1),
                'compression_ratio': 1.0,
                'original_length': seq_len,
                'compressed_length': seq_len
            }
        
        # Compute importance scores
        importance_scores = self.compute_token_importance(hidden_states, attention_weights)
        
        # Select important tokens
        compressed_states, selected_indices = self.select_important_tokens(
            hidden_states, importance_scores, target_length
        )
        
        actual_ratio = compressed_states.shape[1] / seq_len
        
        return {
            'compressed_states': compressed_states,
            'selected_indices': selected_indices,
            'importance_scores': importance_scores,
            'compression_ratio': actual_ratio,
            'original_length': seq_len,
            'compressed_length': compressed_states.shape[1]
        }


class HierarchicalContext:
    """
    Hierarchical context management with different attention resolutions.
    Recent context gets full attention, older context gets progressively sparser attention.
    """
    
    def __init__(self, levels: List[int] = [1, 4, 16], 
                 level_lengths: List[int] = [512, 1024, 2048]):
        self.levels = levels  # Downsampling factors for each level
        self.level_lengths = level_lengths  # Max length for each level
        assert len(levels) == len(level_lengths), "Levels and lengths must match"
        
    def create_hierarchical_representation(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Create hierarchical representation of the context.
        
        Args:
            hidden_states: (B, L, D) input hidden states
            
        Returns:
            Dictionary with representations at different levels
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        representations = {}
        
        current_pos = seq_len
        
        for level_idx, (downsample_factor, max_length) in enumerate(zip(self.levels, self.level_lengths)):
            level_name = f"level_{level_idx}"
            
            # Determine the range for this level
            level_tokens = min(max_length * downsample_factor, current_pos)
            start_pos = max(0, current_pos - level_tokens)
            
            if start_pos >= current_pos:
                # No tokens left for this level
                representations[level_name] = torch.empty(batch_size, 0, hidden_dim, 
                                                         device=hidden_states.device, 
                                                         dtype=hidden_states.dtype)
                continue
            
            # Extract tokens for this level
            level_hidden = hidden_states[:, start_pos:current_pos, :]
            
            if downsample_factor > 1:
                # Downsample by averaging groups of tokens
                level_len = level_hidden.shape[1]
                pad_len = (downsample_factor - level_len % downsample_factor) % downsample_factor
                
                if pad_len > 0:
                    # Pad with zeros or repeat last token
                    padding = level_hidden[:, -1:, :].repeat(1, pad_len, 1)
                    level_hidden = torch.cat([level_hidden, padding], dim=1)
                
                # Reshape and average
                reshaped = level_hidden.view(batch_size, -1, downsample_factor, hidden_dim)
                downsampled = reshaped.mean(dim=2)  # Average over downsample dimension
                
                representations[level_name] = downsampled[:, :max_length, :]  # Trim to max length
            else:
                representations[level_name] = level_hidden[:, :max_length, :]
            
            # Update current position for next level
            current_pos = start_pos
            
            if current_pos <= 0:
                break
        
        return representations
    
    def create_hierarchical_attention_mask(self, query_length: int) -> torch.Tensor:
        """
        Create attention mask for hierarchical attention.
        
        Args:
            query_length: Length of query sequence
            
        Returns:
            Attention mask tensor
        """
        total_kv_length = sum(min(length, query_length // factor) 
                             for factor, length in zip(self.levels, self.level_lengths))
        
        # Create mask - queries can attend to all hierarchical keys
        mask = torch.zeros(query_length, total_kv_length)
        
        return mask  # All zeros = no masking (full attention to hierarchical context)
    
    def combine_hierarchical_outputs(self, outputs: Dict[str, torch.Tensor], 
                                   target_length: int) -> torch.Tensor:
        """
        Combine outputs from different hierarchical levels.
        
        Args:
            outputs: Dictionary of outputs from each level
            target_length: Target output length
            
        Returns:
            Combined output tensor
        """
        if not outputs:
            return torch.empty(0)
        
        # Get dimensions from first output
        first_output = next(iter(outputs.values()))
        batch_size, _, hidden_dim = first_output.shape
        
        combined = torch.zeros(batch_size, target_length, hidden_dim, 
                              device=first_output.device, dtype=first_output.dtype)
        
        current_pos = 0
        
        # Combine from finest to coarsest resolution
        for level_idx in range(len(self.levels)):
            level_name = f"level_{level_idx}"
            
            if level_name in outputs:
                level_output = outputs[level_name]
                level_length = min(level_output.shape[1], target_length - current_pos)
                
                if level_length > 0:
                    combined[:, current_pos:current_pos + level_length, :] = level_output[:, :level_length, :]
                    current_pos += level_length
                
                if current_pos >= target_length:
                    break
        
        return combined


class AdaptiveContextManager:
    """
    Adaptive context manager that dynamically adjusts compression strategy
    based on sequence characteristics and available resources.
    """
    
    def __init__(self, max_context_length: int = 8192,
                 memory_threshold_gb: float = 6.0):
        self.max_context_length = max_context_length
        self.memory_threshold = memory_threshold_gb
        
        self.compressor = ContextCompressor()
        self.hierarchical = HierarchicalContext()
        
        # Adaptive parameters
        self.compression_history = []
        self.performance_history = []
        
    def estimate_memory_usage(self, seq_length: int, hidden_dim: int = 4096, 
                            batch_size: int = 1, dtype: torch.dtype = torch.float16) -> float:
        """Estimate memory usage for sequence in GB"""
        element_size = torch.tensor(0, dtype=dtype).element_size()
        
        # Hidden states
        hidden_memory = batch_size * seq_length * hidden_dim * element_size
        
        # Attention matrices (approximate)
        attention_memory = batch_size * seq_length * seq_length * element_size
        
        total_memory = hidden_memory + attention_memory
        return total_memory / (1024**3)
    
    def choose_compression_strategy(self, seq_length: int, **kwargs) -> str:
        """Choose optimal compression strategy based on sequence characteristics"""
        
        estimated_memory = self.estimate_memory_usage(seq_length, **kwargs)
        
        if seq_length <= 512:
            return "none"  # No compression needed
        elif seq_length <= 2048 and estimated_memory < self.memory_threshold:
            return "light"  # Light compression
        elif seq_length <= 8192:
            return "standard"  # Standard compression
        else:
            return "hierarchical"  # Use hierarchical approach
    
    def compress_adaptively(self, hidden_states: torch.Tensor, 
                          attention_weights: Optional[torch.Tensor] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Apply adaptive compression based on sequence characteristics.
        
        Args:
            hidden_states: Input hidden states
            attention_weights: Optional attention weights
            **kwargs: Additional parameters
            
        Returns:
            Compression result with metadata
        """
        seq_length = hidden_states.shape[1]
        strategy = self.choose_compression_strategy(seq_length, **kwargs)
        
        if strategy == "none":
            return {
                'compressed_states': hidden_states,
                'strategy': strategy,
                'compression_ratio': 1.0
            }
        
        elif strategy == "light":
            # Light compression - 20% reduction
            result = self.compressor.compress_context(
                hidden_states, attention_weights, target_ratio=0.8
            )
            result['strategy'] = strategy
            return result
        
        elif strategy == "standard":
            # Standard compression - 50% reduction
            result = self.compressor.compress_context(
                hidden_states, attention_weights, target_ratio=0.5
            )
            result['strategy'] = strategy
            return result
        
        elif strategy == "hierarchical":
            # Hierarchical compression
            hierarchical_repr = self.hierarchical.create_hierarchical_representation(hidden_states)
            
            # Combine hierarchical levels into single compressed representation
            combined = []
            for level_name, level_repr in hierarchical_repr.items():
                combined.append(level_repr)
            
            if combined:
                compressed_states = torch.cat(combined, dim=1)
            else:
                compressed_states = hidden_states[:, :self.max_context_length, :]
            
            return {
                'compressed_states': compressed_states,
                'hierarchical_repr': hierarchical_repr,
                'strategy': strategy,
                'compression_ratio': compressed_states.shape[1] / seq_length,
                'original_length': seq_length,
                'compressed_length': compressed_states.shape[1]
            }
        
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")