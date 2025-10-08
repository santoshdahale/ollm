"""
Intelligent Attention Routing for oLLM
Dynamically routes different parts of sequences to optimal attention mechanisms.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import time
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class AttentionMetrics:
    """Metrics for attention mechanism performance"""
    computation_time: float
    memory_usage: float
    quality_score: float
    mechanism_name: str
    sequence_length: int
    timestamp: float

class AttentionRouter:
    """
    Intelligent router that selects optimal attention mechanisms based on content and constraints.
    """
    
    def __init__(self, available_mechanisms: Optional[Dict[str, Any]] = None):
        # Initialize available attention mechanisms
        self.mechanisms = available_mechanisms or {
            'full': self._full_attention,
            'sliding_window': self._sliding_window_attention,
            'sparse': self._sparse_attention,
            'multiscale': self._multiscale_attention,
            'hybrid': self._hybrid_attention
        }
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.routing_decisions = []
        
        # Configuration
        self.memory_threshold_gb = 6.0
        self.quality_threshold = 0.95
        self.adaptive_routing = True
        
        # Routing statistics
        self.stats = {
            "total_routes": 0,
            "mechanism_usage": defaultdict(int),
            "quality_maintained": [],
            "performance_gains": []
        }
        
        # Learned patterns
        self.content_patterns = {}
        self.optimal_mechanisms = {}
    
    def route_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None,
                       position_ids: Optional[torch.Tensor] = None,
                       context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Route attention computation to optimal mechanism based on input characteristics.
        """
        
        # Analyze input characteristics
        characteristics = self._analyze_input(query, key, value, attention_mask, context)
        
        # Choose optimal mechanism
        mechanism_name = self._choose_mechanism(characteristics)
        
        # Execute attention with chosen mechanism
        start_time = time.time()
        result = self.mechanisms[mechanism_name](query, key, value, attention_mask, position_ids)
        computation_time = time.time() - start_time
        
        # Track performance
        self._record_performance(mechanism_name, characteristics, computation_time, result)
        
        # Update routing statistics
        self.stats["total_routes"] += 1
        self.stats["mechanism_usage"][mechanism_name] += 1
        
        return result
    
    def _analyze_input(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze input characteristics to inform routing decisions"""
        
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        characteristics = {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "sequence_length": seq_len,
            "head_dim": head_dim,
            "total_elements": query.numel() + key.numel() + value.numel(),
            "estimated_memory_gb": (query.numel() + key.numel() + value.numel()) * 4 / (1024**3),  # fp32 estimate
        }
        
        # Content analysis
        characteristics.update(self._analyze_content(query, key, value))
        
        # Context analysis
        if context:
            characteristics.update({
                "content_type": context.get("content_type", "general"),
                "priority": context.get("priority", "normal"),
                "quality_requirement": context.get("quality_requirement", 0.95)
            })
        
        return characteristics
    
    def _analyze_content(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Dict[str, Any]:
        """Analyze content patterns in the attention tensors"""
        
        # Statistical analysis
        query_stats = {
            "query_mean": query.mean().item(),
            "query_std": query.std().item(),
            "query_entropy": self._calculate_entropy(query),
        }
        
        key_stats = {
            "key_mean": key.mean().item(),
            "key_std": key.std().item(),
            "key_entropy": self._calculate_entropy(key),
        }
        
        value_stats = {
            "value_mean": value.mean().item(),
            "value_std": value.std().item(),
            "value_entropy": self._calculate_entropy(value),
        }
        
        # Pattern analysis
        pattern_analysis = {
            "locality_score": self._calculate_locality_score(query, key),
            "sparsity_potential": self._calculate_sparsity_potential(query, key),
            "repetition_score": self._calculate_repetition_score(value),
        }
        
        return {**query_stats, **key_stats, **value_stats, **pattern_analysis}
    
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """Calculate approximated entropy of tensor values"""
        # Discretize tensor values into bins for entropy calculation
        flat = tensor.flatten()
        hist = torch.histc(flat, bins=100)
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # Remove zero probabilities
        
        if len(probs) == 0:
            return 0.0
        
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        return entropy
    
    def _calculate_locality_score(self, query: torch.Tensor, key: torch.Tensor) -> float:
        """Calculate how local the attention patterns are likely to be"""
        # Simplified locality analysis - measures correlation between adjacent tokens
        seq_len = query.shape[-2]
        if seq_len < 2:
            return 0.0
        
        # Compare adjacent query vectors
        query_flat = query.flatten(0, 1)  # [batch*heads, seq_len, head_dim]
        adjacent_similarities = []
        
        for i in range(min(seq_len - 1, 10)):  # Limit computation
            sim = torch.cosine_similarity(query_flat[:, i], query_flat[:, i + 1], dim=-1)
            adjacent_similarities.append(sim.mean().item())
        
        return sum(adjacent_similarities) / len(adjacent_similarities) if adjacent_similarities else 0.0
    
    def _calculate_sparsity_potential(self, query: torch.Tensor, key: torch.Tensor) -> float:
        """Calculate potential for sparse attention patterns"""
        # Estimate how sparse attention would be by sampling attention scores
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        if seq_len < 4:
            return 0.0
        
        # Sample a small subset for efficiency
        sample_size = min(seq_len, 16)
        indices = torch.randperm(seq_len)[:sample_size]
        
        sample_query = query[:, :, indices]
        sample_key = key[:, :, indices]
        
        # Compute attention scores
        scores = torch.matmul(sample_query, sample_key.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = torch.softmax(scores, dim=-1)
        
        # Calculate sparsity (proportion of near-zero scores)
        threshold = 0.01
        sparse_count = (scores < threshold).sum().item()
        total_count = scores.numel()
        
        return sparse_count / total_count if total_count > 0 else 0.0
    
    def _calculate_repetition_score(self, value: torch.Tensor) -> float:
        """Calculate how repetitive the value vectors are"""
        seq_len = value.shape[-2]
        if seq_len < 2:
            return 0.0
        
        # Sample pairs of value vectors and compute similarity
        sample_size = min(seq_len, 10)
        similarities = []
        
        for i in range(sample_size - 1):
            for j in range(i + 1, min(i + 5, sample_size)):  # Compare with nearby vectors
                if j < seq_len:
                    sim = torch.cosine_similarity(
                        value[:, :, i].flatten(),
                        value[:, :, j].flatten(),
                        dim=0
                    )
                    similarities.append(sim.item())
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _choose_mechanism(self, characteristics: Dict[str, Any]) -> str:
        """Choose optimal attention mechanism based on characteristics"""
        
        seq_len = characteristics["sequence_length"]
        memory_usage = characteristics["estimated_memory_gb"]
        locality_score = characteristics.get("locality_score", 0.0)
        sparsity_potential = characteristics.get("sparsity_potential", 0.0)
        
        # Memory-constrained choice
        if memory_usage > self.memory_threshold_gb:
            if locality_score > 0.7:
                return "sliding_window"
            elif sparsity_potential > 0.6:
                return "sparse"
            else:
                return "multiscale"
        
        # Quality-focused choice
        quality_requirement = characteristics.get("quality_requirement", 0.95)
        if quality_requirement > 0.98:
            if seq_len <= 2048:
                return "full"
            else:
                return "hybrid"  # Best quality for long sequences
        
        # Performance-focused choice
        if seq_len <= 512:
            return "full"
        elif seq_len <= 2048:
            if locality_score > 0.6:
                return "sliding_window"
            else:
                return "sparse"
        elif seq_len <= 8192:
            if sparsity_potential > 0.5:
                return "sparse"
            else:
                return "multiscale"
        else:
            return "hybrid"  # Best for very long sequences
    
    def _full_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None,
                       position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard full attention mechanism"""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        result = torch.matmul(attention_weights, value)
        
        return result
    
    def _sliding_window_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None,
                                 position_ids: Optional[torch.Tensor] = None,
                                 window_size: int = 512) -> torch.Tensor:
        """Sliding window attention with configurable window size"""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        result = torch.zeros_like(query)
        
        for i in range(seq_len):
            # Define window boundaries
            start_idx = max(0, i - window_size // 2)
            end_idx = min(seq_len, i + window_size // 2 + 1)
            
            # Extract window
            window_key = key[:, :, start_idx:end_idx]
            window_value = value[:, :, start_idx:end_idx]
            current_query = query[:, :, i:i+1]
            
            # Compute attention within window
            scores = torch.matmul(current_query, window_key.transpose(-2, -1)) / math.sqrt(head_dim)
            attention_weights = torch.softmax(scores, dim=-1)
            window_result = torch.matmul(attention_weights, window_value)
            
            result[:, :, i:i+1] = window_result
        
        return result
    
    def _sparse_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         position_ids: Optional[torch.Tensor] = None,
                         sparsity_pattern: str = "strided") -> torch.Tensor:
        """Sparse attention with configurable sparsity patterns"""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Create sparsity mask
        if sparsity_pattern == "strided":
            stride = max(1, seq_len // 32)  # Adapt stride to sequence length
            mask = self._create_strided_mask(seq_len, stride)
        elif sparsity_pattern == "local_global":
            mask = self._create_local_global_mask(seq_len)
        else:
            mask = self._create_random_mask(seq_len, sparsity=0.1)
        
        # Compute full attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply sparsity mask
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax and compute result
        attention_weights = torch.softmax(scores, dim=-1)
        result = torch.matmul(attention_weights, value)
        
        return result
    
    def _multiscale_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None,
                             position_ids: Optional[torch.Tensor] = None,
                             scales: List[int] = None) -> torch.Tensor:
        """Multi-scale attention at different resolutions"""
        scales = scales or [1, 2, 4, 8]
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Split heads across scales
        heads_per_scale = num_heads // len(scales)
        results = []
        
        head_start = 0
        for scale in scales:
            head_end = head_start + heads_per_scale
            if head_start >= num_heads:
                break
            
            # Extract heads for this scale
            scale_query = query[:, head_start:head_end]
            scale_key = key[:, head_start:head_end]
            scale_value = value[:, head_start:head_end]
            
            # Downsample for this scale
            if scale > 1:
                scale_query = scale_query[:, :, ::scale]
                scale_key = scale_key[:, :, ::scale]
                scale_value = scale_value[:, :, ::scale]
            
            # Compute attention at this scale
            scale_result = self._full_attention(scale_query, scale_key, scale_value)
            
            # Upsample back if needed
            if scale > 1:
                # Simple upsampling - repeat each token
                upsampled = scale_result.repeat_interleave(scale, dim=-2)
                # Trim to original sequence length
                upsampled = upsampled[:, :, :seq_len]
                results.append(upsampled)
            else:
                results.append(scale_result)
            
            head_start = head_end
        
        # Handle remaining heads with scale 1
        if head_start < num_heads:
            remaining_query = query[:, head_start:]
            remaining_key = key[:, head_start:]
            remaining_value = value[:, head_start:]
            
            remaining_result = self._full_attention(remaining_query, remaining_key, remaining_value)
            results.append(remaining_result)
        
        # Concatenate results
        return torch.cat(results, dim=1)
    
    def _hybrid_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Hybrid attention combining multiple mechanisms"""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Split sequence into segments
        segment_size = 1024
        results = []
        
        for start_idx in range(0, seq_len, segment_size):
            end_idx = min(start_idx + segment_size, seq_len)
            
            # Extract segment
            seg_query = query[:, :, start_idx:end_idx]
            seg_key = key[:, :, start_idx:end_idx]
            seg_value = value[:, :, start_idx:end_idx]
            
            seg_len = end_idx - start_idx
            
            # Choose mechanism for this segment
            if seg_len <= 256:
                seg_result = self._full_attention(seg_query, seg_key, seg_value)
            elif seg_len <= 512:
                seg_result = self._sliding_window_attention(seg_query, seg_key, seg_value, window_size=256)
            else:
                seg_result = self._sparse_attention(seg_query, seg_key, seg_value)
            
            results.append(seg_result)
        
        return torch.cat(results, dim=-2)
    
    def _create_strided_mask(self, seq_len: int, stride: int) -> torch.Tensor:
        """Create strided sparsity mask"""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        for i in range(seq_len):
            # Local attention
            start_local = max(0, i - 32)
            end_local = min(seq_len, i + 33)
            mask[i, start_local:end_local] = True
            
            # Strided attention
            for j in range(i, seq_len, stride):
                mask[i, j] = True
            for j in range(i, -1, -stride):
                mask[i, j] = True
        
        return mask
    
    def _create_local_global_mask(self, seq_len: int) -> torch.Tensor:
        """Create local + global sparsity mask"""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        for i in range(seq_len):
            # Local attention (window of 128)
            start_local = max(0, i - 64)
            end_local = min(seq_len, i + 65)
            mask[i, start_local:end_local] = True
            
            # Global attention (every 64th token)
            for j in range(0, seq_len, 64):
                mask[i, j] = True
        
        return mask
    
    def _create_random_mask(self, seq_len: int, sparsity: float) -> torch.Tensor:
        """Create random sparsity mask"""
        mask = torch.rand(seq_len, seq_len) > sparsity
        
        # Ensure diagonal is always attended
        mask.fill_diagonal_(True)
        
        return mask
    
    def _record_performance(self, mechanism_name: str, characteristics: Dict[str, Any],
                           computation_time: float, result: torch.Tensor):
        """Record performance metrics for learning"""
        
        # Estimate quality (simplified - could use more sophisticated metrics)
        quality_score = 1.0 - min(0.1, computation_time / 1.0)  # Simple time-based quality estimate
        
        metrics = AttentionMetrics(
            computation_time=computation_time,
            memory_usage=characteristics["estimated_memory_gb"],
            quality_score=quality_score,
            mechanism_name=mechanism_name,
            sequence_length=characteristics["sequence_length"],
            timestamp=time.time()
        )
        
        self.performance_history[mechanism_name].append(metrics)
        self.stats["quality_maintained"].append(quality_score)
        
        # Keep history bounded
        if len(self.performance_history[mechanism_name]) > 100:
            self.performance_history[mechanism_name] = self.performance_history[mechanism_name][-100:]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        
        avg_quality = sum(self.stats["quality_maintained"]) / max(len(self.stats["quality_maintained"]), 1)
        
        mechanism_performance = {}
        for mechanism, metrics_list in self.performance_history.items():
            if metrics_list:
                avg_time = sum(m.computation_time for m in metrics_list) / len(metrics_list)
                avg_quality_mech = sum(m.quality_score for m in metrics_list) / len(metrics_list)
                mechanism_performance[mechanism] = {
                    "avg_computation_time": avg_time,
                    "avg_quality": avg_quality_mech,
                    "usage_count": len(metrics_list)
                }
        
        return {
            "total_routes": self.stats["total_routes"],
            "mechanism_usage": dict(self.stats["mechanism_usage"]),
            "average_quality": avg_quality,
            "mechanism_performance": mechanism_performance,
            "adaptive_routing_enabled": self.adaptive_routing
        }
    
    def optimize_routing(self):
        """Optimize routing decisions based on performance history"""
        if not self.adaptive_routing:
            return
        
        # Analyze performance patterns
        best_mechanisms = {}
        
        for mechanism, metrics_list in self.performance_history.items():
            if len(metrics_list) < 5:  # Need sufficient data
                continue
            
            # Group by sequence length ranges
            length_ranges = [(0, 512), (512, 2048), (2048, 8192), (8192, float('inf'))]
            
            for min_len, max_len in length_ranges:
                relevant_metrics = [m for m in metrics_list 
                                  if min_len < m.sequence_length <= max_len]
                
                if len(relevant_metrics) < 3:
                    continue
                
                # Calculate combined score (balance time and quality)
                scores = []
                for m in relevant_metrics:
                    # Lower time is better, higher quality is better
                    score = m.quality_score / max(m.computation_time, 0.001)
                    scores.append(score)
                
                avg_score = sum(scores) / len(scores)
                range_key = f"{min_len}-{max_len}"
                
                if range_key not in best_mechanisms or avg_score > best_mechanisms[range_key][1]:
                    best_mechanisms[range_key] = (mechanism, avg_score)
        
        # Update optimal mechanisms
        self.optimal_mechanisms = {k: v[0] for k, v in best_mechanisms.items()}
        
        print(f"Routing optimization completed. Optimal mechanisms: {self.optimal_mechanisms}")


class ContentAwareAttentionRouter(AttentionRouter):
    """
    Advanced attention router that considers content semantics for routing decisions.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Content type classifiers
        self.content_classifiers = {
            "code": self._is_code_content,
            "structured": self._is_structured_content,
            "natural_language": self._is_natural_language,
            "repetitive": self._is_repetitive_content
        }
        
        # Content-specific optimizations
        self.content_optimizations = {
            "code": "sparse",           # Code has structured patterns
            "structured": "multiscale", # Structured data benefits from hierarchy
            "natural_language": "sliding_window", # Natural text has locality
            "repetitive": "hybrid"      # Repetitive content needs special handling
        }
    
    def _analyze_content(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Dict[str, Any]:
        """Enhanced content analysis with semantic understanding"""
        base_analysis = super()._analyze_content(query, key, value)
        
        # Classify content type
        content_type = self._classify_content(query, key, value)
        base_analysis["detected_content_type"] = content_type
        
        # Content-specific metrics
        if content_type == "code":
            base_analysis["code_structure_score"] = self._analyze_code_structure(value)
        elif content_type == "structured":
            base_analysis["structure_regularity"] = self._analyze_structure_regularity(value)
        elif content_type == "natural_language":
            base_analysis["language_flow_score"] = self._analyze_language_flow(value)
        
        return base_analysis
    
    def _classify_content(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> str:
        """Classify the type of content being processed"""
        
        # Test each classifier
        content_scores = {}
        for content_type, classifier in self.content_classifiers.items():
            content_scores[content_type] = classifier(query, key, value)
        
        # Return type with highest score
        return max(content_scores.keys(), key=lambda k: content_scores[k])
    
    def _is_code_content(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> float:
        """Detect if content is likely code (structured, sparse patterns)"""
        # Code typically has high sparsity and low entropy in certain dimensions
        sparsity = self._calculate_sparsity_potential(query, key)
        entropy = self._calculate_entropy(value)
        
        # Code score combines high sparsity with moderate entropy
        code_score = sparsity * (1.0 - min(entropy / 10.0, 1.0))
        return code_score
    
    def _is_structured_content(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> float:
        """Detect structured content (tables, JSON, etc.)"""
        # Structured content has regular patterns and repetition
        repetition = self._calculate_repetition_score(value)
        regularity = self._calculate_regularity_score(key)
        
        return (repetition + regularity) / 2.0
    
    def _is_natural_language(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> float:
        """Detect natural language content"""
        # Natural language has high locality and moderate entropy
        locality = self._calculate_locality_score(query, key)
        entropy = self._calculate_entropy(value)
        
        # Natural language sweet spot: high locality, moderate entropy
        lang_score = locality * min(entropy / 8.0, 1.0)
        return lang_score
    
    def _is_repetitive_content(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> float:
        """Detect highly repetitive content"""
        repetition = self._calculate_repetition_score(value)
        return repetition
    
    def _calculate_regularity_score(self, tensor: torch.Tensor) -> float:
        """Calculate how regular/periodic the tensor patterns are"""
        # Simplified regularity detection using autocorrelation
        seq_len = tensor.shape[-2]
        if seq_len < 4:
            return 0.0
        
        # Flatten sequence dimension
        flat = tensor.flatten(0, 1).flatten(1)  # [batch*heads, seq_len*head_dim]
        
        regularities = []
        for period in [2, 3, 4, 8]:  # Test common periods
            if period >= seq_len:
                continue
            
            # Calculate correlation with shifted version
            shifted = torch.roll(flat, period, dims=-1)
            correlation = torch.corrcoef(torch.stack([flat.flatten(), shifted.flatten()]))[0, 1]
            regularities.append(abs(correlation.item()) if not torch.isnan(correlation) else 0.0)
        
        return max(regularities) if regularities else 0.0
    
    def _analyze_code_structure(self, value: torch.Tensor) -> float:
        """Analyze code-like structural patterns"""
        # Code has hierarchical structure - analyze value patterns at different scales
        seq_len = value.shape[-2]
        if seq_len < 8:
            return 0.0
        
        # Look for hierarchical patterns
        structure_scores = []
        for window in [4, 8, 16]:
            if window >= seq_len:
                continue
            
            # Calculate variance within windows vs between windows
            num_windows = seq_len // window
            within_var = 0.0
            between_var = 0.0
            
            window_means = []
            for i in range(num_windows):
                start_idx = i * window
                end_idx = min(start_idx + window, seq_len)
                window_data = value[:, :, start_idx:end_idx]
                window_mean = window_data.mean()
                window_var = window_data.var()
                
                window_means.append(window_mean)
                within_var += window_var
            
            if len(window_means) > 1:
                between_var = torch.tensor(window_means).var().item()
                structure_score = between_var / (within_var + 1e-8)
                structure_scores.append(structure_score)
        
        return max(structure_scores) if structure_scores else 0.0
    
    def _analyze_structure_regularity(self, value: torch.Tensor) -> float:
        """Analyze regularity in structured data"""
        return self._calculate_regularity_score(value)
    
    def _analyze_language_flow(self, value: torch.Tensor) -> float:
        """Analyze natural language flow patterns"""
        # Natural language has smooth transitions between adjacent tokens
        seq_len = value.shape[-2]
        if seq_len < 2:
            return 0.0
        
        # Calculate smoothness of transitions
        transitions = []
        for i in range(min(seq_len - 1, 20)):  # Limit computation
            current = value[:, :, i]
            next_token = value[:, :, i + 1]
            
            # Calculate transition smoothness (cosine similarity)
            similarity = torch.cosine_similarity(current.flatten(), next_token.flatten(), dim=0)
            transitions.append(similarity.item())
        
        # Natural language should have moderate similarity between adjacent tokens
        avg_transition = sum(transitions) / len(transitions) if transitions else 0.0
        
        # Optimal range for natural language: 0.3 - 0.7
        if 0.3 <= avg_transition <= 0.7:
            return 1.0 - abs(avg_transition - 0.5) * 2  # Peak at 0.5
        else:
            return max(0.0, 1.0 - abs(avg_transition - 0.5) * 4)  # Penalty outside range
    
    def _choose_mechanism(self, characteristics: Dict[str, Any]) -> str:
        """Enhanced mechanism selection considering content type"""
        
        # Get content-aware recommendation
        detected_content = characteristics.get("detected_content_type", "natural_language")
        content_recommendation = self.content_optimizations.get(detected_content, "sliding_window")
        
        # Get base recommendation
        base_recommendation = super()._choose_mechanism(characteristics)
        
        # Combine recommendations with content-awareness having higher weight
        seq_len = characteristics["sequence_length"]
        memory_usage = characteristics["estimated_memory_gb"]
        
        # Memory constraints override content preferences
        if memory_usage > self.memory_threshold_gb:
            if seq_len > 4096:
                return "multiscale"
            elif detected_content == "code":
                return "sparse"
            else:
                return "sliding_window"
        
        # For shorter sequences, content type matters more
        if seq_len <= 1024:
            return content_recommendation
        
        # For longer sequences, balance content type with base recommendation
        content_weight = 0.7
        base_weight = 0.3
        
        # Simple voting mechanism
        recommendations = {
            content_recommendation: content_weight,
            base_recommendation: base_weight
        }
        
        return max(recommendations.keys(), key=lambda k: recommendations[k])