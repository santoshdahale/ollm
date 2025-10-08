"""
Optimization configuration profiles for different use cases.
Each profile optimizes for specific scenarios like memory efficiency, speed, or long context.
"""

# Memory-optimized profile for systems with limited GPU memory
MEMORY_OPTIMIZED = {
    "name": "memory_optimized",
    "description": "Optimized for systems with limited GPU memory (4-6GB)",
    
    # Memory management
    "memory_pool_size_gb": 3.0,
    "max_sequence_length": 4096,
    
    # Attention optimization
    "attention_method": "sliding_window",
    "window_size": 1024,
    "overlap_size": 128,
    
    # KV cache compression
    "kv_compression": "quantization",
    "compression_bits": 4,
    "compression_threshold": 512,
    
    # Context compression
    "context_compression_ratio": 0.5,
    "preserve_recent_tokens": 256,
    
    # Prefetching
    "prefetch_distance": 1,
    "max_cache_size": 2,
    
    # Batching
    "max_batch_size": 2,
    "batch_timeout_ms": 100.0,
    
    # Speculative decoding
    "speculative_candidates": 0,  # Disabled to save memory
    
    # Streaming
    "chunk_size": 256,
    "overlap_size": 32
}

# Speed-optimized profile for maximum throughput
SPEED_OPTIMIZED = {
    "name": "speed_optimized", 
    "description": "Optimized for maximum inference speed (8GB+ GPU)",
    
    # Memory management
    "memory_pool_size_gb": 8.0,
    "max_sequence_length": 8192,
    
    # Attention optimization
    "attention_method": "sparse",
    "sparsity_pattern": "strided",
    "stride": 64,
    "local_window": 512,
    
    # KV cache compression
    "kv_compression": "pruning",
    "prune_ratio": 0.2,
    "compression_threshold": 1024,
    
    # Context compression
    "context_compression_ratio": 0.8,
    "preserve_recent_tokens": 512,
    
    # Prefetching
    "prefetch_distance": 4,
    "max_cache_size": 6,
    
    # Batching
    "max_batch_size": 8,
    "batch_timeout_ms": 25.0,
    
    # Speculative decoding
    "speculative_candidates": 4,
    "acceptance_threshold": 0.8,
    
    # Streaming
    "chunk_size": 1024,
    "overlap_size": 128
}

# Balanced profile for general use
BALANCED = {
    "name": "balanced",
    "description": "Balanced optimization for general use cases (6-8GB GPU)",
    
    # Memory management
    "memory_pool_size_gb": 6.0,
    "max_sequence_length": 8192,
    
    # Attention optimization
    "attention_method": "adaptive",
    "short_seq_threshold": 512,
    "long_seq_threshold": 4096,
    
    # KV cache compression
    "kv_compression": "quantization",
    "compression_bits": 8,
    "compression_threshold": 1024,
    
    # Context compression
    "context_compression_ratio": 0.7,
    "preserve_recent_tokens": 384,
    
    # Prefetching
    "prefetch_distance": 2,
    "max_cache_size": 4,
    
    # Batching
    "max_batch_size": 4,
    "batch_timeout_ms": 50.0,
    
    # Speculative decoding
    "speculative_candidates": 2,
    "acceptance_threshold": 0.75,
    
    # Streaming
    "chunk_size": 512,
    "overlap_size": 64
}

# Long context profile for very long sequences
LONG_CONTEXT = {
    "name": "long_context",
    "description": "Optimized for very long sequences (100k+ tokens)",
    
    # Memory management
    "memory_pool_size_gb": 5.0,
    "max_sequence_length": 131072,  # 128k tokens
    
    # Attention optimization
    "attention_method": "hierarchical",
    "levels": [1, 4, 16],
    "level_lengths": [1024, 2048, 4096],
    
    # KV cache compression
    "kv_compression": "clustering",
    "num_clusters": 512,
    "compression_threshold": 2048,
    
    # Context compression
    "context_compression_ratio": 0.3,
    "preserve_recent_tokens": 1024,
    "compression_strategy": "hierarchical",
    
    # Prefetching
    "prefetch_distance": 3,
    "max_cache_size": 3,
    
    # Batching
    "max_batch_size": 2,
    "batch_timeout_ms": 200.0,
    
    # Speculative decoding
    "speculative_candidates": 1,  # Limited for long contexts
    "acceptance_threshold": 0.7,
    
    # Streaming
    "chunk_size": 2048,
    "overlap_size": 256,
    "enable_streaming": True
}

# Research profile for experimentation
RESEARCH = {
    "name": "research",
    "description": "Experimental settings for research and development",
    
    # Memory management
    "memory_pool_size_gb": 10.0,
    "max_sequence_length": 16384,
    
    # Attention optimization
    "attention_method": "multiscale",
    "scales": [1, 2, 4, 8],
    
    # KV cache compression
    "kv_compression": "all",  # Test all methods
    "compression_bits": [4, 8],
    "compression_threshold": 512,
    
    # Context compression
    "context_compression_ratio": 0.6,
    "preserve_recent_tokens": 512,
    "adaptive_compression": True,
    
    # Prefetching
    "prefetch_distance": 3,
    "max_cache_size": 8,
    "adaptive_prefetching": True,
    
    # Batching
    "max_batch_size": 6,
    "batch_timeout_ms": 75.0,
    "adaptive_batching": True,
    
    # Speculative decoding
    "speculative_candidates": 3,
    "acceptance_threshold": 0.8,
    "tree_speculation": True,
    
    # Streaming
    "chunk_size": 768,
    "overlap_size": 96,
    "incremental_processing": True
}

# Production profile for deployment
PRODUCTION = {
    "name": "production",
    "description": "Stable settings for production deployment",
    
    # Memory management
    "memory_pool_size_gb": 7.0,
    "max_sequence_length": 4096,
    
    # Attention optimization
    "attention_method": "sliding_window",
    "window_size": 2048,
    "overlap_size": 256,
    
    # KV cache compression
    "kv_compression": "quantization",
    "compression_bits": 8,
    "compression_threshold": 1024,
    
    # Context compression
    "context_compression_ratio": 0.75,
    "preserve_recent_tokens": 512,
    
    # Prefetching
    "prefetch_distance": 2,
    "max_cache_size": 4,
    
    # Batching
    "max_batch_size": 8,
    "batch_timeout_ms": 50.0,
    
    # Speculative decoding
    "speculative_candidates": 2,
    "acceptance_threshold": 0.8,
    
    # Streaming
    "chunk_size": 512,
    "overlap_size": 64,
    
    # Monitoring and logging
    "enable_monitoring": True,
    "log_performance": True,
    "adaptation_interval": 200
}

# All available profiles
PROFILES = {
    "memory_optimized": MEMORY_OPTIMIZED,
    "speed_optimized": SPEED_OPTIMIZED,
    "balanced": BALANCED,
    "long_context": LONG_CONTEXT,
    "research": RESEARCH,
    "production": PRODUCTION
}

def get_profile(name):
    """Get optimization profile by name"""
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Profile '{name}' not found. Available: {available}")
    return PROFILES[name].copy()

def list_profiles():
    """List all available profiles with descriptions"""
    for name, profile in PROFILES.items():
        print(f"{name}: {profile['description']}")

def create_custom_profile(base_profile, **overrides):
    """Create custom profile based on existing profile with overrides"""
    if isinstance(base_profile, str):
        base_profile = get_profile(base_profile)
    
    custom = base_profile.copy()
    custom.update(overrides)
    custom["name"] = "custom"
    custom["description"] = f"Custom profile based on {base_profile.get('name', 'unknown')}"
    
    return custom

def auto_select_profile(gpu_memory_gb=None, sequence_length=None, use_case=None):
    """Automatically select optimal profile based on system characteristics"""
    
    # Try to detect GPU memory if not provided
    if gpu_memory_gb is None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                gpu_memory_gb = 0
        except:
            gpu_memory_gb = 4  # Conservative default
    
    # Default sequence length
    if sequence_length is None:
        sequence_length = 2048
    
    # Select based on constraints
    if gpu_memory_gb < 6:
        return "memory_optimized"
    elif sequence_length > 32768:
        return "long_context"
    elif use_case == "production":
        return "production"
    elif use_case == "research":
        return "research"
    elif gpu_memory_gb >= 12:
        return "speed_optimized"
    else:
        return "balanced"

# Usage examples
if __name__ == "__main__":
    print("Available optimization profiles:")
    print("=" * 40)
    list_profiles()
    
    print("\nAuto-selected profile for current system:")
    recommended = auto_select_profile()
    profile = get_profile(recommended)
    print(f"Recommended: {profile['name']}")
    print(f"Description: {profile['description']}")
    
    print("\nExample custom profile:")
    custom = create_custom_profile(
        "balanced",
        max_batch_size=16,
        speculative_candidates=6,
        description="High-throughput custom profile"
    )
    print(f"Custom profile: max_batch_size={custom['max_batch_size']}")