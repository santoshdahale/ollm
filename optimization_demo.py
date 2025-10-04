#!/usr/bin/env python3
"""
Example script demonstrating oLLM optimization enhancements.
Shows usage of memory pooling, KV compression, attention optimizations,
speculative decoding, prefetching, and adaptive optimization.
"""

import torch
import time
import asyncio
from ollm import Inference
from ollm.optimizations import *

def demonstrate_memory_pool():
    """Demonstrate GPU memory pool optimization"""
    print("=== Memory Pool Demonstration ===")
    
    # Initialize memory pool
    pool = GPUMemoryPool(pool_size_gb=4.0)
    manager = MemoryManager()
    
    # Allocate and use tensors
    print("Allocating tensors through memory pool...")
    
    tensors = []
    for i in range(10):
        tensor = manager.allocate_tensor(
            (1024, 4096), 
            torch.float16, 
            name=f"tensor_{i}"
        )
        tensors.append(tensor)
    
    # Get statistics
    report = manager.get_memory_report()
    print(f"Peak memory usage: {report['peak_memory_gb']:.2f} GB")
    print(f"Pool utilization: {report['pool_stats']['utilization']:.1%}")
    
    # Release tensors
    for i, tensor in enumerate(tensors):
        manager.release_tensor(tensor, name=f"tensor_{i}")
    
    print("Memory pool demonstration complete.\n")

def demonstrate_kv_compression():
    """Demonstrate KV cache compression"""
    print("=== KV Cache Compression Demonstration ===")
    
    # Test different compression methods
    methods = ["quantization", "pruning", "clustering"]
    
    for method in methods:
        print(f"\nTesting {method} compression:")
        
        cache = CompressedKVCache(
            compression_method=method,
            bits=8 if method == "quantization" else None,
            prune_ratio=0.3 if method == "pruning" else None,
            num_clusters=256 if method == "clustering" else None
        )
        
        # Simulate KV states
        batch_size, num_heads, seq_len, head_dim = 1, 32, 1024, 64
        key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Update cache (triggers compression)
        start_time = time.time()
        cache.update(key_states, value_states, layer_idx=0, 
                    cache_kwargs={"compression_threshold": 512})
        compression_time = time.time() - start_time
        
        print(f"  Compression time: {compression_time*1000:.1f}ms")
        print(f"  Usable length: {cache.get_usable_length(0)}")

def demonstrate_attention_optimizations():
    """Demonstrate attention mechanism optimizations"""
    print("=== Attention Optimizations Demonstration ===")
    
    batch_size, num_heads, seq_len, head_dim = 2, 16, 2048, 64
    
    # Create test tensors
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    
    # Test different attention mechanisms
    mechanisms = {
        "sliding_window": SlidingWindowAttention(window_size=512),
        "sparse": SparseAttention(sparsity_pattern="strided"),
        "multiscale": MultiScaleAttention(scales=[1, 2, 4]),
        "adaptive": AdaptiveAttention()
    }
    
    for name, mechanism in mechanisms.items():
        print(f"\nTesting {name} attention:")
        
        start_time = time.time()
        with torch.no_grad():
            output = mechanism(query, key, value)
        computation_time = time.time() - start_time
        
        print(f"  Computation time: {computation_time*1000:.1f}ms")
        print(f"  Output shape: {output.shape}")
        
        # Estimate memory usage
        memory_gb = AttentionOptimizer.estimate_memory_usage(
            batch_size, num_heads, seq_len, head_dim
        )
        print(f"  Estimated memory: {memory_gb:.2f}GB")

def demonstrate_speculative_decoding():
    """Demonstrate speculative decoding (requires models)"""
    print("=== Speculative Decoding Demonstration ===")
    
    print("Note: This requires loaded models. Showing configuration example:")
    
    # Example configuration (would need actual models)
    config_example = """
    # Load main and draft models
    main_model = MyLargeModel()
    draft_model = MySmallModel()  # 2-4x smaller
    
    # Initialize speculative decoder
    decoder = SpeculativeDecoder(
        main_model=main_model,
        draft_model=draft_model,
        num_candidates=4,
        acceptance_threshold=0.8
    )
    
    # Generate with speculation
    result = decoder.generate(
        input_ids=input_tokens,
        max_new_tokens=100
    )
    
    print(f"Speedup: {result['speedup_ratio']:.1f}x")
    print(f"Acceptance rate: {result['acceptance_rate']:.1%}")
    """
    
    print(config_example)

async def demonstrate_streaming():
    """Demonstrate streaming inference"""
    print("=== Streaming Inference Demonstration ===")
    
    # Mock streaming input
    async def mock_input_stream():
        chunks = [
            "The benefits of renewable energy include ",
            "reduced greenhouse gas emissions, ",
            "energy independence, and ",
            "long-term cost savings. ",
            "Solar and wind power are ",
            "becoming increasingly affordable."
        ]
        
        for chunk in chunks:
            await asyncio.sleep(0.1)  # Simulate streaming delay
            yield chunk
    
    print("Simulating streaming input processing:")
    
    # This would use actual models in practice
    print("Input chunks:")
    async for chunk in mock_input_stream():
        print(f"  Received: '{chunk}'")
    
    print("\nStreaming processing would generate responses in real-time.\n")

def demonstrate_dynamic_batching():
    """Demonstrate dynamic batching"""
    print("=== Dynamic Batching Demonstration ===")
    
    # Mock batcher for demonstration
    class MockBatcher:
        def __init__(self):
            self.requests = []
            self.stats = {
                "total_requests": 0,
                "average_batch_size": 0,
                "throughput_requests_per_second": 0,
                "padding_ratio": 0.15
            }
        
        def add_request(self, request_id, input_text, **kwargs):
            self.requests.append({
                "id": request_id,
                "text": input_text,
                "length": len(input_text.split())
            })
            self.stats["total_requests"] += 1
            return request_id
        
        def get_stats(self):
            return self.stats
    
    batcher = MockBatcher()
    
    # Add sample requests
    requests = [
        ("req_1", "What is AI?"),
        ("req_2", "Explain machine learning in detail with examples."),
        ("req_3", "Hello"),
        ("req_4", "How does deep learning work and what are its applications?"),
        ("req_5", "Good morning")
    ]
    
    print("Adding requests to batcher:")
    for req_id, text in requests:
        batcher.add_request(req_id, text, max_new_tokens=50)
        print(f"  {req_id}: '{text}' ({len(text.split())} words)")
    
    # Show batching efficiency
    lengths = [len(text.split()) for _, text in requests]
    print(f"\nRequest lengths: {lengths}")
    print("Batcher would group similar lengths to minimize padding.")
    
    stats = batcher.get_stats()
    print(f"Padding ratio: {stats['padding_ratio']:.1%}")

def demonstrate_adaptive_optimization():
    """Demonstrate adaptive optimization"""
    print("=== Adaptive Optimization Demonstration ===")
    
    # Mock optimizer
    optimizer = AdaptiveOptimizer(
        model=None,  # Would be actual model
        device="cuda:0",
        monitoring_window=10
    )
    
    # Simulate performance metrics
    print("Simulating performance metrics collection:")
    
    scenarios = [
        {"name": "Normal Load", "tokens_per_sec": 15.2, "memory_gb": 4.1, "seq_len": 512},
        {"name": "High Memory", "tokens_per_sec": 8.3, "memory_gb": 7.8, "seq_len": 1024},
        {"name": "Long Context", "tokens_per_sec": 5.1, "memory_gb": 6.2, "seq_len": 4096},
        {"name": "Speed Critical", "tokens_per_sec": 3.2, "memory_gb": 3.9, "seq_len": 256}
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Tokens/sec: {scenario['tokens_per_sec']}")
        print(f"  Memory: {scenario['memory_gb']:.1f}GB")
        print(f"  Sequence length: {scenario['seq_len']}")
        
        # Mock bottleneck analysis
        if scenario['memory_gb'] > 7.0:
            bottleneck = "memory"
            strategy = "memory_optimized"
        elif scenario['tokens_per_sec'] < 6.0:
            bottleneck = "speed"
            strategy = "speed_optimized" 
        elif scenario['seq_len'] > 2048:
            bottleneck = "attention"
            strategy = "long_context"
        else:
            bottleneck = "balanced"
            strategy = "balanced"
        
        print(f"  Detected bottleneck: {bottleneck}")
        print(f"  Recommended strategy: {strategy}")

def demonstrate_integrated_usage():
    """Demonstrate integrated usage with optimized inference"""
    print("=== Integrated Usage Demonstration ===")
    
    print("Example of using optimized inference:")
    
    code_example = """
# Initialize with optimizations enabled
inference = Inference(
    model_id="llama3-8B-chat",
    device="cuda:0", 
    enable_optimizations=True
)

# Configure optimization profile
optimization_config = {
    'prefetch_distance': 3,
    'kv_compression': 'quantization',
    'compression_bits': 8,
    'max_batch_size': 6
}

# Generate with optimizations
result = inference.generate_optimized(
    input_text="Explain the benefits of renewable energy",
    max_new_tokens=200,
    temperature=0.7,
    optimization_config=optimization_config
)

# Get optimization statistics
stats = inference.get_optimization_stats()
print(f"Memory usage: {stats['memory']['current_memory_gb']:.1f}GB")
print(f"Cache hit rate: {stats['prefetcher']['cache_hit_rate']:.1%}")
"""
    
    print(code_example)

def main():
    """Main demonstration function"""
    print("oLLM Optimization Enhancements Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_memory_pool()
    demonstrate_kv_compression()
    
    if torch.cuda.is_available():
        demonstrate_attention_optimizations()
    else:
        print("=== Attention Optimizations ===")
        print("CUDA not available, skipping GPU-dependent demonstrations.\n")
    
    demonstrate_speculative_decoding()
    
    # Run async demonstration
    asyncio.run(demonstrate_streaming())
    
    demonstrate_dynamic_batching()
    demonstrate_adaptive_optimization()
    demonstrate_integrated_usage()
    
    print("=" * 50)
    print("Demonstration complete!")
    print("\nFor production usage:")
    print("1. Install: pip install -e .")
    print("2. Load model: inference = Inference('llama3-8B-chat', enable_optimizations=True)")
    print("3. Generate: result = inference.generate_optimized('Your prompt here')")
    print("4. Monitor: stats = inference.get_optimization_stats()")

if __name__ == "__main__":
    main()