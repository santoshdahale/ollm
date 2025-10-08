#!/usr/bin/env python3
"""
Example usage of optimization profiles with oLLM.
Demonstrates how to use different optimization configurations for various scenarios.
"""

import torch
import time
from ollm import Inference
from ollm.optimization_profiles import get_profile, auto_select_profile, list_profiles

def demonstrate_profile_usage():
    """Demonstrate using different optimization profiles"""
    
    print("=== oLLM Optimization Profiles Demo ===\n")
    
    # Show available profiles
    print("Available optimization profiles:")
    list_profiles()
    
    print("\n" + "="*50 + "\n")
    
    # Auto-select profile
    recommended_profile = auto_select_profile()
    print(f"Auto-recommended profile: {recommended_profile}")
    
    # Get profile configuration
    config = get_profile(recommended_profile)
    print(f"Profile description: {config['description']}")
    print(f"Memory pool size: {config['memory_pool_size_gb']}GB")
    print(f"Max sequence length: {config['max_sequence_length']:,} tokens")
    print(f"Attention method: {config['attention_method']}")
    
    return config

def example_memory_optimized():
    """Example using memory-optimized profile for limited GPU memory"""
    
    print("\n=== Memory-Optimized Example ===")
    
    config = get_profile("memory_optimized")
    print(f"Using profile: {config['name']}")
    print(f"Memory budget: {config['memory_pool_size_gb']}GB")
    
    # Example initialization (would use real model in practice)
    example_code = '''
# Initialize with memory-optimized settings
inference = Inference(
    model_id="llama3-8B-chat",
    device="cuda:0",
    enable_optimizations=True
)

# Configure for memory efficiency
optimization_config = {
    "memory_pool_size_gb": 3.0,
    "kv_compression": "quantization",
    "compression_bits": 4,
    "attention_method": "sliding_window",
    "window_size": 1024,
    "max_batch_size": 2,
    "prefetch_distance": 1
}

# Generate with memory constraints
result = inference.generate_optimized(
    input_text="Explain renewable energy benefits",
    max_new_tokens=150,
    optimization_config=optimization_config
)

print(f"Generated: {result}")

# Check memory usage
stats = inference.get_optimization_stats()
print(f"Memory usage: {stats['memory']['current_memory_gb']:.1f}GB")
'''
    
    print("Code example:")
    print(example_code)

def example_speed_optimized():
    """Example using speed-optimized profile for maximum throughput"""
    
    print("\n=== Speed-Optimized Example ===")
    
    config = get_profile("speed_optimized")
    print(f"Using profile: {config['name']}")
    print(f"Target: Maximum throughput with {config['memory_pool_size_gb']}GB")
    
    example_code = '''
# Initialize for maximum speed
inference = Inference(
    model_id="llama3-8B-chat",
    device="cuda:0", 
    enable_optimizations=True
)

# Load draft model for speculative decoding
draft_model = load_draft_model("llama3-1B-chat")
inference.set_draft_model(draft_model)

# Configure for speed
optimization_config = {
    "memory_pool_size_gb": 8.0,
    "attention_method": "sparse",
    "kv_compression": "pruning",
    "speculative_candidates": 4,
    "max_batch_size": 8,
    "prefetch_distance": 4
}

# High-throughput generation
results = []
start_time = time.time()

for prompt in batch_prompts:
    result = inference.generate_optimized(
        input_text=prompt,
        max_new_tokens=100,
        optimization_config=optimization_config
    )
    results.append(result)

total_time = time.time() - start_time
throughput = len(batch_prompts) / total_time

print(f"Throughput: {throughput:.1f} requests/second")
'''
    
    print("Code example:")
    print(example_code)

def example_long_context():
    """Example using long context profile for very long sequences"""
    
    print("\n=== Long Context Example ===")
    
    config = get_profile("long_context")
    print(f"Using profile: {config['name']}")
    print(f"Max sequence length: {config['max_sequence_length']:,} tokens")
    
    example_code = '''
# Initialize for long context processing
inference = Inference(
    model_id="llama3-8B-chat",
    device="cuda:0",
    enable_optimizations=True
)

# Configure for very long sequences
optimization_config = {
    "max_sequence_length": 131072,  # 128k tokens
    "attention_method": "hierarchical",
    "kv_compression": "clustering",
    "context_compression_ratio": 0.3,
    "chunk_size": 2048,
    "enable_streaming": True
}

# Process very long document
long_document = load_long_document("100k_token_document.txt")

# Use streaming for memory efficiency
result = inference.generate_optimized(
    input_text=long_document + "\\n\\nSummarize the key points:",
    max_new_tokens=500,
    strategy="streaming",
    optimization_config=optimization_config
)

print(f"Summary: {result}")

# Check compression stats
stats = inference.get_optimization_stats()
compression_ratio = stats.get('context_compression_ratio', 1.0)
print(f"Context compressed by {(1-compression_ratio)*100:.1f}%")
'''
    
    print("Code example:")
    print(example_code)

def example_production_deployment():
    """Example production deployment with monitoring"""
    
    print("\n=== Production Deployment Example ===")
    
    config = get_profile("production")
    print(f"Using profile: {config['name']}")
    print("Features: Stability, monitoring, adaptive optimization")
    
    example_code = '''
import asyncio
from ollm import Inference
from ollm.optimizations import DynamicBatcher, SystemMonitor

# Production-ready setup
class ProductionInference:
    def __init__(self):
        self.inference = Inference(
            model_id="llama3-8B-chat",
            device="cuda:0",
            enable_optimizations=True
        )
        
        # Setup production configuration
        self.config = {
            "memory_pool_size_gb": 7.0,
            "max_batch_size": 8,
            "batch_timeout_ms": 50.0,
            "kv_compression": "quantization",
            "compression_bits": 8,
            "enable_monitoring": True,
            "adaptation_interval": 200
        }
        
        # Initialize monitoring
        self.monitor = SystemMonitor(monitoring_interval=1.0)
        self.monitor.start_monitoring()
        
        # Setup batching for concurrent requests
        self.batcher = DynamicBatcher(
            model=self.inference.model,
            tokenizer=self.inference.tokenizer,
            max_batch_size=self.config["max_batch_size"]
        )
        self.batcher.start_processing()
    
    async def process_request(self, request_id, prompt, **kwargs):
        """Process single request through batcher"""
        
        def callback(req_id, result):
            # Handle result (e.g., send to client)
            print(f"Request {req_id} completed: {result[:50]}...")
        
        self.batcher.add_request(
            request_id=request_id,
            input_text=prompt,
            callback=callback,
            **kwargs
        )
    
    def get_health_status(self):
        """Get system health metrics"""
        return {
            "batcher_stats": self.batcher.get_stats(),
            "system_metrics": self.monitor.get_current_metrics(),
            "optimization_stats": self.inference.get_optimization_stats()
        }

# Usage
async def main():
    service = ProductionInference()
    
    # Process concurrent requests
    requests = [
        ("req_1", "What is machine learning?"),
        ("req_2", "Explain neural networks in detail."),
        ("req_3", "How does backpropagation work?"),
    ]
    
    tasks = []
    for req_id, prompt in requests:
        task = service.process_request(req_id, prompt, max_new_tokens=200)
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    # Monitor health
    health = service.get_health_status()
    print(f"System health: {health}")

# Run production service
asyncio.run(main())
'''
    
    print("Code example:")
    print(example_code)

def example_custom_profile():
    """Example creating and using custom profiles"""
    
    print("\n=== Custom Profile Example ===")
    
    from ollm.optimization_profiles import create_custom_profile
    
    # Create custom profile based on balanced
    custom_config = create_custom_profile(
        "balanced",
        # Custom overrides
        memory_pool_size_gb=10.0,
        max_batch_size=16,
        speculative_candidates=6,
        kv_compression="clustering",
        attention_method="multiscale",
        description="High-performance custom profile"
    )
    
    print(f"Custom profile: {custom_config['name']}")
    print(f"Description: {custom_config['description']}")
    print(f"Memory pool: {custom_config['memory_pool_size_gb']}GB")
    print(f"Batch size: {custom_config['max_batch_size']}")
    print(f"Attention: {custom_config['attention_method']}")
    
    example_code = '''
# Create custom optimization profile
from ollm.optimization_profiles import create_custom_profile

custom_profile = create_custom_profile(
    base_profile="balanced",
    # Customize for your specific needs
    memory_pool_size_gb=12.0,
    max_batch_size=16,
    attention_method="multiscale",
    scales=[1, 2, 4, 8],
    kv_compression="clustering",
    num_clusters=1024,
    speculative_candidates=6,
    description="Custom high-performance profile"
)

# Use custom profile
inference = Inference(
    model_id="llama3-8B-chat",
    enable_optimizations=True
)

result = inference.generate_optimized(
    input_text="Your prompt here", 
    optimization_config=custom_profile
)
'''
    
    print("Code example:")
    print(example_code)

def benchmark_profiles():
    """Benchmark different profiles (mock demonstration)"""
    
    print("\n=== Profile Benchmarking ===")
    
    profiles_to_test = ["memory_optimized", "balanced", "speed_optimized"]
    
    print("Simulated benchmark results:")
    print("Profile | Memory (GB) | Speed (tok/s) | Max Context")
    print("-" * 50)
    
    # Mock benchmark results
    results = {
        "memory_optimized": {"memory": 3.2, "speed": 12.4, "context": 4096},
        "balanced": {"memory": 5.8, "speed": 18.7, "context": 8192},
        "speed_optimized": {"memory": 7.9, "speed": 31.2, "context": 8192}
    }
    
    for profile_name in profiles_to_test:
        config = get_profile(profile_name)
        result = results[profile_name]
        
        print(f"{profile_name:15} | {result['memory']:8.1f} | {result['speed']:10.1f} | {result['context']:8,}")
    
    print("\nBenchmarking code example:")
    
    benchmark_code = '''
import time
from ollm import Inference
from ollm.optimization_profiles import get_profile

def benchmark_profile(profile_name, test_prompts):
    """Benchmark a specific optimization profile"""
    
    config = get_profile(profile_name)
    
    inference = Inference(
        model_id="llama3-8B-chat",
        enable_optimizations=True
    )
    
    # Warmup
    inference.generate_optimized("Warmup", max_new_tokens=10, optimization_config=config)
    
    # Benchmark
    start_time = time.time()
    total_tokens = 0
    
    for prompt in test_prompts:
        result = inference.generate_optimized(
            input_text=prompt,
            max_new_tokens=100,
            optimization_config=config
        )
        total_tokens += len(result.split())
    
    elapsed_time = time.time() - start_time
    
    stats = inference.get_optimization_stats()
    
    return {
        "profile": profile_name,
        "tokens_per_second": total_tokens / elapsed_time,
        "memory_usage_gb": stats['memory']['current_memory_gb'],
        "cache_hit_rate": stats.get('prefetcher', {}).get('cache_hit_rate', 0),
        "total_time": elapsed_time
    }

# Run benchmarks
profiles = ["memory_optimized", "balanced", "speed_optimized"]
test_prompts = ["What is AI?", "Explain ML", "How do neural networks work?"]

for profile in profiles:
    result = benchmark_profile(profile, test_prompts)
    print(f"{profile}: {result['tokens_per_second']:.1f} tok/s, {result['memory_usage_gb']:.1f}GB")
'''
    
    print(benchmark_code)

def main():
    """Main demonstration function"""
    
    # Demonstrate basic profile usage
    config = demonstrate_profile_usage()
    
    # Show specific profile examples
    example_memory_optimized()
    example_speed_optimized()
    example_long_context()
    example_production_deployment()
    example_custom_profile()
    benchmark_profiles()
    
    print("\n" + "="*60)
    print("OPTIMIZATION PROFILES SUMMARY")
    print("="*60)
    
    print("""
Key Features:
• 6 pre-configured optimization profiles
• Automatic profile selection based on system specs
• Custom profile creation with inheritance
• Production-ready configurations with monitoring
• Benchmark tools for performance evaluation

Quick Start:
1. Auto-select: profile = auto_select_profile()
2. Get config: config = get_profile(profile)  
3. Initialize: inference = Inference(model_id, enable_optimizations=True)
4. Generate: result = inference.generate_optimized(text, optimization_config=config)

Profiles:
• memory_optimized: For 4-6GB GPU memory
• speed_optimized: For maximum throughput (8GB+)
• balanced: General purpose (6-8GB) 
• long_context: For 100k+ token sequences
• production: Stable deployment settings
• research: Experimental features

For detailed documentation, see OPTIMIZATION_ENHANCEMENTS.md
""")

if __name__ == "__main__":
    main()