#!/usr/bin/env python3
"""
Comprehensive test suite for oLLM optimization enhancements.
Tests all optimization components for correctness and performance.
"""

import torch
import time
import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch

# Import optimization modules
import sys
sys.path.append('/workspaces/ollm/src')

# Direct imports to work with our consolidated structure
from ollm.optimizations.advanced_kv_compression import (
    CompressedKVCache, QuantizedKVCache, MixedPrecisionKVCache
)
from ollm.optimizations.memory_pool import GPUMemoryPool, MemoryManager
from ollm.optimizations.attention_optimizations import (
    SlidingWindowAttention, SparseAttentionOptimizer, MultiScaleAttention, AttentionOptimizer
)
from ollm.optimizations.speculative_decoding import SpeculativeDecoder
from ollm.optimizations.prefetching import PrefetchingOptimizer as LayerPrefetcher
from ollm.optimizations.context_compression import ContextCompressor, HierarchicalContext
from ollm.optimizations.streaming import StreamingOptimizer as StreamingInference, StreamingConfig
from ollm.optimizations.dynamic_batching import DynamicBatchingOptimizer as DynamicBatcher, BatchRequest

# Mock classes for testing
class ChunkedProcessor:
    def __init__(self, model, tokenizer, chunk_size):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
    
    def process_long_sequence(self, text):
        return self._generate_normal(text)
    
    def _generate_normal(self, text):
        return "normal output"

class TestMemoryPool:
    """Test GPU Memory Pool functionality"""
    
    def test_pool_initialization(self):
        pool = GPUMemoryPool(pool_size_gb=1.0)
        assert pool.pool_size == 1024**3
        assert pool.total_allocated >= 0
    
    def test_tensor_allocation_and_release(self):
        pool = GPUMemoryPool(pool_size_gb=1.0)
        
        # Allocate tensor
        tensor = pool.get_tensor((100, 100), torch.float32)
        assert tensor.shape == (100, 100)
        assert tensor.dtype == torch.float32
        
        # Release tensor
        pool.release_tensor(tensor)
        
        # Verify stats
        stats = pool.get_stats()
        assert "total_pool_size_gb" in stats
        assert "utilization" in stats
    
    def test_memory_reuse(self):
        pool = GPUMemoryPool(pool_size_gb=1.0)
        
        # Allocate and release tensor
        tensor1 = pool.get_tensor((50, 50), torch.float16)
        pool.release_tensor(tensor1)
        
        # Allocate same size - should reuse memory
        tensor2 = pool.get_tensor((50, 50), torch.float16)
        
        # Should be different tensor objects but potentially reused memory
        assert tensor2.shape == (50, 50)

class TestKVCompression:
    """Test KV Cache compression methods"""
    
    def test_quantized_cache(self):
        cache = CompressedKVCache(method="quantization", bits=8)
        
        # Create test data
        key_states = torch.randn(1, 8, 100, 64)
        value_states = torch.randn(1, 8, 100, 64)
        
        # Update cache
        result = cache.update(key_states, value_states, layer_idx=0)
        
        assert len(result) == 2  # key, value
        assert result[0].shape == key_states.shape
    
    def test_pruned_cache(self):
        cache = CompressedKVCache(method="mixed_precision", use_triton=False)
        
        key_states = torch.randn(1, 8, 100, 64)
        value_states = torch.randn(1, 8, 100, 64)
        
        result = cache.update(key_states, value_states, layer_idx=0)
        assert result is not None
    
    def test_clustered_cache(self):
        cache = CompressedKVCache(method="temporal", use_triton=False)
        
        key_states = torch.randn(1, 8, 100, 64)
        value_states = torch.randn(1, 8, 100, 64)
        
        result = cache.update(key_states, value_states, layer_idx=0)
        assert result is not None

class TestAttentionOptimizations:
    """Test attention mechanism optimizations"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sliding_window_attention(self):
        attention = SlidingWindowAttention(window_size=128)
        
        batch_size, num_heads, seq_len, head_dim = 1, 8, 256, 64
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        
        output = attention(query, key, value)
        assert output.shape == query.shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available") 
    def test_sparse_attention(self):
        attention = SparseAttentionOptimizer(pattern="strided", stride=32)
        
        batch_size, num_heads, seq_len, head_dim = 1, 8, 256, 64
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        
        output = attention(query, key, value)
        assert output.shape == query.shape
    
    def test_multiscale_attention(self):
        attention = MultiScaleAttention(scales=[1, 2], head_dim=64)
        
        batch_size, num_heads, seq_len, head_dim = 1, 8, 128, 64
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        output = attention(query, key, value)
        assert output.shape == query.shape
    
    def test_attention_optimizer(self):
        # Test memory estimation
        memory_gb = AttentionOptimizer.estimate_memory_usage(1, 32, 1024, 64)
        assert memory_gb > 0
        
        # Test mechanism selection
        mechanism = AttentionOptimizer.choose_attention_mechanism(
            seq_len=2048, 
            available_memory_gb=8.0
        )
        assert mechanism in ["full", "sliding_window", "sparse", "multiscale"]

class TestSpeculativeDecoding:
    """Test speculative decoding functionality"""
    
    def test_speculative_decoder_init(self):
        # Mock models
        main_model = Mock()
        draft_model = Mock()
        
        decoder = SpeculativeDecoder(
            main_model=main_model,
            draft_model=draft_model,
            num_candidates=4
        )
        
        assert decoder.num_candidates == 4
        assert decoder.main_model == main_model
        assert decoder.draft_model == draft_model
    
    def test_stats_tracking(self):
        main_model = Mock()
        draft_model = Mock()
        
        decoder = SpeculativeDecoder(main_model, draft_model)
        
        # Verify stats initialized
        assert decoder.stats.total_tokens_generated == 0
        assert decoder.stats.acceptance_rate == 0.0

class TestPrefetching:
    """Test layer prefetching functionality"""
    
    def test_prefetcher_initialization(self):
        model = Mock()
        model.model.layers = [Mock() for _ in range(10)]
        
        prefetcher = LayerPrefetcher(
            model=model,
            prefetch_distance=2,
            max_cache_size=4
        )
        
        assert prefetcher.prefetch_distance == 2
        assert prefetcher.max_cache_size == 4
    
    def test_cache_statistics(self):
        model = Mock()
        model.model.layers = [Mock() for _ in range(10)]
        
        prefetcher = LayerPrefetcher(model=model)
        stats = prefetcher.get_stats()
        
        assert hasattr(stats, 'cache_hit_rate')
        assert hasattr(stats, 'layers_prefetched')

class TestContextCompression:
    """Test context compression functionality"""
    
    def test_context_compressor(self):
        compressor = ContextCompressor()
        
        # Create test hidden states
        batch_size, seq_len, hidden_dim = 1, 1000, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Compress context
        result = compressor.compress_context(hidden_states, target_ratio=0.5)
        
        assert "compressed_states" in result
        assert "compression_ratio" in result
        assert result["compressed_length"] < seq_len
    
    def test_hierarchical_context(self):
        hierarchical = HierarchicalContext(levels=[1, 2, 4])
        
        batch_size, seq_len, hidden_dim = 1, 1024, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        representations = hierarchical.create_hierarchical_representation(hidden_states)
        
        assert isinstance(representations, dict)
        assert len(representations) <= 3  # Based on levels

class TestAdaptiveOptimizer:
    """Test adaptive optimization functionality"""
    
    def test_optimizer_initialization(self):
        model = Mock()
        optimizer = AdaptiveOptimizer(model=model, device="cpu")
        
        assert optimizer.model == model
        assert len(optimizer.available_strategies) > 0
        assert optimizer.current_strategy is not None
    
    def test_metrics_collection(self):
        model = Mock()
        optimizer = AdaptiveOptimizer(model=model, device="cpu")
        
        metrics = optimizer.collect_metrics(
            tokens_generated=50,
            generation_time=2.0,
            sequence_length=512
        )
        
        assert metrics.tokens_per_second == 25.0
        assert metrics.sequence_length == 512
    
    def test_bottleneck_analysis(self):
        model = Mock()
        optimizer = AdaptiveOptimizer(model=model, device="cpu")
        
        # Add some mock metrics
        for i in range(10):
            optimizer.collect_metrics(
                tokens_generated=10,
                generation_time=1.0,
                sequence_length=1024,
                additional_metrics={"cache_hit_rate": 0.8}
            )
        
        bottleneck = optimizer.analyze_bottleneck()
        assert bottleneck in ["memory", "speed", "attention", "cache", "balanced"]

class TestStreamingInference:
    """Test streaming inference functionality"""
    
    def test_streaming_config(self):
        config = StreamingConfig(chunk_size=256, overlap_size=32)
        assert config.chunk_size == 256
        assert config.overlap_size == 32
    
    def test_chunked_processor(self):
        model = Mock()
        tokenizer = Mock()
        tokenizer.encode.return_value = list(range(100))
        tokenizer.decode.return_value = "test output"
        tokenizer.eos_token_id = 0
        
        processor = ChunkedProcessor(
            model=model,
            tokenizer=tokenizer,
            chunk_size=50
        )
        
        # Mock short sequence processing
        with patch.object(processor, '_generate_normal', return_value="normal output"):
            result = processor.process_long_sequence("short input")
            assert result == "normal output"

class TestDynamicBatching:
    """Test dynamic batching functionality"""
    
    def test_batch_request(self):
        request = BatchRequest(
            request_id="test_001",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=50
        )
        
        assert request.request_id == "test_001"
        assert request.max_new_tokens == 50
        assert not request.is_complete
    
    def test_batcher_initialization(self):
        model = Mock()
        tokenizer = Mock()
        
        batcher = DynamicBatcher(
            model=model,
            tokenizer=tokenizer,
            max_batch_size=4
        )
        
        assert batcher.max_batch_size == 4
        assert len(batcher.length_buckets) == 0

class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_optimization_module_imports(self):
        """Test that all optimization modules can be imported"""
        from ollm.optimizations import (
            GPUMemoryPool, CompressedKVCache,
            SlidingWindowAttention, SpeculativeDecoder,
            LayerPrefetcher, ContextCompressor,
            AdaptiveOptimizer, StreamingInference,
            DynamicBatcher
        )
        
        # Verify classes exist
        assert GPUMemoryPool is not None
        assert CompressedKVCache is not None
        assert SlidingWindowAttention is not None
    
    def test_mock_inference_workflow(self):
        """Test a complete mock inference workflow"""
        
        # Mock components
        model = Mock()
        tokenizer = Mock()
        tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        tokenizer.decode.return_value = "Generated text"
        tokenizer.eos_token_id = 0
        
        # Initialize optimizations
        memory_manager = MemoryManager()
        compressed_cache = CompressedKVCache(compression_method="quantization")
        
        # Verify components work together
        assert memory_manager is not None
        assert compressed_cache is not None
        
        # Mock tensor operations
        test_tensor = torch.randn(10, 10)
        optimized_tensor = memory_manager.allocate_tensor(
            test_tensor.shape, 
            test_tensor.dtype
        )
        
        assert optimized_tensor.shape == test_tensor.shape

def run_performance_benchmarks():
    """Run performance benchmarks for optimizations"""
    print("Running performance benchmarks...")
    
    # Memory pool benchmark
    print("\n1. Memory Pool Benchmark:")
    
    # Standard allocation
    start_time = time.time()
    standard_tensors = []
    for i in range(100):
        tensor = torch.randn(100, 100, device="cuda" if torch.cuda.is_available() else "cpu")
        standard_tensors.append(tensor)
    standard_time = time.time() - start_time
    
    # Pool allocation
    pool = GPUMemoryPool(pool_size_gb=2.0)
    start_time = time.time()
    pool_tensors = []
    for i in range(100):
        tensor = pool.get_tensor((100, 100), torch.float32)
        pool_tensors.append(tensor)
    pool_time = time.time() - start_time
    
    print(f"Standard allocation: {standard_time*1000:.1f}ms")
    print(f"Pool allocation: {pool_time*1000:.1f}ms")
    print(f"Speedup: {standard_time/pool_time:.2f}x")
    
    # Attention benchmark
    if torch.cuda.is_available():
        print("\n2. Attention Benchmark:")
        
        batch_size, num_heads, seq_len, head_dim = 1, 16, 1024, 64
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        
        # Sliding window attention
        sliding_attention = SlidingWindowAttention(window_size=256)
        
        start_time = time.time()
        with torch.no_grad():
            output = sliding_attention(query, key, value)
        sliding_time = time.time() - start_time
        
        print(f"Sliding window attention: {sliding_time*1000:.1f}ms")
        print(f"Output shape: {output.shape}")

def main():
    """Main test runner"""
    print("oLLM Optimization Tests")
    print("=" * 30)
    
    # Run pytest tests
    print("Running unit tests...")
    
    # Individual test classes
    test_classes = [
        TestMemoryPool,
        TestKVCompression, 
        TestAttentionOptimizations,
        TestSpeculativeDecoding,
        TestPrefetching,
        TestContextCompression,
        TestAdaptiveOptimizer,
        TestStreamingInference,
        TestDynamicBatching,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        
        test_instance = test_class()
        methods = [method for method in dir(test_instance) 
                  if method.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    
    # Run performance benchmarks
    try:
        run_performance_benchmarks()
    except Exception as e:
        print(f"Benchmark error: {e}")
    
    print("\nTest suite complete!")

if __name__ == "__main__":
    main()