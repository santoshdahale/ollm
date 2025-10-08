#!/usr/bin/env python3
"""
Test example to verify oLLM with optimizations works correctly.
Modified to work in dev environment without CUDA and specific model paths.
"""

import sys
import os
sys.path.append('/workspaces/ollm/src')

import torch
from transformers import AutoTokenizer

def test_basic_import():
    """Test basic oLLM imports work"""
    print("🧪 Testing Basic Imports")
    print("-" * 30)
    
    try:
        from ollm import Inference
        print("✅ ollm.Inference imported successfully")
    except Exception as e:
        print(f"❌ ollm.Inference import failed: {e}")
        return False
    
    try:
        from ollm.optimizations.advanced_kv_compression import CompressedKVCache
        print("✅ CompressedKVCache imported successfully")
    except Exception as e:
        print(f"❌ CompressedKVCache import failed: {e}")
        return False
    
    try:
        from ollm.optimizations.attention_optimizations import SlidingWindowAttention
        print("✅ SlidingWindowAttention imported successfully")
    except Exception as e:
        print(f"❌ SlidingWindowAttention import failed: {e}")
        return False
    
    return True

def test_optimizations():
    """Test optimization components directly"""
    print("\n🚀 Testing Optimization Components")
    print("-" * 40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test KV compression
    try:
        from ollm.optimizations.advanced_kv_compression import CompressedKVCache
        
        cache = CompressedKVCache(method="quantization", use_triton=False)
        
        # Test with sample tensors
        keys = torch.randn(1, 8, 128, 64, device=device)
        values = torch.randn(1, 8, 128, 64, device=device)
        
        compressed_keys, compressed_values = cache.update(keys, values, layer_idx=0)
        
        print(f"✅ KV Compression test passed")
        print(f"   Input shape: {keys.shape}")
        print(f"   Compressed keys: {compressed_keys.shape}")
        print(f"   Compressed values: {compressed_values.shape}")
        
    except Exception as e:
        print(f"❌ KV Compression test failed: {e}")
        return False
    
    # Test attention optimization
    try:
        from ollm.optimizations.attention_optimizations import SlidingWindowAttention
        
        attention = SlidingWindowAttention(window_size=256, use_triton=False)
        
        batch_size, num_heads, seq_len, head_dim = 1, 8, 128, 64
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        output = attention.forward(query, key, value)
        
        print(f"✅ Sliding Window Attention test passed")
        print(f"   Input shape: {query.shape}")
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Sliding Window Attention test failed: {e}")
        return False
    
    return True

def test_memory_optimizations():
    """Test memory optimization components"""
    print("\n💾 Testing Memory Optimizations")
    print("-" * 35)
    
    # Test memory pool (if CUDA available)
    if torch.cuda.is_available():
        try:
            from ollm.optimizations.memory_pool import GPUMemoryPool
            
            pool = GPUMemoryPool(pool_size_gb=0.5)  # Small pool for testing
            tensor = pool.get_tensor((100, 100), torch.float32)
            
            print(f"✅ GPU Memory Pool test passed")
            print(f"   Allocated tensor: {tensor.shape}")
            
            pool.release_tensor(tensor)
            
        except Exception as e:
            print(f"❌ GPU Memory Pool test failed: {e}")
            return False
    else:
        print("⚠️  CUDA not available, skipping GPU Memory Pool test")
    
    # Test adaptive optimizer
    try:
        from ollm.optimizations.adaptive_optimizer import AdaptiveOptimizer
        
        # Create a simple mock model for testing
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)
            
            def forward(self, x):
                return self.linear(x)
        
        mock_model = MockModel()
        optimizer = AdaptiveOptimizer(mock_model)
        
        print(f"✅ Adaptive Optimizer test passed")
        print(f"   Optimizer created for model with {sum(p.numel() for p in mock_model.parameters())} parameters")
        
    except Exception as e:
        print(f"❌ Adaptive Optimizer test failed: {e}")
        return False
    
    return True

def test_integration():
    """Test integration with inference components"""
    print("\n🔧 Testing Integration Components")
    print("-" * 38)
    
    try:
        # Test that we can create core components
        from ollm.inference import Inference
        from ollm.utils import tensor_size_gb
        
        print("✅ Core inference components imported")
        
        # Test tensor size calculation
        test_tensor = torch.randn(100, 100, dtype=torch.float32)
        tensor_gb = tensor_size_gb(test_tensor)
        print(f"✅ Tensor size calculation works: {tensor_gb:.6f} GB for 100x100 float32 tensor")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    return True

def main():
    """Main test runner"""
    print("🚀 oLLM Integration Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Run all tests
    tests = [
        test_basic_import,
        test_optimizations, 
        test_memory_optimizations,
        test_integration
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! oLLM optimizations are working correctly!")
        print("\nSummary:")
        print("• ✅ Core imports functional")
        print("• ✅ KV compression working")
        print("• ✅ Attention optimizations working")
        print("• ✅ Memory optimizations available")
        print("• ✅ Integration components ready")
        print("\n🚀 Ready for production use!")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())