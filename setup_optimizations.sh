#!/bin/bash

# Setup script for oLLM optimization enhancements
# Installs required dependencies and runs basic tests

set -e

echo "=== oLLM Optimization Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc) -eq 1 ]]; then
    echo "Error: Python 3.8+ required"
    exit 1
fi

# Install core dependencies
echo "Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate

# Install optimization dependencies
echo "Installing optimization dependencies..."
pip install psutil asyncio pytest

# Try to install optional GPU dependencies
echo "Installing optional GPU dependencies..."
pip install kvikio || echo "Warning: kvikio not installed (GPU Direct I/O disabled)"

# Install package in development mode
echo "Installing oLLM in development mode..."
pip install -e .

# Run basic import test
echo "Testing imports..."
python3 -c "
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    import ollm
    print('oLLM: imported successfully')
    
    from ollm.optimizations import GPUMemoryPool, CompressedKVCache
    print('Optimizations: imported successfully')
    
    print('✓ All imports successful')
except Exception as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

# Run optimization demo
echo "Running optimization demonstration..."
python3 optimization_demo.py

# Run tests
echo "Running optimization tests..."
python3 test_optimizations.py

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start:"
echo "1. Load model: from ollm import Inference; inf = Inference('llama3-8B-chat', enable_optimizations=True)"
echo "2. Generate: result = inf.generate_optimized('Your prompt here')"
echo "3. Check stats: stats = inf.get_optimization_stats()"
echo ""
echo "For detailed documentation, see OPTIMIZATION_ENHANCEMENTS.md"