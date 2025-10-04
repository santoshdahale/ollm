<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://ollm.s3.us-east-1.amazonaws.com/files/logo2.png">
    <img alt="vLLM" src="https://ollm.s3.us-east-1.amazonaws.com/files/logo2.png" width=52%>
  </picture>
</p>

<h3 align="center">
LLM Inference for Large-Context Offline Workloads
</h3>

oLLM is a lightweight Python library for large-context LLM inference, built on top of Huggingface Transformers and PyTorch. It enables running models like [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b), [qwen3-next-80B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) or [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on 100k context using ~$200 consumer GPU with 8GB VRAM.  No quantization is used—only fp16/bf16 precision. 
<p dir="auto"><em>Latest updates (0.6.0)</em> 🔥</p>
<ul dir="auto">
<li><b>🚀 Advanced Optimization Suite</b> - 9 comprehensive optimization modules for memory, speed, and long-context efficiency</li>
<li><b>📊 Optimization Profiles</b> - Pre-configured settings for memory-optimized, speed-optimized, balanced, and production scenarios</li>
<li><b>🧠 Intelligent Memory Management</b> - GPU memory pooling, KV cache compression (quantization/pruning/clustering)</li>
<li><b>⚡ Advanced Attention</b> - Sliding window, sparse, multi-scale, and adaptive attention mechanisms</li>
<li><b>🎯 Speculative Decoding</b> - Parallel token generation with draft model verification for faster inference</li>
<li><b>💾 Smart Prefetching</b> - Asynchronous layer loading and memory-aware prefetching</li>
<li><b>📈 Dynamic Batching</b> - Intelligent request batching with length-based bucketing</li>
<li><b>🌊 Streaming Inference</b> - Process infinite-length sequences with bounded memory</li>
<li><b>📱 Auto-Adaptation</b> - Automatic performance monitoring and optimization strategy adjustment</li>
</ul>

<p dir="auto"><em>Previous updates (0.5.0)</em></p>
<ul dir="auto">
<li>Multimodal <b>gemma3-12B</b> (image+text) added. <a href="https://github.com/Mega4alik/ollm/blob/main/example_multimodality.py">[sample with image]</a> </li>
<li>.safetensor files are now read without `mmap` so they no longer consume RAM through page cache</li>
<li>qwen3-next-80B DiskCache support added</li>
<li><b>qwen3-next-80B</b> (160GB model) added with <span style="color:blue">⚡️1tok/2s</span> throughput (our fastest model so far)</li>
<li>gpt-oss-20B flash-attention-like implementation added to reduce VRAM usage </li>
<li>gpt-oss-20B chunked MLP added to reduce VRAM usage </li>
</ul>

---
###  8GB Nvidia 3060 Ti Inference memory usage:

| Model   | Weights | Context length | KV cache |  Baseline VRAM (no offload) | oLLM GPU VRAM | oLLM Disk (SSD) |
| ------- | ------- | -------- | ------------- | ------------ | ---------------- | --------------- |
| [qwen3-next-80B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | 160 GB (bf16) | 50k | 20 GB | ~190 GB   | ~7.5 GB | 180 GB  |
| [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b) | 13 GB (packed bf16) | 10k | 1.4 GB | ~40 GB   | ~7.3GB | 15 GB  |
| [gemma3-12B](https://huggingface.co/google/gemma-3-12b-it)  | 25 GB (bf16) | 50k   | 18.5 GB          | ~45 GB   | ~6.7 GB       | 43 GB  |
| [llama3-1B-chat](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)  | 2 GB (fp16) | 100k   | 12.6 GB          | ~16 GB   | ~5 GB       | 15 GB  |
| [llama3-3B-chat](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  | 7 GB (fp16) | 100k  | 34.1 GB | ~42 GB   | ~5.3 GB     | 42 GB |
| [llama3-8B-chat](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  | 16 GB (fp16) | 100k  | 52.4 GB | ~71 GB   | ~6.6 GB     | 69 GB  |

<small>By "Baseline" we mean typical inference without any offloading</small>

How do we achieve this:

**Core Techniques:**
- Loading layer weights from SSD directly to GPU one by one
- Offloading KV cache to SSD and loading back directly to GPU, no quantization or PagedAttention
- Offloading layer weights to CPU if needed
- FlashAttention-2 with online softmax. Full attention matrix is never materialized. 
- Chunked MLP. Intermediate upper projection layers may get large, so we chunk MLP as well

**Advanced Optimizations (New in 0.6.0):**
- **GPU Memory Pool**: Reduces fragmentation with pre-allocated memory blocks and LRU caching
- **KV Cache Compression**: 4-8bit quantization, importance-based pruning, clustering compression
- **Advanced Attention**: Sliding window (O(n) complexity), sparse patterns, multi-scale hierarchical
- **Speculative Decoding**: Draft model verification for 2-4x speedup on generation tasks
- **Smart Prefetching**: Adaptive layer loading with memory pressure awareness
- **Context Compression**: Hierarchical compression for 100k+ token sequences
- **Dynamic Batching**: Length-based bucketing and adaptive batch sizing
- **Streaming Processing**: Infinite sequence handling with bounded memory usage 
---
Typical use cases include:
- Analyze contracts, regulations, and compliance reports in one pass
- Summarize or extract insights from massive patient histories or medical literature
- Process very large log files or threat reports locally
- Analyze historical chats to extract the most common issues/questions users have
---
Supported **Nvidia GPUs**: Ampere (RTX 30xx, A30, A4000,  A10),  Ada Lovelace (RTX 40xx,  L4), Hopper (H100), and newer

## Getting Started

It is recommended to create venv or conda environment first
```bash
python3 -m venv ollm_env
source ollm_env/bin/activate
```

Install oLLM with `pip install ollm` or [from source](https://github.com/Mega4alik/ollm):

```bash
git clone https://github.com/Mega4alik/ollm.git
cd ollm
pip install -e .
pip install kvikio-cu{cuda_version} Ex, kvikio-cu12
```
> 💡 **Note**  
> **qwen3-next** requires 4.57.0.dev version of transformers to be installed as `pip install git+https://github.com/huggingface/transformers.git`


## Examples

### Basic Usage

```python
from ollm import Inference, file_get_contents, TextStreamer
o = Inference("llama3-1B-chat", device="cuda:0", logging=True) #llama3-1B/3B/8B-chat, gpt-oss-20B, qwen3-next-80B
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2) #(optional) offload some layers to CPU for speed boost
past_key_values = o.DiskCache(cache_dir="./kv_cache/") #set None if context is small
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

messages = [{"role":"system", "content":"You are helpful AI assistant"}, {"role":"user", "content":"List planets"}]
input_ids = o.tokenizer.apply_chat_template(messages, reasoning_effort="minimal", tokenize=True, add_generation_prompt=True, return_tensors="pt").to(o.device)
outputs = o.model.generate(input_ids=input_ids,  past_key_values=past_key_values, max_new_tokens=500, streamer=text_streamer).cpu()
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
```

### Optimized Inference (New!)

```python
from ollm import Inference
from ollm.optimization_profiles import get_profile, auto_select_profile

# Auto-select best profile for your system
profile_name = auto_select_profile()  # Returns "memory_optimized", "balanced", or "speed_optimized"
config = get_profile(profile_name)

# Initialize with optimizations
inference = Inference("llama3-8B-chat", device="cuda:0", enable_optimizations=True)
inference.ini_model(models_dir="./models/")

# Generate with optimizations
result = inference.generate_optimized(
    input_text="Explain quantum computing in detail",
    max_new_tokens=300,
    optimization_config=config
)

print(f"Generated with {profile_name} profile: {result}")

# Check optimization stats
stats = inference.get_optimization_stats()
print(f"Memory usage: {stats['memory']['current_memory_gb']:.1f}GB")
print(f"KV cache compression: {stats.get('kv_compression_ratio', 1.0):.2f}")
```

### Production Deployment

```python
import asyncio
from ollm import Inference
from ollm.optimizations import DynamicBatcher
from ollm.optimization_profiles import get_profile

# Production setup with monitoring
config = get_profile("production")
inference = Inference("llama3-8B-chat", enable_optimizations=True)
inference.setup_optimizations(config)

# Dynamic batching for concurrent requests
batcher = DynamicBatcher(
    model=inference.model,
    tokenizer=inference.tokenizer,
    max_batch_size=8
)

async def process_request(request_id, prompt):
    def callback(req_id, result):
        print(f"Request {req_id}: {result[:50]}...")
    
    batcher.add_request(request_id, prompt, callback=callback, max_new_tokens=200)

# Process multiple requests concurrently
await asyncio.gather(*[
    process_request("req_1", "What is machine learning?"),
    process_request("req_2", "Explain neural networks"),
    process_request("req_3", "How does backpropagation work?")
])
```

Run basic example: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python example.py`

**Additional Examples:**
- [gemma3-12B image+text](https://github.com/Mega4alik/ollm/blob/main/example_multimodality.py)
- [Optimization profiles usage](https://github.com/Mega4alik/ollm/blob/main/profile_examples.py) 
- [Production deployment demo](https://github.com/Mega4alik/ollm/blob/main/optimization_demo.py)

### Quick Setup

For optimized inference setup:
```bash
# Run optimization setup
chmod +x setup_optimizations.sh
./setup_optimizations.sh

# Run optimization demo
python optimization_demo.py

# Test optimizations
python test_optimizations.py
```

## Roadmap
*For visibility of what's coming next (subject to change)*
- Voxtral-small-24B ASR model coming on Oct 5, Sun
- Qwen3-VL or alternative vision model by Oct 12, Sun
- Qwen3-Next MultiTokenPrediction in R&D
- Efficient weight loading in R&D


## Contact us
If there’s a model you’d like to see supported, feel free to reach out at anuarsh@ailabs.us—I’ll do my best to make it happen.
