<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://ollm.s3.us-east-1.amazonaws.com/files/logo.png">
    <img alt="vLLM" src="https://ollm.s3.us-east-1.amazonaws.com/files/logo.png" width=52%>
  </picture>
</p>

<h3 align="center">
LLM Inference for Large-Context Offline Workloads
</h3>

oLLM is a lightweight Python library for large-context LLM inference, built on top of Huggingface Transformers and PyTorch. It enables running models like [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b), [qwen3-next-80B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) or [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on 100k context using ~$200 consumer GPU with 8GB VRAM.  No quantization is used—only fp16/bf16 precision. 
<p dir="auto"><em>Latest updates (0.4.1)</em> 🔥</p>
<ul dir="auto">
<li>.safetensors files are now read without `mmap` so they no longer consume RAM through page cache</li>
<li>qwen3-next-80B DiskCache support added</li>
<li><b>qwen3-next-80B</b> (160GB model) added with <span style="color:blue">⚡️1tok/2s</span> throughput (our fastest model so far)</li>
<li>Llama3 custom chunked attention replaced with flash-attention2 for stability</li>
<li>gpt-oss-20B flash-attention-like implementation added to reduce VRAM usage </li>
<li>gpt-oss-20B chunked MLP added to reduce VRAM usage </li>
</ul>

---
###  8GB Nvidia 3060 Ti Inference memory usage:

| Model   | Weights | Context length | KV cache |  Baseline VRAM (no offload) | oLLM GPU VRAM | oLLM Disk (SSD) |
| ------- | ------- | -------- | ------------- | ------------ | ---------------- | --------------- |
| [qwen3-next-80B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | 160 GB (bf16) | 50k | 2 GB | ~170 GB   | ~7.5 GB | 162 GB  |
| [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b) | 13 GB (packed bf16) | 10k | 1.4 GB | ~40 GB   | ~7.3GB | 15 GB  |
| [llama3-1B-chat](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)  | 2 GB (fp16) | 100k   | 12.6 GB          | ~16 GB   | ~5 GB       | 15 GB  |
| [llama3-3B-chat](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  | 7 GB (fp16) | 100k  | 34.1 GB | ~42 GB   | ~5.3 GB     | 42 GB |
| [llama3-8B-chat](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  | 16 GB (fp16) | 100k  | 52.4 GB | ~71 GB   | ~6.6 GB     | 69 GB  |

<small>By "Baseline" we mean typical inference without any offloading</small>

How do we achieve this:

- Loading layer weights from SSD directly to GPU one by one
- Offloading KV cache to SSD and loading back directly to GPU, no quantization or PagedAttention
- Offloading layer weights to CPU if needed
- FlashAttention-2 with online softmax. Full attention matrix is never materialized. 
- Chunked MLP. Intermediate upper projection layers may get large, so we chunk MLP as well 
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


## Example

Code snippet sample 

```bash
from ollm import Inference, file_get_contents, TextStreamer
o = Inference("llama3-1B-chat", device="cuda:0") #llama3-1B/3B/8B-chat, gpt-oss-20B, qwen3-next-80B
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
or run sample python script as `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python example.py` 

## Contact us
If there’s a model you’d like to see supported, feel free to reach out at anuarsh@ailabs.us—I’ll do my best to make it happen.

