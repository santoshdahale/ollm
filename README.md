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

---

## About

oLLM is a lightweight Python library for large-context LLM inference, built on top of Huggingface Transformers and PyTorch. It enables running models like [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on 100k context using ~$200 consumer GPU with 8GB VRAM.  [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b)  is also supported, but without large context (coming). No quantization is used—only fp16/bf16 precision. 

###  8GB Nvidia 3060 Ti 100k context inference memory usage:

| Model   | Weights | KV cache | Hidden states | Baseline VRAM (no offload) | oLLM GPU VRAM | oLLM Disk (SSD) |
| ------- | ------- | -------- | ------------- | ------------ | ---------------- | --------------- |
| [llama3-1B-chat](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)  | 2 GB (fp16)    | 12.6 GB  | 0.4 GB        | ~16 GB   | ~5 GB       | 18 GB  |
| [llama3-3B-chat](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  | 7 GB (fp16)   | 34.1 GB  | 0.61 GB       | ~42 GB   | ~5.3 GB     | 45 GB |
| [llama3-8B-chat](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  | 16 GB (fp16)  | 52.4 GB  | 0.8 GB        | ~71 GB   | ~6.6 GB     | 75 GB  |
| [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b) | 13 GB (packed bf16) |  | 0.6GB  |    | ~6.4GB, large context support is on the way | 20GB  |

<small>By  "Baseline" we mean typical inference without any offloading. Its VRAM usage does not include full attention materialization (it would be **600GB**)</small>

How do we achieve this:

- Loading layer weights from SSD directly to GPU one by one
- Offloading KV cache to SSD and loading back directly to GPU, no quantization or PagedAttention
- Offloading layer weights to RAM if needed
- Chunked attention with online softmax. Full attention matrix is never materialized. 
- Chunked MLP. Intermediate upper projection layers may get large, so we chunk MLP as well 
---
Typical use cases include:
- Analyze contracts, regulations, and compliance reports in one pass
- Summarize or extract insights from massive patient histories or medical literature
- Process very large log files or threat reports locally
- Analyze historical chats to extract the most common issues/questions users have
---
Supported **Nvidia GPUs**: RTX 20xx (Turing) -- only Llama3, RTX 30xx (Ampere), RTX 40xx (Ada Lovelace), T4, L4, A10, and newer

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

## Example

Code snippet sample 

```bash
from ollm import Inference, KVCache
o = Inference("llama3-1B-chat", device="cuda:0") #llama3-1B/3B/8B-chat, gpt-oss-20B
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2) #(optional) offload some layers to RAM for speed increase
past_key_values = KVCache(cache_dir="./kv_cache/", stats=o.stats) #set None for small context

messages = [{"role":"system", "content":"You are helpful AI assistant"}, {"role":"user", "content":"List planets"}]
input_ids = o.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(o.device)
outputs = o.model.generate(input_ids=input_ids,  past_key_values=past_key_values, max_new_tokens=20).cpu()
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
```
or run sample python script as `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python example.py` 

## Contact us
If there’s a model you’d like to see supported, feel free to reach out at anuarsh@ailabs.us—I’ll do my best to make it happen.
