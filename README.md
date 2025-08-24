# localLLM or oxllm or crocollm, cllm, ollm
curl -s https://pypi.org/pypi/<name>/json | jq .

| Model   | Weights | KV cache | Hidden states | Emb+Head | **Total**    | 
| ------- | ------- | -------- | ------------- | -------- | ------------ |
| **1B**  | 2 GB    | 12.6 GB  | 0.4 GB        | 1.0 GB   | **\~16 GB**  |
| **8B**  | 16 GB   | 52.4 GB  | 0.8 GB        | 2.0 GB   | **\~71 GB**  |
| **70B** | 140 GB  | 262 GB   | 1.6 GB        | 4.0 GB   | **\~408 GB** |

## llama3-1B "meta-llama/Llama-3.2-1B-Instruct"

## llama3-3B (meta-llama/Llama-3.2-3B-Instruct) with model weights on SSD NVMe
- on simple test (30 tokens generation with simple output) took
  no parallel = 58s, loading next in thread(2 total) = 49s, up to 3 in parallel = 48s
- on 10k test (max_tokens=30) took: 1 next in thread = 5min52s (after 1st token, following ones took only 2/3 seconds each)


## llama3-8B (meta-llama/Llama-3.1-8B-Instruct). Default: NoKVCache, attention(q_block_size=64, k_block_size=512)
- on simple test (input=46, output=30 tokens) took: 1NextInThread =  3min54s
- on 10k test (max_tokens=30) took: 1NextInThread  = 8min57s (after 1st token, following ones took 7-8 seconds each)

run as PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  python llama.py