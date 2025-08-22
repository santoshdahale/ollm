# localLLM
 
| Model   | Weights | KV cache | Hidden states | Emb+Head | **Total**    | 
| ------- | ------- | -------- | ------------- | -------- | ------------ |
| **1B**  | 2 GB    | 12.6 GB  | 0.4 GB        | 1.0 GB   | **\~16 GB**  |
| **8B**  | 16 GB   | 52.4 GB  | 0.8 GB        | 2.0 GB   | **\~71 GB**  |
| **70B** | 140 GB  | 262 GB   | 1.6 GB        | 4.0 GB   | **\~408 GB** |

## llama3-1B "meta-llama/Llama-3.2-1B-Instruct"

## llama3-3B (meta-llama/Llama-3.2-3B-Instruct) with model weights on SSD NVMe
- on simple test (30 tokens generation with simple output) took
  no parallel = 58s, loading next in thread = 49s, up to 3 in parallel = 48s

- on 10k test (max_tokens=30) took: 1 next in thread = 5min52s (after 1st token, following ones took only 2/3 seconds each)


## llama3-8B (meta-llama/Llama-3.1-8B-Instruct)



