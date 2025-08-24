#export PYTORCH_CUDA_CUBLASLT_DISABLE=1
#export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python llama.py
