# gds_loader.py
import json, os
import torch
from torch.utils.dlpack import from_dlpack
import cupy as cp
import kvikio

DTYPE_MAP = {
    "float16": cp.float16,
    "bfloat16": cp.float16, #cp.dtype('bfloat16'),
    "float32": cp.float32,
    "float64": cp.float64,
    "int8": cp.int8,
    "int32": cp.int32,
}

class GDSWeights:
    def __init__(self, path: str): #<model_dir>/gds_export/
        self.path = path
        manifest_path = os.path.join(path, 'manifest.json')
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def load_param_to_cuda(self, name: str) -> torch.Tensor:
        meta = self.manifest[name]
        path, shape, dtype = os.path.join(self.path, meta["path"]), meta["shape"], meta["dtype"]
        return self.load_from_disk_to_cuda(path, shape, dtype)

    def load_from_disk_to_cuda(self, path, shape, dtype): #str, list, str
        cp_dtype = DTYPE_MAP[dtype]
        n_elems = 1
        for s in shape:
            n_elems *= s
        nbytes = n_elems * cp.dtype(cp_dtype).itemsize

        # Allocate on GPU
        with cp.cuda.Device(0):
            buf = cp.empty(n_elems, dtype=cp_dtype)

        # DMA read directly into GPU buffer
        with kvikio.CuFile(path, "r") as f:
            # Read raw bytes straight into GPU memory
            n = f.read(buf)
            if n != nbytes:
                raise IOError(f"Short read: {n} of {nbytes} bytes from {path}")

        # Reshape and hand to torch via DLPack
        buf = buf.reshape(shape)
        t = from_dlpack(buf.toDlpack())  # torch.cuda.Tensor shares memory
        return t    


    def has(self, name: str) -> bool:
        return name in self.manifest



if __name__=="__main__":
    q = GDSWeights("./gds_export/")
    x = q.load_param_to_cuda("model.layers.0.self_attn.q_proj.weight")
    print(x, x.dtype)
