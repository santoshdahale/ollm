import json, os, time, math
import torch
from torch.utils.dlpack import from_dlpack
import cupy as cp
import kvikio

stats = None

DTYPE_MAP = {
	"float16": cp.float16,
	"bfloat16": cp.float16, #cp.dtype('bfloat16'),
	"float32": cp.float32,
	"float64": cp.float64,
	"int8": cp.int8,
	"int32": cp.int32,
}

class GDSWeights:
	def __init__(self, path: str, device="cuda:0"):
		self.path = path #<model_dir>/gds_export/
		manifest_path = os.path.join(path, 'manifest.json')
		with open(manifest_path) as f:
			self.manifest = json.load(f)
		self.device = torch.device(device)
		self.offloaded_cpu = {}

	def load_param_to_cuda(self, name: str) -> torch.Tensor:
		meta = self.manifest[name]
		path, shape, dtype = os.path.join(self.path, meta["path"]), meta["shape"], meta["dtype"]
		t = self.get_offloaded_from_cpu_to_cuda(name)
		if t is not None: return t

		if meta.get("packed")=="mxfp4":
			return self.load_mxfp4_from_disk(path, shape, dtype)
		elif meta.get("dtype").startswith("torch"):
			return self.load_torch_from_disk(path)
		else: #kvikio, numpy
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


	def load_torch_from_disk(self, path):
		tensor = torch.load(path, map_location=self.device)
		return tensor

	def load_mxfp4_from_disk(self, path, shape, dtype):
		packed = torch.load(path, map_location=self.device) #{_blocks:t, _scales:t}		
		tensor = convert_moe_packed_tensors(packed["_blocks"], packed["_scales"]).to(self.device)		
		return tensor

	def offload_param_to_cpu(self, name):
		meta = self.manifest[name]
		path, shape, dtype, packed = os.path.join(self.path, meta["path"]), meta["shape"], meta["dtype"], meta.get("packed")
		if packed=="mxfp4" or dtype.startswith("torch"):
			tensor = torch.load(path, map_location="cpu")
		else: #kvikio, numpy
			tensor = self.load_from_disk_to_cuda(path, shape, dtype).cpu() #should be without GPU
		self.offloaded_cpu[name] = {"shape":shape, "dtype":dtype, "packed":packed, "tensor":tensor}

	def get_offloaded_from_cpu_to_cuda(self, name):
		if name in self.offloaded_cpu:
			meta = self.offloaded_cpu[name]
			t, packed = meta["tensor"], meta["packed"]
			t1 = time.perf_counter()
			if packed=="mxfp4":
				tensor = convert_moe_packed_tensors(t["_blocks"].to(self.device), t["_scales"].to(self.device))
			else:
				tensor = t.to(self.device)
			if stats: stats.set("offloaded_cpu_to_cuda", t1)
			return tensor
		return None


#=========================================================================

FP4_VALUES = [
	+0.0,
	+0.5,
	+1.0,
	+1.5,
	+2.0,
	+3.0,
	+4.0,
	+6.0,
	-0.0,
	-0.5,
	-1.0,
	-1.5,
	-2.0,
	-3.0,
	-4.0,
	-6.0,
]

def convert_moe_packed_tensors( #copied from transformers/integrations/mxfp4.py
	blocks,
	scales,
	*,
	dtype: torch.dtype = torch.bfloat16,
	rows_per_chunk: int = 32768 * 1024,  # TODO these values are not here by mistake ;)
) -> torch.Tensor:
	"""
	Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
	pass of GPT_OSS.
	"""

	# Check if blocks and scales are on CPU, and move to GPU if so
	#if not blocks.is_cuda and torch.cuda.is_available():
	#    blocks = blocks.cuda()
	#    scales = scales.cuda()

	scales = scales.to(torch.int32) - 127  # TODO that's because 128=2**7

	assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

	lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

	*prefix_shape, G, B = blocks.shape
	rows_total = math.prod(prefix_shape) * G

	blocks = blocks.reshape(rows_total, B)
	scales = scales.reshape(rows_total, 1)

	out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

	for r0 in range(0, rows_total, rows_per_chunk):
		r1 = min(r0 + rows_per_chunk, rows_total)

		blk = blocks[r0:r1]
		exp = scales[r0:r1]

		# nibble indices -> int64
		idx_lo = (blk & 0x0F).to(torch.long)
		idx_hi = (blk >> 4).to(torch.long)

		sub = out[r0:r1]
		sub[:, 0::2] = lut[idx_lo]
		sub[:, 1::2] = lut[idx_hi]

		torch.ldexp(sub, exp, out=sub)
		del idx_lo, idx_hi, blk, exp, sub

	out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
	del blocks, scales, lut
	return out.transpose(1, 2).contiguous()

#=========================================================================

if __name__=="__main__":
	q = GDSWeights("/media/mega4alik/ssd/models/gpt-oss-20B/gds_export/")
	t = q.load_param_to_cuda("model.layers.0.self_attn.q_proj.weight")
	print(t, t.dtype, t.shape)
