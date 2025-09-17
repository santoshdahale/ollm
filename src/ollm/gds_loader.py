import json, os, time, math, re
import torch
from torch.utils.dlpack import from_dlpack
import cupy as cp
import kvikio
#from safetensors import safe_open
from safetensors._safetensors_rust import safe_open

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
		self.offloaded_map = {}

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
		self.offloaded_map[name] = {"shape":shape, "dtype":dtype, "packed":packed, "tensor":tensor}

	def get_offloaded_from_cpu_to_cuda(self, name):
		if name in self.offloaded_map:
			meta = self.offloaded_map[name]
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

class KvikIOLoader:
	def safetensors_read_header(self, path):
		"""Return parsed JSON header (dict) for safetensors file."""
		with open(path, "rb") as fh:
			hdr_len = int.from_bytes(fh.read(8), "little")
			hdr = fh.read(hdr_len).decode("utf-8")
		return json.loads(hdr)

	def dtype_info_from_safetensors(self, dtype_str):
		"""
		Return (cupy_dtype, torch_dtype, bytes_per_elem).
		For BF16 we return cupy uint16 buffer and torch.bfloat16 as final dtype.
		"""
		if dtype_str in ("F32", "float32"):
			return (cp.float32, torch.float32, 4)
		if dtype_str in ("F16", "float16"):
			return (cp.float16, torch.float16, 2)
		if dtype_str in ("BF16", "bfloat16"):
			# CuPy often lacks native bfloat16; read raw uint16 then reinterpret in torch
			return (cp.uint16, torch.bfloat16, 2)
		if dtype_str in ("I32", "int32"):
			return (cp.int32, torch.int32, 4)
		if dtype_str in ("I64", "int64"):
			return (cp.int64, torch.int64, 8)
		raise NotImplementedError(f"dtype {dtype_str} not supported by this helper")

	def __init__(self, filename):
		assert os.path.exists(filename)
		self.filename = filename
		self.meta = self.safetensors_read_header(filename)

	def get_tensor_gpu(self, key, device="cuda:0"):
		"""
		Load tensor `key` from safetensors directly into GPU memory using kvikio + CuPy.
		Falls back to CPU numpy read if something is not supported.
		"""
		info = self.meta[key]
		shape = tuple(info["shape"])
		offset_start, offset_end = info["data_offsets"]
		nbytes_expected = offset_end - offset_start
		cupy_dtype, torch_target_dtype, bytes_per_elem = self.dtype_info_from_safetensors(info["dtype"])

		# cross-check
		n_elems = 1
		for s in shape:
			n_elems *= int(s)
		nbytes_calc = n_elems * bytes_per_elem
		if nbytes_calc != nbytes_expected:
			# header mismatch? warn but prefer header offsets (more authoritative)
			# Sometimes safetensors stores bytes in different ordering—still, prefer offsets.
			# We'll trust header offsets for size.
			pass

		# allocate a raw byte buffer on GPU large enough to hold nbytes_expected
		raw = cp.empty(nbytes_expected, dtype=cp.uint8)

		# Try direct-GPU read. kvikio has slightly different APIs across versions.
		try:
			with kvikio.CuFile(self.filename, "r") as f:
				# Preferred API: pread(buffer, offset)
				try:
					f.pread(raw, offset_start)
				except TypeError:
					# Some kvikio expects (buffer, nbytes, offset)
					f.pread(raw, nbytes_expected, offset_start)
		except Exception as e:
			# If direct GPU read fails, fallback to CPU read + copy
			# (This path always works but is slower.)
			import numpy as np
			with open(self.filename, "rb") as fh:
				fh.seek(offset_start)
				arr = np.fromfile(fh, dtype=np.uint8, count=nbytes_expected)
			# move to GPU
			raw = cp.asarray(arr)

		# Reinterpret raw bytes as the actual dtype/shape
		# If cupy_dtype is uint16 for BF16, we need to view as uint16 then convert to torch.bfloat16
		if cupy_dtype is cp.uint16:
			# view raw (nbytes) as uint16
			arr = raw.view(cp.uint16)
			arr = arr.reshape(shape)
			# convert to torch: use DLPack to move GPU buffer -> torch tensor (uint16)
			torch_uint16 = from_dlpack(arr.toDlpack()).to(device)
			# reinterpret the uint16 storage as bfloat16 (bitwise view)
			# PyTorch doesn't provide a direct reinterpret_view API for dtype, but we can view the storage:
			torch_bf16 = torch_uint16.view(torch.bfloat16)
			return torch_bf16
		else:
			# view raw bytes as cupy_dtype and reshape
			arr = raw.view(cupy_dtype).reshape(shape)
			# transform to torch via DLPack (zero-copy)
			torch_tensor = from_dlpack(arr.toDlpack()).to(device=device)
			# ensure final dtype exactly matches target (some conversions may be needed)
			if torch_tensor.dtype != torch_target_dtype:
				torch_tensor = torch_tensor.to(dtype=torch_target_dtype)
			return torch_tensor



class MoEWeightsLoader(): #qwen3_next
	def __init__(self, path: str, device="cuda:0"):
		self.path = path #<model_dir>	
		index_path = os.path.join(path, 'model.safetensors.index.json')
		with open(index_path) as f: indexes = json.load(f)
		self.manifest, self.safetensors = {}, {}
		for manifest_name, filename in indexes["weight_map"].items():
			match1 = re.search(r"(model\.layers\.\d+\.mlp\.experts\.\d+\.)", manifest_name)
			match2 = re.search(r"(model\.layers\.\d+\.)", manifest_name)
			if match1 or match2:
				base = match1.group(1) if match1 else match2.group(1)
				if base not in self.manifest: self.manifest[base] = {}
				attr_path = manifest_name.replace(base, "")
				self.manifest[base][attr_path] = filename

		self.device = torch.device(device)
		self.offloaded_map = {}

	def load_dict_to_cuda(self, base):
		t = self.get_offloaded_dict_to_cuda(base)
		if t: return t
		return self.load_dict_from_disk(base, device=self.device)

	def offload_dict_to_gpu_cpu(self, base, gpu=False):
		d = self.load_dict_from_disk(base, device=self.device if gpu else 'cpu')
		self.offloaded_map[base] = d

	def get_offloaded_dict_to_cuda(self, base):
		if base in self.offloaded_map:
			d, d2 = self.offloaded_map[base], {}
			for attr_path, tensor in d.items():
				d2[attr_path] = tensor.to(self.device)
			return d2
		return None
	
	def load_dict_from_disk1(self, base, device='cpu'): #legacy base.pt(attr=>tensor)
		return torch.load(self.path+base.replace(".","__")+".pt", map_location=device) #{self_attn.weight=tensor}

	def load_dict_from_disk(self, base, device='cpu'): #original safetensors
		dbase, d = self.manifest[base], {}
		for attr_path, filename in dbase.items():
			d[attr_path] = self.safetensors[filename].get_tensor(base+attr_path).to(device)
		return d

	def preload_layer_safetensors(self, base): #load only couple instead of 48
		del self.safetensors
		self.safetensors = {}
		for base1 in list(self.manifest.keys()):
			if base1.startswith(base):
				for attr_path, filename in self.manifest[base1].items():
					if filename not in self.safetensors:
						self.safetensors[filename] = safe_open(os.path.join(self.path, filename), framework="pt") #KvikIOLoader


#=========================================================================

if __name__=="__main__":
	q = GDSWeights("/media/mega4alik/ssd/models/gpt-oss-20B/gds_export/")
	t = q.load_param_to_cuda("model.layers.0.self_attn.q_proj.weight")
	print(t, t.dtype, t.shape)
