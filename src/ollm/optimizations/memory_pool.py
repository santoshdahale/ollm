"""
GPU Memory Pool Management for efficient memory allocation and reuse.
Reduces memory fragmentation and allocation overhead.
"""
import torch
import threading
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import gc

@dataclass
class MemoryBlock:
    """Represents a memory block in the pool"""
    tensor: torch.Tensor
    size: int
    dtype: torch.dtype
    is_free: bool = True
    last_used: float = 0.0

class GPUMemoryPool:
    """
    Pre-allocate and reuse GPU memory to avoid fragmentation.
    
    Usage:
        pool = GPUMemoryPool(pool_size_gb=6)
        tensor = pool.get_tensor((1024, 1024), torch.float16)
        # ... use tensor ...
        pool.release_tensor(tensor)
    """
    
    def __init__(self, pool_size_gb: float = 6.0, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.pool_size = int(pool_size_gb * 1024**3)  # Convert to bytes
        self.blocks: Dict[int, MemoryBlock] = {}
        self.free_blocks: Dict[Tuple[torch.Size, torch.dtype], List[int]] = {}
        self.allocated_blocks: Dict[int, MemoryBlock] = {}
        self.lock = threading.Lock()
        self.next_id = 0
        self.total_allocated = 0
        
        # Pre-allocate common tensor sizes
        self._preallocate_common_sizes()
    
    def _preallocate_common_sizes(self):
        """Pre-allocate common tensor sizes for model layers"""
        common_shapes = [
            # Common hidden state sizes
            (1, 4096),      # Single token hidden state
            (1, 8192),      # Larger models
            (512, 4096),    # Batch processing
            (1024, 4096),   # Attention matrices
            (4096, 4096),   # Weight matrices
            # KV cache sizes
            (32, 128, 64),  # Multi-head attention
            (32, 256, 64),  # Longer sequences
        ]
        
        common_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        
        for shape in common_shapes:
            for dtype in common_dtypes:
                size_bytes = torch.tensor(1, dtype=dtype).element_size()
                for dim in shape:
                    size_bytes *= dim
                
                if self.total_allocated + size_bytes < self.pool_size:
                    self._allocate_block(shape, dtype)
    
    def _allocate_block(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Allocate a new memory block"""
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        size_bytes = tensor.numel() * tensor.element_size()
        
        block_id = self.next_id
        self.next_id += 1
        
        block = MemoryBlock(
            tensor=tensor,
            size=size_bytes,
            dtype=dtype,
            is_free=True
        )
        
        self.blocks[block_id] = block
        
        # Add to free blocks index
        key = (tuple(shape), dtype)
        if key not in self.free_blocks:
            self.free_blocks[key] = []
        self.free_blocks[key].append(block_id)
        
        self.total_allocated += size_bytes
        return block_id
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        Get a tensor from the pool or allocate new one.
        
        Args:
            shape: Desired tensor shape
            dtype: Desired tensor dtype
            
        Returns:
            Tensor from pool or newly allocated
        """
        with self.lock:
            key = (tuple(shape), dtype)
            
            # Try to find exact match first
            if key in self.free_blocks and self.free_blocks[key]:
                block_id = self.free_blocks[key].pop()
                block = self.blocks[block_id]
                block.is_free = False
                block.last_used = torch.cuda.Event().elapsed_time(torch.cuda.Event())
                self.allocated_blocks[id(block.tensor)] = block
                return block.tensor
            
            # Try to find larger block that can be reshaped
            required_size = torch.tensor(1, dtype=dtype).element_size()
            for dim in shape:
                required_size *= dim
            
            for (existing_shape, existing_dtype), block_ids in self.free_blocks.items():
                if existing_dtype == dtype and block_ids:
                    existing_size = torch.tensor(1, dtype=existing_dtype).element_size()
                    for dim in existing_shape:
                        existing_size *= dim
                    
                    if existing_size >= required_size:
                        block_id = block_ids.pop()
                        block = self.blocks[block_id]
                        
                        # Reshape if possible
                        if block.tensor.numel() >= torch.Size(shape).numel():
                            reshaped = block.tensor.view(shape)
                            block.is_free = False
                            block.last_used = torch.cuda.Event().elapsed_time(torch.cuda.Event())
                            self.allocated_blocks[id(reshaped)] = block
                            return reshaped
            
            # Allocate new block if pool has space
            size_bytes = torch.tensor(1, dtype=dtype).element_size()
            for dim in shape:
                size_bytes *= dim
            
            if self.total_allocated + size_bytes < self.pool_size:
                block_id = self._allocate_block(shape, dtype)
                block = self.blocks[block_id]
                block.is_free = False
                self.allocated_blocks[id(block.tensor)] = block
                return block.tensor
            
            # Pool is full, try garbage collection and retry
            self._cleanup_unused_blocks()
            if self.total_allocated + size_bytes < self.pool_size:
                block_id = self._allocate_block(shape, dtype)
                block = self.blocks[block_id]
                block.is_free = False
                self.allocated_blocks[id(block.tensor)] = block
                return block.tensor
            
            # Fall back to regular allocation (outside pool)
            return torch.empty(shape, dtype=dtype, device=self.device)
    
    def release_tensor(self, tensor: torch.Tensor):
        """Release a tensor back to the pool"""
        with self.lock:
            tensor_id = id(tensor)
            if tensor_id in self.allocated_blocks:
                block = self.allocated_blocks.pop(tensor_id)
                block.is_free = True
                
                # Add back to free blocks
                key = (tuple(tensor.shape), tensor.dtype)
                if key not in self.free_blocks:
                    self.free_blocks[key] = []
                
                # Find the block ID
                for block_id, stored_block in self.blocks.items():
                    if stored_block is block:
                        self.free_blocks[key].append(block_id)
                        break
    
    def _cleanup_unused_blocks(self):
        """Clean up old unused blocks to free memory"""
        gc.collect()
        torch.cuda.empty_cache()
        
        # Remove blocks that haven't been used recently
        current_time = torch.cuda.Event().elapsed_time(torch.cuda.Event())
        threshold = 60000  # 60 seconds in milliseconds
        
        blocks_to_remove = []
        for block_id, block in self.blocks.items():
            if block.is_free and (current_time - block.last_used) > threshold:
                blocks_to_remove.append(block_id)
        
        for block_id in blocks_to_remove:
            block = self.blocks.pop(block_id)
            self.total_allocated -= block.size
            
            # Remove from free blocks index
            for key, block_list in self.free_blocks.items():
                if block_id in block_list:
                    block_list.remove(block_id)
                    break
            
            del block.tensor
    
    def get_stats(self) -> Dict[str, any]:
        """Get memory pool statistics"""
        with self.lock:
            free_blocks = sum(len(blocks) for blocks in self.free_blocks.values())
            allocated_blocks = len(self.allocated_blocks)
            
            return {
                "total_pool_size_gb": self.pool_size / (1024**3),
                "total_allocated_gb": self.total_allocated / (1024**3),
                "free_blocks": free_blocks,
                "allocated_blocks": allocated_blocks,
                "utilization": self.total_allocated / self.pool_size if self.pool_size > 0 else 0
            }
    
    def __del__(self):
        """Cleanup when pool is destroyed"""
        with self.lock:
            for block in self.blocks.values():
                del block.tensor
            self.blocks.clear()
            self.free_blocks.clear()
            self.allocated_blocks.clear()


class MemoryManager:
    """High-level memory management coordinator"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.pool = GPUMemoryPool(device=device)
        self.peak_memory = 0
        self.memory_events = []
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                       name: str = "") -> torch.Tensor:
        """Allocate tensor with tracking"""
        tensor = self.pool.get_tensor(shape, dtype)
        
        # Track memory usage
        current_memory = torch.cuda.memory_allocated(self.device)
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        self.memory_events.append({
            "action": "allocate",
            "name": name,
            "shape": shape,
            "dtype": str(dtype),
            "size_mb": tensor.numel() * tensor.element_size() / (1024**2),
            "total_memory_gb": current_memory / (1024**3)
        })
        
        return tensor
    
    def release_tensor(self, tensor: torch.Tensor, name: str = ""):
        """Release tensor with tracking"""
        size_mb = tensor.numel() * tensor.element_size() / (1024**2)
        self.pool.release_tensor(tensor)
        
        self.memory_events.append({
            "action": "release", 
            "name": name,
            "size_mb": size_mb,
            "total_memory_gb": torch.cuda.memory_allocated(self.device) / (1024**3)
        })
    
    def get_memory_report(self) -> Dict[str, any]:
        """Get comprehensive memory usage report"""
        pool_stats = self.pool.get_stats()
        
        return {
            "peak_memory_gb": self.peak_memory / (1024**3),
            "current_memory_gb": torch.cuda.memory_allocated(self.device) / (1024**3),
            "pool_stats": pool_stats,
            "recent_events": self.memory_events[-10:],  # Last 10 events
            "gpu_memory_summary": torch.cuda.memory_summary(self.device)
        }