"""
Enhanced Memory Management for oLLM Optimizations
Provides more intelligent memory pooling, fragmentation detection, and auto-tuning.
"""

import torch
import gc
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import threading
import psutil

class EnhancedGPUMemoryPool:
    """
    Enhanced GPU memory pool with auto-tuning, fragmentation detection, and intelligent cleanup.
    Backward compatible with existing GPUMemoryPool.
    """
    
    def __init__(self, device: str = "cuda:0", pool_size_gb: float = 4.0, 
                 auto_tune: bool = True, fragmentation_threshold: float = 0.3):
        self.device = device
        self.pool_size_gb = pool_size_gb
        self.auto_tune = auto_tune
        self.fragmentation_threshold = fragmentation_threshold
        
        # Memory pool storage
        self.memory_pool = {}
        self.allocated_tensors = {}
        self.free_blocks = defaultdict(list)
        
        # Enhanced tracking
        self.allocation_history = deque(maxlen=1000)
        self.fragmentation_history = deque(maxlen=100)
        self.auto_tune_history = deque(maxlen=50)
        
        # Thread safety
        self.pool_lock = threading.RLock()
        
        # Performance metrics
        self.stats = {
            "total_allocations": 0,
            "cache_hits": 0,
            "fragmentations_detected": 0,
            "auto_tunes": 0,
            "memory_saved_gb": 0.0
        }
        
        # Initialize pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize memory pool with enhanced pre-allocation"""
        try:
            # Detect available GPU memory
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                available_memory = total_memory - torch.cuda.memory_allocated()
                max_pool_size = min(self.pool_size_gb * (1024**3), available_memory * 0.8)
            else:
                max_pool_size = self.pool_size_gb * (1024**3)
            
            # Auto-tune pool size if enabled
            if self.auto_tune:
                self.pool_size_gb = self._calculate_optimal_pool_size()
            
            self._create_initial_blocks()
            
        except Exception as e:
            print(f"Warning: Enhanced memory pool initialization failed: {e}")
            # Fall back to basic initialization
            self._basic_initialization()
    
    def _calculate_optimal_pool_size(self) -> float:
        """Calculate optimal pool size based on system characteristics"""
        try:
            if not torch.cuda.is_available():
                return min(self.pool_size_gb, 2.0)  # Conservative for CPU
            
            # Get GPU memory info
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / (1024**3)
            
            # Auto-tune based on GPU memory
            if total_memory_gb >= 24:  # High-end GPU
                optimal_size = min(total_memory_gb * 0.6, 12.0)
            elif total_memory_gb >= 12:  # Mid-range GPU
                optimal_size = min(total_memory_gb * 0.5, 8.0)
            elif total_memory_gb >= 8:  # Entry-level GPU
                optimal_size = min(total_memory_gb * 0.4, 4.0)
            else:  # Limited GPU
                optimal_size = min(total_memory_gb * 0.3, 2.0)
            
            self.stats["auto_tunes"] += 1
            self.auto_tune_history.append({
                "timestamp": time.time(),
                "old_size": self.pool_size_gb,
                "new_size": optimal_size,
                "total_gpu_memory": total_memory_gb
            })
            
            return optimal_size
            
        except Exception:
            return self.pool_size_gb  # Fall back to original size
    
    def _create_initial_blocks(self):
        """Create initial memory blocks with intelligent sizing"""
        common_sizes = [
            (1024, 1024),      # 1M elements - common tensor size
            (2048, 2048),      # 4M elements - attention matrices
            (4096, 1024),      # 4M elements - embeddings
            (8192, 512),       # 4M elements - compressed representations
            (512, 512),        # 256K elements - small tensors
        ]
        
        total_allocated = 0
        target_size = self.pool_size_gb * (1024**3)
        
        for shape in common_sizes:
            if total_allocated >= target_size * 0.8:  # Use 80% of target
                break
            
            # Allocate multiple blocks of each size
            blocks_per_size = 3
            for _ in range(blocks_per_size):
                try:
                    size_bytes = shape[0] * shape[1] * 4  # fp32 size
                    if total_allocated + size_bytes > target_size:
                        break
                    
                    tensor = torch.empty(shape, dtype=torch.float32, device=self.device)
                    self.free_blocks[shape].append(tensor)
                    total_allocated += size_bytes
                    
                except torch.cuda.OutOfMemoryError:
                    break
        
        print(f"Enhanced memory pool initialized: {total_allocated / (1024**3):.2f}GB allocated")
    
    def _basic_initialization(self):
        """Basic initialization fallback"""
        self.memory_pool = {}
        self.free_blocks = defaultdict(list)
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, name: str = "") -> torch.Tensor:
        """Get tensor from pool with enhanced tracking and auto-optimization"""
        with self.pool_lock:
            self.stats["total_allocations"] += 1
            
            # Record allocation request
            self.allocation_history.append({
                "timestamp": time.time(),
                "shape": shape,
                "dtype": str(dtype),
                "name": name
            })
            
            # Try to get from pool first
            tensor = self._get_from_pool(shape, dtype)
            if tensor is not None:
                self.stats["cache_hits"] += 1
                return tensor
            
            # Create new tensor
            try:
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                
                # Track for potential pooling
                tensor_id = id(tensor)
                self.allocated_tensors[tensor_id] = {
                    "tensor": tensor,
                    "shape": shape,
                    "dtype": dtype,
                    "created_at": time.time(),
                    "name": name
                }
                
                return tensor
                
            except torch.cuda.OutOfMemoryError:
                # Try cleanup and retry
                self._emergency_cleanup()
                return torch.empty(shape, dtype=dtype, device=self.device)
    
    def _get_from_pool(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Get tensor from pool with intelligent matching"""
        # Exact match first
        if shape in self.free_blocks and self.free_blocks[shape]:
            tensor = self.free_blocks[shape].pop()
            if tensor.dtype == dtype:
                return tensor.zero_()  # Clear and return
            
            # Type conversion if needed
            if tensor.dtype != dtype:
                return tensor.to(dtype).zero_()
        
        # Try compatible sizes (within 20% size difference)
        target_size = torch.numel(torch.empty(shape))
        
        for pooled_shape, tensors in self.free_blocks.items():
            if not tensors:
                continue
            
            pooled_size = torch.numel(torch.empty(pooled_shape))
            size_ratio = pooled_size / target_size
            
            # Accept if within 20% of target size and not too large
            if 1.0 <= size_ratio <= 1.2:
                tensor = tensors.pop()
                # Reshape if possible
                if tensor.numel() >= target_size:
                    return tensor.view(shape).zero_()
        
        return None
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool with fragmentation detection"""
        with self.pool_lock:
            tensor_id = id(tensor)
            
            if tensor_id in self.allocated_tensors:
                info = self.allocated_tensors.pop(tensor_id)
                shape = info["shape"]
                
                # Add to free blocks if pool not full
                if len(self.free_blocks[shape]) < 10:  # Limit pool size per shape
                    self.free_blocks[shape].append(tensor.detach())
                
                # Calculate memory saved
                memory_saved = tensor.numel() * tensor.element_size()
                self.stats["memory_saved_gb"] += memory_saved / (1024**3)
            
            # Check for fragmentation
            self._check_fragmentation()
    
    def _check_fragmentation(self):
        """Detect and handle memory fragmentation"""
        try:
            if not torch.cuda.is_available():
                return
            
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            
            if reserved > 0:
                fragmentation_ratio = 1.0 - (allocated / reserved)
                self.fragmentation_history.append(fragmentation_ratio)
                
                if fragmentation_ratio > self.fragmentation_threshold:
                    self.stats["fragmentations_detected"] += 1
                    self._defragment_memory()
        
        except Exception:
            pass  # Silently handle any issues
    
    def _defragment_memory(self):
        """Defragment GPU memory"""
        try:
            # Clear unused cached memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Aggressive garbage collection
            gc.collect()
            
            # Consolidate free blocks
            self._consolidate_free_blocks()
            
        except Exception:
            pass
    
    def _consolidate_free_blocks(self):
        """Consolidate free blocks to reduce fragmentation"""
        # Remove empty lists
        empty_shapes = [shape for shape, tensors in self.free_blocks.items() if not tensors]
        for shape in empty_shapes:
            del self.free_blocks[shape]
        
        # Limit number of tensors per shape to prevent memory hoarding
        for shape, tensors in self.free_blocks.items():
            if len(tensors) > 5:
                # Keep only the 5 most recently used
                self.free_blocks[shape] = tensors[-5:]
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup when allocation fails"""
        print("Performing emergency memory cleanup...")
        
        # Clear all free blocks
        self.free_blocks.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_enhanced_stats(self) -> Dict:
        """Get comprehensive memory pool statistics"""
        current_fragmentation = 0.0
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                if reserved > 0:
                    current_fragmentation = 1.0 - (allocated / reserved)
            except:
                pass
        
        return {
            **self.stats,
            "pool_size_gb": self.pool_size_gb,
            "current_fragmentation": current_fragmentation,
            "avg_fragmentation": sum(self.fragmentation_history) / max(len(self.fragmentation_history), 1),
            "free_blocks_count": sum(len(tensors) for tensors in self.free_blocks.values()),
            "allocated_tensors_count": len(self.allocated_tensors),
            "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["total_allocations"], 1),
            "auto_tune_enabled": self.auto_tune
        }
    
    def adaptive_resize(self, target_memory_usage: float = 0.8):
        """Adaptively resize pool based on usage patterns"""
        if not self.auto_tune:
            return
        
        try:
            # Analyze recent allocation patterns
            recent_allocations = list(self.allocation_history)[-100:]  # Last 100 allocations
            
            if len(recent_allocations) < 10:
                return  # Not enough data
            
            # Calculate average allocation rate
            if len(recent_allocations) >= 2:
                time_span = recent_allocations[-1]["timestamp"] - recent_allocations[0]["timestamp"]
                allocation_rate = len(recent_allocations) / max(time_span, 1)
                
                # Adjust pool size based on allocation rate
                if allocation_rate > 50:  # High allocation rate
                    new_size = min(self.pool_size_gb * 1.2, 16.0)  # Increase by 20%
                elif allocation_rate < 5:  # Low allocation rate
                    new_size = max(self.pool_size_gb * 0.9, 1.0)  # Decrease by 10%
                else:
                    return  # No change needed
                
                if abs(new_size - self.pool_size_gb) > 0.5:  # Significant change
                    print(f"Auto-tuning pool size: {self.pool_size_gb:.1f}GB -> {new_size:.1f}GB")
                    self.pool_size_gb = new_size
                    self.stats["auto_tunes"] += 1
        
        except Exception:
            pass  # Silently handle any issues


class EnhancedMemoryManager:
    """
    Enhanced memory manager with predictive allocation and intelligent cleanup.
    """
    
    def __init__(self, device: str = "cuda:0", enable_predictions: bool = True):
        self.device = device
        self.pool = EnhancedGPUMemoryPool(device=device, auto_tune=True)
        self.enable_predictions = enable_predictions
        
        # Prediction and analysis
        self.allocation_patterns = defaultdict(list)
        self.peak_memory = 0
        self.memory_events = []
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._memory_monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
    
    def _memory_monitor(self):
        """Background memory monitoring thread"""
        while self.monitoring_active:
            try:
                # Check fragmentation and auto-tune
                self.pool._check_fragmentation()
                self.pool.adaptive_resize()
                
                # Sleep for monitoring interval
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception:
                continue
    
    def allocate_tensor_smart(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                            name: str = "", predict_usage: bool = True) -> torch.Tensor:
        """Allocate tensor with smart prediction and optimization"""
        
        # Record allocation pattern for prediction
        if predict_usage and self.enable_predictions:
            self.allocation_patterns[name].append({
                "timestamp": time.time(),
                "shape": shape,
                "dtype": str(dtype)
            })
        
        # Get tensor from enhanced pool
        tensor = self.pool.get_tensor(shape, dtype, name)
        
        # Track memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(self.device)
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
        
        # Record memory event
        self.memory_events.append({
            "action": "allocate",
            "name": name,
            "shape": shape,
            "dtype": str(dtype),
            "size_mb": tensor.numel() * tensor.element_size() / (1024**2),
            "timestamp": time.time()
        })
        
        return tensor
    
    def predict_next_allocations(self, context: str = "") -> List[Dict]:
        """Predict likely next allocations based on patterns"""
        if not self.enable_predictions:
            return []
        
        predictions = []
        
        # Analyze patterns for each allocation name
        for name, history in self.allocation_patterns.items():
            if len(history) < 3:  # Need at least 3 data points
                continue
            
            recent = history[-10:]  # Last 10 allocations
            
            # Find common patterns
            common_shapes = defaultdict(int)
            for alloc in recent:
                common_shapes[alloc["shape"]] += 1
            
            # Predict most likely next allocation
            if common_shapes:
                most_common_shape = max(common_shapes.keys(), key=lambda x: common_shapes[x])
                predictions.append({
                    "name": name,
                    "shape": most_common_shape,
                    "confidence": common_shapes[most_common_shape] / len(recent),
                    "context": context
                })
        
        return sorted(predictions, key=lambda x: x["confidence"], reverse=True)
    
    def get_comprehensive_report(self) -> Dict:
        """Get comprehensive memory management report"""
        pool_stats = self.pool.get_enhanced_stats()
        
        # System memory info
        system_memory = {}
        try:
            memory_info = psutil.virtual_memory()
            system_memory = {
                "total_gb": memory_info.total / (1024**3),
                "available_gb": memory_info.available / (1024**3),
                "used_percent": memory_info.percent
            }
        except:
            pass
        
        # GPU memory info
        gpu_memory = {}
        if torch.cuda.is_available():
            try:
                gpu_memory = {
                    "allocated_gb": torch.cuda.memory_allocated(self.device) / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved(self.device) / (1024**3),
                    "peak_gb": self.peak_memory / (1024**3)
                }
            except:
                pass
        
        return {
            "pool_statistics": pool_stats,
            "system_memory": system_memory,
            "gpu_memory": gpu_memory,
            "allocation_patterns": len(self.allocation_patterns),
            "total_events": len(self.memory_events),
            "monitoring_active": self.monitoring_active,
            "predictions_enabled": self.enable_predictions
        }