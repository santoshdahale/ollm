"""
Layer prefetching to overlap computation with memory I/O.
Loads next layers asynchronously while current layers are computing.
"""
import torch
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Callable, Any, Tuple
import time
from dataclasses import dataclass
import queue
import gc

@dataclass
class PrefetchStats:
    """Statistics for prefetching performance"""
    layers_prefetched: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    total_prefetch_time_ms: float = 0.0
    total_load_time_ms: float = 0.0
    cache_hit_rate: float = 0.0

class LayerPrefetcher:
    """
    Asynchronous layer prefetching system that loads model layers ahead of time.
    Overlaps I/O operations with computation to reduce waiting time.
    """
    
    def __init__(self, model, prefetch_distance: int = 2, max_cache_size: int = 4,
                 device: str = "cuda:0", num_workers: int = 2):
        self.model = model
        self.prefetch_distance = prefetch_distance
        self.max_cache_size = max_cache_size
        self.device = torch.device(device)
        self.num_workers = num_workers
        
        # Cache for prefetched layers
        self.layer_cache: Dict[int, torch.nn.Module] = {}
        self.cache_access_order: List[int] = []
        
        # Async execution
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_queue = asyncio.Queue()
        self.active_prefetches: Dict[int, asyncio.Task] = {}
        
        # Statistics
        self.stats = PrefetchStats()
        
        # Thread-safe locks
        self.cache_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Layer loading functions
        self.layer_loaders: Dict[int, Callable] = {}
        self._setup_layer_loaders()
    
    def _setup_layer_loaders(self):
        """Setup layer loading functions for different model types"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Standard transformer model
            layers = self.model.model.layers
            for i, layer in enumerate(layers):
                self.layer_loaders[i] = lambda idx=i: self._load_transformer_layer(idx)
        
        elif hasattr(self.model, 'layers'):
            # Direct layer access
            layers = self.model.layers
            for i, layer in enumerate(layers):
                self.layer_loaders[i] = lambda idx=i: self._load_direct_layer(idx)
    
    def _load_transformer_layer(self, layer_idx: int) -> torch.nn.Module:
        """Load a transformer layer from storage/CPU to GPU"""
        start_time = time.perf_counter()
        
        try:
            layer = self.model.model.layers[layer_idx]
            
            # If layer is on CPU, move to GPU
            if next(layer.parameters()).device != self.device:
                layer = layer.to(self.device)
            
            load_time = (time.perf_counter() - start_time) * 1000
            
            with self.stats_lock:
                self.stats.total_load_time_ms += load_time
            
            return layer
            
        except Exception as e:
            print(f"Error loading layer {layer_idx}: {e}")
            return None
    
    def _load_direct_layer(self, layer_idx: int) -> torch.nn.Module:
        """Load layer with direct access"""
        start_time = time.perf_counter()
        
        try:
            layer = self.model.layers[layer_idx]
            
            if next(layer.parameters()).device != self.device:
                layer = layer.to(self.device)
            
            load_time = (time.perf_counter() - start_time) * 1000
            
            with self.stats_lock:
                self.stats.total_load_time_ms += load_time
            
            return layer
            
        except Exception as e:
            print(f"Error loading layer {layer_idx}: {e}")
            return None
    
    async def _prefetch_layer_async(self, layer_idx: int):
        """Asynchronously prefetch a layer"""
        if layer_idx in self.layer_cache or layer_idx in self.active_prefetches:
            return  # Already cached or being prefetched
        
        start_time = time.perf_counter()
        
        try:
            # Run layer loading in thread pool
            loop = asyncio.get_event_loop()
            if layer_idx in self.layer_loaders:
                layer = await loop.run_in_executor(
                    self.executor, 
                    self.layer_loaders[layer_idx]
                )
                
                if layer is not None:
                    with self.cache_lock:
                        # Add to cache
                        self.layer_cache[layer_idx] = layer
                        self.cache_access_order.append(layer_idx)
                        
                        # Evict oldest layers if cache is full
                        while len(self.layer_cache) > self.max_cache_size:
                            oldest_idx = self.cache_access_order.pop(0)
                            if oldest_idx in self.layer_cache:
                                # Move evicted layer back to CPU to free GPU memory
                                evicted_layer = self.layer_cache.pop(oldest_idx)
                                evicted_layer.cpu()
                                del evicted_layer
                                gc.collect()
                    
                    prefetch_time = (time.perf_counter() - start_time) * 1000
                    
                    with self.stats_lock:
                        self.stats.layers_prefetched += 1
                        self.stats.total_prefetch_time_ms += prefetch_time
                        
        except Exception as e:
            print(f"Error prefetching layer {layer_idx}: {e}")
        
        finally:
            # Remove from active prefetches
            if layer_idx in self.active_prefetches:
                del self.active_prefetches[layer_idx]
    
    def prefetch_layers(self, current_layer_idx: int):
        """
        Prefetch upcoming layers based on current layer index.
        
        Args:
            current_layer_idx: Index of currently executing layer
        """
        # Determine which layers to prefetch
        layers_to_prefetch = []
        for i in range(1, self.prefetch_distance + 1):
            next_layer_idx = current_layer_idx + i
            if (next_layer_idx in self.layer_loaders and 
                next_layer_idx not in self.layer_cache and
                next_layer_idx not in self.active_prefetches):
                layers_to_prefetch.append(next_layer_idx)
        
        # Start prefetching tasks
        for layer_idx in layers_to_prefetch:
            task = asyncio.create_task(self._prefetch_layer_async(layer_idx))
            self.active_prefetches[layer_idx] = task
    
    def get_layer(self, layer_idx: int) -> torch.nn.Module:
        """
        Get a layer, using cache if available or loading if not.
        
        Args:
            layer_idx: Index of layer to retrieve
            
        Returns:
            The requested layer module
        """
        with self.cache_lock:
            if layer_idx in self.layer_cache:
                # Cache hit
                with self.stats_lock:
                    self.stats.prefetch_hits += 1
                
                # Update access order
                if layer_idx in self.cache_access_order:
                    self.cache_access_order.remove(layer_idx)
                self.cache_access_order.append(layer_idx)
                
                return self.layer_cache[layer_idx]
        
        # Cache miss - need to load synchronously
        with self.stats_lock:
            self.stats.prefetch_misses += 1
        
        if layer_idx in self.layer_loaders:
            layer = self.layer_loaders[layer_idx]()
            
            # Add to cache
            with self.cache_lock:
                self.layer_cache[layer_idx] = layer
                self.cache_access_order.append(layer_idx)
                
                # Evict if necessary
                while len(self.layer_cache) > self.max_cache_size:
                    oldest_idx = self.cache_access_order.pop(0)
                    if oldest_idx in self.layer_cache:
                        evicted_layer = self.layer_cache.pop(oldest_idx)
                        evicted_layer.cpu()
                        del evicted_layer
                        gc.collect()
            
            return layer
        
        # Fallback to direct model access
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'layers'):
            return self.model.layers[layer_idx]
        else:
            raise ValueError(f"Cannot access layer {layer_idx}")
    
    def update_stats(self):
        """Update cache hit rate statistics"""
        with self.stats_lock:
            total_requests = self.stats.prefetch_hits + self.stats.prefetch_misses
            if total_requests > 0:
                self.stats.cache_hit_rate = self.stats.prefetch_hits / total_requests
    
    def get_stats(self) -> PrefetchStats:
        """Get current prefetching statistics"""
        self.update_stats()
        return self.stats
    
    def clear_cache(self):
        """Clear the layer cache"""
        with self.cache_lock:
            for layer in self.layer_cache.values():
                layer.cpu()
            self.layer_cache.clear()
            self.cache_access_order.clear()
            gc.collect()
    
    def __del__(self):
        """Cleanup when prefetcher is destroyed"""
        self.clear_cache()
        self.executor.shutdown(wait=True)


class AdaptivePrefetcher(LayerPrefetcher):
    """
    Adaptive prefetcher that adjusts prefetch distance based on performance.
    """
    
    def __init__(self, model, initial_distance: int = 2, **kwargs):
        super().__init__(model, prefetch_distance=initial_distance, **kwargs)
        
        self.min_distance = 1
        self.max_distance = 6
        self.adaptation_window = 20  # Number of requests to consider
        self.recent_hit_rates = []
        
    def adapt_prefetch_distance(self):
        """Adapt prefetch distance based on recent cache hit rates"""
        if len(self.recent_hit_rates) < self.adaptation_window:
            return
        
        avg_hit_rate = sum(self.recent_hit_rates) / len(self.recent_hit_rates)
        
        if avg_hit_rate < 0.7:  # Low hit rate
            # Increase prefetch distance if possible
            self.prefetch_distance = min(self.max_distance, self.prefetch_distance + 1)
        elif avg_hit_rate > 0.95:  # Very high hit rate
            # Decrease prefetch distance to save memory
            self.prefetch_distance = max(self.min_distance, self.prefetch_distance - 1)
        
        # Reset recent hit rates
        self.recent_hit_rates = []
    
    def get_layer(self, layer_idx: int) -> torch.nn.Module:
        """Get layer with adaptive behavior"""
        layer = super().get_layer(layer_idx)
        
        # Track hit rate for adaptation
        current_hit_rate = self.stats.cache_hit_rate
        self.recent_hit_rates.append(current_hit_rate)
        
        # Adapt if we have enough samples
        if len(self.recent_hit_rates) >= self.adaptation_window:
            self.adapt_prefetch_distance()
        
        return layer


class MemoryAwarePrefetcher(LayerPrefetcher):
    """
    Memory-aware prefetcher that adjusts cache size based on available GPU memory.
    """
    
    def __init__(self, model, memory_threshold: float = 0.8, **kwargs):
        super().__init__(model, **kwargs)
        self.memory_threshold = memory_threshold  # Fraction of GPU memory to use
        self.base_cache_size = self.max_cache_size
        
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage as fraction of total"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            return allocated / total
        return 0.0
    
    def _adjust_cache_size(self):
        """Adjust cache size based on memory usage"""
        memory_usage = self._get_gpu_memory_usage()
        
        if memory_usage > self.memory_threshold:
            # High memory usage - reduce cache size
            self.max_cache_size = max(1, self.max_cache_size - 1)
        elif memory_usage < self.memory_threshold - 0.1:
            # Low memory usage - can increase cache size
            self.max_cache_size = min(self.base_cache_size, self.max_cache_size + 1)
        
        # Evict layers if current cache exceeds new limit
        with self.cache_lock:
            while len(self.layer_cache) > self.max_cache_size:
                oldest_idx = self.cache_access_order.pop(0)
                if oldest_idx in self.layer_cache:
                    evicted_layer = self.layer_cache.pop(oldest_idx)
                    evicted_layer.cpu()
                    del evicted_layer
                    gc.collect()
                    torch.cuda.empty_cache()
    
    def prefetch_layers(self, current_layer_idx: int):
        """Prefetch with memory awareness"""
        self._adjust_cache_size()
        super().prefetch_layers(current_layer_idx)


class PipelinedModelExecutor:
    """
    Executor that uses prefetching for pipelined model execution.
    Coordinates layer execution with prefetching for optimal performance.
    """
    
    def __init__(self, model, prefetcher: LayerPrefetcher):
        self.model = model
        self.prefetcher = prefetcher
        
    def forward_with_prefetching(self, x: torch.Tensor, layer_indices: List[int]) -> torch.Tensor:
        """
        Execute forward pass with prefetching.
        
        Args:
            x: Input tensor
            layer_indices: List of layer indices to execute
            
        Returns:
            Output tensor after all layers
        """
        current_output = x
        
        for i, layer_idx in enumerate(layer_indices):
            # Prefetch upcoming layers
            self.prefetcher.prefetch_layers(layer_idx)
            
            # Get current layer (may hit cache)
            layer = self.prefetcher.get_layer(layer_idx)
            
            # Execute layer
            if hasattr(layer, 'forward'):
                current_output = layer(current_output)
            else:
                # Handle special layer types
                current_output = self._execute_layer(layer, current_output)
        
        return current_output
    
    def _execute_layer(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Execute a layer with appropriate handling for different types"""
        try:
            return layer(x)
        except Exception as e:
            print(f"Error executing layer: {e}")
            return x  # Return input unchanged as fallback
    
    def execute_with_stats(self, x: torch.Tensor, layer_indices: List[int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute with detailed performance statistics.
        
        Args:
            x: Input tensor
            layer_indices: List of layer indices to execute
            
        Returns:
            Tuple of (output_tensor, performance_stats)
        """
        start_time = time.perf_counter()
        
        output = self.forward_with_prefetching(x, layer_indices)
        
        total_time = time.perf_counter() - start_time
        prefetch_stats = self.prefetcher.get_stats()
        
        stats = {
            "total_execution_time_ms": total_time * 1000,
            "layers_executed": len(layer_indices),
            "prefetch_stats": prefetch_stats,
            "throughput_layers_per_second": len(layer_indices) / total_time if total_time > 0 else 0
        }
        
        return output, stats


# Alias for backward compatibility
PrefetchingOptimizer = MemoryAwarePrefetcher