"""
Adaptive optimizer that monitors performance and automatically adjusts optimization strategies.
Dynamically reconfigures the system based on current bottlenecks and resource usage.
"""
import torch
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import os

@dataclass
class PerformanceMetrics:
    """Performance metrics for adaptive optimization"""
    timestamp: float
    tokens_per_second: float
    memory_usage_gb: float
    gpu_utilization: float
    cache_hit_rate: float
    attention_time_ms: float
    kv_cache_size_mb: float
    sequence_length: int
    batch_size: int

@dataclass
class OptimizationStrategy:
    """Configuration for optimization strategy"""
    name: str
    attention_method: str = "full"
    kv_compression: str = "none"
    prefetch_distance: int = 2
    memory_pool_size_gb: float = 6.0
    speculative_candidates: int = 0
    context_compression_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "attention_method": self.attention_method,
            "kv_compression": self.kv_compression,
            "prefetch_distance": self.prefetch_distance,
            "memory_pool_size_gb": self.memory_pool_size_gb,
            "speculative_candidates": self.speculative_candidates,
            "context_compression_ratio": self.context_compression_ratio
        }

class AdaptiveOptimizer:
    """
    Adaptive optimizer that monitors system performance and automatically
    adjusts optimization strategies to maximize throughput and efficiency.
    """
    
    def __init__(self, model, device: str = "cuda:0", 
                 monitoring_window: int = 50,
                 adaptation_interval: int = 100):
        self.model = model
        self.device = device
        self.monitoring_window = monitoring_window
        self.adaptation_interval = adaptation_interval
        # Performance monitoring
        self.metrics_history = deque(maxlen=monitoring_window)
        self.iteration_count = 0
        # Available strategies
        self.available_strategies = self._initialize_strategies()
        self.current_strategy = self.available_strategies["balanced"]
        # Adaptation state
        self.last_adaptation = 0
        self.strategy_performance = {}
        # Thread-safe monitoring
        self.metrics_lock = threading.Lock()
        self.adaptation_lock = threading.Lock()
        # Callbacks for strategy changes
        self.strategy_callbacks: List[Callable] = []
        
    def _initialize_strategies(self) -> Dict[str, OptimizationStrategy]:
        """Initialize available optimization strategies"""
        return {
            "memory_optimized": OptimizationStrategy(
                name="memory_optimized",
                attention_method="sliding_window",
                kv_compression="quantization",
                prefetch_distance=1,
                memory_pool_size_gb=4.0,
                speculative_candidates=0,
                context_compression_ratio=0.6
            ),
            "speed_optimized": OptimizationStrategy(
                name="speed_optimized", 
                attention_method="sparse",
                kv_compression="none",
                prefetch_distance=4,
                memory_pool_size_gb=8.0,
                speculative_candidates=4,
                context_compression_ratio=0.8
            ),
            "balanced": OptimizationStrategy(
                name="balanced",
                attention_method="adaptive",
                kv_compression="pruning",
                prefetch_distance=2,
                memory_pool_size_gb=6.0,
                speculative_candidates=2,
                context_compression_ratio=0.7
            ),
            "long_context": OptimizationStrategy(
                name="long_context",
                attention_method="hierarchical",
                kv_compression="clustering",
                prefetch_distance=3,
                memory_pool_size_gb=5.0,
                speculative_candidates=1,
                context_compression_ratio=0.4
            )
        }
    
    def collect_metrics(self, tokens_generated: int, generation_time: float,
                       sequence_length: int, batch_size: int = 1,
                       additional_metrics: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """
        Collect performance metrics for the current iteration.
        Args:
            tokens_generated: Number of tokens generated
            generation_time: Time taken for generation (seconds)  
            sequence_length: Current sequence length
            batch_size: Batch size
            additional_metrics: Additional metrics to include
            
        Returns:
            PerformanceMetrics object
        """
        # Calculate tokens per second
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        # Get memory usage
        memory_usage_gb = 0
        if torch.cuda.is_available():
            memory_usage_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
        # Get GPU utilization (approximation)
        gpu_utilization = min(memory_usage_gb / 8.0, 1.0)  # Assume 8GB GPU
        # Default values for metrics that might not be available
        cache_hit_rate = additional_metrics.get("cache_hit_rate", 0.0) if additional_metrics else 0.0
        attention_time_ms = additional_metrics.get("attention_time_ms", 0.0) if additional_metrics else 0.0
        kv_cache_size_mb = additional_metrics.get("kv_cache_size_mb", 0.0) if additional_metrics else 0.0
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            tokens_per_second=tokens_per_second,
            memory_usage_gb=memory_usage_gb,
            gpu_utilization=gpu_utilization,
            cache_hit_rate=cache_hit_rate,
            attention_time_ms=attention_time_ms,
            kv_cache_size_mb=kv_cache_size_mb,
            sequence_length=sequence_length,
            batch_size=batch_size
        )
        with self.metrics_lock:
            self.metrics_history.append(metrics)
            self.iteration_count += 1
        return metrics
    
    def analyze_bottleneck(self) -> str:
        """Analyze current bottleneck based on recent metrics"""
        if len(self.metrics_history) < 5:
            return "insufficient_data"
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 iterations
        
        # Calculate averages
        avg_memory = sum(m.memory_usage_gb for m in recent_metrics) / len(recent_metrics)
        avg_tokens_per_sec = sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics)
        avg_attention_time = sum(m.attention_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        # Determine bottleneck
        if avg_memory > 7.0:  # High memory usage
            return "memory"
        elif avg_tokens_per_sec < 10:  # Low throughput
            return "speed"
        elif avg_attention_time > 100:  # High attention computation time
            return "attention"
        elif avg_cache_hit_rate < 0.5:  # Low cache efficiency
            return "cache"
        else:
            return "balanced"
    
    def choose_optimal_strategy(self, bottleneck: str, sequence_length: int) -> OptimizationStrategy:
        """Choose optimal strategy based on bottleneck analysis"""
        if bottleneck == "memory":
            return self.available_strategies["memory_optimized"]
        elif bottleneck == "speed":
            return self.available_strategies["speed_optimized"]
        elif bottleneck == "attention" and sequence_length > 4096:
            return self.available_strategies["long_context"]
        else:
            return self.available_strategies["balanced"]
    
    def should_adapt(self) -> bool:
        """Determine if adaptation should be triggered"""
        return (self.iteration_count - self.last_adaptation) >= self.adaptation_interval
    
    def adapt_strategy(self) -> Optional[OptimizationStrategy]:
        """
        Adapt optimization strategy based on current performance.
        Returns:
            New strategy if adaptation occurred, None otherwise
        """
        with self.adaptation_lock:
            if not self.should_adapt():
                return None
            
            # Analyze current bottleneck
            bottleneck = self.analyze_bottleneck()
            
            if bottleneck == "insufficient_data":
                return None
            
            # Get current sequence length (use most recent)
            current_seq_len = self.metrics_history[-1].sequence_length if self.metrics_history else 1024
            
            # Choose new strategy
            new_strategy = self.choose_optimal_strategy(bottleneck, current_seq_len)
            
            # Check if strategy change is needed
            if new_strategy.name != self.current_strategy.name:
                old_strategy = self.current_strategy
                self.current_strategy = new_strategy
                self.last_adaptation = self.iteration_count
                
                # Record strategy performance
                if old_strategy.name in self.strategy_performance:
                    recent_perf = [m.tokens_per_second for m in list(self.metrics_history)[-10:]]
                    avg_perf = sum(recent_perf) / len(recent_perf) if recent_perf else 0
                    self.strategy_performance[old_strategy.name].append(avg_perf)
                else:
                    self.strategy_performance[old_strategy.name] = []
                
                # Notify callbacks
                self._notify_strategy_change(old_strategy, new_strategy, bottleneck)
                
                return new_strategy
            
            return None
    
    def _notify_strategy_change(self, old_strategy: OptimizationStrategy, 
                               new_strategy: OptimizationStrategy, bottleneck: str):
        """Notify registered callbacks about strategy change"""
        for callback in self.strategy_callbacks:
            try:
                callback(old_strategy, new_strategy, bottleneck)
            except Exception as e:
                print(f"Error in strategy change callback: {e}")
    
    def register_strategy_callback(self, callback: Callable):
        """Register callback for strategy changes"""
        self.strategy_callbacks.append(callback)
    
    def get_current_strategy(self) -> OptimizationStrategy:
        """Get current optimization strategy"""
        return self.current_strategy
    
    def force_strategy(self, strategy_name: str) -> bool:
        """
        Force a specific strategy.
        Args:
            strategy_name: Name of strategy to use
            
        Returns:
            True if strategy was set, False if not found
        """
        if strategy_name in self.available_strategies:
            old_strategy = self.current_strategy
            self.current_strategy = self.available_strategies[strategy_name]
            self._notify_strategy_change(old_strategy, self.current_strategy, "manual")
            return True
        return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 iterations
        report = {
            "current_strategy": self.current_strategy.to_dict(),
            "iteration_count": self.iteration_count,
            "metrics_summary": {
                "avg_tokens_per_second": sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics),
                "avg_memory_usage_gb": sum(m.memory_usage_gb for m in recent_metrics) / len(recent_metrics),
                "avg_gpu_utilization": sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics),
                "avg_cache_hit_rate": sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
                "avg_sequence_length": sum(m.sequence_length for m in recent_metrics) / len(recent_metrics)
            },
            "bottleneck_analysis": self.analyze_bottleneck(),
            "strategy_performance": self.strategy_performance,
            "available_strategies": [name for name in self.available_strategies.keys()]
        }
        return report
    
    def save_performance_log(self, filepath: str):
        """Save performance metrics to file"""
        data = {
            "metrics": [
                {
                    "timestamp": m.timestamp,
                    "tokens_per_second": m.tokens_per_second,
                    "memory_usage_gb": m.memory_usage_gb,
                    "gpu_utilization": m.gpu_utilization,
                    "cache_hit_rate": m.cache_hit_rate,
                    "attention_time_ms": m.attention_time_ms,
                    "kv_cache_size_mb": m.kv_cache_size_mb,
                    "sequence_length": m.sequence_length,
                    "batch_size": m.batch_size
                }
                for m in self.metrics_history
            ],
            "strategy_performance": self.strategy_performance,
            "current_strategy": self.current_strategy.to_dict()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_performance_log(self, filepath: str):
        """Load performance metrics from file"""
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Reconstruct metrics history
        self.metrics_history.clear()
        for metric_data in data.get("metrics", []):
            metrics = PerformanceMetrics(**metric_data)
            self.metrics_history.append(metrics)
        # Load strategy performance
        self.strategy_performance = data.get("strategy_performance", {})
        # Set current strategy if available
        current_strategy_data = data.get("current_strategy")
        if current_strategy_data:
            strategy_name = current_strategy_data.get("name")
            if strategy_name in self.available_strategies:
                self.current_strategy = self.available_strategies[strategy_name]


class SystemMonitor:
    """System resource monitor for adaptive optimization"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.system_metrics = deque(maxlen=60)  # Keep 1 minute of data
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if self.is_monitoring:
            return
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                gpu_memory_allocated = 0
                gpu_memory_total = 0
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "gpu_memory_allocated_gb": gpu_memory_allocated,
                    "gpu_memory_total_gb": gpu_memory_total,
                    "gpu_memory_percent": (gpu_memory_allocated / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
                }
                
                self.system_metrics.append(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in system monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get most recent system metrics"""
        return self.system_metrics[-1] if self.system_metrics else None
    
    def get_average_metrics(self, window_seconds: int = 30) -> Dict[str, float]:
        """Get average metrics over time window"""
        if not self.system_metrics:
            return {}
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        recent_metrics = [m for m in self.system_metrics if m["timestamp"] >= cutoff_time]
        if not recent_metrics:
            return {}
        return {
            "avg_cpu_percent": sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics),
            "avg_memory_percent": sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics),
            "avg_gpu_memory_percent": sum(m["gpu_memory_percent"] for m in recent_metrics) / len(recent_metrics)
        }