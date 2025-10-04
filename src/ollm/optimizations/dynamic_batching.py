"""
Dynamic batching system to efficiently process multiple requests with different sequence lengths.
Minimizes padding overhead and maximizes GPU utilization.
"""
import torch
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import threading
from queue import Queue, Empty
import heapq

@dataclass
class BatchRequest:
    """Represents a single request in the batch"""
    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    stop_tokens: List[int] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    generated_tokens: List[int] = field(default_factory=list)
    is_complete: bool = False
    callback: Optional[Callable] = None

@dataclass
class BatchStats:
    """Statistics for dynamic batching"""
    total_requests: int = 0
    completed_requests: int = 0
    average_batch_size: float = 0.0
    average_sequence_length: float = 0.0
    throughput_requests_per_second: float = 0.0
    gpu_utilization: float = 0.0
    padding_ratio: float = 0.0

class DynamicBatcher:
    """
    Dynamic batching system that groups requests by similar characteristics
    to minimize padding and maximize throughput.
    """
    
    def __init__(self, model, tokenizer,
                 max_batch_size: int = 8,
                 max_sequence_length: int = 2048,
                 batch_timeout_ms: float = 50.0,
                 length_bucket_size: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.batch_timeout_ms = batch_timeout_ms
        self.length_bucket_size = length_bucket_size
        
        # Request queues organized by sequence length buckets
        self.length_buckets: Dict[int, deque] = {}
        self.pending_requests: Dict[str, BatchRequest] = {}
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.request_queue = Queue()
        self.result_callbacks: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = BatchStats()
        self.batch_history = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
        # Synchronization
        self.batch_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
    def _get_length_bucket(self, sequence_length: int) -> int:
        """Get the bucket key for a given sequence length"""
        return (sequence_length // self.length_bucket_size) * self.length_bucket_size
    
    def add_request(self, request_id: str, input_text: str, 
                   max_new_tokens: int = 50, **generation_kwargs) -> str:
        """
        Add a new generation request to the batch queue.
        
        Args:
            request_id: Unique identifier for the request
            input_text: Input text to process
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Request ID for tracking
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
        
        # Create request object
        request = BatchRequest(
            request_id=request_id,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=generation_kwargs.get("temperature", 1.0),
            top_p=generation_kwargs.get("top_p", 1.0),
            stop_tokens=generation_kwargs.get("stop_tokens", []),
            callback=generation_kwargs.get("callback")
        )
        
        # Add to appropriate bucket
        seq_len = input_ids.shape[1]
        bucket_key = self._get_length_bucket(seq_len)
        
        with self.batch_lock:
            if bucket_key not in self.length_buckets:
                self.length_buckets[bucket_key] = deque()
            
            self.length_buckets[bucket_key].append(request)
            self.pending_requests[request_id] = request
            
            with self.stats_lock:
                self.stats.total_requests += 1
        
        return request_id
    
    def _select_batch(self) -> List[BatchRequest]:
        """
        Select optimal batch of requests to process together.
        Prioritizes similar sequence lengths to minimize padding.
        """
        with self.batch_lock:
            if not self.length_buckets:
                return []
            
            # Find bucket with most requests
            best_bucket = max(self.length_buckets.keys(), 
                            key=lambda k: len(self.length_buckets[k]))
            
            bucket_requests = self.length_buckets[best_bucket]
            
            # Select requests from this bucket
            batch = []
            while bucket_requests and len(batch) < self.max_batch_size:
                request = bucket_requests.popleft()
                
                # Check if request is still valid
                if request.request_id in self.pending_requests:
                    batch.append(request)
            
            # Clean up empty bucket
            if not bucket_requests:
                del self.length_buckets[best_bucket]
            
            return batch
    
    def _create_padded_batch(self, requests: List[BatchRequest]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Create padded batch tensors from requests.
        
        Returns:
            Tuple of (input_ids, attention_mask, original_lengths)
        """
        if not requests:
            return None, None, []
        
        # Find maximum sequence length in batch
        max_len = max(req.input_ids.shape[1] for req in requests)
        max_len = min(max_len, self.max_sequence_length)
        
        batch_size = len(requests)
        original_lengths = []
        
        # Create padded tensors
        input_ids = torch.full((batch_size, max_len), 
                              self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                              dtype=torch.long, device=self.model.device)
        
        attention_mask = torch.zeros((batch_size, max_len), 
                                   dtype=torch.long, device=self.model.device)
        
        # Fill tensors
        for i, request in enumerate(requests):
            seq_len = min(request.input_ids.shape[1], max_len)
            original_lengths.append(seq_len)
            
            input_ids[i, :seq_len] = request.input_ids[0, :seq_len]
            attention_mask[i, :seq_len] = 1
        
        return input_ids, attention_mask, original_lengths
    
    def _process_batch(self, batch: List[BatchRequest]):
        """
        Process a batch of requests through the model.
        """
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Create padded batch
            input_ids, attention_mask, original_lengths = self._create_padded_batch(batch)
            
            if input_ids is None:
                return
            
            batch_size, seq_len = input_ids.shape
            
            # Calculate padding ratio for stats
            total_tokens = batch_size * seq_len
            actual_tokens = sum(original_lengths)
            padding_ratio = 1.0 - (actual_tokens / total_tokens) if total_tokens > 0 else 0.0
            
            # Process batch through model
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=True
                )
                
                # Generate tokens for each request in batch
                for step in range(max(req.max_new_tokens for req in batch)):
                    # Get logits for last position
                    logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
                    
                    # Apply temperature and top-p for each request
                    next_tokens = []
                    for i, request in enumerate(batch):
                        if request.is_complete or len(request.generated_tokens) >= request.max_new_tokens:
                            next_tokens.append(self.tokenizer.eos_token_id)
                            continue
                        
                        # Apply temperature
                        scaled_logits = logits[i] / request.temperature
                        
                        # Apply top-p filtering
                        if request.top_p < 1.0:
                            scaled_logits = self._apply_top_p(scaled_logits, request.top_p)
                        
                        # Sample next token
                        probs = torch.softmax(scaled_logits, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                        
                        # Check for stop conditions
                        if (next_token == self.tokenizer.eos_token_id or 
                            next_token in request.stop_tokens):
                            request.is_complete = True
                            next_token = self.tokenizer.eos_token_id
                        
                        request.generated_tokens.append(next_token)
                        next_tokens.append(next_token)
                    
                    # Update input for next step
                    next_token_tensor = torch.tensor([next_tokens], device=self.model.device).T
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                    
                    # Update attention mask
                    attention_mask = torch.cat([
                        attention_mask, 
                        torch.ones((batch_size, 1), device=self.model.device, dtype=torch.long)
                    ], dim=1)
                    
                    # Check if all requests are complete
                    if all(req.is_complete for req in batch):
                        break
                    
                    # Get next outputs
                    if not all(req.is_complete for req in batch):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=False  # Don't accumulate KV cache for batched generation
                        )
            
            # Process completed requests
            processing_time = time.time() - start_time
            
            for request in batch:
                # Generate output text
                if request.generated_tokens:
                    output_text = self.tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
                else:
                    output_text = ""
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(request.request_id, output_text)
                    except Exception as e:
                        print(f"Error in callback for request {request.request_id}: {e}")
                
                # Remove from pending
                with self.batch_lock:
                    if request.request_id in self.pending_requests:
                        del self.pending_requests[request.request_id]
                
                with self.stats_lock:
                    self.stats.completed_requests += 1
            
            # Update statistics
            with self.stats_lock:
                self.batch_history.append({
                    "batch_size": len(batch),
                    "sequence_length": seq_len,
                    "processing_time": processing_time,
                    "padding_ratio": padding_ratio
                })
                
                self.processing_times.append(processing_time)
                
                # Update running averages
                if self.batch_history:
                    self.stats.average_batch_size = sum(b["batch_size"] for b in self.batch_history) / len(self.batch_history)
                    self.stats.average_sequence_length = sum(b["sequence_length"] for b in self.batch_history) / len(self.batch_history)
                    self.stats.padding_ratio = sum(b["padding_ratio"] for b in self.batch_history) / len(self.batch_history)
                
                if self.processing_times:
                    total_time = sum(self.processing_times)
                    self.stats.throughput_requests_per_second = len(self.processing_times) / total_time if total_time > 0 else 0
        
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Mark all requests in batch as complete with empty output
            for request in batch:
                if request.callback:
                    try:
                        request.callback(request.request_id, "")
                    except:
                        pass
                
                with self.batch_lock:
                    if request.request_id in self.pending_requests:
                        del self.pending_requests[request.request_id]
    
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        # Set logits to -inf for removed tokens
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def start_processing(self):
        """Start the batch processing thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the batch processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
    
    def _processing_loop(self):
        """Main processing loop for batching"""
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                time_since_last_batch = (current_time - last_batch_time) * 1000  # Convert to ms
                
                # Check if we should process a batch
                should_process = False
                
                with self.batch_lock:
                    total_pending = sum(len(bucket) for bucket in self.length_buckets.values())
                
                if total_pending >= self.max_batch_size:
                    # Have enough requests for full batch
                    should_process = True
                elif total_pending > 0 and time_since_last_batch >= self.batch_timeout_ms:
                    # Timeout reached with pending requests
                    should_process = True
                
                if should_process:
                    batch = self._select_batch()
                    if batch:
                        self._process_batch(batch)
                        last_batch_time = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def get_stats(self) -> BatchStats:
        """Get current batching statistics"""
        with self.stats_lock:
            # Update GPU utilization (approximation)
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.model.device)
                total = torch.cuda.get_device_properties(self.model.device).total_memory
                self.stats.gpu_utilization = allocated / total
            
            return self.stats
    
    def get_pending_count(self) -> int:
        """Get number of pending requests"""
        with self.batch_lock:
            return len(self.pending_requests)
    
    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending request.
        
        Args:
            request_id: ID of request to cancel
            
        Returns:
            True if request was cancelled, False if not found
        """
        with self.batch_lock:
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                
                # Remove from bucket
                seq_len = request.input_ids.shape[1]
                bucket_key = self._get_length_bucket(seq_len)
                
                if bucket_key in self.length_buckets:
                    try:
                        self.length_buckets[bucket_key].remove(request)
                    except ValueError:
                        pass  # Request might have already been removed
                
                # Remove from pending
                del self.pending_requests[request_id]
                return True
        
        return False


class AdaptiveBatcher(DynamicBatcher):
    """
    Adaptive batcher that automatically adjusts batch size and timeout
    based on current load and performance characteristics.
    """
    
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        
        # Adaptive parameters
        self.min_batch_size = 1
        self.base_batch_size = self.max_batch_size
        self.min_timeout_ms = 10.0
        self.max_timeout_ms = 200.0
        self.base_timeout_ms = self.batch_timeout_ms
        
        # Adaptation history
        self.adaptation_history = deque(maxlen=50)
    
    def _adapt_parameters(self):
        """Adapt batch size and timeout based on recent performance"""
        if len(self.batch_history) < 10:
            return
        
        recent_batches = list(self.batch_history)[-10:]
        
        # Calculate metrics
        avg_batch_size = sum(b["batch_size"] for b in recent_batches) / len(recent_batches)
        avg_processing_time = sum(b["processing_time"] for b in recent_batches) / len(recent_batches)
        avg_padding_ratio = sum(b["padding_ratio"] for b in recent_batches) / len(recent_batches)
        
        # Adapt batch size
        if avg_padding_ratio > 0.3:  # High padding - reduce batch size
            self.max_batch_size = max(self.min_batch_size, int(self.max_batch_size * 0.9))
        elif avg_padding_ratio < 0.1 and avg_processing_time < 0.1:  # Low padding, fast processing - increase batch size
            self.max_batch_size = min(self.base_batch_size, int(self.max_batch_size * 1.1))
        
        # Adapt timeout
        current_load = self.get_pending_count()
        if current_load > self.max_batch_size * 2:  # High load - reduce timeout
            self.batch_timeout_ms = max(self.min_timeout_ms, self.batch_timeout_ms * 0.9)
        elif current_load < self.max_batch_size // 2:  # Low load - increase timeout
            self.batch_timeout_ms = min(self.max_timeout_ms, self.batch_timeout_ms * 1.1)
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": time.time(),
            "batch_size": self.max_batch_size,
            "timeout_ms": self.batch_timeout_ms,
            "avg_padding_ratio": avg_padding_ratio
        })
    
    def _processing_loop(self):
        """Enhanced processing loop with adaptation"""
        last_batch_time = time.time()
        last_adaptation = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Periodic adaptation
                if current_time - last_adaptation > 10.0:  # Adapt every 10 seconds
                    self._adapt_parameters()
                    last_adaptation = current_time
                
                # Normal processing logic
                time_since_last_batch = (current_time - last_batch_time) * 1000
                
                should_process = False
                with self.batch_lock:
                    total_pending = sum(len(bucket) for bucket in self.length_buckets.values())
                
                if total_pending >= self.max_batch_size:
                    should_process = True
                elif total_pending > 0 and time_since_last_batch >= self.batch_timeout_ms:
                    should_process = True
                
                if should_process:
                    batch = self._select_batch()
                    if batch:
                        self._process_batch(batch)
                        last_batch_time = current_time
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in adaptive processing loop: {e}")
                time.sleep(0.1)


# Alias for backward compatibility
DynamicBatchingOptimizer = AdaptiveBatcher