"""
Streaming inference for processing very long sequences by streaming tokens as they arrive.
Enables processing of infinite-length sequences with bounded memory usage.
"""
import torch
import asyncio
from typing import AsyncIterator, Iterator, Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import time
from collections import deque
import threading
from queue import Queue, Empty

@dataclass
class StreamingConfig:
    """Configuration for streaming inference"""
    chunk_size: int = 512
    overlap_size: int = 64
    max_buffer_size: int = 2048
    processing_timeout: float = 30.0
    output_buffer_size: int = 1024

class StreamingInference:
    """
    Streaming inference system that processes tokens as they arrive,
    without waiting for the complete input sequence.
    """
    
    def __init__(self, model, tokenizer, config: StreamingConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or StreamingConfig()
        
        # Streaming state
        self.input_buffer = deque(maxlen=self.config.max_buffer_size)
        self.output_buffer = deque(maxlen=self.config.output_buffer_size)
        self.processing_position = 0
        self.kv_cache = None
        
        # Thread management
        self.processing_thread = None
        self.is_streaming = False
        self.stream_lock = threading.Lock()
        
        # Statistics
        self.processed_chunks = 0
        self.total_tokens_processed = 0
        self.processing_times = []
    
    async def stream_generate(self, input_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """
        Generate tokens from streaming input.
        
        Args:
            input_stream: Async iterator of input text chunks
            
        Yields:
            Generated text chunks
        """
        self.is_streaming = True
        
        try:
            # Start processing task
            processing_task = asyncio.create_task(self._process_stream())
            
            # Feed input stream
            async for input_chunk in input_stream:
                await self._add_input_chunk(input_chunk)
            
            # Signal end of input
            await self._add_input_chunk(None)  # End marker
            
            # Yield generated outputs
            while self.is_streaming or self.output_buffer:
                output_chunk = await self._get_output_chunk()
                if output_chunk is not None:
                    yield output_chunk
                else:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
            
            # Wait for processing to complete
            await processing_task
            
        finally:
            self.is_streaming = False
    
    async def _add_input_chunk(self, chunk: Optional[str]):
        """Add input chunk to buffer"""
        with self.stream_lock:
            if chunk is None:
                self.input_buffer.append(None)  # End marker
            else:
                # Tokenize chunk
                tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
                self.input_buffer.extend(tokens)
    
    async def _get_output_chunk(self) -> Optional[str]:
        """Get output chunk from buffer"""
        with self.stream_lock:
            if self.output_buffer:
                tokens = []
                # Collect a reasonable chunk of tokens
                chunk_size = min(32, len(self.output_buffer))
                for _ in range(chunk_size):
                    if self.output_buffer:
                        tokens.append(self.output_buffer.popleft())
                
                if tokens:
                    return self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return None
    
    async def _process_stream(self):
        """Main processing loop for streaming inference"""
        while self.is_streaming:
            try:
                # Check if we have enough tokens to process
                processable_tokens = await self._get_processable_tokens()
                
                if processable_tokens:
                    # Process chunk
                    start_time = time.perf_counter()
                    output_tokens = await self._process_chunk(processable_tokens)
                    processing_time = time.perf_counter() - start_time
                    
                    # Add to output buffer
                    with self.stream_lock:
                        self.output_buffer.extend(output_tokens)
                        self.processed_chunks += 1
                        self.total_tokens_processed += len(processable_tokens)
                        self.processing_times.append(processing_time)
                
                else:
                    # Check for end condition
                    with self.stream_lock:
                        if None in self.input_buffer and len(self.input_buffer) <= 1:
                            # End of input and no more tokens to process
                            break
                    
                    await asyncio.sleep(0.01)  # Wait for more input
                    
            except Exception as e:
                print(f"Error in stream processing: {e}")
                await asyncio.sleep(0.1)
    
    async def _get_processable_tokens(self) -> Optional[List[int]]:
        """Get tokens ready for processing"""
        with self.stream_lock:
            available_tokens = [t for t in self.input_buffer if t is not None]
            
            # Need at least chunk_size tokens or end of stream
            if len(available_tokens) >= self.config.chunk_size:
                # Extract chunk with overlap handling
                if self.processing_position > 0:
                    # Include overlap from previous chunk
                    start_idx = max(0, self.processing_position - self.config.overlap_size)
                else:
                    start_idx = 0
                
                end_idx = min(len(available_tokens), start_idx + self.config.chunk_size)
                
                chunk_tokens = available_tokens[start_idx:end_idx]
                self.processing_position = end_idx - self.config.overlap_size
                
                return chunk_tokens
            
            elif None in self.input_buffer and available_tokens:
                # End of stream - process remaining tokens
                remaining_tokens = available_tokens[self.processing_position:]
                self.processing_position = len(available_tokens)
                return remaining_tokens if remaining_tokens else None
        
        return None
    
    async def _process_chunk(self, tokens: List[int]) -> List[int]:
        """Process a chunk of tokens"""
        input_ids = torch.tensor([tokens], device=self.model.device)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _forward():
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=self.kv_cache,
                    use_cache=True,
                    return_dict=True
                )
                
                # Update KV cache
                self.kv_cache = outputs.past_key_values
                
                # Generate next token(s)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                return next_token.cpu().tolist()[0]
        
        result = await loop.run_in_executor(None, _forward)
        return result
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics"""
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            "processed_chunks": self.processed_chunks,
            "total_tokens_processed": self.total_tokens_processed,
            "average_processing_time_ms": avg_processing_time * 1000,
            "tokens_per_second": self.total_tokens_processed / sum(self.processing_times) if self.processing_times else 0,
            "input_buffer_size": len(self.input_buffer),
            "output_buffer_size": len(self.output_buffer)
        }


class ChunkedProcessor:
    """
    Processes very long sequences by breaking them into manageable chunks
    with proper context preservation between chunks.
    """
    
    def __init__(self, model, tokenizer, 
                 chunk_size: int = 1024, 
                 overlap_size: int = 128,
                 max_context_length: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_context_length = max_context_length
    
    def process_long_sequence(self, input_text: str, max_new_tokens: int = 100) -> str:
        """
        Process a very long input sequence by chunking.
        
        Args:
            input_text: Long input text
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Tokenize full input
        input_tokens = self.tokenizer.encode(input_text, add_special_tokens=True)
        
        if len(input_tokens) <= self.max_context_length:
            # Short enough for normal processing
            return self._generate_normal(input_tokens, max_new_tokens)
        
        # Process in chunks
        return self._generate_chunked(input_tokens, max_new_tokens)
    
    def _generate_normal(self, input_tokens: List[int], max_new_tokens: int) -> str:
        """Generate using normal (non-chunked) method"""
        input_ids = torch.tensor([input_tokens], device=self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][len(input_tokens):]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def _generate_chunked(self, input_tokens: List[int], max_new_tokens: int) -> str:
        """Generate using chunked processing"""
        generated_text = ""
        kv_cache = None
        
        # Process input in chunks to build context
        num_chunks = (len(input_tokens) + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(input_tokens))
            
            # Add overlap from previous chunk
            if chunk_idx > 0 and self.overlap_size > 0:
                overlap_start = max(0, start_idx - self.overlap_size)
                chunk_tokens = input_tokens[overlap_start:end_idx]
            else:
                chunk_tokens = input_tokens[start_idx:end_idx]
            
            # Process chunk
            chunk_ids = torch.tensor([chunk_tokens], device=self.model.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=chunk_ids,
                    past_key_values=kv_cache,
                    use_cache=True,
                    return_dict=True
                )
                
                # Update KV cache for next chunk
                kv_cache = outputs.past_key_values
        
        # Generate new tokens using final context
        current_length = 0
        
        while current_length < max_new_tokens:
            with torch.no_grad():
                # Generate next token
                dummy_input = torch.tensor([[self.tokenizer.eos_token_id]], device=self.model.device)
                outputs = self.model(
                    input_ids=dummy_input,
                    past_key_values=kv_cache,
                    use_cache=True,
                    return_dict=True
                )
                
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits / 0.7, dim=-1)  # temperature=0.7
                next_token = torch.multinomial(probs, 1)
                
                # Update cache and add token
                kv_cache = outputs.past_key_values
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                generated_text += token_text
                current_length += 1
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return generated_text


class IncrementalProcessor:
    """
    Incremental processor that maintains state across multiple processing calls.
    Useful for interactive applications where input arrives incrementally.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.reset_state()
    
    def reset_state(self):
        """Reset processor state"""
        self.accumulated_tokens = []
        self.kv_cache = None
        self.generation_length = 0
    
    def add_input(self, text: str) -> str:
        """
        Add new input text and generate response incrementally.
        
        Args:
            text: New input text to add
            
        Returns:
            Generated response to the new input
        """
        # Tokenize new input
        new_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.accumulated_tokens.extend(new_tokens)
        
        # Process new tokens
        input_ids = torch.tensor([new_tokens], device=self.model.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=self.kv_cache,
                use_cache=True,
                return_dict=True
            )
            
            # Update KV cache
            self.kv_cache = outputs.past_key_values
            
            # Generate response
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            response_token = torch.multinomial(probs, 1)
            
            response_text = self.tokenizer.decode(response_token[0], skip_special_tokens=True)
            self.generation_length += 1
            
            return response_text
    
    def get_full_context(self) -> str:
        """Get the full accumulated context"""
        return self.tokenizer.decode(self.accumulated_tokens, skip_special_tokens=True)
    
    def trim_context(self, max_length: int = 2048):
        """Trim context to maximum length, keeping recent tokens"""
        if len(self.accumulated_tokens) > max_length:
            # Keep recent tokens
            self.accumulated_tokens = self.accumulated_tokens[-max_length:]
            # Reset KV cache since context changed
            self.kv_cache = None


class StreamingServer:
    """
    Server wrapper for streaming inference with WebSocket-like interface.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.active_streams = {}
        self.stream_counter = 0
    
    def create_stream(self, config: StreamingConfig = None) -> int:
        """Create a new streaming session"""
        stream_id = self.stream_counter
        self.stream_counter += 1
        
        self.active_streams[stream_id] = StreamingInference(
            self.model, self.tokenizer, config
        )
        
        return stream_id
    
    def close_stream(self, stream_id: int):
        """Close a streaming session"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id].is_streaming = False
            del self.active_streams[stream_id]
    
    async def process_stream(self, stream_id: int, input_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Process input stream and yield outputs"""
        if stream_id not in self.active_streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream_processor = self.active_streams[stream_id]
        async for output in stream_processor.stream_generate(input_stream):
            yield output
    
    def get_stream_stats(self, stream_id: int) -> Optional[Dict[str, Any]]:
        """Get statistics for a stream"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id].get_streaming_stats()
        return None
    
    def list_active_streams(self) -> List[int]:
        """Get list of active stream IDs"""
        return list(self.active_streams.keys())


# Alias for backward compatibility
StreamingOptimizer = StreamingInference

