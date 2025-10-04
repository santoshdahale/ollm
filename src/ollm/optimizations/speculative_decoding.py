"""
Speculative decoding implementation to generate multiple tokens in parallel.
Uses a smaller draft model to propose candidates, verified by the main model.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import time
from dataclasses import dataclass

@dataclass
class SpeculativeStats:
    """Statistics for speculative decoding performance"""
    total_tokens_generated: int = 0
    total_candidates_proposed: int = 0
    accepted_candidates: int = 0
    acceptance_rate: float = 0.0
    speedup_ratio: float = 1.0
    draft_time_ms: float = 0.0
    verification_time_ms: float = 0.0

class SpeculativeDecoder:
    """
    Speculative decoding using a smaller draft model for acceleration.
    
    The draft model generates candidate tokens quickly, and the main model
    verifies them in parallel, accepting correct predictions and rejecting others.
    """
    
    def __init__(self, main_model, draft_model, num_candidates: int = 4,
                 acceptance_threshold: float = 0.8, temperature: float = 1.0):
        self.main_model = main_model
        self.draft_model = draft_model
        self.num_candidates = num_candidates
        self.acceptance_threshold = acceptance_threshold
        self.temperature = temperature
        self.stats = SpeculativeStats()
        
        # Ensure models are in eval mode
        self.main_model.eval()
        self.draft_model.eval()
    
    def generate_candidates(self, input_ids: torch.Tensor, 
                          past_key_values=None, 
                          draft_past_key_values=None) -> Tuple[torch.Tensor, List]:
        """
        Generate candidate tokens using the draft model.
        
        Args:
            input_ids: Current input token IDs (B, L)
            past_key_values: KV cache for main model
            draft_past_key_values: KV cache for draft model
            
        Returns:
            Tuple of (candidate_tokens, draft_kv_cache_states)
        """
        start_time = time.perf_counter()
        
        candidates = []
        current_input = input_ids
        current_draft_kv = draft_past_key_values
        draft_kv_states = []
        
        with torch.no_grad():
            for i in range(self.num_candidates):
                # Forward pass through draft model
                outputs = self.draft_model(
                    input_ids=current_input,
                    past_key_values=current_draft_kv,
                    use_cache=True,
                    return_dict=True
                )
                
                # Sample next token
                logits = outputs.logits[:, -1, :] / self.temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                candidates.append(next_token)
                draft_kv_states.append(outputs.past_key_values)
                
                # Prepare for next iteration
                current_input = next_token
                current_draft_kv = outputs.past_key_values
        
        candidate_tokens = torch.cat(candidates, dim=1)  # (B, num_candidates)
        
        self.stats.draft_time_ms += (time.perf_counter() - start_time) * 1000
        self.stats.total_candidates_proposed += self.num_candidates
        
        return candidate_tokens, draft_kv_states
    
    def verify_candidates(self, input_ids: torch.Tensor, candidate_tokens: torch.Tensor,
                         past_key_values=None) -> Tuple[torch.Tensor, int, any]:
        """
        Verify candidate tokens using the main model.
        
        Args:
            input_ids: Original input tokens (B, L)
            candidate_tokens: Candidate tokens to verify (B, num_candidates)
            past_key_values: KV cache for main model
            
        Returns:
            Tuple of (accepted_tokens, num_accepted, new_kv_cache)
        """
        start_time = time.perf_counter()
        
        batch_size = input_ids.shape[0]
        
        # Create sequence with all candidates for parallel verification
        # Shape: (B, L + num_candidates)
        extended_sequence = torch.cat([input_ids, candidate_tokens], dim=1)
        
        with torch.no_grad():
            # Forward pass through main model with full sequence
            outputs = self.main_model(
                input_ids=extended_sequence,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            # Get logits for positions where we want to verify candidates
            verification_logits = outputs.logits[:, -self.num_candidates-1:-1, :]  # (B, num_candidates, V)
            
            # Convert to probabilities
            verification_probs = F.softmax(verification_logits / self.temperature, dim=-1)
            
            # Check which candidates are accepted
            accepted_tokens = []
            num_accepted = 0
            
            for i in range(self.num_candidates):
                candidate_token = candidate_tokens[:, i:i+1]  # (B, 1)
                
                # Get probability of candidate token from main model
                candidate_prob = torch.gather(
                    verification_probs[:, i, :], 
                    dim=1, 
                    index=candidate_token
                ).squeeze(-1)  # (B,)
                
                # Accept if probability is above threshold
                if candidate_prob.mean().item() >= self.acceptance_threshold:
                    accepted_tokens.append(candidate_token)
                    num_accepted += 1
                else:
                    break  # Reject this and all subsequent candidates
            
            # Prepare accepted sequence
            if accepted_tokens:
                accepted_sequence = torch.cat(accepted_tokens, dim=1)  # (B, num_accepted)
            else:
                # If no candidates accepted, sample from main model distribution
                last_logits = verification_logits[:, 0, :] / self.temperature
                last_probs = F.softmax(last_logits, dim=-1)
                accepted_sequence = torch.multinomial(last_probs, 1)  # (B, 1)
                num_accepted = 1
        
        self.stats.verification_time_ms += (time.perf_counter() - start_time) * 1000
        self.stats.accepted_candidates += num_accepted
        self.stats.total_tokens_generated += num_accepted
        
        # Update acceptance rate
        if self.stats.total_candidates_proposed > 0:
            self.stats.acceptance_rate = self.stats.accepted_candidates / self.stats.total_candidates_proposed
        
        return accepted_sequence, num_accepted, outputs.past_key_values
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 do_sample: bool = True, temperature: float = 1.0,
                 pad_token_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate tokens using speculative decoding.
        
        Args:
            input_ids: Input token IDs (B, L)
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            pad_token_id: Padding token ID
            
        Returns:
            Dictionary with generated tokens and statistics
        """
        self.temperature = temperature
        start_time = time.perf_counter()
        
        # Initialize
        current_ids = input_ids.clone()
        generated_tokens = []
        main_kv_cache = None
        draft_kv_cache = None
        
        total_iterations = 0
        total_generated = 0
        
        while total_generated < max_new_tokens:
            total_iterations += 1
            
            # Generate candidates with draft model
            candidates, draft_kv_states = self.generate_candidates(
                current_ids, main_kv_cache, draft_kv_cache
            )
            
            # Verify candidates with main model
            accepted_tokens, num_accepted, main_kv_cache = self.verify_candidates(
                current_ids, candidates, main_kv_cache
            )
            
            # Update sequence
            current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
            generated_tokens.append(accepted_tokens)
            total_generated += num_accepted
            
            # Update draft model KV cache to accepted position
            if num_accepted > 0 and num_accepted <= len(draft_kv_states):
                draft_kv_cache = draft_kv_states[num_accepted - 1]
            else:
                # Reset draft cache if no acceptance or index out of range
                draft_kv_cache = None
            
            # Early stopping if we've generated enough
            if total_generated >= max_new_tokens:
                break
        
        # Combine all generated tokens
        if generated_tokens:
            all_generated = torch.cat(generated_tokens, dim=1)
            final_sequence = torch.cat([input_ids, all_generated], dim=1)
        else:
            final_sequence = input_ids
        
        # Calculate performance metrics
        total_time = time.perf_counter() - start_time
        tokens_per_second = total_generated / total_time if total_time > 0 else 0
        
        # Estimate speedup (compared to autoregressive)
        expected_autoregressive_time = total_generated * (self.stats.verification_time_ms / self.num_candidates) / 1000
        actual_time = total_time
        self.stats.speedup_ratio = expected_autoregressive_time / actual_time if actual_time > 0 else 1.0
        
        return {
            "sequences": final_sequence,
            "generated_tokens": total_generated,
            "iterations": total_iterations,
            "tokens_per_second": tokens_per_second,
            "acceptance_rate": self.stats.acceptance_rate,
            "speedup_ratio": self.stats.speedup_ratio,
            "stats": self.stats
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = SpeculativeStats()


class TreeAttentionSpeculation:
    """
    Advanced speculative decoding using tree-based attention for multiple candidate paths.
    Explores multiple possible continuations simultaneously.
    """
    
    def __init__(self, main_model, draft_model, tree_depth: int = 3, branching_factor: int = 2):
        self.main_model = main_model
        self.draft_model = draft_model
        self.tree_depth = tree_depth
        self.branching_factor = branching_factor
        
    def build_speculation_tree(self, input_ids: torch.Tensor, 
                              past_key_values=None) -> Dict[str, Any]:
        """
        Build a tree of candidate token sequences.
        
        Args:
            input_ids: Input token IDs (B, L)
            past_key_values: KV cache
            
        Returns:
            Tree structure with candidate paths
        """
        tree = {"root": {"input_ids": input_ids, "children": []}}
        
        def build_level(node, current_depth):
            if current_depth >= self.tree_depth:
                return
            
            current_input = node["input_ids"]
            
            # Generate multiple candidates at this level
            with torch.no_grad():
                outputs = self.draft_model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                
                # Sample multiple candidates
                top_candidates = torch.topk(probs, self.branching_factor, dim=-1)
                
                for i in range(self.branching_factor):
                    candidate_token = top_candidates.indices[:, i:i+1]
                    new_sequence = torch.cat([current_input, candidate_token], dim=1)
                    
                    child_node = {
                        "input_ids": new_sequence,
                        "token": candidate_token,
                        "prob": top_candidates.values[:, i],
                        "children": []
                    }
                    
                    node["children"].append(child_node)
                    
                    # Recursively build deeper levels
                    build_level(child_node, current_depth + 1)
        
        build_level(tree["root"], 0)
        return tree
    
    def verify_tree_paths(self, tree: Dict[str, Any]) -> List[Tuple[torch.Tensor, float]]:
        """
        Verify all paths in the speculation tree using the main model.
        
        Args:
            tree: Speculation tree
            
        Returns:
            List of (path_tokens, verification_score) tuples
        """
        def collect_paths(node, current_path=[]):
            if not node["children"]:
                # Leaf node - complete path
                return [current_path + [node.get("token")]] if "token" in node else [current_path]
            
            paths = []
            for child in node["children"]:
                child_paths = collect_paths(child, current_path + ([node.get("token")] if "token" in node else []))
                paths.extend(child_paths)
            
            return paths
        
        # Collect all paths
        all_paths = collect_paths(tree["root"])
        
        verified_paths = []
        for path in all_paths:
            if path and path[0] is not None:  # Skip empty paths
                # Combine path tokens
                path_tokens = torch.cat([token for token in path if token is not None], dim=1)
                
                # Verify with main model
                full_sequence = torch.cat([tree["root"]["input_ids"], path_tokens], dim=1)
                
                with torch.no_grad():
                    outputs = self.main_model(full_sequence, return_dict=True)
                    # Calculate verification score (e.g., perplexity)
                    logits = outputs.logits[:, -len(path):, :]
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Score based on probability of the path
                    path_score = 0.0
                    for i, token in enumerate(path):
                        if token is not None:
                            token_score = torch.gather(log_probs[:, i, :], dim=1, index=token)
                            path_score += token_score.mean().item()
                
                verified_paths.append((path_tokens, path_score))
        
        # Sort by verification score (higher is better)
        verified_paths.sort(key=lambda x: x[1], reverse=True)
        
        return verified_paths


class AdaptiveSpeculativeDecoder:
    """
    Adaptive speculative decoder that adjusts strategy based on acceptance rates.
    """
    
    def __init__(self, main_model, draft_model):
        self.main_model = main_model
        self.draft_model = draft_model
        self.base_decoder = SpeculativeDecoder(main_model, draft_model)
        self.tree_decoder = TreeAttentionSpeculation(main_model, draft_model)
        
        # Adaptive parameters
        self.min_candidates = 2
        self.max_candidates = 8
        self.target_acceptance_rate = 0.7
        self.adaptation_window = 50  # Number of steps to consider for adaptation
        
    def adapt_strategy(self):
        """Adapt decoding strategy based on recent performance"""
        current_rate = self.base_decoder.stats.acceptance_rate
        
        if current_rate < self.target_acceptance_rate - 0.1:
            # Low acceptance rate - reduce candidates
            new_candidates = max(self.min_candidates, 
                                self.base_decoder.num_candidates - 1)
        elif current_rate > self.target_acceptance_rate + 0.1:
            # High acceptance rate - increase candidates
            new_candidates = min(self.max_candidates,
                                self.base_decoder.num_candidates + 1)
        else:
            # Acceptance rate is good - keep current strategy
            new_candidates = self.base_decoder.num_candidates
        
        self.base_decoder.num_candidates = new_candidates
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 **kwargs) -> Dict[str, Any]:
        """Generate with adaptive strategy"""
        
        # Use base decoder for most generation
        result = self.base_decoder.generate(input_ids, max_new_tokens, **kwargs)
        
        # Periodically adapt strategy
        if self.base_decoder.stats.total_tokens_generated % self.adaptation_window == 0:
            self.adapt_strategy()
        
        return result