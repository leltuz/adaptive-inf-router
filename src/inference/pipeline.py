"""
Unified inference pipeline that hides routing complexity from callers.
Implements two-stage routing: pre-inference and post-inference.
Supports latency-budget-aware routing and decision traceability.
"""

import time
from typing import Dict, Any, Tuple, Optional, List
from ..router import Router, DecisionTrace
from ..models.fast_model import FastModel
from ..models.slow_model import SlowModel
from ..utils.preprocess import preprocess_input
from ..utils.logging import RoutingLogger


class AdaptiveInferencePipeline:
    """
    Main inference pipeline that routes inputs adaptively between
    fast and slow inference paths using two-stage routing.
    Supports latency-budget-aware routing and structured decision traces.
    """
    
    def __init__(
        self,
        fast_model: FastModel,
        slow_model: SlowModel,
        router: Router,
        logger: Optional[RoutingLogger] = None,
        latency_stats: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            fast_model: Fast approximate inference model
            slow_model: Slow accurate inference model
            router: Router for making routing decisions
            logger: Optional logger for routing decisions
            latency_stats: Optional latency statistics for budget-aware routing
        """
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.router = router
        self.logger = logger
        self.latency_stats = latency_stats
        self.input_counter = 0
        self.latency_history: List[float] = []
    
    def _update_latency_stats(self, latency_ms: float) -> Dict[str, float]:
        """
        Update latency statistics from history for budget-aware routing.
        Computes rolling percentiles deterministically.
        
        Args:
            latency_ms: Latest latency measurement
            
        Returns:
            Dictionary with latency statistics (p50, p95, etc.)
        """
        self.latency_history.append(latency_ms)
        
        if len(self.latency_history) < 10:
            # Need minimum samples for meaningful percentiles
            return {}
        
        sorted_latencies = sorted(self.latency_history)
        n = len(sorted_latencies)
        
        stats = {
            'p50': sorted_latencies[int(0.50 * n)],
            'p75': sorted_latencies[int(0.75 * n)],
            'p90': sorted_latencies[int(0.90 * n)],
            'p95': sorted_latencies[int(0.95 * n)],
            'p99': sorted_latencies[int(0.99 * n)] if n > 50 else sorted_latencies[-1],
        }
        
        return stats
    
    def infer(self, text: str, input_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform adaptive inference on input text with two-stage routing.
        Generates structured decision traces for observability.
        
        Args:
            text: Input text string
            input_id: Optional unique identifier for this input
            
        Returns:
            Dictionary containing prediction, route, latency, and metadata
        """
        if input_id is None:
            self.input_counter += 1
            input_id = f"input_{self.input_counter}"
        
        start_time = time.time()
        
        # Preprocess input
        preprocessed = preprocess_input(text)
        input_features = preprocessed['features']
        
        # Get current latency statistics for budget-aware routing
        current_latency_stats = self.latency_stats
        if current_latency_stats is None and len(self.latency_history) >= 10:
            current_latency_stats = self._update_latency_stats(0.0)  # Will be updated after this inference
        
        # Stage 1: Pre-inference routing (input-only signals)
        should_route_slow_pre, pre_reason, pre_fired_signals = self.router.route_pre_inference(
            input_features, current_latency_stats
        )
        
        routing_stage = None
        fast_latency = 0.0
        fast_model_output = None
        post_fired_signals: List[str] = []
        
        # Check latency budget status
        config = self.router.policy.config
        latency_budget_active = config.latency_budget_enabled
        latency_budget_influenced = False
        current_latency_percentile = None
        
        if latency_budget_active and current_latency_stats:
            percentile_key = f"p{int(config.latency_budget_percentile * 100)}"
            current_latency_percentile = current_latency_stats.get(percentile_key)
            if current_latency_percentile and current_latency_percentile > config.latency_budget_ms:
                latency_budget_influenced = True
        
        if should_route_slow_pre:
            # Skip fast model, route directly to slow model
            routing_stage = "pre_inference"
            routing_reason = pre_reason
            fired_signals = pre_fired_signals
        else:
            # Run fast model for post-inference routing
            fast_start = time.time()
            fast_label, fast_confidence, fast_probs = self.fast_model.predict(text)
            fast_latency = (time.time() - fast_start) * 1000
            fast_model_output = (fast_label, fast_confidence, fast_probs)
            
            # Update latency stats after fast model
            if current_latency_stats is None:
                current_latency_stats = self._update_latency_stats(fast_latency)
            
            # Stage 2: Post-inference routing (fast model outputs)
            should_route_slow_post, post_reason, post_fired_signals = self.router.route_post_inference(
                fast_model_output,
                input_features,
                current_latency_stats
            )
            
            routing_stage = "post_inference"
            routing_reason = post_reason
            fired_signals = post_fired_signals
            
            # Check if latency budget influenced post-inference decision
            if latency_budget_active and current_latency_stats:
                percentile_key = f"p{int(config.latency_budget_percentile * 100)}"
                current_latency_percentile = current_latency_stats.get(percentile_key)
                if current_latency_percentile and current_latency_percentile > config.latency_budget_ms:
                    # Check if signals fired but were overridden by budget
                    if post_fired_signals and not should_route_slow_post:
                        latency_budget_influenced = True
        
        # Route to appropriate model
        if should_route_slow_pre or (fast_model_output is not None and should_route_slow_post):
            # Route to slow model
            slow_start = time.time()
            slow_label, slow_confidence, slow_probs = self.slow_model.predict(text)
            slow_latency = (time.time() - slow_start) * 1000
            total_latency = (time.time() - start_time) * 1000
            
            prediction = slow_label
            confidence = slow_confidence
            probabilities = slow_probs
            route = "slow"
        else:
            # Use fast model output
            prediction = fast_label
            confidence = fast_confidence
            probabilities = fast_probs
            total_latency = (time.time() - start_time) * 1000
            route = "fast"
        
        # Update latency history
        self._update_latency_stats(total_latency)
        
        # Compute all signals for logging
        signals = self.router.compute_signals(fast_model_output, input_features)
        
        # Create structured decision trace
        decision_trace = self.router.create_decision_trace(
            input_id=input_id,
            routing_stage=routing_stage,
            signals=signals,
            fired_signals=fired_signals,
            latency_budget_active=latency_budget_active,
            latency_budget_influenced=latency_budget_influenced,
            current_latency_percentile=current_latency_percentile,
            route=route,
            routing_reason=routing_reason,
            prediction=prediction,
            confidence=confidence,
            latency_ms=total_latency
        )
        
        result = {
            'input_id': input_id,
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities,
            'route': route,
            'routing_stage': routing_stage,
            'routing_reason': routing_reason,
            'latency_ms': total_latency,
            'fast_latency_ms': fast_latency,
            'slow_latency_ms': total_latency - fast_latency if route == "slow" else 0,
            'fast_model_skipped': fast_model_output is None,
            'signals': signals,
            'fired_signals': fired_signals,
            'latency_budget_influenced': latency_budget_influenced,
            'decision_trace': decision_trace
        }
        
        # Log routing decision and trace
        if self.logger:
            self.logger.log_decision_trace(decision_trace)
        
        return result
    
    def infer_batch(self, texts: list, input_ids: Optional[list] = None) -> list:
        """
        Perform batch adaptive inference on multiple texts.
        
        Args:
            texts: List of input text strings
            input_ids: Optional list of unique identifiers
            
        Returns:
            List of inference result dictionaries
        """
        if input_ids is None:
            input_ids = [f"input_{i}" for i in range(len(texts))]
        
        results = []
        for text, input_id in zip(texts, input_ids):
            result = self.infer(text, input_id)
            results.append(result)
        
        return results
