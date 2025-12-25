"""
Core routing logic for adaptive inference.
Implements two-stage routing with pluggable policy abstraction.
Supports latency-budget-aware routing and structured decision traces.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RoutingConfig:
    """Configuration for routing thresholds and latency budget."""
    # Pre-inference routing thresholds (input-only signals)
    pre_inference_length_threshold: int
    pre_inference_word_count_threshold: int
    pre_inference_lexical_diversity_threshold: float
    
    # Post-inference routing thresholds (fast model output signals)
    confidence_margin_threshold: float
    entropy_threshold: float
    min_confidence_threshold: float
    
    # Latency budget constraint (optional)
    latency_budget_enabled: bool
    latency_budget_ms: float
    latency_budget_percentile: float  # e.g., 0.95 for P95


@dataclass
class DecisionTrace:
    """
    Structured trace of a routing decision for observability.
    Records all signals, which fired, and final outcome.
    """
    # Input information
    input_id: str
    routing_stage: str
    
    # All evaluated signals
    signals: Dict[str, Any]
    
    # Which signals fired (caused escalation)
    fired_signals: List[str]
    
    # Latency budget influence
    latency_budget_active: bool
    latency_budget_influenced: bool
    current_latency_percentile: Optional[float]
    
    # Final decision
    route: str
    routing_reason: str
    
    # Outcome
    prediction: Any
    confidence: float
    latency_ms: float


class RoutingPolicy(ABC):
    """
    Abstract base class for routing policies.
    Enables pluggable routing strategies without modifying the core router.
    """
    
    @abstractmethod
    def should_route_to_slow_pre_inference(
        self,
        input_features: Dict[str, Any],
        latency_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Determine if input should be routed to slow model before running fast model.
        Uses only input-derived signals (no model inference required).
        
        Args:
            input_features: Dictionary of extracted input features
            latency_stats: Optional latency statistics for budget-aware routing
            
        Returns:
            Tuple of (should_route_to_slow, reason, fired_signals)
            where reason is None if routing to fast, fired_signals is list of signal names that triggered
        """
        pass
    
    @abstractmethod
    def should_route_to_slow_post_inference(
        self,
        fast_model_output: Tuple[int, float, np.ndarray],
        input_features: Dict[str, Any],
        latency_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str, List[str]]:
        """
        Determine if input should be routed to slow model after running fast model.
        Uses fast model outputs and input features.
        
        Args:
            fast_model_output: Tuple of (label, confidence, probabilities) from fast model
            input_features: Dictionary of extracted input features
            latency_stats: Optional latency statistics for budget-aware routing
            
        Returns:
            Tuple of (should_route_to_slow, reason, fired_signals)
            where fired_signals is list of signal names that triggered
        """
        pass


class ThresholdBasedRoutingPolicy(RoutingPolicy):
    """
    Concrete routing policy using configurable thresholds.
    Implements two-stage routing with input-only and model-output signals.
    Supports latency-budget-aware routing constraints.
    """
    
    def __init__(self, config: RoutingConfig):
        """
        Initialize threshold-based routing policy.
        
        Args:
            config: RoutingConfig containing threshold values and latency budget settings
        """
        self.config = config
    
    def compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute Shannon entropy of probability distribution.
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Entropy value
        """
        probs = probabilities + 1e-10
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)
    
    def compute_confidence_margin(self, probabilities: np.ndarray) -> float:
        """
        Compute confidence margin: difference between max and second-max probabilities.
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Confidence margin value
        """
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) < 2:
            return float(sorted_probs[0])
        margin = sorted_probs[0] - sorted_probs[1]
        return float(margin)
    
    def _check_latency_budget(
        self,
        latency_stats: Optional[Dict[str, float]]
    ) -> Tuple[bool, bool]:
        """
        Check if latency budget constraint should influence routing decision.
        
        Args:
            latency_stats: Dictionary with latency statistics (e.g., {'p95': 45.0})
            
        Returns:
            Tuple of (budget_active, should_bias_toward_fast)
            where budget_active indicates if budget checking is enabled,
            and should_bias_toward_fast indicates if we should bias toward fast path
        """
        if not self.config.latency_budget_enabled:
            return False, False
        
        if latency_stats is None:
            return True, False
        
        percentile_key = f"p{int(self.config.latency_budget_percentile * 100)}"
        current_percentile = latency_stats.get(percentile_key)
        
        if current_percentile is None:
            return True, False
        
        # If current latency percentile exceeds budget, bias toward fast path
        should_bias = current_percentile > self.config.latency_budget_ms
        
        return True, should_bias
    
    def should_route_to_slow_pre_inference(
        self,
        input_features: Dict[str, Any],
        latency_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Pre-inference routing using only input-derived signals.
        
        Args:
            input_features: Dictionary of extracted input features
            latency_stats: Optional latency statistics for budget-aware routing
            
        Returns:
            Tuple of (should_route_to_slow, reason, fired_signals)
        """
        fired_signals = []
        length = input_features.get('length', 0)
        word_count = input_features.get('word_count', 0)
        lexical_diversity = input_features.get('lexical_diversity', 0.0)
        
        # Check latency budget constraint
        budget_active, bias_toward_fast = self._check_latency_budget(latency_stats)
        
        # Check length threshold
        if length > self.config.pre_inference_length_threshold:
            # If latency budget is active and biasing toward fast, don't escalate
            if not (budget_active and bias_toward_fast):
                fired_signals.append('length')
                return True, "pre_inference_long_input", fired_signals
        
        # Check word count threshold
        if word_count > self.config.pre_inference_word_count_threshold:
            if not (budget_active and bias_toward_fast):
                fired_signals.append('word_count')
                return True, "pre_inference_high_word_count", fired_signals
        
        # Check lexical diversity threshold
        if lexical_diversity > self.config.pre_inference_lexical_diversity_threshold:
            if not (budget_active and bias_toward_fast):
                fired_signals.append('lexical_diversity')
                return True, "pre_inference_high_lexical_diversity", fired_signals
        
        # Default: proceed to fast model
        return False, None, fired_signals
    
    def should_route_to_slow_post_inference(
        self,
        fast_model_output: Tuple[int, float, np.ndarray],
        input_features: Dict[str, Any],
        latency_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str, List[str]]:
        """
        Post-inference routing using fast model outputs.
        
        Args:
            fast_model_output: Output from fast model
            input_features: Dictionary of extracted input features
            latency_stats: Optional latency statistics for budget-aware routing
            
        Returns:
            Tuple of (should_route_to_slow, reason, fired_signals)
        """
        fired_signals = []
        _, confidence, probabilities = fast_model_output
        
        # Check latency budget constraint
        budget_active, bias_toward_fast = self._check_latency_budget(latency_stats)
        
        # Check minimum confidence threshold
        if confidence < self.config.min_confidence_threshold:
            if not (budget_active and bias_toward_fast):
                fired_signals.append('min_confidence')
                return True, "post_inference_low_confidence", fired_signals
        
        # Check confidence margin threshold
        margin = self.compute_confidence_margin(probabilities)
        if margin < self.config.confidence_margin_threshold:
            if not (budget_active and bias_toward_fast):
                fired_signals.append('confidence_margin')
                return True, "post_inference_low_confidence_margin", fired_signals
        
        # Check entropy threshold
        entropy = self.compute_entropy(probabilities)
        if entropy > self.config.entropy_threshold:
            if not (budget_active and bias_toward_fast):
                fired_signals.append('entropy')
                return True, "post_inference_high_entropy", fired_signals
        
        # Default: accept fast model prediction
        return False, "post_inference_accept_fast", fired_signals


class Router:
    """
    Policy-agnostic router that delegates routing decisions to a RoutingPolicy.
    Supports two-stage routing: pre-inference and post-inference.
    Generates structured decision traces for observability.
    """
    
    def __init__(self, policy: RoutingPolicy):
        """
        Initialize the router with a routing policy.
        
        Args:
            policy: RoutingPolicy implementation for making routing decisions
        """
        self.policy = policy
    
    def route_pre_inference(
        self,
        input_features: Dict[str, Any],
        latency_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Make pre-inference routing decision using only input features.
        
        Args:
            input_features: Dictionary of extracted input features
            latency_stats: Optional latency statistics for budget-aware routing
            
        Returns:
            Tuple of (should_route_to_slow, reason, fired_signals)
        """
        return self.policy.should_route_to_slow_pre_inference(input_features, latency_stats)
    
    def route_post_inference(
        self,
        fast_model_output: Tuple[int, float, np.ndarray],
        input_features: Dict[str, Any],
        latency_stats: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str, List[str]]:
        """
        Make post-inference routing decision using fast model output.
        
        Args:
            fast_model_output: Output from fast model
            input_features: Dictionary of extracted input features
            latency_stats: Optional latency statistics for budget-aware routing
            
        Returns:
            Tuple of (should_route_to_slow, reason, fired_signals)
        """
        return self.policy.should_route_to_slow_post_inference(
            fast_model_output,
            input_features,
            latency_stats
        )
    
    def compute_signals(
        self,
        fast_model_output: Optional[Tuple[int, float, np.ndarray]],
        input_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute all difficulty signals for logging and analysis.
        
        Args:
            fast_model_output: Optional output from fast model (None if skipped)
            input_features: Dictionary of extracted input features
            
        Returns:
            Dictionary containing all difficulty signals
        """
        signals = {
            'input_length': input_features.get('length', 0),
            'word_count': input_features.get('word_count', 0),
            'char_count': input_features.get('char_count', 0),
            'avg_word_length': input_features.get('avg_word_length', 0),
            'lexical_diversity': input_features.get('lexical_diversity', 0.0),
        }
        
        if fast_model_output is not None:
            _, confidence, probabilities = fast_model_output
            signals['fast_model_confidence'] = confidence
            # Only compute these if policy has the methods (ThresholdBasedRoutingPolicy)
            if hasattr(self.policy, 'compute_confidence_margin'):
                signals['confidence_margin'] = self.policy.compute_confidence_margin(probabilities)
            if hasattr(self.policy, 'compute_entropy'):
                signals['entropy'] = self.policy.compute_entropy(probabilities)
        else:
            signals['fast_model_confidence'] = None
            signals['confidence_margin'] = None
            signals['entropy'] = None
        
        return signals
    
    def create_decision_trace(
        self,
        input_id: str,
        routing_stage: str,
        signals: Dict[str, Any],
        fired_signals: List[str],
        latency_budget_active: bool,
        latency_budget_influenced: bool,
        current_latency_percentile: Optional[float],
        route: str,
        routing_reason: str,
        prediction: Any,
        confidence: float,
        latency_ms: float
    ) -> DecisionTrace:
        """
        Create a structured decision trace for observability.
        
        Args:
            input_id: Unique identifier for the input
            routing_stage: "pre_inference" or "post_inference"
            signals: All evaluated signals
            fired_signals: List of signal names that triggered escalation
            latency_budget_active: Whether latency budget checking is enabled
            latency_budget_influenced: Whether latency budget influenced the decision
            current_latency_percentile: Current latency percentile value
            route: Final routing decision ("fast" or "slow")
            routing_reason: Reason for routing decision
            prediction: Model prediction
            confidence: Model confidence
            latency_ms: Total latency in milliseconds
            
        Returns:
            DecisionTrace object
        """
        return DecisionTrace(
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
            latency_ms=latency_ms
        )
