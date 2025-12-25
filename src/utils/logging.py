"""
Logging and instrumentation module for routing decisions.
Tracks all routing decisions with their signals, stages, and outcomes.
Supports structured decision traces for observability.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from ..router import DecisionTrace


class RoutingLogger:
    """
    Logger for tracking routing decisions and inference metrics.
    Supports two-stage routing logging and structured decision traces.
    """
    
    def __init__(self, log_file: Optional[str] = None, level: str = "INFO"):
        """
        Initialize the routing logger.
        
        Args:
            log_file: Optional path to log file. If None, logs only to console.
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger("adaptive_router")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.decisions = []
        self.decision_traces: List[DecisionTrace] = []
    
    def log_routing_decision(
        self,
        input_id: str,
        route: str,
        routing_stage: Optional[str],
        signals: Dict[str, Any],
        latency_ms: float,
        confidence: float,
        prediction: Any,
        routing_reason: str
    ):
        """
        Log a routing decision with all relevant signals and stage information.
        
        Args:
            input_id: Unique identifier for the input
            route: Either "fast" or "slow"
            routing_stage: "pre_inference" or "post_inference"
            signals: Dictionary of difficulty signals that influenced the decision
            latency_ms: Inference latency in milliseconds
            confidence: Model confidence score
            prediction: Model prediction
            routing_reason: Reason for routing decision
        """
        decision = {
            'timestamp': datetime.now().isoformat(),
            'input_id': input_id,
            'route': route,
            'routing_stage': routing_stage,
            'routing_reason': routing_reason,
            'signals': signals,
            'latency_ms': latency_ms,
            'confidence': confidence,
            'prediction': prediction
        }
        
        self.decisions.append(decision)
        
        self.logger.info(
            f"Routing decision: {route.upper()} | "
            f"Stage: {routing_stage} | "
            f"Input ID: {input_id} | "
            f"Latency: {latency_ms:.2f}ms | "
            f"Confidence: {confidence:.3f} | "
            f"Reason: {routing_reason}"
        )
    
    def log_decision_trace(self, trace: DecisionTrace):
        """
        Log a structured decision trace for observability.
        
        Args:
            trace: DecisionTrace object containing full decision information
        """
        self.decision_traces.append(trace)
        
        # Also log to standard decision log
        self.log_routing_decision(
            input_id=trace.input_id,
            route=trace.route,
            routing_stage=trace.routing_stage,
            signals=trace.signals,
            latency_ms=trace.latency_ms,
            confidence=trace.confidence,
            prediction=trace.prediction,
            routing_reason=trace.routing_reason
        )
        
        # Log latency budget influence if applicable
        if trace.latency_budget_influenced:
            self.logger.info(
                f"Latency budget influenced routing decision for {trace.input_id}: "
                f"P{int(trace.current_latency_percentile * 100) if trace.current_latency_percentile else 0} = "
                f"{trace.current_latency_percentile:.2f}ms"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics from logged routing decisions.
        
        Returns:
            Dictionary containing routing statistics
        """
        if not self.decisions:
            return {}
        
        fast_count = sum(1 for d in self.decisions if d['route'] == 'fast')
        slow_count = sum(1 for d in self.decisions if d['route'] == 'slow')
        total = len(self.decisions)
        
        pre_inference_count = sum(1 for d in self.decisions if d.get('routing_stage') == 'pre_inference')
        post_inference_count = sum(1 for d in self.decisions if d.get('routing_stage') == 'post_inference')
        
        latencies = [d['latency_ms'] for d in self.decisions]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        sorted_latencies = sorted(latencies)
        p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))] if sorted_latencies else 0
        
        # Decision trace statistics
        trace_stats = {}
        if self.decision_traces:
            # Most frequently fired signals
            signal_counts = {}
            budget_influenced_count = sum(1 for t in self.decision_traces if t.latency_budget_influenced)
            
            for trace in self.decision_traces:
                for signal in trace.fired_signals:
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            trace_stats = {
                'total_traces': len(self.decision_traces),
                'budget_influenced_count': budget_influenced_count,
                'budget_influenced_percentage': (budget_influenced_count / len(self.decision_traces) * 100) if self.decision_traces else 0,
                'most_frequent_signals': dict(sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            }
        
        stats = {
            'total_decisions': total,
            'fast_route_count': fast_count,
            'slow_route_count': slow_count,
            'fast_route_percentage': (fast_count / total * 100) if total > 0 else 0,
            'slow_route_percentage': (slow_count / total * 100) if total > 0 else 0,
            'pre_inference_count': pre_inference_count,
            'post_inference_count': post_inference_count,
            'pre_inference_percentage': (pre_inference_count / total * 100) if total > 0 else 0,
            'post_inference_percentage': (post_inference_count / total * 100) if total > 0 else 0,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
        }
        
        if trace_stats:
            stats['decision_trace_stats'] = trace_stats
        
        return stats
    
    def export_decisions(self, output_path: str):
        """
        Export all routing decisions to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.decisions, f, indent=2)
        
        self.logger.info(f"Exported {len(self.decisions)} routing decisions to {output_path}")
    
    def export_decision_traces(self, output_path: str):
        """
        Export all decision traces to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        traces_dict = [
            {
                'input_id': t.input_id,
                'routing_stage': t.routing_stage,
                'signals': t.signals,
                'fired_signals': t.fired_signals,
                'latency_budget_active': t.latency_budget_active,
                'latency_budget_influenced': t.latency_budget_influenced,
                'current_latency_percentile': t.current_latency_percentile,
                'route': t.route,
                'routing_reason': t.routing_reason,
                'prediction': t.prediction,
                'confidence': t.confidence,
                'latency_ms': t.latency_ms
            }
            for t in self.decision_traces
        ]
        
        with open(output_path, 'w') as f:
            json.dump(traces_dict, f, indent=2)
        
        self.logger.info(f"Exported {len(self.decision_traces)} decision traces to {output_path}")
