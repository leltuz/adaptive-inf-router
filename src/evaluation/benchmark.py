"""
Offline evaluation and benchmarking module.
Evaluates always-fast, always-slow, and adaptive routing strategies.
Provides conditional, slice-based, failure-mode, and counterfactual metrics.
"""

import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from ..models.fast_model import FastModel
from ..models.slow_model import SlowModel
from ..router import Router
from ..inference.pipeline import AdaptiveInferencePipeline
from ..utils.logging import RoutingLogger
from ..utils.preprocess import preprocess_input


class Benchmark:
    """
    Benchmarking system for evaluating inference strategies.
    Provides conditional, slice-based, failure-mode, and counterfactual analysis.
    """
    
    def __init__(
        self,
        fast_model: FastModel,
        slow_model: SlowModel,
        router: Router,
        fast_cost: float = 1.0,
        slow_cost: float = 10.0
    ):
        """
        Initialize the benchmark.
        
        Args:
            fast_model: Fast inference model
            slow_model: Slow inference model
            router: Router for adaptive routing
            fast_cost: Relative cost per fast inference
            slow_cost: Relative cost per slow inference
        """
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.router = router
        self.fast_cost = fast_cost
        self.slow_cost = slow_cost
    
    def evaluate_always_fast(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate always-fast inference strategy.
        
        Args:
            texts: List of input texts
            labels: List of ground truth labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        start_time = time.time()
        predictions = []
        latencies = []
        
        for text in texts:
            inference_start = time.time()
            label, confidence, probs = self.fast_model.predict(text)
            latency = (time.time() - inference_start) * 1000
            predictions.append(label)
            latencies.append(latency)
        
        total_time = time.time() - start_time
        
        # Compute accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(labels) if labels else 0
        
        # Compute latency statistics
        latencies_sorted = sorted(latencies)
        avg_latency = np.mean(latencies)
        p95_latency = latencies_sorted[int(0.95 * len(latencies_sorted))] if latencies_sorted else 0
        
        # Compute cost
        total_cost = len(texts) * self.fast_cost
        
        return {
            'strategy': 'always_fast',
            'num_samples': len(texts),
            'accuracy': accuracy,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'total_time_s': total_time,
            'total_cost': total_cost,
            'fast_route_percentage': 100.0,
            'slow_route_percentage': 0.0
        }
    
    def evaluate_always_slow(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate always-slow inference strategy.
        
        Args:
            texts: List of input texts
            labels: List of ground truth labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        start_time = time.time()
        predictions = []
        latencies = []
        
        for text in texts:
            inference_start = time.time()
            label, confidence, probs = self.slow_model.predict(text)
            latency = (time.time() - inference_start) * 1000
            predictions.append(label)
            latencies.append(latency)
        
        total_time = time.time() - start_time
        
        # Compute accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(labels) if labels else 0
        
        # Compute latency statistics
        latencies_sorted = sorted(latencies)
        avg_latency = np.mean(latencies)
        p95_latency = latencies_sorted[int(0.95 * len(latencies_sorted))] if latencies_sorted else 0
        
        # Compute cost
        total_cost = len(texts) * self.slow_cost
        
        return {
            'strategy': 'always_slow',
            'num_samples': len(texts),
            'accuracy': accuracy,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'total_time_s': total_time,
            'total_cost': total_cost,
            'fast_route_percentage': 0.0,
            'slow_route_percentage': 100.0
        }
    
    def evaluate_adaptive(
        self,
        texts: List[str],
        labels: List[int],
        logger: RoutingLogger
    ) -> Dict[str, Any]:
        """
        Evaluate adaptive routing strategy with conditional, slice-based, failure-mode, and counterfactual metrics.
        
        Args:
            texts: List of input texts
            labels: List of ground truth labels
            logger: Logger for tracking routing decisions
            
        Returns:
            Dictionary containing evaluation metrics with comprehensive analysis
        """
        pipeline = AdaptiveInferencePipeline(
            self.fast_model,
            self.slow_model,
            self.router,
            logger
        )
        
        start_time = time.time()
        results = pipeline.infer_batch(texts)
        total_time = time.time() - start_time
        
        # Get fast and slow model predictions for all inputs (for failure-mode and counterfactual analysis)
        fast_predictions_all = []
        slow_predictions_all = []
        
        for text in texts:
            fast_label, _, _ = self.fast_model.predict(text)
            slow_label, _, _ = self.slow_model.predict(text)
            fast_predictions_all.append(fast_label)
            slow_predictions_all.append(slow_label)
        
        # Extract predictions and latencies
        predictions = [r['prediction'] for r in results]
        latencies = [r['latency_ms'] for r in results]
        
        # Compute overall accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(labels) if labels else 0
        
        # Compute latency statistics
        latencies_sorted = sorted(latencies)
        avg_latency = np.mean(latencies)
        p95_latency = latencies_sorted[int(0.95 * len(latencies_sorted))] if latencies_sorted else 0
        
        # Compute routing statistics
        fast_count = sum(1 for r in results if r['route'] == 'fast')
        slow_count = sum(1 for r in results if r['route'] == 'slow')
        fast_percentage = (fast_count / len(results) * 100) if results else 0
        slow_percentage = (slow_count / len(results) * 100) if results else 0
        
        # Compute cost
        total_cost = fast_count * self.fast_cost + slow_count * self.slow_cost
        
        # Conditional metrics: accuracy by routing path
        fast_indices = [i for i, r in enumerate(results) if r['route'] == 'fast']
        slow_indices = [i for i, r in enumerate(results) if r['route'] == 'slow']
        
        fast_accuracy = 0.0
        if fast_indices:
            fast_correct = sum(1 for i in fast_indices if predictions[i] == labels[i])
            fast_accuracy = fast_correct / len(fast_indices)
        
        slow_accuracy = 0.0
        if slow_indices:
            slow_correct = sum(1 for i in slow_indices if predictions[i] == labels[i])
            slow_accuracy = slow_correct / len(slow_indices)
        
        # Routing stage distribution
        pre_inference_count = sum(1 for r in results if r.get('routing_stage') == 'pre_inference')
        post_inference_count = sum(1 for r in results if r.get('routing_stage') == 'post_inference')
        
        # Slice-based metrics: escalation rates and accuracy by routing signal
        signal_stats = defaultdict(lambda: {'count': 0, 'correct': 0, 'escalated': 0})
        
        for r in results:
            reason = r.get('routing_reason', 'unknown')
            signals = r.get('signals', {})
            
            # Track by routing reason
            signal_stats[reason]['count'] += 1
            idx = results.index(r)
            if predictions[idx] == labels[idx]:
                signal_stats[reason]['correct'] += 1
            if r['route'] == 'slow':
                signal_stats[reason]['escalated'] += 1
        
        # Convert signal stats to dict
        signal_metrics = {}
        for reason, stats in signal_stats.items():
            signal_metrics[reason] = {
                'count': stats['count'],
                'accuracy': stats['correct'] / stats['count'] if stats['count'] > 0 else 0,
                'escalation_rate': stats['escalated'] / stats['count'] if stats['count'] > 0 else 0
            }
        
        # Failure-mode analysis
        fast_accepted_incorrect = 0  # Fast path accepted but prediction was wrong
        escalation_corrected_error = 0  # Fast was wrong, slow was right, escalation helped
        escalation_no_improvement = 0  # Escalation didn't help (both wrong, or fast was right)
        escalation_count = 0  # Total escalations
        
        for i, (result, label) in enumerate(zip(results, labels)):
            fast_pred = fast_predictions_all[i]
            slow_pred = slow_predictions_all[i]
            adaptive_pred = result['prediction']
            route = result['route']
            
            # Case 1: Fast path accepted but incorrect
            if route == 'fast' and adaptive_pred != label:
                fast_accepted_incorrect += 1
            
            # Cases 2 and 3: Escalation analysis
            if route == 'slow':
                escalation_count += 1
                fast_correct = (fast_pred == label)
                slow_correct = (slow_pred == label)
                adaptive_correct = (adaptive_pred == label)
                
                # Case 2: Escalation corrected an error
                # Fast was wrong, slow was right, and we got the right answer
                if not fast_correct and slow_correct and adaptive_correct:
                    escalation_corrected_error += 1
                
                # Case 3: Escalation did not improve
                # Either both were wrong, or fast was right but slow was wrong
                if (not fast_correct and not slow_correct) or (fast_correct and not slow_correct):
                    escalation_no_improvement += 1
        
        failure_mode_metrics = {
            'fast_accepted_incorrect_count': fast_accepted_incorrect,
            'fast_accepted_incorrect_rate': fast_accepted_incorrect / len(results) if results else 0,
            'escalation_corrected_error_count': escalation_corrected_error,
            'escalation_corrected_error_rate': escalation_corrected_error / escalation_count if escalation_count > 0 else 0,
            'escalation_no_improvement_count': escalation_no_improvement,
            'escalation_no_improvement_rate': escalation_no_improvement / escalation_count if escalation_count > 0 else 0,
            'total_escalations': escalation_count
        }
        
        # Counterfactual routing analysis
        # Compare adaptive routing decision against alternative paths for each input
        cost_savings_without_sacrifice = 0  # Adaptive used fast, fast was correct, slow would have been correct too
        cost_savings_with_sacrifice = 0  # Adaptive used fast, fast was correct, slow would have been wrong
        unnecessary_escalation = 0  # Adaptive escalated, slow was correct, fast would have been correct too
        escalation_changed_correctness = 0  # Adaptive changed correctness vs fast-only (either improved or degraded)
        correctness_improved = 0  # Adaptive escalated and improved correctness vs fast-only
        correctness_degraded = 0  # Adaptive escalated but degraded correctness vs fast-only
        
        for i, (result, label) in enumerate(zip(results, labels)):
            fast_pred = fast_predictions_all[i]
            slow_pred = slow_predictions_all[i]
            adaptive_pred = result['prediction']
            route = result['route']
            
            fast_correct = (fast_pred == label)
            slow_correct = (slow_pred == label)
            adaptive_correct = (adaptive_pred == label)
            
            if route == 'fast':
                # Adaptive chose fast path
                if fast_correct:
                    if slow_correct:
                        # Both correct: adaptive saved cost without sacrificing correctness
                        cost_savings_without_sacrifice += 1
                    else:
                        # Fast correct, slow wrong: adaptive saved cost and avoided error
                        cost_savings_with_sacrifice += 1
            else:
                # Adaptive chose slow path
                if slow_correct:
                    if fast_correct:
                        # Both correct: adaptive escalated unnecessarily
                        unnecessary_escalation += 1
                
                # Check if escalation changed correctness
                if fast_correct != adaptive_correct:
                    escalation_changed_correctness += 1
                    if adaptive_correct and not fast_correct:
                        correctness_improved += 1
                    elif not adaptive_correct and fast_correct:
                        correctness_degraded += 1
        
        total_samples = len(results)
        counterfactual_metrics = {
            'cost_savings_without_sacrifice_count': cost_savings_without_sacrifice,
            'cost_savings_without_sacrifice_rate': cost_savings_without_sacrifice / total_samples if total_samples > 0 else 0,
            'cost_savings_with_sacrifice_count': cost_savings_with_sacrifice,
            'cost_savings_with_sacrifice_rate': cost_savings_with_sacrifice / total_samples if total_samples > 0 else 0,
            'unnecessary_escalation_count': unnecessary_escalation,
            'unnecessary_escalation_rate': unnecessary_escalation / total_samples if total_samples > 0 else 0,
            'escalation_changed_correctness_count': escalation_changed_correctness,
            'escalation_changed_correctness_rate': escalation_changed_correctness / total_samples if total_samples > 0 else 0,
            'correctness_improved_count': correctness_improved,
            'correctness_improved_rate': correctness_improved / total_samples if total_samples > 0 else 0,
            'correctness_degraded_count': correctness_degraded,
            'correctness_degraded_rate': correctness_degraded / total_samples if total_samples > 0 else 0
        }
        
        return {
            'strategy': 'adaptive',
            'num_samples': len(texts),
            'accuracy': accuracy,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'total_time_s': total_time,
            'total_cost': total_cost,
            'fast_route_percentage': fast_percentage,
            'slow_route_percentage': slow_percentage,
            'fast_route_count': fast_count,
            'slow_route_count': slow_count,
            # Conditional metrics
            'fast_path_accuracy': fast_accuracy,
            'slow_path_accuracy': slow_accuracy,
            'fast_path_count': len(fast_indices),
            'slow_path_count': len(slow_indices),
            # Routing stage distribution
            'pre_inference_count': pre_inference_count,
            'post_inference_count': post_inference_count,
            'pre_inference_percentage': (pre_inference_count / len(results) * 100) if results else 0,
            'post_inference_percentage': (post_inference_count / len(results) * 100) if results else 0,
            # Slice-based metrics
            'signal_metrics': signal_metrics,
            # Failure-mode metrics
            'failure_mode_metrics': failure_mode_metrics,
            # Counterfactual metrics
            'counterfactual_metrics': counterfactual_metrics
        }
    
    def run_full_benchmark(
        self,
        texts: List[str],
        labels: List[int],
        output_dir: str = "results",
        logger: Optional[RoutingLogger] = None
    ) -> Dict[str, Any]:
        """
        Run complete benchmark comparing all three strategies.
        
        Args:
            texts: List of input texts
            labels: List of ground truth labels
            output_dir: Directory to save results
            logger: Optional logger for adaptive routing
            
        Returns:
            Dictionary containing all benchmark results
        """
        print("Running benchmark: always-fast strategy...")
        always_fast_results = self.evaluate_always_fast(texts, labels)
        
        print("Running benchmark: always-slow strategy...")
        always_slow_results = self.evaluate_always_slow(texts, labels)
        
        print("Running benchmark: adaptive routing strategy...")
        if logger is None:
            logger = RoutingLogger()
        adaptive_results = self.evaluate_adaptive(texts, labels, logger)
        
        results = {
            'always_fast': always_fast_results,
            'always_slow': always_slow_results,
            'adaptive': adaptive_results
        }
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results_path = Path(output_dir) / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBenchmark results saved to {results_path}")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """
        Print formatted benchmark results including conditional, failure-mode, and counterfactual metrics.
        
        Args:
            results: Dictionary containing benchmark results
        """
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        for strategy_name, strategy_results in results.items():
            print(f"\n{strategy_name.upper().replace('_', ' ')}")
            print("-" * 80)
            print(f"  Accuracy:              {strategy_results['accuracy']:.4f}")
            print(f"  Avg Latency:           {strategy_results['avg_latency_ms']:.2f} ms")
            print(f"  P95 Latency:           {strategy_results['p95_latency_ms']:.2f} ms")
            print(f"  Total Time:            {strategy_results['total_time_s']:.2f} s")
            print(f"  Total Cost:            {strategy_results['total_cost']:.2f}")
            print(f"  Fast Route %:          {strategy_results['fast_route_percentage']:.2f}%")
            print(f"  Slow Route %:          {strategy_results['slow_route_percentage']:.2f}%")
            
            # Print conditional metrics for adaptive strategy
            if strategy_name == 'adaptive':
                print(f"\n  Conditional Metrics:")
                print(f"    Fast Path Accuracy:  {strategy_results.get('fast_path_accuracy', 0):.4f} ({strategy_results.get('fast_path_count', 0)} samples)")
                print(f"    Slow Path Accuracy: {strategy_results.get('slow_path_accuracy', 0):.4f} ({strategy_results.get('slow_path_count', 0)} samples)")
                print(f"\n  Routing Stage Distribution:")
                print(f"    Pre-inference:      {strategy_results.get('pre_inference_percentage', 0):.2f}% ({strategy_results.get('pre_inference_count', 0)} samples)")
                print(f"    Post-inference:     {strategy_results.get('post_inference_percentage', 0):.2f}% ({strategy_results.get('post_inference_count', 0)} samples)")
                
                # Print signal-based metrics
                signal_metrics = strategy_results.get('signal_metrics', {})
                if signal_metrics:
                    print(f"\n  Signal-Based Metrics:")
                    for reason, metrics in sorted(signal_metrics.items()):
                        print(f"    {reason}:")
                        print(f"      Count: {metrics['count']}, Accuracy: {metrics['accuracy']:.4f}, Escalation Rate: {metrics['escalation_rate']:.4f}")
                
                # Print failure-mode metrics
                failure_metrics = strategy_results.get('failure_mode_metrics', {})
                if failure_metrics:
                    print(f"\n  Failure-Mode Analysis:")
                    print(f"    Fast Accepted but Incorrect: {failure_metrics.get('fast_accepted_incorrect_count', 0)} ({failure_metrics.get('fast_accepted_incorrect_rate', 0):.4f})")
                    print(f"    Escalation Corrected Error: {failure_metrics.get('escalation_corrected_error_count', 0)} ({failure_metrics.get('escalation_corrected_error_rate', 0):.4f})")
                    print(f"    Escalation No Improvement: {failure_metrics.get('escalation_no_improvement_count', 0)} ({failure_metrics.get('escalation_no_improvement_rate', 0):.4f})")
                    print(f"    Total Escalations: {failure_metrics.get('total_escalations', 0)}")
                
                # Print counterfactual metrics
                counterfactual_metrics = strategy_results.get('counterfactual_metrics', {})
                if counterfactual_metrics:
                    print(f"\n  Counterfactual Routing Analysis:")
                    print(f"    Cost Savings (No Sacrifice): {counterfactual_metrics.get('cost_savings_without_sacrifice_count', 0)} ({counterfactual_metrics.get('cost_savings_without_sacrifice_rate', 0):.4f})")
                    print(f"    Cost Savings (Avoided Error): {counterfactual_metrics.get('cost_savings_with_sacrifice_count', 0)} ({counterfactual_metrics.get('cost_savings_with_sacrifice_rate', 0):.4f})")
                    print(f"    Unnecessary Escalation: {counterfactual_metrics.get('unnecessary_escalation_count', 0)} ({counterfactual_metrics.get('unnecessary_escalation_rate', 0):.4f})")
                    print(f"    Escalation Changed Correctness: {counterfactual_metrics.get('escalation_changed_correctness_count', 0)} ({counterfactual_metrics.get('escalation_changed_correctness_rate', 0):.4f})")
                    print(f"      Improved: {counterfactual_metrics.get('correctness_improved_count', 0)} ({counterfactual_metrics.get('correctness_improved_rate', 0):.4f})")
                    print(f"      Degraded: {counterfactual_metrics.get('correctness_degraded_count', 0)} ({counterfactual_metrics.get('correctness_degraded_rate', 0):.4f})")
        
        print("\n" + "="*80)
