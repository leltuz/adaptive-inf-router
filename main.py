"""
Main entry point for Adaptive Inference Router.
Supports both interactive inference and offline evaluation.
"""

import argparse
import yaml
import numpy as np
from pathlib import Path

from src.models.fast_model import FastModel
from src.models.slow_model import SlowModel
from src.router import Router, RoutingConfig, ThresholdBasedRoutingPolicy
from src.inference.pipeline import AdaptiveInferencePipeline
from src.evaluation.benchmark import Benchmark
from src.utils.logging import RoutingLogger
from data.sample_data import generate_sample_texts


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_models(config: dict):
    """Initialize fast and slow models from configuration."""
    fast_config = config['models']['fast']
    slow_config = config['models']['slow']
    
    fast_model = FastModel(
        latency_ms=fast_config['latency_ms'],
        accuracy=fast_config['accuracy']
    )
    
    slow_model = SlowModel(
        latency_ms=slow_config['latency_ms'],
        accuracy=slow_config['accuracy']
    )
    
    return fast_model, slow_model


def initialize_router(config: dict) -> Router:
    """Initialize router with policy from configuration."""
    routing_config = config['routing']
    router_config = RoutingConfig(
        pre_inference_length_threshold=routing_config['pre_inference_length_threshold'],
        pre_inference_word_count_threshold=routing_config['pre_inference_word_count_threshold'],
        pre_inference_lexical_diversity_threshold=routing_config.get('pre_inference_lexical_diversity_threshold', 0.85),
        confidence_margin_threshold=routing_config['confidence_margin_threshold'],
        entropy_threshold=routing_config['entropy_threshold'],
        min_confidence_threshold=routing_config['min_confidence_threshold'],
        latency_budget_enabled=routing_config.get('latency_budget_enabled', False),
        latency_budget_ms=routing_config.get('latency_budget_ms', 50.0),
        latency_budget_percentile=routing_config.get('latency_budget_percentile', 0.95)
    )
    
    # Create policy and router
    policy = ThresholdBasedRoutingPolicy(router_config)
    router = Router(policy)
    
    return router


def train_models(fast_model: FastModel, slow_model: SlowModel, texts: list, labels: list):
    """Train both models on the provided data."""
    print("Training fast model...")
    fast_model.train(texts, labels)
    
    print("Training slow model...")
    slow_model.train(texts, labels)
    
    print("Models trained successfully.")


def run_inference(text: str, config_path: str = "config.yaml"):
    """Run inference on a single text input."""
    config = load_config(config_path)
    fast_model, slow_model = initialize_models(config)
    
    # Generate training data and train models
    train_texts, train_labels = generate_sample_texts(num_samples=500, seed=42)
    train_models(fast_model, slow_model, train_texts, train_labels)
    
    # Initialize router and pipeline
    router = initialize_router(config)
    logger = RoutingLogger(
        log_file=config['logging'].get('log_file'),
        level=config['logging'].get('level', 'INFO')
    )
    
    pipeline = AdaptiveInferencePipeline(fast_model, slow_model, router, logger)
    
    # Run inference
    result = pipeline.infer(text)
    
    print("\n" + "="*80)
    print("INFERENCE RESULT")
    print("="*80)
    print(f"Input:              {text}")
    print(f"Prediction:         {result['prediction']}")
    print(f"Confidence:         {result['confidence']:.4f}")
    print(f"Route:              {result['route'].upper()}")
    print(f"Routing Stage:      {result['routing_stage']}")
    print(f"Routing Reason:     {result['routing_reason']}")
    print(f"Fast Model Skipped: {result['fast_model_skipped']}")
    print(f"Fired Signals:      {result.get('fired_signals', [])}")
    print(f"Latency Budget Influenced: {result.get('latency_budget_influenced', False)}")
    print(f"Latency:            {result['latency_ms']:.2f} ms")
    print(f"  Fast Latency:     {result['fast_latency_ms']:.2f} ms")
    print(f"  Slow Latency:     {result['slow_latency_ms']:.2f} ms")
    print(f"Signals:            {result['signals']}")
    print("="*80)
    
    return result


def run_evaluation(config_path: str = "config.yaml"):
    """Run offline evaluation benchmark."""
    config = load_config(config_path)
    
    # Initialize models
    fast_model, slow_model = initialize_models(config)
    
    # Generate evaluation dataset
    eval_config = config['evaluation']
    print(f"Generating {eval_config['num_samples']} evaluation samples...")
    texts, labels = generate_sample_texts(
        num_samples=eval_config['num_samples'],
        seed=eval_config['seed']
    )
    
    # Split into train and test
    split_idx = len(texts) // 2
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    test_texts = texts[split_idx:]
    test_labels = labels[split_idx:]
    
    # Train models
    print("\nTraining models...")
    train_models(fast_model, slow_model, train_texts, train_labels)
    
    # Initialize router and logger
    router = initialize_router(config)
    logger = RoutingLogger(
        log_file=config['logging'].get('log_file'),
        level=config['logging'].get('level', 'INFO')
    )
    
    # Initialize benchmark
    cost_config = config['cost']
    benchmark = Benchmark(
        fast_model=fast_model,
        slow_model=slow_model,
        router=router,
        fast_cost=cost_config['fast_model_cost'],
        slow_cost=cost_config['slow_model_cost']
    )
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = benchmark.run_full_benchmark(
        texts=test_texts,
        labels=test_labels,
        output_dir=eval_config['output_dir'],
        logger=logger
    )
    
    # Print results
    benchmark.print_results(results)
    
    # Export routing decisions and traces
    if logger:
        decisions_path = Path(eval_config['output_dir']) / "routing_decisions.json"
        logger.export_decisions(str(decisions_path))
        
        traces_path = Path(eval_config['output_dir']) / "decision_traces.json"
        logger.export_decision_traces(str(traces_path))
        
        stats = logger.get_statistics()
        print(f"\nRouting Statistics:")
        print(f"  Total Decisions: {stats.get('total_decisions', 0)}")
        print(f"  Fast Route: {stats.get('fast_route_percentage', 0):.2f}%")
        print(f"  Slow Route: {stats.get('slow_route_percentage', 0):.2f}%")
        print(f"  Pre-inference: {stats.get('pre_inference_percentage', 0):.2f}%")
        print(f"  Post-inference: {stats.get('post_inference_percentage', 0):.2f}%")
        
        # Print decision trace statistics
        trace_stats = stats.get('decision_trace_stats', {})
        if trace_stats:
            print(f"\nDecision Trace Statistics:")
            print(f"  Total Traces: {trace_stats.get('total_traces', 0)}")
            print(f"  Budget Influenced: {trace_stats.get('budget_influenced_count', 0)} ({trace_stats.get('budget_influenced_percentage', 0):.2f}%)")
            most_frequent = trace_stats.get('most_frequent_signals', {})
            if most_frequent:
                print(f"  Most Frequent Signals:")
                for signal, count in list(most_frequent.items())[:5]:
                    print(f"    {signal}: {count}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive Inference Router - Dynamic routing between fast and slow inference paths"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['inference', 'evaluate'],
        default='evaluate',
        help='Mode: inference for single input, evaluate for offline benchmark'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Input text for inference mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'inference':
        if not args.text:
            print("Error: --text required for inference mode")
            return
        run_inference(args.text, args.config)
    elif args.mode == 'evaluate':
        run_evaluation(args.config)


if __name__ == "__main__":
    main()
