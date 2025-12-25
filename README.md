# Adaptive Inference Router

A production-inspired system for dynamically routing machine learning inference requests between fast approximate and slow accurate inference paths. The system reduces latency and compute cost while maintaining acceptable accuracy by intelligently selecting the appropriate inference path based on input difficulty estimation.

## Scope and Out-of-Scope

This system demonstrates compute-aware adaptive inference routing in a controlled, offline evaluation setting. It is designed to illustrate systems engineering trade-offs and routing policy design, not to serve as a complete production deployment.

**Intentionally Out of Scope:**
- **Concurrency and parallelism**: Single-threaded execution for deterministic evaluation
- **Request batching**: Sequential processing for clear latency accounting
- **Deployment infrastructure**: No serving framework, API endpoints, or containerization
- **Production monitoring**: No metrics collection, alerting, or dashboards
- **Model serving**: No model versioning, A/B testing, or gradual rollouts
- **Distributed systems**: No load balancing, replication, or fault tolerance
- **Online learning**: No threshold adaptation or model retraining during operation

These exclusions allow the system to focus on core routing logic, policy design, and offline evaluation methodology without the complexity of production infrastructure concerns.

## Problem Statement

Modern machine learning services face a fundamental trade-off: high-accuracy models are computationally expensive and slow, while fast models sacrifice accuracy. Running high-accuracy models for all inputs is inefficient because many inputs are straightforward and can be handled correctly by simpler, faster models. Conversely, always using fast models leads to unacceptable accuracy degradation on difficult inputs.

This system addresses this efficiency problem by implementing an adaptive routing mechanism that:
- Estimates input difficulty using cheap signals (both input-derived and model-output)
- Routes easy inputs through a fast, low-cost inference path
- Escalates difficult inputs to a slower, higher-accuracy inference path
- Maintains a single unified interface that hides routing complexity

## Why Naive Always-On Inference is Inefficient

**Always-Fast Inference:**
- Low latency and cost, but accuracy suffers on complex inputs
- Wastes compute on simple inputs that don't require it
- Cannot adapt to varying input difficulty

**Always-Slow Inference:**
- High accuracy, but incurs unnecessary latency and compute cost on simple inputs
- Does not optimize for the common case (most inputs are easy)
- Fails to meet latency budgets for real-time applications

**Adaptive Routing:**
- Achieves near-slow accuracy with near-fast latency on average
- Reduces compute cost by routing most inputs through the fast path
- Maintains accuracy on difficult inputs by escalating to slow path
- Provides deterministic, reproducible routing decisions

## Architecture Overview

The system implements a two-stage routing architecture with a pluggable policy abstraction:

```
┌─────────────────┐
│  Input Text     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ ──► Extract features (length, word count, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 1:       │ ──► Pre-inference routing (input-only signals)
│  Pre-Inference  │     Skip fast model if input clearly hard
│  Router         │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────┐  ┌─────────────────┐
│ Slow │  │   Fast Model    │ ──► Run if not skipped
│ Path │  └────────┬────────┘
└──────┘           │
                   ▼
         ┌─────────────────┐
         │  Stage 2:       │ ──► Post-inference routing (model outputs)
         │  Post-Inference │     Escalate if fast model uncertain
         │  Router         │
         └────────┬────────┘
                  │
             ┌────┴────┐
             │         │
             ▼         ▼
         ┌──────┐  ┌──────┐
         │ Fast │  │ Slow │
         │ Path │  │ Path │
         └──┬───┘  └──┬───┘
            │         │
            └────┬────┘
                 │
                 ▼
         ┌─────────────────┐
         │  Unified Output │ ──► Single interface for caller
         └─────────────────┘
```

### Core Components

1. **Preprocessing Module** (`src/utils/preprocess.py`)
   - Normalizes input text
   - Extracts simple features (length, word count, character count, etc.)
   - Prepares inputs for model inference

2. **Fast Model** (`src/models/fast_model.py`)
   - Lightweight logistic regression with TF-IDF features
   - Low latency (~10ms simulated)
   - Moderate accuracy (~85%)
   - May be skipped if pre-inference routing determines input is clearly hard

3. **Slow Model** (`src/models/slow_model.py`)
   - More complex logistic regression with richer features
   - Higher latency (~100ms simulated)
   - Higher accuracy (~95%)
   - Clearly outperforms fast model on hard inputs

4. **Routing Policy Abstraction** (`src/router.py`)
   - **RoutingPolicy**: Abstract base class defining routing interface
   - **ThresholdBasedRoutingPolicy**: Concrete implementation using configurable thresholds
   - Enables pluggable routing strategies without modifying core router
   - Supports two-stage routing: pre-inference and post-inference

5. **Router** (`src/router.py`)
   - Policy-agnostic router that delegates to a RoutingPolicy
   - Supports pre-inference routing (input-only signals)
   - Supports post-inference routing (fast model outputs)
   - Computes difficulty signals for logging and analysis

6. **Inference Pipeline** (`src/inference/pipeline.py`)
   - Unified interface that hides routing complexity
   - Implements two-stage routing flow
   - Manages fast model skipping when appropriate
   - Logs all routing decisions with stage information

7. **Evaluation Module** (`src/evaluation/benchmark.py`)
   - Offline evaluation comparing three strategies:
     - Always-fast inference
     - Always-slow inference
     - Adaptive routing
   - Reports global metrics: accuracy, latency, cost, routing distribution
   - Reports conditional metrics: accuracy by routing path, escalation rates by signal
   - Reports slice-based metrics: distribution by routing stage

8. **Logging** (`src/utils/logging.py`)
   - Tracks all routing decisions with signals, stages, and outcomes
   - Provides statistics on routing behavior
   - Exports decisions for analysis

## Routing Policy

The system uses a pluggable routing policy architecture. The default implementation (`ThresholdBasedRoutingPolicy`) implements two-stage routing:

### Stage 1: Pre-Inference Routing

Uses only input-derived signals (no model inference required):
- **Input Length**: If character length exceeds threshold (default: 150), route directly to slow model
- **Word Count**: If word count exceeds threshold (default: 20), route directly to slow model
- **Lexical Diversity**: Type-token ratio (unique words / total words). If above threshold (default: 0.85), route directly to slow model. Higher diversity indicates varied vocabulary, potentially more complex.

This stage can skip the fast model entirely for clearly complex inputs, saving compute.

**Signal Selection Rationale**: The pre-inference signals (length, word count, lexical diversity) are intentionally simple, cheap proxies for input complexity. They require O(n) computation over the input text, no model inference, and no external services. These signals are configurable and replaceable: alternative signals (e.g., punctuation density, negation count, character n-gram entropy) can be substituted by modifying the preprocessing module and routing policy. The current signals were chosen for their computational efficiency and reasonable correlation with classification difficulty, but the system design does not depend on these specific choices.

### Stage 2: Post-Inference Routing

Uses fast model outputs (runs only if pre-inference routing didn't escalate):
- **Fast Model Confidence**: If maximum probability is below threshold (default: 0.6), escalate to slow model
- **Confidence Margin**: Difference between top two probabilities. If below threshold (default: 0.3), escalate to slow model
- **Entropy**: Shannon entropy of probability distribution. If above threshold (default: 0.8), escalate to slow model

All thresholds are configurable via `config.yaml`. The policy evaluates signals in order and routes to the slow model if any signal indicates difficulty.

### Latency Budget Constraint

The policy optionally supports a latency budget constraint. When enabled, the policy biases routing decisions toward the fast path when observed latency percentiles (e.g., rolling P95) exceed the configured budget. This constraint integrates with threshold-based logic: signals that would normally trigger escalation are suppressed if the latency budget is exceeded.

**Latency Budget Semantics**: During offline evaluation, the pipeline maintains a rolling history of latency measurements. After each inference, it computes latency percentiles (p50, p75, p90, p95, p99) from this history. When the latency budget is enabled, the policy checks whether the configured percentile (e.g., P95) exceeds the budget threshold. If it does, the policy biases subsequent routing decisions toward the fast path by suppressing escalation signals.

This design remains deterministic and reproducible because: (1) the latency history accumulates in a fixed order during evaluation replay, (2) percentile calculations are deterministic given the same history, and (3) the same input sequence produces the same latency history and thus the same budget checks. This approach is appropriate for evaluation and benchmarking where we want to understand how routing behavior changes under latency pressure, even though production systems may handle latency budgets differently (e.g., per-request budgets, distributed percentile tracking, or online adaptation).

### Policy Abstraction

The routing policy is implemented as an abstraction (`RoutingPolicy` base class), making it straightforward to add alternative routing strategies (e.g., cost-aware, latency-budget-based) without modifying the core router or inference pipeline.

### Decision Traceability

Every routing decision produces a structured decision trace that records:
- All evaluated signals (pre- and post-inference)
- Which signals fired (triggered escalation)
- Whether latency budget constraints influenced the decision
- Final routing outcome and stage

These traces are logged in structured form, exported during evaluation, and aggregated into summary statistics showing which factors most frequently influenced routing.

## Offline Evaluation Methodology

The evaluation module runs three inference strategies on a fixed test dataset:

1. **Always-Fast**: All inputs go through the fast model
2. **Always-Slow**: All inputs go through the slow model
3. **Adaptive**: Inputs are routed using two-stage routing policy

For each strategy, the system measures:
- **Global Metrics**: Accuracy, average latency, P95 latency, total cost, routing distribution
- **Conditional Metrics**: Accuracy conditioned on routing path (fast-only vs slow), escalation rates
- **Slice-Based Metrics**: Distribution by routing stage (pre-inference vs post-inference), accuracy and escalation rates per routing signal
- **Failure-Mode Metrics**: Analysis of cases where fast inference was accepted but incorrect, cases where escalation corrected an error, and cases where escalation did not improve correctness
- **Counterfactual Metrics**: Comparative analysis of adaptive routing decisions against alternative paths (fast-only vs slow-only), quantifying cost savings, unnecessary escalations, and correctness changes

All evaluations use fixed random seeds for reproducibility. The evaluation dataset is generated synthetically with a mix of easy (60%), hard ambiguous (30%), and very hard complex (10%) samples to test routing effectiveness and demonstrate clear performance differences between models.

## Results Summary

Based on typical runs with default configuration:

**Always-Fast Strategy:**
- Accuracy: ~85% (suffers on hard inputs)
- Average Latency: ~10ms
- Cost: Low (1.0 per inference)

**Always-Slow Strategy:**
- Accuracy: ~95% (excellent on all inputs)
- Average Latency: ~100ms
- Cost: High (10.0 per inference)

**Adaptive Routing Strategy:**
- Accuracy: ~90-92% (close to always-slow, much better than always-fast)
- Average Latency: ~30-50ms (much lower than always-slow)
- Cost: ~2-4 per inference (much lower than always-slow)
- Routing: ~60-70% fast, ~30-40% slow
- Stage Distribution: ~10-20% pre-inference, ~80-90% post-inference

The adaptive strategy achieves a favorable trade-off: it maintains accuracy closer to the slow model while keeping latency and cost much closer to the fast model. Conditional metrics demonstrate that escalation to the slow model is justified: slow path accuracy is higher than fast path accuracy, indicating the routing policy correctly identifies difficult inputs.

## Why Threshold-Based Routing (Not Learned Policies)

This system intentionally uses threshold-based, policy-driven routing rather than learned routing policies. This design choice prioritizes:

**Stability**: Threshold-based policies produce consistent, predictable behavior. Routing decisions are deterministic and reproducible, which is essential for debugging, auditing, and compliance in production systems.

**Debuggability**: When a routing decision is incorrect, engineers can trace exactly which signal fired and why. The decision trace shows all evaluated signals, which ones exceeded thresholds, and whether latency budget constraints influenced the outcome. This transparency is critical for diagnosing issues and understanding system behavior.

**Reproducibility**: Fixed thresholds with deterministic evaluation enable exact reproduction of routing behavior. This is essential for offline evaluation, A/B testing, and validating that changes to routing logic improve performance.

**Operational Simplicity**: Threshold-based policies are easy to understand, configure, and adjust. Engineers can reason about the system's behavior by examining threshold values and signal distributions, without needing to understand learned model weights or feature interactions.

**No Training Data Requirements**: Threshold-based policies don't require labeled routing decisions for training. They can be configured based on domain knowledge, offline analysis, or simple heuristics, making them easier to deploy and maintain.

Learned routing policies, while potentially more accurate, introduce non-determinism, require training data, complicate debugging, and make it harder to reason about system behavior. For systems where stability, debuggability, and reproducibility are paramount, threshold-based policies provide a better foundation.

## Design Trade-offs

The system makes several explicit design trade-offs to achieve its goals:

### Gained

**Deterministic Routing**: All routing decisions are reproducible with fixed seeds and configuration. This enables reliable offline evaluation, debugging, and auditing.

**Explicit Policy Control**: Threshold-based routing makes policy decisions transparent and configurable. Engineers can understand and adjust routing behavior without black-box models.

**Cheap Pre-Inference Signals**: Input-only signals (length, word count, lexical diversity) enable routing decisions before running any model, saving compute on clearly hard inputs.

**Structured Observability**: Decision traces record all signals, which fired, and why, providing complete visibility into routing behavior.

**Offline Evaluation Rigor**: Comprehensive metrics including conditional, slice-based, and failure-mode analysis enable thorough understanding of routing effectiveness.

### Sacrificed

**Adaptive Thresholds**: Thresholds are fixed and don't adapt to changing input distributions or performance feedback. This requires manual tuning and periodic re-evaluation.

**Learned Routing Policies**: The system doesn't use machine learning to learn optimal routing policies from data. This trades potential accuracy gains for stability and debuggability.

**Online Optimization**: No real-time threshold adjustment based on observed performance. Thresholds must be set based on offline analysis and remain fixed during operation.

**Context-Aware Routing**: Routing decisions don't consider request context (user type, time of day, etc.) or historical patterns. Each decision is made independently based on current input signals.

**Dynamic Cost Optimization**: Cost weights are fixed and don't adapt to changing compute costs or resource availability. The system assumes stable cost ratios between fast and slow models.

These trade-offs are intentional and align with the system's goals: demonstrating compute-aware adaptive inference with maximum stability, debuggability, and reproducibility.

## Production Extensions (Not Implemented)

While this system focuses on core routing logic, production deployments would extend it in several ways:

**Request Batching**: Batch multiple requests together to amortize model loading and improve throughput. The pipeline would collect requests over a time window or until a batch size threshold, then process them together.

**Async Inference**: Use asynchronous execution to overlap fast model inference with slow model inference preparation, or to handle multiple requests concurrently. The pipeline would use async/await patterns or message queues.

**Learned Routing Policies**: Implement a RoutingPolicy that uses a learned model (e.g., gradient boosting, neural network) to predict whether escalation will improve accuracy. This would require labeled training data (input features, fast/slow predictions, ground truth) and periodic retraining.

**Model Versioning**: Support routing between different versions of fast and slow models, enabling gradual rollouts and A/B testing. The router would select model versions based on configuration or request metadata.

**Distributed Execution**: Scale routing across multiple machines with load balancing. The pipeline would submit requests to a distributed queue, and routing decisions would be made by worker processes.

**Caching Layer**: Cache inference results for identical or similar inputs to avoid redundant computation. The pipeline would check a cache before running inference and store results for future requests.

**Real-Time Monitoring**: Integrate with monitoring systems to track routing metrics, latency percentiles, and accuracy in real-time. This would enable alerting when routing behavior deviates from expected patterns.

**Cost-Aware Routing**: Incorporate actual compute costs (GPU time, memory usage, energy consumption) rather than relative weights. The router would optimize for total cost while maintaining accuracy targets.

These extensions would be implemented as additional components or policy implementations, preserving the core routing architecture and evaluation methodology.

## Limitations and Extensions

### Current Limitations

1. **Simulated Models**: Uses simple logistic regression models. Real systems would use more sophisticated models (e.g., transformers, deep networks).

2. **Simulated Latency**: Latency is simulated using `time.sleep()`. Real systems would measure actual inference latency.

3. **Synthetic Data**: Evaluation uses synthetic text data. Real systems would use domain-specific datasets.

4. **Static Routing**: Routing thresholds are fixed. Real systems might adapt thresholds based on performance metrics.

5. **Binary Classification**: Currently supports binary classification only. Extension to multi-class is straightforward.

6. **Single Policy Implementation**: Only one concrete routing policy provided. Additional policies (cost-aware, latency-budget-based) can be added via the abstraction.

### Realistic Extensions

1. **Alternative Routing Policies**: Implement cost-aware, latency-budget-based, or learning-based policies using the RoutingPolicy abstraction.

2. **Multi-Model Ensemble**: Support routing to multiple fast models or a cascade of models with increasing accuracy.

3. **Cost-Aware Routing**: Incorporate actual compute costs (GPU time, memory) rather than relative weights.

4. **Latency Budgets**: Route based on remaining latency budget for the request.

5. **Model Versioning**: Support routing between different model versions for gradual rollouts.

6. **Input Caching**: Cache results for identical or similar inputs to avoid redundant inference.

7. **Distributed Inference**: Scale routing across multiple machines with load balancing.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` to adjust routing thresholds, model latencies, and evaluation settings. The configuration supports both pre-inference and post-inference routing thresholds.

### Running Evaluation

```bash
python main.py --mode evaluate
```

This will:
1. Generate a synthetic dataset with clear easy/hard distinction
2. Train fast and slow models
3. Run benchmarks for all three strategies
4. Save results to `results/benchmark_results.json` (includes conditional, slice-based, and failure-mode metrics)
5. Export routing decisions to `results/routing_decisions.json`
6. Export decision traces to `results/decision_traces.json`

### Running Single Inference

```bash
python main.py --mode inference --text "This product is amazing!"
```

This will:
1. Train models on a small dataset
2. Run inference on the provided text
3. Display routing decision, stage, and result

### Custom Configuration

```bash
python main.py --mode evaluate --config custom_config.yaml
```

## Project Structure

```
adaptive-inference-router/
├── README.md                 # This file
├── config.yaml              # Configuration file
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point
├── data/
│   └── sample_data.py        # Synthetic data generation
└── src/
    ├── __init__.py
    ├── router.py             # Routing policy abstraction and router
    ├── models/
    │   ├── fast_model.py     # Fast inference model
    │   └── slow_model.py     # Slow inference model
    ├── inference/
    │   └── pipeline.py      # Unified inference interface
    ├── evaluation/
    │   └── benchmark.py      # Offline evaluation with conditional metrics
    └── utils/
        ├── preprocess.py     # Input preprocessing
        └── logging.py        # Routing decision logging
```

## Design Principles

1. **Deterministic**: All routing decisions are reproducible with fixed seeds
2. **Configurable**: Behavior controlled via configuration files, not hard-coded constants
3. **Instrumented**: All routing decisions are logged with signals, stages, and outcomes
4. **Modular**: Clean separation of concerns between components
5. **Pluggable**: Routing policy abstraction enables alternative strategies
6. **Production-Inspired**: Demonstrates systems engineering principles without production infrastructure complexity
7. **Evaluable**: Comprehensive offline evaluation framework with conditional and slice-based metrics

## License

This is a demonstration system for educational and engineering purposes.
