#!/usr/bin/env bash
# scripts/benchmark_untrained.sh
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Enterprise benchmark suite for untrained HSMN models.
# Tests architecture properties, scaling, and long-context capabilities.
#
# Usage:
#   ./scripts/benchmark_untrained.sh              # Full benchmark
#   ./scripts/benchmark_untrained.sh quick        # Quick validation
#   ./scripts/benchmark_untrained.sh ultra        # 1M token context
#   ./scripts/benchmark_untrained.sh throughput   # Throughput only
#   ./scripts/benchmark_untrained.sh long-context # Long-context only

set -euo pipefail

# Script directory for imports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
MODE="${1:-full}"
OUTPUT_DIR="${2:-benchmarks/reports}"
DATASET="${3:-wikitext}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║            HSMN Enterprise Benchmark Suite (Production Mode)        ║"
echo "║                     Untrained Model Evaluation                       ║"
echo "║           All benchmarks use full production architecture            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if available
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo -e "${GREEN}Using venv:${NC} $PROJECT_ROOT/venv"
fi

# Set PYTHONPATH to project root for module imports
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"



# Header info
echo -e "${GREEN}Mode:${NC} $MODE"
echo -e "${GREEN}Output:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Dataset:${NC} $DATASET"
echo ""

# Hardware detection (direct import to avoid TensorFlow load)
echo -e "${BLUE}━━━ Hardware Detection ━━━${NC}"
HIGHNOON_SKIP_TF=1 python3 benchmarks/hardware_detection.py --format text 2>/dev/null || {
    echo -e "${YELLOW}Warning: Hardware detection not available. Continuing...${NC}"
}
echo ""

HIGHNOON_SKIP_TF=1 python3 benchmarks/hardware_detection.py --format json -o "$OUTPUT_DIR/hardware_info.json" 2>/dev/null || true
HIGHNOON_SKIP_TF=1 python3 benchmarks/hardware_detection.py --format markdown -o "$OUTPUT_DIR/hardware_info.md" 2>/dev/null || true

# Architecture verification (ensures all production features are active)
echo -e "${BLUE}━━━ Architecture Verification ━━━${NC}"
python3 benchmarks/verify_architecture.py --verbose 2>/dev/null || {
    echo -e "${YELLOW}Warning: Some architecture components not available. Using fallbacks.${NC}"
}
echo ""




# Run benchmarks based on mode
run_benchmark() {
    local name="$1"
    local cmd="$2"

    echo -e "${BLUE}━━━ $name ━━━${NC}"
    if eval "$cmd"; then
        echo -e "${GREEN}✅ $name complete${NC}"
    else
        echo -e "${RED}❌ $name failed${NC}"
        return 1
    fi
    echo ""
}

case $MODE in
    quick)
        echo -e "${YELLOW}Running quick validation (smaller model, fewer iterations)${NC}"
        echo ""
        run_benchmark "Throughput Benchmark" \
            "python3 benchmarks/bench_throughput.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Memory Benchmark" \
            "python3 benchmarks/bench_memory.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Perplexity Benchmark" \
            "python3 benchmarks/bench_perplexity.py --quick --dataset $DATASET --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Quantum Features" \
            "python3 benchmarks/bench_quantum.py --output '$OUTPUT_DIR/quantum_results.json'"
        ;;

    full)
        echo -e "${GREEN}Running full benchmark suite${NC}"
        echo ""
        run_benchmark "Throughput Benchmark" \
            "python3 benchmarks/bench_throughput.py --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Memory Benchmark" \
            "python3 benchmarks/bench_memory.py --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Perplexity Benchmark" \
            "python3 benchmarks/bench_perplexity.py --dataset $DATASET --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Confidence Benchmark" \
            "python3 benchmarks/bench_confidence.py --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Architecture Comparison" \
            "python3 benchmarks/bench_comparison.py --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Quantum Features" \
            "python3 benchmarks/bench_quantum.py --output '$OUTPUT_DIR/quantum_results.json'"
        run_benchmark "Long-Context (up to 128K)" \
            "python3 benchmarks/bench_long_context.py --max-context 131072 --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    ultra)
        echo -e "${GREEN}Running ultra benchmark suite (1M token context)${NC}"
        echo ""
        run_benchmark "Long-Context Benchmark (up to 1M)" \
            "python3 benchmarks/bench_long_context.py --max-context 1048576 --preset 100m-benchmark --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Throughput Benchmark" \
            "python3 benchmarks/bench_throughput.py --verbose --output-dir '$OUTPUT_DIR'"
        run_benchmark "Memory Benchmark" \
            "python3 benchmarks/bench_memory.py --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    throughput)
        echo -e "${GREEN}Running throughput benchmark only${NC}"
        echo ""
        run_benchmark "Throughput Benchmark" \
            "python3 benchmarks/bench_throughput.py --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    memory)
        echo -e "${GREEN}Running memory benchmark only${NC}"
        echo ""
        run_benchmark "Memory Benchmark" \
            "python3 benchmarks/bench_memory.py --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    perplexity)
        echo -e "${GREEN}Running perplexity benchmark only${NC}"
        echo ""
        run_benchmark "Perplexity Benchmark" \
            "python3 benchmarks/bench_perplexity.py --dataset $DATASET --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    long-context)
        echo -e "${GREEN}Running long-context benchmark only (100M model, 1M tokens)${NC}"
        echo ""
        run_benchmark "Long-Context Benchmark" \
            "python3 benchmarks/bench_long_context.py --max-context 1048576 --preset 100m-benchmark --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    quantum)
        echo -e "${GREEN}Running quantum features benchmark only${NC}"
        echo ""
        run_benchmark "Quantum Features" \
            "python3 benchmarks/bench_quantum.py --output '$OUTPUT_DIR/quantum_results.json'"
        ;;

    comparison)
        echo -e "${GREEN}Running architecture comparison only${NC}"
        echo ""
        run_benchmark "Architecture Comparison" \
            "python3 benchmarks/bench_comparison.py --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    leaderboard)
        echo -e "${GREEN}Running Open LLM Leaderboard benchmarks (untrained baseline)${NC}"
        echo ""
        run_benchmark "Leaderboard Evaluation" \
            "python3 benchmarks/bench_leaderboard.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    mmlu)
        echo -e "${GREEN}Running MMLU benchmark only${NC}"
        echo ""
        run_benchmark "MMLU Benchmark" \
            "python3 benchmarks/bench_mmlu.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    hellaswag)
        echo -e "${GREEN}Running HellaSwag benchmark only${NC}"
        echo ""
        run_benchmark "HellaSwag Benchmark" \
            "python3 benchmarks/bench_hellaswag.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    arc)
        echo -e "${GREEN}Running ARC benchmark only${NC}"
        echo ""
        run_benchmark "ARC Benchmark" \
            "python3 benchmarks/bench_arc.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    gsm8k)
        echo -e "${GREEN}Running GSM8K math benchmark only${NC}"
        echo ""
        run_benchmark "GSM8K Benchmark" \
            "python3 benchmarks/bench_gsm8k.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    winogrande)
        echo -e "${GREEN}Running WinoGrande benchmark only${NC}"
        echo ""
        run_benchmark "WinoGrande Benchmark" \
            "python3 benchmarks/bench_winogrande.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    truthfulqa)
        echo -e "${GREEN}Running TruthfulQA benchmark only${NC}"
        echo ""
        run_benchmark "TruthfulQA Benchmark" \
            "python3 benchmarks/bench_truthfulqa.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    humaneval)
        echo -e "${GREEN}Running HumanEval code benchmark only${NC}"
        echo ""
        run_benchmark "HumanEval Benchmark" \
            "python3 benchmarks/bench_humaneval.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    gradient-flow)
        echo -e "${GREEN}Running gradient flow analysis${NC}"
        echo ""
        run_benchmark "Gradient Flow Analysis" \
            "python3 benchmarks/bench_gradient_flow.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    activations)
        echo -e "${GREEN}Running activation statistics analysis${NC}"
        echo ""
        run_benchmark "Activation Statistics" \
            "python3 benchmarks/bench_activation_stats.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    architecture)
        echo -e "${GREEN}Running architecture analysis${NC}"
        echo ""
        run_benchmark "Architecture Analysis" \
            "python3 benchmarks/bench_architecture.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    stability)
        echo -e "${GREEN}Running numerical stability analysis${NC}"
        echo ""
        run_benchmark "Numerical Stability" \
            "python3 benchmarks/bench_numerical_stability.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    scaling)
        echo -e "${GREEN}Running scaling analysis${NC}"
        echo ""
        run_benchmark "Scaling Analysis" \
            "python3 benchmarks/bench_scaling.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    needle-haystack)
        echo -e "${GREEN}Running needle-in-haystack retrieval tests${NC}"
        echo ""
        run_benchmark "Needle-in-Haystack" \
            "python3 benchmarks/bench_needle_haystack.py --quick --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    all)
        echo -e "${GREEN}Running complete benchmark suite${NC}"
        echo ""
        python3 -m benchmarks --dataset "$DATASET" --output-dir "$OUTPUT_DIR" --verbose
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo ""
        echo "Usage: $0 {quick|full|ultra|throughput|memory|perplexity|long-context|quantum|comparison|...}"
        echo ""
        echo "Architecture & Performance:"
        echo "  quick          - Quick validation (smaller model)"
        echo "  full           - Full benchmark suite (default)"
        echo "  ultra          - Ultra benchmark with 1M token context"
        echo "  throughput     - Throughput benchmark only"
        echo "  memory         - Memory benchmark only"
        echo "  perplexity     - Perplexity benchmark only"
        echo "  long-context   - Long-context benchmark (up to 1M tokens)"
        echo "  quantum        - Quantum features benchmark"
        echo "  comparison     - Architecture comparison"
        echo ""
        echo "Architecture Analysis:"
        echo "  gradient-flow  - Gradient flow analysis (vanishing/exploding)"
        echo "  activations    - Activation statistics per layer"
        echo "  architecture   - Parameter/FLOPs/memory breakdown"
        echo "  stability      - Numerical stability (NaN/Inf/Lipschitz)"
        echo "  scaling        - Batch/sequence scaling efficiency"
        echo "  needle-haystack - Long-context retrieval tests"
        echo ""
        echo "SOTA LLM Benchmarks:"
        echo "  leaderboard    - All Open LLM Leaderboard benchmarks"
        echo "  mmlu           - MMLU knowledge benchmark"
        echo "  hellaswag      - HellaSwag commonsense reasoning"
        echo "  arc            - ARC science reasoning"
        echo "  gsm8k          - GSM8K math reasoning"
        echo "  winogrande     - WinoGrande coreference resolution"
        echo "  truthfulqa     - TruthfulQA factual accuracy"
        echo "  humaneval      - HumanEval code generation"
        echo ""
        echo "  all            - Complete unified benchmark"
        exit 1
        ;;
esac

# Generate unified report if running full/ultra/all
if [[ "$MODE" == "full" || "$MODE" == "ultra" || "$MODE" == "all" ]]; then
    echo -e "${BLUE}━━━ Generating Unified Report ━━━${NC}"
    python3 benchmarks/generate_report.py --output-dir "$OUTPUT_DIR" --dataset "$DATASET" --verbose || true
    echo ""
fi

# Summary
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                         Benchmark Complete                           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Results saved to:${NC} $OUTPUT_DIR"
echo ""

# List generated files
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}Generated files:${NC}"
    ls -la "$OUTPUT_DIR"/*.{json,md} 2>/dev/null || echo "  (no output files found)"
fi

echo ""
echo -e "${GREEN}✅ Benchmark suite completed successfully${NC}"
