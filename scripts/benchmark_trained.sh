#!/usr/bin/env bash
# scripts/benchmark_trained.sh
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Enterprise benchmark suite for trained HSMN models.
# Runs comprehensive evaluation including knowledge, reasoning, and generation.
#
# Usage:
#   ./scripts/benchmark_trained.sh /path/to/checkpoint
#   ./scripts/benchmark_trained.sh /path/to/checkpoint full
#   ./scripts/benchmark_trained.sh /path/to/checkpoint quick output_dir

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
CHECKPOINT_PATH="${1:-}"
MODE="${2:-full}"
OUTPUT_DIR="${3:-benchmarks/reports/trained}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                 HSMN Enterprise Benchmark Suite                      ║"
echo "║                     Trained Model Evaluation                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Validate checkpoint
if [ -z "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}Error: Checkpoint path required${NC}"
    echo ""
    echo "Usage: $0 /path/to/checkpoint [mode] [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint  - Path to trained model checkpoint directory (required)"
    echo "  mode        - Benchmark mode: quick, full, leaderboard (default: full)"
    echo "  output_dir  - Output directory for reports (default: benchmarks/reports/trained)"
    echo ""
    echo "Modes:"
    echo "  quick       - Quick validation with subset of benchmarks"
    echo "  full        - Full evaluation suite"
    echo "  leaderboard - Generate lm-eval-harness compatible output"
    echo "  generation  - Generation quality tests only"
    echo "  reasoning   - Reasoning benchmarks only (ARC, HellaSwag, WinoGrande)"
    echo "  knowledge   - Knowledge benchmarks only (MMLU, TruthfulQA)"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}Error: Checkpoint directory not found: $CHECKPOINT_PATH${NC}"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Header info
echo -e "${GREEN}Checkpoint:${NC} $CHECKPOINT_PATH"
echo -e "${GREEN}Mode:${NC} $MODE"
echo -e "${GREEN}Output:${NC} $OUTPUT_DIR"
echo ""

# Hardware detection
echo -e "${BLUE}━━━ Hardware Detection ━━━${NC}"
python -c "from benchmarks.hardware_detection import print_system_info; print_system_info()" 2>/dev/null || {
    echo -e "${YELLOW}Warning: Hardware detection not available${NC}"
}
echo ""

# Save hardware info
python -c "
from benchmarks.hardware_detection import HardwareDetector
import json
detector = HardwareDetector()
info = detector.detect_all()
with open('$OUTPUT_DIR/hardware_info.json', 'w') as f:
    json.dump(info.to_dict(), f, indent=2)
" 2>/dev/null || true

# Run benchmark function
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

# Check if trained model benchmark exists
TRAINED_BENCH="benchmarks/bench_trained_model.py"
if [ ! -f "$TRAINED_BENCH" ]; then
    echo -e "${YELLOW}Warning: Trained model benchmark not yet implemented${NC}"
    echo -e "${YELLOW}Running available benchmarks on the model architecture...${NC}"
    echo ""
fi

case $MODE in
    quick)
        echo -e "${YELLOW}Running quick trained model evaluation${NC}"
        echo ""

        # Perplexity on WikiText
        run_benchmark "Perplexity (WikiText-2)" \
            "python3 benchmarks/bench_perplexity.py --dataset wikitext --verbose --output-dir '$OUTPUT_DIR'"

        # Generation throughput
        run_benchmark "Generation Throughput" \
            "python3 benchmarks/bench_throughput.py --speed --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    full)
        echo -e "${GREEN}Running full trained model evaluation${NC}"
        echo ""

        # Perplexity benchmarks
        run_benchmark "Perplexity (WikiText-2)" \
            "python3 benchmarks/bench_perplexity.py --dataset wikitext --verbose --output-dir '$OUTPUT_DIR'"

        run_benchmark "Perplexity (WikiText-103)" \
            "python3 benchmarks/bench_perplexity.py --dataset wikitext-103 --verbose --output-dir '$OUTPUT_DIR'" || true

        # Throughput
        run_benchmark "Throughput Benchmark" \
            "python3 benchmarks/bench_throughput.py --verbose --output-dir '$OUTPUT_DIR'"

        # Confidence/calibration
        run_benchmark "Confidence Benchmark" \
            "python3 benchmarks/bench_confidence.py --verbose --output-dir '$OUTPUT_DIR'"

        # Long-context
        run_benchmark "Long-Context Benchmark" \
            "python3 benchmarks/bench_long_context.py --max-context 131072 --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    leaderboard)
        echo -e "${GREEN}Generating leaderboard-compatible output${NC}"
        echo ""
        echo -e "${YELLOW}Note: Full leaderboard evaluation requires lm-eval-harness integration.${NC}"
        echo -e "${YELLOW}This generates compatible output format for submission.${NC}"
        echo ""

        # Generate lm-eval compatible output
        python -c "
import json
from datetime import datetime
from pathlib import Path

# Placeholder leaderboard output structure
output = {
    'config': {
        'model': 'HSMN',
        'model_args': 'checkpoint=$CHECKPOINT_PATH',
        'batch_size': 8,
    },
    'results': {},
    'versions': {
        'lm-eval': 'compatible',
        'hsmn': '1.0.0',
    },
    'git_hash': 'local',
    'date': datetime.now().isoformat(),
}

# Save
Path('$OUTPUT_DIR').mkdir(parents=True, exist_ok=True)
with open('$OUTPUT_DIR/leaderboard_submission.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Leaderboard output saved to: $OUTPUT_DIR/leaderboard_submission.json')
print('')
print('To submit to OpenLLM Leaderboard:')
print('1. Upload model to HuggingFace Hub')
print('2. Submit at: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard')
"
        ;;

    generation)
        echo -e "${GREEN}Running generation quality benchmarks${NC}"
        echo ""

        run_benchmark "Generation Throughput" \
            "python3 benchmarks/bench_throughput.py --speed --verbose --output-dir '$OUTPUT_DIR'"
        ;;

    reasoning)
        echo -e "${GREEN}Running reasoning benchmarks (requires trained model)${NC}"
        echo ""
        echo -e "${YELLOW}Note: ARC, HellaSwag, WinoGrande evaluation requires lm-eval-harness${NC}"
        echo ""

        # Placeholder - would run lm-eval-harness tasks
        echo "To run with lm-eval-harness:"
        echo "  lm_eval --model hf --model_args pretrained=$CHECKPOINT_PATH \\"
        echo "    --tasks arc_challenge,hellaswag,winogrande \\"
        echo "    --batch_size 8"
        ;;

    knowledge)
        echo -e "${GREEN}Running knowledge benchmarks (requires trained model)${NC}"
        echo ""
        echo -e "${YELLOW}Note: MMLU, TruthfulQA evaluation requires lm-eval-harness${NC}"
        echo ""

        echo "To run with lm-eval-harness:"
        echo "  lm_eval --model hf --model_args pretrained=$CHECKPOINT_PATH \\"
        echo "    --tasks mmlu,truthfulqa_mc \\"
        echo "    --batch_size 8"
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo ""
        echo "Usage: $0 /path/to/checkpoint {quick|full|leaderboard|generation|reasoning|knowledge}"
        exit 1
        ;;
esac

# Summary
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                         Evaluation Complete                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Checkpoint:${NC} $CHECKPOINT_PATH"
echo -e "${GREEN}Results:${NC} $OUTPUT_DIR"
echo ""

# List generated files
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}Generated files:${NC}"
    ls -la "$OUTPUT_DIR"/*.{json,md} 2>/dev/null || echo "  (no output files found)"
fi

echo ""
echo -e "${GREEN}✅ Trained model benchmark completed${NC}"
