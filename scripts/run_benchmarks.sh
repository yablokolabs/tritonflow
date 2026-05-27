#!/usr/bin/env bash
set -euo pipefail

echo "=== TritonFlow Benchmark Suite ==="

SUITE="${1:-all}"
OUTPUT_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "Suite: $SUITE"
echo "Output: $OUTPUT_DIR"
echo ""

tritonflow benchmark run \
    --suite "$SUITE" \
    --export-md "$OUTPUT_DIR/benchmark_${TIMESTAMP}.md" \
    --export-csv "$OUTPUT_DIR/benchmark_${TIMESTAMP}.csv"

echo ""
echo "Generating charts..."
tritonflow visualize charts \
    "$OUTPUT_DIR/benchmark_${TIMESTAMP}.csv" \
    --output-dir "$OUTPUT_DIR/charts"

echo ""
echo "=== Complete ==="
echo "Report: $OUTPUT_DIR/benchmark_${TIMESTAMP}.md"
echo "Data:   $OUTPUT_DIR/benchmark_${TIMESTAMP}.csv"
echo "Charts: $OUTPUT_DIR/charts/"
