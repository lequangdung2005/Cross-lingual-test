#!/bin/bash

# OpenAI GPT-4o Test Generation Script
# This script runs test generation with GPT-4o API in multiple configurations

set -e  # Exit on error

# Configuration
MODEL="gpt-4o"
SPLIT="test"
MAX_TOKENS=1024
NUM_SEQUENCES=5
TEMPERATURE=0.2
TOP_P=0.95
REQUEST_DELAY=1.0  # Delay between API requests (seconds)

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Languages to process
LANGUAGES=("Rust" "Go" "Julia")

# Modes: standard, arg_context, standard+arg_context
MODES=("standard" "arg_context" "standard_arg_context")

echo "========================================"
echo "OpenAI GPT-4o Test Generation"
echo "========================================"
echo "Model: $MODEL"
echo "Languages: ${LANGUAGES[@]}"
echo "Modes: ${MODES[@]}"
echo "========================================"
echo ""

# Function to run generation
run_generation() {
    local lang=$1
    local mode=$2
    local output_dir=$3
    local extra_args=$4
    
    echo "----------------------------------------"
    echo "Running: $lang - $mode"
    echo "Output: $output_dir"
    echo "----------------------------------------"
    
    python generate_openai.py \
        --model "$MODEL" \
        --lang "$lang" \
        --split "$SPLIT" \
        --save_dir "$output_dir" \
        --max_tokens $MAX_TOKENS \
        --num_return_sequences $NUM_SEQUENCES \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --request_delay $REQUEST_DELAY \
        --instruct_model \
        $extra_args
    
    echo "✓ Completed: $lang - $mode"
    echo ""
}

# Main execution loop
for LANG in "${LANGUAGES[@]}"; do
    echo "========================================"
    echo "Processing Language: $LANG"
    echo "========================================"
    echo ""
    
    # 1. Standard mode
    MODE="standard"
    OUTPUT_DIR="../../evaluation/evaluation_rs/${LANG}/${MODE}/${MODEL}"
    run_generation "$LANG" "$MODE" "$OUTPUT_DIR" ""
    
    # 2. Arg context mode
    MODE="arg_context"
    OUTPUT_DIR="../../evaluation/evaluation_rs/${LANG}/${MODE}/${MODEL}"
    run_generation "$LANG" "$MODE" "$OUTPUT_DIR" "--arg_context"
    
    # 3. Standard + Arg context mode
    MODE="standard_arg_context"
    OUTPUT_DIR="../../evaluation/evaluation_rs/${LANG}/${MODE}/${MODEL}"
    run_generation "$LANG" "$MODE" "$OUTPUT_DIR" "--arg_context"
    
    echo "✓ Completed all modes for $LANG"
    echo ""
done

echo "========================================"
echo "✓ All generations completed!"
echo "========================================"
echo ""
echo "Results saved in: ../../evaluation/evaluation_rs/"
echo ""
echo "Next steps:"
echo "1. Run evaluation on the generated tests"
echo "2. Compare results with: python ../../compare_results.py"
