#! /bin/bash

# Set cache directories (use absolute paths)
export HF_HOME="$(pwd)/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$(pwd)/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$(pwd)/.cache/huggingface/datasets"
export VLLM_CACHE_ROOT="$(pwd)/.cache/vllm"
export WANDB_DISABLED="true"

# Create cache directories if they don't exist
mkdir -p "$HF_HOME"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$VLLM_CACHE_ROOT"

echo "Cache directories:"
echo "  HF_HOME: $HF_HOME"
echo "  VLLM_CACHE_ROOT: $VLLM_CACHE_ROOT"
echo ""

# Model configurations
# INSTRUCT_MODELS=("Qwen/Qwen2.5-Coder-0.5B-Instruct" "Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct" "deepseek-ai/deepseek-coder-1.3b-instruct" "deepseek-ai/deepseek-coder-6.7b-instruct" "codellama/CodeLlama-7b-Instruct-hf" "codellama/CodeLlama-13b-Instruct-hf")
# BASE_MODELS=("Qwen/Qwen2.5-Coder-3B" "Qwen/Qwen2.5-Coder-7B" "deepseek-ai/deepseek-coder-1.3b-base" "deepseek-ai/deepseek-coder-6.7b-base" "bigcode/starcoder2-3b" "bigcode/starcoder2-15b") 
MODELS=("Qwen/Qwen2.5-Coder-0.5B-Instruct" "Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct" "deepseek-ai/deepseek-coder-1.3b-instruct" "deepseek-ai/deepseek-coder-6.7b-instruct" "codellama/CodeLlama-7b-Instruct-hf" "codellama/CodeLlama-13b-Instruct-hf")

# Generation modes to run
MODES=("standard" "fewshot")  # Run both standard and few-shot modes

for MODE in "${MODES[@]}"
do
    echo "=========================================="
    echo "Starting $MODE mode generation"
    echo "=========================================="
    echo ""
    
    for LANG in Rust Go Julia
    do
        for MODEL in "${MODELS[@]}"; do
            IFS='/' read -a array <<< "$MODEL"
            MODEL_NAME=${array[1]}
            
            # Output directory based on mode
            if [ "$MODE" = "fewshot" ]; then
                SAVE_DIR=./evaluation_rs/$LANG/fewshot/$MODEL_NAME
            elif [ "$MODE" = "rag" ]; then
                SAVE_DIR=./evaluation_rs/$LANG/rag/$MODEL_NAME
            else
                SAVE_DIR=./evaluation_rs/$LANG/standard/$MODEL_NAME
            fi
            
            echo "=================================================="
            echo "Running: $MODEL_NAME on $LANG ($MODE mode)"
            echo "Output: $SAVE_DIR"
            echo "=================================================="
            
            # Build command based on mode
            CMD="CUDA_VISIBLE_DEVICES=0,1 python3 generator/generate.py \
                --model $MODEL \
                --instruct_model \
                --split train \
                --lang $LANG \
                --task_name $LANG \
                --max_tokens 1024 \
                --batch_size 4 \
                --cache_dir \"$HF_HOME\" \
                --save_dir $SAVE_DIR \
                --num_return_sequences 1 \
                --repetition_penalty 1.2 \
                --top_p 1 \
                --top_k -1 \
                --temperature 0"
            
            # Add mode-specific flags
            if [ "$MODE" = "fewshot" ]; then
                CMD="$CMD --use_fewshot_jsonl"
            elif [ "$MODE" = "rag" ]; then
                CMD="$CMD --rag_type dense"  # Change to "bm25" if needed
            fi
            
            # Execute command
            eval $CMD
            
            echo ""
            echo "✓ Completed: $MODEL_NAME on $LANG ($MODE mode)"
            echo ""
        done
    done
    
    echo "=========================================="
    echo "Completed all languages for $MODE mode"
    echo "=========================================="
    echo ""
done

echo "=================================================="
echo "All generations completed!"
echo "=================================================="