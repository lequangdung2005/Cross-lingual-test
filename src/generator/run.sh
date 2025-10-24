#! /bin/bash
export HF_HOME=".cache"
export HF_DATASETS_CACHE=".cache"
export VLLM_CACHE_ROOT=".cache"
export WANDB_DISABLED="true"

# Model configurations
# INSTRUCT_MODELS=("Qwen/Qwen2.5-Coder-0.5B-Instruct" "Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct" "deepseek-ai/deepseek-coder-1.3b-instruct" "deepseek-ai/deepseek-coder-6.7b-instruct" "codellama/CodeLlama-7b-Instruct-hf" "codellama/CodeLlama-13b-Instruct-hf")
# BASE_MODELS=("Qwen/Qwen2.5-Coder-3B" "Qwen/Qwen2.5-Coder-7B" "deepseek-ai/deepseek-coder-1.3b-base" "deepseek-ai/deepseek-coder-6.7b-base" "bigcode/starcoder2-3b" "bigcode/starcoder2-15b") 
MODELS=("Qwen/Qwen2.5-Coder-0.5B-Instruct") 
# Generation mode: standard, fewshot, or rag
MODE="fewshot"  # Using few-shot with retrieved examples

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
            --batch_size 5 \
            --cache_dir \".cache\" \
            --save_dir $SAVE_DIR \
            --num_return_sequences 1 \
            --repetition_penalty 1 \
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
        echo "✓ Completed: $MODEL_NAME on $LANG"
        echo ""
    done
done

echo "=================================================="
echo "All generations completed!"
echo "=================================================="