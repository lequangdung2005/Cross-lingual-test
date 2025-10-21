#! /bin/bash
export HF_HOME=".cache"
export HF_DATASETS_CACHE=".cache"
export VLLM_CACHE_ROOT=".cache"
export WANDB_DISABLED="true"



# INSTRUCT_MODELS= ( "Qwen/Qwen2.5-Coder-0.5B-Instruct" "Qwen/Qwen2.5-Coder-1.5B-Instruct" "Qwen/Qwen2.5-Coder-3B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct" "deepseek-ai/deepseek-coder-1.3b-instruct" "deepseek-ai/deepseek-coder-6.7b-instruct" "codellama/CodeLlama-7b-Instruct-hf" "codellama/CodeLlama-13b-Instruct-hf" )
# BASE_MODELS=("Qwen/Qwen2.5-Coder-3B" "Qwen/Qwen2.5-Coder-7B" "deepseek-ai/deepseek-coder-1.3b-base" "deepseek-ai/deepseek-coder-6.7b-base" "bigcode/starcoder2-3b" "bigcode/starcoder2-15b") 
MODELS=( "Qwen/Qwen2.5-Coder-0.5B-Instruct" "deepseek-ai/deepseek-coder-6.7b-instruct" "codellama/CodeLlama-7b-Instruct-hf" )

for LANG in Rust Go Julia
do
    for MODEL in "${MODELS[@]}"; do
        IFS='/' read -a array <<< "$MODEL"
        MODEL_NAME=${array[1]}
        SAVE_DIR=./evaluation_rs/$LANG/test

        CUDA_VISIBLE_DEVICES=0,1 python3 generate.py \
            --model $MODEL \
            --instruct_model \
            --split ${LANG,,} \
            --lang $LANG \
            --task_name $LANG \
            --max_tokens 1024 \
            --batch_size 5 \
            --cache_dir ".cache" \
            --save_dir $SAVE_DIR \
            --num_return_sequences 1 \
            --repetition_penalty 1.2 \
            --top_p 1 \
            --top_k -1 \
            --temperature 0
    done
done