# OpenAI GPT-4o Test Generation Commands

This file contains example commands for running GPT-4o test generation.

## Prerequisites

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Individual Commands

### 1. Standard Mode (No Context)

**Rust:**
```bash
python generate_openai.py \
    --model gpt-4o \
    --lang Rust \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Rust/standard/gpt-4o \
    --max_tokens 1024 \
    --num_return_sequences 5 \
    --temperature 0.2 \
    --top_p 0.95 \
    --request_delay 1.0 \
    --instruct_model
```

**Go:**
```bash
python generate_openai.py \
    --model gpt-4o \
    --lang Go \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Go/standard/gpt-4o \
    --max_tokens 1024 \
    --num_return_sequences 5 \
    --temperature 0.2 \
    --top_p 0.95 \
    --request_delay 1.0 \
    --instruct_model
```

**Julia:**
```bash
python generate_openai.py \
    --model gpt-4o \
    --lang Julia \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Julia/standard/gpt-4o \
    --max_tokens 1024 \
    --num_return_sequences 5 \
    --temperature 0.2 \
    --top_p 0.95 \
    --request_delay 1.0 \
    --instruct_model
```

### 2. Arg Context Mode

**Rust:**
```bash
python generate_openai.py \
    --model gpt-4o \
    --lang Rust \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Rust/arg_context/gpt-4o \
    --max_tokens 1024 \
    --num_return_sequences 5 \
    --temperature 0.2 \
    --top_p 0.95 \
    --request_delay 1.0 \
    --instruct_model \
    --arg_context
```

**Go:**
```bash
python generate_openai.py \
    --model gpt-4o \
    --lang Go \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Go/arg_context/gpt-4o \
    --max_tokens 1024 \
    --num_return_sequences 5 \
    --temperature 0.2 \
    --top_p 0.95 \
    --request_delay 1.0 \
    --instruct_model \
    --arg_context
```

**Julia:**
```bash
python generate_openai.py \
    --model gpt-4o \
    --lang Julia \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Julia/arg_context/gpt-4o \
    --max_tokens 1024 \
    --num_return_sequences 5 \
    --temperature 0.2 \
    --top_p 0.95 \
    --request_delay 1.0 \
    --instruct_model \
    --arg_context
```

### 3. Standard + Arg Context Mode (Combined)

**Rust:**
```bash
python generate_openai.py \
    --model gpt-4o \
    --lang Rust \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Rust/standard_arg_context/gpt-4o \
    --max_tokens 1024 \
    --num_return_sequences 5 \
    --temperature 0.2 \
    --top_p 0.95 \
    --request_delay 1.0 \
    --instruct_model \
    --arg_context
```

**Go:**
```bash
python generate_openai.py \
    --model gpt-4o \
    --lang Go \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Go/standard_arg_context/gpt-4o \
    --max_tokens 1024 \
    --num_return_sequences 5 \
    --temperature 0.2 \
    --top_p 0.95 \
    --request_delay 1.0 \
    --instruct_model \
    --arg_context
```

**Julia:**
```bash
python generate_openai.py \
    --model gpt-4o \
    --lang Julia \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Julia/standard_arg_context/gpt-4o \
    --max_tokens 1024 \
    --num_return_sequences 5 \
    --temperature 0.2 \
    --top_p 0.95 \
    --request_delay 1.0 \
    --instruct_model \
    --arg_context
```

## Using the Automated Script

Instead of running individual commands, use the automated script:

```bash
cd src/generator
chmod +x run_openai.sh
./run_openai.sh
```

This will run all configurations for all languages automatically.

## Parameter Explanations

- `--model gpt-4o`: Use GPT-4o model
- `--lang`: Programming language (Rust/Go/Julia)
- `--split test`: Use test split of the dataset
- `--save_dir`: Output directory for generated tests
- `--max_tokens 1024`: Maximum tokens per generation
- `--num_return_sequences 5`: Generate 5 test cases per function
- `--temperature 0.2`: Low temperature for more deterministic output
- `--top_p 0.95`: Nucleus sampling threshold
- `--request_delay 1.0`: Wait 1 second between API calls (avoid rate limits)
- `--instruct_model`: Use instruction-based prompt template
- `--arg_context`: Include argument type definitions and context

## Cost Estimation

**GPT-4o Pricing (as of Nov 2025):**
- Input: $2.50 per 1M tokens
- Output: $10.00 per 1M tokens

**Approximate costs per run:**
- Each function: ~500 input tokens + 5 × 1024 output tokens = ~5,620 tokens
- Full test set (~300 functions): ~1.7M tokens total
- Estimated cost: ~$10-15 per language per mode

**Total for all modes:** ~$90-135 for 3 languages × 3 modes

## Alternative: Use GPT-3.5-turbo for Cost Savings

Replace `gpt-4o` with `gpt-3.5-turbo` in the commands:

```bash
python generate_openai.py \
    --model gpt-3.5-turbo \
    --lang Rust \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Rust/standard/gpt-3.5-turbo \
    --instruct_model
```

GPT-3.5-turbo is ~10x cheaper but may have lower quality outputs.

## Custom API Endpoint

If using Azure OpenAI or compatible API:

```bash
python generate_openai.py \
    --model gpt-4o \
    --api_key "your-key" \
    --base_url "https://your-endpoint.openai.azure.com/" \
    --lang Rust \
    --split test \
    --save_dir ../../evaluation/evaluation_rs/Rust/standard/gpt-4o \
    --instruct_model
```
