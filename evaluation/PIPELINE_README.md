# Evaluation Pipeline

This pipeline automates the process of:
1. Processing raw evaluation files from `evaluation_rs/` using `preprocess.py`
2. Running Docker-based evaluation on the processed files using `run_evaluation.sh`

## Files

- **`pipeline.sh`**: Main pipeline script that processes all files and runs evaluations
- **`preprocess.py`**: Preprocesses raw test generation outputs
- **`run_evaluation.sh`**: Runs Docker-based evaluation on processed files

## Directory Structure

```
evaluation/
├── pipeline.sh                    # Main pipeline script
├── preprocess.py                  # Preprocessing script
├── run_evaluation.sh              # Docker evaluation script
├── evaluation_rs/                 # Input: Raw evaluation files
│   ├── Go/
│   │   ├── fewshot/
│   │   │   └── [model-name]/
│   │   │       └── Go.final.generated.jsonl
│   │   └── standard/
│   │       └── [model-name]/
│   │           └── Go.final.generated.jsonl
│   ├── Rust/
│   └── Julia/
└── evaluation/data/processed_input/  # Output: Processed files
    ├── Go/
    ├── Rust/
    └── Julia/
```

## Usage

### Run Complete Pipeline

To process all files and run all evaluations:

```bash
./pipeline.sh
```

This will:
1. Find all `.jsonl` files in `evaluation_rs/`
2. Process each file using `preprocess.py`
3. Save processed files to `evaluation/data/processed_input/[lang]/[mode]/[model]/`
4. Run Docker evaluation on each processed file
5. Generate `summary.json` and `detailed_results.jsonl` in each output directory

### Run Individual Steps

#### Step 1: Preprocess a single file

```bash
python3 preprocess.py \
    --input_directory evaluation_rs/Go/fewshot/CodeLlama-13b-Instruct-hf/Go.final.generated.jsonl \
    --processed_input_directory evaluation/data/processed_input/Go/fewshot/CodeLlama-13b-Instruct-hf/processed_Go.jsonl \
    --lang go
```

#### Step 2: Run Docker evaluation on processed file

```bash
./run_evaluation.sh \
    evaluation/data/processed_input/Go/fewshot/CodeLlama-13b-Instruct-hf/processed_Go.jsonl \
    go
```

## Requirements

- Python 3.x with required packages:
  - tree-sitter
  - tree-sitter-rust
  - tree-sitter-go
  - tree-sitter-julia
  - datasets
  
- Docker with the following images:
  - `tessera2025testgen/tessera:go_environment`
  - `tessera2025testgen/tessera:rust_environment`
  - `tessera2025testgen/tessera:julia_environment`

## Output

For each processed file, the pipeline generates:
- **`summary.json`**: Summary statistics of the evaluation
- **`detailed_results.jsonl`**: Detailed results for each test case

These files are saved in the same directory as the processed input file.

## Troubleshooting

### Issue: "Permission denied" when running pipeline.sh

```bash
chmod +x pipeline.sh
```

### Issue: Docker image not found

Make sure Docker images are available:
```bash
docker pull tessera2025testgen/tessera:go_environment
docker pull tessera2025testgen/tessera:rust_environment
docker pull tessera2025testgen/tessera:julia_environment
```

### Issue: Python dependencies missing

Install required packages:
```bash
pip install tree-sitter tree-sitter-rust tree-sitter-go tree-sitter-julia datasets
```

## Notes

- The pipeline processes files in parallel-friendly structure but runs sequentially
- Each language requires its specific Docker image
- Empty directories in `evaluation_rs/` will be skipped automatically
- Progress is displayed with color-coded output for easy monitoring
