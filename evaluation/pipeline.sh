#!/bin/bash

# ====== Configuration ======
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_RS_DIR="$PROJECT_ROOT/evaluation_rs"

# ====== Colors for output ======
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Evaluation Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"

# ====== Step 1: Process files with preprocess.py ======
echo -e "\n${GREEN}[Step 1] Processing files with preprocess.py${NC}\n"

# Counter for processed files
TOTAL_FILES=0
PROCESSED_FILES=0
FAILED_FILES=0
SKIPPED_FILES=0

# Find all .jsonl files in evaluation_rs (excluding already processed ones)
echo "Searching for files to process..."
while IFS= read -r input_file; do
    ((TOTAL_FILES++))
    
    # Extract language and path information
    # Example path: evaluation_rs/Go/fewshot/CodeLlama-13b-Instruct-hf/Go.final.generated.jsonl
    relative_path="${input_file#$EVALUATION_RS_DIR/}"
    
    # Extract language (first directory)
    lang=$(echo "$relative_path" | cut -d'/' -f1)
    
    # Extract model directory (same directory as input file)
    input_dir=$(dirname "$input_file")
    
    # Output file in the same directory as input
    file_basename=$(basename "$input_file" .jsonl)
    output_file="$input_dir/processed_${file_basename}.jsonl"
    
    # Check if processed file already exists
    if [ -f "$output_file" ]; then
        ((SKIPPED_FILES++))
        echo -e "${BLUE}Skipping (already processed):${NC} $relative_path"
        echo ""
        continue
    fi
    
    echo -e "${YELLOW}Processing:${NC} $relative_path"
    echo -e "  Language: $lang"
    echo -e "  Output: ${output_file#$PROJECT_ROOT/}"
    
    # Run preprocess.py
    if python3 "$PROJECT_ROOT/preprocess.py" \
        --input_directory "$input_file" \
        --processed_input_directory "$output_file" \
        --lang "$lang" 2>&1; then
        ((PROCESSED_FILES++))
        echo -e "${GREEN}  ✓ Processed successfully${NC}"
    else
        ((FAILED_FILES++))
        echo -e "\033[0;31m  ✗ Failed to process${NC}"
    fi
    echo ""
    
done < <(find "$EVALUATION_RS_DIR" -name "*.jsonl" -type f ! -name "processed_*")

echo "Found $TOTAL_FILES files total"

echo -e "${GREEN}[Step 1 Complete] Processed: $PROCESSED_FILES | Skipped: $SKIPPED_FILES | Failed: $FAILED_FILES | Total: $TOTAL_FILES${NC}\n"

# ====== Step 2: Run Docker evaluation ======
echo -e "\n${GREEN}[Step 2] Running Docker evaluation${NC}\n"

# Counter for Docker runs
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
SKIPPED_RUNS=0

echo "Searching for processed files..."
# Find all processed files and run Docker evaluation
while IFS= read -r processed_file; do
    ((TOTAL_RUNS++))
    
    # Extract language from path
    relative_path="${processed_file#$EVALUATION_RS_DIR/}"
    lang=$(echo "$relative_path" | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')
    
    # Check if summary.json already exists in the same directory
    processed_dir=$(dirname "$processed_file")
    summary_file="$processed_dir/summary.json"
    
    if [ -f "$summary_file" ]; then
        ((SKIPPED_RUNS++))
        echo -e "${BLUE}Skipping (already evaluated):${NC} $relative_path"
        echo ""
        continue
    fi
    
    echo -e "${YELLOW}Running Docker evaluation:${NC} $relative_path"
    echo -e "  Language: $lang"
    
    # Get the relative path to the processed file from project root
    relative_file_path="${processed_file#$PROJECT_ROOT/}"
    
    # Run Docker evaluation
    if bash "$PROJECT_ROOT/run_evaluation.sh" "$relative_file_path" "$lang" 2>&1; then
        ((SUCCESSFUL_RUNS++))
        echo -e "${GREEN}  ✓ Docker evaluation completed${NC}"
    else
        ((FAILED_RUNS++))
        echo -e "\033[0;31m  ✗ Docker evaluation failed${NC}"
    fi
    echo ""
    
done < <(find "$EVALUATION_RS_DIR" -name "processed_*.jsonl" -type f)

echo "Found $TOTAL_RUNS processed files total"

echo -e "${GREEN}[Step 2 Complete] Completed: $SUCCESSFUL_RUNS | Skipped: $SKIPPED_RUNS | Failed: $FAILED_RUNS | Total: $TOTAL_RUNS${NC}\n"

# ====== Summary ======
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Pipeline Complete${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Step 1 - Preprocessing:"
echo -e "  Processed: $PROCESSED_FILES | Skipped: $SKIPPED_FILES | Failed: $FAILED_FILES | Total: $TOTAL_FILES"
echo -e ""
echo -e "Step 2 - Docker Evaluation:"
echo -e "  Completed: $SUCCESSFUL_RUNS | Skipped: $SKIPPED_RUNS | Failed: $FAILED_RUNS | Total: $TOTAL_RUNS"
echo -e "${BLUE}============================================${NC}"
