

set -e  # Exit immediately if any command fails

PROJECT_ROOT="$(pwd)"
# ====== Parse arguments ======
INPUT_DIR=$1     # Directory of processed JSONL file
LANG=$2          # Language: go, rust, or julia

INPUT_FOLDER_DIR=$(dirname "$INPUT_DIR")

# Build Docker image name dynamically based on language
IMAGE_NAME="tessera2025testgen/tessera:${LANG}_environment"
FILE_NAME=$(basename "$INPUT_DIR")
# ====== Define output file paths ======
OUTPUT_SUMMARY="/data/summary.json"
OUTPUT_DATA_DIR="/data/detailed_results.jsonl"

# ====== Display run information ======
echo ">>> Running main.py with the following arguments:"
echo "    --dataset_dir $INPUT_DIR"
echo "    --output_summary $INPUT_DIR/summary.json"
echo "    --output_data_dir $INPUT_DIR/detailed_results.jsonl"
echo "Docker image: $IMAGE_NAME"
echo "-------------------------------------------"

# ====== Run Docker container ======
docker run --rm \
    -v "$PROJECT_ROOT/$INPUT_FOLDER_DIR":/data \
    "$IMAGE_NAME" \
    python3 /app/main.py \
        --dataset_dir "/data/$FILE_NAME" \
        --output_summary "$OUTPUT_SUMMARY" \
        --output_data_dir "$OUTPUT_DATA_DIR"

echo "Done"

