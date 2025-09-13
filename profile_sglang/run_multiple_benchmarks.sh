#!/bin/bash

# Configuration
MODEL="Qwen/Qwen3-Embedding-8B"
NUM_RUNS=3
HARDWARE="H100"  # Change this to "4090" or "Mi300x" for different hardware
RESULTS_DIR="embedding_results"  # Directory to store results

# Sequence lengths to test
SEQUENCE_LENGTHS=(100 200 400 800 1000 2000 4000 8000)

# Batch sizes to test (powers of 2 from 1 to 256)
BATCH_SIZES=(1 2 4 8 16 32 64 128 256)

# Convert batch sizes array to comma-separated string
BATCH_SIZES_STR=$(IFS=,; echo "${BATCH_SIZES[*]}")

echo "Starting embedding benchmark runs..."
echo "Model: $MODEL"
echo "Hardware: $HARDWARE"
echo "Results directory: $RESULTS_DIR"
echo "Sequence lengths: ${SEQUENCE_LENGTHS[*]}"
echo "Batch sizes: $BATCH_SIZES_STR"
echo "Number of runs per configuration: $NUM_RUNS"
echo ""

# Create results directory if it doesn't exist
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Creating results directory: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
    echo ""
fi

# Loop through each sequence length
for seq_len in "${SEQUENCE_LENGTHS[@]}"; do
    echo "Running benchmark for sequence length: $seq_len"
    
    # Create results filename with directory path
    results_file="$RESULTS_DIR/embedding_results_${seq_len}_${HARDWARE,,}.json"  # ${HARDWARE,,} converts to lowercase
    
    # Create description
    description="Qwen3 8B Embedding TP1 $HARDWARE"
    
    # Run the benchmark
    echo "Command: uv run tokenomics/embedding_benchmark.py --model $MODEL --sequence_lengths \"$seq_len\" --batch_sizes \"$BATCH_SIZES_STR\" --num_runs $NUM_RUNS --description \"$description\" --results_file $results_file"
    
    uv run tokenomics/embedding_benchmark.py \
        --model "$MODEL" \
        --sequence_lengths "$seq_len" \
        --batch_sizes "$BATCH_SIZES_STR" \
        --num_runs $NUM_RUNS \
        --description "$description" \
        --results_file "$results_file"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed benchmark for sequence length $seq_len"
        echo "  Results saved to: $results_file"
    else
        echo "✗ Failed to run benchmark for sequence length $seq_len"
        echo "  Continuing with next sequence length..."
    fi
    
    echo ""
done

echo "All benchmark runs completed!"
echo ""
echo "Results files generated:"
for seq_len in "${SEQUENCE_LENGTHS[@]}"; do
    results_file="$RESULTS_DIR/embedding_results_${seq_len}_${HARDWARE,,}.json"
    if [ -f "$results_file" ]; then
        echo "  ✓ $results_file"
    else
        echo "  ✗ $results_file (not found)"
    fi
done