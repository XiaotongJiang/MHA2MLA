#!/bin/bash

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <yaml_directory>"
    exit 1
fi

YAML_DIR="$1"

# Check if directory exists
if [ ! -d "$YAML_DIR" ]; then
    echo "Error: Directory '$YAML_DIR' does not exist"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to run training for a single YAML file
run_training() {
    local yaml_file="$1"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="logs/training_${timestamp}.log"
    
    echo "Starting training with config: $yaml_file"
    echo "Log file: $log_file"
    
    TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 1 \
        ../src/mha2mla_nt/run_train.py \
        --config-file "$yaml_file" 2>&1 | tee "$log_file"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for $yaml_file"
    else
        echo "Training failed for $yaml_file"
    fi
}

# Process all YAML files in the directory
for yaml_file in "$YAML_DIR"/*.yaml; do
    if [ -f "$yaml_file" ]; then
        run_training "$yaml_file"
    fi
done

echo "All training jobs completed" 