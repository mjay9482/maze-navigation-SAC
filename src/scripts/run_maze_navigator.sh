#!/bin/bash

# Activate the virtual environment
source activate /home/mrityunjay/Documents/Manipulator/venv

# Create necessary directories if they don't exist
mkdir -p logs checkpoints results runs

# Training script
echo "Starting training with simplified maze..."
python3 main.py --mode train --config config.py --log_dir logs/ --checkpoints_dir checkpoints/

# Evaluation script
echo "Starting evaluation with simplified maze..."
python3 main.py --mode eval --config config.py --checkpoint checkpoints/latest_checkpoint

# Analyze metrics
echo "Analyzing metrics..."
python3 analyze_metrics.py --results_dir results/

# Run tests
echo "Running tests with simplified maze..."
python3 test.py

echo "All operations complete! :)" 