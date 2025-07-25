#!/bin/bash
# Startup script for CS2 Skin Prediction Training

echo "Starting CS2 Skin Prediction Training..."
echo "GPU Info:"
nvidia-smi

echo "Installing dependencies..."
pip install -r requirements_gpu.txt

echo "Starting training..."
python gpu_trainer.py

echo "Training completed!"
