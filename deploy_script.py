#!/usr/bin/env python3
"""
Deployment script for remote GPU training on 公益云 (Gongjiyun)
Based on the documentation: https://www.gongjiyun.com/docs/y/nnnkwiwlkid3m1ksgxdcrvsknrj/l3a6wgrppity1qkorxdx6aqnvf/
"""

import os
import subprocess
import json
import zipfile
import shutil
from pathlib import Path

class GongjiyunDeployer:
    def __init__(self):
        self.project_name = "cs2-skin-prediction"
        self.files_to_upload = [
            "data_processor.py",
            "gpu_trainer.py", 
            "requirements_gpu.txt",
            "22313.json",
            "deploy_script.py"
        ]
        
    def create_deployment_package(self):
        """Create deployment package for upload"""
        print("Creating deployment package...")
        
        # Create deployment directory
        deploy_dir = Path("deployment")
        deploy_dir.mkdir(exist_ok=True)
        
        # Copy files
        for file_name in self.files_to_upload:
            if os.path.exists(file_name):
                shutil.copy2(file_name, deploy_dir / file_name)
                print(f"Added {file_name}")
            else:
                print(f"Warning: {file_name} not found")
        
        # Create startup script
        startup_script = """#!/bin/bash
# Startup script for CS2 Skin Prediction Training

echo "Starting CS2 Skin Prediction Training..."
echo "GPU Info:"
nvidia-smi

echo "Installing dependencies..."
pip install -r requirements_gpu.txt

echo "Starting training..."
python gpu_trainer.py

echo "Training completed!"
"""
        
        with open(deploy_dir / "start.sh", "w") as f:
            f.write(startup_script)
        
        # Create configuration file
        config = {
            "project_name": self.project_name,
            "python_version": "3.9",
            "gpu_required": True,
            "memory_gb": 16,
            "storage_gb": 50,
            "estimated_runtime_hours": 2
        }
        
        with open(deploy_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Create zip file
        zip_path = f"{self.project_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(deploy_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, deploy_dir)
                    zipf.write(file_path, arcname)
        
        print(f"Deployment package created: {zip_path}")
        return zip_path
    
    def generate_docker_setup(self):
        """Generate Docker setup for local testing"""
        dockerfile = """FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_gpu.txt .
RUN pip3 install --no-cache-dir -r requirements_gpu.txt

# Copy application files
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Run the application
CMD ["./start.sh"]
"""
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile)
        
        docker_compose = """version: '3.8'

services:
  cs2-trainer:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
        
        with open("docker-compose.yml", "w") as f:
            f.write(docker_compose)
        
        print("Docker setup files created (Dockerfile, docker-compose.yml)")
    
    def create_batch_processing_script(self):
        """Create script for processing multiple JSON files"""
        batch_script = """#!/usr/bin/env python3
import os
import glob
import json
from data_processor import SkinDataProcessor
from gpu_trainer import GPUSkinTrainer

def process_all_json_files():
    processor = SkinDataProcessor()
    trainer = GPUSkinTrainer()
    
    # Load all JSON files
    json_files = glob.glob("*.json")
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) > 1:
        # Process multiple files
        df = processor.load_multiple_json_files("*.json")
    else:
        # Process single file
        df = processor.load_single_json('22313.json')
    
    if df is None:
        print("No data loaded")
        return
    
    print(f"Total records: {len(df)}")
    
    # Prepare and train
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(df)
    models = trainer.train_all_models(X_train, X_test, y_train, y_test)
    trainer.save_models()
    
    print("Batch processing completed!")

if __name__ == "__main__":
    process_all_json_files()
"""
        
        with open("batch_process.py", "w") as f:
            f.write(batch_script)
        
        print("Batch processing script created: batch_process.py")
    
    def generate_deployment_instructions(self):
        """Generate deployment instructions"""
        instructions = """
# CS2 Skin Prediction - Remote GPU Training Deployment

## 公益云 (Gongjiyun) Deployment Instructions

### Step 1: Prepare Your Environment
1. Register at https://www.gongjiyun.com/
2. Apply for GPU resources (recommend RTX 3080/4090 or better)
3. Upload the deployment package: cs2-skin-prediction.zip

### Step 2: Environment Setup
```bash
# After connecting to your GPU instance
unzip cs2-skin-prediction.zip
cd cs2-skin-prediction
chmod +x start.sh
```

### Step 3: Install Dependencies
```bash
pip install -r requirements_gpu.txt
```

### Step 4: Verify GPU Access
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 5: Start Training
```bash
python gpu_trainer.py
```

### Step 6: Monitor Training
- Check logs for training progress
- Models will be saved to ./models/ directory
- Use wandb.ai for experiment tracking (optional)

### Step 7: Download Results
After training completes, download:
- ./models/ directory (contains trained models)
- Training logs and metrics

## Local Testing with Docker
```bash
# Build and run locally (requires NVIDIA Docker)
docker-compose up --build
```

## Processing Multiple Files
If you have 2000+ JSON files:
1. Upload all JSON files to the same directory
2. Run: python batch_process.py

## Expected Training Time
- Single file (22313.json): ~30 minutes
- 2000+ files: ~4-6 hours (depending on GPU)

## GPU Requirements
- Minimum: GTX 1080 Ti (11GB VRAM)
- Recommended: RTX 3080/4090 (16GB+ VRAM)
- Memory: 16GB+ RAM
- Storage: 50GB+ free space

## Troubleshooting
1. CUDA out of memory: Reduce batch_size in gpu_trainer.py
2. JSON loading errors: Check file format consistency
3. Model training fails: Try individual model training functions
"""
        
        with open("DEPLOYMENT_INSTRUCTIONS.md", "w") as f:
            f.write(instructions)
        
        print("Deployment instructions created: DEPLOYMENT_INSTRUCTIONS.md")

def main():
    deployer = GongjiyunDeployer()
    
    print("CS2 Skin Prediction - Deployment Setup")
    print("=" * 50)
    
    # Create deployment package
    zip_path = deployer.create_deployment_package()
    
    # Generate Docker setup for local testing
    deployer.generate_docker_setup()
    
    # Create batch processing script
    deployer.create_batch_processing_script()
    
    # Generate instructions
    deployer.generate_deployment_instructions()
    
    print("\n" + "=" * 50)
    print("Deployment setup completed!")
    print(f"Upload {zip_path} to 公益云 for remote GPU training")
    print("See DEPLOYMENT_INSTRUCTIONS.md for detailed steps")

if __name__ == "__main__":
    main()