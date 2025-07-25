
# CS2 Skin Prediction - Remote GPU Training Deployment

## ¹«ÒæÔÆ (Gongjiyun) Deployment Instructions

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
