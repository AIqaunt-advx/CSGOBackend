# 公益云 (Gongjiyun) Remote GPU Training Setup Guide

## Account Setup & Billing

### 1. Account Registration & Verification
- Go to https://www.gongjiyun.com/
- Register with your email/phone number
- Complete identity verification (required for GPU access)
- Add payment method (credit card/Alipay/WeChat Pay)

### 2. GPU Resource Application
- Navigate to "GPU Computing" section
- Apply for GPU quota (may require approval)
- Choose GPU type:
  - **RTX 3080** (12GB VRAM) - ~¥2-3/hour
  - **RTX 4090** (24GB VRAM) - ~¥4-6/hour
  - **A100** (40GB VRAM) - ~¥8-12/hour (for large datasets)

### 3. Billing Configuration
- Set up auto-payment to avoid interruptions
- Monitor usage in real-time dashboard
- Set spending limits/alerts

## Deployment Steps

### Step 1: Upload Your Training Package
```bash
# Upload the deployment package we created
# File: cs2-skin-prediction.zip (contains all your code)
```

### Step 2: Create GPU Instance
1. Select GPU type based on your needs:
   - For single JSON file (22313.json): RTX 3080 sufficient
   - For 2000+ files: RTX 4090 or A100 recommended

2. Choose OS: Ubuntu 20.04 LTS (recommended)

3. Configure resources:
   - CPU: 8+ cores
   - RAM: 16GB+
   - Storage: 100GB+ SSD

### Step 3: Environment Setup
```bash
# After connecting to your instance
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y

# Create virtual environment
python3 -m venv cs2_env
source cs2_env/bin/activate

# Install CUDA (if not pre-installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda-11-8 -y
```

### Step 4: Deploy Your Code
```bash
# Extract your deployment package
unzip cs2-skin-prediction.zip
cd cs2-skin-prediction

# Install dependencies
pip install -r requirements_gpu.txt

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Step 5: Start Training
```bash
# For single file training
python gpu_trainer.py

# For batch processing (if you have multiple JSON files)
python batch_process.py
```

## Cost Estimation

### Training Time Estimates:
- **Single JSON file (22313.json)**: 30-60 minutes
- **2000+ JSON files**: 4-8 hours

### Cost Breakdown:
- **RTX 3080**: ¥2.5/hour × 1 hour = ¥2.5 (~$0.35)
- **RTX 4090**: ¥5/hour × 6 hours = ¥30 (~$4.20)
- **Storage**: ¥0.1/GB/month
- **Network**: ¥0.5/GB transfer

### Total Estimated Cost:
- **Small dataset**: ¥3-5 (~$0.50-0.70)
- **Large dataset**: ¥30-50 (~$4-7)

## Monitoring & Management

### Real-time Monitoring:
- GPU utilization dashboard
- Training progress logs
- Cost tracking
- Auto-shutdown when complete

### Best Practices:
1. **Set spending alerts** to avoid unexpected charges
2. **Use spot instances** for 50-70% cost savings (if available)
3. **Auto-shutdown** after training completion
4. **Download results immediately** to avoid storage costs

## Troubleshooting

### Common Issues:
1. **GPU quota exceeded**: Apply for higher quota or wait
2. **CUDA version mismatch**: Use pre-configured ML images
3. **Out of memory**: Reduce batch size or use larger GPU
4. **Network timeout**: Use screen/tmux for long training

### Support:
- 24/7 technical support via WeChat/QQ
- Documentation: https://docs.gongjiyun.com/
- Community forum for ML users

## Alternative Platforms (if 公益云 unavailable):

1. **AutoDL** (autodl.com) - Similar pricing, good for ML
2. **阿里云 PAI** - Alibaba Cloud ML platform
3. **腾讯云 TI** - Tencent Cloud ML platform
4. **百度云 AI Studio** - Baidu's ML platform

Would you like me to help you with any specific part of the setup process?