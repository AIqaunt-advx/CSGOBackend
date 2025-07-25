# Cell 1: Environment Setup and Dependencies
# Run this first to install required packages

import subprocess
import sys

def install_packages():
    """Install required packages for the remote GPU environment"""
    packages = [
        'pandas==2.0.3',
        'numpy==1.24.3', 
        'scikit-learn==1.3.0',
        'matplotlib==3.7.2',
        'seaborn==0.12.2',
        'xgboost==2.0.3',
        'lightgbm==4.1.0',
        'catboost==1.2.2',
        'joblib==1.3.2',
        'tqdm==4.66.1',
        'optuna==3.4.0'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

# Uncomment and run if packages need to be installed
# install_packages()

# Verify GPU access
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")