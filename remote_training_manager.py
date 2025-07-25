#!/usr/bin/env python3
"""
Remote GPU Training Manager for 公益云 (Gongjiyun)
Handles deployment, monitoring, and cost management
"""

import os
import time
import json
import subprocess
from datetime import datetime, timedelta

class RemoteTrainingManager:
    def __init__(self):
        self.config = {
            "platform": "gongjiyun",
            "project_name": "cs2-skin-prediction",
            "estimated_training_time": 2,  # hours
            "max_cost_limit": 50,  # RMB
            "auto_shutdown": True
        }
        
    def check_account_setup(self):
        """Guide user through account setup checklist"""
        print("=== 公益云 Account Setup Checklist ===")
        
        checklist = [
            "✓ Registered account at gongjiyun.com",
            "✓ Completed identity verification",
            "✓ Added payment method (credit card/Alipay/WeChat)",
            "✓ Applied for GPU quota approval",
            "✓ Set up spending alerts/limits",
            "✓ Have sufficient balance for training"
        ]
        
        print("\nPlease confirm you have completed:")
        for item in checklist:
            print(f"  {item}")
        
        response = input("\nHave you completed all items above? (y/n): ")
        if response.lower() != 'y':
            print("\nPlease complete the account setup first:")
            print("1. Visit: https://www.gongjiyun.com/")
            print("2. Follow the setup guide in: gongjiyun_setup_guide.md")
            return False
        
        return True
    
    def estimate_costs(self, dataset_size="single"):
        """Estimate training costs based on dataset size"""
        cost_estimates = {
            "single": {
                "gpu_type": "RTX 3080",
                "training_time": 1,  # hours
                "gpu_cost_per_hour": 2.5,  # RMB
                "storage_cost": 0.5,  # RMB
                "network_cost": 1.0,  # RMB
            },
            "large": {
                "gpu_type": "RTX 4090", 
                "training_time": 6,  # hours
                "gpu_cost_per_hour": 5.0,  # RMB
                "storage_cost": 2.0,  # RMB
                "network_cost": 3.0,  # RMB
            }
        }
        
        estimate = cost_estimates.get(dataset_size, cost_estimates["single"])
        
        total_gpu_cost = estimate["training_time"] * estimate["gpu_cost_per_hour"]
        total_cost = total_gpu_cost + estimate["storage_cost"] + estimate["network_cost"]
        
        print(f"\n=== Cost Estimate ({dataset_size} dataset) ===")
        print(f"GPU Type: {estimate['gpu_type']}")
        print(f"Training Time: {estimate['training_time']} hours")
        print(f"GPU Cost: ¥{total_gpu_cost:.2f}")
        print(f"Storage Cost: ¥{estimate['storage_cost']:.2f}")
        print(f"Network Cost: ¥{estimate['network_cost']:.2f}")
        print(f"Total Estimated Cost: ¥{total_cost:.2f} (~${total_cost/7:.2f} USD)")
        
        return total_cost
    
    def create_deployment_config(self):
        """Create deployment configuration for remote training"""
        config = {
            "instance_config": {
                "gpu_type": "RTX-4090",
                "cpu_cores": 8,
                "memory_gb": 32,
                "storage_gb": 100,
                "os": "Ubuntu-20.04-CUDA-11.8"
            },
            "training_config": {
                "python_version": "3.9",
                "requirements_file": "requirements_gpu.txt",
                "main_script": "gpu_trainer.py",
                "data_files": ["22313.json"],
                "output_dir": "models"
            },
            "monitoring": {
                "auto_shutdown": True,
                "max_training_time": 8,  # hours
                "cost_alert_threshold": 40,  # RMB
                "save_interval": 30  # minutes
            }
        }
        
        with open("deployment_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("Deployment configuration saved to: deployment_config.json")
        return config
    
    def create_startup_script(self):
        """Create startup script for remote instance"""
        startup_script = """#!/bin/bash
# CS2 Skin Prediction Training - Startup Script

set -e  # Exit on any error

echo "=== Starting CS2 Skin Prediction Training ==="
echo "Time: $(date)"
echo "GPU Info:"
nvidia-smi

# Setup environment
echo "Setting up Python environment..."
python3 -m venv cs2_env
source cs2_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements_gpu.txt

# Verify GPU access
echo "Verifying GPU access..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Start training with monitoring
echo "Starting training..."
python gpu_trainer.py 2>&1 | tee training.log

# Save results
echo "Training completed. Saving results..."
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz models/ *.log *.png

echo "=== Training Completed Successfully ==="
echo "Results saved to: results_*.tar.gz"
echo "Download this file before instance shutdown!"

# Auto-shutdown (optional)
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "Auto-shutdown in 10 minutes..."
    sleep 600
    sudo shutdown -h now
fi
"""
        
        with open("startup.sh", "w") as f:
            f.write(startup_script)
        
        os.chmod("startup.sh", 0o755)
        print("Startup script created: startup.sh")
    
    def create_monitoring_script(self):
        """Create monitoring script for training progress"""
        monitoring_script = """#!/usr/bin/env python3
import time
import psutil
import GPUtil
import json
from datetime import datetime

def monitor_training():
    start_time = datetime.now()
    max_cost = 50  # RMB
    cost_per_hour = 5.0  # RMB for RTX 4090
    
    while True:
        # Get system stats
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Get GPU stats
        try:
            gpus = GPUtil.getGPUs()
            gpu_util = gpus[0].load * 100 if gpus else 0
            gpu_memory = gpus[0].memoryUtil * 100 if gpus else 0
        except:
            gpu_util = 0
            gpu_memory = 0
        
        # Calculate running cost
        elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
        current_cost = elapsed_hours * cost_per_hour
        
        # Create status report
        status = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_hours": round(elapsed_hours, 2),
            "current_cost_rmb": round(current_cost, 2),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "gpu_utilization": round(gpu_util, 1),
            "gpu_memory_percent": round(gpu_memory, 1)
        }
        
        # Save status
        with open("training_status.json", "w") as f:
            json.dump(status, f, indent=2)
        
        # Print status
        print(f"[{status['timestamp']}] "
              f"Cost: ¥{status['current_cost_rmb']:.2f} | "
              f"GPU: {status['gpu_utilization']:.1f}% | "
              f"Memory: {status['gpu_memory_percent']:.1f}%")
        
        # Cost alert
        if current_cost > max_cost * 0.8:
            print(f"WARNING: Cost approaching limit! Current: ¥{current_cost:.2f}")
        
        time.sleep(60)  # Update every minute

if __name__ == "__main__":
    monitor_training()
"""
        
        with open("monitor_training.py", "w") as f:
            f.write(monitoring_script)
        
        print("Monitoring script created: monitor_training.py")
    
    def prepare_deployment_package(self):
        """Prepare complete deployment package"""
        print("Preparing deployment package for remote GPU training...")
        
        # Create all necessary files
        self.create_deployment_config()
        self.create_startup_script()
        self.create_monitoring_script()
        
        # Update requirements for GPU monitoring
        gpu_requirements = """# GPU Training Requirements
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
requests==2.31.0
joblib==1.3.2

# GPU-accelerated ML libraries
xgboost==1.7.6
lightgbm==4.0.0
catboost==1.2.2

# Deep learning frameworks (GPU support)
torch==2.0.1
tensorflow==2.13.0

# Monitoring and utilities
tqdm==4.66.1
wandb==0.15.8
optuna==3.3.0
psutil==5.9.5
gputil==1.4.0
"""
        
        with open("requirements_gpu_full.txt", "w") as f:
            f.write(gpu_requirements)
        
        print("✓ Deployment package prepared")
        print("✓ Files created:")
        print("  - deployment_config.json")
        print("  - startup.sh")
        print("  - monitor_training.py") 
        print("  - requirements_gpu_full.txt")
    
    def deploy_to_remote(self):
        """Guide user through remote deployment"""
        print("\n=== Remote Deployment Guide ===")
        
        if not self.check_account_setup():
            return False
        
        # Estimate costs
        dataset_type = input("Dataset size? (single/large): ").lower()
        if dataset_type not in ["single", "large"]:
            dataset_type = "single"
        
        estimated_cost = self.estimate_costs(dataset_type)
        
        confirm = input(f"\nProceed with estimated cost of ¥{estimated_cost:.2f}? (y/n): ")
        if confirm.lower() != 'y':
            print("Deployment cancelled.")
            return False
        
        # Prepare deployment package
        self.prepare_deployment_package()
        
        print("\n=== Next Steps ===")
        print("1. Upload cs2-skin-prediction.zip to your 公益云 instance")
        print("2. Run: bash startup.sh")
        print("3. Monitor progress: python monitor_training.py")
        print("4. Download results when complete")
        
        print(f"\n=== Important Reminders ===")
        print(f"• Training will auto-shutdown after completion")
        print(f"• Download results immediately to avoid storage costs")
        print(f"• Monitor costs in real-time via dashboard")
        print(f"• Support: WeChat/QQ customer service")
        
        return True

def main():
    manager = RemoteTrainingManager()
    
    print("CS2 Skin Prediction - Remote GPU Training Manager")
    print("=" * 55)
    
    action = input("Choose action:\n1. Setup deployment\n2. Cost estimation only\nChoice (1/2): ")
    
    if action == "1":
        manager.deploy_to_remote()
    elif action == "2":
        dataset_type = input("Dataset size? (single/large): ").lower()
        manager.estimate_costs(dataset_type)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()