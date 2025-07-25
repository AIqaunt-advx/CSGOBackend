# CS2 Skin Trading Prediction - Jupyter Notebook Setup Guide

## Environment: Ubuntu 20.04, Python 3.12, PyTorch 2.7.1, CUDA 12.8

### Step 1: Copy Files to Jupyter Environment

Upload these Python files to your Jupyter notebook environment:
- `notebook_setup.py` - Environment setup and GPU verification
- `notebook_npz_loader.py` - NPZ file loader and inspector  
- `notebook_data_processor.py` - Data processing and feature engineering
- `notebook_gpu_trainer.py` - GPU-optimized model training
- `notebook_main_workflow.py` - Complete training workflow

### Step 2: Jupyter Notebook Cell Structure

Create a new Jupyter notebook with the following cells:

#### Cell 0: File Checker (NEW - Run this first!)
```python
# Copy content from notebook_file_checker.py
exec(open('notebook_file_checker.py').read())
```

#### Cell 1: Environment Setup
```python
# Copy content from notebook_setup.py
exec(open('notebook_setup.py').read())
```

#### Cell 2: NPZ Data Inspection
```python
# Copy content from notebook_npz_loader.py
exec(open('notebook_npz_loader.py').read())
```

#### Cell 3: Data Processor
```python
# Copy content from notebook_data_processor.py
exec(open('notebook_data_processor.py').read())
```

#### Cell 4: GPU Trainer
```python
# Copy content from notebook_gpu_trainer.py
exec(open('notebook_gpu_trainer.py').read())
```

#### Cell 5: Main Workflow
```python
# Copy content from notebook_main_workflow.py
exec(open('notebook_main_workflow.py').read())
```

### Step 3: Alternative - Direct Copy-Paste Method

Instead of using `exec(open(...))`, you can directly copy-paste the content of each file into separate cells:

1. **Cell 1**: Copy entire content of `notebook_setup.py`
2. **Cell 2**: Copy entire content of `notebook_npz_loader.py`
3. **Cell 3**: Copy entire content of `notebook_data_processor.py`
4. **Cell 4**: Copy entire content of `notebook_gpu_trainer.py`
5. **Cell 5**: Copy entire content of `notebook_main_workflow.py`

### Step 4: Running the Training

1. **Run Cell 1**: Install dependencies and verify GPU
2. **Run Cell 2**: Inspect your NPZ files structure
3. **Run Cell 3**: Initialize data processor
4. **Run Cell 4**: Initialize GPU trainer
5. **Run Cell 5**: Execute complete training workflow

### Step 5: Expected Output

The training will:
- ✅ Load and inspect NPZ files
- ✅ Process and engineer features
- ✅ Train multiple GPU-accelerated models:
  - XGBoost (GPU)
  - LightGBM (GPU)
  - CatBoost (GPU)
  - Deep Neural Network (PyTorch)
  - Random Forest (baseline)
- ✅ Compare model performance
- ✅ Save best model
- ✅ Generate visualizations

### Step 6: Troubleshooting

#### NPZ File Issues:
- Run the inspection cell first to understand your data structure
- The loader handles various NPZ formats automatically
- Check column mappings in `post_process_dataframe()`

#### GPU Issues:
```python
# Verify GPU access
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

#### Memory Issues:
- Reduce batch size in deep learning training
- Process fewer files initially for testing
- Use `max_files` parameter in `load_all_npz_files()`

#### Package Issues:
```python
# Install missing packages
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'package_name'])
```

### Step 7: Customization

#### Adjust Model Parameters:
```python
# In the GPU trainer, modify parameters:
params = {
    'max_depth': 10,  # Increase for more complex models
    'learning_rate': 0.05,  # Decrease for better convergence
    'n_estimators': 2000,  # Increase for better performance
}
```

#### Feature Engineering:
```python
# Add custom features in create_features() method
df_features['custom_feature'] = df_features['price'] / df_features['onSaleQuantity']
```

#### Target Variable:
```python
# Change target in prepare_data() method
X_train, X_test, y_train, y_test, features = processor.prepare_data(df, target_col='your_target')
```

### Step 8: Results and Model Usage

After training, you'll have:
- `best_cs2_model.pkl` - Best performing model
- Performance visualizations
- Feature importance analysis
- Model comparison charts

#### Using the Trained Model:
```python
import joblib

# Load saved model
model_data = joblib.load('best_cs2_model.pkl')
model = model_data['model']
model_name = model_data['model_name']

# Make predictions
prediction = model.predict([[on_sale_qty, seek_qty, price, seek_price, ...]])
```

### Expected Training Time:
- **Single NPZ file**: 5-15 minutes
- **10 NPZ files**: 30-60 minutes  
- **100+ NPZ files**: 2-4 hours

### GPU Memory Usage:
- **XGBoost/LightGBM**: ~2-4GB VRAM
- **Deep Neural Network**: ~4-8GB VRAM
- **Total system**: ~8-16GB RAM

Ready to start training! 🚀