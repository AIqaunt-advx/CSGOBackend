# Cell 4: Main Training Workflow
# Copy this entire cell and run it to execute the complete training pipeline

import os
import time
from datetime import datetime

def run_complete_training():
    """Complete training workflow"""
    
    print("🎯 CS2 Skin Trading Volume Prediction - GPU Training")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    
    # Step 1: Load Data
    print("\n📁 Step 1: Loading Data...")
    
    # Check for NPZ files in dataset2 directory
    dataset_dir = './dataset2/'
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory {dataset_dir} not found!")
        print("💡 Please make sure your dataset2 folder exists")
        return
    
    npz_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
    print(f"Found NPZ files in {dataset_dir}: {len(npz_files)} files")
    
    if not npz_files:
        print(f"❌ No NPZ files found in {dataset_dir}!")
        print("💡 Make sure your .npz files are in the dataset2 directory")
        print("💡 You can also check other file types:")
        all_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.npz', '.json', '.csv', '.pkl'))]
        print(f"   Available data files in {dataset_dir}: {all_files}")
        return
    
    # Load data using smart NPZ loader
    print(f"\n📁 Loading ALL {len(npz_files)} skin files from {dataset_dir}...")
    print("   Each file represents data for one skin - combining all for comprehensive training")
    
    # Load all files (each represents one skin)
    df = load_all_npz_files(dataset_dir, max_files=None)  # Load ALL files, not just 10
    
    if df is None:
        print("❌ Failed to load data!")
        return
    
    # Step 1.5: Extract complex data structures
    print("\n🔧 Step 1.5: Processing Complex Data Structure...")
    df = process_complex_data(df)
    
    # Step 2: Data Exploration
    print("\n📊 Step 2: Data Exploration...")
    processor.explore_data(df)
    
    # Step 3: Data Preparation
    print("\n🔧 Step 3: Data Preparation...")
    try:
        X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(df)
        print(f"✓ Data prepared successfully!")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Test samples: {X_test.shape[0]}")
        print(f"  - Features: {X_train.shape[1]}")
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return
    
    # Step 4: Model Training
    print("\n🚀 Step 4: GPU Model Training...")
    start_time = time.time()
    
    models = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    training_time = time.time() - start_time
    print(f"\n⏱️ Total training time: {training_time:.2f} seconds")
    
    # Step 5: Results Visualization
    print("\n📈 Step 5: Results Visualization...")
    trainer.plot_results(y_test)
    
    # Step 6: Save Best Model
    print("\n💾 Step 6: Saving Best Model...")
    trainer.save_best_model()
    
    # Step 7: Model Summary
    print("\n📋 Step 7: Training Summary")
    print("=" * 40)
    for name, model_info in models.items():
        score = model_info['score']
        print(f"{name:20} | R² Score: {score:.4f}")
    
    print(f"\n🏆 Best Model: {trainer.best_model_name}")
    print(f"🎯 Best Score: {trainer.best_score:.4f}")
    print(f"⏱️ Training Time: {training_time:.2f}s")
    print(f"✅ Training completed at: {datetime.now()}")
    
    return models

# Run the complete training workflow
models = run_complete_training()