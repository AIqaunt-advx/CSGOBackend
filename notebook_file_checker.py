# Cell 0: File Checker and Directory Inspector
# Run this first to see what files are available

import os
import glob
import numpy as np

def check_current_directory():
    """Check what files are available in the current directory and dataset2"""
    print("🔍 Directory Contents:")
    print("=" * 50)
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"📁 Working Directory: {current_dir}")
    
    # Check current directory
    print(f"\n📂 Current Directory (.):")
    all_files = os.listdir('.')
    print(f"   Total files: {len(all_files)}")
    
    # Check dataset2 directory
    dataset_dir = './dataset2/'
    dataset_files = []
    if os.path.exists(dataset_dir):
        dataset_files = os.listdir(dataset_dir)
        print(f"\n📂 Dataset Directory ({dataset_dir}):")
        print(f"   Total files: {len(dataset_files)}")
    else:
        print(f"\n📂 Dataset Directory ({dataset_dir}): NOT FOUND")
    
    # Categorize files from both directories
    data_files = {
        'NPZ files (current)': [f for f in all_files if f.endswith('.npz')],
        'NPZ files (dataset2)': [f for f in dataset_files if f.endswith('.npz')],
        'JSON files (current)': [f for f in all_files if f.endswith('.json')],
        'JSON files (dataset2)': [f for f in dataset_files if f.endswith('.json')],
        'CSV files (current)': [f for f in all_files if f.endswith('.csv')],
        'CSV files (dataset2)': [f for f in dataset_files if f.endswith('.csv')],
        'Python files': [f for f in all_files if f.endswith('.py')],
        'Other files': [f for f in all_files if not any(f.endswith(ext) for ext in ['.npz', '.json', '.csv', '.pkl', '.py'])]
    }
    
    for category, files in data_files.items():
        if files:
            print(f"\n{category}: {len(files)}")
            for i, file in enumerate(files[:10]):  # Show first 10
                file_size = os.path.getsize(file) / (1024*1024)  # MB
                print(f"  {i+1:2d}. {file:30} ({file_size:.2f} MB)")
            if len(files) > 10:
                print(f"     ... and {len(files) - 10} more files")
    
    return data_files

def quick_npz_peek(file_path, max_keys=5):
    """Quick peek at NPZ file structure"""
    try:
        print(f"\n🔍 Quick peek at: {file_path}")
        print("-" * 40)
        
        data = np.load(file_path, allow_pickle=True)
        keys = list(data.keys())
        
        print(f"Keys found: {len(keys)}")
        
        for i, key in enumerate(keys[:max_keys]):
            array = data[key]
            print(f"  {key:15} | Shape: {str(array.shape):15} | Type: {array.dtype}")
        
        if len(keys) > max_keys:
            print(f"  ... and {len(keys) - max_keys} more keys")
        
        data.close()
        return True
        
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def suggest_next_steps(data_files):
    """Suggest what to do next based on available files"""
    print("\n💡 Next Steps:")
    print("=" * 30)
    
    # Check for NPZ files in either location
    npz_current = data_files.get('NPZ files (current)', [])
    npz_dataset2 = data_files.get('NPZ files (dataset2)', [])
    
    if npz_dataset2:
        print("✅ NPZ files found in dataset2 directory! You can proceed with training.")
        print("   → The workflow will automatically use ./dataset2/ directory")
        print("   → Run the NPZ loader cell next")
        
        # Quick peek at first NPZ file in dataset2
        first_npz = os.path.join('./dataset2/', npz_dataset2[0])
        quick_npz_peek(first_npz)
        
    elif npz_current:
        print("✅ NPZ files found in current directory!")
        print("   → Consider moving them to ./dataset2/ directory")
        print("   → Or modify the workflow to use current directory")
        
        # Quick peek at first NPZ file
        first_npz = npz_current[0]
        quick_npz_peek(first_npz)
        
    elif data_files.get('JSON files (dataset2)', []) or data_files.get('JSON files (current)', []):
        print("📄 JSON files found, but NPZ expected.")
        print("   → You may need to convert JSON to NPZ format")
        print("   → Or modify the loader to handle JSON files")
        
    elif data_files.get('CSV files (dataset2)', []) or data_files.get('CSV files (current)', []):
        print("📊 CSV files found, but NPZ expected.")
        print("   → You may need to convert CSV to NPZ format")
        
    else:
        print("❌ No data files found!")
        print("   → Please upload your .npz files to ./dataset2/ directory")
        print("   → Make sure the dataset2 folder exists and contains your NPZ files")

# Run the checker
data_files = check_current_directory()
suggest_next_steps(data_files)