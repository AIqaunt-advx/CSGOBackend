# Cell 1.5: NPZ File Debugger
# Run this to debug specific NPZ file issues

import numpy as np
import pandas as pd
import os

def debug_npz_file(file_path):
    """Debug a specific NPZ file to understand its structure"""
    print(f"🔍 Debugging NPZ file: {file_path}")
    print("=" * 60)
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        print(f"📁 File: {os.path.basename(file_path)}")
        print(f"🔑 Keys: {list(data.keys())}")
        print()
        
        for key in data.keys():
            array = data[key]
            print(f"Key: {key}")
            print(f"  Type: {type(array)}")
            print(f"  Dtype: {array.dtype}")
            print(f"  Shape: {array.shape}")
            print(f"  Size: {array.size}")
            print(f"  Ndim: {array.ndim}")
            
            # Try to show content safely
            try:
                if array.ndim == 0:  # Scalar
                    item = array.item()
                    print(f"  Scalar value: {item} (type: {type(item)})")
                    
                    # If it's a complex object, try to inspect it
                    if hasattr(item, '__dict__'):
                        print(f"  Object attributes: {dir(item)}")
                    elif isinstance(item, (dict, list, tuple)):
                        print(f"  Container length: {len(item)}")
                        if isinstance(item, dict):
                            print(f"  Dict keys: {list(item.keys())}")
                        
                elif array.ndim == 1:
                    print(f"  Sample values: {array[:min(5, len(array))]}")
                    
                elif array.ndim == 2:
                    print(f"  Sample shape: {array[:min(3, array.shape[0]), :min(3, array.shape[1])]}")
                    
                else:
                    print(f"  High-dimensional array, first few elements: {array.flat[:5]}")
                    
            except Exception as e:
                print(f"  ⚠️ Error accessing content: {e}")
            
            print()
        
        data.close()
        
    except Exception as e:
        print(f"❌ Error opening file: {e}")
        import traceback
        traceback.print_exc()

def try_load_strategies(file_path):
    """Try different loading strategies for an NPZ file"""
    print(f"🧪 Trying different loading strategies for: {os.path.basename(file_path)}")
    print("=" * 60)
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Strategy 1: Look for 'data' key
        if 'data' in data.keys():
            print("📊 Strategy 1: Found 'data' key")
            main_data = data['data']
            print(f"   Data type: {type(main_data)}")
            print(f"   Data shape: {main_data.shape}")
            print(f"   Data dtype: {main_data.dtype}")
            
            try:
                if main_data.ndim == 0:
                    item = main_data.item()
                    print(f"   Extracted item: {type(item)}")
                    
                    if isinstance(item, dict):
                        print(f"   Dict keys: {list(item.keys())}")
                        # Try to create DataFrame
                        df = pd.DataFrame(item)
                        print(f"   ✅ DataFrame created: {df.shape}")
                        return df
                    elif isinstance(item, (list, tuple)):
                        print(f"   List/tuple length: {len(item)}")
                        df = pd.DataFrame(item)
                        print(f"   ✅ DataFrame created: {df.shape}")
                        return df
                        
            except Exception as e:
                print(f"   ❌ Strategy 1 failed: {e}")
        
        # Strategy 2: Multiple keys as columns
        print("\n📊 Strategy 2: Multiple keys as columns")
        df_dict = {}
        
        for key in data.keys():
            array_data = data[key]
            print(f"   Processing key '{key}': shape {array_data.shape}, dtype {array_data.dtype}")
            
            try:
                if array_data.ndim == 0:
                    df_dict[key] = [array_data.item()]
                    print(f"     ✅ Scalar converted to list")
                elif array_data.ndim == 1:
                    df_dict[key] = array_data
                    print(f"     ✅ 1D array added")
                elif array_data.ndim == 2 and array_data.shape[1] == 1:
                    df_dict[key] = array_data.flatten()
                    print(f"     ✅ Column vector flattened")
                else:
                    print(f"     ⚠️ Skipping complex shape")
                    
            except Exception as e:
                print(f"     ❌ Error: {e}")
        
        if df_dict:
            print(f"\n   Collected {len(df_dict)} columns")
            
            # Check lengths
            lengths = {}
            for k, v in df_dict.items():
                if hasattr(v, '__len__'):
                    lengths[k] = len(v)
                else:
                    lengths[k] = 1
            
            print(f"   Column lengths: {lengths}")
            
            if len(set(lengths.values())) == 1:
                df = pd.DataFrame(df_dict)
                print(f"   ✅ DataFrame created: {df.shape}")
                return df
            else:
                print(f"   ⚠️ Inconsistent lengths, need alignment")
        
        data.close()
        
    except Exception as e:
        print(f"❌ Error in strategies: {e}")
        import traceback
        traceback.print_exc()
    
    return None

# Test with first NPZ file from dataset2
dataset_dir = './dataset2/'
if os.path.exists(dataset_dir):
    npz_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
    if npz_files:
        test_file = os.path.join(dataset_dir, npz_files[0])
        print(f"🧪 Testing with: {test_file}")
        debug_npz_file(test_file)
        print("\n" + "="*60 + "\n")
        result_df = try_load_strategies(test_file)
        
        if result_df is not None:
            print(f"\n🎉 Success! DataFrame shape: {result_df.shape}")
            print(f"Columns: {list(result_df.columns)}")
            print(f"Sample data:\n{result_df.head()}")
        else:
            print("\n❌ All strategies failed")
    else:
        print(f"❌ No NPZ files found in {dataset_dir}")
else:
    print(f"❌ Dataset directory {dataset_dir} not found!")