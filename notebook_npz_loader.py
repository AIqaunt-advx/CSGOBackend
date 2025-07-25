# Cell 1.5: NPZ Data Loader and Inspector
# Run this cell first to inspect your NPZ files and understand their structure

import numpy as np
import pandas as pd
import os
import glob

def inspect_all_npz_files(directory='.'):
    """Inspect all NPZ files in directory to understand data structure"""
    npz_files = glob.glob(os.path.join(directory, '*.npz'))
    
    if not npz_files:
        print("❌ No NPZ files found in current directory!")
        return
    
    print(f"🔍 Found {len(npz_files)} NPZ files")
    print("=" * 60)
    
    for i, file_path in enumerate(npz_files[:5]):  # Show first 5 files
        print(f"\n📁 File {i+1}: {os.path.basename(file_path)}")
        print("-" * 40)
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            for key in data.keys():
                array = data[key]
                print(f"{key:20} | Shape: {str(array.shape):15} | Dtype: {str(array.dtype):10}")
                
                # Show sample values
                if array.size <= 10:
                    print(f"{'':20} | Values: {array}")
                elif array.ndim == 1 and len(array) > 0:
                    print(f"{'':20} | Sample: [{array[0]}, {array[1] if len(array) > 1 else '...'}, ...]")
                elif array.ndim == 2 and array.size > 0:
                    print(f"{'':20} | Sample: {array[0, :min(3, array.shape[1])]}")
            
            data.close()
            
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    
    if len(npz_files) > 5:
        print(f"\n... and {len(npz_files) - 5} more files")

def load_npz_smart(file_path):
    """Smart NPZ loader that handles different data structures"""
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Strategy 1: Direct DataFrame reconstruction
        if 'data' in data.keys():
            # If there's a 'data' key, try to use it
            main_data = data['data']
            try:
                if hasattr(main_data, 'item'):
                    # Try to extract item (for 0-d arrays)
                    item_data = main_data.item()
                    if isinstance(item_data, dict):
                        # Pickled dictionary
                        df = pd.DataFrame(item_data)
                    elif isinstance(item_data, (list, tuple)):
                        # List or tuple of records
                        df = pd.DataFrame(item_data)
                    else:
                        # Single value
                        df = pd.DataFrame({'value': [item_data]})
                elif main_data.ndim == 2:
                    # 2D array - need column names
                    columns = data.get('columns', [f'col_{i}' for i in range(main_data.shape[1])])
                    if hasattr(columns, 'item'):
                        columns = columns.item() if columns.ndim == 0 else columns
                    df = pd.DataFrame(main_data, columns=columns)
                elif main_data.ndim == 1:
                    # 1D array
                    df = pd.DataFrame({'data': main_data})
                else:
                    # 0D array (scalar)
                    df = pd.DataFrame({'data': [main_data.item()]})
            except (ValueError, AttributeError) as e:
                print(f"⚠️ Error processing 'data' key: {e}")
                # Fall back to treating as regular array
                if main_data.ndim >= 1:
                    df = pd.DataFrame({'data': main_data.flatten()})
                else:
                    df = pd.DataFrame({'data': [main_data]})
        
        # Strategy 2: Multiple arrays as columns
        else:
            df_dict = {}
            for key in data.keys():
                array_data = data[key]
                
                try:
                    # Handle different array types
                    if array_data.ndim == 0:  # Scalar
                        # Convert scalar to single-element array
                        df_dict[key] = [array_data.item()]
                    elif array_data.ndim == 1:  # 1D array
                        df_dict[key] = array_data
                    elif array_data.ndim == 2 and array_data.shape[1] == 1:  # Column vector
                        df_dict[key] = array_data.flatten()
                    elif array_data.ndim == 2:  # 2D array - split into columns
                        for i in range(array_data.shape[1]):
                            df_dict[f'{key}_{i}'] = array_data[:, i]
                    else:
                        print(f"Skipping {key} with shape {array_data.shape}")
                        continue
                except (ValueError, AttributeError) as e:
                    print(f"⚠️ Error processing key '{key}': {e}")
                    continue
            
            if df_dict:
                # Ensure all arrays have the same length
                lengths = [len(v) if hasattr(v, '__len__') else 1 for v in df_dict.values()]
                if len(set(lengths)) > 1:
                    min_length = min(lengths)
                    print(f"⚠️ Arrays have different lengths, truncating to {min_length}")
                    for k, v in df_dict.items():
                        if hasattr(v, '__len__') and len(v) > min_length:
                            df_dict[k] = v[:min_length]
                        elif not hasattr(v, '__len__'):
                            # Scalar value, repeat it
                            df_dict[k] = [v] * min_length
                
                df = pd.DataFrame(df_dict)
            else:
                print(f"❌ No suitable data found in {file_path}")
                data.close()
                return None
        
        data.close()
        
        # Post-process the DataFrame
        df = post_process_dataframe(df, file_path)
        
        print(f"✅ Loaded {len(df)} records from {os.path.basename(file_path)}")
        print(f"   Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        import traceback
        print(f"   Details: {traceback.format_exc()}")
        return None

def post_process_dataframe(df, file_path):
    """Post-process loaded DataFrame to standardize column names and data types"""
    
    print(f"   Post-processing {os.path.basename(file_path)}...")
    
    # Column name standardization
    column_mappings = {
        # Common variations for trading data
        'timestamp': 'timestamp',
        'time': 'timestamp', 
        'ts': 'timestamp',
        'price': 'price',
        'prices': 'price',
        'on_sale_quantity': 'onSaleQuantity',
        'onsale_quantity': 'onSaleQuantity',
        'current_sales_volume': 'onSaleQuantity',
        'supply': 'onSaleQuantity',
        'seek_quantity': 'seekQuantity',
        'wanted_sales_volume': 'seekQuantity',
        'demand': 'seekQuantity',
        'seek_price': 'seekPrice',
        'seekprice': 'seekPrice',
        'bid_price': 'seekPrice',
        'transaction_amount': 'transactionAmount',
        'success_sales_volume': 'transactionAmount',
        'volume': 'transactionAmount',
        'transaction_num': 'transcationNum',
        'transcation_num': 'transcationNum',
        'survive_num': 'surviveNum'
    }
    
    # Apply mappings
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Handle NULL values early - convert string nulls to proper NaN
    print(f"     Cleaning NULL values...")
    original_shape = df.shape
    
    # Replace various NULL representations with NaN
    null_values = ['NULL', 'null', 'None', 'none', 'NaN', 'nan', '', ' ', 'N/A', 'n/a']
    df = df.replace(null_values, np.nan)
    
    # Count NULL values per column
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        print(f"     Found {total_nulls} NULL values across {(null_counts > 0).sum()} columns")
        for col, count in null_counts.items():
            if count > 0:
                print(f"       {col}: {count} nulls ({count/len(df)*100:.1f}%)")
    
    # Handle timestamp conversion
    if 'timestamp' in df.columns:
        try:
            # Try different timestamp formats
            if df['timestamp'].dtype == 'object':
                df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
            else:
                # Assume Unix timestamp
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df['date'] = df['datetime'].dt.date
        except Exception as e:
            print(f"     ⚠️ Could not convert timestamp: {e}")
    
    # Convert to numeric and handle errors
    numeric_columns = ['price', 'onSaleQuantity', 'seekQuantity', 'seekPrice', 
                      'transactionAmount', 'transcationNum', 'surviveNum']
    
    for col in numeric_columns:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Count conversion errors
            conversion_nulls = df[col].isnull().sum() - null_counts.get(col, 0)
            if conversion_nulls > 0:
                print(f"     ⚠️ {col}: {conversion_nulls} values couldn't be converted to numeric")
    
    # Fill transaction metrics with 0 (these are often legitimately 0)
    for col in ['transactionAmount', 'transcationNum', 'surviveNum']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Report final NULL status
    final_nulls = df.isnull().sum().sum()
    if final_nulls > 0:
        print(f"     Final: {final_nulls} NULL values remaining (will be cleaned later)")
    else:
        print(f"     ✓ No NULL values remaining")
    
    return df

def load_all_npz_files(directory='.', max_files=None):
    """Load and combine all NPZ files in directory"""
    npz_files = glob.glob(os.path.join(directory, '*.npz'))
    
    if not npz_files:
        print("❌ No NPZ files found!")
        return None
    
    if max_files:
        npz_files = npz_files[:max_files]
    
    print(f"📁 Loading {len(npz_files)} NPZ files...")
    
    all_dataframes = []
    
    for i, file_path in enumerate(npz_files):
        print(f"\n📄 Processing file {i+1}/{len(npz_files)}: {os.path.basename(file_path)}")
        
        df = load_npz_smart(file_path)
        if df is not None:
            df['source_file'] = os.path.basename(file_path)
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("❌ No data could be loaded from any file!")
        return None
    
    # Find common columns
    common_columns = set(all_dataframes[0].columns)
    for df in all_dataframes[1:]:
        common_columns = common_columns.intersection(set(df.columns))
    
    print(f"\n🔗 Common columns across all files: {sorted(list(common_columns))}")
    
    # Combine dataframes using common columns
    if len(common_columns) < 3:
        print("⚠️ Very few common columns found. Combining all unique columns...")
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    else:
        aligned_dfs = [df[list(common_columns)] for df in all_dataframes]
        combined_df = pd.concat(aligned_dfs, ignore_index=True)
    
    print(f"\n✅ Successfully combined {len(all_dataframes)} files")
    print(f"   Total records: {len(combined_df)}")
    print(f"   Final columns: {list(combined_df.columns)}")
    
    return combined_df

# Run inspection
print("🔍 Inspecting NPZ files in current directory...")
inspect_all_npz_files()