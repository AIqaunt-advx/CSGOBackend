# Cell 2.7: Data Extractor for Complex NPZ Structures
# Run this to extract actual trading data from complex NPZ structures

import pandas as pd
import numpy as np
import json

def extract_trading_data(df):
    """Extract actual trading data from complex DataFrame structures"""
    print("🔧 Extracting trading data from complex structure...")
    print(f"   Input shape: {df.shape}")
    print(f"   Input columns: {list(df.columns)}")
    
    # Strategy 1: If we have a 'data' column with complex data
    if 'data' in df.columns and len(df.columns) <= 3:  # Likely all data is in 'data' column
        print("   Strategy 1: Extracting from 'data' column...")
        
        # Look at the first few entries to understand structure
        sample_data = df['data'].iloc[0]
        print(f"   Sample data type: {type(sample_data)}")
        
        if isinstance(sample_data, (list, tuple)):
            print(f"   Data is list/tuple of length: {len(sample_data)}")
            
            # Try to convert list/tuple to columns
            try:
                # Assume it's [timestamp, price, onSaleQuantity, seekQuantity, seekPrice, ...]
                data_arrays = []
                max_length = 0
                
                for i, row_data in enumerate(df['data']):
                    if isinstance(row_data, (list, tuple)) and len(row_data) > 0:
                        data_arrays.append(row_data)
                        max_length = max(max_length, len(row_data))
                    else:
                        print(f"   ⚠️ Row {i} has invalid data: {type(row_data)}")
                
                if data_arrays and max_length > 0:
                    # Convert to DataFrame
                    data_matrix = []
                    for row_data in data_arrays:
                        # Pad shorter rows with NaN
                        padded_row = list(row_data) + [np.nan] * (max_length - len(row_data))
                        data_matrix.append(padded_row[:max_length])
                    
                    # Create column names based on common trading data structure
                    if max_length >= 5:
                        column_names = ['timestamp', 'price', 'onSaleQuantity', 'seekQuantity', 'seekPrice']
                        if max_length > 5:
                            column_names.extend([f'extra_{i}' for i in range(5, max_length)])
                    else:
                        column_names = [f'col_{i}' for i in range(max_length)]
                    
                    extracted_df = pd.DataFrame(data_matrix, columns=column_names[:max_length])
                    print(f"   ✅ Extracted to {extracted_df.shape} with columns: {list(extracted_df.columns)}")
                    
                    return extracted_df
                    
            except Exception as e:
                print(f"   ❌ List/tuple extraction failed: {e}")
        
        elif isinstance(sample_data, dict):
            print(f"   Data is dictionary with keys: {list(sample_data.keys())}")
            
            # Try to extract from dictionary
            try:
                all_records = []
                for row_data in df['data']:
                    if isinstance(row_data, dict):
                        all_records.append(row_data)
                
                if all_records:
                    extracted_df = pd.DataFrame(all_records)
                    print(f"   ✅ Extracted to {extracted_df.shape} with columns: {list(extracted_df.columns)}")
                    return extracted_df
                    
            except Exception as e:
                print(f"   ❌ Dictionary extraction failed: {e}")
        
        elif isinstance(sample_data, str):
            print(f"   Data is string, sample: {sample_data[:100]}...")
            
            # Try to parse as JSON
            try:
                all_records = []
                for row_data in df['data']:
                    if isinstance(row_data, str):
                        try:
                            parsed = json.loads(row_data)
                            all_records.append(parsed)
                        except:
                            # Try eval as last resort (dangerous but sometimes necessary)
                            try:
                                parsed = eval(row_data)
                                all_records.append(parsed)
                            except:
                                continue
                
                if all_records:
                    extracted_df = pd.DataFrame(all_records)
                    print(f"   ✅ Extracted to {extracted_df.shape} with columns: {list(extracted_df.columns)}")
                    return extracted_df
                    
            except Exception as e:
                print(f"   ❌ String parsing failed: {e}")
        
        elif np.isscalar(sample_data) or isinstance(sample_data, np.ndarray):
            print(f"   Data is scalar/array: {sample_data}")
            
            # If it's numeric data, maybe it's a time series
            try:
                if df['data'].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    # Create a simple time series DataFrame
                    extracted_df = pd.DataFrame({
                        'value': df['data'],
                        'index': range(len(df))
                    })
                    print(f"   ✅ Created time series with {extracted_df.shape}")
                    return extracted_df
            except Exception as e:
                print(f"   ❌ Scalar extraction failed: {e}")
    
    # Strategy 2: Data is already in separate columns
    print("   Strategy 2: Using existing column structure...")
    
    # Remove non-data columns
    exclude_cols = ['source_file', 'file_path', 'index']
    data_cols = [col for col in df.columns if col not in exclude_cols]
    
    if data_cols:
        extracted_df = df[data_cols].copy()
        print(f"   ✅ Using existing structure: {extracted_df.shape} with columns: {list(extracted_df.columns)}")
        return extracted_df
    
    # Strategy 3: Last resort - return as is
    print("   Strategy 3: Returning original data...")
    return df

def smart_column_mapping(df):
    """Apply smart column name mapping based on data patterns"""
    print("🔄 Applying smart column mapping...")
    
    # Common patterns in trading data
    mappings = {
        # Timestamp patterns
        'timestamp': ['timestamp', 'time', 'ts', 'date', 'datetime'],
        # Price patterns  
        'price': ['price', 'current_price', 'market_price', 'sell_price'],
        # Volume patterns
        'onSaleQuantity': ['on_sale_quantity', 'supply', 'available', 'stock', 'current_sales_volume'],
        'seekQuantity': ['seek_quantity', 'demand', 'wanted', 'buy_orders', 'wanted_sales_volume'],
        'seekPrice': ['seek_price', 'bid_price', 'buy_price', 'wanted_price'],
        # Transaction patterns
        'transactionAmount': ['transaction_amount', 'volume', 'success_sales_volume', 'traded_amount'],
        'transcationNum': ['transaction_num', 'trade_count', 'num_trades'],
        'surviveNum': ['survive_num', 'remaining', 'leftover']
    }
    
    # Apply mappings
    renamed_cols = {}
    for standard_name, variations in mappings.items():
        for variation in variations:
            if variation in df.columns and standard_name not in df.columns:
                renamed_cols[variation] = standard_name
                break
    
    if renamed_cols:
        df = df.rename(columns=renamed_cols)
        print(f"   ✅ Renamed columns: {renamed_cols}")
    
    # If we have numeric columns without clear names, try to infer
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    unnamed_numeric = [col for col in numeric_cols if col.startswith(('col_', 'extra_', 'data'))]
    
    if len(unnamed_numeric) >= 2:
        print(f"   🔍 Found {len(unnamed_numeric)} unnamed numeric columns, inferring names...")
        
        # Common pattern: [timestamp, price, supply, demand, bid_price]
        name_suggestions = ['timestamp', 'price', 'onSaleQuantity', 'seekQuantity', 'seekPrice']
        
        rename_map = {}
        for i, col in enumerate(unnamed_numeric[:len(name_suggestions)]):
            if name_suggestions[i] not in df.columns:
                rename_map[col] = name_suggestions[i]
        
        if rename_map:
            df = df.rename(columns=rename_map)
            print(f"   ✅ Inferred column names: {rename_map}")
    
    return df

# Example usage function
def process_complex_data(df):
    """Complete processing pipeline for complex data structures"""
    print("🚀 Processing complex data structure...")
    
    # Step 1: Extract actual data
    extracted_df = extract_trading_data(df)
    
    # Step 2: Apply smart column mapping
    mapped_df = smart_column_mapping(extracted_df)
    
    # Step 3: Basic data cleaning
    print("🧹 Basic data cleaning...")
    
    # Convert timestamp if it exists and is numeric
    if 'timestamp' in mapped_df.columns:
        try:
            if mapped_df['timestamp'].dtype in [np.int64, np.float64]:
                mapped_df['datetime'] = pd.to_datetime(mapped_df['timestamp'], unit='s', errors='coerce')
                print("   ✅ Converted timestamp to datetime")
        except Exception as e:
            print(f"   ⚠️ Timestamp conversion failed: {e}")
    
    # Ensure numeric columns are properly typed
    numeric_candidates = ['price', 'onSaleQuantity', 'seekQuantity', 'seekPrice', 'transactionAmount']
    for col in numeric_candidates:
        if col in mapped_df.columns:
            mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce')
    
    print(f"✅ Final processed shape: {mapped_df.shape}")
    print(f"✅ Final columns: {list(mapped_df.columns)}")
    
    return mapped_df