# Cell 2: Data Processing Class
# Copy this entire cell into your Jupyter notebook

import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CS2SkinDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = ['onSaleQuantity', 'seekQuantity', 'price', 'seekPrice']
        self.target_column = 'transactionAmount'
        self.skin_features = []  # Store features for each skin
        
    def load_single_npz(self, file_path):
        """Load single NPZ file as one data point for one skin"""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Extract skin name from file path
            skin_name = file_path.split('/')[-1].replace('.npz', '') if '/' in file_path else file_path.replace('.npz', '')
            
            # Create single row dictionary for this skin
            skin_data = {'skin_name': skin_name, 'source_file': file_path}
            
            # Process each key in NPZ file
            for key in data.keys():
                array_data = data[key]
                
                if array_data.ndim == 0:  # Scalar value
                    skin_data[key] = float(array_data)
                elif array_data.ndim == 1:  # 1D array - use statistical features
                    if len(array_data) > 0:
                        skin_data[f'{key}_mean'] = np.mean(array_data)
                        skin_data[f'{key}_std'] = np.std(array_data)
                        skin_data[f'{key}_min'] = np.min(array_data)
                        skin_data[f'{key}_max'] = np.max(array_data)
                        skin_data[f'{key}_median'] = np.median(array_data)
                        skin_data[f'{key}_sum'] = np.sum(array_data)
                        skin_data[f'{key}_count'] = len(array_data)
                        # Use the latest/last value as current state
                        skin_data[key] = array_data[-1]
                    else:
                        skin_data[key] = 0
                elif array_data.ndim == 2:  # 2D array - flatten or use stats
                    if array_data.size > 0:
                        flat_data = array_data.flatten()
                        skin_data[f'{key}_mean'] = np.mean(flat_data)
                        skin_data[f'{key}_std'] = np.std(flat_data)
                        skin_data[f'{key}_sum'] = np.sum(flat_data)
                        skin_data[key] = flat_data[-1] if len(flat_data) > 0 else 0
                    else:
                        skin_data[key] = 0
                else:
                    # Higher dimensional arrays - use summary stats
                    if array_data.size > 0:
                        flat_data = array_data.flatten()
                        skin_data[f'{key}_mean'] = np.mean(flat_data)
                        skin_data[f'{key}_sum'] = np.sum(flat_data)
                        skin_data[key] = flat_data[-1] if len(flat_data) > 0 else 0
                    else:
                        skin_data[key] = 0
            
            # Common column name mappings for CS2 trading data
            column_mappings = {
                'on_sale_quantity': 'onSaleQuantity',
                'onsale_quantity': 'onSaleQuantity',
                'current_sales_volume': 'onSaleQuantity',
                'seek_quantity': 'seekQuantity',
                'wanted_sales_volume': 'seekQuantity',
                'seek_price': 'seekPrice',
                'transaction_amount': 'transactionAmount',
                'success_sales_volume': 'transactionAmount',
                'transaction_num': 'transcationNum',
                'survive_num': 'surviveNum'
            }
            
            # Apply column mappings
            mapped_data = {}
            for key, value in skin_data.items():
                mapped_key = column_mappings.get(key, key)
                mapped_data[mapped_key] = value
            
            # Fill null values with 0 for transaction metrics
            for col in ['transactionAmount', 'transcationNum', 'surviveNum']:
                if col not in mapped_data:
                    mapped_data[col] = 0
            
            print(f"✓ Processed skin: {skin_name} ({len(mapped_data)} features)")
            return mapped_data
            
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
            return None
    
    def load_multiple_npz_files(self, file_pattern="*.npz", max_files=None):
        """Load multiple NPZ files, each as one data point for one skin"""
        npz_files = glob.glob(file_pattern)
        if max_files:
            npz_files = npz_files[:max_files]
        
        print(f"Found {len(npz_files)} NPZ files to process")
        
        all_skin_data = []
        successful_loads = 0
        
        for i, file_path in enumerate(npz_files):
            if i % 1000 == 0:
                print(f"Processing file {i+1}/{len(npz_files)}...")
            
            skin_data = self.load_single_npz(file_path)
            if skin_data is not None:
                all_skin_data.append(skin_data)
                successful_loads += 1
        
        if all_skin_data:
            # Create DataFrame from list of dictionaries
            df = pd.DataFrame(all_skin_data)
            
            # Fill missing values with 0 for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            print(f"✓ Successfully loaded {successful_loads}/{len(npz_files)} skins")
            print(f"✓ Combined dataset: {df.shape[0]} skins, {df.shape[1]} features")
            print(f"✓ Sample features: {list(df.columns[:10])}")
            
            return df
        else:
            print("✗ No valid skin data found")
            return None
    
    def inspect_npz_structure(self, file_path):
        """Inspect the structure of an NPZ file"""
        try:
            data = np.load(file_path, allow_pickle=True)
            print(f"\n=== NPZ File Structure: {file_path} ===")
            
            for key in data.keys():
                array = data[key]
                print(f"{key:20} | Shape: {str(array.shape):15} | Dtype: {str(array.dtype)}")
                
                # Show sample values for small arrays
                if array.size <= 10:
                    print(f"{'':20} | Values: {array}")
                elif array.ndim == 1 and array.size > 0:
                    print(f"{'':20} | Sample: {array[:5]}...")
                elif array.ndim == 2 and array.size > 0:
                    print(f"{'':20} | Sample: {array[:3, :min(3, array.shape[1])]}...")
            
            data.close()
            
        except Exception as e:
            print(f"✗ Error inspecting {file_path}: {e}")
    
    def explore_data(self, df):
        """Basic data exploration and visualization"""
        print("\n=== Data Overview ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for datetime column
        if 'datetime' in df.columns:
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        elif 'timestamp' in df.columns:
            print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print("No datetime/timestamp column found")
        
        print("\n=== Column Info ===")
        print(df.info())
        
        print("\n=== Statistical Summary ===")
        print(df.describe())
        
        print("\n=== Missing Values ===")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Time series (if datetime available)
        if 'datetime' in df.columns and 'price' in df.columns:
            axes[0,0].plot(df['datetime'], df['price'])
            axes[0,0].set_title('Price Over Time')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Price')
        elif 'timestamp' in df.columns and 'price' in df.columns:
            axes[0,0].plot(df['timestamp'], df['price'])
            axes[0,0].set_title('Price Over Time')
            axes[0,0].set_xlabel('Timestamp')
            axes[0,0].set_ylabel('Price')
        else:
            # Show first numeric column distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].hist(bins=50, ax=axes[0,0])
                axes[0,0].set_title(f'Distribution of {numeric_cols[0]}')
            else:
                axes[0,0].text(0.5, 0.5, 'No numeric data for plotting', ha='center', va='center')
                axes[0,0].set_title('No Data Available')
        
        # Plot 2: Volume distributions (if available)
        volume_cols = [col for col in ['onSaleQuantity', 'seekQuantity'] if col in df.columns]
        if len(volume_cols) >= 2:
            axes[0,1].hist(df[volume_cols[0]], bins=50, alpha=0.7, label=volume_cols[0])
            axes[0,1].hist(df[volume_cols[1]], bins=50, alpha=0.7, label=volume_cols[1])
            axes[0,1].set_title('Volume Distributions')
            axes[0,1].set_xlabel('Quantity')
            axes[0,1].legend()
        elif len(volume_cols) == 1:
            df[volume_cols[0]].hist(bins=50, ax=axes[0,1])
            axes[0,1].set_title(f'Distribution of {volume_cols[0]}')
        else:
            # Show distribution of second numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                df[numeric_cols[1]].hist(bins=50, ax=axes[0,1])
                axes[0,1].set_title(f'Distribution of {numeric_cols[1]}')
            else:
                axes[0,1].text(0.5, 0.5, 'No volume data', ha='center', va='center')
                axes[0,1].set_title('Volume Data (Not Available)')
        
        # Plot 3: Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
            axes[1,0].set_title('Correlation Matrix')
        else:
            axes[1,0].text(0.5, 0.5, 'Need 2+ numeric columns', ha='center', va='center')
            axes[1,0].set_title('Correlation (Not Available)')
        
        # Plot 4: Transaction amount over time (if available)
        if 'transactionAmount' in df.columns and df['transactionAmount'].sum() > 0:
            if 'datetime' in df.columns:
                axes[1,1].plot(df['datetime'], df['transactionAmount'])
                axes[1,1].set_xlabel('Date')
            elif 'timestamp' in df.columns:
                axes[1,1].plot(df['timestamp'], df['transactionAmount'])
                axes[1,1].set_xlabel('Timestamp')
            else:
                axes[1,1].plot(df['transactionAmount'])
                axes[1,1].set_xlabel('Index')
            axes[1,1].set_title('Transaction Amount Over Time')
            axes[1,1].set_ylabel('Transaction Amount')
        else:
            # Show any remaining numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            remaining_cols = [col for col in numeric_cols if col not in ['price', volume_cols[0] if volume_cols else '']]
            if remaining_cols:
                df[remaining_cols[0]].plot(ax=axes[1,1])
                axes[1,1].set_title(f'{remaining_cols[0]} Over Index')
            else:
                axes[1,1].text(0.5, 0.5, 'No Transaction Data', ha='center', va='center')
                axes[1,1].set_title('Transaction Amount (No Data)')
        
        plt.tight_layout()
        plt.show()
        
        # Print sample data
        print("\n=== Sample Data ===")
        print(df.head(10))
    
    def create_features(self, df):
        """Engineer features for skin-level data (each row is one skin)"""
        df_features = df.copy()
        
        print(f"🔧 Creating features for {len(df_features)} skins...")
        
        # Basic ratio features (only if columns exist)
        if 'seekQuantity' in df_features.columns and 'onSaleQuantity' in df_features.columns:
            df_features['demand_ratio'] = df_features['seekQuantity'] / (df_features['onSaleQuantity'] + 1)
            df_features['supply_demand_diff'] = df_features['onSaleQuantity'] - df_features['seekQuantity']
            df_features['market_balance'] = (df_features['onSaleQuantity'] - df_features['seekQuantity']) / (df_features['onSaleQuantity'] + df_features['seekQuantity'] + 1)
        
        if 'price' in df_features.columns and 'seekPrice' in df_features.columns:
            df_features['price_spread'] = df_features['price'] - df_features['seekPrice']
            df_features['price_spread_pct'] = df_features['price_spread'] / (df_features['seekPrice'] + 0.01)
            df_features['price_premium'] = df_features['price'] / (df_features['seekPrice'] + 0.01)
        
        # Volatility-based features (using _std columns if available)
        volatility_features = []
        for base_col in ['price', 'onSaleQuantity', 'seekQuantity']:
            std_col = f'{base_col}_std'
            mean_col = f'{base_col}_mean'
            if std_col in df_features.columns and mean_col in df_features.columns:
                # Coefficient of variation
                df_features[f'{base_col}_cv'] = df_features[std_col] / (df_features[mean_col] + 0.01)
                volatility_features.append(f'{base_col}_cv')
        
        # Range-based features (using min/max columns if available)
        for base_col in ['price', 'onSaleQuantity', 'seekQuantity']:
            min_col = f'{base_col}_min'
            max_col = f'{base_col}_max'
            mean_col = f'{base_col}_mean'
            if min_col in df_features.columns and max_col in df_features.columns:
                df_features[f'{base_col}_range'] = df_features[max_col] - df_features[min_col]
                if mean_col in df_features.columns:
                    df_features[f'{base_col}_range_norm'] = df_features[f'{base_col}_range'] / (df_features[mean_col] + 0.01)
        
        # Transaction efficiency features
        if 'transactionAmount' in df_features.columns:
            if 'seekQuantity' in df_features.columns:
                df_features['transaction_rate'] = df_features['transactionAmount'] / (df_features['seekQuantity'] + 1)
            if 'onSaleQuantity' in df_features.columns:
                df_features['sell_through_rate'] = df_features['transactionAmount'] / (df_features['onSaleQuantity'] + 1)
        
        # Market activity features
        activity_indicators = []
        for col in ['transcationNum', 'surviveNum']:
            if col in df_features.columns:
                activity_indicators.append(col)
        
        if len(activity_indicators) >= 2:
            df_features['activity_ratio'] = df_features['transcationNum'] / (df_features['surviveNum'] + 1)
        
        # Price tier classification (if price exists)
        if 'price' in df_features.columns:
            price_percentiles = df_features['price'].quantile([0.25, 0.5, 0.75])
            df_features['price_tier'] = pd.cut(
                df_features['price'], 
                bins=[-np.inf, price_percentiles[0.25], price_percentiles[0.5], price_percentiles[0.75], np.inf],
                labels=['low', 'medium', 'high', 'premium']
            )
            # Convert to numeric
            df_features['price_tier_num'] = df_features['price_tier'].cat.codes
        
        # Market position features (relative to all skins)
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in ['price', 'onSaleQuantity', 'seekQuantity', 'transactionAmount']:
            if col in numeric_cols:
                df_features[f'{col}_percentile'] = df_features[col].rank(pct=True)
                df_features[f'{col}_zscore'] = (df_features[col] - df_features[col].mean()) / (df_features[col].std() + 0.01)
        
        # Log transformations for skewed data (only if columns exist and positive)
        for col in ['onSaleQuantity', 'seekQuantity', 'price', 'transactionAmount']:
            if col in df_features.columns:
                # Only log transform positive values
                positive_mask = df_features[col] > 0
                if positive_mask.sum() > 0:
                    df_features[f'{col}_log'] = 0
                    df_features.loc[positive_mask, f'{col}_log'] = np.log1p(df_features.loc[positive_mask, col])
        
        # Fill any remaining NaN values
        df_features = df_features.fillna(0)
        
        # Remove any infinite values
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_cols] = df_features[numeric_cols].replace([np.inf, -np.inf], 0)
        
        print(f"✓ Created {df_features.shape[1] - df.shape[1]} new features")
        print(f"✓ Total features: {df_features.shape[1]}")
        
        return df_features
    
    def clean_null_values(self, df):
        """Clean NULL values from the dataset with intelligent strategy"""
        print(f"🧹 Cleaning NULL values...")
        print(f"   Original shape: {df.shape}")
        
        # Check for different types of NULL values
        null_patterns = [
            df.isnull(),  # Standard pandas NULL
            df.isna(),    # Standard pandas NA
            df == 'NULL', # String 'NULL'
            df == 'null', # String 'null'
            df == 'None', # String 'None'
            df == '',     # Empty strings
            df == ' ',    # Space strings
        ]
        
        # Combine all null patterns
        is_null = pd.DataFrame(False, index=df.index, columns=df.columns)
        for pattern in null_patterns:
            try:
                is_null = is_null | pattern
            except:
                continue
        
        # Report NULL statistics
        null_counts = is_null.sum()
        if null_counts.sum() > 0:
            print(f"   NULL values found:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"     {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # INTELLIGENT CLEANING STRATEGY
        
        # Step 1: Drop columns that are mostly NULL (>95%)
        high_null_cols = []
        for col, count in null_counts.items():
            if count > len(df) * 0.95:
                high_null_cols.append(col)
        
        if high_null_cols:
            print(f"   🗑️ Dropping columns with >95% NULL: {high_null_cols}")
            df = df.drop(columns=high_null_cols)
            is_null = is_null.drop(columns=high_null_cols)
            null_counts = is_null.sum()
        
        # Step 2: Fill transaction columns with 0 (they're often legitimately 0)
        transaction_cols = ['transactionAmount', 'transcationNum', 'surviveNum']
        for col in transaction_cols:
            if col in df.columns:
                null_count_before = is_null[col].sum()
                df[col] = df[col].fillna(0)
                is_null[col] = False  # Mark as no longer null
                if null_count_before > 0:
                    print(f"   🔧 Filled {col} NULLs with 0: {null_count_before} values")
        
        # Step 3: Only remove rows with NULLs in ESSENTIAL columns
        essential_cols = ['timestamp', 'price', 'onSaleQuantity', 'seekQuantity', 'seekPrice']
        available_essential = [col for col in essential_cols if col in df.columns]
        
        if available_essential:
            print(f"   🎯 Checking essential columns for NULLs: {available_essential}")
            essential_nulls = is_null[available_essential].any(axis=1)
            clean_df = df[~essential_nulls].copy()
            
            removed_rows = len(df) - len(clean_df)
            print(f"   Removed {removed_rows} rows with NULLs in essential columns ({removed_rows/len(df)*100:.1f}%)")
        else:
            print("   ⚠️ No essential columns found, keeping all rows")
            clean_df = df.copy()
        
        print(f"   Clean shape: {clean_df.shape}")
        
        if len(clean_df) == 0:
            print("❌ All rows have NULLs in essential columns!")
            print("💡 Trying more lenient strategy...")
            
            # Fallback: Only require non-null values in at least 2 numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                # Keep rows that have at least 2 non-null numeric values
                numeric_null_counts = is_null[numeric_cols].sum(axis=1)
                max_allowed_nulls = len(numeric_cols) - 2
                clean_df = df[numeric_null_counts <= max_allowed_nulls].copy()
                
                removed_rows = len(df) - len(clean_df)
                print(f"   Fallback: Removed {removed_rows} rows, kept rows with ≥2 numeric values")
                print(f"Fallback clean shape: {clean_df.shape}")
        
        if len(clean_df) == 0:
            raise ValueError("❌ All rows contain too many NULL values! Cannot proceed with training.")
        
        if len(clean_df) < len(df) * 0.1:
            print("⚠️ Warning: Removed >90% of data due to NULL values. Check data quality!")
        
        return clean_df
    
    def prepare_data(self, df, target_col='transactionAmount', test_size=0.2):
        """Prepare skin-level data for training (each row is one skin)"""
        print(f"📊 Input data shape: {df.shape[0]} skins, {df.shape[1]} features")
        print(f"📊 Sample columns: {list(df.columns[:10])}")
        
        # Step 1: Clean NULL values FIRST
        df_clean = self.clean_null_values(df)
        
        # Step 2: Create features
        df_processed = self.create_features(df_clean)
        print(f"📊 After feature engineering: {df_processed.shape}")
        
        # Step 3: Handle target column
        if target_col not in df_processed.columns or df_processed[target_col].sum() == 0:
            print(f"⚠️ Warning: {target_col} not available or all zeros. Creating optimized target.")
            
            # Create transaction success rate as target (0-1 range)
            if 'seekQuantity' in df_processed.columns and 'transactionAmount' in df_processed.columns:
                # Transaction rate = transactionAmount / seekQuantity (capped at 1.0)
                df_processed['transaction_success_rate'] = np.minimum(
                    1.0, 
                    df_processed['transactionAmount'] / (df_processed['seekQuantity'] + 1)
                )
                target_col = 'transaction_success_rate'
                print("✓ Created transaction_success_rate target (transactionAmount/seekQuantity)")
            
            elif 'seekQuantity' in df_processed.columns and 'onSaleQuantity' in df_processed.columns:
                # Market efficiency based on supply/demand balance
                df_processed['market_efficiency'] = (
                    df_processed['seekQuantity'] / 
                    (df_processed['onSaleQuantity'] + df_processed['seekQuantity'] + 1)
                )
                target_col = 'market_efficiency'
                print("✓ Created market_efficiency target based on supply/demand")
            
            else:
                # Use available numeric columns to create a composite target
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                non_meta_cols = [col for col in numeric_cols if not col.startswith('skin_') and col != 'source_file']
                
                if len(non_meta_cols) >= 2:
                    # Create composite score from top features
                    col1, col2 = non_meta_cols[0], non_meta_cols[1]
                    df_processed['composite_score'] = (
                        (df_processed[col1] - df_processed[col1].min()) / (df_processed[col1].max() - df_processed[col1].min() + 0.01) * 0.6 +
                        (df_processed[col2] - df_processed[col2].min()) / (df_processed[col2].max() - df_processed[col2].min() + 0.01) * 0.4
                    )
                    target_col = 'composite_score'
                    print(f"✓ Created composite_score target using {col1} and {col2}")
                else:
                    raise ValueError("❌ Insufficient data to create meaningful target variable")
        
        # Step 4: Select features for training - exclude metadata and target
        exclude_cols = [
            'skin_name', 'source_file', target_col, 
            'price_tier'  # Categorical column
        ]
        
        # Get all numeric columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        available_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Prioritize engineered features
        priority_features = [
            'demand_ratio', 'supply_demand_diff', 'market_balance',
            'price_spread', 'price_spread_pct', 'price_premium',
            'transaction_rate', 'sell_through_rate', 'activity_ratio',
            'price_percentile', 'onSaleQuantity_percentile', 'seekQuantity_percentile',
            'price_zscore', 'onSaleQuantity_zscore', 'seekQuantity_zscore'
        ]
        
        # Add priority features that exist
        final_features = []
        for feat in priority_features:
            if feat in available_features:
                final_features.append(feat)
        
        # Add remaining features
        for feat in available_features:
            if feat not in final_features:
                final_features.append(feat)
        
        # Limit to reasonable number of features to avoid overfitting
        if len(final_features) > 50:
            final_features = final_features[:50]
            print(f"⚠️ Limited to top 50 features to prevent overfitting")
        
        print(f"✓ Selected {len(final_features)} features for training")
        print(f"✓ Top features: {final_features[:10]}")
        
        if len(final_features) == 0:
            raise ValueError("No suitable features found for training!")
        
        X = df_processed[final_features]
        y = df_processed[target_col]
        
        # Step 5: Final data validation
        print(f"🔍 Final data validation...")
        
        # Check for any remaining NaN/inf values
        X_nulls = X.isnull().sum().sum()
        y_nulls = y.isnull().sum()
        X_infs = np.isinf(X).sum().sum()
        y_infs = np.isinf(y).sum() if np.issubdtype(y.dtype, np.number) else 0
        
        print(f"   X nulls: {X_nulls}, X infs: {X_infs}")
        print(f"   y nulls: {y_nulls}, y infs: {y_infs}")
        
        if X_nulls > 0 or y_nulls > 0 or X_infs > 0 or y_infs > 0:
            print("⚠️ Found remaining NULL/inf values, cleaning...")
            
            # Create mask for valid rows
            valid_mask = (
                ~X.isnull().any(axis=1) & 
                ~y.isnull() & 
                ~np.isinf(X).any(axis=1) &
                ~np.isinf(y)
            )
            
            X = X[valid_mask]
            y = y[valid_mask]
            
            print(f"   Final clean shape: X={X.shape}, y={len(y)}")
        
        # Step 6: Ensure we have enough data
        if len(X) < 100:
            raise ValueError(f"❌ Too few samples after cleaning: {len(X)}. Need at least 100 skins for training.")
        
        print(f"✓ Training samples: {len(X)} skins")
        print(f"✓ Features: {len(final_features)}")
        print(f"✓ Target column: {target_col}")
        print(f"✓ Target stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
        
        # Step 7: Random split (not chronological since each skin is independent)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        print(f"✓ Train set: {X_train.shape[0]} skins")
        print(f"✓ Test set: {X_test.shape[0]} skins")
        
        # Step 8: Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Final validation - check for any NaN in scaled data
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            print("❌ NaN values found in scaled data! This shouldn't happen.")
            raise ValueError("Scaling produced NaN values")
        
        print("✅ Data preparation completed successfully!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, final_features

# Initialize the processor
processor = CS2SkinDataProcessor()
print("✓ Data processor initialized")