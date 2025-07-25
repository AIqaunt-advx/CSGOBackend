# Cell 2.5: Data Quality Checker
# Run this to analyze data quality and NULL patterns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_data_quality(df):
    """Comprehensive data quality analysis"""
    print("🔍 Data Quality Analysis")
    print("=" * 50)
    
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📊 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 1. NULL Value Analysis
    print(f"\n🚫 NULL Value Analysis:")
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    
    if null_counts.sum() > 0:
        null_summary = pd.DataFrame({
            'Column': null_counts.index,
            'NULL_Count': null_counts.values,
            'NULL_Percentage': null_percentages.values
        }).sort_values('NULL_Count', ascending=False)
        
        print(null_summary[null_summary['NULL_Count'] > 0])
        
        # Visualize NULL patterns
        if null_counts.sum() > 0:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            null_cols = null_summary[null_summary['NULL_Count'] > 0]
            plt.bar(range(len(null_cols)), null_cols['NULL_Count'])
            plt.xticks(range(len(null_cols)), null_cols['Column'], rotation=45)
            plt.title('NULL Counts by Column')
            plt.ylabel('Count')
            
            plt.subplot(1, 2, 2)
            plt.bar(range(len(null_cols)), null_cols['NULL_Percentage'])
            plt.xticks(range(len(null_cols)), null_cols['Column'], rotation=45)
            plt.title('NULL Percentages by Column')
            plt.ylabel('Percentage (%)')
            
            plt.tight_layout()
            plt.show()
    else:
        print("✅ No NULL values found!")
    
    # 2. Data Type Analysis
    print(f"\n📋 Data Types:")
    dtype_summary = df.dtypes.value_counts()
    print(dtype_summary)
    
    # 3. Unique Value Analysis
    print(f"\n🔢 Unique Values per Column:")
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        print(f"  {col:20}: {unique_count:6} unique ({unique_pct:5.1f}%)")
    
    # 4. Numeric Column Statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Numeric Column Statistics:")
        print(df[numeric_cols].describe())
        
        # Check for infinite values
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            print(f"\n⚠️ Infinite Values Found:")
            for col, count in inf_counts.items():
                print(f"  {col}: {count} infinite values")
        else:
            print(f"\n✅ No infinite values found in numeric columns")
    
    # 5. String Column Analysis
    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        print(f"\n📝 String Column Analysis:")
        for col in string_cols:
            print(f"\n  {col}:")
            value_counts = df[col].value_counts().head(10)
            print(f"    Top values: {list(value_counts.index)}")
            
            # Check for potential NULL representations
            potential_nulls = ['NULL', 'null', 'None', 'none', 'NaN', 'nan', '', ' ', 'N/A', 'n/a']
            found_nulls = []
            for null_val in potential_nulls:
                if null_val in df[col].values:
                    count = (df[col] == null_val).sum()
                    found_nulls.append(f"{null_val}({count})")
            
            if found_nulls:
                print(f"    ⚠️ Potential NULL representations: {', '.join(found_nulls)}")
    
    # 6. Row Completeness Analysis
    print(f"\n📋 Row Completeness Analysis:")
    complete_rows = df.dropna().shape[0]
    incomplete_rows = len(df) - complete_rows
    
    print(f"  Complete rows: {complete_rows} ({complete_rows/len(df)*100:.1f}%)")
    print(f"  Incomplete rows: {incomplete_rows} ({incomplete_rows/len(df)*100:.1f}%)")
    
    if incomplete_rows > 0:
        # Analyze patterns of missing data
        null_pattern = df.isnull()
        null_combinations = null_pattern.value_counts().head(10)
        print(f"\n  Top NULL patterns:")
        for pattern, count in null_combinations.items():
            if any(pattern):  # Only show patterns with at least one NULL
                null_cols = [col for col, is_null in zip(df.columns, pattern) if is_null]
                print(f"    {count:4} rows missing: {null_cols}")
    
    # 7. Recommendations
    print(f"\n💡 Data Quality Recommendations:")
    
    total_null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    if total_null_pct > 50:
        print("  🔴 CRITICAL: >50% of data is NULL. Consider data source quality.")
    elif total_null_pct > 20:
        print("  🟡 WARNING: >20% of data is NULL. Review data collection process.")
    elif total_null_pct > 5:
        print("  🟡 CAUTION: >5% of data is NULL. Monitor data quality.")
    else:
        print("  🟢 GOOD: <5% NULL data. Quality looks acceptable.")
    
    if incomplete_rows > len(df) * 0.8:
        print("  🔴 CRITICAL: >80% of rows have missing data. Training may be difficult.")
    elif incomplete_rows > len(df) * 0.5:
        print("  🟡 WARNING: >50% of rows have missing data. Consider imputation strategies.")
    
    if len(df) < 1000:
        print("  🟡 WARNING: Small dataset (<1000 rows). Consider collecting more data.")
    
    return {
        'total_rows': len(df),
        'total_nulls': df.isnull().sum().sum(),
        'null_percentage': total_null_pct,
        'complete_rows': complete_rows,
        'incomplete_rows': incomplete_rows,
        'numeric_columns': len(numeric_cols),
        'string_columns': len(string_cols)
    }

def recommend_cleaning_strategy(df):
    """Recommend data cleaning strategy based on analysis"""
    print(f"\n🛠️ Recommended Cleaning Strategy:")
    
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        print("  ✅ No cleaning needed - data is already clean!")
        return
    
    # Strategy 1: Drop columns with too many NULLs
    high_null_cols = null_counts[null_counts > len(df) * 0.7]
    if len(high_null_cols) > 0:
        print(f"  1. 🗑️ Consider dropping columns with >70% NULL:")
        for col, count in high_null_cols.items():
            print(f"     - {col}: {count/len(df)*100:.1f}% NULL")
    
    # Strategy 2: Handle rows with NULLs
    complete_rows = df.dropna().shape[0]
    if complete_rows > len(df) * 0.3:
        print(f"  2. ✂️ Drop rows with ANY NULL (keeps {complete_rows} rows, {complete_rows/len(df)*100:.1f}%)")
    else:
        print(f"  2. ⚠️ Dropping NULL rows would remove too much data ({len(df)-complete_rows} rows)")
        print(f"     Consider imputation strategies instead")
    
    # Strategy 3: Imputation suggestions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if null_counts[col] > 0 and null_counts[col] < len(df) * 0.3:
            print(f"  3. 🔧 {col}: Consider median imputation ({null_counts[col]} NULLs)")
    
    print(f"\n  📊 Final expected data size after cleaning: ~{complete_rows} rows")

# Example usage - run this after loading your data
# analyze_data_quality(df)
# recommend_cleaning_strategy(df)