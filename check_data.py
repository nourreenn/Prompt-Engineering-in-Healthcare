import pandas as pd
import os

def check_preprocessed_data():
    # Check unified dataset
    print("Checking unified dataset...")
    unified_df = pd.read_csv('preprocessed_data/unified_dataset.csv')
    print("\nUnified Dataset Info:")
    print("Shape:", unified_df.shape)
    print("\nColumns:", unified_df.columns.tolist())
    print("\nSample data (first 2 rows):")
    print(unified_df.head(2).to_string())
    
    # Check individual datasets
    print("\nChecking individual datasets...")
    for file in os.listdir('preprocessed_data'):
        if file.endswith('_preprocessed.csv') and file != 'unified_dataset.csv':
            print(f"\n{file}:")
            df = pd.read_csv(f'preprocessed_data/{file}')
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print("Sample data (first row):")
            print(df.head(1).to_string())

if __name__ == "__main__":
    check_preprocessed_data() 