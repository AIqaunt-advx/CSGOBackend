#!/usr/bin/env python3
import os
import glob
import json
from data_processor import SkinDataProcessor
from gpu_trainer import GPUSkinTrainer

def process_all_json_files():
    processor = SkinDataProcessor()
    trainer = GPUSkinTrainer()
    
    # Load all JSON files
    json_files = glob.glob("*.json")
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) > 1:
        # Process multiple files
        df = processor.load_multiple_json_files("*.json")
    else:
        # Process single file
        df = processor.load_single_json('22313.json')
    
    if df is None:
        print("No data loaded")
        return
    
    print(f"Total records: {len(df)}")
    
    # Prepare and train
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(df)
    models = trainer.train_all_models(X_train, X_test, y_train, y_test)
    trainer.save_models()
    
    print("Batch processing completed!")

if __name__ == "__main__":
    process_all_json_files()
