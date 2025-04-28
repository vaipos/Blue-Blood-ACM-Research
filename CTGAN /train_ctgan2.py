import argparse
import os
import pandas as pd
import numpy as np
import joblib
import re
from ctgan import CTGAN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', ''))
    return parser.parse_args()

def clean_and_convert(x):
    if pd.notna(x):
        try:
            x = re.sub(r'[\[\]]', '', x)
            cleaned = re.sub(r'\s+', ' ', x.strip())
            return np.array([float(i) for i in cleaned.split(' ')])
        except Exception as e:
            print(f"Error processing value: {x}. Error: {e}")
            return x
    return x

def main():
    args = parse_args()
    print(f"Args: {args}")
    
    train_file = os.path.join(args.train, 'final_data_april22.csv')
    print(f"Reading training data from: {train_file}")
    df = pd.read_csv(train_file)

    
    # Train the CTGAN model
    model = CTGAN(
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True
    )

    print("Training CTGAN...")
    model.fit(df)
    print("Training completed.")

    # Generate synthetic data
    synthetic_data_apr22 = model.sample(2000)
    print("Synthetic data sample:")
    print(synthetic_data_apr22.head())


    # Save outputs
    os.makedirs(args.output_data_dir, exist_ok=True)
    synthetic_data_apr22.to_csv(os.path.join(args.output_data_dir, 'synthetic_data_apr22.csv'), index=False)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, 'ctgan_model.pkl'))

if __name__ == '__main__':
    main()