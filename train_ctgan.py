import argparse
import os
import pandas as pd
import numpy as np
import joblib
import re
from ctgan import CTGAN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', ''))
    #added
   

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
    
    train_file = os.path.join(args.train, 'final_df.csv')
    print(f"Reading training data from: {train_file}")
    df = pd.read_csv(train_file)

    # ---- PREPROCESSING STARTS HERE ----
    print("Preprocessing data...")

    # Drop unnecessary columns
    df = df.drop(columns=['subject_id', 'pre_charttime', 'post_charttime', 'prescription_start'], errors='ignore')

    # Clean embeddings column
    df['prescription_rx_embeddings'] = df['prescription_rx_embeddings'].apply(clean_and_convert)

    # Expand embeddings into individual columns
    expanded_columns = pd.DataFrame(df['prescription_rx_embeddings'].tolist(), index=df.index)
    expanded_columns.columns = [f'P{i}' for i in range(expanded_columns.shape[1])]
    df = pd.concat([df.drop(columns=['prescription_rx_embeddings']), expanded_columns], axis=1)

    # Fill any remaining NaNs
    df = df.fillna(0)

    print("Finished preprocessing.")
    print(df.head())
    # ---- PREPROCESSING ENDS HERE ----

    # Train the CTGAN model
    model = CTGAN(
        epochs=args.epochs,
        batch_size=args.batch_size,
        generator_lr=args.lr,
        discriminator_lr=args.lr,
        verbose=True
    )

    print("Training CTGAN...")
    model.fit(df)
    print("Training completed.")

    # Generate synthetic data
    synthetic_data = model.sample(100)
    print("Synthetic data sample:")
    print(synthetic_data.head())

    # Save outputs
    os.makedirs(args.output_data_dir, exist_ok=True)
    synthetic_data.to_csv(os.path.join(args.output_data_dir, 'synthetic_data.csv'), index=False)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, 'ctgan_model.pkl'))

if __name__ == '__main__':
    main()
