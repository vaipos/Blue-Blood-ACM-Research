import os
import argparse
import pandas as pd
import numpy as np
import re
import io
import json
import boto3
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
import logging
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, skew, kurtosis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET_NAME = "blue-blood-data"

def get_presc_cols(df):
    presc_cols = []

    for col in df.columns:
        # check if column starts with 'P'
        if col.startswith('P'):
            presc_cols.append(col)

    presc_cols.append('prescription_dose_val_rx')
    presc_cols.append('prescription_dose_unit_rx')

    return presc_cols

def get_presc_input(df):
    presc_cols = get_presc_cols(df)

    prescriptions = []
        
    # Iterate through rows of the DataFrame
    for _, row in df.iterrows():
        # Extract values from each row
        presc = np.array(row[presc_cols].values)
        prescriptions.append(presc)
    
    prescriptions = np.array(prescriptions)
    print(prescriptions.shape)

    return prescriptions

def add_padding(pre_treatment, post_treatment):
    # Compute the number of zeros to pad (130 - current length)
    pad_width = 130 - pre_treatment.shape[0]
    padded_pre_treatment = np.pad(pre_treatment, (0, pad_width), mode='constant')
    pad_width = 130 - post_treatment.shape[0]
    padded_post_treatment = np.pad(post_treatment, (0, pad_width), mode='constant')
    
    return padded_pre_treatment, padded_post_treatment

def build_model(lstm_units=64, dropout_rate=0.2, learning_rate=0.01):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(3, 130)),
        Dropout(0.2),
        LSTM(64 // 2, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(130)
    ])

    # Compile the model with specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def prepare_training_data(df):
    X_train_list = []
    y_train_list = []

    presc_cols = get_presc_cols(df)

    # for each row in the df
    for _, row in df.iterrows():
        # Extract pre_treatment and post_treatment from the current row
        pre_cols = [col for col in df.columns if col.startswith('pre_')]
        post_cols = [col for col in df.columns if col.startswith('post_')]
        
        # Get values for the current row
        pre_treatment = np.array(row[pre_cols].values)
        post_treatment = np.array(row[post_cols].values)
        
        # Get prescription data for current row (assuming this is already defined elsewhere)
        prescriptions = np.array(row[presc_cols].values)
        
        # Add padding to the inputs
        padded_pre_treatment, padded_post_treatment = add_padding(pre_treatment, post_treatment)
        
        # Create the full sequence (1 patient, 3 time steps, 130 features)
        X = np.array([[
            padded_pre_treatment,     # Time Step 1: Pre-Treatment
            prescriptions,            # Time Step 2: Prescription
            padded_post_treatment     # Time Step 3: Post-Treatment
        ]])
        
        y = X[:, -1, :]  # Target is the last time step (Post-Treatment)
        
        X_train_list.append(X[0])
        y_train_list.append(y[0])
    
    return np.array(X_train_list), np.array(y_train_list)

# Function to calculate skewness
def calculate_skewness(df1, df2):
    return df1.apply(lambda x: skew(x, nan_policy='omit')), df2.apply(lambda x: skew(x, nan_policy='omit'))

# Function to calculate kurtosis
def calculate_kurtosis(df1, df2):
    return df1.apply(lambda x: kurtosis(x, nan_policy='omit')), df2.apply(lambda x: kurtosis(x, nan_policy='omit'))

def train_model(df, model, epochs=10, job_name=None):
    X, y = prepare_training_data(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=1, 
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )

    y_pred = model.predict(X_val)

    logger.info(f"History: {history.history}")
    
    s3_model_path = f"models/{job_name}/lstm_model.pkl"
    s3_chart_path = f"models/{job_name}/training-validation-loss.png"
    
    # Save the model to in-memory buffer
    buf = io.BytesIO()
    pickle.dump(model, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)

    s3_client = boto3.client("s3")
    
    # Upload the model to S3
    s3_client.upload_fileobj(buf, BUCKET_NAME, s3_model_path)
    logger.info(f"Model successfully uploaded to s3://{BUCKET_NAME}/{s3_model_path}")

    return history.history, s3_model_path, s3_chart_path

def chart_model_performance(history, figsize=(8, 6), train_marker='o', val_marker='s', job_name=None):
    print("STARTING CHARTING! \n")
    
    # Force Matplotlib to use a non-GUI backend
    matplotlib.use("Agg")
    
    # Extract loss values
    train_loss = history['loss']
    val_loss = history.get('val_loss', None)
    epochs_range = range(1, len(train_loss) + 1)

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(epochs_range, train_loss, label='Train Loss', marker=train_marker)
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Validation Loss', marker=val_marker)
    
    print("Creating PLOT \n")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    s3_chart_key = f"models/{job_name}/training-validation-loss.png"

    # Upload to S3
    s3 = boto3.client("s3")
    s3.upload_fileobj(buf, BUCKET_NAME, s3_chart_key)
    
    print(f"Plot saved to S3: s3://{BUCKET_NAME}/{s3_chart_key}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--lstm-units', type=int, default=64)
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument("--job_name", type=str)
    
    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=None, help='Directory to save model artifacts')
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    
    args, _ = parser.parse_known_args()
    
    job_name = args.job_name
    
    # Load training data
    print(f"Loading data from {args.train}")
    df = pd.read_csv(os.path.join(args.train, 'synthetic_data.csv'))
    
    # Build model with specified hyperparameters
    print("Building model...")
    model = build_model(lstm_units=args.lstm_units, dropout_rate=args.dropout_rate, learning_rate=args.learning_rate)

    # Print model summary to logs
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    logger.info(f"Model Summary:\n{model_summary}")
    
    # Train the model
    print("Starting model training...")
    history, model_path, chart_path = train_model(
        df, 
        model, 
        epochs=args.epochs,
        job_name=job_name
    )
    
    print("Training complete!")
    print(f"Model saved to: {model_path}")
    print("History object:", history)
    print("Keys:", history.keys())

    chart_model_performance(history, job_name=job_name)
