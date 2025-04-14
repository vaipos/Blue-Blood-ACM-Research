'''
- get model to predict based on pre-treatment cbc and rx data --> 57 predictions, of 20rows 130 columsn each....
- 
'''

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
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Reshape
from keras.callbacks import EarlyStopping
import logging
import pickle
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#AWS Credentials
BUCKET_NAME = "blue-blood-data"

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

def build_model(lstm_units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=(2, 130)),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dense(130)
    ])

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
        
        # Get prescription data for current row
        prescriptions = np.array(row[presc_cols].values)
        
        # Add padding to the inputs
        padded_pre_treatment, padded_post_treatment = add_padding(pre_treatment, post_treatment)
        
        # Create the full sequence (now directly using the padded post_treatment as target)
        X = np.array([
            padded_pre_treatment,     # Pre-Treatment
            prescriptions             # Prescription
        ])
        
        y = padded_post_treatment  
        
        X_train_list.append(X)
        y_train_list.append(y)
    
    return np.array(X_train_list), np.array(y_train_list)

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

    #return history.history, s3_model_path, s3_chart_path
    return history.history, s3_model_path, s3_chart_path, X_val, y_val

def plot_training_val_loss(history, figsize=(8, 6), train_marker='o', val_marker='s', job_name=None):
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
    
'''
Need to look into this again --> check if the way the predicted CBC-actual CBC error is properly being flagged, because the distribution doesn't look right
'''
def plot_error_distribution(y_pred, y_val, threshold=0.002, job_name=None):
    # Force Matplotlib to use a non-GUI backend
    matplotlib.use("Agg")
    
    epsilon = 1e-8  # small value to prevent division by zero
    relative_error = np.abs(y_pred - y_val) / (np.abs(y_val) + epsilon)
    error_mask = relative_error > threshold
    
    # Change this line - remove axis=2 since we now have 2D arrays
    error_counts_per_sample = np.sum(error_mask, axis=1)  # Changed from axis=(1, 2)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(error_counts_per_sample, bins=range(0, np.max(error_counts_per_sample) + 2), edgecolor='black')
    plt.xlabel(f"Number of Features Outside {threshold*100:.3f}% Relative Error (per sample)")
    plt.ylabel("Number of Samples")
    plt.title('Relative Error Distribution')
    plt.grid(True)

    # Rest of the function remains the same
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    s3_chart_key = f"models/{job_name}/relative-error-distribution.png"
    s3 = boto3.client("s3")
    s3.upload_fileobj(buf, BUCKET_NAME, s3_chart_key)

def evaluate_model_performance(model, X_val, y_val, epochs, lstm_units, dropout_rate, learning_rate):
    y_pred = model.predict(X_val)

    # Unweighted Statistical Metrics
    mse = np.mean((y_pred - y_val) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_val))

    # Weighted Statistical metrics
    weights = np.zeros_like(y_val)
    weights[:, :25] = 1.0

    squared_error = (y_pred - y_val) ** 2
    abs_error = np.abs(y_pred - y_val)

    weighted_squared_error = squared_error * weights
    weighted_abs_error = abs_error * weights

    # Weighted metrics
    weighted_mse = np.sum(weighted_squared_error) / np.sum(weights)
    weighted_rmse = np.sqrt(weighted_mse)
    weighted_mae = np.sum(weighted_abs_error) / np.sum(weights)

    #MD File Contents
    markdown = f"""# Model Performance Evaluation Report
    ## Hyperparameter Configuration:
    - **Epochs** : {epochs}
    - **LSTM Units** : {lstm_units}
    - **Dropout Rate** : {dropout_rate}
    - **Learning Rate** : {learning_rate}

    ## Unweighted Metrics
    - **Mean Square Error**  : {mse:.9f}
    - **RMSE** : {rmse:.9f}
    - **MAE**  : {mae:.9f}

    ## Weighted Metrics
    - **Weighted MSE**  : {weighted_mse:.9f}
    - **Weighted RMSE** : {weighted_rmse:.9f}
    - **Weighted MAE**  : {weighted_mae:.9f}
    """

    # Upload to S3
    s3_chart_key = f"models/{job_name}/model_evaluation.md"
    buf = io.BytesIO(markdown.encode("utf-8"))
    s3 = boto3.client("s3")
    s3.upload_fileobj(buf, BUCKET_NAME, s3_chart_key)

    return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lstm_units', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument("--job_name", type=str)
    
    # SageMaker paramebters
    parser.add_argument('--model-dir', type=str, default=None, help='Directory to save model artifacts')
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    args, _ = parser.parse_known_args()
    
    # Log job name from environment
    job_name = args.job_name

    # Load training data
    print(f"Loading data from {args.train}")
    df = pd.read_csv(os.path.join(args.train, 'synthetic_data.csv'))
    
    # Build model with specified hyperparameters
    print("Building & training the model...")
    model = build_model(lstm_units=args.lstm_units, dropout_rate=args.dropout_rate, learning_rate=args.learning_rate)

    # Print Model Summary
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    logger.info(f"Model Summary:\n{model_summary}")
    
    # Model Training
    print("Starting model training...")
    history, model_path, chart_path, X_val, y_val = train_model(
        df, 
        model, 
        epochs=args.epochs,
        job_name=job_name
    )
    plot_training_val_loss(history, job_name=job_name)

    # Testing & Evaluation
    y_pred = evaluate_model_performance(model, 
                                        X_val, 
                                        y_val, 
                                        epochs=args.epochs, 
                                        lstm_units=args.lstm_units, 
                                        dropout_rate=args.dropout_rate, 
                                        learning_rate=args.learning_rate)
    

    #Incorrectly displaying histograms --> get this fixed
    plot_error_distribution(y_pred, y_val, threshold=0.002)


