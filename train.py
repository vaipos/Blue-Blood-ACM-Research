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
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Reshape
from keras.callbacks import EarlyStopping
import logging
import pickle
import matplotlib
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#AWS Credentials
BUCKET_NAME = "blue-blood-data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_and_convert(x):
    if isinstance(x, np.ndarray):  # If already a NumPy array, return as-is
        return x
    if isinstance(x, str):  # Only process strings
        try:
            x = re.sub(r'[\[\]]', '', x)  # Remove square brackets
            cleaned = re.sub(r'\s+', ' ', x.strip())  # Remove extra spaces
            return np.array([float(i) for i in cleaned.split(' ')])  # Convert to NumPy array
        except Exception as e:
            return x  # Return original value in case of error
    return x  # If NaN or unexpected type, return as-is

def get_unique_pairs(df):
    subject_ids = df['subject_id'].unique()
    patient_date_pairs = {id: set() for id in subject_ids}

    for subj in subject_ids:
        df[df['subject_id'] == subj]['prescription_start'].apply(lambda x: patient_date_pairs[subj].add(x))
        # convert set to list
        patient_date_pairs[subj] = list(patient_date_pairs[subj])
    return patient_date_pairs

def get_presc_input(df):
    prescriptions = []
        
    # Iterate through rows of the DataFrame
    for _, row in df.iterrows():
        # Extract values from each row
        presc = row['prescription_rx_embeddings']
        dose_val = row['prescription_dose_val_rx']
        dose_unit = row['prescription_dose_unit_rx']
        
        # Concatenate the prescription embedding with the dose value and unit
        combined = np.concatenate((presc, np.array([dose_val, dose_unit])))
        prescriptions.append(combined)
    
    # Convert list to numpy array
    prescriptions = np.array(prescriptions)

    return prescriptions

# function that adds the proper padding to our input arrays
def add_padding(prescriptions, pre_treatment, post_treatment):
    # reshape pre_treatment and post_treatment to be 2D arrays
    pre_treatment = pre_treatment.reshape(1, -1)
    post_treatment = post_treatment.reshape(1, -1)
    
    # Pad or truncate to 20 rows
    if prescriptions.shape[0] < 20:
        # Pad with zeros to reach 20 rows
        prescriptions = np.pad(prescriptions, ((0, 20 - prescriptions.shape[0]), (0, 0)), mode='constant')
    elif prescriptions.shape[0] > 20:
        # Truncate to 20 rows
        prescriptions = prescriptions[:20, :]
    
    # Pad pre_treatment and post_treatment to 20 rows
    pre_treatment = np.pad(pre_treatment, ((0, 19), (0, 0)), mode='constant')  # pad to (20, 25)
    post_treatment = np.pad(post_treatment, ((0, 19), (0, 0)), mode='constant')  # pad to (20, 25)
    
    # Now pad columns to reach 180 features for each
    padded_prescriptions = np.pad(prescriptions, ((0, 0), (0, 0)), mode='constant')
    padded_pre_treatment = np.pad(pre_treatment, ((0, 0), (0, 105)), mode='constant')
    padded_post_treatment = np.pad(post_treatment, ((0, 0), (0, 105)), mode='constant') 
    
    return padded_prescriptions, padded_pre_treatment, padded_post_treatment

def build_model(lstm_units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        TimeDistributed(Flatten(), input_shape=(3, 20, 130)),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(20 * 130, activation="linear"),
        Reshape((20, 130))
    ])

    # Compile the model with specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def prepare_training_data(df):
    # Clean and convert the DataFrame
    df['prescription_rx_embeddings'] = df['prescription_rx_embeddings'].apply(clean_and_convert)

    # Get unique patient/date pairs
    patient_date_pairs = get_unique_pairs(df)
    
    X_train_list = []
    y_train_list = []
    
    # Iterate through the patient/date pairs
    for patient in patient_date_pairs:
        for date in patient_date_pairs[patient]:
            # Get the data for the current patient/date pair
            patient_data = df[(df['subject_id'] == patient) & (df['prescription_start'] == date)]
            
            if len(patient_data) == 0:
                continue
                
            # Drop unnecessary columns for processing
            processing_data = patient_data.drop(['subject_id', 'prescription_start', 'pre_charttime', 'post_charttime'], axis=1)
            
            # Get the prescription input (2DArray with shape (num_prescriptions, 130))
            prescriptions = get_presc_input(processing_data)
            
            # pre_treatment and post_treatment are 1D arrays
            pre_treatment = np.array(processing_data[[col for col in processing_data.columns if col.startswith('pre_')]].values[0])
            post_treatment = np.array(processing_data[[col for col in processing_data.columns if col.startswith('post_')]].values[0])
            
            # Add padding to the inputs
            padded_prescriptions, padded_pre_treatment, padded_post_treatment = add_padding(prescriptions, pre_treatment, post_treatment)
            
            # Create the full sequence (1 patient, 3 time steps, 180 features)
            X = np.array([[
                padded_pre_treatment,     # Time Step 1: Pre-Treatment
                padded_prescriptions,     # Time Step 2: Prescription
                padded_post_treatment     # Time Step 3: Post-Treatment
            ]])
            
            y = X[:, -1, :]  # Target is the last time step (Post-Treatment)
            
            X_train_list.append(X[0])
            y_train_list.append(y[0])
    
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
def plot_error_distribution(y_pred, y_val, threshold=0.002):
    # Force Matplotlib to use a non-GUI backend
    matplotlib.use("Agg")
    
    epsilon = 1e-8  # Keep this as a small value to prevent division by zero
    relative_error = np.abs(y_pred - y_val) / (np.abs(y_val) + epsilon)
    
    weights = np.zeros_like(y_val)
    weights[:, 0, :25] = 1.0  # only valid (non-padded) values

    error_mask = (relative_error > 0.02) * weights
    errors_per_sample = np.sum(error_mask, axis=(1, 2))  # shape: (n_samples,)

    error_distribution = collections.Counter(errors_per_sample)
    
    x_axis = list(error_distribution.keys())
    y_axis = list(error_distribution.values())

    plt.figure(figsize=(10, 6))
    plt.scatter(x_axis, y_axis)
    plt.xlabel("Number of Errors (per sample)")
    plt.ylabel("Number of Samples")
    plt.title("Samples vs. Error (Relative Error > 0.2%)")
    plt.grid(True)

    # Save plot to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    # Upload to S3
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
    weights[:, 0, :25] = 1.0

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
    df = pd.read_csv(os.path.join(args.train, 'final_df.csv'))
    
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

















