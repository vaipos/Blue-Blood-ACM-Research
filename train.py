import os
import argparse
import pandas as pd
import numpy as np
import re
import json
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, Reshape
import logging
from sklearn.model_selection import train_test_split
import joblib

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

def build_model(lstm_units=64, dropout_rate=0.2):
    model = Sequential([
        TimeDistributed(Flatten(), input_shape=(3, 20, 130)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(20 * 130, activation="linear"),
        Reshape((20, 130))
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_training_data(df):
    # Process embeddings
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

def train_model(df, model, epochs=10, batch_size=1):
    
    X, y = prepare_training_data(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_val, y_val)
    )

    logger.info(f"History: {history}")
        
    # Save metrics to model_dir if provided
    if args.model_dir:
        

        # Save the model to s3
        os.makedirs(args.model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(args.model_dir, 'lstm_model.pkl'))

    return history.history
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lstm-units', type=int, default=64)
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    
    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=None, help='Directory to save model artifacts')
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    
    args, _ = parser.parse_known_args()
    
    # Load training data
    print(f"Loading data from {args.train}")
    df = pd.read_csv(os.path.join(args.train, 'final_df.csv'))
    
    # Build model with specified hyperparameters
    print("Building model...")
    model = build_model(lstm_units=args.lstm_units, dropout_rate=args.dropout_rate)

    # Print model summary to logs
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    logger.info(f"Model Summary:\n{model_summary}")
    
    # Train the model
    print("Starting model training...")
    history = train_model(df, model, epochs=args.epochs, batch_size=args.batch_size)
    
    print("Training complete!")
