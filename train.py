import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import sys
import json
import logging
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import joblib

# Set up logging to appear in SageMaker logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ----------------- PARSE HYPERPARAMETERS -----------------
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by SageMaker
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lstm_units", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # SageMaker adds this automatically - we need to handle it
    parser.add_argument("--model_dir", type=str, default=None,
                       help="S3 path for saving model artifacts")

    # SageMaker data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    
    # Debug parameter - accept both flag and value versions
    parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help="Enable debug mode (--debug or --debug True)")

    return parser.parse_args()

# ----------------- PREPROCESSING FUNCTIONS -----------------
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
    """Get unique pairs of subject_id and prescription_start"""
    subject_ids = df['subject_id'].unique()
    patient_date_pairs = {}
    
    for subj in subject_ids:
        subj_df = df[df['subject_id'] == subj]
        dates = subj_df['prescription_start'].unique()
        patient_date_pairs[subj] = list(dates)
        
    return patient_date_pairs

def get_presc_input(df):
    """Create prescription array with the proper format"""
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

def add_padding(prescriptions, pre_treatment, post_treatment):
    """Add proper padding to input arrays"""
    padded_prescriptions = np.pad(prescriptions, ((0, 0), (0, 50)), mode='constant')
    padded_pre_treatment = np.pad(pre_treatment, ((0, 0), (130, 25)), mode='constant')
    padded_post_treatment = np.pad(post_treatment, ((0, 0), (155, 0)), mode='constant')
    
    return padded_prescriptions, padded_pre_treatment, padded_post_treatment

# ----------------- PREPARE TRAINING DATA -----------------
def prepare_training_data(df):
    """Prepare training data from DataFrame"""
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
            
            # Pre_treatment and post_treatment are 1D arrays
            pre_columns = [col for col in processing_data.columns if col.startswith('pre_')]
            post_columns = [col for col in processing_data.columns if col.startswith('post_')]
            
            if len(pre_columns) == 0 or len(post_columns) == 0:
                continue
                
            pre_treatment = np.array(processing_data[pre_columns].values[0])
            post_treatment = np.array(processing_data[post_columns].values[0])
            
            # Add padding to the inputs
            padded_prescriptions, padded_pre_treatment, padded_post_treatment = add_padding(
                prescriptions, pre_treatment.reshape(1, -1), post_treatment.reshape(1, -1))
            
            # Create the full sequence (1 patient, 3 time steps, 180 features)
            X = np.array([[
                padded_pre_treatment[0],     # Time Step 1: Pre-Treatment
                padded_prescriptions[0],     # Time Step 2: Prescription
                padded_post_treatment[0]     # Time Step 3: Post-Treatment
            ]])
            
            y = X[:, -1, :]  # Target is the last time step (Post-Treatment)
            
            X_train_list.append(X[0])
            y_train_list.append(y[0])
    
    return np.array(X_train_list), np.array(y_train_list)

# ----------------- LOAD DATA FROM S3 -----------------
def load_data(data_path):
    """Load and preprocess data from S3 or local filesystem, focusing on final_df.csv"""
    # Check if data_path exists
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        sys.exit(1)
    
    # In SageMaker, data_path will be a directory containing the downloaded files
    if os.path.isdir(data_path):
        # Look for any CSV file
        csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".csv")]
        if not csv_files:
            logger.error(f"No CSV files found in directory: {data_path}")
            sys.exit(1)
        
        # Use the first CSV file found
        file_path = csv_files[0]
    elif os.path.isfile(data_path) and data_path.endswith('.csv'):
        # If data_path is already a file, use it directly
        file_path = data_path
    else:
        logger.error(f"Invalid data path: {data_path}. Must be a directory containing CSV files or a CSV file.")
        sys.exit(1)
    
    # Load the single CSV file
    try:
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ['subject_id', 'prescription_start']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            sys.exit(1)
        
        # Try to prepare training data
        return prepare_training_data(df)
    except Exception as e:
        logger.error(f"Failed to process data file {file_path}: {str(e)}", exc_info=True)
        sys.exit(1)

# ----------------- BUILD LSTM MODEL -----------------
def build_model(lstm_units=64, dropout_rate=0.2, learning_rate=0.001):
    """Build the LSTM model matching the architecture in the notebook"""
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(3, 180)),  # 3 time steps, 180 features
        Dropout(dropout_rate),
        LSTM(lstm_units//2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(16, activation="relu"),
        Dense(180)  # Predicting post-treatment blood profile
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# ----------------- TRAIN MODEL -----------------
def train():
    try:
        args = parse_args()
        
        # Load data (will abort if data loading fails)
        logger.info(f"Loading data from: {args.train}")
        X, y = load_data(args.train)
        logger.info(f"Successfully loaded data X: {X.shape}, y: {y.shape}")
        
        # Implement train-test split
        test_size = 0.2  # 20% for test set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Build and train model with minimal epochs for testing
        model = build_model(
            lstm_units=args.lstm_units,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate
        )
        
        # Print model summary to logs
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        logger.info(f"Model Summary:\n{model_summary}")
        
        # Use only a few epochs for debugging
        actual_epochs = min(args.epochs, 5) if args.debug else args.epochs
        
        # Train with reduced epochs
        history = model.fit(
            X_train, y_train,
            epochs=actual_epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, y_val)
        )
        
        # Get final metrics
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        # Save metrics to model_dir if provided
        if args.model_dir:
            metrics = {
                'train_loss': float(final_train_loss),
                'val_loss': float(final_val_loss)
            }
            
            # Log metrics to the console
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

            # Save the model to s3
            os.makedirs(args.model_dir, exist_ok=True)
            joblib.dump(model, os.path.join(args.model_dir, 'lstm_model.pkl'))
        
        return {
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'history': history.history
        }
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        logger.info("Starting training script")
        train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
