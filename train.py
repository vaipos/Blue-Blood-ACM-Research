# STL Utils
import os
import io
import argparse
import logging
import pickle
import collections
import boto3

# Data Handling
import pandas as pd
import numpy as np

# Model Architecture & Training
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Statistical Analysis & Plots
import matplotlib
import matplotlib.pyplot as plt
import collections
from scipy.interpolate import make_interp_spline

#AWS Credentials
BUCKET_NAME = "blue-blood-data"

# Initialize Logger for Debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Read both time steps forwards/backwards
        Bidirectional(
            LSTM(
                lstm_units, 
                return_sequences=False,
                kernel_regularizer=l2(0.001)
            ), 
            input_shape=(2, 130)
        ),

        # Force minimum of 0.3 dropout
        Dropout(max(dropout_rate, 0.3)),
        
        # Regularization to favor generalization based on training data, rather than memorization
        Dense(130, activation='linear', kernel_regularizer=l2(0.001))
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

def plot_training_val_loss(history, figsize=(10, 6), train_marker='o', val_marker='s', job_name=None):
    # Force Matplotlib to use a non-GUI backend
    matplotlib.use("Agg")
    
    # Extract loss values
    train_loss = history['loss']
    val_loss = history.get('val_loss', None)
    epochs_range = range(1, len(train_loss) + 1)

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(epochs_range, train_loss, label='Training Loss', marker=train_marker, color='crimson')
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Validation Loss', marker=val_marker, color='blue')
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training vs Validation Loss', fontsize=18, fontweight='bold', color='black')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save plot to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    s3_chart_key = f"models/{job_name}/training-validation-loss.png"
    # Upload to S3
    s3 = boto3.client("s3")
    s3.upload_fileobj(buf, BUCKET_NAME, s3_chart_key)
    
def plot_error_distribution(y_pred, y_val, threshold=0.02, job_name=None):
    # Force Matplotlib to use a non-GUI backend
    matplotlib.use("Agg")
    
    epsilon = 1e-8
    absolute_error = np.abs(y_pred - y_val)
    relative_error = absolute_error / (np.abs(y_val) + epsilon)

    # Treat small ground truth values differently
    error_mask = np.where(
        np.abs(y_val) < 1e-4,  # if ground truth is (close to) 0
        absolute_error > 0.01,  # then flag if absolute error > 1%
        relative_error > threshold  # else use relative error
    )
    
    weights = (y_val != -1.0).astype(float)
    weights[:, 25:] = 0

    #error_mask = (relative_error > threshold) * weights
    errors_per_sample = np.sum(error_mask*weights, axis=1)  # shape: (n_samples,)

    per_sample_column_errors = []

    # Iterate over each sample
    for i in range(y_val.shape[0]):
        # Only check real data (row 0, columns 0–24)
        missed_indices = np.where((error_mask[i, :25] * weights[i, :25]) == 1)[0]
        per_sample_column_errors.append(missed_indices.tolist())
    
    for i, missed in enumerate(per_sample_column_errors[:5]):
        print(f"Sample {i} - Columns where Error > 2%: {missed}")

    error_distribution = collections.Counter(errors_per_sample)
    
    x_vals = np.array(sorted(error_distribution.keys()))
    y_vals = np.array([error_distribution[x] for x in x_vals])

    # Generalized Error Curve
    if len(x_vals) > 2:  # Need at least 3 points for spline
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
        spline = make_interp_spline(x_vals, y_vals, k=2)
        y_smooth = spline(x_smooth)
    else:
        x_smooth, y_smooth = x_vals, y_vals  # fallback to raw

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_smooth, y_smooth, label="Generalized Error Shape", color='blue', linewidth=2.5)
    # Shaded region under the curve
    plt.fill_between(x_smooth, y_smooth, color='blue', alpha=0.15)
    plt.scatter(x_vals, y_vals, color='red', s=80, edgecolors='black', label='Observed Error Counts', zorder=5)
    
    plt.suptitle("Sample Distribution vs Prediction Errors", fontsize=18, fontweight='bold', color='black')
    plt.title(f"Relative Error > {threshold*100:.1f}%", fontsize=14, color='black')
    plt.xlabel("Number of Errors per Sample", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Samples", fontsize=14, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save plot to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    # Upload to S3
    s3_chart_key = f"models/{job_name}/relative-error-distribution.png"
    s3 = boto3.client("s3")
    s3.upload_fileobj(buf, BUCKET_NAME, s3_chart_key)

def evaluate_model_performance(model, X_val, y_val, epochs, lstm_units, dropout_rate, learning_rate, job_name=None):
    y_pred = model.predict(X_val)

    # Unweighted Statistical Metrics
    mse = np.mean((y_pred - y_val) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_val))

    # (a) Mask out invalid values: either padding (after col 25) or -1.0
    mask = np.ones_like(y_val)
    mask[:, 25:] = 0                      # padding columns
    mask = mask * (y_val != -1.0)        # only real, non-null values

    # (b) Apply to error metrics
    squared_error = (y_pred - y_val) ** 2
    abs_error = np.abs(y_pred - y_val)

    weighted_squared_error = squared_error * mask
    weighted_abs_error = abs_error * mask

    weighted_mse = np.sum(weighted_squared_error) / np.sum(mask)
    weighted_rmse = np.sqrt(weighted_mse)
    weighted_mae = np.sum(weighted_abs_error) / np.sum(mask)

    #MD File Contents
    markdown = f"""# **Model Performance Evaluation Report**
## Hyperparameter Configuration:
- **Epochs** : {epochs}
- **LSTM Units** : {lstm_units}
- **Dropout Rate** : {dropout_rate}
- **Learning Rate** : {learning_rate}

## Unweighted Metrics
- **Mean Squared Error (MSE)**        : {mse:.9f}
- **Root Mean Squared Error (RMSE)**  : {rmse:.9f}
- **Mean Absolute Error (MAE)**       : {mae:.9f}

## Weighted Metrics
- **Weighted Mean Squared Error (MSE)**        : {weighted_mse:.9f}
- **Weighted Root Mean Squared Error (RMSE)**  : {weighted_rmse:.9f}
- **Weighted Mean Absolute Error (MAE)**       : {weighted_mae:.9f}

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
    df = pd.read_csv(os.path.join(args.train, 'blue-blood-synthetic-final.csv'))
    
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
                                        learning_rate=args.learning_rate,
                                        job_name=job_name)
    plot_error_distribution(y_pred, y_val, threshold=0.25, job_name=job_name)

    np.savetxt("y_pred_sample.csv", y_pred, delimiter=",")
    np.savetxt("y_val_sample.csv", y_val, delimiter=",")



