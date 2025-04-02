# Blue Blood Project Documentation

## Overview
This project uses machine learning to predict post-treatment blood profiles based on pre-treatment data and medication prescriptions. The system uses LSTM networks to analyze the temporal relationship between pre-treatment blood values, medication prescriptions, and post-treatment outcomes.

## Files

### `lstm.ipynb`
Notebook for configuring and executing AWS SageMaker training jobs.
- **Features:**
  - Setting up SageMaker execution role and permissions
  - Verifying S3 data and code accessibility
  - Configuring TensorFlow estimator parameters
  - Launching training jobs with customizable hyperparameters
  - Job monitoring and model deployment
- **Training Configuration:**
  - Instance type: ml.m5.xlarge
  - Framework: TensorFlow 2.9
  - Python version: 3.9
  - Customizable hyperparameters include:
    - Epochs
    - Learning rate
    - LSTM units
    - Dropout rate
    - Batch size

### `train.py`
TensorFlow script for training an LSTM model on AWS SageMaker.
- **Key Functions:**
  - `clean_and_convert()`: Preprocesses embedding strings to numpy arrays
  - `get_unique_pairs()`: Creates dictionary mapping patient IDs to prescription dates
  - `get_presc_input()`: combines embeddings w/ dosage vals and units
  - `add_padding()`: Standardizes input shapes to (20, 130) for each timestep
  - `build_model()`: Creates LSTM model architecture with configurable parameters
  - `prepare_training_data()`: Transforms dataframes into data that is usable by the model
  - `train_model()`: Handles training loop with validation split
- **Model Architecture:**
  - TimeDistributed layer to process each time step
  - LSTM layer for sequence processing
  - Dropout for regularization
  - Dense layer followed by reshape for output formatting
- **Execution:** 
  - Called by SageMaker with custom hyperparameters
  - Loads data from S3 paths
  - Saves trained model and metrics back to S3 (not yet)

## Data Processing Workflow
1. Load raw data from S3 containing prescription and blood test information
2. Process prescription embedding strings into numpy arrays
3. Group data by patient and prescription date
4. For each patient/date pair:
   - Extract pre-treatment blood values
   - Format prescription information with embeddings and dosage
   - Include post-treatment blood values
5. Create 3D sequence data with shape (n_samples, 3, 20, 130):
   - First dimension: Patient/date samples
   - Second dimension: 3 Time sequences (pre-treatment, prescriptions, post-treatment)
   - Third dimension: 20 rows for multiple prescriptions/values (padded if fewer)
   - Fourth dimension: 130 features per row

## Model Training Process
1. Load and preprocess data
2. Split data into training and validation sets
3. Train LSTM model to predict post-treatment values from pre-treatment and prescription data
4. Validate performance against test data
5. Save model and performance metrics (not yet)

### `model_testing.ipynb`
Basically just the local version of train.py