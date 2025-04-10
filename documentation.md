# Blue Blood Project Documentation

## Overview
This project uses machine learning to predict post-treatment blood profiles based on pre-treatment data and medication prescriptions. The system uses LSTM networks to analyze the temporal relationship between pre-treatment blood values, medication prescriptions, and post-treatment outcomes.

## Files

### `model_testing.ipynb`(currently being used to test out stat tests)
Basically just the local version of train.py
**Newest Updates**
  - some stat tests added and working
  - stat tests run for each row of y_pred and y_val
  - columns where error exceeds 2% are being printed

### `hyperparam-loop.ipynb`
Notebook for automated hyperparameter tuning of LSTM models using AWS SageMaker.
- **Features:**
  - Setting up AWS SageMaker session and IAM role permissions
  - Verifying S3 data and code accessibility
  - Systematically testing multiple hyperparameter combinations
  - Running training jobs in parallel on SageMaker
  - Fetching and displaying training/validation loss plots from S3
- **Hyperparameters Tested:**
  - Epochs: 10, 20, 50
  - Learning rates: 0.001, 0.01, 0.1
  - LSTM units: 32, 64, 128
  - Dropout rates: 0.2, 0.3, 0.4
- **Infrastructure:**
  - Instance type: ml.m5.4xlarge
  - Framework: TensorFlow 2.9
  - Python version: 3.9

### `train.py` (stat tests haven't been tested from sagemaker)
TensorFlow script for training an LSTM model on AWS SageMaker.
- **Key Functions:**
  - `clean_and_convert()`: Preprocesses embedding strings to numpy arrays
  - `get_unique_pairs()`: Creates dictionary mapping patient IDs to prescription dates
  - `get_presc_input()`: Combines embeddings with dosage values and units
  - `add_padding()`: Standardizes input shapes to (20, 130) for each timestep
  - `build_model()`: Creates LSTM model architecture with configurable parameters
  - `prepare_training_data()`: Transforms dataframes into model-ready format
  - `train_model()`: Handles training loop with validation split and model saving
  - `chart_model_performance()`: Creates and saves training/validation loss plots to S3
- **Model Architecture:**
  - TimeDistributed layer to process each time step
  - LSTM layer for sequence processing
  - Dropout for regularization
  - Dense layer followed by reshape for output formatting
- **Hyperparameters:**
  - Configurable epochs, learning rate, LSTM units, and dropout rate
  - Command-line arguments for SageMaker integration
- **Execution:** 
  - Invoked by SageMaker with custom hyperparameters
  - Loads data from S3 paths
  - Saves trained model to S3 as pickle file
  - Generates and uploads performance visualization plots to S3

### `bb-dev-lstm.ipynb`
Notebook for configuring and executing AWS SageMaker training jobs. 
- **Features:**
  - Setting up SageMaker execution role and permissions
  - Verifying S3 data and code accessibility
  - Configuring TensorFlow estimator parameters
  - Launching training jobs with customizable hyperparameters
  - Job monitoring and model deployment
  - Fetching and displaying training/validation loss plots from S3
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

### `preprocessing.ipynb`
Preprocessing steps taken to get from original data to final_df.csv