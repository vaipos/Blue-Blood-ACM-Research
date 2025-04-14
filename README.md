# Blue Blood - Project Repository
## Overview
Welcome to BlueBlood! This project utilizes a machine learning pipeline to predict post-treatment blood profiles based on pre-treatment data and a series of prescriptions medication. By leveraging LSTM networks, the project aims to analyze the temporal relationship between blood test results and prescriptions, enabling better patient outcome predictions. This has potential applications in personalized medicine and treatment optimization.

## BlueBlood Workflow
### 1. Data Extraction & Pre-Processing
1. **Load Data:** Retrieve raw data from AWS S3 including tabular data with pre-treatment CBC, post-treatment CBC and series of prescriptions assigned to patients
2. **Scaling/Normalization:** Applied to all numerical columns (pre/post treatment CBC) & accounted for NULL/missing values in raw data 
3. **Process Prescription Data:** 
    - Convert prescription strings into BERT vector embeddings, and reduce to LSTM-compatible dimensions using UMAP
    - Ordinal encoding for all prescription labels & units
4. **Feature Engineering:**
   - Includes **pre & post-treatment blood values**.
   - Reformatted prescription information (embedding + dosage)

#### Source Code: `bb-dev-preprocessing.ipynb`
Notebook which includes all preprocessing and input formatting scripts, to format dataset and save into AWS S3 for future retrieval

### **2. Model Training & Evaluation**  
1. **Data Preparation:** Split the dataset into training, validation, and test sets.  

2. **Input Sequence Construction:** Format data into structured 3D sequences with shape **(n_samples, 3, 20, 130)**:  
   - **First axis:** Unique patient-date samples.  
   - **Second axis:** Three time steps—pre-treatment, prescriptions, and post-treatment.  
   - **Third axis:** Up to 20 rows for multiple prescriptions (padded if fewer).  
   - **Fourth axis:** 130 feature values per row.  

3. **LSTM Model Training:** Train the model to predict post-treatment blood values based on pre-treatment data and prescriptions.  

4. **Validation & Testing:** Assess performance on unseen data using validation and test sets.  

5. **Model Storage:** Save the trained model and evaluation metrics to AWS S3 for future use.  
  
#### Source Code: `bb-dev-lstms.ipynb`
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

#### Training & Evaluation Script: `train.py`
TensorFlow & Keras-based script for training & evaluating an LSTM model on AWS SageMaker which is executed on a Docker container; all outputs (model metrics, weights and biases), graphs and evaluation metrics saved to S3

- **Key Functions:**
  - *Training*
    - `clean_and_convert()`: Preprocesses embedding strings to numpy arrays
    - `get_unique_pairs()`: Creates dictionary mapping patient IDs to prescription dates
    - `get_presc_input()`: combines embeddings w/ dosage vals and units
    - `add_padding()`: Standardizes input shapes to (20, 130) for each timestep
    - `build_model()`: Creates LSTM model architecture with configurable parameters
    - `prepare_training_data()`: Transforms dataframes into data that is usable by the model
    - `train_model()`: Handles training loop with validation split
  - *Testing & Evaluation*
    - `plot_training_val_loss()`: Maps training and validation loss over epochs to a graph and saves to S3 model directory
    - `plot_error_distribution()`: Maps # of errors per row graphically and saves to S3 model directory
    - `evaluate_model_performance()`: Fetches model's prediction post-treatment CBC and evaluates statistical accuracy using:
      - Mean Sqauare Error
      - Root Mean Square Error
      - Mean Absolute Error
        - Two sets of testing are handled, for before and after accounting for padded data fields

- **Model Architecture:**
  - Time-Distributed layer to process each time step
  - LSTM layer for sequence processing
  - Dropout for regularization
  - Dense layer followed by reshape for output formatting
- **Execution:** 
  - Called by SageMaker with custom hyperparameters
  - Loads data from S3 paths
  - Saves trained model and metrics back to S3, with each model version (with distinct hyperparameters) saved to its own directory along with corresponding charts
