# Blue-Blood

Welcome to the **Blue-Blood-ACM-Research** repository! This project is part of an ongoing research effort aimed at integrating simulated drug interaction data and transfer learning to predict changes in a patient's blood profile after drug administration.

## Project Overview

The primary goal of this project is to develop a machine learning (ML) model that can integrate simulated drug interaction data with Complete Blood Count (CBC) blood profile data. The model will predict how a patient's blood profile changes following drug administration, even though post-treatment data is not yet available.

### Key Objectives:
- Leverage drug compound datasets and CBC blood profile data to predict potential changes.
- Use transfer learning to improve model performance despite the lack of post-treatment data.
- Analyze how drug interactions may affect a patient's blood profile in real-world scenarios.

## Technologies Used

- **Python** for model development
- **TensorFlow/PyTorch** for deep learning model training
- **Pandas** and **NumPy** for data manipulation
- **Scikit-learn** for model evaluation
- **Matplotlib/Seaborn** for data visualization

## Dataset (MIMIC-III Clinical Database):

The project utilizes:
- **Drug Compounds Dataset**: Contains information on various drug compounds.
- **CBC Blood Profile Data**: Comprises patient blood test results, focusing on various blood components like white blood cells, red blood cells, and platelets.

## Installation

To start the project, clone the repository and install the required dependencies. **More setup instructions will be provided to you during Build Night 1**. 

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/Blue-Blood_ACM-Research.git
cd Blue-Blood_ACM-Research
```

### 2. Install dependencies:

#### Check if you have Python and Conda installed:
Before installing, check if you already have Python and Conda installed on your system by running the following commands:

- **Check Python**:
  ```bash
  python --version
  ```

- **Check Conda**:
  ```bash
  conda --version
  ```

If you see version numbers in response, you already have Python and/or Conda installed! If not, follow the steps below:

#### If you don't have Python:
- Download Python from [here](https://www.python.org/downloads/).

#### If you don't have Conda:
- Download Conda from [here](https://www.anaconda.com/download/success).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
