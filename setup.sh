#!/bin/bash

command_exists() {
    command -v "$1" &>/dev/null
}

if command_exists conda; then
    echo "Conda is installed."
else
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

current_env=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$current_env" != "bb_dev" ]; then
    echo "Activating the 'bb_dev' environment..."
    conda activate bb_dev
else
    echo "Already in the 'bb_dev' environment."
fi

if command_exists aws; then
    echo "AWS CLI is already installed."
else
    echo "AWS CLI is not installed. Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws
fi

if command_exists jupyter; then
    echo "Jupyter Notebook is already installed."
else
    echo "Jupyter Notebook is not installed. Installing Jupyter Notebook..."
    conda run -n bb_dev conda install notebook -y
fi

if command_exists git; then
    echo "Git is already installed."
else
    echo "Git is not installed. Installing Git..."
    sudo apt-get update
    sudo apt-get install git -y
fi

if [ -f "requirements.txt" ]; then
    echo "Installing Python libraries from requirements.txt using conda..."
    conda install --file requirements.txt -y
else
    echo "requirements.txt not found. Please make sure the file exists in the current directory."
fi

echo "Setup complete!"
