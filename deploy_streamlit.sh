#!/bin/bash

# Set environment name
ENV_NAME=streamlit_env

# Create a new Conda environment
echo "Creating a new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.8 -y

# Activate the environment
echo "Activating the environment"
source activate $ENV_NAME

# Install packages
echo "Installing packages"
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch -y
conda install -c pyg pyg=2.2.0 -y
conda install openbabel fsspec rdkit graph-tool -c conda-forge -y

# Export the environment to environment.yaml
echo "Exporting the environment to environment.yaml"
conda env export --no-builds > environment.yaml

# Create requirements.txt for Python dependencies
echo "Creating requirements.txt"
cat > requirements.txt <<EOL
streamlit==1.9.0
pytorch==1.12.0
torchvision==0.13.0
torchaudio==0.12.0
pyg==2.2.0
fsspec
rdkit
matplotlib
ogb
wandb
networkx
grinpy
EOL

# Create packages.txt for system dependencies (if any)
echo "Creating packages.txt"
cat > packages.txt <<EOL
openbabel
graph-tool
EOL

# Inform the user
echo "Setup complete. Files environment.yaml, requirements.txt, and packages.txt have been created."
echo "Please push these files to your GitHub repository for Streamlit deployment."
