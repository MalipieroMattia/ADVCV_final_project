#!/bin/bash

# -----------------------------
# Info message helpers
# -----------------------------
function echo_info {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

function echo_error {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

echo_info "Detected Operating System: $(uname)"

# -----------------------------
# Install system deps (Linux only)
# -----------------------------
if [[ "$(uname)" == "Linux" ]]; then
    echo_info "Updating package lists..."
    sudo apt-get update

    echo_info "Installing Python3, venv, pip, git..."
    sudo apt-get install -y python3 python3-venv python3-pip git
fi

# -----------------------------
# Clone the repository
# -----------------------------
REPO_DIR="ADVCV_final_project"
REPO_URL="https://github.com/MalipieroMattia/ADVCV_final_project.git"
BRANCH="seadronessea"

if [ ! -d "$REPO_DIR" ]; then
    echo_info "Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo_info "Repository already exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull
    cd ..
fi

# -----------------------------
# Enter repo and checkout branch
# -----------------------------
cd "$REPO_DIR"

echo_info "Checking out branch: $BRANCH"
git checkout "$BRANCH"

# -----------------------------
# Create venv inside repository
# -----------------------------
if [ ! -d ".venv" ]; then
    echo_info "Creating virtual environment..."
    python3 -m venv .venv
else
    echo_info "Virtual environment already exists."
fi

# Activate venv
echo_info "Activating virtual environment..."
source .venv/bin/activate

# -----------------------------
# Install Python requirements
# -----------------------------
echo_info "Upgrading pip..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo_info "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo_error "requirements.txt NOT FOUND"
fi

# -----------------------------
# Setup W&B
# -----------------------------
echo_info "Setting up Wandb..."

if [ -z "$WANDB_API_KEY" ]; then
    echo_info "Enter your Wandb API key:"
    read -s WANDB_API_KEY
    export WANDB_API_KEY
fi

echo "WANDB_API_KEY=$WANDB_API_KEY" > .env
echo_info ".env file created."

echo_info "Logging into Wandb..."
wandb login "$WANDB_API_KEY"

echo_info "Initialization complete."