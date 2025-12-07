#!/bin/bash

# =============================================================================
# Project Initialization Script
# =============================================================================
# Usage:
#   bash init.sh                              # Use defaults
#   bash init.sh --data /work/dataset         # Override data path
#   bash init.sh --help                       # Show all options
# =============================================================================

# =============================================================================
# CONFIGURATION - Edit these for your project
# =============================================================================
DEFAULT_REPO_NAME="ADVCV_final_project"
DEFAULT_REPO_URL="https://github.com/MalipieroMattia/ADVCV_final_project.git"
DEFAULT_BRANCH="main"
DEFAULT_DATA_PATH="raw_data/Data_YOLO"
DEFAULT_NUM_CLASSES=9
DEFAULT_CLASS_NAMES="0:SH 1:SP 2:SC 3:OP 4:MB 5:HB 6:CS 7:CFO 8:BMFO"

VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"

# =============================================================================
# Helper Functions
# =============================================================================
function echo_info { echo -e "\033[1;34m[INFO]\033[0m $1"; }
function echo_success { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
function echo_error { echo -e "\033[1;31m[ERROR]\033[0m $1"; }
function echo_warning { echo -e "\033[1;33m[WARNING]\033[0m $1"; }

function show_help {
    echo ""
    echo "Usage: bash init.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --data PATH         Path to dataset (default: $DEFAULT_DATA_PATH)"
    echo "  --repo URL          Git repository URL"
    echo "  --repo-name NAME    Repository directory name"
    echo "  --branch BRANCH     Git branch to checkout"
    echo "  --skip-clone        Skip cloning/pulling repository"
    echo "  --skip-venv         Skip virtual environment setup"
    echo "  --skip-deps         Skip installing dependencies"
    echo "  --skip-wandb        Skip Weights & Biases setup"
    echo "  --wandb-key KEY     Weights & Biases API key"
    echo "  --help              Show this help message"
    echo ""
    exit 0
}

# =============================================================================
# Parse Command Line Arguments
# =============================================================================
REPO_NAME="$DEFAULT_REPO_NAME"
REPO_URL="$DEFAULT_REPO_URL"
BRANCH="$DEFAULT_BRANCH"
DATA_PATH="$DEFAULT_DATA_PATH"
SKIP_CLONE=false
SKIP_VENV=false
SKIP_DEPS=false
SKIP_WANDB=false
WANDB_KEY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data) DATA_PATH="$2"; shift 2 ;;
        --repo) REPO_URL="$2"; shift 2 ;;
        --repo-name) REPO_NAME="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --skip-clone) SKIP_CLONE=true; shift ;;
        --skip-venv) SKIP_VENV=true; shift ;;
        --skip-deps) SKIP_DEPS=true; shift ;;
        --skip-wandb) SKIP_WANDB=true; shift ;;
        --wandb-key) WANDB_KEY="$2"; shift 2 ;;
        --help|-h) show_help ;;
        *) DATA_PATH="$1"; shift ;;
    esac
done

# =============================================================================
# Main Script
# =============================================================================
echo ""
echo "=============================================="
echo "  Project Environment Setup"
echo "=============================================="
echo ""

OS="$(uname)"
echo_info "Operating System: $OS"

IS_UCLOUD=false
if [[ "$OS" == "Linux" ]] && [[ -d "/work" ]]; then
    IS_UCLOUD=true
    echo_info "Environment: UCloud"
else
    echo_info "Environment: Local"
fi

# Install packages (local only)
if [[ "$IS_UCLOUD" == false ]]; then
    if [[ "$OS" == "Linux" ]]; then
        echo_info "Installing system packages..."
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-venv python3-pip git > /dev/null 2>&1
    elif [[ "$OS" == "Darwin" ]]; then
        if command -v brew &> /dev/null; then
            brew install python3 git > /dev/null 2>&1
        fi
    fi
fi

# Clone/Update Repository
if [[ "$SKIP_CLONE" == false ]]; then
    if [ ! -d "$REPO_NAME" ]; then
        echo_info "Cloning repository..."
        git clone --branch "$BRANCH" "$REPO_URL" "$REPO_NAME"
        cd "$REPO_NAME"
    else
        echo_info "Updating repository..."
        cd "$REPO_NAME"
        git pull origin "$BRANCH" 2>/dev/null || git pull
    fi
else
    echo_info "Skipping repository clone/update"
    [ -d "$REPO_NAME" ] && cd "$REPO_NAME"
fi

# Virtual Environment
if [[ "$SKIP_VENV" == false ]]; then
    [ ! -d "$VENV_DIR" ] && python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    if [[ "$SKIP_DEPS" == false ]]; then
        pip install --upgrade pip --quiet
        [ -f "$REQUIREMENTS_FILE" ] && pip install -r "$REQUIREMENTS_FILE" --quiet
    fi
fi

# Generate data.yaml
mkdir -p configs
if [[ "$DATA_PATH" == /* ]]; then
    YAML_PATH="$DATA_PATH"
else
    YAML_PATH="../$DATA_PATH"
fi

cat > configs/data.yaml << EOF
path: $YAML_PATH
train: images/train
val: images/val
nc: $DEFAULT_NUM_CLASSES
names:
EOF

for class_entry in $DEFAULT_CLASS_NAMES; do
    echo "  ${class_entry%%:*}: ${class_entry#*:}" >> configs/data.yaml
done

echo_success "Created configs/data.yaml"

# Verify Dataset
if [ -d "$DATA_PATH/images/train" ]; then
    TRAIN_COUNT=$(ls -1 "$DATA_PATH/images/train" 2>/dev/null | wc -l)
    VAL_COUNT=$(ls -1 "$DATA_PATH/images/val" 2>/dev/null | wc -l)
    echo_success "Dataset found: Train=$TRAIN_COUNT, Val=$VAL_COUNT"
else
    echo_warning "Dataset not found at: $DATA_PATH"
fi

# Wandb Setup
if [[ "$SKIP_WANDB" == false ]]; then
    [ -n "$WANDB_KEY" ] && export WANDB_API_KEY="$WANDB_KEY"
    if [ -n "$WANDB_API_KEY" ]; then
        echo "WANDB_API_KEY=$WANDB_API_KEY" > .env
        wandb login "$WANDB_API_KEY" 2>/dev/null && echo_success "Logged into W&B"
    else
        echo "# WANDB_API_KEY=your_key_here" > .env
        echo_warning "WANDB_API_KEY not set"
    fi
fi

# Summary
echo ""
echo "=============================================="
echo_success "Setup Complete!"
echo "=============================================="
echo ""
echo "  Data path: $DATA_PATH"
echo "  Directory: $(pwd)"
echo ""
echo "Next steps:"
echo "  source $VENV_DIR/bin/activate"
echo "  python main.py --help"
echo ""
