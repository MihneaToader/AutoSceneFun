#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)

# Check if the script is not executed from the offlineProcessing folder
if [[ "$CURRENT_DIR" != *"offlineProcessing"* ]]; then
    echo "Please navigate to the 'offlineProcessing' directory and run the setup script from there."
    exit 1
fi

# Default values for models to download
DOWNLOAD_LITE=true
DOWNLOAD_FULL=true
DOWNLOAD_HEAVY=true

# Parse command-line arguments (only download desired models)
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            model="$2"
            case $model in
                lite)
                    DOWNLOAD_FULL=false
                    DOWNLOAD_HEAVY=false
                    ;;
                full)
                    DOWNLOAD_LITE=false
                    DOWNLOAD_HEAVY=false
                    ;;
                heavy)
                    DOWNLOAD_LITE=false
                    DOWNLOAD_FULL=false
                    ;;
                *)
                    echo "Invalid model specified. Valid options are 'lite', 'full', or 'heavy'."
                    exit 1
                    ;;
            esac
            shift
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Create virtual environment (if not already created)
if [ ! -d ".3dv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .3dv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Exiting."
        echo "Tip: Make sure python is available, i.e. activate conda base environment"
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .3dv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

# Install dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install --upgrade pip
# python -m pip install mediapipe
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Exiting."
    exit 1
fi

# Create models folder if it doesn't exist
mkdir -p models

# Change directory to models folder
cd models || exit

echo "Starting download of body-pose models..."

# Download files
if [ "$DOWNLOAD_LITE" = true ]; then
    echo "Downloading Pose landmarker (lite)..."
    curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
fi

if [ "$DOWNLOAD_FULL" = true ]; then
    echo "Downloading Pose landmarker (full)..."
    curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
fi

if [ "$DOWNLOAD_HEAVY" = true ]; then
    echo "Downloading Pose landmarker (heavy)..."
    curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
fi

echo "Download completed."

echo "Setup completed. Use 'source .3dv/bin/activate' to activate virtual environment."