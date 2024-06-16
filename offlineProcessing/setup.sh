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

# Create conda environment (if not already created)
if ! conda env list | grep -q '3dv'; then
    echo "Creating conda environment..."
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment. Exiting."
        echo "Tip: Make sure conda is available and the environment.yml file is correctly configured."
        exit 1
    fi
else
    echo "Conda environment already exists."
fi

# Activate the conda environment
echo "Activating conda environment..."
source activate 3dv
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment. Exiting."
    exit 1
fi

# Install open3d
echo "Installing open3d..."
conda install open3d -y
if [ $? -ne 0 ]; then
    echo "Failed to install open3d. Exiting."
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

echo "Setup completed. Use 'conda activate 3dv' to activate virtual environment."