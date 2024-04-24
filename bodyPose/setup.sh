#!/bin/bash

# Create models folder if it doesn't exist
mkdir -p models

# Change directory to models folder
cd models || exit

echo "Starting download of body-pose models..."

# Download files
echo "Downloading Pose landmarker (lite)..."
curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

echo "Downloading Pose landmarker (full)..."
curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

echo "Downloading Pose landmarker (heavy)..."
curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

echo "Download completed."
