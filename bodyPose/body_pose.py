# Pose estimation
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Visualisation
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow

import argparse
import os

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        
    return annotated_image

def get_body_pose(model_path, mode, data):
    # Load model
    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    # Set running mode
    if mode == "Image":
        running_mode = VisionRunningMode.IMAGE
    elif mode == "Video":
        running_mode = VisionRunningMode.VIDEO
    elif mode == "Stream":
        running_mode = VisionRunningMode.STREAM

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running_mode)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Process image
        results = landmarker.detect(data)

        # Get rid of alpha (transparency) channel
        bgr_image = data.numpy_view()[:, :, :3]
        annotated_image = draw_landmarks_on_image(bgr_image, results)
        # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # Convert RGB to BGR for displaying with OpenCV if necessary
        bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Annotated Image', bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="Image", help="Generate pose from Image, Video or Stream")
    parser.add_argument("--model", type=str, default="models/pose_landmarker_lite.task", help="Path to model")
    parser.add_argument("--data", type=str, default="data", help="Path to image, video or stream folder")
    parser.add_argument("--process_folder", type=bool, default=True, help="Process all files in folder")
    args = parser.parse_args()

    # Raise warning if mode not supported
    if args.mode != "Image":
        raise ValueError(f"{args.mode} not supported. Video and Stream to be implemented")
    
    # Load all images in folder if process_folder is True
    if args.process_folder:
        images = [i for i in os.listdir(args.data) if i.endswith(".jpg")]
        for image in images:
            i = mp.Image.create_from_file(os.path.join(args.data, image))
            get_body_pose(model_path=args.model, mode=args.mode, data=i)
    else:
        i = mp.Image.create_from_file(args.data)
        get_body_pose(model_path=args.model, mode=args.mode, data=i)
    



if __name__ == "__main__":
    main()
