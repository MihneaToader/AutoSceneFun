import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

from utils import OUTPUT_DIR
import json


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Get landmarks in world coords (debuggin)
    pose_world_landmarks_list = detection_result.pose_world_landmarks

    # Loop through the detected poses to visualize.
    for pose_landmarks, pose_world_landmarks in zip(pose_landmarks_list, pose_world_landmarks_list):
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        norm_landmarks = [
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ]
        pose_landmarks_proto.landmark.extend(norm_landmarks)
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())

        world_landmarks = [
            landmark_pb2.Landmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_world_landmarks
        ]

        # Add position text near each landmark.
        for landmark, world_landmark in zip(norm_landmarks, world_landmarks):
            x, y = int(landmark.x * annotated_image.shape[1]), int(landmark.y * annotated_image.shape[0])
            cv2.putText(annotated_image, f'({world_landmark.x:.2f}, {world_landmark.y:.2f}, {world_landmark.z:.2f})', (x, y -20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) # 1.0 is font size, 2 is thickness
        
    return annotated_image


def draw_processed_landmarks_on_image(new_pose_world_landmarks_list, org_img_path, output_path):
    """NOTE: WORKS FOR IMAGES ONLY!"""

    # Load image
    rgb_image = mp.Image.create_from_file(org_img_path)
    
    bgr_image = rgb_image.numpy_view()[:, :, :3]
    annotated_image = np.copy(bgr_image)

    org_img_name = os.path.splitext(os.path.basename(org_img_path))[0]

    joints_path = os.path.join(OUTPUT_DIR, "body_pose", "debugg", "landmarks", f"{org_img_name}_landmarks.json")

    with open(joints_path, 'r') as file:
        joints = json.load(file)

    pose_landmarks_list = joints.get("pose_landmarks")

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    
    norm_landmarks = [
        landmark_pb2.NormalizedLandmark(x=landmark['x'], y=landmark['y'], z=landmark['z']) for landmark in pose_landmarks_list
    ]
    pose_landmarks_proto.landmark.extend(norm_landmarks)

    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())

    # Get world landmarks for debugging
    world_landmarks = [
        landmark_pb2.Landmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in new_pose_world_landmarks_list
    ]

    # Add position text near each landmark.
    for landmark, world_landmark in zip(norm_landmarks, world_landmarks):
        x, y = int(landmark.x * annotated_image.shape[1]), int(landmark.y * annotated_image.shape[0])
        cv2.putText(annotated_image, f'({world_landmark.x:.2f}, {world_landmark.y:.2f}, {world_landmark.z:.2f})', (x, y -20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) # 1.0 is font size, 2 is thickness
    
    bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, bgr_image)


# Used in sync_hand_poses.py to visualise hands after transformation
def plot_hand_projections(np_relevant_meta_data, np_relevant_bodypose_data, transformed_bodypose_data):
    """(Helper function)"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot MetaQuest data
    ax.scatter(np_relevant_meta_data[:, 0], np_relevant_meta_data[:, 1], np_relevant_meta_data[:, 2], c='r', label='MetaQuest Data')

    # Plot BodyPose data
    ax.scatter(np_relevant_bodypose_data[:, 0], np_relevant_bodypose_data[:, 1], np_relevant_bodypose_data[:, 2], c='b', label='BodyPose Data')

    # Plot Transformed BodyPose data
    ax.scatter(transformed_bodypose_data[:, 0], transformed_bodypose_data[:, 1], transformed_bodypose_data[:, 2], c='g', label='Transformed BodyPose Data')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()