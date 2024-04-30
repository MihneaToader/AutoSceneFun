import json
import numpy as np
import argparse
import os

# Visualisation
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

from tqdm import tqdm

DEBUGG = False


class Pose:
    def __init__(self, timestamp, joint_names, joint_data):
        self.timestamp = timestamp
        self.joint_names = joint_names
        self.joints = np.array(joint_data) # Contains Point3D objects

    @staticmethod
    def load_from_data(data, timestamp):
        if 'Landmark' not in data:
            raise KeyError("Pose only supports 'Landmark'-data, no relative poses.")

        joint_names_path = 'models/pose_landmarkers.json'
        with open(joint_names_path, 'r') as file:
            joint_names = json.load(file).values()

        joint_data = [data['Landmark'][str(idx)] for idx in range(len(joint_names))]
        joint_data = [[j['x'], j['y'], j['z']] for j in joint_data]
        return Pose(timestamp, list(joint_names), joint_data)

    def spo_between_eyes(self):
        left_eye_idx = [self.joint_names.index(name) for name in ['left_eye_inner', 'left_eye', 'left_eye_outer']]
        right_eye_idx = [self.joint_names.index(name) for name in ['right_eye_inner', 'right_eye', 'right_eye_outer']]

        left_eye = np.mean(self.joints[left_eye_idx], axis=0)
        right_eye = np.mean(self.joints[right_eye_idx], axis=0)

        center = (left_eye + right_eye) / 2

        self.joints -= center  # Shift all joints to center the pose between the eyes

    def pose_to_dict(self):
        return {'Landmark': {i: {'x': lm[0], 'y': lm[1], 'z': lm[2]} for i, lm in enumerate(self.joints)}}
    
    def debug_draw(self, org_img_path, output_path):
        draw_landmarks_on_image(self.joints, org_img_path, output_path)

    def __repr__(self):
        return f"Pose({', '.join(f'{name}={point}' for name, point in zip(self.joint_names, self.joints))})"



def process_data(args):

    # NOTE: Same processing for videos and images
    processed_folder = os.path.join(args.output, "processed")
    debugg_folder = os.path.join(args.output, "debugg")

    # Check if --data is a folder
    if os.path.isdir(args.data):
        # Iterate over all files in folder
        for file in os.listdir(args.data):
            if file.endswith(".json") and not "_p." in file and not "landmarks" in file:

                file_path = os.path.join(args.data, file)

                # Get output path
                file_name = file.split('.')[0]
                output_path = os.path.join(processed_folder, file_name + "_p.json")

                # Load data
                with open(file_path, 'r') as f:
                    data = json.load(f)

                processed_data = {}

                # Iterate over all timestamps
                for timestamp, d in tqdm(data.items(), desc=f"Processing {file_name}, timestamp"):
                    p = Pose.load_from_data(d, timestamp)

                    # Shift pose origin to between eyes (change with desired pose shift function)
                    p.spo_between_eyes()

                    if args.mode == "Image" and DEBUGG:
                        output_path = os.path.join(debugg_folder, file_name + f"_debugg.jpg")
                        p.debug_draw()

                    # Save data
                    processed_data[timestamp] = p.pose_to_dict()

                with open(output_path, 'w') as file:
                    json.dump(processed_data, file, indent=4)
                    print(f"Processed data saved at {output_path}")

    # Data is single file
    elif os.path.isfile(args.data):
        # Check if path is a json file
        if not args.data.endswith('.json'):
            raise ValueError(f"File at {args.data} is not a json file")
        
        if "_p." in args.data:
            print(f"File at {args.data} is already processed")
            return
        
        file_path = args.data

        # Get output path
        file_name = args.data.split('/')[-1][:-5]
        output_path = os.path.join(processed_folder, file_name + "_p.json")

        # Load data
        with open(file_path, 'r') as f:
            data = json.load(f)

        processed_data = {}

        # Iterate over all timestamps
        for timestamp, d in tqdm(data.items(), desc=f"Processing {file_name}, timestamp"):
            
            p = Pose.load_from_data(d, timestamp)
            p.spo_between_eyes()
            processed_data[timestamp] = p.pose_to_dict()

        with open(output_path, 'w') as file:
            json.dump(processed_data, file, indent=4)
            print(f"Processed data saved at {output_path}")


def draw_landmarks_on_image(new_pose_world_landmarks_list, org_img_path, output_path):
    """NOTE: WORKS FOR IMAGES ONLY!"""

    rgb_image = mp.Image.create_from_file(org_img_path)
    bgr_image = rgb_image.numpy_view()[:, :, :3]
    annotated_image = np.copy(bgr_image)

    org_img_name = org_img_path.split('/')[-1].split('.')[0]

    with open(f'output/raw/{org_img_name}.json', 'r') as file:
        joints = json.load(file)

    pose_landmarks_list = joints.get("pose_landmarks")

    # Get landmarks in world coords (debuggin)
    pose_world_landmarks_list = new_pose_world_landmarks_list

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

    world_landmarks = [
        landmark_pb2.Landmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in pose_world_landmarks_list
    ]

    # Add position text near each landmark.
    for landmark, world_landmark in zip(norm_landmarks, world_landmarks):
        x, y = int(landmark.x * annotated_image.shape[1]), int(landmark.y * annotated_image.shape[0])
        cv2.putText(annotated_image, f'({world_landmark.x:.2f}, {world_landmark.y:.2f}, {world_landmark.z:.2f})', (x, y -20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) # 1.0 is font size, 2 is thickness
        
    bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, bgr_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default="output/raw", help="Path to image, video or stream data folder")
    parser.add_argument("--mode", type=str, default="Image", help="Choose mode [Image, Video]")
    parser.add_argument("-out", "--output", type=str, default="output", help="Path to save output")
    args = parser.parse_args()

    # Check if output is folder and not file
    if not os.path.isdir(args.output):
        raise ValueError(f"Output path {args.output} is not a folder")
    
    else:
        # Check if output folder exists, else create it
        if not os.path.exists(args.output):
            os.makedirs(args.output)

            # Add processed folder
            os.makedirs(os.path.join(args.output, "processed"))
        
        else:
            # Add output folder for processed data
            processed_folder = os.path.join(args.output, "processed")
            if not os.path.exists(processed_folder):
                os.makedirs(processed_folder)

    # TODO: apply new folder structure to code!

    process_data(args)
    
if __name__ == "__main__":
    main()