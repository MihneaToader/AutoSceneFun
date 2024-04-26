import json
import numpy as np
import argparse
import os

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"
    
    def __add__(self, other):
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __truediv__(self, other):
        return Point3D(self.x / other, self.y / other, self.z / other)
    

class Pose:
    def __init__(self, timestamp, **joints):
        self.timestamp = timestamp
        
        # Set joint attribute dynamically
        for joint_name, point in joints.items():
            setattr(self, joint_name, point)

    @staticmethod
    def load_from_data(data, timestamp):

        # Read available joint names
        joint_names_path = 'models/pose_landmarkers.json'
        with open(joint_names_path, 'r') as file:
            f =  json.load(file)

        joint_names = [name for name in f.values()]

        # Create joints dictionary
        joints = {
            joint_name: Point3D(data['Landmark'][str(idx)]['x'], data['Landmark'][str(idx)]['y'], data['Landmark'][str(idx)]['z'])
            for idx, joint_name in enumerate(joint_names)
        }
        return Pose(timestamp, **joints)

    def get_joint(self, joint_name):
        # Return None if joint does not exist
        return getattr(self, joint_name, None)
    
    def spo_between_eyes(self):
        left_eye = (self.left_eye_inner + self.left_eye + self.left_eye_outer)/3
        right_eye = (self.right_eye_inner + self.right_eye + self.right_eye_outer)/3

        vec_to_new_ori = (left_eye + right_eye)/2

        # TODO: add vec to each datapoint


    
    def __repr__(self) -> str:
        pass

def process_data(args, data):
    
    # Process Images
    if args.mode.lower() == "image":
        # Get first key in data
        for timestamp, d in data.items():
            p = Pose.load_from_data(d, timestamp)
       
    elif args.mode.lower() == 'video':
        for timestamp, d in data.items():
            p = Pose.load_from_data(d, timestamp)



def load_data(data_path):
    # Check if path is a json file
    if not data_path.endswith('.json'):
        raise ValueError(f"File at {data_path} is not a json file")
    
    # Load json file
    with open(data_path, 'r') as f:
        data = json.load(f)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default="output/verona.json", help="Path to image, video or stream data folder")
    parser.add_argument("--mode", type=str, default="Image", help="Choose mode [Image, Video]")
    parser.add_argument("-out", "--output", type=str, default="output", help="Path to save output")
    args = parser.parse_args()
    
    # Check if output folder exists, else create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load data
    data = load_data(args.data)

    # Process data
    process_data(args, data)
    
if __name__ == "__main__":
    main()