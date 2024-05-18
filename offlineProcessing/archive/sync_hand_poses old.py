import json
import os
from datetime import datetime, timezone, timedelta
from dateutil import tz
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

# Local modules
import utils
from body_pose.visualisation import plot_hand_projections


# NOTE: ADAPT FORMAT_DATA FUNCTION TO NEW DATA FORMAT
# NOTE 2: THIS SCRIPT ONLY MAPS DATA USING THE LEFT HAND!


# TODO: output processed data starting at common timestamp t

def _format_meta_data(data) -> json:
    """
    (Helper function) Bring data in compatible format for further processing
    NOTE: Edit funciton as MetaQuest-data changes in format
    """
    
    # Adjust timestamps (convert to unix timestamp)
    for t in data:
        # print(t['Timestamp'])
        base_str = t['Timestamp'][:t['Timestamp'].rfind(':')]
        millies = t['Timestamp'][t['Timestamp'].rfind(':')+1:]
        
        base_int = datetime.strptime(base_str, "%Y-%m-%d %H:%M:%S")
        
        # Assemble unix timestamp
        unix = base_int.replace(tzinfo=tz.tzlocal()).timestamp() + int(millies) / 1000

        data[data.index(t)]["Timestamp"] = unix

    # Position map for renaming
    position_map = {
        "PositionX": "x", "PositionY": "y", "PositionZ": "z",
        "RotationW": "rw", "RotationX": "rx", "RotationY": "ry", "RotationZ": "rz"
    }
    # Joints map for renaming
    with open(os.path.join(utils.BODY_POSE_DIR, "hand_mapping.json"), 'r') as file:
        joints_mapping = json.load(file)

    new_dataset = {}

    # Rename joints and axis labels
    for t in data:
        new_joints = {}
        joints = t['Position_rotation']
        
        for joint, values in joints.items():
            # Rename axis labels
            new_values = {position_map.get(key, key): value for key, value in values.items()}
            
            # Rename joint and assign the new values
            new_joints[joints_mapping.get(joint, joint)] = new_values
        
        # # Replace old joints with new joints
        # t['Position_rotation'] = new_joints

        # Save data to new dataset
        new_dataset[t['Timestamp']] = new_joints

    return new_dataset

def _get_data_creation_date(data_path):
    """(Helper function) Get creation time of video from metadata, including milliseconds if available."""

    try:  
        timestamp = os.stat(data_path).st_birthtime

        return timestamp
    
    except Exception as e:
        print("Warning: Unsupported media type.")
        return None
    
def _dict_to_numpy(data):
    """(Helper function) Converts a dictionary of coordinates to a NumPy array."""
    coordinates = []
    for key, coords in data.items():
        # Ensure 'x', 'y', 'z' are in the coords dictionary
        if all(k in coords for k in ['x', 'y', 'z']):
            coordinates.append((coords['x'], coords['y'], coords['z']))
    
    # Convert list of tuples to a NumPy array
    return np.array(coordinates)

def _calculate_distance(coord1, coord2):
    """(Helper function) Calculate the Euclidean distance between two 3D points."""
    return math.sqrt((coord1['x'] - coord2['x']) ** 2 + 
                     (coord1['y'] - coord2['y']) ** 2 + 
                     (coord1['z'] - coord2['z']) ** 2)

def get_metaquest_data_at_t(data_metaquest, t):
    """
    Returns data +- 1 sec around a specific timestamp
    """
    # Find the index of the timestamp
    one_second = 1  # One second in terms of timestamp units
    data_at_t = []

    # Early exit if the timestamp is out of the range of the data
    if data_metaquest:
        min_time = data_metaquest[0]["Timestamp"]
        max_time = data_metaquest[-1]["Timestamp"]
        if t < min_time - one_second or t > max_time + one_second:
            print("Timestamp out of range")
            return data_at_t

    # Iterate through each entry and check if it's within the ±1 second range
    for entry in data_metaquest:
        if t - one_second <= entry["Timestamp"] <= t + one_second:
            data_at_t.append(entry)

    return data_at_t

def get_bodypose_data_at_t(data_bodypose, t):
    return data_bodypose[str(t)]

def get_first_common_t(data_metaquest, data_bodypose, delta_ms):
    """Returns first common timestamp within threshold delta_ms."""

    delta_ms = delta_ms/1000

    # Iterate through each timestamp in the first dataset
    for entry1 in data_metaquest:
        t_meta = entry1['Timestamp']

        # Check each timestamp in the second dataset
        for t_body in data_bodypose.keys():
            # Compare the timestamps within the allowed delta
            if abs(t_meta - float(t_body)) <= delta_ms:
                return t_meta, t_body

    return None

def get_matching_timestamps_from_t(data_metaquest, data_bodypose, first_common_t, delta):
    """
    Match data to the nearest target time.

    retunrs: List[tuple(t_metaquest, t_bodypose)]
    """

    t_bodypose = [float(t) for t in data_bodypose.keys() if float(t) >= first_common_t]
    t_metaquest = [entry['Timestamp'] for entry in data_metaquest if entry['Timestamp'] >= first_common_t]

    reg_idx = 0
    matched_results = []
    t_diff = []

    # Loop through the irregular timestamps and match to the nearest target time
    for t_meta in t_metaquest:
        # Advance the regular index to find the closest regular timestamp
        while reg_idx < len(t_bodypose) - 1 and abs(t_bodypose[reg_idx + 1] - t_meta) < abs(t_bodypose[reg_idx] - t_meta):
            reg_idx += 1

        if reg_idx < len(t_bodypose) and abs(t_bodypose[reg_idx] - t_meta) <= delta/1000:
            matched_results.append((t_meta, t_bodypose[reg_idx]))
            
            # Keep track of time differences
            t_diff.append(abs(t_bodypose[reg_idx] - t_meta))       
    
    return matched_results, t_diff


def get_optimal_meta_timestamp(data_metaquest, data_bodypose):
    """
    Finds the timestamp in the detailed dataset that has the closest match to the simple dataset.
    
    :return: (Closest timestamp meta, relevant meta data, min distance)
    """

    # NOTE: THIS CODE MIGHT BE PRONE TO ROTATIONS!!!

    min_distance = float('inf')
    closest_timestamp_meta = None
    rel_meta_data = {}

    # Joint mapping based on assumed similarity or logical match
    joint_mapping = {
        '16': 'Wrist',  # Example mapping, adjust according to actual joint correspondence
        '18': 'Hand_PinkyProximal',  # Assuming you map these based on your knowledge of the datasets
        '20': 'Hand_MiddleProximal',
    }

    for entry in data_metaquest:
        total_distance = 0
        for simple_key, detailed_key in joint_mapping.items():
            if detailed_key in entry['Position_rotation']:
                total_distance += _calculate_distance(data_bodypose[simple_key], entry['Position_rotation'][detailed_key])
        
        if total_distance < min_distance:
            min_distance = total_distance
            closest_timestamp_meta = entry['Timestamp']
            rel_meta_data = {simple_key: entry['Position_rotation'][joint_mapping[simple_key]] for simple_key in joint_mapping.keys()}


    return closest_timestamp_meta, rel_meta_data, min_distance





def kabsch_scaling(points_metaquest, points_bodypose):
    """
    Calculates the optimal rotation matrix and scaling factor to align two sets of points.
    
    :param points_metaquest: Nx3 numpy array of points (simple data).
    :param points_bodypose: Nx3 numpy array of corresponding points (detailed data).
    :return: Tuple containing the rotation matrix, scaling factor, and translation vector.
    """
    
    # Step 1: Translate points to their centroids
    centroid_p = np.mean(points_metaquest, axis=0)
    centroid_q = np.mean(points_bodypose, axis=0)
    p_centered = points_metaquest - centroid_p
    q_centered = points_bodypose - centroid_q
    
    # Step 2: Compute the covariance matrix
    H = np.dot(p_centered.T, q_centered)

    U, S, Vt = np.linalg.svd(H) # Singular Value Decomposition

    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    scale = np.sum(S) / np.sum(p_centered ** 2)
    t = - np.dot(R,centroid_q.T) + centroid_p.T

    return R, scale, t


def transform_points(points, R, scale, t):
    """
    Applies the rotation matrix, scaling factor, and translation vector to the points.
    """
    scaled_points = points * scale
    rotated_points = np.dot(scaled_points, R.T)
    transformed_points = rotated_points + t.T
    return transformed_points


def _output_data(data, output_path):
    """(Helper function) Output the transformed data to a JSON file."""
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)


def process_data(meta_data_path, body_pose_path, output_path, delta_ms=30):

    # =====================
    # Step 1: Load the data
    # =====================

    # Load LEFT HAND from metaquest data
    left_hand_data_filename = [i for i in os.listdir(meta_data_path) if "left" and "hand" in i.lower()][0]
    left_hand_data_path = os.path.join(meta_data_path, left_hand_data_filename)
    with open(left_hand_data_path, 'r') as file:
        data_lefthand = json.load(file)
        data_lefthand = _format_meta_data(data_lefthand['Entries']) # Remove Entries if not in data anymore

        # _output_data(data_lefthand, os.path.join(output_path, "meta_lefthand.json"))

    # Load body pose data
    with open(body_pose_path, 'r') as file:
        data_bodypose = json.load(file)

        # Check if the data is in the correct format
        assert data_bodypose['landmark_type'].lower() == 'landmark', f"Landmark type is not 'landmark' but {data_bodypose['landmark_type']}. This script only supports world poses."
        data_bodypose.pop('landmark_type')
        
        # _output_data(data_bodypose, os.path.join(output_path, "bodypose.json"))

    # ============================
    # Step 2: Synchronize the data
    # ============================

    # Goal is to find the first common timestamp of the two datasets
    # Common timestamp = first datapoint of bodypose and metaquest that are within delta_ms of each other
    first_t_meta, first_t_body = get_first_common_t(data_lefthand, data_bodypose, delta_ms)

    # Get data at first common timestamp
    # MetaQuest data is gathered around 1 sec before and after the timestamp
    # This is due to the exact timestamp of the MetaQuest (bodypose only second precision)
    metaquest_data_at_t = get_metaquest_data_at_t(data_lefthand, first_t_meta)
    body_pose_data_at_t = get_bodypose_data_at_t(data_bodypose, first_t_body)

    
    # Get timestamp of MetaQuest data that is closest to the bodypose data (min. distance of relevant joints)
    closest_t_meta, relevant_meta_data, _ = get_optimal_meta_timestamp(metaquest_data_at_t, body_pose_data_at_t)
    # print(f"Best matching MetaQuest timestamp around first common timestamp: {closest_t_meta} with min_distance: {min_distance}")

    # Isolate timestamps of common data points (within delta_ms of each other), starting from the first common timestamp
    # TODO: IS FIRST COMMON TIMESTAMP INCLUDED IN COMMON T?
    common_t, _ = get_matching_timestamps_from_t(data_lefthand, data_bodypose, closest_t_meta, delta_ms)

    # for debugging
    db = []
    da = []

    # Iterate over all common timestamps and calculate the optimal transformation
    for idx, (t_meta, t_body) in enumerate(common_t):

        metaquest_data_at_t = get_metaquest_data_at_t(data_lefthand, t_meta)
        body_pose_data_at_t = get_bodypose_data_at_t(data_bodypose, t_body)

        _, relevant_meta_data, _ = get_optimal_meta_timestamp(metaquest_data_at_t, body_pose_data_at_t)

        # ================================
        # Step 3: Calculate transformation
        # ================================

        # 15: left_Wrist, 17: left_Hand_PinkyProximal, 19: Hand_MiddleProximal
        # only keep left hand data of bodypose = keep keys of list
        relevant_bodypose_data_at_t = {key: value for key, value in body_pose_data_at_t.items() if key in ['15', '17', '19']}
        
        # Convert dictionaries to numpy arrays
        np_relevant_meta_data = _dict_to_numpy(relevant_meta_data)
        np_relevant_bodypose_data = _dict_to_numpy(relevant_bodypose_data_at_t)

        # print("Before transformation distances:", np.linalg.norm(np_relevant_meta_data - np_relevant_bodypose_data, axis=1))

        # Calculate the optimal rotation matrix, scaling factor, and translation vector
        R, scale, t = kabsch_scaling(np_relevant_meta_data, np_relevant_bodypose_data)

        # print(f"Rotation matrix:\n{R}")
        # print(f"Scaling factor: {scale}")
        # print(f"Translation vector: {t}")

        transformed_bodypose_data = transform_points(np_relevant_bodypose_data, R, scale, t)

        # if idx == 0:
        #     plot_hand_projections(np_relevant_meta_data, np_relevant_bodypose_data, transformed_bodypose_data)

        # print("After transformation distances:", np.linalg.norm(np_rel_meta_data - transformed_body_pose_data, axis=1))

        # ====================================
        # (DEBUGGING) Calculate mean distances
        # ====================================

        distances_before = np.abs(np_relevant_meta_data - np_relevant_bodypose_data)
        distances_after = np.abs(np_relevant_meta_data - transformed_bodypose_data)

        db.append(distances_before)
        da.append(distances_after)

    db = np.concatenate(db, axis=0)
    da = np.concatenate(da, axis=0)

    print(f"Mean distance before transformation: {np.mean(db, axis=0)}")
    print(f"Mean distance after transformation: {np.mean(da, axis=0)}")



def main():

    parser = argparse.ArgumentParser(description="Synchronize hand pose data from MetaQuest and body pose data.")
    parser.add_argument("--data_csv", type=str, default=os.path.join(utils.DATA_DIR, "data_mapping.csv"),
                        help="Path to the MetaQuest left hand pose data file.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(utils.OUTPUT_DIR, "body_pose", "final_recordings"),
                        help="Path to the output directory.")
    parser.add_argument("--delta", type=int, default=30,
                        help="Time difference threshold in milliseconds.")

    args = parser.parse_args()

    # Check if data mapping file exists
    if not os.path.isfile(args.data_csv):
        print(f"Error: Data file not found at {args.data_csv}.")
        exit(1)

    # Ensure that output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mappings = pd.read_csv(args.data_csv, index_col="id")

    for index, recording in mappings.iterrows():
        
        # =====================================
        # Set paths and ensure folder structure
        # =====================================

        if os.path.exists(recording["meta_data"]):
            meta_data_path = recording["meta_data"]
        else:
            meta_data_path = os.path.join(utils.DATA_DIR, "meta_quest", recording["meta_data"])

        if os.path.exists(recording["body_pose_media"]):
            body_pose_path = recording["body_pose_media"]
        else:
            body_pose_path = os.path.join(utils.OUTPUT_DIR, "body_pose", "raw", recording["body_pose_media"].split(".")[0] + ".json")
    
        # Set output path
        if pd.isna(recording["final_name"]) or recording["final_name"].strip() == "":
            output_path = os.path.join(args.output_dir, f"session_{index}")
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = os.path.join(args.output_dir, recording["final_name"])
            os.makedirs(output_path, exist_ok=True)

        process_data(meta_data_path, body_pose_path, output_path, args.delta)

if __name__ == "__main__":
    main()