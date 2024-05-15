import json
import os
from datetime import datetime, timezone, timedelta
from dateutil import tz
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Local modules
import utils


# NOTE: ADAPT FORMAT_DATA FUNCTION TO NEW DATA FORMAT


def _format_data(data) -> json:
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

    # Rename joints and axis labels
    for t in data:
        new_joints = {}
        joints = t['Position_rotation']
        
        for joint, values in joints.items():
            # Rename axis labels
            new_values = {position_map.get(key, key): value for key, value in values.items()}
            
            # Rename joint and assign the new values
            new_joints[joints_mapping.get(joint, joint)] = new_values
        
        # Replace old joints with new joints
        t['Position_rotation'] = new_joints

    return data

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
    return data_bodypose[str(t)]["Landmark"]

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




def get_optimal_timestamp_of_sequence(data_metaquest, data_bodypose):
    """
    Finds the timestamp in the detailed dataset that has the closest match to the simple dataset.
    
    :return: The timestamp of the closest match and the minimum distance.
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


    return closest_timestamp_meta, rel_meta_data ,min_distance





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

    U, S, Vt = np.linalg.svd(H)

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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Synchronize hand pose data from MetaQuest and body pose data.")
    parser.add_argument("--data_csv", type=str, default=os.path.join(utils.DATA_DIR, "hand_meta_quest"),
                        help="Path to the MetaQuest left hand pose data file.")
    parser.add_argument("--data_body", type=str, default=os.path.join(utils.OUTPUT_DIR, "body_pose", "processed"),  
                        help="Path to the body pose data file.")



    with open(os.path.join(utils.DATA_DIR, "hand_meta_quest", "2024-05-09 13:39:27nd Tracking leftHand Tracking left (OVRHand)_hand_data.json"), 'r') as file:
        data = json.load(file)

    entries = _format_data(data["Entries"])


    video = "data/media/IMG_0821.MOV"

    birth_time = _get_data_creation_date(video)

    with open("output/body_pose/processed/1715254764.0_p.json", 'r') as file:
        data_bodypose = json.load(file)


    delta = 30
    first_t_meta, first_t_body = get_first_common_t(entries, data_bodypose, delta)

    # print(f"First common timestamp: meta: {first_t_meta}, body: {first_t_body}")

    # Find first timestamp where hands overlap, use this timestamp as the first common timestamp
    metaquest_data_at_t = get_metaquest_data_at_t(entries, first_t_meta)
    body_pose_data_at_t = get_bodypose_data_at_t(data_bodypose, first_t_body)

    # 16: Wrist, 18: Hand_PinkyProximal, 20: Hand_MiddleProximal
    rel_body_pose_data_at_t = {key: value for key, value in body_pose_data_at_t.items() if key in ['16', '18', '20']}

    # only keep right hand data of bodypose = keep keys of list

    closest_t_meta, rel_meta_data, min_distance = get_optimal_timestamp_of_sequence(metaquest_data_at_t, body_pose_data_at_t)
    # print(f"Best matching MetaQuest timestamp around first common timestamp: {closest_t_meta} with min_distance: {min_distance}")

    np_rel_meta_data = _dict_to_numpy(rel_meta_data)
    np_body_pose_data = _dict_to_numpy(rel_body_pose_data_at_t)


    print("Before transformation distances:", np.linalg.norm(np_rel_meta_data - np_body_pose_data, axis=1))

    R, scale, t = kabsch_scaling(np_rel_meta_data, np_body_pose_data)

    print(f"Rotation matrix:\n{R}")
    print(f"Scaling factor: {scale}")
    print(f"Translation vector: {t}")

    transformed_body_pose_data = transform_points(np_body_pose_data, R, scale, t)

    print("After transformation distances:", np.linalg.norm(np_rel_meta_data - transformed_body_pose_data, axis=1))


    # Isolate all frames that are close
    common_t, differences = get_matching_timestamps_from_t(entries, data_bodypose, closest_t_meta, delta)

    db = []
    da = []

    for t_meta, t_body in common_t:
        metaquest_data_at_t = get_metaquest_data_at_t(entries, t_meta)
        body_pose_data_at_t = get_bodypose_data_at_t(data_bodypose, t_body)

        rel_body_pose_data_at_t = {key: value for key, value in body_pose_data_at_t.items() if key in ['16', '18', '20']}
        _, rel_meta_data, _ = get_optimal_timestamp_of_sequence(metaquest_data_at_t, body_pose_data_at_t)

        np_rel_meta_data = _dict_to_numpy(rel_meta_data)
        np_body_pose_data = _dict_to_numpy(rel_body_pose_data_at_t)

        R, scale, t = kabsch_scaling(np_rel_meta_data, np_body_pose_data)

        # transformed_body_pose_data = transform_points(_dict_to_numpy(body_pose_data_at_t), R, scale, t)
        transformed_body_pose_data = transform_points(np_body_pose_data, R, scale, t)

        distances_before = np.abs(np_rel_meta_data - np_body_pose_data)
        distances_after = np.abs(np_rel_meta_data - transformed_body_pose_data)

        db.append(distances_before)
        da.append(distances_after)


    db = np.concatenate(db, axis=0)
    da = np.concatenate(da, axis=0)

    print(f"Mean distance before transformation: {np.mean(db, axis=0)}")
    print(f"Mean distance after transformation: {np.mean(da, axis=0)}")