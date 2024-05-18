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
# TODO: (OPTIONAL) update match_data_around_d in a way that it chooses the right t based on the average distance after processing

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


def _calculate_distance(coord1:np.array, coord2:np.array):
    """(Helper function) Calculate the Euclidean distance between two 3D points."""
    return np.linalg.norm(coord1 - coord2)

def _get_data_at_t(data, t):
    return {t: data[str(t)]}

def get_data_around_t(data:dict, timestamp) -> dict:
    """
    Returns data +- 1 sec around a specific timestamp
    """
    # Find the index of the timestamp
    one_second = 1  # One second in terms of timestamp units
    data_around_t = {}

    # Early exit if the timestamp is out of the range of the data
    if data:
        timestamps = list(data.keys())
        min_time = timestamps[0]
        max_time = timestamps[-1]
        if timestamp < min_time - one_second or timestamp > max_time + one_second:
            print("Timestamp out of range")
            return data_around_t

    # Iterate through each entry and check if it's within the ±1 second range
    for t, d in data.items():
        if timestamp - one_second <= t <= timestamp + one_second:
            data_around_t[t] = d

    return data_around_t

def get_first_common_t(data1:dict, data2:dict, delta_ms) -> tuple:
    """
    Returns first common timestamp within threshold delta_ms.
    
    :return: (t1, t2) or None if no common timestamp is found.
    """

    # Convert delta to seconds
    delta_ms = delta_ms/1000

    # Iterate through each timestamp in the first dataset
    for t1 in data1.keys():
        # Check each timestamp in the second dataset
        for t2 in data2.keys():
            # Compare the timestamps within the allowed delta
            if abs(float(t1) - float(t2)) <= delta_ms:
                return float(t1), float(t2)

    return None

def _dict_to_numpy(data:dict):
    """(Helper function) Converts a dictionary of coordinates to a NumPy array."""
    coordinates = []

    for coords in data.values():
        coordinates.append((coords['x'], coords['y'], coords['z']))
    
    # Convert list of tuples to a NumPy array
    return np.array(coordinates)


def get_relevant_data(data:dict):
    """
    Returns only the relevant data for the transformation.
    """
    def _is_bodypose_data(d): # Helper function
        try:
            int(d)
            return True
        except ValueError:
            return False
    
    # Check if data is bodypose or metaquest
    is_bodypose = _is_bodypose_data(next(iter(data[next(iter(data))])))

    if is_bodypose:
        rel_data = {}
        for t, d in data.items():
            # Only keep left of hand data
            rel_data[t] = {key: value for key, value in d.items() if key in ['15', '17', '19']}

        return rel_data

    else:
        rel_data = {}
        for t, d in data.items():
            # Only keep left of hand data
            rel_data[t] = {key: value for key, value in d.items() if key in ['Wrist', 'Hand_PinkyProximal', 'Hand_MiddleProximal']}
        
        return rel_data


def sync_data(data_bodypose:dict, data_lefthand:dict, t_start_bodypose, t_start_lefthand):
    """
    Sets datasets into same time frame, starting from the first common timestamp

    :return: (data_bodypose, data_lefthand)
    """

    # Sychronise starts of data
    data_bodypose = {t: d for t, d in data_bodypose.items() if float(t) >= t_start_bodypose}
    data_lefthand = {t: d for t, d in data_lefthand.items() if float(t) >= t_start_lefthand}

    # Add difference of timestamps to bodypose timestamps to synchronise start
    delta_t = t_start_lefthand - t_start_bodypose
    new_data_bodypose = {round(1000*(float(t) + delta_t))/1000: d for t, d in data_bodypose.items()}

    return new_data_bodypose, data_lefthand


def match_data(data_lefthand:dict, data_bodypose:dict, delta_ms):
    """
    Finds point in time in both datasets where points are closest.
    
    :return: (data_lefthand, data_bodypose)
    """
    # NOTE: THIS CODE MIGHT BE PRONE TO ROTATIONS!!!

    # Goal is to find the first common timestamp of the two datasets
    # Common timestamp = first datapoint of bodypose and metaquest that are within delta_ms of each other
    first_t_meta, first_t_body = get_first_common_t(data_lefthand, data_bodypose, delta_ms)

    # Get data at first common timestamp
    # MetaQuest data is gathered around 1 sec before and after the timestamp
    # This is due to the exact timestamp of the MetaQuest (bodypose only second precision)
    metaquest_data_at_t = get_data_around_t(data_lefthand, first_t_meta)
    body_pose_data_at_t = _get_data_at_t(data_bodypose, first_t_body)

    r_meta_lefthand = get_relevant_data(metaquest_data_at_t)
    r_bodypose = get_relevant_data(body_pose_data_at_t)

    # Plan: transform data1 to data2 and calculate the distance for each timestamp
    # Return data for which distance is minimal
    distances = []

    for t2, d2 in r_bodypose.items():
        for t1, d1 in r_meta_lefthand.items():
            # Convert dictionaries to numpy arrays
            np_d1 = _dict_to_numpy(d1)
            np_d2 = _dict_to_numpy(d2)

            # Calculate the optimal rotation matrix, scaling factor, and translation vector
            R, scale, t = kabsch_scaling(np_d1, np_d2)

            transformed_d1 = transform_points(np_d1, R, scale, t)

            distances.append((_calculate_distance(transformed_d1, np_d2), t1, t2))

    # Find the minimum distance
    min_distance, closest_t_meta, closest_t_body = min(distances, key=lambda x: x[0])
    print(f"Min distance: {min_distance} at t_meta: {closest_t_meta} and t_body: {closest_t_body}")

    data_bodypose, data_lefthand = sync_data(data_bodypose, data_lefthand, closest_t_body, closest_t_meta)

    return data_lefthand, data_bodypose


def align_data(data1, data2, delta_ms):
    """
    Aligns two datasets based on their timestamps. Outputs datasets with common timestamps and same length.

    retunrs: List[tuple(t_metaquest, t_bodypose)], List[time_diff]
    """

    # First timestamp is already synchronised
    t1_list = [float(t) for t in data1.keys()]
    t2_list = [float(t) for t in data2.keys()]

    reg_idx = 0
    new_data1 = {}
    new_data2 = {}
    t_diff = []

    # Loop through the irregular timestamps and match to the nearest target time
    for t1 in t1_list:
        # Advance the regular index to find the closest regular timestamp
        while reg_idx < len(t2_list) - 1 and abs(t2_list[reg_idx + 1] - t1) < abs(t2_list[reg_idx] - t1):
            reg_idx += 1

        if reg_idx < len(t2_list) and abs(t2_list[reg_idx] - t1) <= delta_ms/1000:
            # Keep only data points that are within the delta_ms threshold
            new_data1[t1] = data1[t1]
            new_data2[t2_list[reg_idx]] = data2[t2_list[reg_idx]]
            
            # Keep track of time differences
            t_diff.append(abs(t2_list[reg_idx] - t1))
    
    return new_data1, new_data2, t_diff


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

    data_lefthand, data_bodypose = match_data(data_lefthand, data_bodypose, delta_ms)

    data_lefthand, data_bodypose, differences = align_data(data_lefthand, data_bodypose, delta_ms)

    print(f"mean time difference: {np.mean(differences)}")

    # Isolate timestamps of common data points (within delta_ms of each other), starting from the first common timestamp
    # common_t, _ = get_matching_timestamps_from_t(data_lefthand, data_bodypose, t_meta, delta_ms)

    # for debugging
    db = []
    da = []

    # Iterate over all common timestamps and calculate the optimal transformation
    for idx, ((t_lefthand, d_lefthand), (t_bodypose, d_bodypose)) in enumerate(zip(data_lefthand.items(), data_bodypose.items())):

        rel_data_lefthand = get_relevant_data({t_lefthand: d_lefthand})
        rel_data_bodypose = get_relevant_data({t_bodypose: d_bodypose})

        # ================================
        # Step 3: Calculate transformation
        # ================================
        
        # Convert dictionaries to numpy arrays

        np_rel_data_lefthand = _dict_to_numpy(next(iter(rel_data_lefthand.values())))
        np_rel_data_bodypose = _dict_to_numpy(next(iter(rel_data_bodypose.values())))

        # Calculate the optimal rotation matrix, scaling factor, and translation vector
        R, scale, t = kabsch_scaling(np_rel_data_lefthand, np_rel_data_bodypose)

        transformed_rel_data_bodypose = transform_points(np_rel_data_bodypose, R, scale, t)

        # if idx == 0:
            # plot_hand_projections(np_rel_data_lefthand, np_rel_data_bodypose, transformed_rel_data_bodypose)


        transformed_data_bodypose = transform_points(_dict_to_numpy(d_bodypose), R, scale, t)

        # ====================================
        # (DEBUGGING) Calculate mean distances
        # ====================================

        distances_before = np.abs(np_rel_data_lefthand - np_rel_data_bodypose)
        distances_after = np.abs(np_rel_data_lefthand - transformed_rel_data_bodypose)

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