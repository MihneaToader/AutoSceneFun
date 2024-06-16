import json
import os
from datetime import datetime
from dateutil import tz
import numpy as np

# Local modules
import utils
# from utils.tools.setup_folder_structure import _create_necessary_folders_sync_hand_poses
from body_pose.postprocess_meta_files import process_folder as post_process_data

# NOTE: exclude rotations in format_meta_data to speed up the code if needed


def _format_meta_data(data, l_r_c="left") -> json:
    """
    (Helper function) Bring data in compatible format for further processing
    NOTE: Edit funciton as MetaQuest-data changes in format
    """

    assert l_r_c in ["left", "right", "camera"], "Invalid 'l_r_c'. Choose 'left', 'right' or 'camera'."

    # Adjust timestamps (convert to unix timestamp)
    for t in data:
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

    new_dataset = {}

    if l_r_c in ['left', 'right']:
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
                new_joints[f"{l_r_c}_{joints_mapping.get(joint, joint)}"] = new_values

            # Save data to new dataset
            new_dataset[float(t['Timestamp'])] = new_joints

    elif l_r_c == 'camera':
        for t in data:
            new_head = {}
            head = t['Position_rotation']
            
            for values in head.values():
                # Rename axis labels
                new_head = {position_map.get(key, key): value for key, value in values.items()}

            # Save data to new dataset
            new_dataset[float(t['Timestamp'])] = {"head":new_head}

    return new_dataset


def _merge_n_datasets(delta_ms, *datasets):
    """
    Merges two datasets based on their timestamps.
    """

    # Check if at least one dataset is provided
    if not datasets:
        raise ValueError("At least one dataset must be provided")
    
    # Initialize the merged dataset with the first dataset
    merged_data = datasets[0]
    
    counter = 0
    # Iterate through the rest of the datasets
    for data in datasets[1:]:
        data1_timestamps = list(merged_data.keys())
        data2_timestamps = list(data.keys())
        
        for ts1 in data1_timestamps:
            ts1_float = float(ts1)
            
            for ts2 in data2_timestamps:
                ts2_float = float(ts2)
                
                if abs(ts1_float - ts2_float) * 1000 <= delta_ms:
                    merged_data[ts1].update(data[ts2])
                    counter += 1

                if ts2 > ts1: # Speed up code
                    break

    print(f"Merging datasets - found {counter} matches")
    
    return merged_data


def _calculate_distance(coord1:np.array, coord2:np.array):
    """(Helper function) Calculate the Euclidean distance between two 3D points."""
    return np.linalg.norm(coord1 - coord2)


def _get_data_at_t(data, t):
    return {t: data[str(t)]}


def prepare_points_for_transformation(data_bodypose, data_meta, tr="translation"):
    """
    Check that the data is in the correct format and extract the relevant points for the transformation.

    This function ensures that if data is missing, the corresponding points in the other dataset are also removed to assure correct input shapes.

    :param tr: either get data for translation or rotation

    :return: (points_meta, points_bodypose)
    """

    points_bodypose = []
    points_meta = []

    if tr == "translation":
        # meta_right_hand = []
        # body_right_hand = []
        # meta_left_hand = []
        # body_left_hand = []
        
        # for index, body_part in META_BODY_MAPPING.items():
        #     if index in data_bodypose and body_part in data_meta:
        #         if "left" in body_part:
        #             meta_left_hand.append([data_meta[body_part]['x'], data_meta[body_part]['y'], data_meta[body_part]['z']])
        #             body_left_hand.append([data_bodypose[index]['x'], data_bodypose[index]['y'], data_bodypose[index]['z']])
        #         elif "right" in body_part:
        #             meta_right_hand.append([data_meta[body_part]['x'], data_meta[body_part]['y'], data_meta[body_part]['z']])
        #             body_right_hand.append([data_bodypose[index]['x'], data_bodypose[index]['y'], data_bodypose[index]['z']])

        # points_bodypose.append(np.mean(body_left_hand, axis=0))
        # points_bodypose.append(np.mean(body_right_hand, axis=0))
        # points_meta.append(np.mean(meta_left_hand, axis=0))
        # points_meta.append(np.mean(meta_right_hand, axis=0))

        if 'head' in data_bodypose and 'head' in data_meta:
            points_bodypose.append([data_bodypose['head']['x'], data_bodypose['head']['y'], data_bodypose['head']['z']])
            points_meta.append([data_meta['head']['x'], data_meta['head']['y'], data_meta['head']['z']])
    
    elif tr == "rotation":
        # for index, body_part in META_BODY_MAPPING.items():
        #     if index in data_bodypose and body_part in data_meta:
        #         points_bodypose.append([data_bodypose[index]['x'], data_bodypose[index]['y'], data_bodypose[index]['z']])
        #         points_meta.append([data_meta[body_part]['x'], data_meta[body_part]['y'], data_meta[body_part]['z']])

        meta_right_hand = []
        body_right_hand = []
        meta_left_hand = []
        body_left_hand = []

        for index, body_part in META_BODY_MAPPING.items():
            if index in data_bodypose and body_part in data_meta:
                if "left" in body_part:
                    meta_left_hand.append([data_meta[body_part]['x'], data_meta[body_part]['y'], data_meta[body_part]['z']])
                    body_left_hand.append([data_bodypose[index]['x'], data_bodypose[index]['y'], data_bodypose[index]['z']])
                elif "right" in body_part:
                    meta_right_hand.append([data_meta[body_part]['x'], data_meta[body_part]['y'], data_meta[body_part]['z']])
                    body_right_hand.append([data_bodypose[index]['x'], data_bodypose[index]['y'], data_bodypose[index]['z']])

        points_bodypose.append(np.mean(body_left_hand, axis=0))
        points_bodypose.append(np.mean(body_right_hand, axis=0))
        points_meta.append(np.mean(meta_left_hand, axis=0))
        points_meta.append(np.mean(meta_right_hand, axis=0))

        if 'head' in data_bodypose and 'head' in data_meta:
            points_bodypose.append([data_bodypose['head']['x'], data_bodypose['head']['y'], data_bodypose['head']['z']])
            points_meta.append([data_meta['head']['x'], data_meta['head']['y'], data_meta['head']['z']])
    
    return np.array(points_meta), np.array(points_bodypose)


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

    raise ValueError("No common timestamp found within threshold.")


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

    # TODO: INCLUDE HEAD IN BOTH DATASETS
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
            head_joints_list = [str(i) for i in range(0, 9)]
            rel_joints = {key: value for key, value in d.items() if key in META_BODY_MAPPING.keys()}

            # Calculate head position
            head_joints = [value for key, value in d.items() if key in head_joints_list]
            head_joints = [[i['x'], i['y'], i['z']] for i in head_joints]
            head = np.mean(head_joints, axis=0)

            # Add head to relevant joints
            rel_joints['head'] = {'x': head[0], 'y': head[1], 'z': head[2]}
            rel_data[t] = rel_joints

    else: # if is metaquest data
        rel_data = {}
        for t, d in data.items():
            # Only keep left of hand data
            rel_joints = {key: value for key, value in d.items() if key in META_BODY_MAPPING.values()}
            rel_joints['head'] = d['head'].copy()
            rel_data[t] = rel_joints

    return rel_data


def sync_data(data_bodypose:dict, data_metaquest:dict, t_start_bodypose, t_start_lefthand):
    """
    Sets datasets into same time frame, starting from the first common timestamp

    :return: (data_bodypose, data_metaquest)
    """

    # Sychronise starts of data
    data_bodypose = {t: d for t, d in data_bodypose.items() if float(t) >= t_start_bodypose}
    data_metaquest = {t: d for t, d in data_metaquest.items() if float(t) >= t_start_lefthand}

    # Add difference of timestamps to bodypose timestamps to synchronise start
    delta_t = t_start_lefthand - t_start_bodypose
    new_data_bodypose = {round(1000*(float(t) + delta_t))/1000: d for t, d in data_bodypose.items()}

    return new_data_bodypose, data_metaquest


def match_data(data_metaquest:dict, data_bodypose:dict, delta_ms):
    """
    Finds point in time in both datasets where points are closest.
    
    :return: (data_metaquest, data_bodypose)
    """
    # NOTE: THIS CODE MIGHT BE PRONE TO ROTATIONS!!!

    # Goal is to find the first common timestamp of the two datasets
    # Common timestamp = first datapoint of bodypose and metaquest that are within delta_ms of each other
    first_t_meta, first_t_body = get_first_common_t(data_metaquest, data_bodypose, delta_ms)

    # Get data at first common timestamp
    # MetaQuest data is gathered around 1 sec before and after the timestamp
    # This is due to the exact timestamp of the MetaQuest (bodypose only second precision)
    metaquest_data_at_t = get_data_around_t(data_metaquest, first_t_meta)
    body_pose_data_at_t = _get_data_at_t(data_bodypose, first_t_body)

    r_meta= get_relevant_data(metaquest_data_at_t)
    r_bodypose = get_relevant_data(body_pose_data_at_t)

    # Plan: transform data1 to data2 and calculate the distance for each timestamp
    # Return data for which distance is minimal
    distances = []

    for t2, d2 in r_bodypose.items():
        for t1, d1 in r_meta.items():
            # Convert dictionaries to numpy arrays
            np_d1_t, np_d2_t = prepare_points_for_transformation(data_meta=d1, data_bodypose=d2, tr="translation")
            np_d1_r, np_d2_r = prepare_points_for_transformation(data_meta=d1, data_bodypose=d2, tr="rotation")

            # Calculate the optimal rotation matrix, scaling factor, and translation vector
            t = kabsch_scaling(np_d1_t, np_d2_t, get="translation")
            R = kabsch_scaling(np_d1_r, np_d2_r, get="rotation")

            transformed_d1 = transform_points(np_d1_r, R, t)

            distances.append((_calculate_distance(transformed_d1, np_d2_r), t1, t2))

    # Find the minimum distance
    min_distance, closest_t_meta, closest_t_body = min(distances, key=lambda x: x[0])
    # print(f"Min distance: {min_distance} at t_meta: {closest_t_meta} and t_body: {closest_t_body}")

    data_bodypose, data_metaquest = sync_data(data_bodypose, data_metaquest, closest_t_body, closest_t_meta)

    return data_metaquest, data_bodypose


def align_data(data1, data2, delta_ms):
    """
    Aligns two datasets based on their timestamps. Outputs datasets with common timestamps and same length.

    returns: (new_data1, new_data2, time_differences)
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


# def kabsch_scaling(points_metaquest, points_bodypose):
#     """
#     Calculates the optimal rotation matrix and scaling factor to align two sets of points.
    
#     :param points_metaquest: Nx3 numpy array of points (simple data).
#     :param points_bodypose: Nx3 numpy array of corresponding points (detailed data).
#     :return: Tuple containing the rotation matrix, scaling factor, and translation vector.
#     """

#     # print(f"Shape points_metaquest: {points_metaquest.shape}, shape points_bodypose: {points_bodypose.shape}")
#     # Step 1: Translate points to their centroids
#     centroid_meta = np.mean(points_metaquest, axis=0)
#     centroid_body = np.mean(points_bodypose, axis=0)
#     points_meta_centered = points_metaquest - centroid_meta
#     points_body_centered = points_bodypose - centroid_body
    
#     # Step 2: Compute the covariance matrix
#     H = np.dot(points_body_centered.T, points_meta_centered)

#     U, S, Vt = np.linalg.svd(H) # Singular Value Decomposition

#     R = np.dot(Vt.T, U.T)

#     # special reflection case
#     if np.linalg.det(R) < 0:
#         Vt[2, :] *= -1
#         R = np.dot(Vt.T, U.T)

#     t = centroid_meta - np.dot(R, centroid_body)

#     return R, t

def kabsch_scaling(points_metaquest, points_bodypose, get):
    """
    Calculates the optimal rotation matrix (around X and Z axes) and translation vector to align two sets of points.
    
    :param points_metaquest: Nx3 numpy array of points (simple data).
    :param points_bodypose: Nx3 numpy array of corresponding points (detailed data).
    :return: Tuple containing the rotation matrix and translation vector.
    """

    assert get in ["rotation", "translation"], "Invalid 'get'. Choose 'rotation' or 'translation'."

    centroid_meta = np.mean(points_metaquest, axis=0)
    centroid_body = np.mean(points_bodypose, axis=0)

    if get == "translation":
        return centroid_meta - centroid_body

    points_meta_centered = points_metaquest - centroid_meta
    points_body_centered = points_bodypose - centroid_body

    H = np.dot(points_body_centered.T, points_meta_centered)
    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Extract the rotation around the Y axis
    beta = np.arctan2(R[0, 2], R[2, 2])

    # Construct the rotation matrix around the Y axis
    Ry1 = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    Ry2 = np.array([
        [np.cos(beta), 0, -np.sin(beta)],
        [0, 1, 0],
        [np.sin(beta), 0, np.cos(beta)]
    ])

    transformed_bodypose1 = np.dot(points_body_centered, Ry1.T)
    transformed_bodypose2 = np.dot(points_body_centered, Ry2.T)

    dist1 = np.linalg.norm(points_meta_centered - transformed_bodypose1)
    dist2 = np.linalg.norm(points_meta_centered - transformed_bodypose2)

    if dist1 < dist2:
        return Ry1
    else:
        return Ry2


def transform_points(points:np.array, R, t):
    return np.dot(points, R.T) + t


def _output_data(data, output_path):
    """(Helper function) Output the transformed data to a JSON file."""
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_formatted_meta_data(meta_data_path, key1:str, key2:str):

    assert key1.lower() in ["left", "right", "camera"], "Invalid key1. Choose 'left', 'right' or 'camera'."

    try:
        data_filename = [i for i in os.listdir(meta_data_path) if key1 in i.lower() and key2 in i.lower()][0]
    except IndexError:
        print(f"Error: No file found for {key1} {key2}.")
        return {}
    
    data_path = os.path.join(meta_data_path, data_filename)
    with open(data_path, 'r') as file:
        data = json.load(file)
        data = _format_meta_data(data['Entries'], l_r_c=key1) # Remove Entries if not in data anymore

    return data

def process_data(args):

    meta_data_path = args.data
    body_pose_path = args.PROCESSED_BODYPOSE_PATH
    output_path = args.HAND_POSE_OUTPUT_PATH
    delta_ms = args.delta

    # =====================
    # Step 1: Load the data
    # =====================

    # Load LEFT HAND from metaquest data
    data_lefthand = get_formatted_meta_data(meta_data_path, "left", "hand")
    data_righthand = get_formatted_meta_data(meta_data_path, "right", "hand")
    data_head = get_formatted_meta_data(meta_data_path, "camera", "position")

    if data_lefthand == {} or data_righthand == {} or data_head == {}: # Check if data is empty
        print(f"Error: No data found in {meta_data_path}.")
        return
    
    data_metaquest = _merge_n_datasets(40, data_lefthand, data_righthand, data_head)

    # Load body pose data
    with open(body_pose_path, 'r') as file:
        data_bodypose = json.load(file)

        # Check if the data is in the correct format
        assert data_bodypose['landmark_type'].lower() == 'landmark', f"Landmark type is not 'landmark' but {data_bodypose['landmark_type']}. This script only supports world poses."
        data_bodypose.pop('landmark_type')

    # ============================
    # Step 2: Synchronize the data
    # ============================

    data_metaquest, data_bodypose = match_data(data_metaquest=data_metaquest, data_bodypose=data_bodypose, delta_ms=delta_ms)

    data_metaquest, data_bodypose, differences = align_data(data1=data_metaquest, data2=data_bodypose, delta_ms=delta_ms)

    # for debugging
    db = []
    da = []

    # Save data
    new_bodypose = {}

    # Iterate over all common timestamps and calculate the optimal transformation
    for (t_meta, d_meta), (t_bodypose, d_bodypose) in zip(data_metaquest.items(), data_bodypose.items()):

        rel_data_metaquest = get_relevant_data({t_meta: d_meta})
        rel_data_bodypose = get_relevant_data({t_bodypose: d_bodypose})

        # ================================
        # Step 3: Calculate transformation
        # ================================
        
        # Convert dictionaries to numpy arrays
        # np_rel_data_meta, np_rel_data_bodypose = prepare_points_for_transformation(data_bodypose=rel_data_bodypose_head[t_bodypose], data_meta=rel_data_metaquest_head[t_meta])
        np_rel_data_meta_t, np_rel_data_bodypose_t = prepare_points_for_transformation(data_bodypose=rel_data_bodypose[t_bodypose], 
                                                                                       data_meta=rel_data_metaquest[t_meta], 
                                                                                       tr="translation")
        
        np_rel_data_meta_r, np_rel_data_bodypose_r = prepare_points_for_transformation(data_bodypose=rel_data_bodypose[t_bodypose], 
                                                                                       data_meta=rel_data_metaquest[t_meta], 
                                                                                       tr="rotation")
        # Calculate the optimal rotation matrix, scaling factor, and translation vector
        t = kabsch_scaling(np_rel_data_meta_t, np_rel_data_bodypose_t, get="translation")
        R = kabsch_scaling(np_rel_data_meta_r, np_rel_data_bodypose_r, get="rotation")
    
        transformed_data_bodypose = transform_points(_dict_to_numpy(d_bodypose), R, t)

        # Bring bodypose data back into dictionary format
        new_bodypose[t_bodypose] = {str(i): {'x': lm[0], 'y': lm[1], 'z': lm[2]} for i, lm in enumerate(transformed_data_bodypose)}
        
        # Performance assessment
        transformed_rel_data_bodypose = transform_points(np_rel_data_bodypose_r, R, t)

        # ====================================
        # (DEBUGGING) Calculate mean distances
        # ====================================

        distances_before = np.abs(np_rel_data_meta_r - np_rel_data_bodypose_r)
        distances_after = np.abs(np_rel_data_meta_r - transformed_rel_data_bodypose)

        db.append(distances_before)
        da.append(distances_after)

    db = np.concatenate(db, axis=0)
    da = np.concatenate(da, axis=0)

    print(f"Mean distance before transformation: {np.mean(db, axis=0)}")
    print(f"Mean distance after transformation: {np.mean(da, axis=0)}")

    # Output the transformed data
    _output_data(new_bodypose, os.path.join(output_path, "bodypose.json"))

    # Output cleaned meta data
    first_meta_t = next(iter(data_metaquest))
    meta_lefthand_from_t = {t: d for t, d in data_lefthand.items() if float(t) >= first_meta_t-delta_ms/1000}
    _output_data(meta_lefthand_from_t, os.path.join(output_path, "meta_lefthand.json"))


    meta_right_hand_from_t = {t: d for t, d in data_righthand.items() if float(t) >= first_meta_t-delta_ms/1000}
    _output_data(meta_right_hand_from_t, os.path.join(output_path, "meta_righthand.json"))

    meta_head_from_t = {t: d for t, d in data_head.items() if float(t) >= first_meta_t-delta_ms/1000}
    _output_data(meta_head_from_t, os.path.join(output_path, "meta_head.json"))

    # Postprocess the files
    if not args.no_postprocess:
        post_process_data(output_path)


def main(args):

    # Load meta bodypose mapping
    with open(os.path.join(utils.BODY_POSE_DIR, "meta_bodypose_mapping.json"), 'r') as file:
        global META_BODY_MAPPING
        META_BODY_MAPPING = json.load(file)
        META_BODY_MAPPING = META_BODY_MAPPING['index_to_body_part']

        # Check if keys are convertable to type int and values are type str
        assert all(isinstance(int(k), int) for k in META_BODY_MAPPING.keys()), f"{os.path.join(utils.BODY_POSE_DIR, 'meta_bodypose_mapping.json')} must contain ints as keys."

    # Get correct bodypose data
    body_pose_name = [v for v in os.listdir(args.data) if v.lower().endswith(".mov") or v.lower().endswith(".mp4")]
    if len(body_pose_name) == 0:
        print(f"Tried to get bodypose name from {args.data_meta} but found no corresponding video file.")
    elif len(body_pose_name) > 1:
        print(f"Found multiple video files in {args.data_meta}. Using {body_pose_name[0].rsplit('.')[0]} as bodypose data.")

    body_pose_name = body_pose_name[0].rsplit('.')[0]
    args.PROCESSED_BODYPOSE_PATH = os.path.join(args.PROCESSED_BODYPOSE_PATH, f"{body_pose_name}.json")

    # Process data
    process_data(args)

if __name__ == "__main__":
    main()