import json
from datetime import datetime
from dateutil import tz
import os
import utils

# NOTE: Goal of this file is to translate the processed data to the format used by unity

def post_process_file(file_path):
    """
    Bring processed data back in unity format
    """
    print(f"Processing file: {file_path}")

    # Check if valid filename
    filename = os.path.basename(file_path)
    assert filename in ['bodypose.json', 'meta_lefthand.json', 'meta_righthand.json', 'meta_head.json'], f"Invalid filename: {filename}, must be in ['bodypose.json', 'meta_lefthand.json', 'meta_righthand.json', 'meta_head.json']"

    def _convert_timestamp(timestamp):
        return datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]

    with open(file_path, 'r') as file:
        data = json.load(file)
        
        if next(iter(data.keys())) == "Entries": # Already processed
            return

    transformed_data = {"Entries": []}

    for timestamp, body_parts in data.items():
        entry = {
            "Timestamp": _convert_timestamp(timestamp),
            "Position_rotation": {}
        }
        for part, values in body_parts.items():
            entry["Position_rotation"][part] = {
                "PositionX": values["x"],
                "PositionY": values["y"],
                "PositionZ": values["z"],
            }
        transformed_data["Entries"].append(entry)

    # Overwrite previous file
    with open(file_path, 'w') as file:
        json.dump(transformed_data, file, indent=2)

def post_process_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isdir(file_path):
            post_process_folder(file_path)
        elif os.path.isfile(file_path):
            post_process_file(file_path)

# NOTE: exclude rotations in format_meta_data to speed up the code if needed

def pre_process_meta_data(data, l_r_c="left") -> json:
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