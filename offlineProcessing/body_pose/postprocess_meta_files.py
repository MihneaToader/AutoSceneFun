import json
from datetime import datetime
import argparse
import os

import utils

# NOTE: Goal of this file is to translate the processed data to the format used by unity

def process_file(file_path):
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

def process_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isdir(file_path):
            process_folder(file_path)
        elif os.path.isfile(file_path):
            process_file(file_path)
        
def main():
    parser = argparse.ArgumentParser(description="Synchronize hand pose data from MetaQuest and body pose data.")
    parser.add_argument("--data", type=str,
                        help="Path to the data directory.")

    args = parser.parse_args()

    # Check if data is a folder
    if not args.data:
        args.data = os.path.join(utils.OUTPUT_DIR, "body_pose", "final_recordings")
    
    # Process valid files in folder
    if os.path.isdir(args.data):
        process_folder(args.data)

    elif os.path.isfile(args.data):
        process_file(args.data)
        

if __name__ == "__main__":
    main()