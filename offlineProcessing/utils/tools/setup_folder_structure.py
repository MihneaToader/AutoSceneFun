import os
import time as t
import utils

def get_session_index():
    return int(t.time())


# Function to create folder if it doesn't exist
def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _create_necessary_folders_bodypose(output_path, debugg=False):
        # Ensure the base output path exists
        ensure_folder(output_path)

        # Create subdirectories
        subdirectories = ["media", "raw"]

        if debugg:
            subdirectories.append("debugg")
            subdirectories.append(os.path.join("debugg", "landmarks"))

        for subdir in subdirectories:
            ensure_folder(os.path.join(output_path, subdir))   

def _create_necessary_folders_sync_hand_poses(output_path, set_output_name=None):

    # Ensure the base output path exists
    ensure_folder(output_path)

    output_name = get_session_index() if set_output_name is None else set_output_name
    session_folder_path = os.path.join(output_path, str(output_name))

    ensure_folder(session_folder_path)

    return session_folder_path