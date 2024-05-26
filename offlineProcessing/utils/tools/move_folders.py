import os
import shutil
from pathlib import Path

def copy_contents(src_folder, dest_folder, include):
    
    # Ensure the destination folder exists
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    
    # List all items in the source folder
    items = os.listdir(src_folder)
    
    for item in items:
        # Skip excluded items
        if not item in include:
            continue
        
        src_item = os.path.join(src_folder, item)
        dest_item = os.path.join(dest_folder, item)
        
        try:
            if os.path.isdir(src_item):
                # Copy directory
                shutil.copytree(src_item, dest_item)
            else:
                # Copy file
                shutil.copy2(src_item, dest_item)
            print(f"Copied: {src_item} -> {dest_item}")
        except Exception as e:
            print(f"Failed to copy {src_item}: {e}")

# if __name__ == "__main__":
#     # Define source and destination folders
#     src_folder = 'data/session1'
#     dest_folder = 'output/body_pose/final_recordings/session_1'
    
#     # List of items to exclude from copying (e.g., subfolder names or filenames)
#     exclude = ['audio_text.json', 'iPhoneMesh.json', 'roomMesh.obj', 'textured_output.obj']
    
#     copy_contents(src_folder, dest_folder, exclude)
