import os


# Function to create folder if it doesn't exist
def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _create_folder_structure(args, stage="preprocess"):

    if stage not in ["preprocess", "postprocess"]:
        raise ValueError("stage must be either 'preprocess' or 'postprocess'")

    # Ensure the base output path exists
    ensure_folder(args.OUTPUT_PATH)
    
    if stage == "preprocess":
        if not args.no_preprocess: # Create preprocess folders
            ensure_folder(args.BODYPOSE_OUTPUT_PATH)

            # Create subdirectories
            subdirectories = ["media", "raw"]

            if args.debug:
                subdirectories.append("debug")
                subdirectories.append(os.path.join("debug", "landmarks"))

            for subdir in subdirectories:
                ensure_folder(os.path.join(args.BODYPOSE_OUTPUT_PATH, subdir))
    
    if stage == "postprocess":
        ensure_folder(args.HAND_POSE_OUTPUT_PATH)