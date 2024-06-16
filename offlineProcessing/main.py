import argparse
import os
import time as t
import subprocess

import utils
from utils.tools.move_folders import copy_contents
from utils.tools.setup_folder_structure import _create_folder_structure
from body_pose.get_body_pose import process_data as get_body_pose
from body_pose.sync_hand_poses import main as sync_hand_poses


# TODO: Change LITE model to HEAVY model

def parse_args():
    parser = argparse.ArgumentParser(description='Offline Processing of Logging Files from Meta Quest')
    parser.add_argument('-n', '--session_name', type=str, default=str(int(t.time())),
                        help='Name of the session, Default is the current time in seconds')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path to data folder with videos')
    parser.add_argument("-npre","--no_preprocess", action="store_true",
                        help="Skip preprocessing of files")
    parser.add_argument('-pd', '--preprocessed_data', type=str, required=False,
                        help='Path to preprocessed data folder if no_preprocess is enabled')
    parser.add_argument("-npost", "--no_postprocess", action="store_true",
                        help="Skip postprocessing of files")
    parser.add_argument("-v", "--visualise", action="store_true",
                        help="Visualise the data")
    
    # Body Pose Retrieval Arguments
    parser.add_argument('--debug', action="store_true",
                        help='Debug mode outputs landmarks visualisation')
    parser.add_argument("--mode", type=str, default="Video", required=False,
                        help="Generate pose from: Video or Image")
    parser.add_argument("--model", type=str, required=False,
                        help="Path to body pose model")
    parser.add_argument("--fps", type=int, default=60, 
                        help="Synchronise all data to this fps")
    
    # Sync Hand Poses Arguments
    parser.add_argument("--delta", type=int, default=50,
                        help="Time difference threshold in milliseconds.")
    
    
    args = parser.parse_args()

    known_args, _ = parser.parse_known_args()

    assert known_args.fps > 0, "FPS must be greater than 0"
    if known_args.fps < 10:
        print("Warning: FPS really low, risks creating data lag.")


    if known_args.no_preprocess and known_args.preprocessed_data is None:
        parser.error("--no_preprocess requires --preprocessed_data path/to/preprocessed/data/folder")

    if known_args.model is None:
        known_args.model = os.path.join(utils.MODELS_DIR, "pose_landmarker_heavy.task")

    if known_args.no_postprocess and known_args.visualise:
        print("Cannot visualise data without postprocessing. Enabeling postprocessing...")
        known_args.no_postprocess = False

    if known_args.debug and known_args.no_preprocess:
        print("Cannot debug without preprocessing. Enabeling preprocessing...")
        known_args.no_preprocess = False

    assert known_args.mode in ["Video", "Image"], f"Mode '{known_args.mode}' not valid, must be either 'Video' or 'Image'."

    # Set relevant paths
    known_args.OUTPUT_PATH = os.path.join(utils.OUTPUT_DIR, str(args.session_name))

    if known_args.preprocessed_data is None:
        known_args.BODYPOSE_OUTPUT_PATH = None
    else:
        known_args.OUTPUT_PATH = known_args.preprocessed_data
        known_args.BODYPOSE_OUTPUT_PATH = os.path.join(args.preprocessed_data, "body_pose")
    

    return known_args


def main():

    args = parse_args()

    # =============================
    # Body Pose Estimation Pipeline
    # =============================

    # Get body pose
    if not args.no_preprocess:

        args.BODYPOSE_OUTPUT_PATH = os.path.join(args.OUTPUT_PATH, "body_pose")
        _create_folder_structure(args, "preprocess")

        # Get body pose
        get_body_pose(args)

    # Synchronise body pose and hand pose
    args.HAND_POSE_OUTPUT_PATH = os.path.join(args.OUTPUT_PATH, "hand_mapping_raw")
    args.PROCESSED_BODYPOSE_PATH = os.path.join(args.BODYPOSE_OUTPUT_PATH, "raw")
    _create_folder_structure(args, "postprocess")

    # Sync hand poses
    sync_hand_poses(args)

    # ==============
    # Visualise Data
    # ==============

    if args.visualise:
        args.VISUALISATION = os.path.join(args.OUTPUT_PATH, "visualisation")

        # Move data to correct folders
        include_files = ['audio_text.json', 'iPhoneMesh.json', 'roomMesh.obj', 'textured_output.obj']
        copy_contents(args.data, args.VISUALISATION, include_files)
        copy_contents(args.HAND_POSE_OUTPUT_PATH, args.VISUALISATION)

        # Visualise data
        cmd_visualise_data = f"python '{os.path.join(utils.PYVIZ_DIR, 'examples', 'time_data.py')}' --data '{args.VISUALISATION}'"
        os.system(cmd_visualise_data)


if __name__ == '__main__':
    main()