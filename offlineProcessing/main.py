import argparse
import os

import utils
from utils.tools.move_folders import copy_contents
import body_pose

# TODO: Change LITE model to HEAVY model

def parse_args():
    parser = argparse.ArgumentParser(description='Offline Processing of Logging Files from Meta Quest')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data folder with videos')
    parser.add_argument("--output", type=str, default=utils.OUTPUT_DIR,
                        help="Path to output folder")
    parser.add_argument("--model", type=str, required=False,
                        help="Path to body pose model")
    parser.add_argument("--set_fps", type=int, default=60, 
                        help="Synchronise all data to this fps")
    parser.add_argument("--delta", type=int, required=False,
                        help="Time difference threshold in milliseconds.")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Skip preprocessing of files")
    parser.add_argument("--no_postprocess", action="store_true",
                        help="Skip postprocessing of files")
    parser.add_argument("--visualise", action="store_true",
                        help="Visualise the data")
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    if args.no_postprocess and args.visualise:
        print("Cannot visualise data without postprocessing. Enabeling postprocessing...")
        args.no_postprocess = False

    # =============================
    # Body Pose Estimation Pipeline
    # =============================

    # print(f"Working on {args.set_fps} FPS")

    # Get body pose
    # TODO: CHANGE TO HEAVY MODEL
    if not args.no_preprocess:
        get_bodypose_data_path = args.data
        get_bodpose_output_path = os.path.join(args.output, "body_pose")

        cmd_get_body_pose = f"python -m body_pose.get_body_pose --mode Video --data '{get_bodypose_data_path}' --output '{get_bodpose_output_path}' --set_fps {args.set_fps} --model models/pose_landmarker_lite.task" 
        
        if args.model:
            cmd_get_body_pose += f"--model {args.model}"
        
        os.system(cmd_get_body_pose)

    # # Synchronise body pose and hand pose
    sync_hand_poses_data_meta_path = args.data
    sync_hand_poses_data_bodypose_path = os.path.join(args.output, "body_pose", "raw")
    sync_hand_poses_output_path = os.path.join(args.output, "final")
    cmd_sync_hand_poses = f"python -m body_pose.sync_hand_poses --data_meta '{sync_hand_poses_data_meta_path}' --data_bodypose '{sync_hand_poses_data_bodypose_path}' --output_dir '{sync_hand_poses_output_path}'"

    if args.delta:
        cmd_sync_hand_poses += f" --delta {args.delta}"

    if not args.no_postprocess:
        cmd_sync_hand_poses += " --postprocess"

    os.system(cmd_sync_hand_poses)

    # ==============
    # Visualise Data
    # ==============

    if args.visualise:
        # Get output path
        visualise_meta_data_path = args.data

        # Get source folder = latest folder generated
        final_folders = [int(f) for f in os.listdir(os.path.join(args.output, 'final'))]
        visualise_source_folder = os.path.join(args.output, 'final', str(max(final_folders)))

        # Move data to correct folders
        include_files = ['audio_text.json', 'iPhoneMesh.json', 'roomMesh.obj', 'textured_output.obj']
        copy_contents(visualise_meta_data_path, visualise_source_folder, include_files)

        # Visualise data
        cmd_visualise_data = f"python '{os.path.join(utils.PYVIZ_DIR, 'examples', 'time_data.py')}' --data '{visualise_source_folder}'"
        os.system(cmd_visualise_data)
    
    

if __name__ == '__main__':
    main()