import argparse
import os

import utils
import body_pose

# TODO: Change LITE model to HEAVY model

def parse_args():
    parser = argparse.ArgumentParser(description='Offline Processing of Logging Files from Meta Quest')
    parser.add_argument('--data', type=str, required=False,
                        help='Path to data folder with videos')
    parser.add_argument("--model", type=str, required=False,
                        help="Path to body pose model")
    parser.add_argument("--set_fps", type=int, default=30, 
                        help="Synchronise all data to this fps")
    parser.add_argument("--output", type=str, required=False,
                        help="Path to output folder")
    parser.add_argument("--delta", type=int, required=False,
                        help="Time difference threshold in milliseconds.")
    parser.add_argument("--not_postprocess", action="store_true",
                        help="Skip postprocessing of files")
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # =============================
    # Body Pose Estimation Pipeline
    # =============================

    print(f"Working on {args.set_fps} FPS")

    # Get body pose
    cmd_get_body_pose = f"python -m body_pose.get_body_pose --mode Video --set_fps {args.set_fps} --model models/pose_landmarker_heavy.task" 
    
    if args.model:
        cmd_get_body_pose += f"--model {args.model}"

    if args.data:
        cmd_get_body_pose += f" --data {args.data}"
    
    if args.output:
        cmd_get_body_pose += f" --output {args.output}"

    os.system(cmd_get_body_pose)

    # Synchronise body pose and hand pose
    cmd_sync_hand_poses = f"python -m body_pose.sync_hand_poses"

    if args.data:
        cmd_sync_hand_poses += f" --data {args.data}"

    if args.output:
        cmd_sync_hand_poses += f" --output {args.output}"

    if args.delta:
        cmd_sync_hand_poses += f" --delta {args.delta}"

    os.system(cmd_sync_hand_poses)

    # Bring files into format for unity
    if not args.not_postprocess:
        print("Postprocessing files...")

        cmd_postprocess_files = "python -m body_pose.postprocess_meta_files"

        if args.output:
            cmd_postprocess_files += f" --data {os.path.join(args.output, 'body_pose', 'final_recordings')}"

        os.system(cmd_postprocess_files)
    

if __name__ == '__main__':
    main()