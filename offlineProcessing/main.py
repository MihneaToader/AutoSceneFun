import argparse
import os

import utils
import body_pose

def parse_args():
    parser = argparse.ArgumentParser(description='Offline Processing of Logging Files from Meta Quest')
    parser.add_argument('--data', type=str, required=False,
                        help='Path to data folder with videos')
    parser.add_argument("--model", type=str, required=False,
                        help="Path to body pose model")
    parser.add_argument("--set_fps", type=int, default=30, 
                        help="Synchronise all data to this fps")
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # =============================
    # Body Pose Estimation Pipeline
    # =============================

    print(f"Working on {args.set_fps} FPS")

    cmd_get_body_pose = f"python -m body_pose.get_body_pose --mode Video --set_fps {args.set_fps} --model models/pose_landmarker_lite.task" 

    if args.data:
        cmd_get_body_pose += f"--data {args.data}"
    
    if args.model:
        cmd_get_body_pose += f"--model {args.model}"

    os.system(cmd_get_body_pose)

    cmd_shift_pose_origin = f"python -m body_pose.shift_pose_origin --mode Video"
    os.system(cmd_shift_pose_origin)



if __name__ == '__main__':
    main()
    