import argparse
import os

import utils
import body_pose

def parse_args():
    parser = argparse.ArgumentParser(description='Offline Processing of Logging Files from Meta Quest')
    parser.add_argument('--data', type=str, default=utils.DATA_DIR,
                        help='Path to data folder with videos')
    parser.add_argument("--model", type=str, default=os.path.join(utils.MODELS_DIR, "pose_landmarker_heavy.task"), 
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

    cmd_get_body_pose = f"python -m body_pose.get_body_pose --data {args.data} --model {args.model} --set_fps {args.set_fps} --mode 'Video'"
    os.system(cmd_get_body_pose)

    cmd_shift_pose_origin = f"python -m body_pose.shift_pose_origin --mode 'Video'"
    os.system(cmd_shift_pose_origin)



if __name__ == '__main__':
    main()
    