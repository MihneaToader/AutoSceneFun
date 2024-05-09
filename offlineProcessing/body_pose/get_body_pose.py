# Pose estimation
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import datetime
import json
from PIL import Image

# Visualisation
from body_pose.visualisation import draw_landmarks_on_image
import cv2

import datetime
from PIL import Image
from pymediainfo import MediaInfo

from tqdm import tqdm
import argparse
import os

from utils import *

def get_body_pose_from_image(model_path, image, creation_time, output_path, image_name, visualise=False):
    # Load model
    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    # Create PoseLandmarker object
    with PoseLandmarker.create_from_options(options) as landmarker:
        
        # Process image
        results = landmarker.detect(image)

        # Save results to json file
        # World joint positions
        if results.pose_world_landmarks:
            landmarks_dict = {str(creation_time):{'Landmark':{i: {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence':lm.presence} for i, lm in enumerate(results.pose_world_landmarks[0])}}}
            with open(os.path.join(output_path, "raw", image_name + ".json"), 'w') as f:
                json.dump(landmarks_dict, f, indent=4)

        # Relative joint positions
        # if results.pose_landmarks:
        #     landmarks_dict = {'NormalizedLandmark':{i: {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence':lm.presence} for i, lm in enumerate(results.pose_landmarks[0])}}
        #     with open(os.path.join(output_path, "raw", image_name + ".json"), 'w') as f:
        #         json.dump(landmarks_dict, f, indent=4)
        
        if DEBUGG:

            # Split filename from output path
            debugg_output_path = os.path.join(output_path, "debugg", "landmarks", image_name + "_landmarks.json")

            results_dict = {"pose_landmarks": [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence':lm.presence} for lm in results.pose_landmarks[0]], 
                            "pose_world_landmarks": [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence':lm.presence} for lm in results.pose_world_landmarks[0]]}

            print(f'Saving landmarks to {debugg_output_path}')
            with open(debugg_output_path, 'w') as file:
                json.dump(results_dict, file, indent=4)

        if visualise:
            # Get rid of alpha (transparency) channel
            bgr_image = image.numpy_view()[:, :, :3]
            annotated_image = draw_landmarks_on_image(bgr_image, results)
            # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            # Convert RGB to BGR for displaying with OpenCV if necessary
            bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(os.path.join(output_path, "media", image_name + "_ann.jpg"), bgr_image)


def get_body_pose_from_video(model_path, video, output_path, video_name, starting_time, target_fps, visualise=False):

    # Read metadata from video
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load pose estimation model
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    
    # Calculate how man frames to skip
    if target_fps == 0: target_fps = fps
    skip_ratio = max(1, int(fps / target_fps))
    fps = int(fps / skip_ratio)
    frame_count = 0

    # Create a tqdm progress bar
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) // skip_ratio
    progress_bar = tqdm(total=total_frames, desc="Processing Video Frames")

    # Define the codec and create VideoWriter object
    if visualise:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        video_output_path = os.path.join(output_path, "media", video_name + "_ann.mov")
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))
    
    # Save joint positions
    landmarks_file_path = os.path.join(output_path, "raw", video_name + ".json")
    pose_data = {}
    try:
        # Create PoseLandmarker object
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            
            # Iterate over each video frame
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                # Skip frames if necessary
                if frame_count % skip_ratio == 0:

                    # Update the progress bar
                    progress_bar.update(1)
                    

                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert to MediaPipe Image format
                    mp_frame = mp.Image(data=rgb_frame, image_format=mp.ImageFormat.SRGB)

                    # Calculate timestamp in milliseconds (make sure it's non-negative)
                    timestamp_ms = int(video.get(cv2.CAP_PROP_POS_MSEC))
                    if timestamp_ms < 0:
                        timestamp_ms = 0  # Reset to zero if negative

                    # Process the frame through MediaPipe
                    try:
                        results = landmarker.detect_for_video(mp_frame, timestamp_ms=timestamp_ms)
                    except Exception as e:
                        print(f"Error processing frame with timestamp {timestamp_ms}: {e}")
                        continue
                    
                    # Save data with global timestamp as key
                    global_timestamp = starting_time + datetime.timedelta(milliseconds=timestamp_ms)
                    global_timestamp = global_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                    # if results.pose_landmarks: # world coord are found under results.pose_world_landmarks
                    #     landmarks_dict = {'NormalizedLandmark':{i: {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence':lm.presence} for i, lm in enumerate(results.pose_landmarks[0])}}
                    #     pose_data[global_timestamp] = landmarks_dict
                    if results.pose_world_landmarks: # world coord are found under results.pose_world_landmarks
                        landmarks_dict = {'Landmark':{i: {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence':lm.presence} for i, lm in enumerate(results.pose_world_landmarks[0])}}
                        pose_data[global_timestamp] = landmarks_dict


                    if visualise and results.pose_landmarks:
                        annotated_image = draw_landmarks_on_image(rgb_frame, results)
                        bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                        out.write(bgr_annotated_image)
                        # cv2.imshow('Annotated Frame', bgr_annotated_image)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                frame_count += 1
    
    finally:
        # Close progress bar on any exit
        progress_bar.close()
        video.release()
        if visualise:
            print("Saving video to: ", video_output_path)
            out.release()
        # cv2.destroyAllWindows()

        with open(landmarks_file_path, 'w') as f:
            json.dump(pose_data, f, indent=4)


def get_data_creation_date(data_path):
    """Get creation time of video from metadata, including milliseconds if available."""

    # Account for videos
    if data_path.lower().endswith(('.mp4', '.mov')):
        media_info = MediaInfo.parse(data_path)
        for track in media_info.tracks:
            if track.track_type == "Video":
                creation_time_str = track.encoded_date
                # Adjust format to possibly include milliseconds
                formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]  # Added format with %f for milliseconds
                for fmt in formats:
                    try:
                        creation_time = datetime.datetime.strptime(creation_time_str.replace(' UTC', ''), fmt)
                        return creation_time
                    except ValueError:
                        continue
                print("Warning: Creation time not found in video metadata or does not match expected format.")
                return None

    # Get image creation date
    elif data_path.lower().endswith(('.jpg', '.jpeg')):
        with Image.open(data_path) as img:
            exif_data = img._getexif()
            if exif_data:
                datetime_org = exif_data.get(36867)  # DateTimeOriginal tag
                if datetime_org:
                    formats = ["%Y:%m:%d %H:%M:%S.%f", "%Y:%m:%d %H:%M:%S"]  # Added format with %f for milliseconds
                    for fmt in formats: # Try to get milliseconds
                        try:
                            creation_time = datetime.datetime.strptime(datetime_org, fmt)
                            return creation_time
                        except ValueError:
                            continue

    else:
        print("Warning: Unsupported media type.")
        return None


def process_data(args):
    """Process data based on mode"""

    # Account for images
    if args.mode.lower() == "image":
        # Load all images in folder
        if os.path.isdir(args.data):
            images = [i for i in os.listdir(args.data) if i.endswith(".jpg")]
            for image in images:
                image_path = os.path.join(args.data, image)
                i = mp.Image.create_from_file(image_path)
                creation_time = get_data_creation_date(image_path)
                get_body_pose_from_image(model_path=args.model, image=i, creation_time=creation_time, visualise=args.visualise, output_path=args.output, image_name=image.split(".")[0])
        
        # If only single image is given
        elif os.path.isfile(args.data):
            i = mp.Image.create_from_file(args.data)
            
            # Get image name
            image = args.data.split("/")[-1]
            creation_time = get_data_creation_date(args.data)
            get_body_pose_from_image(model_path=args.model, image=i, creation_time=creation_time, visualise=args.visualise, output_path=args.output, image_name=image.split(".")[0])
        
        else:
            raise ValueError("Invalid path provided for video data")
        
    # Account for videos
    elif args.mode.lower() == "video":
        # Process whole folder
        if os.path.isdir(args.data):
            # Isolate videos in folder
            videos = [v for v in os.listdir(args.data) if v.endswith(".mov") or v.endswith(".mp4")]
            for video in videos:
                video_path = os.path.join(args.data, video)
                v = cv2.VideoCapture(video_path)
                
                # Check if video is opened
                if not v.isOpened():
                    raise IOError("Cannot open video: " + os.path.join(args.data, video))
                
                creation_time = get_data_creation_date(video_path)
                new_video_name = str(creation_time)
                get_body_pose_from_video(model_path=args.model, video=v, visualise=args.visualise, output_path=args.output, video_name=new_video_name, starting_time=creation_time, target_fps=args.set_fps)
        # Process single video
        elif os.path.isfile(args.data):
            v = cv2.VideoCapture(args.data)
            
            # Check if video is opened
            if not v.isOpened():
                raise IOError("Cannot open video: " + args.data)
            
            creation_time = get_data_creation_date(args.data)
            new_video_name = str(creation_time)
            get_body_pose_from_video(model_path=args.model, video=v, visualise=args.visualise, output_path=args.output, video_name=new_video_name, starting_time=creation_time, target_fps=args.set_fps)
        
        else:
            raise ValueError("Invalid path provided for video data")
        
def create_necessary_folders(output_path):
    # Function to create folder if it doesn't exist
    def ensure_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    # Ensure the base output path exists
    ensure_folder(output_path)

    # Create subdirectories
    subdirectories = ["media", "raw"]
    if DEBUGG:
        subdirectories.append("debugg")
        subdirectories.append("debugg/landmarks")
    
    for subdir in subdirectories:
        ensure_folder(os.path.join(output_path, subdir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="Video", 
                        help="Generate pose from [Image, Video, Stream]")
    parser.add_argument("--model", type=str, default=os.path.join(MODELS_DIR, "pose_landmarker_lite.task"), 
                        help="Path to model")
    parser.add_argument("-d", "--data", type=str, default=DATA_DIR, 
                        help="Path to image, video or stream data folder")
    parser.add_argument("-v", "--visualise", type=bool, default=False, 
                        help="Visualise results")
    parser.add_argument("-out", "--output", type=str, default=os.path.join(OUTPUT_DIR, "body_pose"),
                        help="Path to save output")
    parser.add_argument("-sf", "--set_fps", type=int, default=0, 
                        help="Set fps for video processing, 0 for original fps")
    parser.add_argument("--debugg", type=bool, default=False, 
                        help="Debugg mode")
    args = parser.parse_args()

    # Raise warning if mode not supported
    if args.mode.lower() not in ["image", "video"]:
        raise ValueError(f"{args.mode} not supported. Video and Stream to be implemented")
    
    # if args.mode.lower() == "image":
    #     print("Warning: Output joint positions is not implemented. Only visualisation available.")
    
    global DEBUGG # Make DEBUGG a global variable
    DEBUGG = args.debugg

    create_necessary_folders(args.output)
    
    process_data(args)
    
if __name__ == "__main__":
    main()