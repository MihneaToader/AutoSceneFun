# Pose estimation
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import datetime
import json

# Visualisation
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from pymediainfo import MediaInfo
# from google.colab.patches import cv2_imshow


from tqdm import tqdm
import argparse
import os

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        
    return annotated_image

def get_body_pose_from_image(model_path, image, output_path, visualise=False):
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

        if visualise:
            # Get rid of alpha (transparency) channel
            bgr_image = image.numpy_view()[:, :, :3]
            annotated_image = draw_landmarks_on_image(bgr_image, results)
            # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            # Convert RGB to BGR for displaying with OpenCV if necessary
            bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(output_path, bgr_image)

def get_body_pose_from_video(model_path, video, output_path, starting_time, target_fps, visualise=False):

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
        out = cv2.VideoWriter(os.path.join(output_path), fourcc, fps, (frame_width, frame_height))
    
    # Save joint positions
    landmarks_file_path = os.path.join(output_path[:-4] + ".json")
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

                    if results.pose_landmarks: # world coord are found under results.pose_world_landmarks
                        landmarks_dict = {'NormalizedLandmark':{i: {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence':lm.presence} for i, lm in enumerate(results.pose_landmarks[0])}}
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
            print("Saving video to: ", output_path)
            out.release()
        # cv2.destroyAllWindows()

        with open(landmarks_file_path, 'w') as f:
            json.dump(pose_data, f, indent=4)

        print("Saving video and landmarks to: ", landmarks_file_path[:-4])


def get_video_creation_date(video_path):
    """Get creation time of video from metadata"""

    media_info = MediaInfo.parse(video_path)
    for track in media_info.tracks:
        if track.track_type == "Video":
            creation_time_str = track.encoded_date
            # Parse the creation time string into a datetime object
            creation_time = datetime.datetime.strptime(creation_time_str.replace(' UTC', ''), "%Y-%m-%d %H:%M:%S")
            return creation_time
    
    print("Warning: Creation time not found in video metadata or does not match expected format.")
    return None


def process_data(args):
    """Process data based on mode"""

    # Account for images
    if args.mode.lower() == "image":
        # Load all images in folder
        if os.path.isdir(args.data):
            images = [i for i in os.listdir(args.data) if i.endswith(".jpg")]
            for image in images:
                i = mp.Image.create_from_file(os.path.join(args.data, image))
                output_path = os.path.join(args.output, image)
                get_body_pose_from_image(model_path=args.model, image=i, visualise=args.visualise, output_path=output_path)
        
        # If only single image is given
        elif os.path.isfile(args.data):
            i = mp.Image.create_from_file(args.data)
            
            # Get image name
            image_name = args.data.split("/")[-1]
            output_path = os.path.join(args.output, image_name)
            get_body_pose_from_image(model_path=args.model, image=i, visualise=args.visualise, output_path=args.output)
        
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
                
                creation_time = get_video_creation_date(video_path)
                output_path = os.path.join(args.output, str(creation_time) + video[-4:])
                get_body_pose_from_video(model_path=args.model, video=v, visualise=args.visualise, output_path=output_path, starting_time=creation_time, target_fps=args.set_fps)
        # Process single video
        elif os.path.isfile(args.data):
            v = cv2.VideoCapture(args.data)
            
            # Check if video is opened
            if not v.isOpened():
                raise IOError("Cannot open video: " + args.data)
            
            video_format = args.data.split("/")[-1][-4:]
            creation_time = get_video_creation_date(args.data)
            output_path = os.path.join(args.output, str(creation_time) + video_format)
            get_body_pose_from_video(model_path=args.model, video=v, visualise=args.visualise, output_path=output_path, starting_time=creation_time, target_fps=args.set_fps)
        
        else:
            raise ValueError("Invalid path provided for video data")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="Video", help="Generate pose from [Image, Video, Stream]")
    parser.add_argument("--model", type=str, default="models/pose_landmarker_lite.task", help="Path to model")
    parser.add_argument("-d", "--data", type=str, default="data", help="Path to image, video or stream data folder")
    parser.add_argument("-v", "--visualise", type=bool, default=False, help="Visualise results")
    parser.add_argument("-out", "--output", type=str, default="output", help="Path to save output")
    parser.add_argument("-sf", "--set_fps", type=int, default=0, help="Set fps for video processing")
    args = parser.parse_args()

    # Raise warning if mode not supported
    if args.mode.lower() not in ["image", "video"]:
        raise ValueError(f"{args.mode} not supported. Video and Stream to be implemented")
    
    if args.mode.lower() == "image":
        print("Warning: Output joint positions is not implemented. Only visualisation available.")
    
    # Check if output folder exists, else create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    process_data(args)
    
if __name__ == "__main__":
    main()


# TODO: add fps cap to synchronise fps with MetaQuest