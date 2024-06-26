# Pose estimation
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')


# Visualisation
from body_pose.visualisation import draw_landmarks_on_image
import cv2

from tqdm import tqdm
import os

from utils import *


class BodyPose():
    def __init__(self, media_path, model_path, output_folder, debug=False, landmarks:str="landmark"):
        
        assert landmarks.lower() in ["landmark", "normalized_landmark"], "Invalid landmark type. Choose either 'landmark' or 'normalized_landmark'"
        
        self.media_path = media_path
        self.model_path = model_path
        self.output_folder = output_folder
        self.visualise = debug
        self.DEBUG = debug
        self.mode = self._set_mode()

        self.filename = os.path.basename(self.media_path).split(".")[0]

        self.landmark_type = landmarks.lower() # Either "landmark" or "normalized_landmark"

        self.creation_time = self._get_data_creation_date(self.media_path)

        # Generate necessary paths
        self._set_paths()


    def _set_paths(self):

        self.output_path = os.path.join(self.output_folder, "raw", self.filename + ".json") # Output path for landmarks
        self.debug_output_path = os.path.join(self.output_folder, "debug", "landmarks", self.filename + "_landmarks.json") # Output path for debug landmarks

        ending = ".jpg" if self.mode == "image" else ".mov"
        self.visualisation_output_path = os.path.join(self.output_folder, "media", self.filename + "_ann" + ending) # Output path for visualisation


    def _set_mode(self):
        if self.media_path.lower().endswith(".jpg"):
            mode = "image"
        elif self.media_path.lower().endswith(".mov") or self.media_path.lower().endswith(".mp4"):
            mode = "video"
        else:
            raise ValueError("Unsupported media type")
        
        return mode

    def get_body_pose_from_image(self):

        # Load image
        image = mp.Image.create_from_file(self.media_path)

        # Load model
        BaseOptions = python.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE)

        # Create PoseLandmarker object
        with PoseLandmarker.create_from_options(options) as landmarker:

            # Process image
            results = landmarker.detect(image)
            landmarks_dict = {}

            # Save results to json file
            # World joint positions
            if self.landmark_type == "landmark":
                if results.pose_world_landmarks:
                    landmarks_dict = {
                        "landmark_type": "Landmark",
                        str(self.creation_time): {i: {
                            'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence': lm.presence} for i, lm in enumerate(results.pose_world_landmarks[0])}}

            elif self.landmark_type == "normalized_landmark":
                if results.pose_landmarks:
                    landmarks_dict = {
                        "landmark_type": "NormalizedLandmark",
                        str(self.creation_time): {i: 
                            {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence':lm.presence} for i, lm in enumerate(results.pose_landmarks[0])}}

            with open(self.output_path, 'w') as f:
                json.dump(landmarks_dict, f, indent=4)

            if self.DEBUG:
                self._save_debug_landmarks(results)
            
            if self.visualise:
                self._visualise_results(image, results)

    def get_body_pose_from_video(self, target_fps):
        print(f"Processing video: {self.media_path}")

        if self.DEBUG:
            print("\n Warning: Full debugging not supported for videos \n")

        # Load video
        video = cv2.VideoCapture(self.media_path)

        # Load pose estimation model
        fps = round(video.get(cv2.CAP_PROP_FPS))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO)

        # Calculate how man frames to skip
        if target_fps == 0: target_fps = fps
        skip_ratio = max(1, int(fps / target_fps))
        fps = int(fps / skip_ratio)
        frame_count = 0
        # Create a tqdm progress bar
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) // skip_ratio

        # Define the codec and create VideoWriter object
        if self.visualise:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.visualisation_output_path, fourcc, fps, (frame_width, frame_height))

        pose_data = {"landmark_type": self.landmark_type}

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

                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Convert to MediaPipe Image format
                        mp_frame = mp.Image(data=rgb_frame, image_format=mp.ImageFormat.SRGB)

                        # Calculate timestamp in milliseconds (make sure it's non-negative)
                        timestamp_ms = int(video.get(cv2.CAP_PROP_POS_MSEC))
                        if timestamp_ms < 0:
                            timestamp_ms = 0

                        # Process the frame through MediaPipe
                        try:
                            results = landmarker.detect_for_video(mp_frame, timestamp_ms=timestamp_ms)
                        except Exception as e:
                            print(f"Error processing frame with timestamp {timestamp_ms}: {e}")
                            continue
                        
                        # Save data with global timestamp as key
                        global_timestamp = self.creation_time + timestamp_ms / 1000

                        # Usually landmarks are saved, not normalised landmarks
                        if self.landmark_type == "landmark":
                            if results.pose_world_landmarks:
                                landmarks_dict = {i: {
                                    'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence': lm.presence} for i, lm in enumerate(results.pose_world_landmarks[0])}
                                pose_data[f"{global_timestamp}"] = landmarks_dict


                        elif self.landmark_type == "normalized_landmark":
                            if results.pose_landmarks:
                                landmarks_dict = {i: {
                                    'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence': lm.presence} for i, lm in enumerate(results.pose_landmarks[0])}
                                pose_data[f"{global_timestamp}"] = landmarks_dict


                        if self.visualise and results.pose_landmarks:
                            annotated_image = draw_landmarks_on_image(rgb_frame, results)
                            bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                            out.write(bgr_annotated_image)

                    frame_count += 1
        except:
            print("Error")

        finally:
            video.release()
            if self.visualise:
                print("Saving video to: ", self.visualisation_output_path)
                out.release()

            with open(self.output_path, 'w') as f:
                json.dump(pose_data, f, indent=4)

    def _save_debug_landmarks(self, results):
        results_dict = {
            "pose_landmarks": [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence': lm.presence} for lm in results.pose_landmarks[0]],
            "pose_world_landmarks": [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility, 'presence': lm.presence} for lm in results.pose_world_landmarks[0]]}
        
        print(f'Saving landmarks to {self.debug_output_path}')
        with open(self.debug_output_path, 'w') as file:
            json.dump(results_dict, file, indent=4)

    def _visualise_results(self, image, results):
        # Get rid of alpha (transparency) channel
        bgr_image = image.numpy_view()[:, :, :3]
        annotated_image = draw_landmarks_on_image(bgr_image, results)

        # Convert RGB to BGR for displaying with OpenCV if necessary
        bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Save image
        cv2.imwrite(self.visualisation_output_path, bgr_image)

    @staticmethod
    def _get_data_creation_date(data_path):
        try:
            timestamp = os.stat(data_path).st_birthtime
            return timestamp
        except Exception as e:
            try :
                timestamp = os.stat(data_path).st_ctime 
                return timestamp
            except:
                print(f"Warning: Cannot read metadata from {data_path}.")
            return None

    def _process_data(self, set_fps=0):
        if self.mode == "image":
            self.get_body_pose_from_image()
        elif self.mode == "video":
            self.get_body_pose_from_video(set_fps)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


def process_data(args):
    if os.path.isdir(args.data):
        if args.mode.lower() == "video":
            filenames = [i for i in os.listdir(args.data) if i.lower().endswith(".mov") or i.lower().endswith(".mp4")]
            data = [os.path.join(args.data, i) for i in filenames]

        elif args.mode.lower() == "image":
            filenames = [i for i in os.listdir(args.data) if i.lower().endswith(".jpg")]
            data = [os.path.join(args.data, i) for i in filenames]
        
    elif {os.path.isfile(args.data) 
          and args.data.lower().endswith(".jpg") 
          or args.data.lower().endswith(".mov") 
          or args.data.lower().endswith(".mp4")}:
        data = [args.data]

    else:
        raise ValueError("Invalid path provided for data")
    
    # Process images
    for d in data:
        body_pose = BodyPose(d, args.model, args.BODYPOSE_OUTPUT_PATH, args.debug)
        body_pose._process_data(args.fps)