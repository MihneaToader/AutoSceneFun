# Human-Scene Interactions by Observing Hand Poses with Meta Quest 3

The objective of this project is to optimise the dataset creation process for 3D indoor environments, with a particular focus on the [SceneFun3D](https://scenefun3d.github.io) dataset. Our approach is divided into two main phases: online and offline processing. 

![Project Overview](docs/Bedroom%201%20Walkaround.webp)

**Online Phase**
* Capture indoor environments using iPhone LiDAR and Meta Quest 3 mesh extraction

**Offline Phase**
* Employ a speech-to-text model for efficient transcription of interaction labels.
* Utilise Google's [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) body pose estimation model for posture tracking.

Additionally, we enhance the [Pyviz3D](https://github.com/francisengelmann/PyViz3D) visualization pipeline to support time-series analysis.

This repository is organized into two sections: online and offline. Detailed guides for environment setup and demonstrations are provided.

## Repo setup
Clone repo
```
git clone git@github.com:MihneaToader/AutoSceneFun.git
```
*Disclaimer*: run 
```
git submodule update --init --recursive
```
to download the necessary submodules.

## :goggles: Online Processing

### Setup instructions for development

<details>

### Prerequisite software
[Unity 2022.3.23f1](https://unity.com/download)\
[Meta Quest Developer Hub](https://developer.oculus.com/meta-quest-developer-hub/)\
[SideQuest](https://sidequestvr.com/setup-howto)\
[Oculus App](https://developer.oculus.com/documentation/unity/unity-link/) (only for Windows)

### Setup
For a visual guide, follow Black Whale Studio's [Get Started with Meta Quest Development in Unity](https://www.youtube.com/watch?v=BU9LYKM2TDc).\
Additionally, in the Oculus App, in the Beta tab of the Settings, toggle "Developer Runtime Features" and then enable Passthrough, Point Cloud and Spatial Data. For ease of development, try to setup the Oculus Link if you're using Windows. It allows for quick debugging without having to build every time. Open the unity folder as a new Unity project.

### File structure
For the sake of privacy, room scans are not included in this repository. Scans can be made with [3DScanner app](https://3dscannerapp.com/) an iPhone app which allows capturing LiDAR scans. Place the scan in the Assets\Scans folder and you're ready to use them. Click on the SampleScene in the Scenes folder. Now replace the textured_output in the scene view with your scan. Take the default element of the scan in the scene view and drag it to AlignementRecorder-> Aligned Object element and NewMeshAlignment->Source Mesh Object element.
**Make sure that your scan prefab has read/write enabled else Unity will not be able to accesse the scan and perform transformations!**

### Building and running
If you've done everything right, you should be able to go to File > Build Settings,set platform to Android and set the build target to the Oculus 3 device and hit Build and Run. 

### Recording data
If the scan does not aligne correctly restart the app and change your orientation or changing the Y rotation of the camera rig in Unity should help.
The app starts to aligne the scan and then will transition to passtrough and hand, head data and audio will be recorded. Once the app is exited all datasets are saved with timestamp and identifier names. Using the SideQuest app the recorded data can be found in Android/data/app_name/files.

</details>

## Offline Processing

### :gear: Setup

<details>

***Disclaimer***: Streaming is not supported

Navigate to the offlineProcessing folder `cd offlineProcessing`

*MacOS* and *Linux*

Run the setup-file to setup the environment with the necessary dependencies and download models.

```
bash setup.sh
```

*Windows*

Run the setup-file to setup the environment with the necessary dependencies and download models.

Run the bat file twice as conda create exits the first time eventhough running correctly 
```
setup.bat
```
#### Additional Useful Flags

* `--model` : only download a specific model. Valid options: 'lite', 'full', or 'heavy'

#### Manual Setup
*MacOS* and  *Linux*
<details>

Run
```
conda env create -f environment.yml
conda activate 3dv
conda install open3d
python -m pip install git+https://github.com/m-bain/whisperx.git
```

Download the desired body pose estimation network:
* [Lite](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)
* [Full](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)
* [Heavy](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task)

Place the model(s) into the `offlineProcessing/models` folder

</details>

*Windows*
<details>

Run
```
conda env create -f windows_environment.yml
conda activate 3dv
python -m pip install git+https://github.com/m-bain/whisperx.git
```

Download the desired body pose estimation network:
* [Lite](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)
* [Full](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)
* [Heavy](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task)

Place the model(s) into the `offlineProcessing/models` folder

</details>

</details>

### :bar_chart: Data

<details>

For each recording, create a new folder within the `data` directory. An example scene can be found [here](https://polybox.ethz.ch/index.php/s/4XrXz0gl9Ev5C8Q). Please move this example to `data/example`. The folder has to be moved not copied as copying changes the folder creation time.

The data must adhere to specific naming conventions:
* Camera positions must include `camera` and `position` in the name.
* Left-hand recordings must include `left` and `hand` in the name.
* Right-hand recordings must include `right` and `hand` in the name.

#### Supported Formats
* **Images**: `.jpg`
* **Videos**: `.mp4`, `.mov` (predominantly using `.mov`)
* **Meta Quest Recordings**: `.json`
* **Room Mesh and Texture**: `.obj`

</details>

### :raised_hands: Hand Pose Mapping

<details>

Utilize `hand_pose_mapping.py` to map body pose recordings to recorded Meta Quest hand poses.

*Disclaimer*: Running the visualization requires adding the PyVis path to the system path. When running the code for the first time, a warning will appear, providing the necessary command to execute.

### Instructions

1. **Run the entire pipeline of body pose extraction, mapping, and visualization**:
    ```
    python hand_pose_mapping.py -d path/to/data/folder --visualise
    ```

2. **Run the pipeline without video processing** (i.e., work with already processed video and visualize):
    ```
    python hand_pose_mapping.py -d path/to/data/folder --visualise -npre --preprocessed_data path/to/already/processed/data
    ```

3. **Run the pipeline without postprocessing** (does not convert data back into Unity format; postprocessing is required for visualization):
    ```
    python hand_pose_mapping.py -d path/to/data/folder -npost
    ```

#### Additional Useful Flags

* `--session_name`: Specifies the name of the output session. The default is the current time in seconds.
* `--mode`: Determines the input type for generating body poses, either from an image or video. Default: Video.
* `--fps`: Synchronizes all data to the specified frames per second (see disclaimer below).
* `--model`: Selects the Pose Landmarker model by providing the relative path.
* `--debug`: Enables debug mode, which outputs all landmarks and the provided media file with annotations (media file output only works for photos).
* `--delta`: Sets the time difference threshold in milliseconds for body pose hand mapping.
* `--audio_transcribe_keyword`:If used will use keyword transcription. 
* `--keyword_transcribe_mode`:If --audio_transcribe_keyword is used sets the start and stop keywords.

*FPS*: Data is synchronized based on the timestamps from the Meta Quest (usually lower than the video fps). If the fps is lower than the provided Quest frames, this may result in data lag.

</details>

### Output formats

* ***Images***: .jpg
* ***Videos***: .mov
* ***Landmarks***: .json

</details>

## Contributors

- [Axel Wagner](https://github.com/Axel2017)
- [Max Kieffer](https://github.com/mkiefferus)
- [Mihnea Toader](https://github.com/MihneaToader)