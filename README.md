# AutoSceneFun
## Setup instructions for development
### Prerequisite software
[Unity 2022.3.23f1](https://unity.com/download)\
[Meta Quest Developer Hub](https://developer.oculus.com/meta-quest-developer-hub/)\
[SideQuest](https://sidequestvr.com/setup-howto)\
[Oculus App](https://developer.oculus.com/documentation/unity/unity-link/) (only for Windows)

### Setup
For a visual guide, follow Black Whale Studio's [Get Started with Meta Quest Development in Unity](https://www.youtube.com/watch?v=BU9LYKM2TDc).\
Additionally, in the Oculus App, in the Beta tab of the Settings, toggle "Developer Runtime Features" and then enable Passthrough, Point Cloud and Spatial Data. For ease of development, try to setup the Oculus Link if you're using Windows. It allows for quick debugging without having to build every time.

### File structure
For the sake of privacy, room scans are not included in this repository. Access polybox and download the needed scans from the `data` folder. Unpack the contents of the archive in the Assets\Scans folder and you're ready to use them.

## Building and running
If you've done everything right, you should be able to open the Unity project, wait for Unity Hub to download all the necessary packages, go to File > Build Settings, set the build target to the Oculus 3 device and hit Build and Run. 

## :dancer: Body-Pose Estimation
<details>

### Setup

***Disclaimer***: Streaming is not supported

Run the setup-file to setup the environment with the necessary dependencies and download models.

```
bash setup.sh
```
Add the ```--model``` flagg if only a specific model is needed (lite, full or heavy)
Example:
```
bash setup.sh --model lite
```

Use ```source .3dv/bin/activate``` to activate the created virtual environment.

### Data
Put your data into `data` folder. For images, `.jpg` is the supported format. For videos, both `.mp4` and `.mov` are supported. No need for dividing folders.

### Body_pose.py

Default runs video-processing and saves results in the `output` folder (will be created if not present)

* `--mode` : Generate pose from [Image, Video, Stream] (by default: Video)
* `--model` : Path to model (by default: models/pose_landmarker_lite.task)
* `--data` : Path to image or video data folder (by default: data) (will process only single file, if path to single file is given)
* `--visualise` : Output visualisation of joints to output folder
* `--output` : Path to output folder (by default: output). Folder will be created if does not exist
* `--set_fps` : Set output fps for data.

The output is sorted by timestamps. A video of `fps` will only be evaluated at `--set_fps`, i.e. a 60 fps video will only generate `--set_fps` datapoints for each second.

### Output formats

* ***Images***: .jpg
* ***Videos***: .mov
* ***Landmarks***: .json

</details>