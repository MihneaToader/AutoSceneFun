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