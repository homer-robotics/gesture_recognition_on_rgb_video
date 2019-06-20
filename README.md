## Overview
This implementation aims at performing human gesture recognition based solely on RGB videos. The underlying techniques used are **OpenPose** to extract the pose in each indivdual frame of the video and **Dynamic Time Warping** to perform the time-series classification. 

### Dependencies:
* You need OpenCV version **3.4.1** or newer. I tested it with version 3.4.3.
* This setup uses the **COCO** model, the models are not included here due to their size. Therefore you need to run ```getModels.sh``` first.

### Pitfalls
* If a pose can't be (fully) extracted due to occlusions, the frame will be skipped. This is not a problem in the UTD-MHAD, but might cause problems if you use this on different data.

### Acknowledgements
* The code for using OpenPose to extract the poses from a video is based on the example code from [this LearnOpenCV tutorial](https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/).
* The Dynamic Time Warping is calculated using the method proposed by Salvador and Chan, which was implemented in the [fastdtw library](https://pypi.org/project/fastdtw/)
