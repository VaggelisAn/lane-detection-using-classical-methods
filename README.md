# Assignment 3 – Introduction to Computer Vision
## Recreating a publicated lane detection algorithm using classical computer vision methods (ie. no ML).
Evangelos Ananiadis, Nikoleta Tsavlidou
Supervising Prof: Dr. Papadimitriou Katerina  
## Project Description
We implement a Lane Detection algorithm using classical computer vision methods deliberately excluding machine learning techniques. This constraint allows us to examine course material in practice. Lane detection is as interesting as it is relevant to the modern car industry, with applications such as Lane Departure Warning, Lane Keep Assist and even autonomous driving. Pipelines such as this are used in modern cars, as classical algorithms are highly efficient and optimizable. After reviewing several articles on the topic of lane detection, we decided to implement the algorithm described in K. Dinakaran, et al. (2021). This article describes an implementation that covers a large part of the course material, and as such it is relevant to our knowledge. The algorithm consists of the following steps: First, in the data pre-processing step, the camera is calibrated to correct lens distortion. The camera distortion parameters and camera intrinsic matrix were calculated using an open-source calibration tool (github.com/yashdeep01/CameraCalibrationCheckerboard). The camera view is then cropped to only include the road. Edge features are extracted using Sobel and HLS edge detection, and their results are combined (or'ed). The image is then distorted into bird’s eye view, making the lanes parallel. A sliding window search is applied to the warped image, and polynomial models are fitted to the detected lanes. Finally, the detected lane region is visualized by projecting the lane boundaries and filled lane area back onto the original image. For evaluation and testing we will construct our own dataset, on the peripheral road of Volos. For this, one person will drive, while the other person takes pictures from their phone. The dataset will be corrected and undistorted after calibrating the phone camera. Our implementation is based on our own code, dataset, materials (camera), specifications and knowledge. A Jupyter notebook was created to cover the process step-by-step on an image. By running the process_video.py script (see Setup) the lane detection polygon is overlaid on a input video. Similar pre-existing implementations and code will be used as references and they will be attributed. Our evaluation will be mostly qualitive, as we will optically examine the illustrated shapes that are overlaid on the lanes, assessing visual coherence.

![lane_detection_pipeline][resources/lane_detection_pipeline.jpg]
## Setup
conda create -n lane_detection_env python=3.10 # create a python 3.10 conda environment
pip install requirements.txt # install required packages

### lane detection on a video:
python process_video.py # process an input video (by default the video should be in the test_videos directory and named test_video.py
![video_results][resources/]

### lane detection on an image:
jupyter notebook # -> and select 
![image_results][resources/]
