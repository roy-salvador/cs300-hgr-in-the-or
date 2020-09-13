# cs300-hgr-in-the-or

A Medical Image Navigation prototype application powered by hand gesture commands, created to demonstrate the feasibility of a Deep Learning and Computer Vision-based Hand Gesture Recognition System for Operating Room usage. A deliverable requirement for CS 300 Thesis course, successfully defended Midyear 2020 at the University of the Philippines Diliman under Sir Prospero Naval.

![Demo Application](https://github.com/roy-salvador/cs300-hgr-in-the-or/blob/master/demo.gif)

For more details, please check [paper](https://www.academia.edu/44021842/Towards_a_Feasible_Hand_Gesture_Recognition_System_as_Sterile_Interface_in_the_Operating_Room_with_3D_Convolutional_Neural_Network) and [YouTube video](https://youtu.be/bR4XhAHzdFk) on how the system would be used.

## Requirements
* Python 3.5
* [OpenCV](https://opencv.org/opencv-3-2/)
* [imutils](https://pypi.org/project/imutils/)
* [Lasagne](https://lasagne.readthedocs.io/en/latest/)
* [Plotly](https://plotly.com/python/)

## Prerequisite
Trained network with Jester Dataset named as `network\jester.npz`. For more detailson training the network, check [training](https://github.com/roy-salvador/cs300-hgr-in-the-or/tree/master/training) folder.

## Running Demo Application
To run the prototype application using your webcam to capture the gestures:

```
python DemoMain.py
```  
