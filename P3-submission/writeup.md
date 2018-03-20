# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model_architecture.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.py for making video from the recorded driving images
* video.mp4 showing the car driving in antonomous mode
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I adopted NVIDIA CNN architecture with extra dropout layers. 

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data provided by the project, which contains images from left, right and centre cameras.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The architecture is adapted from the NVIDIA network architecture, with two dropout layers between the fully connected layers to avoid overfitting.

![Model Overview][image1]

#### 2. Creation of the Training Set 

I used the training set provided by the course. 

##### use side camera

I noticed that a lot of the time the car was heading straight ahead with the steering angle being 0. To add more training cases to the set I selected images from the left and right camera, treating them as they are from the the centre camera and modify the steering angle by add/substracting a correction of 0.2.

##### add augamentation

I also applied image augamentation by flipping the images to simulate the anti-clockwise driving on the same track.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was bout 5-7 as after that the validation error would start increasing again. I used an adam optimizer so that manually training the learning rate wasn't necessary.
