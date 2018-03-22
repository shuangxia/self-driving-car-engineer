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
[image2]: ./example_center_camera.png "Center Camera"
[image3]: ./example_left_camera.png "Left Camera"
[image4]: ./example_right_camera.png "Right Camera"
[image5]: ./example_flipped_image.png "Flipped Image"
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

#### 1. Architecture Design Approach

I started the NVIDIA network architecture, as previous work has shown good performance. The model consists of five convolutional layers followed by three fully connected layers. I added a cropping layer at the very front to crop off the top and bottom part of images, to keep only the part where it shows the road.

#### 2. Avoid overfitting

The first training suggested overfitting, given the validation loss was much higher than the training loss. To avoid overfitting, I added one dropout layer with a drop rate of 0.5 after each of the first two fully connected layers. It turned out that the model was not overfitting anymore but with a tradeoff of accuracy. It seemed that the dropout was too strong.

![Initial Model Overview][image1]

I then tried to use a lower dropout rate (0.8) and also by removing one of the dropout layer. After a couple of trials, I settled the model with one dropout layer after the first FC layer with a rate of 0.8 as it showed the best validation score.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

After reading some comments on the choice of optimizer, I tried out a different loss function. Instead of MSE I used MAE and it turned out to improve the performance a bit more. MSE penalise larger errors more while MAE penalise errors equally. It's probably a more appropriate metrics in this case. 

The ideal number of epochs was about 7-10 as after that the validation error would start increasing again. 

### Creation of the Training Set 

I used the data provided by the project, which contains images from left, right and centre cameras.

![Left Camera][image3]![Center Camera][image2]![Right Camera][image4]

##### use side camera

I noticed that a lot of the time the car was heading straight ahead with the steering angle being 0. To add more training cases to the set I selected images from the left and right camera, treating them as they are from the the centre camera and modify the steering angle by add/substracting a correction of 0.2.

##### add augamentation

I also applied image augamentation by flipping the images to simulate the anti-clockwise driving on the same track.

![Flipped Image][image5]

##### cross validation

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

### Run the model on Autonomous mode

I first ran the car with default speed setting (9 MPH) and the car drove fine. I then started thinking "what if I increase the speed of car?". I checked the training data and saw that the speed of the car when the data was collected was about 20-30 MPH. I modified the file `drive.py` and used `set_speed` to set the speed to 20. At a faster speed the car still drove ok so that worked.

Both videos included. 
