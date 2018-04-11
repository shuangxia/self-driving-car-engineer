**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./undistorted.png "Undistorted"
[image2]: ./road_transformed.png "Road Transformed"
[image3]: ./binary_combo_example.png "Binary Example"
[image4]: ./warped.png "Warp Example"
[image5]: ./color_fitted_lines.png "Fit Visual"
[image6]: ./example_output.png "Output"
[video1]: ./project_video_solution.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### Calibrate the camera using examples of chessboard images

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

Use the camera calibration step described above to get a distortion-correct image:
![alt text][image2]

#### 2. Example of a binary image resulted from a combined thresholding pipeline of color transforms, gradients and other methods.

I used a combination of color and gradient thresholds to generate a binary image (gradient x, gradient y, gradient magnitude, gradient direction, HLS color space).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Example of a perspective transformed image.

The `perspective_transform()` function takes as inputs an image (`img`), and define source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points as below:

```python
src = np.float32([[200,720],[1200,720],[750,470],[580,470]])
dst = np.float32([[200,720],[1020,720],[1020,0],[200,0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Example of fitted curve from detected edges.

Take the left and right line pixels identified from the binary warped image from the previous step, and fit with polynomial function.

There are two ways of identifying pixels from the binary image. One is defined in `slide_window_search` function, which use the color histogram to detect the peaks (position of the lanes), and then divide the image horizontally into n windonws and search in each window for the line pixels. This method is used when there is no good lines detected previously. 

The other way is defined in `continuous_search` function, which takes the pixels detected from the previous image and use those as the starting points to create two searching windows. The method saves time but can only be used when good lanes are detected from previous frames. If no good lane has been detected from the past three frames, the pipeline will go back to using the `slide_window_search`.

![alt text][image5]

#### 5. Sanity check on the detected lanes.

Three checks are done to see if the lanes are valid.

* Checking that they have similar curvature
* Checking that they are separated by approximately the right distance horizontally
* Checking that they are roughly parallel

I implement this step in function `sanity_check()`

#### 6. Smooth over the past detected lanes.

To avoid the lanes jumping around from frame to frame, I take the past 4 detected lanes and use the average positions as the current best fit.

#### 7. Calculate radius of curvature of the lane and the position of the vehicle with respect to center.

Take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature is given in meters assuming the curve of the road follows a circle. For the position of the vehicle, assume the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset. 

I implemented this two steps in function `cal_curvature()` and `center_offset()`.

#### 9. Visualise the detected valid road area.

Take the detected lanes and color fill the area in between.

#### 10. Example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1.  Here is a link to my final video output. 

I have some wobbly lines when the car driving through shadows, but no catastrophic failure.

Here's a [link to my video result][video1]

---

### Discussion

#### 1. The project video

Current pipleline works fairly good on the project video. It has some wobbly lines but no major failure. To get rid of the wobbly lines, a better smoothing process might be needed. I currently smooth over the last 4 valid fitted lines, which might not be high enough. I should probably also smooth the fit coefficients.

#### 2. The challenge video

I had a go with the challenge video and the result turned out to be worse than the project video. The pipeline can't detect the lane lines accurately especially when the brightness of the images changes. The car position on the road is also different from the project video - the car bares on the left of the road a lot. This means the hard coded coordinates used for the perspective transformation are not valid anymore. The sanity check gaurantees that the fitted lanes are still reasonably smooth through the video but it just not always cover the right area.

To make the pipeline works better on the challeng video, the performance of the binary thresholding needs to be improved. Image augmentation might need to be considered, such as brightness adjustment. The thresholds need to be tuned too.
