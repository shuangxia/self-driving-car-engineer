# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidYellowLeft.jpg 

---

## Reflection		

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps. 
1) resize the image to the standard 960 *540 size as this is what the current pipeline works on, 
2) converted the images to grayscale, 
3) applied Gaussian kernel to smooth the noise, 
4) applied Canny transform to detect the edges, 
5) applied region filter to select the region of interest only, 
6) applied Hough transform to find lines from Canny edges, and 
7) draw the lanes based on the detected line segments.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by 
1) separting the line candidates into left lane or right lane group, given the left lane has a positive slope and the right lane has a negative one, 
2) calculating the median slope and intercept of the left and right line candidates, and 
3) using the calculated slope and intercept for the final lanes, and extrapolate it to the region of interest.

### 2. Identify potential shortcomings with your current pipeline

- One potential shortcoming would be in the chanllenge video when the car drives in tree shadow, it failed to detect the actual lane for a second.

- Another would be when there is more than one left or right lane in the region of interest, the currently pipeline draws only one lane on each side, so it will draw one lane somewhere between the actual lanes.

- Another one could be the region of interest. Currently it's hard coded, but it could change if the size of the vehicle changes or the side of driver changes, or if the angle of the camera changes.


### 3. Suggest possible improvements to your pipeline

- A possible improvement would be to cluster the slopes of the line candidates (e.g. KNN or something similar) to find out the actual number of lanes, filter out the lines unlikely to be a road lane based on the slope (for instance, the front of the car in the challenge video), and then do extrapolation for each lane accordingly. At the moment, I use a filter of > 0.5 and < -0.5 for legit lanes.

- Another potential improvement could be to add region of interest detection. I can see the select of area of interest can impact the performance of the pipeline. At the moment the area is hard coded, which is not ideal. It can probably be calculated based on the height and angle of the camera, the driver side, the heading of the car, etc.
