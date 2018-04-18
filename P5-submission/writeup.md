**Vehicle Detection Project**

The steps of this project are the following:

* Apply a color transform and extract binned color features and histograms of color, and HOG features on a labeled training set of images.
* Train a Linear SVM classifier to classify cars and non-cars.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject false positives.
* Draw a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/slide_windows.png
[image4]: ./examples/heatmap.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

#### ****** All helper functions are saved in `helper.py` and got imported to the notebook. ******

### 1. Extract features from the training images.

I started by reading in all the `car` and `non-car` images. I included all images provided by the course. Here is an example of one of each of the `car` and `non-car` classes:

![alt text][image1]

#### 1.1 Extract color features.

I explored the impact of different color spaces on the color histograms and binned color features. It doesn't seem to make much difference as long as I include all channels to capture all the color information. 

#### 1.2 Extract HOG features.

I then explored color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

The code for this step is contained in the function `get_hog_features` and it uses the function called `hog` comes in `skimage`.

I tried various combinations of parameters and settled with `YUV` and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. I chose this combination because it gave good balance between the quality of the features and the time cost to extract the features. Increase the number of HOG features would only give marginal improvement to the features but significantly increase the time it takes to extract the features.

### 2. Train a classifier using the selected HOG features and color features.

Features are scaled to zero mean and unit variance before training the classifier. I normalised the feature values using `scaler` provided in scikit learn. 

I trained a linear SVM using the features I mentioned above. I split the dataset into 80% training and 20% testing. The reported accuracy on the test dataset is 99.2%. I also tried different `C` values for the SVM classifier to see if the regularization would make a difference. It doesn't seem to make much difference in this case so I stick with the default `C=1`.

### 3. Sliding Window Search.

The code to do this is contained in function `slide_window`. I started with three different scales. This was decided depends on the rough sizes of the cars in the video far and close to the camera. 

There are two ways on deciding how the window moves, either with an overlap ratio or define how many cells to step. For different sizes of windows, you'd want to define a different overlap ratio. I ended up using `cells_per_step` in the final pipeline as this is self-adjustable depending on the window size. 

Here is an example of all the windows used in search. The search is constrained to area of interest only.

![alt text][image3]

### 4. Heatmap labels

The last step of the pipeline is to use a heatmap to label the confident detections and filter out potential false positives. From the positive detections I created a heatmap. I then masked the heatmap with a defined threshold, for instance 1, so that only the areas which are detected as positive twice at least would then be labeled as positive detection. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap and constructed bounding boxes to cover the area of each blob detected. Here are some example images:

![alt text][image4]
---

### Video Implementation

In addition to the heatmap filtering method above, for detection on video stream I generated the heatmap over multiple frames and then apply a threshold. I settled with 10 frames and threshold of 5, which roughly means an object has to be detected on half of the frames.

Here's a [link to my video result](./project_video_output.mp4)

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the labels on the integrated heatmap from all six frames, after applied threshold filter:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

#### 1. False positive

I started with the default parameters and the classifier seems to work OK on some test images. However when running on the video stream I got a lot false positives even with an applied heatmap filter. I then tried different combinations of parameters for HOG features and color spaces, but none of which could fully get rid of false positives. 

I then tried to obtain more `non-car` images in the training dataset by doing data augmentation. I tried flipped images, blurred images and croped/resized images. Again those improved the accuracy on the training dataset, but didn't help much with the performance on video images.

I finally discovered that the search window sizes and positions play an important role. I saved the frames from the video where the pipeline wasn't working well on, and tested different window sizes, start/stop positions and overlap ratio on them. This seemed to help. 

#### 2. Follow detected cars

What I could do to improve the detector is to follow the car once it's detected. The idea is to track the surrounding area of detected cars and draw the bounding box more accuractly. The full image scanning can then to deminished to happen every 1 second or so to detect new appearing cars. 

#### 3. Use deep learning model

People reported that neural network model such as RNN works better on such problem. There is this model for Real-Time Object Detection called YOLO (you only look once). this could be the next thing to try.

