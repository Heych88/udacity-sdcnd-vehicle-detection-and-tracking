**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/14.jpeg
[image2]: ./output_images/extra02.jpeg
[image3]: ./output_images/HOG_map.png
[image4]: ./output_images/YOLO_test1.jpg
[image5]: ./output_images/YOLO_test2.jpg
[image6]: ./output_images/YOLO_test3.jpg
[image7]: ./output_images/YOLO_test4.jpg
[image8]: ./output_images/YOLO_test5.jpg
[image9]: ./output_images/YOLO_test6.jpg
[image10]: ./output_images/box_output_test4.jpg
[image11]: ./output_images/heatmap_thresh_3_test1.jpg
[image12]: ./output_images/heatmap_thresh_4_test1.jpg
[image13]: ./output_images/HOG_test4.jpg

[image14]: ./output_images/heatmap_test1.jpg
[image15]: ./output_images/heatmap_threshold_6_test1.jpg
[image16]: ./output_images/objects_test1.jpg

[image17]: ./output_images/heatmap_test2.jpg
[image18]: ./output_images/heatmap_threshold_6_test2.jpg
[image19]: ./output_images/objects_test2.jpg

[image20]: ./output_images/heatmap_test3.jpg
[image21]: ./output_images/heatmap_threshold_6_test3.jpg
[image22]: ./output_images/objects_test3.jpg

[image23]: ./output_images/heatmap_test4.jpg
[image24]: ./output_images/heatmap_threshold_6_test4.jpg
[image25]: ./output_images/objects_test4.jpg

[image26]: ./output_images/heatmap_test5.jpg
[image27]: ./output_images/heatmap_threshold_6_test5.jpg
[image28]: ./output_images/objects_test5.jpg

[image29]: ./output_images/heatmap_test6.jpg
[image30]: ./output_images/heatmap_threshold_6_test6.jpg
[image31]: ./output_images/objects_test6.jpg

[video1]: ./output_images/YOLO_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.   

You're reading it!

Note: The following outlines a HOG and SVM method that was not used in the final program due to inaccuracies, lack of easy in object adaptability, as well as speed. The best method implemented is the [fast YOLO](https://arxiv.org/pdf/1612.08242.pdf) object tracking method which is outlined in the discussion section below as to why this method was choosen. The [fast YOLO](https://arxiv.org/pdf/1612.08242.pdf) was implemented using the python implementation on Github called [darkflow](https://github.com/thtrieu/darkflow).

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG features code for this step is contained in functions `hog_extract()` and `get_hog_features()`, lines 151 through 210 of the file called `objectdetection_HOG.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car data][image1] ![non car data][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(13, 13)` and `cells_per_block=(2, 2)`:


![HOG map][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and through testing the above parameters provided the best accuracy results from the SVM classifier.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training code for the SVM classifier is located in the function `train_svm()`, lines 451 through 489 of the file called `objectdetection_HOG.py`. All the images to be classified both in training and run time were passed to the function `get_object_features()`, lines 258 through 310 of the file `objectdetection_HOG.py`. In this function, all the channels of the 'HSV' HOG image, `spatial_features` and `color histogram features` are stacked together to create the engineered features of the data.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search each image in only the area of the image that an object was expected. The images were rescaled after each search loop to make them smaller, thus enabling the search box to stay a static size and search a larger area of the image for closerup objects. Each search box overlaps is neighboring box by 50%. This both minimises the amount of area overlap resulting in multiple redundent calculations, but also minimises missed or false detections on the boarder of the search boxes.

The original window size is of the full height of the search area to help find close objects. After each pass of the image with the sliding windows and window classification, the windows are scaled down by 1.5 of the previous window size to a min size of 48 pixels to help classify objects that are further away or of a smaller size.

The code for this can be found in the function `slide_window()`, at lines 213 to 256 and the function `search_windows()`, at lines 312 to 342 in the file `objectdetection_HOG.py`.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, combining this with the svm classifier and the search windows methods, the following is produced.

![window classification of objects][image10]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]. The video show the use of the yolo method which is outlined below in Discussion.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of each positive detections in each frame of the video as coordinates of the boxes location. From the positive detections I created a heatmap, function `add_heat()`, at lines 100 to 114 in the file `objectdetection_HOG.py`. and then thresholded that map setting the threshold variable `heat_thresh=2` in the function `locate_object()`, line 348 to 401 in the same file, to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
Below demonstraights the sliding window/HOG object detection in the form of a heatmap. The heatmap threshold is set at 6 resulting in an object if there is 6 or more overlaping windows.

###### The first image is the resulting heatmap of the sliding windows:

###### The second image is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

###### The third image is the resulting bounding boxes drawn onto the last frame in the series:

![alt text][image14]

![alt text][image15]

![alt text][image16]

![alt text][image17]

![alt text][image18]

![alt text][image19]

![alt text][image20]

![alt text][image21]

![alt text][image22]

![alt text][image23]

![alt text][image24]

![alt text][image25]

![alt text][image26]

![alt text][image27]

![alt text][image28]

![alt text][image29]

![alt text][image30]

![alt text][image31]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

During implementation and testing of the HOG classifier, it was observed that the methods require many parameters to be tuned. These parameters are also hapthazard as tuning for all possible cases requires alot of experimentation with different enviroment conditions and enviroment positions, such as object position. The above method is not reliable under different lighting conditions and object positions, as can be seen with image `test4.jpg`. The sliding window method also only uses 5 different sized windows adding to the error of object detection should it fit between windows.

During testing and implementing of the project, The method outlined above with multiple sliding windows, SVM and feature extraction was deemed to slow (2.45s / image) as well as unreliable. A possible replacment method is the [fast YOLO](https://arxiv.org/pdf/1612.08242.pdf) method. This not only produced more reliable detection but also much master processing, (0.24s / image). Below is a comparision of the HOG_SVM vs the Yolo method respectivly.

![HOG_SVM classifier][image16]
![YOLO classifier][image4]

The Yolo code has been implemented by the python wrapper [darkflow](https://github.com/thtrieu/darkflow). The code is avalible on [github](https://github.com/thtrieu/darkflow). I have only interfaced with this code in the function `find_objects()` at lines 33 to 37 in the file `objectdetection.py`. **Yolo code not submitted with project**

The use of a VOC or COCO classifier also helps classify different objects easily, where as the HOG and SVM method must be retrained to destinguiesh other objects. Slowing down the code more.

While both the yolo and HOG methods of object detection are not perfect, the Yolo method is more reliable, accurate and faster making it a winner between the two methods.

Note: The proccessing times mentioned above are not suitable for real time detection due to their slow speeds. Both function speeds are slow due to the large amount of python code and wraper required to be interpreted during run time.

Below is the output images of the YOLO method combined with project4 - Advanced lane finding.

![alt text][image4] ![alt text][image5]![alt text][image6] ![alt text][image7]![alt text][image8] ![alt text][image9]
