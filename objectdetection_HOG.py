"""
The following implements object detection using HOG and color space features.
The code is a modified version of the code supplied during the lessons of
object detection section of the Udacity Self-driving-car Nano-degree

The code is supplied 'AS IS'
"""

import cv2
import numpy as np
from skimage.feature import hog
from scipy.ndimage.measurements import label
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class object():

    def __init__(self, cspace='RGB', spatial_size=(32, 32), orient=9,
                 hist_bins=32, hist_range=(0, 256), pix_per_cell=8,
                 cell_per_block=2, hog_channel=0, spatial_feat=True,
                 hist_feat=True, hog_feat=True):

        self.cspace = cspace
        self.spatial_size = spatial_size
        self.orient = orient
        self.hist_bins = hist_bins
        self.hist_range = hist_range
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.X_scaler = None
        self.svc = None

        # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # draws boxes of each box in bboxes
        # img : image to be converted
        # labels : 2D-array of heatmap binary
        # color : box edge color
        # thick : thickness of the box edge
        # return : image with boxes drawn

        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def draw_labeled_bboxes(self, img, labels, color=(0, 0, 255), thick=6):
        # draws boxes around a labeled heat map image
        # img : image to be converted
        # labels : 2D-array of heatmap binary
        # color : box edge color
        # thick : thickness of the box edge
        # return : image with heatmap boxes drawn on top off

        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], color, thick)
        # Return the image
        return img

    def convert_color(self, img):
        # converts the image to the class color space
        # img : image to be converted
        # return : image in modified color space

        if self.cspace != 'RGB':
            if self.cspace == 'HSV':
                cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.cspace == 'LUV':
                cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif self.cspace == 'HLS':
                cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif self.cspace == 'YUV':
                cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif self.cspace == 'YCrCb':
                cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            cvt_image = np.copy(img)

        return cvt_image

    def add_heat(self, heatmap, bbox_list):
        # combines overlapping boxes to create a heat map
        # heatmap : 2D array with a shape equal or greater than the coordinates
        #           in bbox_list.
        # bbox_list : 2D tuple with box coordinates
        # return : heatmap of the box coordinates

        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):
        # removes low counts from the heatmap
        # heatmap : 2D array of heatmap
        # threshold : min value to kep from the map
        # return : a 2D thresholded heatmap

        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    # Computes the images spatial binned colour features
    # img : input image data to extract features from
    # size : 2D array with the number of features to collect 
    # return : features vector
    def bin_spatial(self, img, size=None):

        if size is None: size = self.spatial_size
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # Computes the images colour histogram
    # img : input image data to extract features from
    # bins : number of bins to group colour values into
    # bin_range : The lower and upper value limits to be included in the bins  
    # return : Return the individual histograms, bin_centers and feature vector
    def color_hist(self, img, nbins=None, bins_range=None):

        if nbins is None: nbins = self.hist_bins
        if bins_range is None: bins_range = self.hist_range

        # Compute the histogram of the colour channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def get_hog_features(self, img, orient=None, pix_per_cell=None, cell_per_block=None,
                            vis=False, feature_vec=False):
        # calculates the HOG features of an image channel
        # img : supplied image for features extract
        # orient : number of different orientations of the HOG
        # pix_per_cell : no. of pixels per cell
        # cell_per_block : no. cells per block
        # vis : True => will return an image of the HOG
        # feature_vec : return HOG data as a features vector
        # return : HOG features for the image channel

        if orient is None: orient=self.orient
        if pix_per_cell is None: pix_per_cell = self.pix_per_cell
        if cell_per_block is None: cell_per_block = self.cell_per_block

        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    def hog_extract(self, img, orient=None, pix_per_cell=None, cell_per_block=None,
                    hog_channel=None, vis=False, feature_vec=False):
        # extracts the Histogram Of Gradients (HOG) of the supplied image
        # img : supplied image for features extract
        # orient : number of different orientations of the HOG
        # pix_per_cell : no. of pixels per cell
        # cell_per_block : no. cells per block
        # hog_channel : Which image channel to do the processing on
        # vis : True => will return an image of the HOG
        # feature_vec : return HOG data as a features vector
        # return : HOG features for the image

        if orient is None: orient = self.orient
        if pix_per_cell is None: pix_per_cell = self.pix_per_cell
        if cell_per_block is None: cell_per_block = self.cell_per_block
        if hog_channel is None: hog_channel = self.hog_channel

        feature_image = np.copy(img)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(self.get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=vis, feature_vec=feature_vec))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=vis, feature_vec=feature_vec)

        return hog_features

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # Define a function that takes an image,
        # start and stop positions in both x and y,
        # window size (x and y dimensions),
        # and overlap fraction (for both x and y)
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))

        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def get_object_features(self, image, spatial_size=None, orient=None,
                            hist_bins=None, hist_range=None, pix_per_cell=None,
                            cell_per_block=None, hog_channel=None, vis=False, feature_vec=False,
                            spatial_feat=None, hist_feat=None, hog_feat=None):
        # gets the supplied image features using call hog_extract,
        # bin_spatial() and color_hist()
        # image : supplied image for features extract
        # spatial_size : size of eachgrid box
        # orient : number of different orientations of the HOG
        # hist_bins : no. of groups for the color histogram features
        # hist_range : min and max values for the color histogram
        # pix_per_cell : no. of pixels per cell
        # cell_per_block : no. cells per block
        # hog_channel : Which image channel to do the processing on
        # vis : True => will return an image of the HOG
        # feature_vec : return HOG data as a features vector
        # spatial_feat : True => include the image spacial features
        # hist_feat : True => include the image color histograme features
        # hog_feat : True => include the image HOG features
        # return : HOG features for the image

        if spatial_size is None: spatial_size = self.spatial_size
        if hist_bins is None: hist_bins = self.hist_bins
        if hist_range is None: hist_range = self.hist_range
        if orient is None: orient=self.orient
        if pix_per_cell is None: pix_per_cell = self.pix_per_cell
        if cell_per_block is None: cell_per_block = self.cell_per_block
        if hog_channel is None: hog_channel = self.hog_channel
        if spatial_feat is None: spatial_feat = self.spatial_feat
        if hist_feat is None: hist_feat = self.hist_feat
        if hog_feat is None: hog_feat = self.hog_feat

        image = self.convert_color(image)

        #if hog_feat is True:
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = np.ravel(self.hog_extract(image, orient=orient, pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block, hog_channel=hog_channel,
                                            vis=vis, feature_vec=feature_vec))
        #features = np.concatenate((features, hog_features))

        #if spatial_feat is True:
        # Apply bin_spatial() to get spatial color features
        spatial_features = self.bin_spatial(image, size=spatial_size)
        #features = np.concatenate((features, spatial_features))

        #if hist_feat is True:
        # Apply color_hist() also with a color space option now
        hist_features = self.color_hist(image, nbins=hist_bins, bins_range=hist_range)
        features = np.concatenate((spatial_features, hist_features, hog_features))

        # Return list of feature vectors
        return features

    def search_windows(self, img, windows, clf, scaler, color_space='RGB',
                       spatial_size=(32, 32), hist_bins=32,
                       hist_range=(0, 256), orient=9,
                       pix_per_cell=8, cell_per_block=2,
                       hog_channel=0, spatial_feat=True,
                       hist_feat=True, hog_feat=True,
                       vis=False, feature_vec=False):
        # Searches a list of windows to extract features present in the window
        # img : image for feature extraction
        # windows : list of each windows bounding box pixel position
        # clf : supervised classifier type
        # scalar: column scalar for the classifier 
        # color_space : image colour space to be converted too for feature extraction
        # spatial_size : size of eachgrid box
        # hist_bins : no. of groups for the color histogram features
        # hist_range : min and max values for the color histogram
        # orient : number of different orientations of the HOG 
        # pix_per_cell : no. of pixels per cell
        # cell_per_block : no. cells per block
        # hog_channel : Which image channel to do the processing on
        # spatial_feat : True => include the image spacial features
        # hist_feat : True => include the image color histograme features
        # hog_feat : True => include the image HOG features
        # vis : True => will return an image of the HOG
        # feature_vec : return HOG data as a features vector

        on_windows = []
        # Iterate over all windows in the list
        for window in windows:
            # Extract the test window from original image an resize
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # Extract features for that window
            features = self.get_object_features(test_img,
                                                spatial_size=spatial_size, hist_bins=hist_bins,
                                                orient=orient, pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block, hist_range=hist_range,
                                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                hist_feat=hist_feat, hog_feat=hog_feat,
                                                vis=vis, feature_vec=feature_vec)


            # Scale extracted features to be fed to classifier and predict
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = clf.predict(test_features)
            # If positive prediction save the window
            if prediction == 1:
                on_windows.append(window)

        return on_windows

    def locate_objects(self, image, ystart, ystop, wstart, wstop, scale,
                       orient=None, pix_per_cell=None, cell_per_block=None,
                       spatial_size=None, hist_bins=None, hist_range=None, spatial_feat=None,
                       hist_feat=None, hog_feat=None, show_obj=False, heat_thresh=1, show_heat=False,
                       show_boxes=False):

        if spatial_size is None: spatial_size = self.spatial_size
        if hist_bins is None: hist_bins = self.hist_bins
        if hist_range is None: hist_range = self.hist_range
        if orient is None: orient=self.orient
        if pix_per_cell is None: pix_per_cell = self.pix_per_cell
        if cell_per_block is None: cell_per_block = self.cell_per_block
        if spatial_feat is None: spatial_feat = self.spatial_feat
        if hist_feat is None: hist_feat = self.hist_feat
        if hog_feat is None: hog_feat = self.hog_feat

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        box_list = []
        px_count = ystop-ystart

        while px_count >= 48:
            windows = self.slide_window(image, x_start_stop=[None, None], y_start_stop=[ystart, ystop],
                                        xy_window=(px_count, px_count), xy_overlap=(0.5, 0.5))

            box_list.extend(self.search_windows(image, windows, self.svc, self.X_scaler, color_space=self.cspace,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=self.hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat))

            # make the search window smaller
            px_count = int(px_count//1.5)

        if show_boxes is True:
            img_copy = image.copy()
            box_img = self.draw_boxes(img_copy, box_list, color=(0, 0, 255), thick=6)
            cv2.imshow('box', box_img)

        # Add heat to each box in box list
        heat = self.add_heat(heat, box_list)

        #cv2.imwrite('output_images/heatmap_test6.jpg', heat*int(255//np.max(heat)))

        # Apply threshold to help remove false positives
        heat[heat < heat_thresh] = 0
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        obj_pos = label(heatmap)
        if show_obj is True:
            draw_img = self.draw_labeled_bboxes(np.copy(image), obj_pos)

        if show_obj is True and show_heat is True:
            return obj_pos, draw_img, heatmap
        elif show_obj is True:
            return obj_pos, draw_img
        elif show_heat is True:
            return obj_pos, heatmap
        else:
            return obj_pos

    """
    Traing section of the object class
    """
    # Define a function to return some characteristics of the dataset
    def data_look(self, car_list, notcar_list):
        data_dict = {}
        # Define a key in data_dict "n_cars" and store the number of car images
        data_dict["n_cars"] = len(car_list)
        # Define a key "n_notcars" and store the number of notcar images
        data_dict["n_notcars"] = len(notcar_list)
        # Read in a test image, either car or notcar
        # Define a key "image_shape" and store the test image shape 3-tuple
        img = cv2.imread(car_list[0])
        data_dict["image_shape"] = np.shape(img)
        # Define a key "data_type" and store the data type of the test image.
        data_dict["data_type"] = img.dtype
        # Return data_dict
        return data_dict

    def get_training_data(self, file_loc, show_summary=True):
        # get all the files from a location and stores them into an array
        # file_loc : <str> address of the files location
        # show_summary : True => displays information about the supplied files
        # return : array of the list of images to open

        data = []
        for images_loc in file_loc:
            images = glob.glob(images_loc + '*.jpeg')
            group_data = []

            for image in images:
                group_data.append(image)

            data.append(group_data)

        data_info = self.data_look(data[0], data[1])
        if show_summary is True:
            print('Your function returned a count of',
                  data_info["n_cars"], ' cars and',
                  data_info["n_notcars"], ' non-cars')
            print('of size: ', data_info["image_shape"], ' and data type:',
                  data_info["data_type"])
        return data

    def train_svm(self, obj_data, not_obj_data):
        # obj_data : data of the images containing the correct class
        # not_obj_data : data images of false information

        img_data = self.get_training_data([obj_data, not_obj_data])

        car_features = []
        notcar_features = []

        for image in img_data[0]:
            img = cv2.imread(image)
            car_features.append(self.get_object_features(img))
        for image in img_data[1]:
            img = cv2.imread(image)
            notcar_features.append(self.get_object_features(img))
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        print("TRAINING")

        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        self.svc.fit(X_train, y_train)

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
