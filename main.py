"""
this program takes in a checkerboard image from a camera and calibrates the
image to remove camera radial and tangential distortion.
"""

import cv2
import objectdetection_YOLO as odYOLO # object detection using YOLO
import objectdetection_HOG as odHOG # object detection using an svm and HOG features
import data
import numpy as np

""" Uncomment below if adding project 4 - advanced lane detection """
#from driveline import Lane
#from camera import CameraImage
#from lane import lane_pipeline

use_yolo = False

def adjust_channel_gamma(channel, gamma=1.):
    # adjusts the brightness of an image channel
    # channel : 2D source channel
    # gamma : brightness correction factor, gamma < 1 => darker image
    # returns : gamma corrected image

    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    # http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    invGamma = 1.0 / np.absolute(gamma)
    table = (np.array([((i / 255.0) ** invGamma) * 255
                       for i in np.arange(0, 256)]).astype("uint8"))

    # apply gamma correction using the lookup table
    return cv2.LUT(channel, table)

def adjust_image_gamma(img, gamma=1.):
    # adjusts the brightness of an image
    # img : source image
    # gamma : brightness correction factor, gamma < 1 => darker image
    # returns : gamma corrected image

    # convert to HSV to adjust gamma by V
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 2] = adjust_channel_gamma(img[:, :, 2], gamma=gamma)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

# Define the codec and create VideoWriter object
if data.isVideo:
    # setup video recording when using a video
    fourcc = cv2.VideoWriter_fourcc(*  'WMV2') #'MJPG')
    filename = 'output_images/YOLO_projectvideo.wmv' # + data.video
    out = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
    # initalise the video capture
    cam = cv2.VideoCapture(data.img_add)

# setup which object detection method to use Yolo or SVM & HOG
if use_yolo is True:
    # define the yolo classifier
    # this calls the python wrapper implemented by darkflow
    # https://github.com/thtrieu/darkflow
    # this is an implementation of the yolo object detection method outlined in papers
    # You Only Look Once: Unified, Real-Time Object Detection, arXiv:1506.02640 [cs.CV],
    # YOLO9000: Better, Faster, Stronger, arXiv:1612.08242 [cs.CV]
    yolo = odYOLO.yolo(model="cfg/tiny-yolo-voc.cfg", chkpt="bin/tiny-yolo-voc.weights", threshold=0.12)
else:
    # define a SVM and HOG classifier
    car_object = odHOG.object(spatial_size=(12,12), hist_bins=34, pix_per_cell=13, hog_channel='ALL', cspace='HLS')

    # location of the training data for the SVM
    car_object.train_svm("data/vehicles_smallset/", "data/non-vehicles_smallset/")

while(1):
    # continually loop if the input is a video until it ends of the user presses 'q'
    # if an image execute once and wait till the user presses a key
    if data.isVideo:
        ret, image = cam.read()
        if ret == False:
            break
    else:
        # read in the image to the program
        image = cv2.imread(data.img_add, -1)

    """ object detection """
    if use_yolo is True:
        # YOLO classifier
        gamma_img = adjust_image_gamma(image.copy(), 2)
        objs = yolo.find_object(gamma_img) # find the objects
        image = yolo.draw_box(image, objs, show_label=True) # add the detected objects to the window
    else:
        h, w = image.shape[:2]
        # SVM and HOG classifier
        gamma_img = adjust_image_gamma(image.copy(), 2)
        obj_pos = car_object.locate_objects(gamma_img, h // 2, h-80, 0, w, scale=2, show_obj=False,
                                                       show_boxes=False, heat_thresh=6, show_heat=False)
        image = car_object.draw_labeled_bboxes(image, obj_pos, color=(0, 0, 255), thick=6)

    cv2.imshow('final', image)

    # wait for a user key interrupt then close all windows
    if data.isVideo:
        out.write(image)  # save image to video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # save the new image
        cv2.imwrite('output_images/objects_' + data.image, image)
        cv2.waitKey(0)
        break

if data.isVideo:
    out.release()
    cam.release()

cv2.destroyAllWindows()
