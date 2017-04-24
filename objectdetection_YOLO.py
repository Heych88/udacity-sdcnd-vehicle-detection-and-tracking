import cv2
""" Uncomment below for yolo """
#from net.build import TFNet

class yolo():

    def __init__(self, model, chkpt, threshold):
        # model : yolo model to be used for detection
        # chkpt : model weights to be used
        # thresh : threshold for object detection
        options = {"model": model, "load": chkpt, "threshold": threshold}
        """ Uncomment below for yolo """
        #self.tfnet = TFNet(options)

    def draw_box(self, image, objects, box_color=(0,0,255), show_label=False):
        # draw boxes around objects
        # image : with the objects
        # objects : x, y coordinate points of the objects
        # box_color : color of the surrounding object box
        # show_label : display the label of the object detected
        # return : image with the box location of the objects

        img = image.copy()
        for obj in objects:
            pt1 = (obj['bottomright']['x'], obj['bottomright']['y'])
            pt2 = (obj['topleft']['x'], obj['topleft']['y'])
            cv2.rectangle(img, pt1, pt2, box_color, 4)

            if show_label is True:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, obj['label'], pt2, font, 0.75, (0,255,0), 2)

        return img

    def find_object(self, image):
        # finds objects in the supplied image
        # image : image with objects
        # return : list of objects found in the image
        return self.tfnet.return_predict(image)