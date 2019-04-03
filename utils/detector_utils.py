# Utilities for object detector.
import imutils as imutils
import math
import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict


detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27

MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def hand_histogram(image_np, left, right, top, bottom):
    top_a = int(top + ((bottom-top) * 2 / 5))
    bottom_a = int(bottom - ((bottom - top) * 2 / 5))
    left_a = int(left + ((right - left) * 2 / 5))
    right_a = int(right - ((right - left) * 2 / 5))

    # p1 = (int(left_a), int(top_a))
    # p2 = (int(right_a), int(bottom_a))

    # cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

    objectColor = image_np[int(top_a):int(bottom_a), int(left_a):int(right_a)].copy()
    hsvObjectColor = cv2.cvtColor(objectColor, cv2.COLOR_BGR2HSV)
    objectHist = cv2.calcHist([hsvObjectColor], [0, 1], None, [12, 15], [0, 180, 0, 256])
    return cv2.normalize(objectHist, objectHist, 0,255,cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    # thresh = cv2.dilate(thresh, None, iterations=5)
    thresh = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, thresh)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("c:/temp/gray.png", gray_hist_mask_image)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def max_contour(contour_list):
    max_i = 0
    max_area = 0
    for i in range(len(contour_list)):
        cnt = contour_list[i]
        area_cnt = cv2.contourArea(cnt)
        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i
    if max_i <= len(contour_list):
        return contour_list[max_i]

# draw fingers
def manage_image_opr(frame, grep_cut_image, hand_hist):
    hist_mask_image = hist_masking(grep_cut_image, hand_hist)
    contour_list = contours(hist_mask_image)
    if (contour_list is not None):
        max_cont = max_contour(contour_list)
        if max_cont is not None:
            # draw hull
            # hull = cv2.convexHull(max_cont)
            # cv2.drawContours(frame, [hull], 0, (0, 255, 0), thickness=1)
            # draw greater than
            return calculateFingers(max_cont, frame)
    else:
        print("No contours found ...")
    return False


# Calulate Fingers
def calculateFingers(res, drawing):
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        middle_tip = tuple()
        edge = tuple()
        pointer_tip = tuple()
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    middle_tip = start
                    edge = (far[0], int(start[1] + (end[1]-start[1])/2))
                    pointer_tip = (start[0], end[1])
                    textStart = (start[0], start[1]-10)
                    # cv2.circle(drawing, far, 3, [211, 84, 0],  thickness=2, lineType=8, shift=0)
                    # cv2.circle(drawing, start, 3, [211, 84, 0],  thickness=2, lineType=8, shift=0)
                    # cv2.circle(drawing, end, 3, [211, 84, 0],  thickness=2, lineType=8, shift=0)

            # draw greater than
            if cnt == 1 and edge[1] > middle_tip[1] and edge[0] > middle_tip[0] and edge[1] < pointer_tip[1]:
                cv2.line(drawing, pointer_tip, edge, (77, 255, 9), 5)
                cv2.line(drawing, edge, middle_tip, (77, 255, 9), 5)
                cv2.putText(drawing, 'IES ASG rocks!', textStart, cv2.FONT_HERSHEY_PLAIN, 1, (77, 255, 9), 2)
                return True
            else:
                return False
    return False


# get hand contours
def draw_hand_contour(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            hand_hist = hand_histogram(image_np, left, right, top, bottom)
            # filtered_frame = image_np[int(top):int(bottom), int(left):int(right)].copy()
            grep_cut_img = grab_cut_image(image_np, int(left), int(top), int(right)-int(left),  int(bottom)-int(top))
            if (grep_cut_img is not None):
                # cv2.imwrite("c:/temp/grep.png", grep_cut_img)
                # draw greater than
                return manage_image_opr(image_np, grep_cut_img, hand_hist)
    return False


def grab_cut_image(image_np, left, top, width, height):
    mask = np.zeros(image_np.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (left, top, width, height)
    try:
        cv2.grabCut(image_np, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return image_np * mask2[:, :, np.newaxis]
    except Exception as inst:
        print(inst)


# Load a frozen infrerence graph into memory
def load_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 1, 1)


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        # flip
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
