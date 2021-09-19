import face_alignment
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')

MIN_CHEEK_PIXELS = 20

def roi_average_rgb(image, landmarks):
    """
    Given frames and facial landmarks, calculate the ROI in each frame, then average the rgb brightness values
    for each frame and associate it to the person. 

    image: (H, W, 3) usual image representation
    landmarks: (68, 2) representing coordinates of 68 landmarks

    returns (3,) average rgb values, or None if not enough pixels on cheek
    """
    contours = {'left': [], 'right': []}

    #add nose ends
    contours['left'].append(2*landmarks[31]-landmarks[33])
    
    #add contours around eyes
    contours['left'].append(landmarks[41])
    contours['left'].append(landmarks[36])

    #add contours around face
    contours['left'].extend(landmarks[0:4])

    #do the same for right contour
    contours['right'].append(2*landmarks[35]-landmarks[33])
    contours['right'].extend(landmarks[46:44:-1])
    contours['right'].extend(landmarks[16:12:-1])

    #get all the pixels on the cheek
    pixels = []
    for contour in contours.values():
        pixels.extend(_get_pixels_in_shape(contour))

    if len(pixels) < MIN_CHEEK_PIXELS:
        return None
    
    return np.mean(np.stack([image[p[1], p[0]] for p in pixels], axis=0), axis=0)


def _get_pixels_in_shape(contour):
    """
    Given a convex contour, get all the points inside it
    Time Complexity: O(range(contour)*|contour|)

    contour: (N, 2)
    """
    pixels = []

    #calculate bounding box rectangle
    contour_x = [p[0] for p in contour]
    contour_y = [p[1] for p in contour]
    for x in range(math.ceil(min(contour_x)), math.ceil(max(contour_x))):
        for y in range(math.ceil(min(contour_y)), math.ceil(max(contour_y))):
            if _is_point_in_hull([x,y], contour):
                pixels.append([x,y])
    return pixels
    

def _is_point_in_hull(point, contour):
    """
    Check if a point is in a convex contour, using cross product
    """
    orientation = None
    for i in range(len(contour)):
        j = (i+1) % len(contour) #next point in contour
        
        #shoelace formula for which direction point is on to the edge
        side = math.copysign(1, _shoelace(contour[j], contour[i], point))
        if orientation is None:
            orientation = side
        elif not orientation == side:
                return False
    return True

def _shoelace(x,y,z):
    """Shoelace formula for signed area"""
    return (x[0]-z[0])*y[1] + (y[0]-x[0])*z[1] + (z[0]-y[0])*x[1]

def landmark_detection(image):
    """detect face landmarks"""
    return fa.get_landmarks_from_image(image)[0]
