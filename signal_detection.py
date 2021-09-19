import face_alignment
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')

MIN_CHEEK_PIXELS = 20
SHRINK_FACTOR = 0.8

def get_signal(rgb):
    """
    Use Skin Tone Normalization to process RGB array signal in order to control
    for illumination, skin tone and spectral reflection

    rgb: (3, ) array of rgb values
    """

    # normalize for illumination by converting signal into what it would look like in white light
    rgb_st = np.array([0.7682, 0.5121, 0.3841]) #average skin tone
    rgb_w = rgb * rgb_st[None, :] / rgb.mean(axis=0)[None, :]
    print(rgb_w.shape)
    
    # calculate chrominance signals, which removes noise due to specular reflection
    X = (rgb_w[:,0] - rgb_w[:,1]) / (rgb_st[0] - rgb_st[1])
    Y = (rgb_w[:,0] + rgb_w[:,1] - 2*rgb_w[:,2]) / (rgb_st[0] + rgb_st[1] - 2*rgb_st[2])

    # calculate signal X/Y which cancels out brightness effects
    return X/Y

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

    contours['left'] = np.stack(contours['left'])
    contours['left'] = (contours['left'] - contours['left'][0]) * SHRINK_FACTOR + contours['left'][0]

    #do the same for right contour
    contours['right'].append(2*landmarks[35]-landmarks[33])
    contours['right'].extend(landmarks[46:44:-1])
    contours['right'].extend(landmarks[16:12:-1])

    contours['right'] = np.stack(contours['right'])
    contours['right'] = (contours['right'] - contours['right'][0]) * SHRINK_FACTOR + contours['right'][0]

    #get all the pixels on the cheek
    pixels = []
    for contour in contours.values():
        
        # we don't allow contour to go outside of the image
        for vertex in contour:
            if vertex[0] > image.shape[1] or vertex[1] > image.shape[0]:
                break
        else:
            pixels.extend(_get_pixels_in_shape(contour))

    if len(pixels) < MIN_CHEEK_PIXELS:
        return None
    
    return np.mean(np.stack([image[p[1], p[0]] for p in pixels], axis=0), axis=0)

def draw_contours(image, landmarks):
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

    contours['left'] = np.stack(contours['left'])
    contours['left'] = (contours['left'] - contours['left'][0]) * SHRINK_FACTOR + contours['left'][0]

    contours['right'] = np.stack(contours['right'])
    contours['right'] = (contours['right'] - contours['right'][0]) * SHRINK_FACTOR + contours['right'][0]
    return contours


def _get_pixels_in_shape(contour):
    """
    Given a convex contour, get all the points inside it
    Time Complexity: O(range(contour)*|contour|)

    contour: (N, 2)
    """
    pixels = []

    for i in range(1, len(contour)-1):
        pixels.extend(_get_pixels_in_triange(contour[0], contour[i], contour[i+1]))
    return pixels
    """
    #calculate bounding box rectangle
    contour_x = [p[0] for p in contour]
    contour_y = [p[1] for p in contour]
    for x in tqdm(range(math.ceil(min(contour_x)), math.ceil(max(contour_x)))):
        for y in range(math.ceil(min(contour_y)), math.ceil(max(contour_y))):
            if _is_point_in_hull([x,y], contour):
                pixels.append([x,y])
    return pixels
    """
    
def _get_pixels_in_triange(A, B, C):
    """
    Given 3 endpoints of a triangle, enumerate every pixel contained in it
    
    Future Optimization: Instead of copying down each pixel, 
    just keep track of the current total rgb value and the count of valid pixels.
    This saves 2 memory copies (with extend) for each pixel in the cheek.
    """
    X,Y,Z = _sort(A,B,C)
    pixels = []
    for row in range(math.ceil(X[0]), math.ceil(Z[0])):
        bound1 = line_horizontal_intersection(row, X, Z)
        if row < Y[0]:
            bound2 = line_horizontal_intersection(row, X, Y)
        else:
            bound2 = line_horizontal_intersection(row, Y, Z)
        if bound1 < bound2:
            pixels.extend([[row, col] for col in range(math.ceil(bound1), math.ceil(bound2))])
        else:
            pixels.extend([[row, col] for col in range(math.ceil(bound2), math.ceil(bound1))])
    return pixels

def line_horizontal_intersection(row, X, Z):
    return X[1] + (Z[1]-X[1])*(row-X[0])/(Z[0]-X[0])


def _sort(A, B, C):
    ab = (A[0] < B[0])
    ac = (A[0] < C[0])
    bc = (B[0] < C[0])
    if ab:
        if bc:
            return A,B,C
        if not ac:
            return C,A,B
        return A,C,B
    if ac:
        return B,A,C
    if not bc:
        return C,B,A
    return B,C,A

    

def _is_point_in_hull(point, contour):
    """
    Check if a point is in a convex contour, using cross product
    """

    """
    coords = point[None, :] - contour
    contour.append(contour[0])
    contour[:1] - contour[-1:]
    """

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
    """

def _shoelace(x,y,z):
    """Shoelace formula for signed area"""
    return (x[0]-z[0])*y[1] + (y[0]-x[0])*z[1] + (z[0]-y[0])*x[1]

def landmark_detection(image):
    """detect face landmarks"""
    return fa.get_landmarks_from_image(image)[0]

def landmark_detection_batch(image_batch):
    """detect face landmarks"""
    return [pred[0] for pred in fa.get_landmarks_from_batch(image_batch)]
