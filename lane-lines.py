#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

import os
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def line_seg_params(x1, y1, x2, y2):
    """
    Given line segment endpoints, 
    return midpoint, direction angle, segment length.
    """
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    dx = x2 - x1
    dy = y2 - y1
    dir = math.atan2(dy, dx)
    L = math.sqrt(dx*dx + dy*dy)
    if dir < -np.pi/2:
        dir += np.pi
    elif dir > np.pi/2:
        dir -= np.pi
    return x, y, dir, L
    
def lane_lines(lines):
    eps = math.sin(np.pi/12)
    weights_L = []
    weights_R = []
    mids_L = [] 
    mids_R = []
    dirs_L = []
    dirs_R = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            x,y,dir,L = line_seg_params(x1,y1,x2,y2)
            if math.fabs(dir) > np.pi/6 and math.fabs(dir) < np.pi/3:
                if dir < 0.0:
                    weights_L.append(L)
                    mids_L.append((x,y))
                    dirs_L.append(dir)
                else:
                    weights_R.append(L)
                    mids_R.append((x,y))
                    dirs_R.append(dir)
                    
    xL = yL = dL = np.nan
    xR = yR = dR = np.nan

    if len(weights_L) > 0:
        xL, yL = np.average(mids_L, axis=0, weights=weights_L)
        dL = np.average(dirs_L, axis=0, weights=weights_L)

    if len(weights_R) > 0:
        xR, yR = np.average(mids_R, axis=0, weights=weights_R)
        dR = np.average(dirs_R, axis=0, weights=weights_R)
        
    return xL, yL, dL, xR, yR, dR

def process_image(image):

    # define ROI
    imwidth = image.shape[1]
    imheight = image.shape[0]

    roi_B = round(imheight*0.62)
    roi_T = imheight-1
    
    roi_BL = (round(imwidth*0.43), roi_B)
    roi_TL = (round(imwidth*0.07), roi_T)
    roi_TR = (round(imwidth*1.00), roi_T)
    roi_BR = (round(imwidth*0.59), roi_B)

    ROI_boundary = np.array([[roi_BL, roi_TL, roi_TR, roi_BR]], dtype=np.int32)
    
    ROI_hole = np.array(
        [[
            (round(imwidth*0.50), round(imheight*0.62)),
            (round(imwidth*0.28), roi_T),
            (round(imwidth*0.79), roi_T)
        ]], 
        dtype=np.int32)
    
    # compute masked canny edge map
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, 5)
    edge_map = canny(blur_gray, 50, 150)
    mask = np.zeros_like(edge_map)
    cv2.fillPoly(mask, ROI_boundary, 255)
    cv2.fillPoly(mask, ROI_hole, 0)
    masked_edge_map = cv2.bitwise_and(edge_map, mask)
    
    # detect line segments in edge map
    hough_lines = cv2.HoughLinesP(
        masked_edge_map, 
        rho = 2, 
        theta = np.pi/180, 
        threshold = 30, 
        lines = np.array([]), 
        minLineLength = 60, 
        maxLineGap = 1000)
        
    # extract left/right lane lines from line segment data
    xL, yL, dirL, xR, yR, dirR = lane_lines(hough_lines)
    
    # draw lane lines overlay image
    lines_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    lane_color = [255,0,0]
    lane_thickness = 7
    
    # clip lane lines to top/bottom of ROI
    y1 = roi_B
    y2 = roi_T
    
    #draw left lane line
    if not np.isnan(dirL):
        uL = math.cos(dirL)
        vL = math.sin(dirL)
        x1 = int(round(xL + uL*((y1-yL)/vL)))
        x2 = int(round(xL + uL*((y2-yL)/vL)))
        cv2.line(lines_img, (x1,y1), (x2,y2), lane_color, lane_thickness)
    
    #draw right lane line
    if not np.isnan(dirR):
        uR = math.cos(dirR)
        vR = math.sin(dirR)
        x1 = int(round(xR + uR*((y1-yR)/vR)))
        x2 = int(round(xR + uR*((y2-yR)/vR)))
        cv2.line(lines_img, (x1,y1), (x2,y2), lane_color, lane_thickness)
    
    # return original image overlayed with detected line segments
    result = weighted_img(lines_img, image)
    #cv2.polylines(result, ROI_boundary, isClosed=True, color=[0,0,255], thickness=1)
    #cv2.polylines(result, ROI_hole, isClosed=True, color=[0,0,255], thickness=1)
    return result
    
test_image_names = os.listdir("test_images/")

for image_name in test_image_names:
    image = mpimg.imread('test_images/' + image_name)
    lanes = process_image(image)
    plt.imsave('test_images_output/' + image_name, lanes)
