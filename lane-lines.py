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
    
def lane_line_edges(image):
    """
    """
    gray = grayscale(image)
    smooth = gaussian_blur(gray, 5)
    return canny(smooth, 50, 150)

def lane_line_hough_lines(edge_map, ROI):
    """
    """
    mask = np.zeros_like(edge_map)
    cv2.fillPoly(mask, ROI, 255)
    masked_edges = cv2.bitwise_and(edge_map, mask)
    
    # detect line segments in edge map
    return cv2.HoughLinesP(
        masked_edges, 
        rho = 2, 
        theta = np.pi/180, 
        threshold = 10, 
        lines = np.array([]), 
        minLineLength = 10, 
        maxLineGap = 10)

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
    
def segments_to_lane_line(lines):
    """
    """
    mids = [] 
    dirs = []
    weights = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            x,y,dir,L = line_seg_params(x1,y1,x2,y2)
            if math.fabs(dir) > np.pi/12 and math.fabs(dir) < 5*np.pi/12:
                mids.append((x,y))
                dirs.append(dir)
                weights.append(L)
                    
    x = y = u = v = np.nan

    if len(weights) > 0:
        x, y = np.average(mids, axis=0, weights=weights)
        dir = np.average(dirs, axis=0, weights=weights)
        u, v = math.cos(dir), math.sin(dir)

    return x, y, u, v
    
def extract_lane_line(edge_map, ROI):
    """
    """
    # detect line segments in edge map
    hough_lines = lane_line_hough_lines(edge_map, ROI)

    # filter/combine segments into single line
    return segments_to_lane_line(hough_lines)
    
def draw_clipped_line(img, x, y, u, v, bot, top, color, thickness):
    """
    """
    y1 = bot
    t1 = (y1-y)/v
    x1 = int(round(x + t1*u))
    
    y2 = top
    t2 = (y2-y)/v
    x2 = int(round(x + t2*u))
    
    cv2.line(img, (x1,y1), (x2,y2), color, thickness)
    
def bottom_of_ROI(image):
    return image.shape[0] - 1

def top_of_ROI(image):
    return int(round(image.shape[0]*0.62))

def left_lane_ROI(image):
    """
    """
    bottom = bottom_of_ROI(image)
    top = top_of_ROI(image)
    imwidth = image.shape[1]
    roi_TL = (int(round(imwidth*0.43)), top)
    roi_BL = (int(round(imwidth*0.07)), bottom)
    roi_BR = (int(round(imwidth*0.28)), bottom)
    roi_TR = (int(round(imwidth*0.50)), top)
    return np.array([[roi_TL, roi_BL, roi_BR, roi_TR]], dtype=np.int32)
    
def right_lane_ROI(image):
    """
    """
    bottom = bottom_of_ROI(image)
    top = top_of_ROI(image)
    imwidth = image.shape[1]
    roi_TL = (int(round(imwidth*0.50)), top)
    roi_BL = (int(round(imwidth*0.79)), bottom)
    roi_BR = (int(round(imwidth*1.00)), bottom)
    roi_TR = (int(round(imwidth*0.59)), top)
    return np.array([[roi_TL, roi_BL, roi_BR, roi_TR]], dtype=np.int32)
    
def process_image(image):

    # compute canny edge map
    edge_map = lane_line_edges(image)
    
    # compute lane line ROI's
    ROI_L = left_lane_ROI(image)
    ROI_R = right_lane_ROI(image)

    # extract left/right lane lines from edge data
    xL, yL, uL, vL = extract_lane_line(edge_map, ROI_L)
    xR, yR, uR, vR = extract_lane_line(edge_map, ROI_R)
        
    # draw lane lines overlay image
    lanes_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    lane_color = [255,0,0]
    lane_thickness = 7
    clip_bottom = bottom_of_ROI(image)
    clip_top = top_of_ROI(image)
    
    #draw left lane line
    if not np.isnan(xL):
        draw_clipped_line(
            lanes_img, 
            xL, yL, uL, vL, 
            clip_bottom, clip_top, 
            lane_color, lane_thickness)
    
    #draw right lane line
    if not np.isnan(xR):
        draw_clipped_line(
            lanes_img, 
            xR, yR, uR, vR, 
            clip_bottom, clip_top, 
            lane_color, lane_thickness)
    
    # overlay lane lines onto original image
    result = weighted_img(lanes_img, image)
    # cv2.polylines(result, ROI_L, isClosed=True, color=[0,0,255], thickness=1)
    # cv2.polylines(result, ROI_R, isClosed=True, color=[0,0,255], thickness=1)
    return result
    
in_dir = 'test_images/'
out_dir = 'test_images_output/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

test_image_names = os.listdir(in_dir)

for image_name in test_image_names:

    # full pipeline
    image = mpimg.imread(in_dir + image_name)
    lanes = process_image(image)
    plt.imsave(out_dir + image_name, lanes)

    # diagnostic image: Canny edges
    edge_map = lane_line_edges(image)
    left_ROI = left_lane_ROI(image)
    right_ROI = right_lane_ROI(image)
    mask = np.zeros_like(edge_map)
    cv2.fillPoly(mask, left_ROI, 255)
    cv2.fillPoly(mask, right_ROI, 255)
    masked_edges = cv2.bitwise_and(edge_map, mask)
    plt.imsave(out_dir + 'edges-' + image_name, masked_edges, cmap='gray')

    # diagnostic image: Hough lines
    hough_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    hough_L = lane_line_hough_lines(edge_map, left_lane_ROI(image))
    hough_R = lane_line_hough_lines(edge_map, right_lane_ROI(image))
    draw_lines(hough_img, hough_L)
    draw_lines(hough_img, hough_R)
    plt.imsave(out_dir + 'lines-' + image_name, hough_img)
