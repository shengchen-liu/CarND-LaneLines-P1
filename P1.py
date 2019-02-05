#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


test_images = os.listdir("test_images/")
output_dir = "test_images_output/"

# Global parameters

# Gaussian smoothing
kernel_size = 5

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.07  # ditto for top edge of trapezoid
trap_height = 0.4 # height of the trapezoid expressed as percentage of image height

# Hough Transform
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 20	 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 5 #minimum number of pixels making up a line
max_line_gap = 20	# maximum gap in pixels between connectable line segments

def filter_colors(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90,100,100])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

    return image2

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
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, trap_height, color=[255, 0, 0], thickness=2):
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

    # Split lines into right_lines and left_lines, representing the right and left lane lines
    # Right/left lane lines must have positive/negative slope, and be on the right/left half of the image

    left_lines = []
    right_lines = []
    img_x_center = img.shape[1] / 2  # x coordinate of center of image

    slope_limit = 0.5  # To filter out flat lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate slope
            if x2 - x1 == 0.:  # corner case, avoiding division by 0
                slope = 999.  # practically infinite slope
            else:
                slope = (y2 - y1) / (x2 - x1)
            #             cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            if slope > slope_limit and x1 > img_x_center and x2 > img_x_center:
                right_lines.append(line)
            elif slope < -slope_limit and x1 < img_x_center and x2 < img_x_center:
                left_lines.append(line)
            else:
                continue
    #             cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    right_lines = np.array(right_lines)  # (n_lines, 1, 4)
    draw_left, draw_right = True, True

    if right_lines.shape[0] > 0:
        right_lines_x = np.concatenate((right_lines[:, :, 0], right_lines[:, :, 2]), axis=0)  # (2*n_lines, 1)
        right_lines_y = np.concatenate((right_lines[:, :, 1], right_lines[:, :, 3]), axis=0)  # (2*n_lines, 1)
        # Run linear regression to find best fit line for right and left lane lines
        right_m, right_b = np.polyfit(right_lines_x.flatten(), right_lines_y.flatten(), 1)  # y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False

    left_lines = np.array(left_lines)  # (n_lines, 1, 4)
    if left_lines.shape[0] > 0:
        left_lines_x = np.concatenate((left_lines[:, :, 0], left_lines[:, :, 2]), axis=0)  # (2*n_lines, 1)
        left_lines_y = np.concatenate((left_lines[:, :, 1], left_lines[:, :, 3]), axis=0)  # (2*n_lines, 1)
        # Run linear regression to find best fit line for right and left lane lines
        left_m, left_b = np.polyfit(left_lines_x.flatten(), left_lines_y.flatten(), 1)  # y = m*x + b
    else:
        left_m, left_b = -1,1
        draw_left = False


    # Find 2 end points for right and left lines, used for drawing the line
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)

    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m

    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m

    # Convert calculated end points from float to int
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #     draw_lines(line_img, lines)
    return (lines, line_img)


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # 1. Filter colors
    # Only keep white and yellow pixels in the image, all other pixels become black
    image_raw = image
    image = filter_colors(image)

    imshape = image.shape
    # 2.Convert to gray scale
    gray_image = grayscale(image)

    # 3. Gaussian smoothing
    blur_gray = gaussian_blur(gray_image, kernel_size)

    # 4.Canny
    edges = canny(blur_gray, low_threshold, high_threshold)

    # 5. Define region of interest
    # This time we are defining a four sided polygon to mask
    vertices = np.array([[ \
        ((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]), \
        ((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height), \
        (imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]] \
        , dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # 6. Run Hough on edge detected image

    lines, line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # 7. Iterate over the output "lines" and draw lines on a blank image
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    draw_lines(line_image, lines, trap_height=0.4, color=[255, 0, 0], thickness=10)

    # 8. Draw lines on origional image
    weighted_image = weighted_img(line_image, image)
    #     plt.figure()
    #     plt.imshow(weighted_image)

    result = weighted_img(line_image, image_raw)
    return result


def main():
    import argparse
    # -----------------------------------------------
    # Arg parser
    # changes
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_NAME", help="NAME OF VIDEO FOR TEST", type=str)
    args = parser.parse_args()
    video_name = args.VIDEO_NAME
    video_output = 'test_videos_output/{}.mp4'.format(video_name)
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(6,7)
    clip1 = VideoFileClip("test_videos/{}.mp4".format(video_name))
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(video_output, audio=False)
if __name__ == '__main__':
    main()