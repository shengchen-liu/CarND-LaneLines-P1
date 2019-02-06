# **Finding Lane Lines on the Road** 
[//]: # (Image References)

[image1]: ./examples/Pipeline.png "Pipeline"
[image_challenge]: ./examples/CarND-LaneLines-P1-challenge.gif "challenge"

![alt text][image_challenge]
---

**Finding Lane Lines on the Road**

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road

---

### How to run the code

This project requires Python3

Type 'python find_lane.py --[VIDEO_NAME]

In the [VIDEO_NAME] option, choose one of 'solidWhiteRight', 'solidYelloLeft' or 'challenge' 

---

### Reflection

### 1. Pipeline description 
My pipeline consisted of 5 steps.

In order to draw a single line on the left and right lanes, the function 'draw_lines()' includes :
1. Calculate slopes for each single line.
2. Filter Hough lines by slope and end points.
3. Seperate filterd lines into left/right candidates.
4. Do linear fittings to identify the full extent of left/right lanes.

The image below shows how the pipeline works: 

![alt text][image1]


### 2. Potential shortcomings with the current pipeline

One shortcoming of this algorithm comes from 'image preprocessing', whihch is filter out all colors except yellow and white pixels. This technique can reduce the wrong detection of tree's shadows and small discrepencies on the road. However, at night or during bad weahters, simply filtering the color will harm the accuracy.


### 3. Possible improvements to the pipeline

A possible improvement would be to use a more robust algorithm to filter out unwanted color pixels.

Another suggestion is to use deep learning methods to do lane detection.

---
### Youtube Videos
Please click the following links to view the results on Youtube:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/VLQvyDKHezw/0.jpg)](https://www.youtube.com/watch?v=VLQvyDKHezw) 
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/lZS761WeNTA/0.jpg)](https://www.youtube.com/watch?v=lZS761WeNTA)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/T7F7wQDoybc/0.jpg)](https://www.youtube.com/watch?v=T7F7wQDoybc)