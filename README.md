# **Finding Lane Lines on the Road** 
[//]: # (Image References)

[image1]: ./examples/Pipeline.png "Pipeline"
[image_challenge]: ./examples/CarND-LaneLines-P1-challenge.gif "challenge"

![alt text][image_challenge]
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road

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