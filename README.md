##Lane Detection

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/selected_pixels.jpg "Selected Line Pixels"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./examples/combined_colored.jpg "Color and Gradient Combined"
[image8]: ./examples/fitted_polynomial.jpg "Fitted Polynomial"
[image9]: ./examples/combined_pipeline.jpg "Combined Pipeline"
[video1]: ./project_video_outpu.mp4 "Output Video"

---
###Usage
Run
`python line-detect.py`

###Dependencies
* python 2.7
* numpy
* scipy
* opencv
* anaconda (reccomended)

###Camera Calibration

The code for this step is located in file `camera-calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 247 through 249 in `utils.py`).  
Here's an example of my output for this step.

![alt text][image7]
![alt text][image3]

####Color Selection
Color selection is done in function `select_white_and_yellow()` in file `utils.py` and is based on converting the image to HSV color space and thresholding the pixels agains precalculated minimum and maximum values of white and yellow colors.
Thresholds:
```
lower_yellow = np.array([0, 0, 40])
upper_yellow = np.array([100, 255, 255])
lower_white = np.array([0, 160, 0])
upper_white = np.array([255, 255, 255])
```
This technique is very usefull to filter out edges created by shadows and thus assosiated with darker colors.

####Gradient Selection
Gradient selection is done in function `pipeline()` in file `utils.py` and is based on magnitute and direction of the gradients. Several techniques are used including sobel gradient detection, gradient magnitute calculation and gradient direction which was thresholded in the range `(pi/10, pi - pi/10)`

To better understand which technique contributes to which part of the binary image I used two colors to represent color and gradient results overlayed on the same image as can be seen in the image below.
![alt text][image7]

####Perspective Transform
The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 95 through 103 in the file `line-detect.py`. The `perspective_transform()` function takes as inputs an image (`img`) and three tuning parameters with defualt values that allowed to experiment with the source and destination object points as follows:
* offset=450    - determines the upper left and right corners in the source image
* btm_offset=30 - determines the bottom left and right corners in the source image
* vcutout=0.66  - determines the horizontal bottom portion of the image to be scaled to full height in transformed image
The source and destination points are then calculated as follows:
```
    src = np.float32([[offset, int(h*vcutout)], [w-offset, int(h*vcutout)], [w-1, h-btm_offset], [0, h-btm_offset]])
    dst = np.float32([[0,      0             ], [w-1,      0             ], [w-1, h-1  ], [0, h-1  ]])
```
This resulted in the following source and destination points:
 
| Source        | Destination   | 
|:-------------:|:-------------:| 
|[  450   475 ] |[    0     0 ] | 
|[  830   475 ] |[ 1279     0 ] |
|[ 1279   690 ] |[ 1279   719 ] |
|[    0   690 ] |[    0   719 ] |
       
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
       
![alt text][image4]
       
####Lane-line Pixel Detection
To detect lane line pixels I used two pass approach to calculate the general direction of the line using linear regression and then extract the pixels around the line found in the first pass. The raw pixels are converted to (x,y) point vectors in funtion `extract_points()` in file `utils.py` which is passed to numpy's `polyfit` function to return the 2nd degree fit to pixels. The following sections contain more information about both passes.
        
####First Pass - General Direction
The code for this method is located in the first half of the function `detect_lane()` in file `utils.py`. I used the [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) (Random Sample Concensus) method to estimate the direction of the line in left and right parts of the image separately. Convinitently, [scipy](https://www.scipy.org/scipylib/download.html) library implements this [algorithm](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html) which is invoked in function `ransac()` in file `utils.py`. The `ransac()` function returns set of points that are considered to be "inliners", which are pixel that lie within certain distance from the general trend. Those points are then passed to `fit_polynomial()` function that uses numpy's `polyfit()` function to estimate a linear fit for the data. It is important to use linear estimation here because parabolas can oscilate wideley when data is noisy.

####Second Pass - Sliced Windows
In this step, which is located in the second half of the `detect_lane()` funtion, the image is divided to several (configurable) slices and points extracted from each slice individually based on the fit found in the first pass plus some slack to account for noisy data. More specifically, for each slice the starting x offset and ending x offset are calculated by applying the fit found in the first pass to top and bottom y coordinates of the current slice in function `calc_focus()` in file `utils.py`. The relevant lane line points, for left and right lines separately, are then extracted by the function `extract_points()` that receives the image and a vector in the form `[starting x, width, starting y, height]` and returns a vector of (x,y) coordinates. The points from all the slices are combined to one vector and returned to the calling function `process_image()` in `line-detect.py`.

The following image visualizes the selected pixels in the binary image after applying both passes.       
![alt text][image5]

####Polynomial Fitting
The lane-line pixels detected in previous step by `detect_lane()` are then fed by the `process_image()` function in 'line-detect.py` file to the `fit_polynomial()` function that uses numpy's `polyfit()` with `degree=2` to fit a polynomial function to the data. The resulting fit is then fed to `updateLine()` function that compares it to the fit from previous images and determines whether to discard the fit. For this purpose the x position at the top of the image (y=0) is calculated for both the previous and the current fits and compared, the current fit is accepted only if the x coordinates in both images are below a threshold (configurable). If the current fit is accepted it is averaged with the previous fit to produce smoother transitions in video and prevent line-jumping. The function `process_image()` then extrapolates the line points for each height coordinate using the updated fits for left and right lanes. The resulting polynomial curvatures for both lines can be seen in the following image.
![alt text][image8]

####Curvature and Center
Curvature and deviation from lane center is done in the end of `process_image()` function in file `line-detect.py`. For this purpose it invokes the function `curvature_m()` from `utils.py` which receives vectors for x and y coordinates of the fitted polynomial, converts them from pixel space to meter space by multiplying by a constant, fits polynomial again, this time in meter space and calculates curvature based on the following formula:
```
curv = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
```
Deviation from center is calculated in function `process_image()` in `line-detect.py` by substructing the distance from image center for each line (assuming the center of the image is the center of the vehicle) and then substructing from each other, under the assumption that the center of the vehicle should be in equal distance from both the left and right lane to be in the center. For convinience the curvature is converted to kilometers and deviation from center to centimeters.
       
####Lane Area Overlay
The code for overlaying the lane area on the original image is located in function `unwarp()` in `utils.py`. This function receives the original image, the warped image, inverted perspective transform matrix, and three vectors - y values from 0 to the height of the image, and x values calculated by applying the fit for each line to y values. The values are computed in function `process_image()`. The polygon created by the points is drawn onto the warped image with light green color and the image is then goes through inverted perspective transform obatained by the caller function by using numpy's `inv()` function on the original perspective transform matrix calculated in `perspective_transform()` function in the beginning of the pipeline. As the final step the information about line curvatures and deviation from center is printed on the unwarped image to achieve the following result:
![alt text][image6]

####Combined Pipeline
To help debug the pipeline the temporary results shown above are collected for each image and combine to one 3x3 grid image. This mode can be turned off or on by modifying the `combined_vis` variable in `line-detect.py`. In addition, the pipeline visualization feature can be turned off or on by modifying the `visualize` variable in `line-detect.py`. The combined pipeline image looks like this:
![alt text][image9]
       
---    
       
###Pipeline (video)
           
Here's a [link to my video result](https://youtu.be/L7T6gimylXo) or to [local video file](./project_video_output.avi)
       
---    
       
###Discussion
       
I will describe here two biggest challenges I had during the project and how I tackled them.

####Shadows
Shadows turned out to be an edge monster, and whats worse they can appear on the road as opposed to other edges on the horizon which can be cut off during the perspective transform. Fortunetely, they also posses one characteristic that can help distinguish them from the edges we want - lane lines. Shadows, by definition, are darker areas on the image relatively to the lane lines which are white or yellow, therefore color filtering was very effective technique to weed out the shadows. The color filtering technique is described in previous sections.

####Close Cars
Cars, even when black have white edges due to reflection from windows and shiny paint. They can also appear close to the lane lines and have direction and length close to the lane segments. My first approach was using linear polyfit to estimate the general direction of the curve but this approach usually breaks with close car edges because they represent many pixels that sway the linear estimation heavily and as expected proportinally to the number of pixels. To solve this problem I used RANSAC algorithm as a robust method to fit linear function to the data points ignoring the outliers. More details about the RANSAC method can be found in previous sections. 

      
