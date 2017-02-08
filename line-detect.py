import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


##################################
# Camera calibration
##################################

# prepare object points
nx = 9 # number of inside corners in x
ny = 5 # number of inside corners in y

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration1.jpg')
#print(images)
# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
#        cv2.imshow('img', img)
#        cv2.waitKey(500)

# Takes an image, object points, and image points
# performs the camera calibration, image distortion correction and
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       None,
                                                       None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


img = cv2.imread('camera_cal/calibration3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

img_size = (img.shape[1], img.shape[0])
offset = 500

# For source points I'm grabbing the outer four detected corners
src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
# For destination points, I'm arbitrarily choosing some points to be
# a nice fit for displaying our warped result
dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                  [img_size[0] - offset, img_size[1] - offset],
                  [offset, img_size[1] - offset]])
# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)

# Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(img, M, img_size)

fname = 'test_images/test1.jpg'
img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

undistorted = cal_undistort(img, objpoints, imgpoints)

def perspective_transform(img, offset=450, btm_offset=30, vcutout=0.66):
    h = img.shape[0]
    w = img.shape[1]
    src = np.float32([[offset, int(h*vcutout)], [w-offset, int(h*vcutout)], [w-1, h-btm_offset], [0, h-btm_offset]])
    dst = np.float32([[0,      0             ], [w-1,      0             ], [w-1, h-1  ], [0, h-1  ]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    return warped, M



import utils

visualize = True


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #confidence of the best fit
        self.best_conf = None
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = None #[np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #array of all fits
        self.fits = []

left_line = Line()
right_line = Line()

#updates the history of detection and determines the best
#latest fit based on current fit and confidence
def updateLine(line, cur_fit, x_diff_thresh = 50):
    if cur_fit is None:
        return line.current_fit
    if line.current_fit is None:
        line.current_fit = cur_fit

    cur_x0 = cur_fit[2]
    best_x0 = line.current_fit[2]
    print('diff:', abs(cur_x0-best_x0))
    if abs(cur_x0-best_x0) < x_diff_thresh:
        line.current_fit = (line.current_fit + cur_fit)/2
        line.fits.append(cur_fit)
        line.detected = True
    else:
        line.detected = False

    return line.current_fit


def calc_line_center(line):
    if line.detected:
        return line.current_fit[2]  #X position at Y=0
    else:
        return None

def visualize_fit(img, fit):
    h = img.shape[0]
    w = img.shape[1]
    yvals = np.arange(0, h, 1)

    xvals = (fit[0] * yvals ** 2 + fit[1] * yvals + fit[2]).astype('int')
    xvals = np.maximum(0, np.minimum(w - 1, xvals))
    img[yvals, xvals] = 1

    return img

# Returns an image with lane area overlaid on it
def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if visualize:
        if not combined_vis:
            axes = []
            for i in range(9):
                f, ax = plt.subplots(1, 1, figsize=(40, 20))
                axes.append(ax)
            axes = np.array(axes).reshape((3,3))
        else:
            f, axes = plt.subplots(3, 3, figsize=(40, 20))
        axes[0][0].set_title('Original Image')
        axes[0][1].set_title('Pipeline Result')
        axes[0][2].set_title('Combined Thresholded')
        axes[1][0].set_title('Perspective Transform')
        axes[1][1].set_title('Column Histogram')
        axes[1][2].set_title('Detected Lane Points')
        axes[2][0].set_title('Fitted polynomial')
        axes[2][1].set_title('Detected Lane Area')

    # undistort
    undistorted = cal_undistort(image, objpoints, imgpoints)

    # Create thresholded binary image
    stacked, binimg = utils.pipeline(undistorted)
    stacked = stacked*255

    if visualize:
        axes[0][0].imshow(image)
        axes[0][0].get_xaxis().set_visible(axis_visible)
        axes[0][0].get_yaxis().set_visible(axis_visible)

        axes[0][1].imshow(stacked)
        axes[0][1].get_xaxis().set_visible(axis_visible)
        axes[0][1].get_yaxis().set_visible(axis_visible)

        axes[0][2].imshow(binimg, cmap='gray')
        axes[0][2].get_xaxis().set_visible(axis_visible)
        axes[0][2].get_yaxis().set_visible(axis_visible)

    # perspective transform
    perspimg, M = perspective_transform(binimg)
    #Deactivate small chunk from the left and right
    #perspimg[:,:200] = 0
    #perspimg[:,-100:] = 0
    if visualize:
        axes[1][0].imshow(perspimg, cmap='gray')
        axes[1][0].get_xaxis().set_visible(axis_visible)
        axes[1][0].get_yaxis().set_visible(axis_visible)

    #Detect lane pixels and fit to find lane boundary
    hist, l_lane, r_lane = utils.detect_lane(perspimg,
                                             l_prev_fit=left_line.current_fit,
                                             r_prev_fit=right_line.current_fit)

    if visualize:
        axes[1][1].plot(hist)
        #axes[idx][4].set_aspect(axes[idx][0].get_aspect())
        axes[1][1].get_xaxis().set_visible(axis_visible)
        axes[1][1].get_yaxis().set_visible(axis_visible)
        # Show detected lane points
        det_lanes_img = np.zeros_like(perspimg)
        det_lanes_img[l_lane[:, 1], l_lane[:, 0]] = 1
        det_lanes_img[r_lane[:, 1], r_lane[:, 0]] = 1
        axes[1][2].imshow(det_lanes_img, cmap='gray')
        axes[1][2].get_xaxis().set_visible(axis_visible)
        axes[1][2].get_yaxis().set_visible(axis_visible)

    #fit polinomial
    left_lane_fit, l_conf = utils.fit_polynomial(l_lane)
    right_lane_fit, r_conf = utils.fit_polynomial(r_lane)
    if left_lane_fit is None or right_lane_fit is None:
        print('no fit')
        left_line.detected = False
        left_line.current_fit = None
        right_line.detected = False
        right_line.current_fit = None
        return image
    '''
    min_lane_w = 400
    max_lane_w = 900
    h = image.shape[0]
    l_basex = utils.calc_point(h-1, left_lane_fit)
    r_basex = utils.calc_point(h-1, right_lane_fit)
    if l_basex > r_basex or abs(l_basex-r_basex) not in range(min_lane_w, max_lane_w):
        print('no fit')
        left_line.detected = False
        left_line.current_fit = None
        right_line.detected = False
        right_line.current_fit = None
        return image
    '''
    print('fit:', list(left_lane_fit), list(right_lane_fit))

    '''
    lane_width_px = 500
    if l_conf == 0 and r_conf != 0:
        left_lane_fit = right_lane_fit + [0, 0, -lane_width_px]
    if l_conf != 0 and r_conf == 0:
        right_lane_fit = left_lane_fit + [0, 0, lane_width_px]
    '''

    if visualize:
        poly_img = np.zeros_like(perspimg)
        poly_img = visualize_fit(poly_img, left_lane_fit)
        poly_img = visualize_fit(poly_img, right_lane_fit)
        axes[2][0].imshow(poly_img, cmap='gray')
        axes[2][0].get_xaxis().set_visible(axis_visible)
        axes[2][0].get_yaxis().set_visible(axis_visible)

    left_lane_fit = updateLine(left_line, left_lane_fit)
    right_lane_fit = updateLine(right_line, right_lane_fit)

    #Extrapolate:
    h = perspimg.shape[0]
    w = perspimg.shape[1]
    yvals = np.arange(0, h, 1)
    center = w/2
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    left_xvals = None
    right_xvals = None
    if left_lane_fit is not None:
        # Extrapolate
        left_xvals  = (left_lane_fit[0]*yvals**2  + left_lane_fit[1]*yvals  + left_lane_fit[2] ).astype('int')
        left_line.bestx = int(left_xvals[h-1])
        left_line.line_base_pos = abs(left_line.bestx - center)*xm_per_pix
        # Fix outliers (created by extrapolation)
        left_xvals = np.maximum(0, np.minimum(w-1, left_xvals))
        #Determine curvature of the lane and vehicle position with respect to center.
        left_curv = utils.curvature_m(left_xvals, yvals)
    if right_lane_fit is not None:
        # Extrapolate
        right_xvals = (right_lane_fit[0]*yvals**2 + right_lane_fit[1]*yvals + right_lane_fit[2]).astype('int')
        right_line.bestx = int(right_xvals[h-1])
        right_line.line_base_pos = abs(right_line.bestx - center)*xm_per_pix
        # Fix outliers (created by extrapolation)
        right_xvals = np.maximum(0, np.minimum(w-1, right_xvals))
        #Determine curvature of the lane and vehicle position with respect to center.
        right_curv = utils.curvature_m(right_xvals, yvals)

    left_curv /= 1000 #km
    right_curv /= 1000#km
    left_curv_str = 'straight' if left_curv > 10 else '%.2f km' % left_curv
    right_curv_str = 'straight' if right_curv > 10 else '%.2f km' % right_curv
    print(left_curv_str, right_curv_str)

    from numpy.linalg import inv
    #Warp the detected lane boundaries back onto the original image
    unwarped = utils.unwarp(image, perspimg, inv(M), left_xvals, right_xvals, yvals)

    curv_str = 'curvature (l,r): (%s, %s)' % (left_curv_str, right_curv_str)
    dev_str =  'dev from center: %0.2fcm' % (100*(left_line.line_base_pos - right_line.line_base_pos)/2)
    #dev_str =  'deviation m  (l,r): (%0.2f, %0.2f)' % (left_line.line_base_pos, right_line.line_base_pos)
    txtsize = 1
    thickness = 2
    txtcolor = (0, 255, 255) #yellow
    cv2.putText(unwarped, curv_str, (100, 100), cv2.FONT_HERSHEY_DUPLEX, txtsize, txtcolor, thickness)
    cv2.putText(unwarped, dev_str, (100, 150), cv2.FONT_HERSHEY_DUPLEX, txtsize, txtcolor, thickness)
    #print the info to the image
    if visualize:
        axes[2][1].imshow(unwarped)
        axes[2][1].get_xaxis().set_visible(axis_visible)
        axes[2][1].get_yaxis().set_visible(axis_visible)

        plt.show()

    return unwarped


#Make a list of calibration images
images = glob.glob('test_images/*.jpg')
for i in range(0):#len(images)):
    # Read in an image and convert to RGB
    image = cv2.imread(images[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = process_image(image)


cap = cv2.VideoCapture("project_video.mp4")
#cap = cv2.VideoCapture("challenge_video.mp4")
#cap = cv2.VideoCapture("harder_challenge_video.mp4")

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = None

frame_cnt = 0
frame_start = 0
frame_end = 0xffffffff
visualize = False
axis_visible = True
combined_vis = True

while True:
    flag, image = cap.read()
    if flag:
        frame_cnt += 1
        if frame_cnt < frame_start:
            continue
        elif frame_cnt > frame_end:
            break
        print('frame_cnt = ', frame_cnt)
        if out == None:
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (image.shape[1], image.shape[0]))
        h = image.shape[0]
        w = image.shape[1]

       # imcpy = np.zeros_like(image)
       # imcpy[i*h/2:(i+1)*h/2][j*w/4:(j+1)*w/4] = image[i*h/2:(i+1)*h/2][j*w/4:(j+1)*w/4]
       # imcpy[i*h/2:(i+1)*h/2][j*w/4+w/2:(j+1)*w/4+w/2] = image[i*h/2:(i+1)*h/2][j*w/4+w/2:(j+1)*w/4+w/2]

        res = process_image(image)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.imshow('video', res)
        out.write(res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
out.release()

import time
time.sleep(2) # delays for 5 seconds
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.tigh