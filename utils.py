import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

##################################
# create a thresholded binary image
##################################


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y')
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sobel_binary = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sobel_binary

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 6) Create a binary mask where mag thresholds are met
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 7) Return this mask as your binary_output image
    return sobel_binary

# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the direction of the gradient
    np.seterr(divide='ignore', invalid='ignore')
    sobel_dir = np.arctan(sobely / sobelx)
    # 4) Take the absolute value
    abs_sobel_dir = np.absolute(sobel_dir)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gray)
    # 6) Return this mask as your binary_output image
    binary_output[(abs_sobel_dir >= thresh[0]) & (abs_sobel_dir <= thresh[1])] = 1
    return binary_output

# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    # 2) Apply a threshold to the S channel
    binary = np.zeros_like(s)
    binary[(s > thresh[0]) & (s <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    binary_output = binary
    return binary_output

# Define a function that thresholds the S-channel of HLS
def select_white_and_yellow(img):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #Filter out all colors except yellow and white:
    lower_yellow = np.array([0, 0, 40])
    upper_yellow = np.array([100, 255, 255])
    lower_white = np.array([0, 160, 0])
    upper_white = np.array([255, 255, 255])
    ymask = cv2.inRange(hls, lower_yellow, upper_yellow)
    wmask = cv2.inRange(hls, lower_white, upper_white)
    mask = np.logical_or(ymask, wmask)
    return mask

def visualize_params_dir(img, win=np.pi/32, step=np.pi/64, tmin=0, tmax=np.pi/2, ksize=3):
    #best - (1.13, 1.23)
    img = np.copy(img)

    nb_subplots = int((tmax - tmin) / step) + 3
    rows = np.ceil(nb_subplots ** 0.5)
    cols = rows
    for i in range(nb_subplots):
        curmin = i * step + tmin
        curmax = min(curmin + win, tmax)
        #mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(curmin, curmax))
        dir_binary = dir_threshold(img, ksize, (curmin, curmax))
        plt.subplot(rows, cols, i+1)
        # hide axes
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        title = ('(%.2f' % curmin) + (', %.2f' % curmax + ')')
        cur_axes.axes.set_title(title)
        plt.imshow(dir_binary, cmap='gray')

    plt.subplot(rows, cols, nb_subplots+1)
    # hide axes
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    title = 'Original'
    cur_axes.axes.set_title(title)
    plt.imshow(img)

    plt.show()

def visualize_params_mag(img, win=150, step=10, tmin=0, tmax=150, ksize = 3):
    # best (50, 150)
    img = np.copy(img)

    nb_subplots = int((tmax - tmin) / step) + 2
    rows = np.ceil(nb_subplots ** 0.5)
    cols = rows
    for i in range(nb_subplots):
        curmin = i * step + tmin
        curmax = min((i * step) + win, tmax)
        mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(curmin, curmax))
        #print(row, col)
        plt.subplot(rows, cols, i+1)
        # hide axes
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        title = '(' + str(curmin) + ', ' + str(curmax) + ')'
        cur_axes.axes.set_title(title)
        plt.imshow(mag_binary, cmap='gray')

    plt.subplot(rows, cols, nb_subplots+1)
    # hide axes
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    title = 'Original'
    cur_axes.axes.set_title(title)
    plt.imshow(img)

    #plt.tight_layout()
    plt.show()

def visualize_params_hls(img, win=150, step=10, tmin=0, tmax=255):
    img = np.copy(img)

    nb_subplots = int((tmax-tmin)/step)+1
    rows = np.ceil(nb_subplots ** 0.5)
    cols = rows
    for i in range(nb_subplots):
        curmin = i*step + tmin
        curmax = min((i*step) + win, tmax)
        hls_binary = hls_select(image, (curmin, curmax))
        #print(row, col)
        plt.subplot(rows, cols, i+1)
        # hide axes
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        title = '(' + str(curmin) + ', ' + str(curmax) + ')'
        cur_axes.axes.set_title(title)
        plt.imshow(hls_binary, cmap='gray')

    plt.subplot(rows, cols, nb_subplots+1)
    # hide axes
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    title = 'Original'
    cur_axes.axes.set_title(title)
    plt.imshow(img)

    plt.tight_layout()
    plt.show()

def visualize_params_sobel(img, win=50, step=10, tmin=0, tmax=100):
    #sobel x best: img, win=20, step=10, tmin=0, tmax=100
    #soble y best: img, win = 50, step = 10, tmin = 0, tmax = 150
    img = np.copy(img)

    nb_subplots = int((tmax - tmin) / step) + 2
    rows = np.ceil(nb_subplots ** 0.5)
    cols = rows
    for i in range(nb_subplots):
        curmin = i * step + tmin
        curmax = min(curmin + win, tmax)
        sxbinary = abs_sobel_thresh(img, 'x', (curmin, curmax))
        # print(row, col)
        plt.subplot(rows, cols, i + 1)
        # hide axes
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        title = '(' + str(curmin) + ', ' + str(curmax) + ')'
        cur_axes.axes.set_title(title)
        plt.imshow(sxbinary, cmap='gray')

    plt.subplot(rows, cols, nb_subplots + 1)
    # hide axes
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    title = 'Original'
    cur_axes.axes.set_title(title)
    plt.imshow(img)

    plt.tight_layout()
    plt.show()


'''
# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
'''

def pipeline(img, ksize=5):
    img = np.copy(img)
    #Get binary mask of broad spectrum of white and yellow pixels
    color_filter_binary = select_white_and_yellow(img)
    #f = plt.figure()
    #imgplot = plt.imshow(color_filter_binary, cmap='gray')
    #plt.show()
    # Sobel x
    sxbinary = abs_sobel_thresh(img, 'x', (20, 255))
    #sxbinary = abs_sobel_thresh(img, 'x', (40, 90))
    s_binary = hls_select(img, (80, 255))
    dir_binary = dir_threshold(img, ksize, (np.pi/10, np.pi - np.pi/10))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(50, 150))
    #plt.imshow(mag_binary)
    #plt.show()

    combined = np.zeros_like(sxbinary)
    combined[((((sxbinary == 1) & (mag_binary == 1)) |
              ((s_binary == 1) & (dir_binary == 1))) &
             (color_filter_binary == 1))] = 1
    #combined[(sxbinary == 1)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary, combined

def process_hist(hist, min_samples=50):
    w = hist.shape[0]
    l_avg = hist[:w/2].mean()
    r_avg = hist[w/2:].mean()
    l_conf = 0 if l_avg ==0 or hist[:w/2].sum < min_samples else hist[:w/2].max()/l_avg
    r_conf = 0 if r_avg ==0 or hist[w/2:].sum < min_samples else hist[w/2:].max()/r_avg

    return hist[:w/2].argmax(), l_conf, hist[w/2:].argmax()+w/2, r_conf
'''
    if confidence < conf_thresh or hist.max() < abs_thresh:
        #Expand search to the whole image
        hist = np.sum(bin_img[:,:], axis=0)
        confidence = hist.max() / havg
        #fig1 = plt.figure()
        #ax1 = fig1.add_subplot(111)
        #ax1.plot(hist)
        #plt.show()

    if confidence > conf_thresh:
        conf_lst.append(confidence)
        lane.append([hist.argmax(), y_val])
'''

def line_len(hough_line):
    import math
    x1, y1, x2, y2 = hough_line
    res = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return res

# Receives an image and ouputs the x coordinates of the longest line
def longest_line(img):
    #edges = cv2.Canny(img*255, 0, 255, apertureSize=3)
    minLineLength = 100
    maxLineGap = 30
    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(111)
    #ax1.imshow(edges, cmap='gray')
    #plt.show()
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    print(lines)
    print(lines.shape)
    lengths = [line_len(line[0]) for line in lines]
    print(lengths)
    longest = lines[np.argmax(lengths)][0]

    print(line_len(longest))
    #for line in lines[0]:
        #cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return min(longest[0], longest[2]), max(longest[0], longest[2])

# From http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x,window_len=100,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def hist_search(bin_img, lane_width_px = 500):
    h = bin_img.shape[0]
    w = bin_img.shape[1]

    l_lane = []
    r_lane = []
    for i in range(h-1, h/2, -1):
        hist = np.sum(bin_img[i-h/2:i], axis=0)
        hist = smooth(hist)
        l_avg = hist[:w/2].mean()
        r_avg = hist[w/2:].mean()
        l_max = hist[:w / 2].argmax()
        r_max = hist[w / 2:].argmax() + w/2
        l_conf = hist[l_max]/l_avg
        r_conf = hist[r_max]/r_avg
        l_lane.append([l_max, i])
        r_lane.append([r_max, i])
    return hist, np.array(l_lane), np.array(r_lane), l_conf, r_conf

#region = [x, w, y, h]
def extract_points(bin_img, region):
    [xr, wr, yr, hr] = region
    #extract the coordinates lane points from the early calculated regions
    points = np.transpose(np.nonzero(bin_img[yr:yr+hr,xr:xr+wr]))
    points[:, 1] += xr
    points[:, 0] += yr
    points[:,[0,1]] = points[:,[1,0]] #reorder to be (x,y) instead of (y,x)

    return np.array(points)

def plot_hists(histograms):
    nb_hists = histograms.shape[0]
    print(nb_hists)
    f, axes = plt.subplots(nb_hists, 1, figsize=(40, 20))
    for i in range(nb_hists):
        axes[i].plot(histograms[i])


def calc_point(y, fit):
    return int((y**2)*fit[0] + y*fit[1] + fit[2])

def visualize_fit(img, fit):
    h = img.shape[0]
    w = img.shape[1]
    yvals = np.arange(0, h, 1)

    xvals = (fit[0] * yvals ** 2 + fit[1] * yvals + fit[2]).astype('int')
    xvals = np.maximum(0, np.minimum(w - 1, xvals))
    img[yvals, xvals] = 1

    return img
'''
def calc_sliced_points(bin_img):
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    nb_slices = 10
    h_slice = h/nb_slices

    points = []
    for i in range(nb_slices):
        start = i*h_slice
        end = (i+1)*h_slice
        hist = np.sum(bin_img[start:end,:], axis=0)
        bestx = hist[:w/2].argmax() + 0

        points_slice = extract_points(bin_img, [bestx-h_slice, h_slice*2, i*h_slice, h_slice])
        points.append(points_slice)
    return points

def exclude_and_stack(points, idx):
    res_points = np.empty((0,2), int)
    for i in range(len(points)):
        if i != idx:
            res_points = np.append(res_points, points[i], axis=0)
    return res_points

def calc_score(bin_img):
    nb_slices = 10
    points = calc_sliced_points(bin_img)
    for i in range(nb_slices):
        points_excluded = exclude_and_stack(points, i)
        fit, conf = fit_polynomial(points_excluded)
        print('conf', conf)
'''

from sklearn import linear_model

def ransac(points, residual_threshold=20, min_samples=20):
    if len(points) < min_samples:
        return np.empty((0,2), int)

    xvals = points[:,0]
    yvals = points[:,1]

    xvals = np.reshape(xvals, (len(xvals), 1))
    yvals = np.reshape(yvals, (len(yvals), 1))

    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=residual_threshold)
    model_ransac.fit(xvals, yvals)
    inlier_mask = model_ransac.inlier_mask_
    #print ('inlier_mask', inlier_mask)
    #outlier_mask = np.logical_not(inlier_mask)
    return points[inlier_mask]
'''
def ransac_slice(bin_img, slice, nb_slices, x_start=None, x_end=None):
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    h_slice = h/nb_slices

    ystart = slice*h_slice
    yend = (slice+1)*h_slice
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = w

    points = extract_points(bin_img, [x_start, x_end-x_start, ystart, yend-ystart])
    points = ransac(points)

    return points
'''

def calc_focus(h, w, fit, slice, nb_slices, slack):
    h_slice = h/nb_slices

    ystart = slice*h_slice
    yend = (slice+1)*h_slice

    if fit is None:
        est_xstart = 0
        est_xend = w
    else:
        x1 = calc_point(ystart, fit)
        x2 = calc_point(yend, fit)
        est_xstart = max(0, min(x1, x2) - slack/2)
        est_xend = min(w-1, max(x1, x2) + slack/2)

    return est_xstart, est_xend

def detect_lane(bin_img, l_prev_fit=None, r_prev_fit=None, slack=200):
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    nb_slices = 10
    h_slice = h/nb_slices

    hist_full = np.sum(bin_img, axis=0)
    hist_full = smooth(hist_full)

    if l_prev_fit is None:
        #use ransac linear model to estimate line
        l_points_full = ransac(extract_points(bin_img, [0, w/2, 0, h]))
        l_prev_fit,_ = fit_polynomial(l_points_full, linear=True)

    if r_prev_fit is None:
        # use ransac linear model to estimate line
        r_points_full = ransac(extract_points(bin_img, [w/2, w/2, 0, h]))
        r_prev_fit,_ = fit_polynomial(r_points_full, linear=True)

    l_points = np.empty((0,2), int)
    r_points = np.empty((0,2), int)
    for i in range(nb_slices):
        ystart = i * h_slice
        yend = (i + 1) * h_slice
        l_xstart, l_xend = calc_focus(h, w, l_prev_fit, i, nb_slices, slack=slack)
        r_xstart, r_xend = calc_focus(h, w, r_prev_fit, i, nb_slices, slack=slack)

        l_points_slice = extract_points(bin_img, [l_xstart, l_xend - l_xstart, ystart, yend - ystart])
        r_points_slice = extract_points(bin_img, [r_xstart, r_xend - r_xstart, ystart, yend - ystart])

        l_points = np.append(l_points, l_points_slice, axis=0)
        r_points = np.append(r_points, r_points_slice, axis=0)

    return hist_full, l_points, r_points


#@profile
def detect_lane_old(bin_img, win=100, lx = 0, rx = 0, lane_width_px = 500, conf_thresh = 3, density_thresh=1000):
    #return hist_search(bin_img, line_dist_px)
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    print('lx, rx', lx, rx)

    hist = np.sum(bin_img, axis=0)
    hist = smooth(hist)
    l_avg = hist[:w/2].mean()
    r_avg = hist[w/2:].mean()
    l_max = hist[:w / 2].argmax()
    r_max = hist[w / 2:].argmax() + w/2
    l_conf = hist[l_max]/l_avg
    r_conf = hist[r_max]/r_avg
    if math.isnan(l_conf):
        l_conf = 0
    if math.isnan(r_conf):
        r_conf = 0

    '''
    #infer on the second lane from the lane with higher confidence
    if l_conf > r_conf:
        lx_start = l_max
        region_start = lx_start + line_dist_px - win/2
        region_end = region_start + win
        rx_start = hist[region_start:region_end].argmax() + region_start
    else:
        rx_start = r_max
        region_start = rx_start - line_dist_px - win/2
        region_end = region_start + win
        lx_start = hist[region_start:region_end].argmax() + region_start
    '''
    lx_start = l_max
    rx_start = r_max
    print('start', lx_start, rx_start)
    lx_end = lx_start
    # find the base of the peak from left
    while lx_start > 0 and hist[lx_start] > hist[lx_start-win:lx_start].mean():
        lx_start -= 1
    # find the base of the peak from right
    while hist[lx_end] > hist[lx_end:lx_end+win].mean():
        lx_end += 1

    rx_end = rx_start
    #find the base of the peak from left
    while hist[rx_start] > hist[rx_start-win:rx_start].mean():
        rx_start -= 1
    #find the base of the peak from right
    while rx_end < w-1 and hist[rx_end] > hist[rx_end:rx_end+win].mean():
        rx_end += 1

    print('confidence: ', int(l_conf), int(r_conf))
    print('search range: ', lx_start, lx_end, rx_start, rx_end)

    '''
    #This version below performs x10 times slower than the uncommmented version using np.nonzero
    l_lane = []
    r_lane = []
    for y in range(h):
        for x in range(lx_start, lx_end):
            if bin_img[y,x] > 0:
                l_lane.append([x,y,20])
        for x in range(rx_start, rx_end):
            if bin_img[y,x] > 0:
                r_lane.append([x,y,20])
    '''

    #extract the coordinates lane points from the early calculated regions
    l_lane = np.transpose(np.nonzero(bin_img[:,lx_start:lx_end]))
    l_lane[:,1] += lx_start
    l_lane[:,[0,1]] = l_lane[:,[1,0]] #reorder to be (x,y) instead of (y,x)
    r_lane = np.transpose(np.nonzero(bin_img[:,rx_start:rx_end]))
    r_lane[:,1] += rx_start
    r_lane[:,[0,1]] = r_lane[:,[1,0]] #reorder to be (x,y) instead of (y,x)
    #print(lx_start, lx_end, rx_start, rx_end)

    #print(len(l_lane), len(r_lane))
    if len(l_lane) < density_thresh:
        l_conf = 0
    if len(r_lane) < density_thresh:
        r_conf = 0

    '''
    fit = fit_polynomial(l_lane, 1)
    x0 = fit[1] #prediction of x where y=0
    upper_layer = np.stack([np.arange(x0-win/2, x0+win/2), np.zeros(win)], axis=-1)
    l_lane = np.concatenate((l_lane, upper_layer)).astype('int')
    upper_layer = np.stack([np.arange(rx_start, rx_end), np.zeros(rx_end-rx_start)], axis=-1)
    r_lane = np.concatenate((r_lane, upper_layer)).astype('int')
    print(l_lane, r_lane)
    '''
    #create gravity points on the edge for polyfit
    if lx > 0:
        np.append(l_lane, [lx, h-1])
    if rx > 0:
        np.append(r_lane, [h - 1, rx])

    return hist, l_lane, r_lane, l_conf, r_conf

import robust

def fit_polynomial(points, linear=False):

    if len(points) == 0:
        return None, 0

    xvals = points[:,0]
    yvals = points[:,1]

    deg = 2
    if linear:
        deg = 1
    #n = np.arange(len(yvals))
    #line_fit, res, _, _, _ = np.polyfit(yvals, xvals, 2, w=np.sqrt(n), full=True)
    fit, res, _, _, _ = np.polyfit(yvals, xvals, deg, full=True)
    if len(res) > 0:
        confidence = math.sqrt(len(xvals)) / res
    else:
        confidence = 0

    if linear:
        fit = np.array([0, fit[0], fit[1]])

    return fit, confidence

def curvature_px(left_fit, right_fit, yvals):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(yvals)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                                 /np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                    /np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1163.9    1213.7
    return left_curverad, right_curverad

#calculate curvature of lane in meters
def curvature_m(xvals, yvals, straight_thresh=0.1**7):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30. / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension

    y_eval = np.max(yvals)
    fit = np.polyfit(yvals * ym_per_pix, xvals * xm_per_pix, 2)

    curv = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * fit[0])
    return curv
    # Example values: 3380.7 m    3189.3 m

def unwarp(undist, warped, Minv, left_fitx, right_fitx, yvals):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

#visualize_params_mag(image) #50, 150
#visualize_params_dir(image) #1.13, 1.23
#visualize_params_sobel(image) # y: 50, 100, x: 40, 90
#visualize_params_hls(image)#110, 255

#cal_fname = 'calibration_pickle.p'
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
#dist_pickle = {}
#dist_pickle["mtx"] = mtx
#dist_pickle["dist"] = dist
#pickle.dump(dist_pickle, open(cal_fname, "wb"))
# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=30)
#ax2.imshow(dst)
#ax2.set_title('Undistorted Image', fontsize=30)

# Read in the saved objpoints and imgpoints
#dist_pickle = pickle.load( open(cal_fname, "rb" ) )
#objpoints = dist_pickle["objpoints"]
#imgpoints = dist_pickle["imgpoints"]
