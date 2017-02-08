import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

from scipy import signal
xs = np.arange(0, np.pi, 0.05)
data = np.sin(xs)
peakind = signal.find_peaks_cwt(data, np.arange(1,10))
print(peakind, xs[peakind], data[peakind])
exit(0)

n_samples = 1000
n_outliers = 50


X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)

# Add outlier data
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# Fit line using all data
model = linear_model.LinearRegression()
model.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(X, y)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(-5, 5)
line_y = model.predict(line_X[:, np.newaxis])
line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])

# Compare estimated coefficients
print("Estimated coefficients (true, normal, RANSAC):")
print(coef, model.coef_, model_ransac.estimator_.coef_)

lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(line_X, line_y, color='navy', linestyle='-', linewidth=lw,
         label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linestyle='-',
         linewidth=lw, label='RANSAC regressor')
plt.legend(loc='lower right')
plt.show()

exit(0)



import pickle
from pypeaks import Data, Intervals

[x, y] = pickle.load(file('examples/sample-histogram.pickle'))
data_obj = Data(x, y, smoothness=11)

#Peaks by slope method
data_obj.get_peaks(method='slope')
#print data_obj.peaks
data_obj.plot()

#Peaks by interval method
ji_intervals = pickle.load('examples/ji_intervals.pickle')
ji_intervals = Intervals(ji_intervals)
data_obj.get_peaks(method='interval', intervals=ji_intervals)
#print data_obj.peaks
data_obj.plot(intervals=ji_intervals)

#Read the help on Data object, and everything else is explained there.
help(Data)


def calc_histograms(bin_img, win=200):
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    nb_slices = 10
    h_slice = h/nb_slices

    hist_full = np.sum(bin_img, axis=0)
    hist_full = smooth(hist_full)
    #l_max_full = hist_full[:w / 2].argmax()
    #r_max_full = hist_full[w / 2:].argmax() + w / 2

    hists = []
    l_points = np.empty((0,2), int)
    r_points = np.empty((0,2), int)
    l_maxes = []
    r_maxes = []
    for i in range(nb_slices):
        start = i*h_slice
        end = (i+1)*h_slice
        hist = np.sum(bin_img[start:end,:], axis=0)
        hists.append(hist)
        l_max = hist[:w/2].argmax() + 0
        r_max = hist[w/2:].argmax() + w/2
        l_maxes.append(l_max)
        r_maxes.append(r_max)

    l_points_slice = extract_points(bin_img, [l_max-h_slice/2, h_slice, i*h_slice, h_slice])
    r_points_slice = extract_points(bin_img, [r_max-h_slice/2, h_slice, i*h_slice, h_slice])
    l_points = np.append(l_points, l_points_slice, axis=0)
    r_points = np.append(r_points, r_points_slice, axis=0)

    l_maxes = np.array(l_maxes)
    l_avgx = l_maxes.mean()

    hists = np.array(hists)
    l_confs = np.array(l_confs)
    r_confs = np.array(r_confs)

    #plot_hists(hists)
    #plt.show()
    return hist_full, l_points, r_points, l_confs.mean(), r_confs.mean()


def calc_histograms(bin_img, win=400):
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    nb_slices = 10
    h_slice = h/nb_slices

    hist_full = np.sum(bin_img, axis=0)
    hist_full = smooth(hist_full)
    l_max_full = hist_full[:w / 2].argmax()
    r_max_full = hist_full[w / 2:].argmax() + w / 2

    l_points = extract_points(bin_img, [l_max_full - win / 2, win, 0, h])
    r_points = extract_points(bin_img, [r_max_full - win / 2, win, 0, h])

    return hist_full, l_points, r_points, 20, 20

    l_fit = fit_polynomial(l_points, linear=True)
    r_fit = fit_polynomial(r_points, linear=True)
    print('r_fit', r_fit)

    hists = []
    l_confs = []
    r_confs = []
    l_points = np.empty((0,2), int)
    r_points = np.empty((0,2), int)
    for i in range(nb_slices):
        start = i*h_slice
        end = (i+1)*h_slice
        hist = np.sum(bin_img[start:end,:], axis=0)
        hists.append(hist)
        l_ctr_x = int(l_fit[1]*(start+end)/2 + l_fit[2])
        r_ctr_x = int(r_fit[1]*(start+end)/2 + r_fit[2])
        l_max = hist[l_ctr_x-win/2:l_ctr_x+win/2].argmax() + l_ctr_x-win/2
        r_max = hist[r_ctr_x-win/2:r_ctr_x+win/2].argmax() + r_ctr_x-win/2
        print(l_max, r_max)
        l_conf = hist[l_max]
        r_conf = hist[r_max]
        l_confs.append(l_conf)
        r_confs.append(r_conf)
        l_points_slice = extract_points(bin_img, [l_max-h_slice/2, h_slice, i*h_slice, h_slice])
        r_points_slice = extract_points(bin_img, [r_max-h_slice/2, h_slice, i*h_slice, h_slice])
        l_points = np.append(l_points, l_points_slice, axis=0)
        r_points = np.append(r_points, r_points_slice, axis=0)


    hists = np.array(hists)
    l_confs = np.array(l_confs)
    r_confs = np.array(r_confs)

    #plot_hists(hists)
    #plt.show()
    return hist_full, l_points, r_points, l_confs.mean(), r_confs.mean()


def calc_histograms(bin_img, max_dev=100, prev_fit=None):
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    nb_slices = 10
    h_slice = h/nb_slices

    hist_full = np.sum(bin_img, axis=0)
    hist_full = smooth(hist_full)
    l_max_full = hist_full[:w / 2].argmax()
    r_max_full = hist_full[w / 2:].argmax() + w / 2

    from scipy import signal
    peakind = signal.find_peaks_cwt(hist_full, np.arange(1, 10))
    print(peakind, hist_full[peakind])

    hists = []
    l_points = np.empty((0,2), int)
    r_points = np.empty((0,2), int)
    for i in range(nb_slices):
        start = i*h_slice
        end = (i+1)*h_slice
        hist = np.sum(bin_img[start:end,:], axis=0)
        hists.append(hist)
        l_max = hist[:w/2].argmax() + 0
        r_max = hist[w/2:].argmax() + w/2

        l_points_slice = extract_points(bin_img, [l_max-h_slice/2, h_slice, i*h_slice, h_slice])
        r_points_slice = extract_points(bin_img, [r_max-h_slice/2, h_slice, i*h_slice, h_slice])
        #fit_polynomial(l_points)
        #fit_polynomial(r_points)

        from scipy import signal
        peakind = signal.find_peaks_cwt(hist, np.arange(1, 50))
        print(peakind, hist[peakind])

        #        l_points_slice = np.array([[l_max, (start+end)/2]])
#        r_points_slice = np.array([[r_max, (start+end)/2]])
        if abs(l_max - l_max_full) < max_dev:
            l_points = np.append(l_points, l_points_slice, axis=0)
        if abs(r_max - r_max_full) < max_dev:
            r_points = np.append(r_points, r_points_slice, axis=0)

    #print(r_points)
    return hist_full, l_points, r_points, 20, 20

def detect_lane_new(bin_img, max_dev=100, est_fit=None, slack=300):
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    nb_slices = 10
    h_slice = h/nb_slices

    #calc_score(bin_img[:w/2])
    #calc_score(bin_img[w/2:])

    hist_full = np.sum(bin_img, axis=0)
    hist_full = smooth(hist_full)
    l_max_full = hist_full[:w / 2].argmax()
    r_max_full = hist_full[w / 2:].argmax() + w / 2

    hists = []
    l_peaks = []
    r_peaks = []
    l_conf_arr = []
    r_conf_arr = []
    l_fits = []
    r_fits = []
    for i in range(nb_slices):
        start = i*h_slice
        end = (i+1)*h_slice
        hist = np.sum(bin_img[start:end,:], axis=0)
        hists.append(hist)
        l_max = hist[:w/2].argmax() + 0
        r_max = hist[w/2:].argmax() + w/2

        l_points_slice = extract_points(bin_img, [l_max-h_slice, h_slice*2, i*h_slice, h_slice])
        r_points_slice = extract_points(bin_img, [r_max-h_slice, h_slice*2, i*h_slice, h_slice])

        l_fit, l_conf = fit_polynomial(l_points_slice, linear=True)
        if l_fit is None:
            l_fit, l_conf = [np.array([0, 0, 0]), 0]

        r_fit, r_conf = fit_polynomial(r_points_slice, linear=True)
        if r_fit is None:
            r_fit, r_conf = [np.array([0, 0, 0]), 0]

        l_conf_arr.append(l_conf)
        r_conf_arr.append(r_conf)
        l_fits.append(l_fit)
        r_fits.append(r_fit)


        from scipy import signal
        peakind = signal.find_peaks_cwt(hist, np.arange(1, 50))
        l_peaks_slice = [x for x in peakind if x < w/2]
        r_peaks_slice = [x for x in peakind if x > w / 2]

        l_peaks.append(l_peaks_slice)
        r_peaks.append(r_peaks_slice)
        #print(peakind, hist[peakind])


    l_conf_arr = np.array(l_conf_arr)
    l_best_slice = l_conf_arr.argmax()
    l_best_fit = l_fits[l_best_slice]
    l_points = np.empty((0,2), int)
    for i in range(nb_slices):
        ystart = i*h_slice
        yend = (i+1)*h_slice
        x1 = calc_point(ystart, l_best_fit)
        x2 = calc_point(yend, l_best_fit)
        est_xstart = max(0, min(x1, x2))
        est_xend = min(w-1, max(x1, x2))
        l_points_slice = extract_points(bin_img, [min(est_xstart, est_xend)-slack/2,
                                                  abs(est_xstart-est_xend) + slack,
                                                  i * h_slice,
                                                  h_slice])
        l_points = np.append(l_points, l_points_slice, axis=0)

    print(r_conf_arr)
    r_conf_arr = np.array(r_conf_arr)
    r_best_slice = r_conf_arr.argmax()
    r_best_fit = r_fits[r_best_slice]
    img = np.zeros_like(bin_img)
    visualize_fit(img, r_best_fit)
    plt.imshow(img)

    r_points = np.empty((0, 2), int)
    for i in range(nb_slices):
        ystart = i * h_slice
        yend = (i + 1) * h_slice
        x1 = calc_point(ystart, r_best_fit)
        x2 = calc_point(yend, r_best_fit)
        est_xstart = max(0, min(x1, x2))
        est_xend = min(w-1, max(x1, x2))

        r_points_slice = extract_points(bin_img, [min(est_xstart, est_xend) - slack / 2,
                                                  abs(est_xstart - est_xend) + slack,
                                                  i * h_slice,
                                                  h_slice])
        r_points = np.append(r_points, r_points_slice, axis=0)

    return hist_full, l_points, r_points, 20, 20
