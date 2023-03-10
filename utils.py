import numpy as np
import os
import errno
import matplotlib.pyplot as plt
import scipy.io as spio
from skimage.feature import plot_matches
from skimage.util import montage
from skimage.io import imread_collection
from skimage.io import concatenate_images
import glob
import re
import cv2



def evaluateLabels(y, ypred, visualize=True):
    classLabels = np.unique(y)
    conf = np.zeros((len(classLabels), len(classLabels)))
    for tc in range(len(classLabels)):
        for pc in range(len(classLabels)):
            conf[tc, pc] = np.sum(np.logical_and(y==classLabels[tc],
                ypred==classLabels[pc]).astype(float))

    acc = np.sum(np.diag(conf))/y.shape[0]

    if visualize:
        plt.figure()
        plt.imshow(conf, cmap='gray')
        plt.ylabel('true labels')
        plt.xlabel('predicted labels')
        plt.title('Confusion matrix (Accuracy={:.2f})'.format(acc*100))
        plt.show()

    return (acc, conf)


# Show matching between a pair of images
def showMatches(im1, im2, c1, c2, matches, title=""):
    disp_matches = np.array([np.arange(matches.shape[0]), matches]).T.astype(int)
    valid_matches = np.where(matches>=0)[0]
    disp_matches = disp_matches[valid_matches, :]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_matches(ax, im1, im2,
            c1[[1, 0], :].astype(int).T, c2[[1,0], :].astype(int).T, disp_matches)
    ax.set_title(title)


#Thanks to mergen from https://stackoverflow.com/questions/7008608
def todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict

def load_images(folder):
    images = []
    image_labels = []

    for filename in glob.glob(folder + '/*.jpg'):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
            image_labels.append(re.sub('_[^_]*$', '', re.split('/', filename)[-1]))

    return images, image_labels


# def imread(path):
#     img = plt.imread(path).astype(float)
#     #Remove alpha channel if it exists
#     if img.ndim > 2 and img.shape[2] == 4:
#         img = img[:, :, 0:3]
#     #Puts images values in range [0,1]
#     if img.max() > 1.0:
#         img /= 255.0
#
#     return img


# def mkdir(dirpath):
#     if not os.path.exists(dirpath):
#         try:
#             os.makedirs(directory)
#         except OSError as e:
#             if e.errno != errno.EEXIST:
#                 raise
#     else:
#         print("Directory {} already exists.".format(dirpath))


# Thanks to ali_m from https://stackoverflow.com/questions/17190649
# def gaussian(hsize=3,sigma=0.5):
#     """
#     2D gaussian mask - should give the same result as MATLAB's
#     fspecial('gaussian',[shape],[sigma])
#     """
#     shape = (hsize, hsize)
#     m,n = [(ss-1.)/2. for ss in shape]
#     y,x = np.ogrid[-m:m+1,-n:n+1]
#     h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
#     h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
#     sumh = h.sum()
#     if sumh != 0:
#         h /= sumh
#     return h
