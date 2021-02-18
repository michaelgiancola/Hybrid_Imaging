#Author: Michael Giancola
#CS4442 Assignment 3 Question 3
#Date: 18 March 2020

import imageio
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#this method is what creates the gaussian filter
#it takes in two parameters which are the size of the kernel along with the sigma value
def makeGaussianFilter(size, sigma):
    aa = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)   #finds the center of the kernel
    x, y = np.meshgrid(aa, aa)
    gaussKernel = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sigma)) #gaussian filter implementation
    return gaussKernel / np.sum(gaussKernel)    #returns the filter kernel matrix and normalizes weights to sum to 1


#this method uses the scipy convolve2d to convolve the image and kernel matricies passed through as parameters
def convolve(image, kernel):
    return signal.convolve2d(image, kernel, boundary='symm', mode='same')


# scale image's intensity to [0,1] with mean value of 0.5 for better visualization.\n",
def intensityscale(raw_img):
    # scale an image's intensity from [min, max] to [0, 1].\n,
    v_min, v_max = raw_img.min(), raw_img.max()
    scaled_im = (raw_img * 1.0 - v_min) / (v_max - v_min)
    # keep the mean to be 0.5.\n",
    meangray = np.mean(scaled_im)
    scaled_im = scaled_im - meangray + 0.5
    # clip to [0, 1]
    scaled_im = np.clip(scaled_im, 0, 1)
    return scaled_im


#this is the main function
if __name__ == "__main__":
    image = imageio.imread("einsteinandwho.png", as_gray=True)

    fig = plt.figure()

    gaussianKernel = makeGaussianFilter(35, 10)        #creates the gaussian filter with size 35 and sigma value of 10
    lowFreqImage = convolve(image,gaussianKernel)      #uses the construction equation to compute the low freq image
    highFreqImage = image - lowFreqImage               #uses the construction equation to compute the high freq image

    #below plots the original image, the low freq image, and the high freq image respectively
    plt.subplot(1,3,1)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.imshow(image, cmap='gray')

    plt.subplot(1,3,2)
    plt.title('Low Spatial Frequency Range'), plt.xticks([]), plt.yticks([])
    lowFreqImage = intensityscale(lowFreqImage)
    plt.imshow(lowFreqImage, cmap='gray')

    plt.subplot(1,3,3)
    plt.title('High Spatial Frequency Range'), plt.xticks([]), plt.yticks([])
    highFreqImage = intensityscale(highFreqImage)
    plt.imshow(highFreqImage, cmap='gray')

    plt.show(block=True)


