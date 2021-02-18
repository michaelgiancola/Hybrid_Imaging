#Author: Michael Giancola
#Created a hybrid image using the michael.jpg and mrbean.jpg
#Used the equation out = blure(B) + (A-blur(A)) and wrote the blur function which low-pass filters an image
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


#This is the method that creates box filter which takes in a size of kernel parameter
def makeBoxFilter(size):
    bKernel = np.ones((size,size), dtype=float)
    return (1/float(size**2))*bKernel    #returns the filter kernel matrix


#this method uses the scipy convolve2d to convolve the image and kernel matricies passed through as parameters
def convolve(image, kernel):
    return signal.convolve2d(image, kernel, boundary='symm', mode='same')


#this method creates the hybrid image by applying the blur function to both images and using the construction equation
def hybrid(imageA, imageB, filter):
    filteredImageA = convolve(imageA, filter)
    filteredImageB = convolve(imageB, filter)
    return filteredImageB + (imageA - filteredImageA)


#this is the main function
if __name__ == "__main__":
    highFreqImage = imageio.imread("michael.jpg", as_gray=True) #reads in image A
    lowFreqImage = imageio.imread("mrbean.jpg", as_gray=True)   #reads in image B

    fig = plt.figure()

    gaussianKernel = makeGaussianFilter(11, 5)      #creates the gaussian filter with size 11 and sigma value of 5
    hybrid1 = hybrid(highFreqImage, lowFreqImage, gaussianKernel)   #creates the first hybrid picture
    plt.subplot(1,2,1)
    plt.title('Hybrid Image using Gaussian'), plt.xticks([]), plt.yticks([])
    plt.imshow(hybrid1, cmap='gray')

    boxKernel = makeBoxFilter(9)    #creates the box filter with size 9
    hybrid2 = hybrid(highFreqImage, lowFreqImage, boxKernel)    #creates the second hybrid picture using the box filter
    plt.subplot(1,2,2)
    plt.title('Hybrid Image using Box'), plt.xticks([]), plt.yticks([])
    plt.imshow(hybrid2, cmap='gray')
    plt.show(block=True)




