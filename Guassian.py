import cv2
import numpy as np
from scipy import ndimage

filename = 'lena.jpg'
img = cv2.imread(filename)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('lena_gray.jpg', img_gray)

kernel_3_3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])
kernel_5_5 = np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
])

kernel_3_3 = kernel_3_3 / kernel_3_3.sum()
kernel_5_5 = kernel_5_5 / kernel_5_5.sum()

img = cv2.imread(filename, 0)
k3 = ndimage.convolve(img, kernel_3_3)
k5 = ndimage.convolve(img, kernel_5_5)

cv2.imshow('3x3', k3)
cv2.imshow('5x5', k5)
cv2.waitKey()