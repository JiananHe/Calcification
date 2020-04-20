import numpy as np
import cv2
import matplotlib.pyplot as plt

a = np.array([[[10,10], [100,10], [100,100], [10,100]]], dtype = np.int32)
b = np.array([[[100,100], [200,230], [150,200], [100,220]]], dtype = np.int32)
print(a.shape)

im = np.zeros([240, 320], dtype = np.uint8)
cv2.polylines(im, a, 1, 255)
cv2.fillPoly(im, b, 255)

dilate_rate = 5
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dilate_rate, dilate_rate))
dilate = cv2.dilate(im, kernel, iterations=1)

cv2.imshow("test", im)
cv2.imshow("dilate", dilate)
cv2.waitKey(0)