import cv2
import os
import numpy as np
from skimage import data,exposure
import matplotlib.pyplot as plt


def histogram_equalize(img):
    plt.subplot(121)
    plt.hist(img.flatten(), bins=256, density=1, edgecolor='None', facecolor='red')
    he_image = exposure.equalize_hist(image)

    plt.subplot(122)
    plt.hist(he_image.flatten(), bins=256, density=1, edgecolor='None', facecolor='red')

    plt.show()
    return he_image


if __name__ == "__main__":
    root_dir = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\ROIs"
    benign_dir = os.path.join(root_dir, "benign")
    malignant_dir = os.path.join(root_dir, "malignant")

    for file in os.listdir(benign_dir):
        if file.split(".")[-1] == "png":
            file_path = os.path.join(benign_dir, file)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            # he_image = histogram_equalize(image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 15
            cv2.imshow("test", (image - np.min(image)) / (np.max(image) - np.min(image)))
            # cv2.imshow("he", he_image)
            cv2.waitKey(0)

            print(image.shape)
