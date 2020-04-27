import SimpleITK as sitk
import numpy as np


def read_dicom(path):
    image = sitk.ReadImage(path)
    print(image.GetSpacing())
    array = sitk.GetArrayFromImage(image)
    print(np.min(array))
    print(np.max(array))
    return array


if __name__ == "__main__":
    read_dicom(r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\raw_data\benign\CASE1\DICOM\19031313\29060000\41885749")

