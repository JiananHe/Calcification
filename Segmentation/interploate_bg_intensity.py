import cv2
import numpy as np
import skimage.morphology as morphology
import skimage.measure as measure
import os

min_cal_area = 3
low_density_thresh = 1000  # Calcification would not appear in low-density areas with a intensity below this value


def bilinear_interpolate(P, H, W):
    """
    bi-linear interpolate for given p0, p1, p2, p3 in a W*H area
    :param P: [p0, p1, p2, p3]^T
    :param W: width
    :param H: height
    :return: a interpolated map with shape (H, W)
    """
    weight = np.array([[(1 - a) * (1 - b), b * (1 - a), (1 - b) * a, a * b] for a in np.arange(0, 1, 1 / H) for b in
                       np.arange(0, 1, 1 / W)])
    interpolated_intensity = np.matmul(weight, P).reshape(H, W)
    return interpolated_intensity


def region_median(img, coord, radius):
    """
    get median intensity in the neighbour of coord
    :param img: raw image
    :param coord: coordinate
    :param radius: radius of neighbour
    :return: median intensity
    """
    shape = img.shape
    nb_left = 0 if coord[1] - radius < 0 else coord[1] - radius
    nb_top = 0 if coord[0] - radius < 0 else coord[0] - radius
    nb_right = shape[1] if coord[1] + radius > shape[1] else coord[1] + radius
    nb_bottom = shape[0] if coord[0] + radius > shape[0] else coord[0] + radius
    neighbour = img[nb_top:nb_bottom, nb_left:nb_right]
    assert len(neighbour) != 0
    return np.median(neighbour)


def background_intensity(img, zero_start=True, local_size=15):
    """
    calculate background intensity for every local area with bi-linear interpolate
    :param img: raw image
    :param zero_start: iterate from left top corner or half local area
    :param local_size: size of local area
    :return: background intensity
    """
    shape = img.shape
    bg_intensity = np.zeros(shape)
    start_point = local_size if zero_start else local_size // 2
    for i in [0] + list(range(start_point, shape[0], local_size)):
        for j in [0] + list(range(start_point, shape[1], local_size)):
            # coordinates of corners of local area
            lp = j
            tp = i
            rp = lp + local_size if lp + local_size < shape[1] else shape[1]
            bp = tp + local_size if tp + local_size < shape[0] else shape[0]

            # get intensity of corners of local area with median pooling
            min_radius = 3
            max_radius = 5
            local_area_intensity = np.zeros((bp - tp, rp - lp))
            for r in range(min_radius, max_radius + 1):
                p0 = region_median(img, (tp, lp), r)
                p1 = region_median(img, (tp, rp), r)
                p2 = region_median(img, (bp, lp), r)
                p3 = region_median(img, (bp, rp), r)
                local_area_intensity += bilinear_interpolate(np.array([[p0], [p1], [p2], [p3]]), bp - tp, rp - lp)

            bg_intensity[tp:bp, lp:rp] = local_area_intensity / (max_radius - min_radius + 1)

    # cv2.imshow("bg", (bg_intensity - np.min(bg_intensity)) / (np.max(bg_intensity) - np.min(bg_intensity)))
    return bg_intensity


def morphology_filter(img, diff_img, show=True, winname=None):
    """
    filter regions with morphology operations
    :param img: raw image
    :param diff_img: different image between raw image and background intensity
    :param show: whether to show images
    :return: image
    """
    # omit negative
    diff_img[diff_img < 0] = 0

    # only preserve the 5% highest positive value
    pos_num = np.sum(diff_img > 0)
    t = sorted(diff_img.flat)[-int(0.05 * pos_num)]
    diff_img[diff_img < t] = 0
    if show:
        cv2.imshow("diff-%s" % winname, (diff_img - np.min(diff_img)) / (np.max(diff_img) - np.min(diff_img)))

    # filter small, holey and low-intensity components, filter components located in low density region
    diff_img_binary = np.where(diff_img > 0, 1, diff_img).astype(np.uint8)
    label_img, num_components = measure.label(diff_img_binary, return_num=True, connectivity=2)
    properties = measure.regionprops(label_img)
    min_cal_thresh = 0.5 * np.max(img[diff_img_binary != 0])
    for i in range(1, num_components + 1):
        area = properties[i - 1]["area"]
        filled_area = properties[i - 1]["filled_area"]
        centre_point = np.array(properties[i - 1]["centroid"]).astype(np.int)
        max_intensity = np.max(img[label_img == i])

        if area <= min_cal_area or area != filled_area or \
                (max_intensity < min_cal_thresh and region_median(img, centre_point, 25) < low_density_thresh):
            diff_img_binary = np.where(label_img == i, 0, diff_img_binary)
            diff_img = np.where(label_img == i, 0, diff_img)

    # filter the remaining areas of lower intensity
    diffs = diff_img[diff_img != 0].astype(np.int)
    diffs_set_sorted = sorted(list(set(diffs.flat)))
    thresh_intensity_id = 1
    while len(diffs) != 0 and np.median(diffs) < np.max(diffs) * 0.4:
        diff_img = np.where(diff_img < diffs_set_sorted[thresh_intensity_id], 0, diff_img)
        diff_img_binary = np.where(diff_img < diffs_set_sorted[thresh_intensity_id], 0, diff_img_binary)
        thresh_intensity_id += 1
        diffs = diff_img[diff_img != 0].astype(np.int)

    # if a region in label_img is removed in diff_img, then remove this region in diff_img_binary
    for i in range(1, num_components + 1):
        if (diff_img[label_img == i] == 0).all():
            diff_img_binary = np.where(label_img == i, 0, diff_img_binary)

    # diff_img_binary = morphology.binary_erosion(diff_img_binary)
    diff_img_binary = morphology.remove_small_objects(diff_img_binary.astype(np.bool), 1, connectivity=2) + 0
    # diff_img_binary = morphology.binary_dilation(diff_img_binary)

    if show:
        cv2.imshow("removed-%s" % winname, diff_img_binary + .0)

    return diff_img_binary


def segment_main(img_path, show=False):
    """
    main function of segmentation
    :param img_path: path of raw image
    :param show: whether to show images
    :return: segmentation mask (np.uint8)
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # get background intensity (twice to enhance robust)
    bg_intensity = background_intensity(img, zero_start=True, local_size=15)
    bg_intensity_1 = background_intensity(img, zero_start=False, local_size=15)

    # get segmentation based on different image
    seg0 = morphology_filter(img, img - bg_intensity, show, winname="0")
    seg1 = morphology_filter(img, img - bg_intensity_1, show, winname="1")

    # get the final segmentation
    segmentation = seg0 + seg1
    segmentation = np.where(segmentation > 0, 255, segmentation).astype(np.uint8)

    # show segmentation overlay
    if show:
        cv2.imshow("image", (img - np.min(img)) / (np.max(img) - np.min(img)))
        cv2.imshow("segmentation", segmentation)
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255 + 0.5).astype(np.uint8)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay_r = np.where(segmentation == 255, 255, overlay[:, :, 0])
        overlay_g = np.where(segmentation == 255, 0, overlay[:, :, 1])
        overlay_b = np.where(segmentation == 255, 0, overlay[:, :, 2])
        overlay = np.stack((overlay_r, overlay_g, overlay_b), axis=2)
        cv2.imshow("overlay", overlay)
        cv2.waitKey(0)

    return segmentation


if __name__ == "__main__":
    rois_path = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\ROIs"
    benign_rois_path = os.path.join(rois_path, "benign")
    malignant_rois_path = os.path.join(rois_path, "malignant")
    seg_save_path = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\ROIs_seg"
    benign_seg_save_path = os.path.join(seg_save_path, "benign")
    malignant_seg_save_path = os.path.join(seg_save_path, "malignant")

    segment_main(os.path.join(benign_rois_path, "B_12RMLO.png"), True)

    # for benign_case in os.listdir(benign_rois_path):
    #     if benign_case[-3:] != "png" or os.path.exists(os.path.join(benign_seg_save_path, benign_case)):
    #         continue
    #     print("benign %s" % benign_case)
    #     segmentation = segment_main(os.path.join(benign_rois_path, benign_case), False)
    #     cv2.imwrite(os.path.join(benign_seg_save_path, benign_case), segmentation)
    #
    # for malignant_case in os.listdir(malignant_rois_path):
    #     if malignant_case[-3:] != "png" or os.path.exists(os.path.join(malignant_seg_save_path, malignant_case)):
    #         continue
    #     print("malignant %s" % malignant_case)
    #     segmentation = segment_main(os.path.join(malignant_rois_path, malignant_case), False)
    #     cv2.imwrite(os.path.join(malignant_seg_save_path, malignant_case), segmentation)
