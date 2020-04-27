import cv2
import numpy as np
import skimage.morphology as morphology
import skimage.measure as measure


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


def median_pool(img, coord, radius):
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
    return np.median(neighbour)


def background_intensity(img, zero_start=True, local_size=15):
    """
    calculate background intensity for every local area with bi-linear interpolate
    :param img: raw gray image
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
            max_radius = 5
            local_area_intensity = np.zeros((bp - tp, rp - lp))
            for r in range(1, max_radius + 1):
                p0 = median_pool(img, (tp, lp), r)
                p1 = median_pool(img, (tp, rp), r)
                p2 = median_pool(img, (bp, lp), r)
                p3 = median_pool(img, (bp, rp), r)
                local_area_intensity += bilinear_interpolate(np.array([[p0], [p1], [p2], [p3]]), bp - tp, rp - lp)

            bg_intensity[tp:bp, lp:rp] = local_area_intensity / max_radius

    # cv2.imshow("bg", (bg_intensity - np.min(bg_intensity)) / (np.max(bg_intensity) - np.min(bg_intensity)))
    return bg_intensity


def remove_tiny_huge_area(img, min_thresh, max_thresh):
    """
    remove area that less than min_thresh or bigger than max_thresh
    :param img: mask image
    :param min_thresh: minimum area threshold
    :param max_thresh: maximum area threshold
    :return: mak image
    """
    label_img, num_components = measure.label(img, return_num=True, connectivity=2)
    properties = measure.regionprops(label_img)
    assert len(properties) == num_components
    # filter components according to areas
    for i in range(1, num_components + 1):
        area = properties[i - 1]["area"]
        filled_area = properties[i - 1]["filled_area"]  # filter the area with holes
        if area <= min_thresh or area >= max_thresh or area != filled_area:
            label_img = np.where(label_img == i, 0, label_img)

    label_img = np.where(label_img != 0, 255, label_img)
    return label_img


def morphology_filter(diff_img, min_thresh=3, max_thresh=400, show=True, winname=None):
    """
    filter different image with morphology operations
    :param diff_img: different image between raw image and background intensity
    :param min_thresh: minimum area threshold
    :param max_thresh: maximum area threshold
    :param show: whether to show images
    :return: image
    """
    assert show and winname is not None
    # omit negative
    diff_img[diff_img < 0] = 0

    # only preserve the 5% highest positive value
    pos_num = np.sum(diff_img > 0)
    t = sorted(diff_img.flat)[-int(0.01 * pos_num)]
    # t = list(set(sorted(diff_img.flat)))[-int(0.05 * pos_num)]
    diff_img[diff_img < t] = 0

    if show:
        cv2.imshow("diff-%s" % winname, (diff_img - np.min(diff_img)) / (np.max(diff_img) - np.min(diff_img)))

    # binary segmentation
    diff_img_binary = np.where(diff_img > 0, 255, diff_img)
    # remove tiny, huge and holey area
    removed_img = remove_tiny_huge_area(diff_img_binary, min_thresh, max_thresh)

    if show:
        cv2.imshow("removed-%s" % winname, (removed_img - np.min(removed_img)) / (np.max(removed_img) - np.min(removed_img)))

    return removed_img


if __name__ == "__main__":
    img = cv2.imread(r'C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\ROIs\benign\B_1RCC.png', cv2.IMREAD_UNCHANGED)
    img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255 + 0.5).astype(np.uint8)

    # get background intensity (twice to enhance robust)
    bg_intensity = background_intensity(img, zero_start=True, local_size=15)
    bg_intensity_1 = background_intensity(img, zero_start=False, local_size=15)
    # get segmentation based on different image
    seg0 = morphology_filter(img - bg_intensity, winname="0")
    seg1 = morphology_filter(img - bg_intensity_1, winname="1")

    # get the final segmentation
    segmentation = seg0 + seg1
    segmentation = np.where(segmentation > 0, 255, segmentation).astype(np.uint8)

    # show segmentation overlay
    cv2.imshow("image", img)
    cv2.imshow("segmentation", segmentation)
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay_r = np.where(segmentation == 255, 255, overlay[:, :, 0])
    overlay_g = np.where(segmentation == 255, 0, overlay[:, :, 1])
    overlay_b = np.where(segmentation == 255, 0, overlay[:, :, 2])
    overlay = np.stack((overlay_r, overlay_g, overlay_b), axis=2)
    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)
