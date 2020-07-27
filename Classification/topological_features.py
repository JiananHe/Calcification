import cv2
import h5py
import os
import numpy as np
from time import time
import itertools
from skimage import morphology
import matplotlib.pyplot as plt
import networkx


DIALTE_KERNEL_TYPE = cv2.MORPH_ELLIPSE  # cv2.MORPH_RECT


def reade_mask_img(file_path):
    """
    :param file_path: path of mask image
    :return: a 2D numpy array
    """
    mask_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    # print(mask_img.shape)
    return mask_img


def initial_connected_component(mask_img, ignore_thresh=2, draw=False):
    """
    extract and split the connected components in initial mask image
    :param img: the initial mask image (2D numpy array, W*H)
    :param ignore_thresh: the calcifications that less that ignore_thresh pixels are ignored
    :param draw: whether to show image during processing
    :return: an list in which every element is a connected component
    """
    image_for_draw = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

    # find contours
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # extract and split the connected components in mask image, ignore the very tiny calcifications
    contours_list = []
    for cid in range(len(contours)):
        if cv2.contourArea(contours[cid]) < ignore_thresh:  # ignore areas less than ignore_thresh
            continue
        c = contours[cid]  # [m, 1, 2]
        # contours_list.append(np.squeeze(c, axis=1))
        contours_list.append(c)
        if draw:
            cv2.drawContours(image_for_draw, contours, cid, (0, 255, 255), 1)

    if draw:
        cv2.imshow("cals", image_for_draw)
        cv2.waitKey(1)

    return contours_list


def dilate_component(shape, contours_list, dilate_rate, draw=False):
    """
    dilate every component in contours_list with dilate_rate
    :param shape: the shape of mask image
    :param contours_list: list of connected components
    :param dilate_rate:
    :return: a list of in which every element is a dilated connected component
    """
    image_for_draw = np.zeros([shape[0], shape[1], 3], dtype=np.uint8)
    dilated_component_list = []
    for contour in contours_list:
        temp_img = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(temp_img, [contour], 255)  # add [] for pts to fill, otherwise only the outline would be drawn

        # dilate
        kernel = cv2.getStructuringElement(DIALTE_KERNEL_TYPE, (dilate_rate, dilate_rate))
        dilated_img = cv2.dilate(temp_img, kernel, iterations=1)
        # kernel = morphology.disk(dilate_rate)
        # dilated_img = morphology.dilation(temp_img, kernel)

        # get the board of dilated component
        new_contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(new_contours) == 1
        dilated_component_list += new_contours
        if draw:
            cv2.drawContours(image_for_draw, new_contours, 0, (0, 255, 255), 1)

    if draw:
        cv2.imshow("dilated", image_for_draw)
        cv2.waitKey(1)

    return dilated_component_list


def graph_generator(shape, components, old_ad_matric):
    """
    generate a graph, every component is a vertex and the connections between components determine the edges.
    also return a networkx.Graph
    :param shape: the shape of mask image
    :param components: a list of components
    :param old_ad_matric
    :return: the adjacent matrix of the graph
    """
    num_components = len(components)
    if old_ad_matric is None:
        ad_matrix = np.zeros((num_components, num_components), dtype=np.uint8)
    else:
        ad_matrix = old_ad_matric.copy()

    kx_G = networkx.Graph()
    kx_G.add_nodes_from(list(range(num_components)))

    dilate_img = np.zeros((num_components, shape[0], shape[1]), dtype=np.uint8)
    dilate_area = np.zeros(num_components)

    for i in range(num_components):
        comp = components[i]
        cv2.fillPoly(dilate_img[i], [comp], 255)
        dilate_area[i] = len(np.where(dilate_img[i] != 0)[0])

    # check whether two components are connected
    for i in range(num_components):
        for j in range(i + 1, num_components):
            # two components that have already connected must connect with current dilation rate
            if ad_matrix[i, j] == 1:
                continue
            if np.min(components[i][:, :, 0]) > np.max(components[j][:, :, 0]) \
                    or np.min(components[j][:, :, 0]) > np.max(components[i][:, :, 0])\
                    or np.min(components[i][:, :, 1]) > np.max(components[j][:, :, 1])\
                    or np.min(components[j][:, :, 1]) > np.max(components[i][:, :, 1]):
                continue
            if dilate_area[i] + dilate_area[j] > len(np.where((dilate_img[i] + dilate_img[j]) != 0)[0]):
                ad_matrix[i, j] = ad_matrix[j, i] = 1
                kx_G.add_edge(i, j)

    return ad_matrix, kx_G


def graph_features_extractor(adjacency_matrix, kx_G):
    """
    generate graph features according to the adjacency matrix
    :param adjacency_matrix:
    :param kx_G:
    :return: graph features
    """
    graph_features = {}
    num_vertexes = adjacency_matrix.shape[0]

    # degree matrix
    degree_matrix = np.zeros(adjacency_matrix.shape)
    for i in range(num_vertexes):
        assert np.sum(adjacency_matrix[:, i]) == np.sum(adjacency_matrix[i, :])
        degree_matrix[i, i] = np.sum(adjacency_matrix[:, i])
    graph_features["Average Vertex Degree"] = np.sum(degree_matrix) / num_vertexes
    graph_features["Maximum Vertex Degree"] = np.max(degree_matrix)

    # Connected Component
    subgraphs = [kx_G.subgraph(c).copy() for c in networkx.algorithms.components.connected_components(kx_G)]
    subgraph_length = [len(c) for c in subgraphs]
    graph_features["Number of Subgraphs"] = len(subgraph_length)
    # assert len(subgraph_vertexes) == len(subgraphs)

    # assert len(subgraph_vertexes) == num_subgraphs
    graph_features["Giant Connected Component Ratio"] = np.max(subgraph_length) / num_vertexes
    graph_features["Percentage of Isolated Points"] = np.sum(np.array(subgraph_length) == 1) / num_vertexes

    # eccentricity
    kx_eccentricity = []
    for s in range(len(subgraphs)):
        ecc = networkx.algorithms.distance_measures.eccentricity(subgraphs[s])
        kx_eccentricity += [i for i in ecc.values()]
    # assert np.all(sorted(np.array(kx_eccentricity)) == sorted(eccentricity))
    graph_features["Average Vertex Eccentricity"] = np.sum(kx_eccentricity) / num_vertexes
    graph_features["Diameter"] = np.max(kx_eccentricity)

    # clustering coefficient
    kx_clustering_coefficient = []
    for s in range(len(subgraphs)):
        coe = networkx.algorithms.cluster.clustering(subgraphs[s])
        kx_clustering_coefficient += [i for i in coe.values()]
    # assert np.all(sorted(np.array(kx_clustering_coefficient)) == sorted(clustering_coefficient))

    graph_features["Average Clustering Coefficient"] = np.sum(kx_clustering_coefficient) / num_vertexes

    return graph_features


def feature_extractor_main(mask_path, show=False):
    """
    main function
    :param mask_path: path of mask image
    :param show: whether to show
    :return: a feature list (512, )
    """
    mask_img = reade_mask_img(mask_path)
    contours_list = initial_connected_component(mask_img)
    # assert len(contours_list) == 0
    print("number of components %d" % len(contours_list))

    all_graph_features = []
    old_ad_matric = None
    time_begin = time()
    for dilate_rate in range(1, 65):
        dilated_component_list = dilate_component(mask_img.shape, contours_list, dilate_rate, show)
        ad_matrix, kx_G = graph_generator(mask_img.shape, dilated_component_list, old_ad_matric)
        graph_features = graph_features_extractor(ad_matrix, kx_G)
        all_graph_features += graph_features.values()
        old_ad_matric = ad_matrix

    assert len(all_graph_features) == 512 and None not in all_graph_features
    print("time: %d s" % (time() - time_begin))
    return all_graph_features


def show_topological(mask_path):
    mask_img = reade_mask_img(mask_path)
    contours_list = initial_connected_component(mask_img)
    # assert len(contours_list) == 0
    print("number of components %d" % len(contours_list))

    # extract centroid (vertex)
    vertex_pos = {}
    vertex_color = []
    shape = mask_img.shape
    for i, contour in enumerate(contours_list):
        temp_img = np.zeros(shape)
        cv2.fillPoly(temp_img, [contour], 1.0)  # add [] for pts to fill, otherwise only the outline would be drawn
        zeros_moment = np.sum(temp_img)
        coord_x, coord_y = np.meshgrid(np.array(range(shape[0])), np.array(range(shape[1])))
        one_moment_x = np.sum(coord_x * temp_img)
        one_moment_y = np.sum(coord_y * temp_img)
        centroid = np.array([one_moment_x / zeros_moment, shape[1] - one_moment_y / zeros_moment], dtype=np.int)

        # cv2.circle(temp_img, tuple(centroid), 2, 0.5)
        # cv2.imshow("single centroid", temp_img)
        # cv2.waitKey(0)
        vertex_pos[i] = centroid
        vertex_color.append(np.random.rand(3))

    # get adjacency relation (edge). Note that the order of vertex in centroids_list is the same as
    # the order of component in dilated_component_list
    old_ad_matric = None
    for dilate_rate in [2, 4, 8, 16, 32, 64]:
        dilated_component_list = dilate_component(mask_img.shape, contours_list, dilate_rate)
        ad_matrix, kx_G = graph_generator(mask_img.shape, dilated_component_list, old_ad_matric)

        # draw
        networkx.draw_networkx(kx_G, pos=vertex_pos, arrows=False, node_size=150, font_size=7, node_color=np.array(vertex_color))
        plt.show()



if __name__ == "__main__":
    mask_path = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\ROIs_seg"
    benign_mask_path = os.path.join(mask_path, "benign")
    malignant_mask_path = os.path.join(mask_path, "malignant")
    benign_features = []
    malignant_features = []

    # Test one mask with one  dilate rate
    # feature_extractor_main("test_image.bmp", True)
    show_topological("test_image.bmp")

    # for benign_case in os.listdir(benign_mask_path):
    #     print("benign %s" % benign_case)
    #     benign_features.append(feature_extractor_main(os.path.join(benign_mask_path, benign_case), True))
    #
    # for malignant_case in os.listdir(malignant_mask_path):
    #     print("malignant %s" % malignant_case)
    #     malignant_features.append(feature_extractor_main(os.path.join(malignant_mask_path, malignant_case), False))
    #
    # benign_features = np.array(benign_features)
    # malignant_features = np.array(malignant_features)
    # print(benign_features.shape, malignant_features.shape)
    #
    # # save as h5 file
    # h5f = h5py.File(r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\Roi_features.h5", 'w')
    # h5f.create_dataset("benign", data=benign_features)
    # h5f.create_dataset("malignant", data=malignant_features)
    # h5f.close()
