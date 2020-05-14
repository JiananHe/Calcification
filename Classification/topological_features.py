import cv2
import h5py
import os
import numpy as np
import itertools
import networkx


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
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_rate, dilate_rate))
        dilated_img = cv2.dilate(temp_img, kernel, iterations=1)

        # get the board of dilated component
        new_contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(new_contours) == 1
        dilated_component_list += new_contours
        if draw:
            cv2.drawContours(image_for_draw, new_contours, 0, (0, 255, 255), 1)

    if draw:
        cv2.imshow("dilated", image_for_draw)
        cv2.waitKey(1)

    return dilated_component_list


def graph_generator(shape, components):
    """
    generate a graph, every component is a vertex and the connections between components determine the edges.
    also return a networkx.Graph
    :param shape: the shape of mask image
    :param components: a list of components
    :return: the adjacent matrix of the graph
    """
    num_components = len(components)
    ad_matrix = np.zeros((num_components, num_components), dtype=np.uint8)
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
            if np.min(components[i][:, :, 0]) > np.max(components[j][:, :, 0]) \
                    or np.min(components[j][:, :, 0]) > np.max(components[i][:, :, 0])\
                    or np.min(components[i][:, :, 1]) > np.max(components[j][:, :, 1])\
                    or np.min(components[j][:, :, 1]) > np.max(components[i][:, :, 1]):
                continue
            if dilate_area[i] + dilate_area[j] > len(np.where((dilate_img[i] + dilate_img[j]) != 0)[0]):
                ad_matrix[i, j] = ad_matrix[j, i] = 1
                kx_G.add_edge(i, j)

    return ad_matrix, kx_G


def floyd_min_dist(adjacency_matrix, subgraph_vertexes):
    """
    Multi-source shortest path with floyd algorithm
    :param adjacency_matrix:
    :param subgraph_vertexes: a list in which every element contains the vertex indexes in every sub-graph
    :return: the shortest distance between every two vertexes
    """
    all_vertexes = adjacency_matrix.shape[0]
    shortest_distances = adjacency_matrix.copy().astype(np.int)
    shortest_distances[shortest_distances == 0] = all_vertexes  # stand for unreachable
    shortest_distances.flat[::(all_vertexes + 1)] = 0  # set all diagonal elements as 0

    for subgraph in subgraph_vertexes:
        num_vertexes = len(subgraph)
        for k in range(num_vertexes):
            kid = subgraph[k]
            for i in range(num_vertexes):
                iid = subgraph[i]
                for j in range(i + 1, num_vertexes):
                    jid = subgraph[j]
                    if shortest_distances[iid, kid] + shortest_distances[kid, jid] < shortest_distances[iid, jid]:
                        shortest_distances[iid, jid] = shortest_distances[jid, iid] = \
                            shortest_distances[iid, kid] + shortest_distances[kid, jid]

    return shortest_distances


def dfs_max_subgraph(adjacency_matrix):
    """
    get the number of vertexes in each sub-graph with DFS algorithm
    :param adjacency_matrix:
    :return: a list in which every element contains the vertex indexes in every sub-graph,
    also return the number of vertexes in every sub-graph
    """
    num_vertexes = adjacency_matrix.shape[0]
    stack_vertex = []
    subgraph_vertexes = []
    subgraph_length = []
    is_checked = np.zeros(num_vertexes)
    while (is_checked == 0).any():
        v = np.argwhere(is_checked == 0).squeeze(1)[0]  # get one unchecked vertex v
        stack_vertex.append(v)
        temp_subgraph_num = 0
        temp_subgraph_ver = []
        is_checked[v] = 1
        while len(stack_vertex) > 0:
            top = stack_vertex.pop(-1)
            temp_subgraph_ver.append(top)
            temp_subgraph_num += 1

            neighbors = np.argwhere(adjacency_matrix[top, :] == 1).squeeze(1)  # get the neighbors of vertex v
            unchecked_neighbors = list(filter(lambda i: is_checked[i] == 0, neighbors))  # push the unchecked neighbors
            stack_vertex += unchecked_neighbors
            is_checked[unchecked_neighbors] = 1

        subgraph_vertexes.append(temp_subgraph_ver)
        subgraph_length.append(temp_subgraph_num)
    return subgraph_vertexes, subgraph_length


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

    # normalized laplacian matrix
    # normalized_laplacian_matrix = np.zeros(adjacency_matrix.shape)
    # for i in range(num_vertexes):
    #     for j in range(i, num_vertexes):
    #         if i == j and degree_matrix[i, j] != 0:
    #             normalized_laplacian_matrix[i, j] = 1
    #         elif adjacency_matrix[i, j] == 1:
    #             if degree_matrix[i, i] == 0 or degree_matrix[j, j] == 0:
    #                 normalized_laplacian_matrix[i, j] = normalized_laplacian_matrix[j, i] = 0
    #             else:
    #                 normalized_laplacian_matrix[i, j] = normalized_laplacian_matrix[j, i] \
    #                     = -1 / (np.sqrt(degree_matrix[i, i] * degree_matrix[j, j]))
    #
    # eigenvalues, _ = np.linalg.eig(normalized_laplacian_matrix)
    # eigenvalues = np.round(eigenvalues, 8)
    # assert (0 <= eigenvalues).all() and (eigenvalues <= 2).all()
    # num_subgraphs = np.sum(eigenvalues == 0)
    # assert num_subgraphs > 0
    # graph_features["Number of Subgraphs"] = num_subgraphs

    # Connected Component
    subgraph_vertexes, subgraph_length = dfs_max_subgraph(adjacency_matrix)
    graph_features["Number of Subgraphs"] = len(subgraph_vertexes)
    subgraphs = [kx_G.subgraph(c).copy() for c in networkx.algorithms.components.connected_components(kx_G)]
    assert len(subgraph_vertexes) == len(subgraphs)

    # assert len(subgraph_vertexes) == num_subgraphs
    graph_features["Giant Connected Component Ratio"] = np.max(subgraph_length) / num_vertexes
    graph_features["Percentage of Isolated Points"] = np.sum(np.array(subgraph_length) == 1) / num_vertexes

    # the shortest distance
    shortest_distances = floyd_min_dist(adjacency_matrix, subgraph_vertexes)
    shortest_distances[shortest_distances == num_vertexes] = -1

    # eccentricity
    eccentricity = np.max(shortest_distances, axis=0)
    kx_eccentricity = []
    for s in range(len(subgraphs)):
        ecc = networkx.algorithms.distance_measures.eccentricity(subgraphs[s])
        kx_eccentricity += [i for i in ecc.values()]
    assert np.all(sorted(np.array(kx_eccentricity)) == sorted(eccentricity))
    graph_features["Average Vertex Eccentricity"] = np.sum(eccentricity) / num_vertexes
    graph_features["Diameter"] = np.max(eccentricity)


    # clustering coefficient
    clustering_coefficient = np.zeros(num_vertexes)
    for i in range(num_vertexes):
        neighbors = np.argwhere(adjacency_matrix[i, :] == 1).squeeze(1)  # get the neighbors of vertex i
        if len(neighbors) <= 1:  # no neighbor or no edge when only one neighbor
            clustering_coefficient[i] = 0
        else:
            # get the number of edges between the neighbors of vertex i
            clustering_coefficient[i] = sum(adjacency_matrix[i] for i in itertools.product(neighbors, neighbors))
            # divided by the number of possible edges between the neighbors of vertex i
            clustering_coefficient[i] /= len(neighbors) * (len(neighbors) - 1)
    kx_clustering_coefficient = []
    for s in range(len(subgraphs)):
        coe = networkx.algorithms.cluster.clustering(subgraphs[s])
        kx_clustering_coefficient += [i for i in coe.values()]
    assert np.all(sorted(np.array(kx_clustering_coefficient)) == sorted(clustering_coefficient))

    graph_features["Average Clustering Coefficient"] = np.sum(clustering_coefficient) / num_vertexes

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
    print("number of components %d" % len(contours_list))

    all_graph_features = []
    for dilate_rate in range(1, 65):
        print(dilate_rate)
        dilated_component_list = dilate_component(mask_img.shape, contours_list, dilate_rate, show)
        ad_matrix, kx_G = graph_generator(mask_img.shape, dilated_component_list)
        graph_features = graph_features_extractor(ad_matrix, kx_G)
        all_graph_features += graph_features.values()

    return all_graph_features


if __name__ == "__main__":
    mask_path = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\ROIs_seg"
    benign_mask_path = os.path.join(mask_path, "benign")
    malignant_mask_path = os.path.join(mask_path, "malignant")
    benign_features = []
    malignant_features = []

    # Test one mask with one  dilate rate
    mask_img = reade_mask_img(os.path.join(benign_mask_path, "B_100RMLO.png"))
    contours_list = initial_connected_component(mask_img)
    dilated_component_list = dilate_component(mask_img.shape, contours_list, 5, True)
    ad_matrix, kx_G = graph_generator(mask_img.shape, dilated_component_list)
    graph_features = graph_features_extractor(ad_matrix, kx_G)

    for benign_case in os.listdir(benign_mask_path):
        print("benign %s" % benign_case)
        benign_features.append(feature_extractor_main(os.path.join(benign_mask_path, benign_case)))

    for malignant_case in os.listdir(malignant_mask_path):
        print("malignant %s" % malignant_case)
        malignant_features.append(feature_extractor_main(os.path.join(malignant_mask_path, malignant_case)))

    benign_features = np.array(benign_features)
    malignant_features = np.array(malignant_features)
    print(benign_features.shape, malignant_features.shape)

    # save as h5 file
    h5f = h5py.File(r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\Roi_features.h5", 'w')
    h5f.create_dataset("benign", data=benign_features)
    h5f.create_dataset("malignant", data=malignant_features)
    h5f.close()
