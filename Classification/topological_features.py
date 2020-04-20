import cv2
import numpy as np
import itertools


def reade_mask_img(file_path):
    """
    :param file_path: path of mask image
    :return: a 2D numpy array
    """
    mask_img = cv2.imread(file_path)
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    print(mask_img.shape)
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
        cv2.imshow("mask", img)
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
    :param shape: the shape of mask image
    :param components: a list of components
    :return: the adjacent matrix of the graph
    """
    num_components = len(components)
    ad_matrix = np.zeros((num_components, num_components), dtype=np.uint8)

    # check whether two components are connected
    for i in range(num_components):
        for j in range(i + 1, num_components):
            com_a = components[i]
            com_b = components[j]
            temp_img_a = np.zeros(shape, dtype=np.uint8)
            temp_img_b = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(temp_img_a, [com_a], 255)
            cv2.fillPoly(temp_img_b, [com_b], 255)

            if np.sum(temp_img_a != 0) + np.sum(temp_img_b != 0) > np.sum((temp_img_a + temp_img_b) != 0):
                ad_matrix[i, j] = ad_matrix[j, i] = 1

    return ad_matrix


def floyd_min_dist(adjacency_matrix):
    """
    Multi-source shortest path with floyd algorithm
    :param adjacency_matrix:
    :return: the shortest distance between every two vertexes
    """
    num_vertexes = adjacency_matrix.shape[0]
    shortest_distances = adjacency_matrix.copy().astype(np.int8)
    shortest_distances[shortest_distances == 0] = num_vertexes  # stand for unreachable
    shortest_distances.flat[::(num_vertexes + 1)] = 0  # set all diagonal elements as 0

    for k in range(num_vertexes):
        for i in range(num_vertexes):
            for j in range(i + 1, num_vertexes):
                if shortest_distances[i, k] + shortest_distances[k, j] < shortest_distances[i, j]:
                    shortest_distances[i, j] = shortest_distances[j, i] = \
                        shortest_distances[i, k] + shortest_distances[k, j]

    return shortest_distances


def dfs_max_subgraph(adjacency_matrix):
    """
    get the number of vertexes in every sub-graph with DFS algorithm
    :param adjacency_matrix:
    :return: a list
    """
    num_vertexes = adjacency_matrix.shape[0]
    stack_vertex = []
    subgraph_vertexes = []
    is_checked = np.zeros(num_vertexes)
    while (is_checked == 0).any():
        v = np.argwhere(is_checked == 0).squeeze(1)[0]  # get one unchecked vertex v
        stack_vertex.append(v)
        temp_num_subgraph = 0
        is_checked[v] = 1
        while len(stack_vertex) > 0:
            top = stack_vertex.pop(-1)
            temp_num_subgraph += 1

            neighbors = np.argwhere(adjacency_matrix[top, :] == 1).squeeze(1)  # get the neighbors of vertex v
            unchecked_neighbors = list(filter(lambda i: is_checked[i] == 0, neighbors))  # push the unchecked neighbors
            stack_vertex += unchecked_neighbors
            is_checked[unchecked_neighbors] = 1

        subgraph_vertexes.append(temp_num_subgraph)
    return subgraph_vertexes


def graph_features_extractor(adjacency_matrix):
    """
    generate graph features according to the adjacency matrix
    :param adjacency_matrix:
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
    normalized_laplacian_matrix = np.zeros(adjacency_matrix.shape)
    for i in range(num_vertexes):
        for j in range(i, num_vertexes):
            if i == j and degree_matrix[i, j] != 0:
                normalized_laplacian_matrix[i, j] = 1
            elif adjacency_matrix[i, j] == 1:
                if degree_matrix[i, i] == 0 or degree_matrix[j, j] == 0:
                    normalized_laplacian_matrix[i, j] = normalized_laplacian_matrix[j, i] = 0
                else:
                    normalized_laplacian_matrix[i, j] = normalized_laplacian_matrix[j, i] \
                        = -1 / (np.sqrt(degree_matrix[i, i] * degree_matrix[j, j]))

    eigenvalues, _ = np.linalg.eig(normalized_laplacian_matrix)
    eigenvalues = np.round(eigenvalues, 8)
    assert (0 <= eigenvalues).all() and (eigenvalues <= 2).all()
    num_subgraphs = np.sum(eigenvalues == 0)
    assert num_subgraphs > 0
    graph_features["Number of Subgraphs"] = num_subgraphs

    # Connected Component
    subgraph_vertexes = dfs_max_subgraph(adjacency_matrix)
    assert len(subgraph_vertexes) == num_subgraphs
    graph_features["Giant Connected Component Ratio"] = np.max(subgraph_vertexes) / num_vertexes
    graph_features["Percentage of Isolated Points"] = np.sum(np.array(subgraph_vertexes) == 1) / num_vertexes

    # the shortest distance
    shortest_distances = floyd_min_dist(adjacency_matrix)
    shortest_distances[shortest_distances == num_vertexes] = -1

    # eccentricity
    eccentricity = np.max(shortest_distances, axis=0)
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
    graph_features["Average Clustering Coefficient"] = np.sum(clustering_coefficient) / num_vertexes

    return graph_features


if __name__ == "__main__":
    test_img_path = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Classification\MC\M_111_LCC_mask.bmp"
    mask_img = reade_mask_img(test_img_path)
    contours_list = initial_connected_component(mask_img)

    all_graph_features = {}
    for dilate_rate in range(1, 65):
        dilated_component_list = dilate_component(mask_img.shape, contours_list, 1, False)
        ad_matrix = graph_generator(mask_img.shape, dilated_component_list)
        graph_features = graph_features_extractor(ad_matrix)
        all_graph_features[dilate_rate] = graph_features

    print(all_graph_features)
