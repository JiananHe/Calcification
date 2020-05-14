from sklearn.datasets import load_iris
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import h5py


def make_dataset(h5_path, split=False):
    h5f = h5py.File(h5_path, 'r')
    benign_features = np.array(h5f["benign"])
    np.random.shuffle(benign_features)
    malignant_features = np.array(h5f["malignant"])
    np.random.shuffle(malignant_features)
    benign_counts = benign_features.shape[0]
    malignant_counts = malignant_features.shape[0]

    if not split:
        features = np.concatenate([benign_features, malignant_features])
        targets = np.array([0] * benign_counts + [1] * malignant_counts)
        np.random.seed(2020)
        np.random.shuffle(features)
        np.random.seed(2020)
        np.random.shuffle(targets)
        return features, targets
    else:
        training_features = np.concatenate([benign_features[:-30], malignant_features[:-30]])
        training_targets = np.array([0] * (benign_counts - 30) + [1] * (malignant_counts - 30))
        valid_features = np.concatenate([benign_features[-30:], malignant_features[-30:]])
        valid_targets = np.array([0] * 30 + [1] * 30)
        np.random.seed(2020)
        np.random.shuffle(training_features)
        np.random.seed(2020)
        np.random.shuffle(training_targets)
        np.random.seed(2020)
        np.random.shuffle(valid_features)
        np.random.seed(2020)
        np.random.shuffle(valid_targets)
        return training_features, training_targets, valid_features, valid_targets


def find_optim_k(features, targets):
    k_range = range(1, 31)
    k_error = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, features, targets, cv=6, scoring='accuracy')
        k_error.append(1 - scores.mean())

    # 画图，x轴为k值，y值为误差值
    plt.plot(k_range, k_error)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    training_features, training_targets, valid_features, valid_targets = make_dataset(
        r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\Roi_features.h5", split=True)
    # find_optim_k(features, targets)
    knn = KNeighborsClassifier(n_neighbors=26, weights="distance")
    knn.fit(training_features, training_targets)

    # predict
    # for feature in features:
    predicts = knn.predict(valid_features)
    print(np.sum(predicts == valid_targets))
    print(predicts.shape)
    print(np.sum(predicts == valid_targets) / predicts.shape[0])
