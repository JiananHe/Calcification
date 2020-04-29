import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class FeaturesDataset(Dataset):
    def __init__(self, benign_h5_path, malignant_h5_path):
