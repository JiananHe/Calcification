import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset


def make_dataloaders(h5_path, batch_size):
    """
    make train dataloader and valid dataloader
    :param h5_path:
    :param batch_size:
    :return:
    """
    h5f = h5py.File(h5_path, 'r')
    benign_features = np.array(h5f["benign"])
    np.random.shuffle(benign_features)
    malignant_features = np.array(h5f["malignant"])
    np.random.shuffle(malignant_features)
    benign_counts = benign_features.shape[0]
    malignant_counts = malignant_features.shape[0]
    print(benign_features.shape, benign_features.dtype)
    print(malignant_features.shape, malignant_features.dtype)

    training_features = torch.from_numpy(np.concatenate([benign_features[:-40], malignant_features[:-40]]))
    training_targets = torch.from_numpy(np.array([0] * (benign_counts - 40) + [1] * (malignant_counts - 40)))
    training_dataset = TensorDataset(training_features, training_targets)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_features = torch.from_numpy(np.concatenate([benign_features[-40:], malignant_features[-40:]]))
    valid_targets = torch.from_numpy(np.array([0] * 40 + [1] * 40))
    valid_dataset = TensorDataset(valid_features, valid_targets)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return training_dataloader, valid_dataloader


class MLP(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = self.make_layers(512, 128)
        self.fc2 = self.make_layers(128, 64)
        self.fc3 = self.make_layers(64, 2)
        self.out = nn.Softmax()

    def make_layers(self, in_features, out_features):
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return nn.Softmax(dim=1)(x)


if __name__ == "__main__":
    training_dataloader, valid_dataloader = make_dataloaders(
        r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\PrivateData\Roi_features.h5", 16)
    print(torch.cuda.is_available())

    net = MLP().double().cuda()
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    loss_func = nn.CrossEntropyLoss().cuda()

    writer = SummaryWriter()
    for epoch in range(100):
        train_losses = []
        for i, (features, targets) in enumerate(training_dataloader):
            features = features.cuda()
            targets = targets.long().cuda()
            predicts = net(features)
            loss = loss_func(predicts, targets)
            print(loss.item())
            train_losses.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
        print("epoch: %d - train loss: %.3f" % (epoch, np.mean(train_losses)))

        valid_losses = []
        for i, (features, targets) in enumerate(valid_dataloader):
            features = features.cuda()
            targets = targets.long().cuda()
            predicts = net(features)
            loss = loss_func(predicts, targets)
            print(loss.item())
            valid_losses.append(loss.item())
        print("epoch: %d - valid loss: %.3f" % (epoch, np.mean(valid_losses)))

        writer.add_scalars('loss', {'train': np.mean(train_losses),
                                    'valid': np.mean(valid_losses)}, epoch)
    writer.close()
