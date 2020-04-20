import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import cv2


def get_data_loader(dir, data_augment, batch_size=64, num_workers=4):
    loader = data.DataLoader(
        datasets.ImageFolder(
            dir,
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=[0.8, 1.5], contrast=[0.8, 1.5]),
                transforms.Grayscale(),
                transforms.ToTensor()  # [0-255] uint8 -> [0.0 - 1.0] float_tensor
            ] if data_augment else
                [transforms.Grayscale(), transforms.ToTensor()]
            )),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


if __name__ == "__main__":
    train_dir = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\INBreast\Sample\train"
    valid_dir = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\INBreast\Sample\valid"
    train_loader = get_data_loader(train_dir, data_augment=True)
    valid_loader = get_data_loader(train_dir, data_augment=False)

    for i, (images, target) in enumerate(valid_loader):
        print(i, images.shape, target.shape, target)
        images = images.cpu().numpy()
        # cv2.imshow("test", np.squeeze(images[0]))
        # cv2.waitKey(0)
