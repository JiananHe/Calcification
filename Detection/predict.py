import numpy as np
import cv2
import torch.nn.parallel
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data_loader import get_data_loader
from model import VGG
from focal_loss import FocalLoss

if __name__ == "__main__":
    r = 16
    thresh = 0.9
    # model
    net = VGG(1).cuda()
    net.load_state_dict(torch.load("./modules/vgg-160-1.433.pth.tar")['state_dict'])
    net.eval()

    # test image
    image = Image.open("./predictions/test5.jpg")

    # to patch tensor
    trans = transforms.Compose([transforms.Pad(r, padding_mode='symmetric'), transforms.Grayscale(), transforms.ToTensor()])
    image = torch.squeeze(trans(image))
    w, h = image.shape
    print(image.shape)

    patches = [image[i - r:i + r, j - r:j + r] for i in range(r, w - r) for j in range(r, h - r)]
    a = patches[23]
    print(len(patches))

    # predict
    segment = np.zeros((w - 2*r)*(h - 2*r), dtype=np.uint8)
    batch_size = 128
    with torch.no_grad():
        for p in range(0, len(patches), batch_size):
            batch = torch.stack(patches[p:p+batch_size]).unsqueeze(dim=1).cuda()
            output = net(batch)
            output = F.softmax(output, dim=1)
            predict = output[:, 1] > thresh
            # predict = torch.argmax(output, dim=1).view(-1)  # (128,)

            segment[p:p+batch_size] = predict.cpu().numpy().astype(np.uint8)
            print(batch.shape)

    segment = segment.reshape((w - 2*r, h - 2*r)) * 255
    print("cal counts: %d" % np.sum(segment))
    cv2.imwrite("./predictions/predct.jpg", segment)
    cv2.imshow("seg", segment)
    cv2.waitKey(0)

