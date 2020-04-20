import torch
import torch.nn as nn
import torch.nn.functional as F

from focal_loss import FocalLoss


vgg16_config = [16, 16, 32, 'M', 32, 32, 64, 'M', 64, 64, 64, 'M']   # 64 * (4*4) = 1024


class VGG(nn.Module):
    def __init__(self, input_channel, dropout_rate=0.2):
        super(VGG, self).__init__()
        self.dropout_rate = dropout_rate
        self.input_channel = input_channel
        self.conv_layers = self.make_layers(vgg16_config)

        # 由于输入size不确定，所以在进行Linear之前，可以通过AdaptiveAvgPool2d来固定feature map size
        self.classifier = nn.Sequential(nn.Linear(1024, 64),
                                        nn.Linear(64, 2))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def make_layers(self, cfg):
        layers = []
        last_channel = self.input_channel
        for c in cfg:
            if c == 'M':
                layers += [nn.Conv2d(last_channel, last_channel, kernel_size=2, stride=2),
                           nn.Dropout2d(self.dropout_rate)]
            else:
                layers += [nn.Conv2d(last_channel, c, kernel_size=3, padding=1),
                           nn.BatchNorm2d(c),
                           nn.PReLU()]
                last_channel = c
        return nn.Sequential(*layers)


if __name__ == "__main__":
    vgg = VGG(1).cuda()
    loss_func = FocalLoss()
    print(vgg)

    img = torch.randn((8, 1, 32, 32)).cuda()
    tgt = torch.randint(0, 2, (8,)).cuda()
    with torch.no_grad():
        pred = vgg(img)
        print(pred)
        loss = loss_func(pred, tgt)
        print(loss.item())


