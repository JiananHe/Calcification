import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data_loader import get_data_loader
from model import VGG
from focal_loss import FocalLoss


def valid_accuracy(valid_loader, net):
    print("valid...")
    net.eval()
    gt_pos = 0
    pd_pos = 0
    true_pos = 0
    correct = 0
    all_num = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(valid_loader):
            images = images.cuda()
            target = target.cuda()

            output = net(images)
            output = F.softmax(output, dim=1)
            predict = torch.argmax(output, dim=1).view(-1)  # (8,)

            gt_pos += torch.sum(target == 1).item()
            pd_pos += torch.sum(predict == 1).item()
            true_pos += torch.sum(target[target == 1] == predict[target == 1]).item()
            correct += torch.sum(target == predict).item()
            all_num += images.shape[0]

    precious = true_pos / pd_pos
    recall = true_pos / gt_pos
    accuracy = correct / all_num
    return precious, recall, accuracy


if __name__ == "__main__":
    # hyper param
    init_lr = 0.0001
    weight_decay = 0.0005
    Epochs = 1000
    train_dir = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\INBreast\Sample\train"
    valid_dir = r"C:\Users\13249\Desktop\20200115-20200205\Calcification\Data\INBreast\Sample\valid"

    # data_loader
    train_loader = get_data_loader(train_dir, data_augment=True)
    valid_loader = get_data_loader(train_dir, data_augment=False, batch_size=128)

    # model
    net = VGG(1).cuda()

    # define loss function (criterion) and optimizer
    loss_func = FocalLoss().cuda()
    opt = optim.Adam(net.parameters(), init_lr, weight_decay=weight_decay)
    lr_decay = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    # train
    writer = SummaryWriter()
    for epoch in range(0, Epochs + 1):
        net.train()
        losses = []
        for i, (images, target) in enumerate(train_loader):
            images = images.cuda()
            target = target.cuda()

            output = net(images)
            loss = loss_func(output, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

            print("Epoch: %4d, step: %4d, loss: %.3f" % (epoch, i, loss.item()))

        mean_loss = np.mean(losses)
        lr_decay.step()

        writer.add_scalar('train/loss', mean_loss, epoch)
        writer.add_scalar('learning rate', opt.param_groups[0]['lr'], epoch)
        print("=============== Mean loss in epoch %4d is: %.3f ===============" % (epoch, mean_loss))

        if epoch % 5 == 0:
            precious, recall, accuracy = valid_accuracy(valid_loader, net)
            print("=============== Epoch %4d - precious: %.3f, recall: %.3f, accuracy: %.3f ==============="
                  % (epoch, precious, recall, accuracy))
            writer.add_scalars('validation', {'precious': precious,
                                              'recall': recall,
                                              'accuracy': accuracy}, epoch)
            torch.save({'epoch': epoch, 'lr': opt.param_groups[0]['lr'], 'state_dict': net.state_dict()},
                       "./modules/vgg-%d-%.3f.pth.tar" % (epoch, mean_loss))

    writer.close()
