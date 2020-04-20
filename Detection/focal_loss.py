import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, output, target):
        if output.dim() > 2:
            output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W => N,C,H*W
            output = output.transpose(1, 2)  # N,C,H*W => N,H*W,C
            output = output.contiguous().view(-1, output.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(output, dim=1)  # 先softmax再求log
        logpt = logpt.gather(1, target)  # 根据target的index信息选择pt
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != output.data.type():
                self.alpha = self.alpha.type_as(output.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
