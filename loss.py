# list all the additional loss functions

import torch
import torch.nn as nn


################## entropy loss (continuous target) #####################
def entropy(pred):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    loss = torch.mean(torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss
