import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config

class MNISTClassifier(nn.Module):
    def __init__(self, cfg:Config):
        super(MNISTClassifier, self).__init__()
        self.cfg = cfg
        c = cfg

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(c.cx * c.cy, c.layer1_size)
        self.layer_2 = torch.nn.Linear(c.layer1_size, c.layer2_size)
        self.layer_3 = torch.nn.Linear(c.layer2_size, c.num_classes)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

