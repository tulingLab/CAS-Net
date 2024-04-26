import torch.nn as nn
import torch
from .backbone import resnet18


class Casnet(nn.Module):
    def __init__(self):
        super(Casnet, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.conv = nn.Conv2d(1024,512,kernel_size=1)

        self.norm = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 1)
        self.act = nn.Sigmoid()

    def forward(self, x1, x2):

        out1, out2 = self.backbone(x1,x2)
        output1 = self.avg_pool(out1)
        output2 = self.avg_pool(out2)

        output = torch.cat([output1, output2], dim=1)
        output = self.relu(self.norm(self.flatten(self.conv(output))))
        out = self.act(self.fc2(output))

        return out
