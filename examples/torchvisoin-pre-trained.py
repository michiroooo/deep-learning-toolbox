import torch
import torch.nn as nn
from torchvision import models

from torchinfo import summary


class ClasifyResNet(nn.Module):
    def __init__(self, num_classes):
        super(ClasifyResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.hidden_size = resnet.fc.in_features  # 最終層手前の次元数

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):

        h = self.backbone(x)
        h = h.view(len(h), -1)
        y = self.classifier(h)

        return y


if __name__ == "__main__":

    net = ClasifyResNet(num_classes=10)

    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(torch.argmax(y))
    summary(net, input_size=(1, 3, 224, 224))
