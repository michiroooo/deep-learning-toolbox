import torch
import torch.nn as nn
from torchsummary import summary

import timm


class FeatureNet(nn.Module):

    def __init__(self, backbone_name, num_classes, embedding_size=16):

        super(FeatureNet, self).__init__()

        backbone = timm.create_model(
            backbone_name,
            num_classes
        )

        backbone.forward_features()

        try:
            feature_dim = backbone.classifier.in_features
        except AttributeError:
            feature_dim = backbone.num_features

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):

        h = self.encoder(x)
        h = h.view(len(h), -1)
        y = self.classifier(h)

        return y

    


class ClasifyNet(nn.Module):

    def __init__(self, backbone_name, num_classes, embedding_size=16):

        super(ClasifyNet, self).__init__()

        backbone = timm.create_model(
            backbone_name,
            num_classes
        )

        try:
            feature_dim = backbone.classifier.in_features
        except AttributeError:
            feature_dim = backbone.num_features

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):

        h = self.encoder(x)
        h = h.view(len(h), -1)
        y = self.classifier(h)

        return y


if __name__ == '__main__':

    print(timm.list_models())

    net = timm.create_model(
        model_name="seresnext50_32x4d",
        num_classes=10,
        in_chans=3,
    )

    net = timm.create_model(
        model_name="efficientnet_b0",
        num_classes=10,
        in_chans=3,
    )


    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(torch.argmax(y))

    summary(net, (3, 224, 224), batch_size=1, device="cpu")