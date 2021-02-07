import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchinfo import summary


class ResnetFPN(nn.Module):
    def __init__(self, backbone_name, embedding_size=16):

        super().__init__()

        self.backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            pretrained=True,
            trainable_layers=3,
            returned_layers=[1],
        )

        # try:
        #     feature_dim = backbone.classifier.in_features
        # except AttributeError:
        #     feature_dim = backbone.num_features

        # self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):

        return self.backbone(x)


if __name__ == "__main__":

    num_classes = 13

    model = ResnetFPN(
        backbone_name="resnet50",
    )

    x = torch.randn(9, 3, 224, 224)
    y = model(x)

    for k, v in y.items():

        # from pprint import pprint

        print(k, v.shape)

    backbone = torchvision.models.resnet50(pretrained=True)
    backbone.out_channels = 2048

    summary(backbone, input_size=(9, 3, 224, 224), device="cpu", verbose=2)

    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=[0], output_size=7, sampling_ratio=2
    )

    model = torchvision.models.detection.FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_feature
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(
    #     in_features_mask, hidden_layer, num_classes
    # )

    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 400, 400)]
    predictions = model(x)

    for k, v in predictions.items():

        print(k.v)

    # summary(model, input_size=(2, 3, 224, 224), device="cpu", verbose=2)
