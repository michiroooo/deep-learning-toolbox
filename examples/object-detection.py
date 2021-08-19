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
            trainable_layers=5,
            # returned_layers=[1],
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

    backbone = ResnetFPN(
        backbone_name="resnet50",
    )

    summary(backbone, input_size=(9, 3, 448, 448), device="cpu", verbose=2)

    #    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

    summary(model, input_size=(1, 3, 224, 224), device="cpu", verbose=2)

    # num_classes = 2  # 1 class (person) + background
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = (
    #     torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    #         in_features, num_classes
    #     )
    # )

    # # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # # backbone.out_channels = 1280

    # # #    summary(backbone, input_size=(9, 3, 224, 224), device="cpu", verbose=2)

    # # anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
    # #     sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    # # )
    # # roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    # #     featmap_names=[0], output_size=56, sampling_ratio=2
    # # )

    # # model = torchvision.models.detection.FasterRCNN(
    # #     backbone=backbone,
    # #     num_classes=num_classes,
    # #     rpn_anchor_generator=anchor_generator,
    # #     box_roi_pool=roi_pooler,
    # # )

    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = (
    #     torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    #         in_features, num_classes
    #     )
    # )

    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(
    #     in_features_mask, hidden_layer, num_classes
    # )

    # model.eval()

    # summary(model, input_size=(2, 3, 224, 224), device="cpu", verbose=2)

    # x = [torch.rand(3, 300, 400), torch.rand(3, 400, 400)]
    # predictions = model(x)

    # for k, v in predictions.items():

    #     print(k.v)
