# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by ZZXXYY
"""
DETR model and criterion classes.
"""
import argparse

import torch
from torch import nn

from misc import (NestedTensor, nested_tensor_from_tensor_list)

from mlp import MLP
from backbone import Backbone, Joiner
from transformer import Transformer
from position_encoding import PositionEmbeddingSine

from yoolkit.arguments import load_arguments


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone_path=""):
        """ Initializes the model.
        """
        super().__init__()

        args = load_arguments('./model.hocon')
        # DETR
        num_classes = args.num_classes
        num_queries = args.transformer.num_queries

        self.transformer = Transformer(
            d_model=args.transformer.hidden_dim,
            dropout=args.transformer.dropout,
            nhead=args.transformer.nheads,
            dim_feedforward=args.transformer.dim_feedforward,
            num_encoder_layers=args.transformer.enc_layers,
            num_decoder_layers=args.transformer.dec_layers,
            normalize_before=args.transformer.pre_norm,
            return_intermediate_dec=True,
        )

        position_embedding = PositionEmbeddingSine(args.transformer.hidden_dim // 2, normalize=True)
        if backbone_path != "":
            self.backbone = Joiner(Backbone(args.backbone.type, train_backbone=True, return_interm_layers=False, dilation=args.backbone.dilation, path=backbone_path), position_embedding)
        else:
            self.backbone = Joiner(Backbone(args.backbone.type, train_backbone=True, return_interm_layers=False, dilation=args.backbone.dilation), position_embedding)

        hidden_dim = self.transformer.d_model
        num_channels = self.backbone.num_channels

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test or Convert Model Weights")
    parser.add_argument('--mode', required=True, choices=['create', 'test'])
    parser.add_argument('--detr-path', type=str, default="../../bins/detr-r101-2c7b67e5.pth")
    parser.add_argument('--backbone-path', type=str, default="../../bins/resnet101-63fe2227.pth")
    parser.add_argument('--model-path', type=str, default="../../bins/DETR_ResNet101_BSZ2.pth")
    args = parser.parse_args()
    #assert sys.argv[2] in set(['bsz1', 'bsz2'])
    #ver = sys.argv[2].upper()
    if args.mode == 'create':
        # create whole
        detr = DETR(args.backbone_path)

        checkpoint = torch.load(args.detr_path, map_location='cpu')

        #print(detr.backbone.named_parameters.keys())
        detr.load_state_dict(checkpoint['model'])
        print(detr)
        torch.save(detr.state_dict(), args.model_path)

        detr_new = DETR()
        params = torch.load(args.model_path, map_location='cpu')
        detr_new.load_state_dict(params)
        for name in list(detr_new.state_dict()):
            assert torch.sum(detr_new.state_dict()[name] != detr.state_dict()[name]) == 0, f"{name}"
            print(f"PASS: {name}")

    if args.mode == 'test':
        detr = DETR()
        params = torch.load(args.model_path, map_location='cpu')
        detr.load_state_dict(params)
        device = torch.device('cuda:0')
        detr.to(device)
        x = torch.randn(4,3,5,6).to(device)
        rs = detr(x)
        print(rs['pred_boxes'].size())
        #print(detr)
