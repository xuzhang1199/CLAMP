# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .untils import tokenize

from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
from mmcv.runner import load_checkpoint_cococlip

from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose
import logging
import pdb

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class TopDown(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        self.init_weights(pretrained=pretrained)

    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img width: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
                Otherwise, return predicted poses, boxes, image paths
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, target_weight)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap +
                                  output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img


@POSENETS.register_module()
class CLAMP(BasePose):
    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder,
                 class_names,
                 context_length,
                 score_concat_index=3,
                 identity_head=None,
                 upconv_head=None,
                 token_embed_dim=512,
                 text_dim=1024,
                 clip_pretrained=None,
                 matching_only=False,
                 visual_dim=256,
                 CL_ratio=1.0,
                 prompt_encoder=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 loss_pose=None,
                 pretrained=None):
        super().__init__()
        self.fp16_enabled = False
        self.backbone = builder.build_backbone(backbone)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        if text_encoder is not None:
            self.text_encoder = builder.build_backbone(text_encoder)

        if context_decoder is not None:
            self.context_decoder = builder.build_backbone(context_decoder)

        self.with_prompt_encoder = False
        if prompt_encoder is not None:
            self.prompt_encoder = builder.build_backbone(prompt_encoder)
            self.with_prompt_encoder = True

        self.init_weights(pretrained=None, clip_pretrained=clip_pretrained)

        self.context_length = context_length
        self.score_concat_index = score_concat_index

        self.with_identity_head = False
        self.with_upconv_head = False
        self._init_identity_head(identity_head)
        self._init_upconv_head(upconv_head)

        self.class_names = class_names
        self.matching_only = matching_only
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.class_names)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_projection = nn.Parameter(torch.empty(text_encoder['embed_dim'], visual_dim))
        nn.init.normal_(self.text_projection, std=text_encoder['embed_dim'] ** -0.5)
        self.CL_visual = nn.CrossEntropyLoss(reduce=False)
        self.CL_text = nn.CrossEntropyLoss(reduce=False)
        self.CL_ratio = CL_ratio

        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)

        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-3)


    @property
    def with_keypoint(self):
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None, clip_pretrained=None):
        self.backbone.init_weights(pretrained=clip_pretrained)
        if self.with_keypoint:
            self.keypoint_head.init_weights()
        self.text_encoder.init_weights()

    def load_checkpoint(self,
                        filename,
                        map_location='cpu',
                        strict=False,
                        revise_keys=[(r'^module.', '')]):
        logger = logging.getLogger()
        return load_checkpoint_cococlip(
            self,
            filename,
            map_location,
            strict,
            logger,
            revise_keys=revise_keys)

    def _init_identity_head(self, identity_head):
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def _init_upconv_head(self, upconv_head):
        if upconv_head is not None:
            self.with_upconv_head = True
            self.upconv_head = builder.build_head(upconv_head)

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def spatial_adapt(self, x):
        x_orig = list(x[0:4])
        cls_token, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape
        text_embeddings = self.text_encoder(self.texts.to(cls_token.device), self.contexts).expand(B, -1, -1)

        # model the relation of prompts
        if self.with_prompt_encoder:
            text_embeddings = self.prompt_encoder(text_embeddings)

        # cross-attn to enhance text emb
        visual_tokens = torch.cat([cls_token.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                   dim=2).permute(0, 2, 1)
        refine_emb = self.context_decoder(text_embeddings, visual_tokens)
        prompt_embeddings = text_embeddings + self.gamma * refine_emb

        # spatial connection
        visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        prompt_embeddings_norm = F.normalize(prompt_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, prompt_embeddings_norm)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)

        return prompt_embeddings, x_orig[self.score_concat_index], score_map

    def feature_adapt(self, visual_embeddings, text_embeddings, target, target_weight):
        # (Batch, C, H, W) (Batch, 256, 64, 64) for x[0]
        B, C, H, W = visual_embeddings.shape
        # (Batch, K, D) (Batch, 17, 1024)
        B, K, D = text_embeddings.shape
        # (Batch, K, D) -> (Batch, K, C)
        if D != C:
            text_embeddings = text_embeddings @ self.text_projection

        target_mask = torch.where(target == 1, 1, 0)
        # (Batch, K, H, W, C) -> (Batch, K, C)
        visual_embeddings = torch.sum(torch.einsum('bkhw,bchw->bkhwc', target_mask, visual_embeddings), dim=(2, 3))

        visual_embeddings = F.normalize(visual_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.einsum('bhc,bwc->bhw', visual_embeddings, text_embeddings)
        logits_per_text = logits_per_image.transpose(1, 2).contiguous()

        losses = dict()
        labels = torch.arange(K, device=logits_per_image.device).expand(B, -1)
        loss_visual = self.CL_visual(logits_per_image, labels) * target_weight.squeeze()
        loss_text = self.CL_text(logits_per_text, labels) * target_weight.squeeze()
        contrastive_loss = (loss_visual.mean() + loss_text.mean()) / 2

        losses['feature_loss'] = contrastive_loss * self.CL_ratio

        return losses

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        target, target_down = target
        target_weight, target_down_weight = target_weight
        x = self.backbone(img)

        text_embeddings, output, score_map = self.spatial_adapt(x)

        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        contrastive_loss = self.feature_adapt(x[4][1], text_embeddings, target_down, target_down_weight)
        losses.update(contrastive_loss)
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, target_weight)
            losses.update(keypoint_losses)
            if not self.matching_only:
                keypoint_accuracy = self.keypoint_head.get_accuracy(
                    output, target, target_weight)
                losses.update(keypoint_accuracy)

        if self.with_upconv_head:
            score_map = self.upconv_head(score_map)

        if self.with_identity_head:
            spatial_losses = self.identity_head.get_loss(
                score_map, target, target_weight)
            losses.update(spatial_losses)
            if self.matching_only:
                keypoint_accuracy = self.identity_head.get_accuracy(
                    score_map, target, target_weight)
                losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)

        text_embeddings, features, score_map = self.spatial_adapt(features)

        if self.with_keypoint:
            if not self.matching_only:
                output_heatmap = self.keypoint_head.inference_model(
                    features, flip_pairs=None)
            else:
                assert self.with_upconv_head
                output_heatmap = self.upconv_head.inference_model(
                    score_map, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)

            text_embeddings, features_flipped, score_map_flipped = self.spatial_adapt(features_flipped)
            if self.with_keypoint:
                if not self.matching_only:
                    output_flipped_heatmap = self.keypoint_head.inference_model(
                        features_flipped, img_metas[0]['flip_pairs'])
                else:
                    assert self.with_upconv_head
                    output_flipped_heatmap = self.upconv_head.inference_model(
                        score_map_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap +
                                  output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            if not self.matching_only:
                keypoint_result = self.keypoint_head.decode(
                    img_metas, output_heatmap, img_size=[img_width, img_height])
            else:
                keypoint_result = self.upconv_head.decode(
                    img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img