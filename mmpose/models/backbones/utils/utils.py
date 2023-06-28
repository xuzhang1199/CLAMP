# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import torch
import torch.nn.functional as F
import pdb
from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint

    state_dict = OrderedDict()
    # strip prefix of state_dict
    for k, v in state_dict_tmp.items():
        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_checkpoint_clip(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint

    state_dict = OrderedDict()
    # strip prefix of state_dict
    for k, v in state_dict_tmp.items():
        if 'attnpool.positional_embedding' in k:
            # pdb.set_trace()
            if model.attnpool.positional_embedding.shape != v.shape:
                print(
                    f'Resize the pos_embed shape from {v.shape} to {model.attnpool.positional_embedding.shape}')
                cls_pos = v[0:1, :]
                if model.same_dim:
                    H = W = model.input_resolution // 32
                else:
                    H = model.input_resolution // 32 #H is longer
                    W = model.short_dim // 32
                # old_h = int(math.sqrt(state_dict[new_k][1:,].shape[0]))
                old_h = 8
                old_w = 6
                spatial_pos = F.interpolate(
                    v[1:, ].reshape(1, old_h, old_w, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W),
                    mode='bilinear')
                spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H * W).permute(1, 0)
                positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                v = positional_embedding
                assert model.attnpool.positional_embedding.shape == v.shape
        elif 'backbone.positional_embedding' in k:
            # pdb.set_trace()
            if model.positional_embedding.shape != v.shape:
                print(
                    f'Resize the pos_embed shape from {v.shape} to {model.positional_embedding.shape}')
                cls_pos = v[0:1, :]
                if model.same_dim:
                    H = W = model.input_resolution // model.patch_size
                else:
                    H = model.input_resolution // model.patch_size #H is longer
                    W = model.short_dim // model.patch_size
                # old_h = int(math.sqrt(state_dict[new_k][1:,].shape[0]))
                old_h = 16
                old_w = 12
                spatial_pos = F.interpolate(
                    v[1:, ].reshape(1, old_h, old_w, 768).permute(0, 3, 1, 2), size=(H, W),
                    mode='bilinear')
                spatial_pos = spatial_pos.reshape(768, H * W).permute(1, 0)
                positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                v = positional_embedding
                assert model.positional_embedding.shape == v.shape

        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def get_state_dict(filename, map_location='cpu'):
    """Get state_dict from a file or URI.

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.

    Returns:
        OrderedDict: The state_dict.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint

    state_dict = OrderedDict()
    # strip prefix of state_dict
    for k, v in state_dict_tmp.items():
        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v

    return state_dict
