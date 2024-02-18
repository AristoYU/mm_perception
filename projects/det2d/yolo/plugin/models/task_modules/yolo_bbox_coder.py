#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
# @Author: AristoYU
# @Date: 2024-02-18 21:47:03
# @LastEditTime: 2024-02-18 21:56:55
# @LastEditors: AristoYU
# @Description: Custom yolo bbox coder
# @FilePath: /mm_perception/projects/det2d/yolo/plugin/models/task_modules/yolo_bbox_coder.py
"""

from typing import Union

import torch
from torch import Tensor

from mmengine.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, HorizontalBoxes, get_box_tensor
from mmdet.models.task_modules.coders.yolo_bbox_coder import YOLOBBoxCoder


@TASK_UTILS.register_module()
class YOLOBBoxCoderV2(YOLOBBoxCoder):
    """ Fix YOLOBBoxCoder Double tensor to Float tensor bug.
    """

    def encode(self, bboxes: Tensor | BaseBoxes, gt_bboxes: Tensor | BaseBoxes, stride: Tensor | int) -> Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor or :obj:`BaseBoxes`): Source boxes,
                e.g., anchors.
            gt_bboxes (torch.Tensor or :obj:`BaseBoxes`): Target of the
                transformation, e.g., ground-truth boxes.
            stride (torch.Tensor | int): Stride of bboxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        bboxes = get_box_tensor(bboxes).float()
        gt_bboxes = get_box_tensor(gt_bboxes).float()
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        w_target = torch.log((w_gt / w).clamp(min=self.eps))
        h_target = torch.log((h_gt / h).clamp(min=self.eps))
        x_center_target = ((x_center_gt - x_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        y_center_target = ((y_center_gt - y_center) / stride + 0.5).clamp(
            self.eps, 1 - self.eps)
        encoded_bboxes = torch.stack(
            [x_center_target, y_center_target, w_target, h_target], dim=-1)
        return encoded_bboxes

    def decode(self, bboxes: Union[Tensor, BaseBoxes], pred_bboxes: Tensor,
               stride: Union[Tensor, int]) -> Union[Tensor, BaseBoxes]:
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor or :obj:`BaseBoxes`): Basic boxes,
                e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            Union[torch.Tensor, :obj:`BaseBoxes`]: Decoded boxes.
        """
        bboxes = get_box_tensor(bboxes).float()
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
            pred_bboxes[..., :2] - 0.5) * stride
        whs = (bboxes[..., 2:] -
               bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
        decoded_bboxes = torch.stack(
            (xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] -
             whs[..., 1], xy_centers[..., 0] + whs[..., 0],
             xy_centers[..., 1] + whs[..., 1]),
            dim=-1)

        if self.use_box_type:
            decoded_bboxes = HorizontalBoxes(decoded_bboxes)
        return decoded_bboxes
