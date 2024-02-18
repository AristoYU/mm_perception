#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
# @Author: AristoYU
# @Date: 2024-02-18 12:22:51
# @LastEditTime: 2024-02-18 12:26:12
# @LastEditors: AristoYU
# @Description: 20 epochs schedule for 2D detection with 8GPUS
# @FilePath: /mm_perception/configs/det2d/_base_/schedules/schedule_20e.py
"""

# training schedule for 20e
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[16, 19],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
