# model settings
model = dict(
    type='HookedRCNN',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0])),
    roi_head=dict(
        type='HookedRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0,
                use_torchvision=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='HookedBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100)))
# dataset settings
dataset_type = 'CocoDataset'
anno_root = 'data/hico_det/annotations/'
img_root = 'data/hico_20160224_det/images/'
img_norm_cfg = dict(
    mean=[117, 111.17, 102.855], std=[63.478, 61.442, 61.773], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=anno_root + 'hico_det_train.json',
        img_prefix=img_root + 'train2015/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=anno_root + 'hico_det_test.json',
        img_prefix=img_root + 'test2015/',
        pipeline=test_pipeline))
# build settings
build = dict(
    train=dict(
        max_per_img=dict(hum=-1, obj=-1),
        score_thr=dict(hum=0.1, obj=0.1),
        iou_thr=dict(pos=0.6, neg=0.4),
        max_h_as_o=-1,
        neg_pos_ub=50),
    test=dict(
        max_per_img=dict(hum=10, obj=20),
        score_thr=dict(hum=0.5, obj=0.1),
        iou_thr=dict(pos=0.5, neg=0.5),
        max_h_as_o=3))
# graph settings
checkpoint_root = 'checkpoints/'
elmo_type = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B'
graph = dict(
    embedder=dict(
        type='ELMo',
        options=checkpoint_root + elmo_type + '_options.json',
        weights=checkpoint_root + elmo_type + '_weights.hdf5',
        level=2),
    uni_weight=dict(act=0.6, obj=0.4),
    tri_weight=dict(hum=0.2, act=0.6, obj=0.2),
    emb_weight=dict(vis=0.6, sem=0.5),
    num_edges=dict(act=2, obj=3, tri=3))
