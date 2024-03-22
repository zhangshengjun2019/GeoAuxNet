_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16 # bs: total bs in all gpus
num_worker = 8
mix_prob = 0.8
empty_cache = False
enable_amp = False
find_unused_parameters = True
# batch_size_test = 8

# trainer
train = dict(
    type="MultiDatasetTrainer",
)

# model settings
model = dict(
    type='PPT-v1m2',
    backbone=dict(
        type='SpUNet-GeoAuxNet',
        in_channels=6,
        num_classes=0,
        base_channels=32,
        context_channels=128,
        channels=(64, 64, 128, 256, 128, 128, 96, 64),
        layers=(4, 6, 8, 10, 8, 6, 6, 4),
        cls_mode=False,
        conditions=('ScanNet', 'S3DIS', 'NuScenes'),
        zero_init=False,
        norm_decouple=True,
        norm_adaptive=False,
        norm_affine=True),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ],
    backbone_out_channels=64,
    context_channels=128,
    conditions=('ScanNet', 'S3DIS', 'NuScenes'),
    num_classes=(20, 13, 16))


# scheduler settings
epoch = 100
optimizer = dict(type="SGD", lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)
# param_dicts = [dict(keyword="modulation", lr=0.005)]

# dataset settings
data = dict(
    num_classes=16,
    ignore_index=-1,
    names=[
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ],
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="NuScenesDataset",
                split=["train", "val"],
                data_root="data/nuscenes",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='x', p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='y', p=0.5),
                    dict(
                        type="PointClip",
                        point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="IntensityExpand", channels=3),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        keys=("coord", "strength", "segment"),
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Add", keys_dict={"condition": "NuScenes"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=-1,
                loop=8,
            ),
            # S3DIS
            dict(
                type="S3DISDataset",
                split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
                data_root="data/s3dis",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.5,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(
                        type="ElasticDistortion",
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "color", "segment"),
                    ),
                    dict(type="SphereCrop", point_max=200000, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ShufflePoint"),
                    dict(type="Add", keys_dict={"condition": "S3DIS"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "color"),
                    ),
                ],
                test_mode=False,
                loop=2,  # sampling weight
            ),
        ],
    ),
    val=dict(
        type="S3DISDataset",
        split="Area_5",
        data_root="data/s3dis",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Add", keys_dict={"condition": "S3DIS"}),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "condition"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type="SemanticKITTIDataset",
        split="val",
        data_root="data/semantic_kitti",
        transform=[
            dict(
                type="PointClip",
                point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
            ),
            dict(type="IntensityExpand", channels=3),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength"),
                return_point_patch=True,
                point_patch_feat_key=("coord", "strength"),
                side=6,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="Add", keys_dict={"condition": "SemanticKITTI"}),
                dict(type="Add", keys_dict={"sensor": "LiDAR"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition", "sensor", 
                          "point_patch_coord", "point_patch_feat", "point_patch_offset"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
