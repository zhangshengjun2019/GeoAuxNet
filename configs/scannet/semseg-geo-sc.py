_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4 # bs: total bs in all gpus
num_worker = 1
mix_prob = 0.8
empty_cache = False
enable_amp = False
find_unused_parameters = True

# trainer
train = dict(
    type="MultiDatasetTrainer",
)

# model settings
model = dict(
    type="PPT-v1m3",
    backbone=dict(
        type='SpUNet-GeoAuxNet',
        in_channels=6,
        pointnet_in_channels=6,
        num_classes=0,
        base_channels=32,
        context_channels=256,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=False,
        conditions=("ScanNet", "S3DIS", "SemanticKITTI"),
        zero_init=False,
        norm_decouple=True,
        norm_adaptive=False,
        norm_affine=True,
        pointnet_base_channels=32,
        pointnet_channels=(32, 64, 128, 256, 256, 128, 96, 96),
        pointnet_layers=(2, 2, 2, 2, 2, 2, 2, 2),
        stride=(4, 4, 4, 4),
        nsample=(24, 24, 24, 24),
        geo_pool_max_size=(32, 64, 128, 256),
        ca_out_channels=(16, 32, 64, 128),
        thresold=0.9,
        update_rate=0.1,
        grid_size=0.02,
        sensors=("RGB-D", "LiDAR")
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=96,
    pointnet_out_channels=96,
    context_channels=256,
    conditions=("ScanNet", "S3DIS", "SemanticKITTI"),
    num_classes=(20, 13, 19),
)

# scheduler settings
epoch = 800
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
    num_classes=20,
    ignore_index=-1,
    names=[
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
        "cabinet",
        "pillow",
        "counter",
        "desk",
        "dresser",
        "clothes",
        "tv",
    ],
    train=dict(
        type="ConcatDataset",
        datasets=[
            # ScanNet
            dict(
                type="ScanNetDataset",
                split="train",
                data_root="data/scannet",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
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
                    # dict(type="PointPatch", grid_size=1.2, point_max=20000, feat_keys=("coord", "color")),
                    dict(
                        type="GridSample",
                        grid_size=0.1,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                        keys=("coord", "color", "segment"),
                        return_point_patch=True,
                        point_patch_feat_key=("coord", "color"),
                        side=1.2,
                    ),
                    dict(type="SphereCrop", point_max=200000, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ShufflePoint"),
                    dict(type="Add", keys_dict={"condition": "ScanNet"}),
                    dict(type="Add", keys_dict={"sensor": "RGB-D"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition", "sensor", "point_patch_coord", "point_patch_label", "point_patch_feat", "point_patch_offset"),
                        feat_keys=("coord", "color"),
                    ),
                ],
                test_mode=False,
                loop=2,  # sampling weight
            ),
        ],
    ),
    val=dict(
        type="ScanNetDataset",
        split="val",
        data_root="data/scannet",
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(type="PointPatch", grid_size=0.8, point_max=20000, feat_keys=("coord", "color")),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "segment"),
                return_grid_coord=True,
                return_point_patch=True,
                point_patch_feat_key=("coord", "color"),
                side=1.2,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Add", keys_dict={"condition": "ScanNet"}),
            dict(type="Add", keys_dict={"sensor": "RGB-D"}),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "condition", "sensor","point_patch_coord", "point_patch_feat", "point_patch_offset"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type="ScanNetDataset",
        split="val",
        data_root="data/scannet",
        transform=[
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "color"),
                return_point_patch=True,
                point_patch_feat_key=("coord", "color"),
                side=1.2,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="Add", keys_dict={"condition": "ScanNet"}),
                dict(type="Add", keys_dict={"sensor": "RGB-D"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition", "sensor", 
                          "point_patch_coord", "point_patch_feat", "point_patch_offset"),
                    feat_keys=("coord", "color"),
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
