# Directly inherit the entire recipe you want to use.
_base_ = 'vitpose-small_8xb64-210e_humanart-256x192.py'
# This line is to import your own modules.

# base settings
data_root = '../data/'
data_mode = 'topdown'

# hooks
custom_hooks = []

# runtime
train_cfg = dict(max_epochs=1000, val_interval=10)

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=1000,
        milestones=[800, 950],
        gamma=0.1,
        by_epoch=True)
]

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(256, 192), heatmap_size=(64, 48), sigma=2)

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(_delete_=True, type='Kaiming'),
        img_size=(256, 192),
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='checkpoint/td-hm_ViTPose-small_8xb64-210e_humanart-256x192-5cbe2bfc_20230611.pth'
    )
)

# dataset settings
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='bizarre_pose/bizarre_pose_dataset/preprocessed/ann_train.json',
        data_prefix=dict(img=''),
        pipeline=_base_['train_pipeline']
    ))
val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='bizarre_pose/bizarre_pose_dataset/preprocessed/ann_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=_base_['val_pipeline']
    ))
test_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='bizarre_pose/bizarre_pose_dataset/preprocessed/ann_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=_base_['val_pipeline']
    ))


# # evaluators
val_evaluator = dict(
    ann_file=f'{data_root}bizarre_pose/bizarre_pose_dataset/preprocessed/ann_val.json'
)
test_evaluator = dict(
    ann_file=f'{data_root}bizarre_pose/bizarre_pose_dataset/preprocessed/ann_test.json'
)

# # preprocessing:
# python projects/HPEIA/tools/preprocess.py
# # training:
# python tools/train.py projects/HPEIA/configs/vitpose-small_8xb64-210e_bizarrepose-256x192.py
# # testing:
# python tools/test.py projects/HPEIA/configs/vitpose-small_8xb64-210e_bizarrepose-256x192.py work_dirs/vitpose-small_8xb64-210e_bizarrepose-256x192/{pth_filename}
# # visualizing:
# tensorboard --logdir work_dirs/vitpose-small_8xb64-210e_bizarrepose-256x192/${TIMESTAMP}/vis_data
