# Directly inherit the entire recipe you want to use.
_base_ = 'mmpose::body_2d_keypoint/topdown_heatmap/humanart/' \
         'td-hm_ViTPose-small_8xb64-210e_humanart-256x192.py'

# # This line is to import your own modules.
custom_imports = dict(imports=[])

# base settings
data_root = '../data/'
data_mode = 'topdown'

# hooks
custom_hooks = []


# dataset settings
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        # overwrite 'bbox_file' to None to run on ground truth
        # overwrite 'bbox_file' to a customized file to run on customized bounding boxes
        # default to run validation & testing on given bboxes
        bbox_file=None,
    ),
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    ann_file=f'{data_root}HumanArt/annotations/validation_humanart.json')
test_evaluator = val_evaluator


# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(
            checkpoint='https://download.openmmlab.com/mmpose/'
            'v1/pretrained_models/mae_pretrain_vit_small_20230913.pth'
        )
    )
)

# visualizer
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])

# # preprocessing:
# python projects/HPEIA/tools/preprocess.py
# # training:
# python tools/train.py projects/HPEIA/configs/vitpose-small_8xb64-210e_humanart-256x192.py
# # testing:
# python tools/test.py projects/HPEIA/configs/vitpose-small_8xb64-210e_humanart-256x192.py work_dirs/vitpose-small_8xb64-210e_humanart-256x192/{pth_filename}
# # visualizing:
# tensorboard --logdir work_dirs/vitpose-small_8xb64-210e_humanart-256x192/${TIMESTAMP}/vis_data
