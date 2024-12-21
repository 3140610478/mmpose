# Directly inherit the entire recipe you want to use.
_base_ = 'vitpose-small_8xb64-210e_humanart-256x192.py'

# # This line is to import your own modules.
custom_imports = dict(
    imports=[
        'projects.HPEIA.datasets',
        'projects.HPEIA.metrics',
    ]
)

# base settings
data_root = '../data/'
data_mode = 'topdown'

# hooks
custom_hooks = []


# dataset settings
train_dataloader = dict(
    dataset=dict(
        type='HumanArt21CocoDataset',
        data_root=data_root,
    )
)
val_dataloader = dict(
    dataset=dict(
        type='HumanArt21CocoDataset',
        data_root=data_root,
        # overwrite 'bbox_file' to None to run on ground truth
        # overwrite 'bbox_file' to a customized file to run on customized bounding boxes
        # default to run validation & testing on given bboxes
        bbox_file=None,
    ),
)
test_dataloader = val_dataloader

# model settings
model = dict(
    head=dict(out_channels=21),
)


# evaluators
val_evaluator = dict(
    type='HumanArt21Metric',
    ann_file=f'{data_root}HumanArt/annotations/validation_humanart.json'
)
test_evaluator = val_evaluator


# # preprocessing:
# python projects/HPEIA/tools/preprocess.py
# # training:
# python tools/train.py projects/HPEIA/configs/vitpose-small_8xb64-210e_humanart21-256x192.py
# # testing:
# python tools/test.py projects/HPEIA/configs/vitpose-small_8xb64-210e_humanart21-256x192.py work_dirs/vitpose-small_8xb64-210e_humanart21-256x192/{pth_filename}
# # visualizing:
# tensorboard --logdir work_dirs/vitpose-small_8xb64-210e_humanart21-256x192/${TIMESTAMP}/vis_data
