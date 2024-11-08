# Copyright (c) OpenMMLab. All rights reserved.
'''
This file is based on mmpose/datasets/datasets/body/humanart21_dataset.py,
rewritten for joint training with HumanArt21 and MSCOCO
'''

from mmpose.datasets.datasets import HumanArt21Dataset
from mmpose.registry import DATASETS
import numpy as np
from typing import Optional
import copy


@DATASETS.register_module()
class HumanArt21CocoDataset(HumanArt21Dataset):
    '''
    joint dataset of HumanArt and MSCOCO, inherted from HumanArt21Dataset
    '''

    METAINFO: dict = dict(from_file='configs/_base_/datasets/humanart21.py')

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        '''Rewritten to handle coco-format annotation with 17 keypoints 
        and convert it into the 21 keypoints format used by HumanArt21Dataset.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict/None: Parsed instance annotation
        '''

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann:
            return None

        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        # this part rewritten to ensure compatibility of HumanArt21 with MSCOCO
        if 'keypoints_21' in ann:
            _keypoints = np.array(
                ann['keypoints_21'], dtype=np.float32).reshape(1, -1, 3)
            keypoints = _keypoints[..., :2]
            keypoints_visible = np.minimum(1, _keypoints[..., 2])
        else:
            _keypoints = np.array(
                ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
            keypoints = np.concatenate(
                (_keypoints[..., :2], np.zeros((1, 4, 2))), axis=1,)
            keypoints_visible = np.concatenate(
                (np.minimum(1, _keypoints[..., 2]), np.zeros((1, 4))), axis=1,)

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': ann['category_id'],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info
