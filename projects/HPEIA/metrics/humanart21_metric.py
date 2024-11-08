# Copyright (c) OpenMMLab. All rights reserved.
# HumanArt21Metric:
# Adapter class to evaluate models with CocoMetric settings on HumanArt21 dataset
from typing import Dict, Optional
from mmpose.registry import METRICS
from mmpose.evaluation.metrics import CocoMetric


@METRICS.register_module()
class HumanArt21Metric(CocoMetric):
    def __init__(self,
                 ann_file: Optional[str] = None,
                 use_area: bool = True,
                 iou_type: str = 'keypoints',
                 score_mode: str = 'bbox_keypoint',
                 keypoint_score_thr: float = 0.2,
                 nms_mode: str = 'oks_nms',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 pred_converter: Dict = None,
                 gt_converter: Dict = None,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        if ann_file is None:
            raise TypeError('ann_file not given in HumanArt21Metric')
        super().__init__(
            ann_file, use_area, iou_type, score_mode, keypoint_score_thr, nms_mode, nms_thr, format_only,
            pred_converter, gt_converter, outfile_prefix, collect_device, prefix
        )
        for v in self.coco.anns.values():
            if 'keypoints_21' in v:
                v['keypoints'] = \
                    v.pop('keypoints_21')
                v['num_keypoints'] = \
                    v.pop('num_keypoints_21')
