'''
Convert Bizarre Pose Dataset into MSCOCO format
'''
import os
import json
import numpy as np
import cv2

data_root = 'data/'

img_src_dir = 'bizarre_pose/bizarre_pose_dataset/raw/images/'
img_dst_dir = 'bizarre_pose/bizarre_pose_dataset/preprocessed/images/'
if not os.path.exists(f'{data_root}{img_dst_dir}'):
    os.makedirs(f'{data_root}{img_dst_dir}')
    img_paths = os.listdir(f'{data_root}{img_src_dir}')
    img_paths = [p for p in img_paths if p.endswith('.png')]
    total = len(img_paths)
    for i, img_path in enumerate(img_paths, start=1):
        if i % 10 == 0:
            print(f'{i:4d}/{total}')
        img_id = img_path.rstrip('.png')
        img = cv2.imread(f'{data_root}{img_src_dir}{img_path}')
        cv2.imwrite(f'{data_root}{img_dst_dir}{img_path}', img)
        pass


ann_src_dir = 'bizarre_pose/bizarre_pose_dataset/raw/'
ann_dst_dir = 'bizarre_pose/bizarre_pose_dataset/preprocessed/'
filter_dir = 'bizarre_pose/bizarre_pose_dataset/_filters/'
with open(f'{data_root}{ann_src_dir}annotations.json') as f:
    anns_all = json.load(f)

idx = 0
catagories = [
    {
        'supercategory': 'person',
        'id': 1,
        'name': 'person',
        'keypoints': [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ],
        'skeleton': [
            [16, 14], [14, 12], [17, 15], [15, 13],
            [12, 13], [6, 12],  [7, 13],  [6, 7],   [6, 8],
            [7, 9],   [8, 10],  [9, 11],  [2, 3],   [1, 2],
            [1, 3],   [2, 4],   [3, 5],   [4, 6],   [5, 7]
        ]
    }
]
for split in ('train', 'val', 'test'):
    with open(f'{data_root}{filter_dir}accountably_{split}.csv') as f:
        keys = [k.strip() for k in f.readlines()]
    imgs, anns = [], []

    for key in keys:
        ann = anns_all[key]
        keypoints = [
            [*(ann['keypoints'][name][::-1]), 1] for name in
            (
                'nose', 'eye_left', 'eye_right', 'ear_left', 'ear_right',
                'shoulder_left', 'shoulder_right', 'elbow_left', 'elbow_right',
                'wrist_left', 'wrist_right', 'hip_left', 'hip_right',
                'knee_left', 'knee_right', 'ankle_left', 'ankle_right',
                # 'body_upper', 'neck_base', 'head_base', 'nose_root',
                # 'trapezium_left', 'trapezium_right', 'tiptoe_left', 'tiptoe_right',
            )
        ]
        keypoints = sum(keypoints, start=[])
        bbox = ann['bbox'][0] + ann['bbox'][1]
        y1, x1, y2, x2 = bbox
        w, h = x2 - x1, y2 - y1
        bbox = [x1, y1, w, h]
        area = np.clip(w * h * 0.53, a_min=1.0, a_max=None)
        img_h, img_w = ann['size']

        imgs.append({
            'file_name': f'{img_dst_dir}{key}.png',
            "height": img_h,
            "width": img_w,
            'id': int(key),
        })
        anns.append({
            'iscrowd': 0,
            'keypoints': keypoints,
            'image_id': int(key),
            'area': area,
            'bbox': bbox,
            'category_id': 1,
            'id': idx,
        })
        idx += 1

    with open(f'{data_root}{ann_dst_dir}ann_{split}.json', 'w') as f:
        json.dump(
            {
                'images': imgs,
                'annotations': anns,
                # from MSCOCO
                'categories': catagories,
            },
            f,
        )
    pass

print('Preprocess completed.')
