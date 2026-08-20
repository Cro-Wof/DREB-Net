from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import torch.utils.data as data

VISDRONE_num_classes = 4
VISDRONE_class_name = ['people', 'car', 'truck', 'bus']
VISDRONE_valid_ids = [0, 1, 2, 3]


class VisDrone2019DET(data.Dataset):
    num_classes = VISDRONE_num_classes
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(VisDrone2019DET, self).__init__()
        self.sharp_data_dir = opt.sharp_data_dir
        self.blur_data_dir = opt.blur_data_dir
        if getattr(opt, 'dataset', 'visdrone') == 'visdrone_vid':
            # VisDrone-VID keeps sharp frames under one directory per
            # sequence, while the prepared DREB root keeps blurred frames
            # and COCO annotations under each split.
            self.sharp_img_dir = os.path.join(
                self.sharp_data_dir,
                'VisDrone2019-VID-{}'.format(split),
                'sequences',
            )
            self.blur_img_dir = os.path.join(
                self.blur_data_dir,
                split,
                'blur_images',
            )
            self.annot_path = os.path.join(
                self.blur_data_dir,
                split,
                'annotations_dreb4.json',
            )
            dataset_name = 'VisDrone-VID'
        else:
            self.sharp_img_dir = os.path.join(self.sharp_data_dir, 'VisDrone2019-DET-{}/images'.format(split))
            self.blur_img_dir = os.path.join(self.blur_data_dir, 'VisDrone2019-DET-{}/images'.format(split))

            if split == 'test-dev':
                self.annot_path = os.path.join(
                    '../dataset/VisDrone/Annotations/annotations' + str(VISDRONE_num_classes),
                    'annotations_VisDrone_dev.json')
            else:
                self.annot_path = os.path.join(
                    '../dataset/VisDrone/Annotations/annotations' + str(VISDRONE_num_classes),
                    'annotations_VisDrone_{}.json').format(split)
            dataset_name = 'VisDrone-DET'

        if getattr(opt, 'dataset', 'visdrone') == 'visdrone_vid':
            for required_path in (self.sharp_img_dir, self.blur_img_dir, self.annot_path):
                if not os.path.exists(required_path):
                    raise FileNotFoundError(
                        'VisDrone-VID path does not exist: {}'.format(required_path)
                    )
            
        print('annot_path:', self.annot_path)
                
        # Some prepared VisDrone-VID frames contain up to 159 valid DREB
        # objects.  Keep headroom so CTDetDataset does not silently truncate
        # annotations during training or evaluation.
        self.max_objs = 256
        self.class_name = VISDRONE_class_name
        self._valid_ids = VISDRONE_valid_ids

        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                  dtype=np.float32)
        self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing {} {} data'.format(dataset_name, split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        if (
            getattr(opt, 'dataset', 'visdrone') == 'visdrone_vid'
            and getattr(opt, 'max_frames_per_sequence', 0) > 0
        ):
            self.images = self._limit_frames_per_sequence(
                self.images,
                opt.max_frames_per_sequence,
            )
            print(
                'VisDrone-VID prefix frame limit: first {} consecutive frames per sequence; '
                'using {} images'.format(opt.max_frames_per_sequence, len(self.images))
            )
        self._temporal_context = {}
        if getattr(opt, 'num_input_frames', 1) > 1:
            if getattr(opt, 'dataset', 'visdrone') != 'visdrone_vid':
                raise ValueError('multi-frame input is currently supported only for visdrone_vid')
            if opt.num_input_frames != 3:
                raise ValueError('the first multi-frame implementation supports exactly 3 frames')
            self._build_temporal_context()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _limit_frames_per_sequence(self, image_ids, max_frames):
        """Keep the first deterministic consecutive frames per VID sequence."""
        grouped = {}
        for image_id in image_ids:
            file_name = self.coco.imgs[image_id]['file_name']
            sequence_id = os.path.dirname(file_name)
            grouped.setdefault(sequence_id, []).append(image_id)

        selected = []
        for sequence_ids in grouped.values():
            # annotations_dreb4.json is generated in frame-number order.
            # Taking the prefix preserves temporal continuity and keeps the
            # exact same sample set across runs.
            selected.extend(sequence_ids[:max_frames])
        return selected

    @staticmethod
    def _frame_sort_key(file_name):
        stem = os.path.splitext(os.path.basename(file_name))[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    def _build_temporal_context(self):
        """Build [previous, center, next] inside the selected VID subset."""
        grouped = {}
        for image_id in self.images:
            file_name = self.coco.imgs[image_id]['file_name']
            sequence_id = os.path.dirname(file_name)
            grouped.setdefault(sequence_id, []).append(file_name)

        for file_names in grouped.values():
            ordered = sorted(file_names, key=self._frame_sort_key)
            for index, file_name in enumerate(ordered):
                previous = ordered[max(0, index - 1)]
                following = ordered[min(len(ordered) - 1, index + 1)]
                self._temporal_context[file_name] = [previous, file_name, following]

    def get_temporal_file_names(self, file_name):
        """Return ordered clip file names for a center-frame file."""
        if not self._temporal_context:
            return [file_name]
        try:
            return self._temporal_context[file_name]
        except KeyError as exc:
            raise KeyError(
                'frame is not part of the selected VID subset: {}'.format(file_name)
            ) from exc

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out  = list(map(self._to_float, bbox[0:4]))

                    detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                            extreme_points = list(map(self._to_float, bbox[5:13]))
                            detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), 
                                open('{}/results.json'.format(save_dir), 'w'))
    
    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        # Evaluate exactly the images used by this dataset instance.  This is
        # important for VisDrone-VID prefix experiments, where self.images may
        # contain only the first N consecutive frames of each sequence.
        coco_eval.params.imgIds = list(self.images)
        coco_eval.evaluate()
        coco_eval.accumulate()
        # coco_eval.summarize()	#原始是这一行，为了保存结果到文本使用下面的代码

        ################### 保存结果到本地
        import io
        import sys
        # 重定向输出到一个字符串流
        old_stdout = sys.stdout
        sys.stdout = my_stdout = io.StringIO()
        # 调用 summarize 方法
        coco_eval.summarize()
        # 恢复标准输出
        sys.stdout = old_stdout
        # 获取方法输出
        results = my_stdout.getvalue()
        print(results)
        # 将输出写入到文本文件
        with open(os.path.join(save_dir, 'result.txt'), 'a') as f:
            f.write(results)
            f.write('\n')
