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

    def _summarize_detection_counts(self, detections, score_thresh=0.3,
                                    iou_thresh=0.5):
        """Return counts from the serialized COCO detections."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        predicted_count = 0
        ground_truth_count = 0
        per_class = {}

        def box_iou(box_a, box_b):
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union_area = area_a + area_b - inter_area
            return inter_area / union_area if union_area > 0 else 0.0

        predictions_by_category = {}
        evaluated_image_ids = set(self.images)
        for detection in detections:
            image_id = int(detection['image_id'])
            category_id = int(detection['category_id'])
            if image_id not in evaluated_image_ids or category_id not in self._valid_ids:
                continue
            if float(detection['score']) >= score_thresh:
                predictions_by_category.setdefault(category_id, []).append(
                    (float(detection['score']), image_id, detection['bbox']))

        for cls_ind, class_name in enumerate(self.class_name, start=1):
            category_id = self._valid_ids[cls_ind - 1]
            class_tp = 0
            class_fp = 0
            class_fn = 0

            predictions = []
            for score, image_id, bbox in predictions_by_category.get(category_id, []):
                x, y, width, height = bbox
                predictions.append((score, image_id,
                                    (x, y, x + width, y + height)))

            predictions.sort(key=lambda item: item[0], reverse=True)
            matched_ground_truth = {}

            for score, image_id, bbox in predictions:
                del score  # The score is only used for matching order.
                gt_boxes = []
                for ann in self.coco.imgToAnns.get(image_id, []):
                    if ann.get('category_id') != category_id or ann.get('ignore', 0):
                        continue
                    x, y, width, height = ann['bbox']
                    gt_boxes.append((x, y, x + width, y + height))

                matched = matched_ground_truth.setdefault(image_id, set())
                best_iou = iou_thresh
                best_gt_index = None
                for gt_index, gt_box in enumerate(gt_boxes):
                    if gt_index in matched:
                        continue
                    overlap = box_iou(bbox, gt_box)
                    if overlap >= best_iou:
                        best_iou = overlap
                        best_gt_index = gt_index

                if best_gt_index is None:
                    class_fp += 1
                else:
                    matched.add(best_gt_index)
                    class_tp += 1

            for image_id in self.images:
                gt_count = sum(
                    1 for ann in self.coco.imgToAnns.get(image_id, [])
                    if ann.get('category_id') == category_id
                    and not ann.get('ignore', 0)
                )
                matched_count = len(matched_ground_truth.get(image_id, set()))
                class_fn += gt_count - matched_count

            class_predicted = class_tp + class_fp
            predicted_count += class_predicted
            ground_truth_count += class_tp + class_fn
            true_positives += class_tp
            false_positives += class_fp
            false_negatives += class_fn
            per_class[class_name] = (class_predicted, class_tp, class_fp, class_fn)

        precision = (true_positives / float(true_positives + false_positives)
                     if true_positives + false_positives else 0.0)
        recall = (true_positives / float(true_positives + false_negatives)
                  if true_positives + false_negatives else 0.0)
        f1 = (2.0 * precision * recall / (precision + recall)
              if precision + recall else 0.0)

        lines = [
            '\nDetection metrics (score >= {:.2f}, IoU >= {:.2f}):'.format(
                score_thresh, iou_thresh),
            '  Predicted boxes: {}'.format(predicted_count),
            '  Ground-truth boxes: {}'.format(ground_truth_count),
            '  TP: {} | FP: {} | FN: {}'.format(
                true_positives, false_positives, false_negatives),
            '  Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}'.format(
                precision, recall, f1),
            '  Per class:',
        ]
        for class_name in self.class_name:
            class_predicted, class_tp, class_fp, class_fn = per_class[class_name]
            class_precision = (class_tp / float(class_tp + class_fp)
                               if class_tp + class_fp else 0.0)
            class_recall = (class_tp / float(class_tp + class_fn)
                            if class_tp + class_fn else 0.0)
            lines.append(
                '    {}: predicted={} TP={} FP={} FN={} precision={:.4f} recall={:.4f}'.format(
                    class_name, class_predicted, class_tp, class_fp, class_fn,
                    class_precision, class_recall))
        return '\n'.join(lines) + '\n'
    
    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        with open('{}/results.json'.format(save_dir), 'r') as f:
            serialized_detections = json.load(f)
        detection_metrics = self._summarize_detection_counts(
            serialized_detections, score_thresh=0.3, iou_thresh=0.5)
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
        print(detection_metrics)
        # 将输出写入到文本文件
        with open(os.path.join(save_dir, 'result.txt'), 'a') as f:
            f.write(results)
            f.write('\n')
            f.write(detection_metrics)
