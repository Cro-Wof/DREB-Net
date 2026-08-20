from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, '..'))

import cv2
import numpy as np
from progress.bar import Bar
import torch

from lib.opts import opts
from lib.logger import Logger
from lib.utils.utils import AverageMeter
from lib.datasets.dataset_factory import dataset_factory
from lib.detectors.ctdet_detector import CtdetDetector as Detector
from lib.utils.debugger import color_list


def report_test_metrics(detector, avg_time_stats, save_dir, num_iters):
    """Print and save model size and end-to-end inference speed metrics."""
    total_params = sum(param.numel() for param in detector.model.parameters())
    trainable_params = sum(
        param.numel() for param in detector.model.parameters() if param.requires_grad)

    avg_latency_s = avg_time_stats['tot'].avg if num_iters > 0 else 0.0
    fps = 1.0 / avg_latency_s if avg_latency_s > 0 else 0.0
    metrics = (
        '\nTest runtime/model metrics:\n'
        '  Total parameters: {:,} ({:.3f} M)\n'
        '  Trainable parameters: {:,} ({:.3f} M)\n'
        '  Average latency: {:.3f} ms/frame\n'
        '  FPS: {:.3f}\n'
    ).format(
        total_params, total_params / 1e6,
        trainable_params, trainable_params / 1e6,
        avg_latency_s * 1000.0, fps)

    print(metrics)
    with open(os.path.join(save_dir, 'result.txt'), 'a') as f:
        f.write(metrics)


def save_detection_visualization(image, results, class_names, save_path, vis_thresh):
    """Draw predictions on the original input image and save it."""
    if image is None:
        raise ValueError('Failed to load image for visualization: {}'.format(save_path))

    show_image = image.copy()
    for cls_ind, class_name in enumerate(class_names, start=1):
        for bbox in results.get(cls_ind, []):
            score = float(bbox[4])
            if score < vis_thresh:
                continue

            x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
            x1 = max(0, min(x1, show_image.shape[1] - 1))
            y1 = max(0, min(y1, show_image.shape[0] - 1))
            x2 = max(0, min(x2, show_image.shape[1] - 1))
            y2 = max(0, min(y2, show_image.shape[0] - 1))
            color = tuple(int(value) for value in color_list[cls_ind - 1])
            label = '{} {:.2f}'.format(class_name, score)

            cv2.rectangle(show_image, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_y = max(text_h + baseline, y1)
            cv2.rectangle(
                show_image,
                (x1, text_y - text_h - baseline),
                (x1 + text_w, text_y),
                color,
                -1,
            )
            cv2.putText(
                show_image,
                label,
                (x1, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, show_image)


def maybe_save_visualization(opt, dataset, image_id):
    if not opt.save_visualizations:
        return

    img_info = dataset.coco.loadImgs(ids=[int(image_id)])[0]
    vis_dir = opt.vis_dir or os.path.join(opt.save_dir, 'visualizations')
    save_path = os.path.join(vis_dir, img_info['file_name'])
    return save_path


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func, pre_process_clip_func=None):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.sharp_img_dir = dataset.sharp_img_dir
        self.blur_img_dir = dataset.blur_img_dir
        self.temporal_context = getattr(dataset, '_temporal_context', {})
        if opt.inp_sharp_or_blur == 'sharp':
            self.img_dir = self.sharp_img_dir
        elif opt.inp_sharp_or_blur == 'blur' or opt.inp_sharp_or_blur == 'SB_deblur':
            self.img_dir = self.blur_img_dir

        self.pre_process_func = pre_process_func
        self.pre_process_clip_func = pre_process_clip_func
        self.opt = opt
    
    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        center_file_name = img_info['file_name']
        if self.opt.num_input_frames > 1:
            file_names = self.temporal_context[center_file_name]
        else:
            file_names = [center_file_name]
        frame_paths = [os.path.join(self.img_dir, name) for name in file_names]
        frames = [cv2.imread(path) for path in frame_paths]
        if any(frame is None for frame in frames):
            raise RuntimeError('failed to decode test clip for {}'.format(center_file_name))
        image = frames[len(frames) // 2]
        images, meta = {}, {}
        for scale in self.opt.test_scales:
            if self.opt.num_input_frames > 1:
                images[scale], meta[scale] = self.pre_process_clip_func(frames, scale)
            else:
                images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)


def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    
    split = 'val' if not opt.trainval else 'test-dev'
    dataset = Dataset(opt, split)
    detector = Detector(opt)
    
    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process, detector.pre_process_clip),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        ret = detector.run(pre_processed_images)
        image_id = int(img_id.numpy().astype(np.int32)[0])
        results[image_id] = ret['results']
        vis_path = maybe_save_visualization(opt, dataset, image_id)
        if vis_path is not None:
            input_image = pre_processed_images['image'][0].numpy()
            save_detection_visualization(
                input_image,
                ret['results'],
                dataset.class_name,
                vis_path,
                opt.vis_thresh,
            )
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                        ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                t, tm = avg_time_stats[t])
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)
    report_test_metrics(detector, avg_time_stats, opt.save_dir, num_iters)


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    
    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        if opt.inp_sharp_or_blur == 'sharp':
            input_dir = dataset.sharp_img_dir
        elif opt.inp_sharp_or_blur == 'blur' or opt.inp_sharp_or_blur == 'SB_deblur':
            input_dir = dataset.blur_img_dir

        if opt.num_input_frames > 1:
            file_names = dataset.get_temporal_file_names(img_info['file_name'])
            img_paths = [os.path.join(input_dir, name) for name in file_names]
            ret = detector.run(img_paths)
            img_path = img_paths[len(img_paths) // 2]
        else:
            img_path = os.path.join(input_dir, img_info['file_name'])
            ret = detector.run(img_path)
        
        results[img_id] = ret['results']
        vis_path = maybe_save_visualization(opt, dataset, img_id)
        if vis_path is not None:
            input_image = cv2.imread(img_path)
            save_detection_visualization(
                input_image,
                ret['results'],
                dataset.class_name,
                vis_path,
                opt.vis_thresh,
            )

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                        ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)
    report_test_metrics(detector, avg_time_stats, opt.save_dir, num_iters)


if __name__ == '__main__':
    opt = opts().parse()
    if opt.not_prefetch_test:
        test(opt)
        print('test')
    else:
        prefetch_test(opt)
        print('prefetch_test')
