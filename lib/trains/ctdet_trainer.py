from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import time
from progress.bar import Bar
from lib.utils.data_parallel import DataParallel

import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path))
sys.path.append(os.path.join(current_path, '..'))
print(sys.path)

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.losses import mse_loss, ssim_loss, PerceptualLoss, Stripformer_Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.utils import AverageMeter
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss, opt):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
        self.opt = opt
    
    def forward(self, batch, phase, epoch):
        if self.opt.inp_sharp_or_blur == 'sharp': 
            outputs = self.model(batch['sharp_input'])
        elif self.opt.inp_sharp_or_blur == 'blur':
            outputs = self.model(batch['blur_input'])
        elif self.opt.inp_sharp_or_blur == 'SB_deblur':
            if getattr(self.opt, 'num_input_frames', 1) > 1:
                outputs = self.model(batch['blur_clip'], phase)
            else:
                outputs = self.model(batch['blur_input'], phase)
        loss, loss_stats = self.loss(outputs, batch, epoch, phase)
        return outputs[-1], loss, loss_stats


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                        RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
                        NormRegL1Loss() if opt.norm_wh else \
                        RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        
        if opt.inp_sharp_or_blur == 'SB_deblur' and opt.deblur_loss == 'Stripformer':
            self.deblur_loss_Stripformer = Stripformer_Loss()

        self.opt = opt

    def _temporal_detection_loss(self, temporal_logits, temporal_target,
                                 temporal_valid):
        """Focal loss for aligned previous/following feature heatmaps."""
        loss = temporal_logits.sum() * 0
        valid_branches = 0
        for neighbor_index in range(temporal_logits.size(1)):
            valid_samples = temporal_valid[:, neighbor_index] > 0
            if valid_samples.any():
                prediction = _sigmoid(
                    temporal_logits[:, neighbor_index][valid_samples]
                )
                target = temporal_target[:, neighbor_index][valid_samples]
                loss = loss + self.crit(prediction, target)
                valid_branches += 1
        if valid_branches == 0:
            return loss
        return loss / valid_branches

    def _hard_negative_loss(self, heatmap, valid_mask):
        """Penalize detached high-confidence local maxima outside GT boxes."""
        opt = self.opt
        detached = heatmap.detach()
        local_maxima = detached.eq(torch.nn.functional.max_pool2d(
            detached, kernel_size=3, stride=1, padding=1
        ))
        candidate_mask = (
            local_maxima
            & (detached >= opt.hard_negative_score_thresh)
            & (valid_mask > 0).expand_as(detached)
        )
        batch_size = heatmap.size(0)
        flattened_scores = detached.reshape(batch_size, -1)
        flattened_mask = candidate_mask.reshape(batch_size, -1)
        masked_scores = flattened_scores.masked_fill(~flattened_mask, -1)
        topk = min(opt.hard_negative_topk, masked_scores.size(1))
        top_scores, top_indices = torch.topk(masked_scores, k=topk, dim=1)
        selected = top_scores >= opt.hard_negative_score_thresh
        if not selected.any():
            return heatmap.sum() * 0
        selected_predictions = heatmap.reshape(batch_size, -1).gather(
            1, top_indices
        )
        negative_loss = -torch.log(
            (1 - selected_predictions).clamp(min=1e-4)
        ) * selected_predictions.pow(2)
        selected_weight = selected.float()
        return (negative_loss * selected_weight).sum() / (
            selected_weight.sum() + 1e-4
        )

    def forward(self, outputs, batch, epoch, phase):
        opt = self.opt
        hm_loss, wh_loss, off_loss, deblur_loss = 0, 0, 0, 0
        temporal_hm_loss, hard_negative_loss = None, None
        for s in range(opt.num_stacks):
            if opt.inp_sharp_or_blur == 'SB_deblur' and phase == 'train':
                output, deblur_out = outputs[0][s], outputs[1]
            else:
                output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if temporal_hm_loss is None:
                temporal_hm_loss = output['hm'].sum() * 0
                hard_negative_loss = output['hm'].sum() * 0

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(), 
                    batch['ind'].detach().cpu().numpy(), 
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(), 
                    batch['ind'].detach().cpu().numpy(), 
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.temporal_det_supervision and phase == 'train':
                if 'temporal_hm' not in output:
                    raise KeyError(
                        'temporal_det_supervision is enabled but the model '
                        'did not return temporal_hm'
                    )
                temporal_hm_loss += self._temporal_detection_loss(
                    output['temporal_hm'],
                    batch['temporal_hm'],
                    batch['temporal_hm_valid'],
                ) / opt.num_stacks
            if (
                    opt.hard_negative
                    and phase == 'train'
                    and epoch > opt.hard_negative_warmup_epochs):
                if 'hard_negative_valid' not in batch:
                    raise KeyError(
                        'hard_negative is enabled but the dataset did not '
                        'return hard_negative_valid'
                    )
                hard_negative_loss += self._hard_negative_loss(
                    output['hm'], batch['hard_negative_valid']
                ) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                        self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                        batch['dense_wh'] * batch['dense_wh_mask']) / 
                        mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks
            
            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.inp_sharp_or_blur == 'SB_deblur':
                # if epoch <= opt.deblur_train_end_epoch and phase == 'train':
                if (phase=='train' and opt.train_mode=='continuous' and epoch<=opt.deblur_train_end_epoch) or \
                   (phase=='train' and opt.train_mode=='interval' and epoch<=opt.deblur_train_end_epoch and epoch%2==0):
                    if opt.deblur_loss == 'mse_ssim':
                        deblur_loss += mse_loss(deblur_out, batch['sharp_input']) + ssim_loss(deblur_out, batch['sharp_input'])
                    elif opt.deblur_loss == 'Stripformer':
                        deblur_loss += self.deblur_loss_Stripformer(deblur_out, batch['sharp_input'], batch['blur_input'])
                    else:
                        raise ValueError("deblur loss not exists!!!")
                else:
                    deblur_loss = mse_loss(batch['sharp_input'], batch['sharp_input'])

        if opt.inp_sharp_or_blur == 'SB_deblur':
            if epoch <= opt.deblur_train_end_epoch and phase == 'train':
                loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.deblur_weight * deblur_loss
            else:
                loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
            loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss, 'deblur_loss': deblur_loss}
        else:
            loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
            loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
            
        if opt.temporal_det_supervision:
            loss = loss + opt.temporal_det_weight * temporal_hm_loss
            loss_stats['temporal_hm_loss'] = temporal_hm_loss
        if opt.hard_negative:
            loss = loss + opt.hard_negative_weight * hard_negative_loss
            loss_stats['hard_negative_loss'] = hard_negative_loss
        loss_stats['loss'] = loss
        return loss, loss_stats



class CtdetTrainer(object):
    def __init__(self, opt, model, optimizer=None, scheduler=None):
        self.opt = opt
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss, opt)


    def run_epoch(self, phase, epoch, data_loader, logger):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
            self.scheduler.step()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        if opt.print_iter > 0:
            # Keep Bar's index/throughput/ETA updated, but suppress its
            # per-iteration carriage-return output.  Status is printed below
            # at the requested interval instead.
            bar.update = lambda: None
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            if self.opt.inp_sharp_or_blur == 'SB_deblur':
                output, loss, loss_stats = model_with_loss(batch, phase, epoch)
            else:
                output, loss, loss_stats = model_with_loss(batch, phase, epoch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['sharp_input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                    '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            bar.next()
            if opt.print_iter > 0 and (
                iter_id % opt.print_iter == 0 or iter_id == num_iters - 1
            ):
                print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix), flush=True)
            
            if opt.debug > 0:
                self.debug(batch, output, iter_id)
            
            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        ret['lr'] = self.scheduler.get_lr()[0]
        return ret, results
    

    def train(self, epoch, data_loader, logger):
        return self.run_epoch('train', epoch, data_loader, logger)
    

    def val(self, epoch, data_loader, logger):
        return self.run_epoch('val', epoch, data_loader, logger)


    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        if opt.inp_sharp_or_blur == 'SB_deblur':
            loss_states.append('deblur_loss')
        if opt.temporal_det_supervision:
            loss_states.append('temporal_hm_loss')
        if opt.hard_negative:
            loss_states.append('hard_negative_loss')
        loss = CtdetLoss(opt)
        return loss_states, loss


    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus, 
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)


    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
            img = batch['sharp_input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1], dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1], dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)


    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
