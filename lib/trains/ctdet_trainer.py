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
from models.utils import _sigmoid, _transpose_and_gather_feat
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

    def _localization_quality_target(self, pred_wh, pred_reg, target_wh,
                                     target_reg, ind, output_width):
        """Compute detached IoU targets at the annotated center locations."""
        grid_x = (ind % output_width).float()
        grid_y = (ind // output_width).float()

        pred_cx = grid_x + pred_reg[..., 0]
        pred_cy = grid_y + pred_reg[..., 1]
        target_cx = grid_x + target_reg[..., 0]
        target_cy = grid_y + target_reg[..., 1]
        pred_w = pred_wh[..., 0].clamp(min=1e-3)
        pred_h = pred_wh[..., 1].clamp(min=1e-3)
        target_w = target_wh[..., 0].clamp(min=1e-3)
        target_h = target_wh[..., 1].clamp(min=1e-3)

        pred_x1, pred_x2 = pred_cx - pred_w / 2, pred_cx + pred_w / 2
        pred_y1, pred_y2 = pred_cy - pred_h / 2, pred_cy + pred_h / 2
        target_x1, target_x2 = target_cx - target_w / 2, target_cx + target_w / 2
        target_y1, target_y2 = target_cy - target_h / 2, target_cy + target_h / 2
        inter_w = (torch.minimum(pred_x2, target_x2) -
                   torch.maximum(pred_x1, target_x1)).clamp(min=0)
        inter_h = (torch.minimum(pred_y2, target_y2) -
                   torch.maximum(pred_y1, target_y1)).clamp(min=0)
        intersection = inter_w * inter_h
        union = pred_w * pred_h + target_w * target_h - intersection
        return (intersection / union.clamp(min=1e-6)).detach().clamp(0, 1)

    def _localization_quality_loss(self, output, batch):
        if 'quality' not in output:
            raise KeyError(
                'localization_quality is enabled but the model did not '
                'return a quality map'
            )
        ind = batch['ind'].long()
        pred_quality = _sigmoid(
            _transpose_and_gather_feat(output['quality'], ind)
        ).squeeze(-1)
        pred_wh = _transpose_and_gather_feat(output['wh'], ind)
        pred_reg = _transpose_and_gather_feat(output['reg'], ind)
        target = self._localization_quality_target(
            pred_wh.detach(), pred_reg.detach(), batch['wh'], batch['reg'],
            ind, output['quality'].shape[-1]
        )
        valid = batch['reg_mask'] > 0
        if not valid.any():
            return pred_quality.sum() * 0
        return torch.nn.functional.smooth_l1_loss(
            pred_quality[valid], target[valid], reduction='mean'
        )

    def forward(self, outputs, batch, epoch, phase):
        opt = self.opt
        hm_loss, wh_loss, off_loss, deblur_loss = 0, 0, 0, 0
        localization_quality_loss = None
        for s in range(opt.num_stacks):
            if opt.inp_sharp_or_blur == 'SB_deblur' and phase == 'train':
                output, deblur_out = outputs[0][s], outputs[1]
            else:
                output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if localization_quality_loss is None:
                localization_quality_loss = output['hm'].sum() * 0

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
            if opt.localization_quality and phase == 'train':
                localization_quality_loss += self._localization_quality_loss(
                    output, batch
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
            
        if opt.localization_quality:
            loss = loss + opt.localization_quality_weight * localization_quality_loss
            loss_stats['localization_quality_loss'] = localization_quality_loss
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
        if opt.localization_quality:
            loss_states.append('localization_quality_loss')
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
