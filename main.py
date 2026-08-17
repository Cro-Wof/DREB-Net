from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime, timedelta

import torch
import torch.utils.data
from torch.optim import lr_scheduler
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.utils.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.ctdet_trainer import CtdetTrainer as Trainer
from lib.utils.general import one_cycle, one_flat_cycle


def format_duration(seconds):
    """Format a duration compactly for epoch-level training summaries."""
    return str(timedelta(seconds=max(0, int(seconds))))

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr) # (SGD=1E-2, Adam=1E-3)

    # Scheduler
    lrf = 0.001
    # lf = one_cycle(1, lrf, opt.num_epochs)  # cosine 1->hyp['lrf']
    # lf = one_flat_cycle(1, lrf, opt.num_epochs)  # flat cosine 1->hyp['lrf']        
    # lf = lambda x: 1.0
    lf = lambda x: (1 - x / opt.num_epochs) * (1.0 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    trainer = Trainer(opt, model, optimizer, scheduler)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'), 
        batch_size=1, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
 
    if opt.test:
        _, preds = trainer.val(0, val_loader, logger)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'), 
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    best = 1e10
    training_start_time = time.time()
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        epoch_start_time = time.time()
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader, logger)
        logger.write('epoch: {} |'.format(epoch))

        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        # save the last 10 epoch
        if epoch >= 190 or (epoch%10 == 0 and epoch != 0):
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)

        log_dict_val = None
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader, logger)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))

            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
        logger.write('\n')

        # Print one epoch-level summary after validation (or after training if
        # validation is disabled).  The loss values here are the weighted
        # total losses, not the individual loss components.
        elapsed_seconds = time.time() - training_start_time
        epoch_seconds = time.time() - epoch_start_time
        average_epoch_seconds = elapsed_seconds / epoch
        remaining_seconds = average_epoch_seconds * (opt.num_epochs - epoch)
        estimated_finish = datetime.now() + timedelta(seconds=remaining_seconds)
        val_loss_text = (
            '{:.4f}'.format(log_dict_val['loss'])
            if log_dict_val is not None else 'N/A'
        )
        best_text = (
            '{:.4f}'.format(best)
            if best < 1e10 else 'N/A'
        )
        epoch_summary = (
            '[Epoch {}/{}] train_loss={:.4f} | val_loss={} | best_val_loss={} | '
            'epoch_time={} | elapsed={} | ETA={} | finish~{}'
        ).format(
            epoch,
            opt.num_epochs,
            log_dict_train['loss'],
            val_loss_text,
            best_text,
            format_duration(epoch_seconds),
            format_duration(elapsed_seconds),
            format_duration(remaining_seconds),
            estimated_finish.strftime('%Y-%m-%d %H:%M:%S'),
        )
        print(epoch_summary, flush=True)
        logger.write(epoch_summary + '\n')


    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
