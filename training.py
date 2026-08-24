import time
import os
import sys
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import datasets
from model import loader
from losses import Depth_Loss
from metrics import AverageMeter, Result
import torch.nn.functional as F

class TeeLogger(object):
    def __init__(self, filepath, terminal):
        self.terminal = terminal
        self.log = open(filepath, 'a')

    def write(self, message):
        self.terminal.write(message)
        if '\r' not in message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    def fileno(self):
        if hasattr(self.terminal, 'fileno'):
            return self.terminal.fileno()
        raise OSError("fileno not supported")

max_depths = {
    'kitti': 80.0,
    'nyu': 10.0,
    'nyu_reduced': 10.0,
}

class Trainer():
    def __init__(self, args):
        self.args = args
        self.debug = True

        from datetime import datetime, timedelta, timezone
        vietnam_tz = timezone(timedelta(hours=7))

        resume_dir = getattr(args, 'resume_dir', '')
        if resume_dir:
            self.results_pth = resume_dir
            self.checkpoint_pth = os.path.join(self.results_pth, 'models')
            if not os.path.isdir(self.checkpoint_pth):
                raise FileNotFoundError(f"Khong tim thay thu muc checkpoint: {self.checkpoint_pth}")
            print(f"[Resume] Tiep tuc training tu: {self.results_pth}")
        else:
            timestamp = datetime.now(vietnam_tz).strftime("%Y_%m_%d_%H_%M_%S")
            self.results_pth = os.path.join(args.save_results, timestamp)
            self.checkpoint_pth = os.path.join(self.results_pth, 'models')
            os.makedirs(self.checkpoint_pth, exist_ok=True)

        log_file_path = os.path.join(self.results_pth, 'training_log.txt')
        self.tee_logger = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = self.tee_logger
        sys.stderr = self.tee_logger

        config_path = os.path.join(self.results_pth, 'configuration.txt')
        with open(config_path, 'w') as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

        self.epoch = 0
        self.val_losses = []
        self.best_val_rmse = float('inf')
        self.best_val_metrics = None
        self.run_id = getattr(args, 'run_id', 'unnamed')
        self.max_epochs = args.num_epochs
        self.dataset = args.dataset
        self.maxDepth = max_depths[args.dataset]
        print('Maximum Depth of Dataset: {}'.format(self.maxDepth))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        spatial_attention = getattr(args, 'spatial_attention', False)
        skip_connection = getattr(args, 'skip_connection', 'single')

        self.model = loader.load_model(
            args.model,
            args.weights_path,
            spatial_attention=spatial_attention,
            skip_connection=skip_connection,
            deep_supervision_enable=getattr(args, 'deep_supervision_enable', True),
            localbins_enable=getattr(args, 'localbins_enable', True),
            localbins_nbins=getattr(args, 'localbins_nbins', 16)
        )
        self.model.to(self.device)

        self.train_loader = datasets.get_dataloader(
            args.dataset,
            path=args.data_path,
            split='train',
            augmentation=args.eval_mode,
            batch_size=args.batch_size,
            resolution=args.resolution,
            workers=args.num_workers,
            cutdepth_enable=getattr(args, 'cutdepth_enable', True),
            cutdepth_mode=getattr(args, 'cutdepth_mode', 'rect'),
            cutdepth_probability=getattr(args, 'cutdepth_probability', 0.75),
            cutdepth_max_area_ratio=getattr(args, 'cutdepth_max_area_ratio', 0.25),
            seed=getattr(args, 'seed', 42),
            val_ratio=getattr(args, 'val_ratio', 0.05)
        )

        self.val_loader = datasets.get_dataloader(
            args.dataset,
            path=args.data_path,
            split='val',
            augmentation=args.eval_mode,
            batch_size=args.batch_size,
            resolution=args.resolution,
            workers=args.num_workers,
            seed=getattr(args, 'seed', 42),
            val_ratio=getattr(args, 'val_ratio', 0.05)
        )

        alpha = getattr(args, 'loss_alpha', 0.1)
        beta = getattr(args, 'loss_beta', 1.0)
        gamma = getattr(args, 'loss_gamma', 1.0)
        depth_loss_type = getattr(args, 'depth_loss_type', 'berhu')
        berhu_threshold = getattr(args, 'berhu_threshold', 0.2)

        self.loss_func = Depth_Loss(
            alpha=alpha, beta=beta, gamma=gamma, maxDepth=self.maxDepth,
            depth_loss_type=depth_loss_type, berhu_threshold=berhu_threshold
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=getattr(args, 'weight_decay', 0.0)
        )

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.scheduler_step_size,
            gamma=0.1
        )

        if args.load_checkpoint != '':
            self.load_checkpoint(args.load_checkpoint)
        elif resume_dir:
            ckpts = [f for f in os.listdir(self.checkpoint_pth) if f.startswith('checkpoint_') and f.endswith('.pth')]
            if ckpts:
                get_epoch = lambda fn: int(fn.split('_')[1].split('.')[0])
                latest = sorted(ckpts, key=get_epoch)[-1]
                print(f"[Resume] Tu dong tai checkpoint moi nhat: {latest}")
                self.load_checkpoint(os.path.join(self.checkpoint_pth, latest))

    def train(self):
        torch.cuda.empty_cache()
        self.start_time = time.time()
        for self.epoch in range(self.epoch, self.max_epochs):
            current_time = time.strftime('%H:%M', time.localtime())
            lr_str = f"Base LR: {self.optimizer.param_groups[0]['lr']:.7f}"
            print('{} - Epoch {} - {}'.format(current_time, self.epoch, lr_str))

            self.train_loop()

            if self.val_loader is not None:
                self.val_loop()

            self.lr_scheduler.step()
            self.save_checkpoint()

        self.save_model()

    def train_loop(self):
        self.model.train()
        accumulated_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.epoch}", leave=True, file=sys.stdout, dynamic_ncols=True, mininterval=2.0)
        for i, data in enumerate(pbar):
            image, gt = self.unpack_and_move(data)

            self.optimizer.zero_grad()

            outputs = self.model(image)
            if isinstance(outputs, tuple):
                prediction, aux_outputs = outputs
            else:
                prediction = outputs
                aux_outputs = {}

            loss_value = self.loss_func(prediction, gt, image=image)

            # Deep Supervision Loss
            if getattr(self.args, 'deep_supervision_enable', True) and getattr(self.args, 'deep_supervision_weight', 0.0) > 0.0:
                ds_weight = self.args.deep_supervision_weight
                for ds_key in ['ds_1', 'ds_2']:
                    pred_lowres = aux_outputs.get(ds_key)
                    if pred_lowres is not None:
                        gt_lowres = F.interpolate(gt, size=pred_lowres.shape[-2:], mode='bilinear')
                        mask_lowres = gt_lowres > 0.0
                        ds_loss = torch.abs(pred_lowres[mask_lowres] - gt_lowres[mask_lowres]).mean()
                        loss_value = loss_value + ds_weight * ds_loss

            # LocalBins Auxiliary Head Loss
            if getattr(self.args, 'localbins_enable', True) and getattr(self.args, 'localbins_weight', 0.0) > 0.0 and 'localbins_depth' in aux_outputs:
                localbins_pred = aux_outputs['localbins_depth']
                gt_lowres_lb = F.interpolate(gt, size=localbins_pred.shape[-2:], mode='bilinear')
                gt_metric_lb = self.inverse_depth_norm(gt_lowres_lb)
                mask_lb = gt_lowres_lb > 0.0
                lb_loss = F.l1_loss(localbins_pred[mask_lb], gt_metric_lb[mask_lb])
                loss_value = loss_value + self.args.localbins_weight * lb_loss

            loss_value.backward()
            self.optimizer.step()

            loss_item = loss_value.item()
            accumulated_loss += loss_item

            pbar.set_postfix(loss=f"{loss_item:.4f}", refresh=False)

        current_time = time.strftime('%H:%M', time.localtime())
        average_loss = accumulated_loss / (len(self.train_loader.dataset) + 1)
        print('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))

    def val_loop(self):
        torch.cuda.empty_cache()
        self.model.eval()
        accumulated_loss = 0.0
        average_meter = AverageMeter()

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {self.epoch} (Test-Protocol)", leave=True, file=sys.stdout, dynamic_ncols=True, mininterval=2.0)
        with torch.no_grad():
            for i, data in enumerate(pbar):
                t0 = time.time()
                image, gt = self.unpack_and_move(data)
                data_time = time.time() - t0

                t0 = time.time()
                inv_prediction = self.model(image)
                if isinstance(inv_prediction, tuple):
                    inv_prediction = inv_prediction[0]
                prediction = self.inverse_depth_norm(inv_prediction)
                gpu_time = time.time() - t0

                gt_lowres = F.interpolate(gt, size=inv_prediction.shape[-2:], mode='bilinear')
                loss_value = self.loss_func(inv_prediction, self.depth_norm(gt_lowres), image=image)
                loss_item = loss_value.item()
                accumulated_loss += loss_item

                prediction_full = F.interpolate(prediction, size=gt.shape[-2:], mode='bilinear', align_corners=False)

                pred_eval = prediction_full.data
                gt_eval = gt.data

                H, W = gt_eval.shape[-2:]
                if self.dataset in ['nyu', 'nyu_reduced']:
                    crop = [20, 460, 24, 616]
                    pred_eval = pred_eval[:, :, crop[0]:crop[1], crop[2]:crop[3]]
                    gt_eval = gt_eval[:, :, crop[0]:crop[1], crop[2]:crop[3]]
                elif self.dataset == 'kitti':
                    crop_h0 = int(0.3324324 * H)
                    crop_h1 = int(0.91351351 * H)
                    crop_w0 = int(0.0359477 * W)
                    crop_w1 = int(0.96405229 * W)
                    pred_eval = pred_eval[:, :, crop_h0:crop_h1, crop_w0:crop_w1]
                    gt_eval = gt_eval[:, :, crop_h0:crop_h1, crop_w0:crop_w1]

                result = Result()
                result.evaluate(pred_eval, gt_eval)
                average_meter.update(result, gpu_time, data_time, image.size(0))

                pbar.set_postfix(loss=f"{loss_item:.4f}", rmse=f"{result.rmse:.4f}", refresh=False)

        avg = average_meter.average()
        current_time = time.strftime('%H:%M', time.localtime())
        average_loss = accumulated_loss / (len(self.val_loader.dataset) + 1)
        self.val_losses.append(average_loss)
        print('{} - Average Validation Loss: {:3.4f}'.format(current_time, average_loss))

        if avg.rmse < self.best_val_rmse:
            self.best_val_rmse = avg.rmse
            self.best_val_metrics = {
                'rmse': avg.rmse, 'mae': avg.mae,
                'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'absrel': avg.absrel, 'lg10': avg.lg10, 'gpu_time': avg.gpu_time,
            }
            self._save_best_model()

        print('\n*\n'
              'RMSE={average.rmse:.4f}\n'
              'MAE={average.mae:.4f}\n'
              'Delta1={average.delta1:.4f}\n'
              'Delta2={average.delta2:.4f}\n'
              'Delta3={average.delta3:.4f}\n'
              'REL={average.absrel:.4f}\n'
              'Lg10={average.lg10:.4f}\n'
              't_GPU={time:.3f}\n'.format(
              average=avg, time=avg.gpu_time))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.epoch = checkpoint['epoch']
        if 'val_losses' in checkpoint and checkpoint['val_losses'] is not None:
            self.val_losses = checkpoint['val_losses']
        if 'best_val_rmse' in checkpoint and checkpoint['best_val_rmse'] is not None:
            self.best_val_rmse = checkpoint['best_val_rmse']
            print(f"[Resume] Khoi phuc best_val_rmse = {self.best_val_rmse:.4f} tu checkpoint.")

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.checkpoint_pth, 'checkpoint_{}.pth'.format(self.epoch))
        torch.save({
            'epoch': self.epoch + 1,
            'val_losses': self.val_losses,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'best_val_rmse': self.best_val_rmse,
            'best_val_metrics': self.best_val_metrics,
        }, checkpoint_dir)
        current_time = time.strftime('%H:%M', time.localtime())
        print('{} - Model saved'.format(current_time))

    def _save_best_model(self):
        best_model_pth = os.path.join(self.checkpoint_pth, 'best_model.pth')
        torch.save(self.model.state_dict(), best_model_pth)
        print('New best model saved with validation RMSE: {:.4f}'.format(self.best_val_rmse))

    def save_model(self):
        best_model_pth = os.path.join(self.checkpoint_pth, 'best_model.pth')
        if not os.path.exists(best_model_pth):
            best_checkpoint_pth = os.path.join(self.checkpoint_pth, 'checkpoint_{}.pth'.format(self.max_epochs - 1))
            checkpoint = torch.load(best_checkpoint_pth)
            torch.save(checkpoint['model'], best_model_pth)
            print('Model saved.')

    def inverse_depth_norm(self, depth):
        zero_mask = depth == 0.0
        depth_safe = torch.clamp(depth, min=1e-6)
        depth_safe = self.maxDepth / depth_safe
        depth_safe = torch.clamp(depth_safe, self.maxDepth / 100, self.maxDepth)
        depth_safe[zero_mask] = 0.0
        return depth_safe

    def depth_norm(self, depth):
        zero_mask = depth == 0.0
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        depth[zero_mask] = 0.0
        return depth

    def unpack_and_move(self, data):
        if isinstance(data, (tuple, list)):
            image = data[0].to(self.device, non_blocking=True)
            gt = data[1].to(self.device, non_blocking=True)
            return image, gt
        if isinstance(data, dict):
            image = data['image'].to(self.device, non_blocking=True)
            gt = data['depth'].to(self.device, non_blocking=True)
            return image, gt
        print('Type not supported')
