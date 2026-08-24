import os
import sys
import argparse
import random
import numpy as np
import torch

from training import Trainer
from evaluate import Evaluater

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    file_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser(description='GuideDepth Monocular Depth Estimation')

    # Mode
    parser.set_defaults(train=False)
    parser.set_defaults(evaluate=False)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--eval', dest='evaluate', action='store_true')

    # Data
    parser.add_argument('--data_path', type=str, help='path to train data', default=os.path.join(file_dir, 'dataset/train'))
    parser.add_argument('--test_path', type=str, help='path to test data', default=os.path.join(file_dir, 'dataset/test/official'))
    parser.add_argument('--dataset', type=str, help='dataset for training', choices=['kitti', 'nyu', 'nyu_reduced'], default='nyu_reduced')
    parser.add_argument('--resolution', type=str, help='Resolution of images for training', choices=['full', 'half', 'mini'], default='half')
    parser.add_argument('--eval_mode', type=str, help='Eval mode', choices=['alhashim'], default='alhashim')

    # Model
    parser.add_argument('--model', type=str, help='name of model to train', default='GuideDepth')
    parser.add_argument('--weights_path', type=str, help='path to model weights', default='')
    parser.add_argument('--spatial_attention', type=str2bool, help='Whether to use spatial attention', default=False)
    parser.add_argument('--skip_connection', type=str, help='Multi-scale semantic skip-connection mode', choices=['none', 'single'], default='single')

    # Checkpoint
    parser.add_argument('--load_checkpoint', type=str, help='path to checkpoint', default='')
    parser.add_argument('--save_checkpoint', type=str, help='path to save checkpoints', default='./checkpoints')
    parser.add_argument('--save_results', type=str, help='path to save results', default='./results')

    # Optimization
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.0)
    parser.add_argument('--num_epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('--scheduler_step_size', type=int, help='step size of scheduler', default=15)
    parser.add_argument('--optimizer_type', type=str, choices=['adam', 'adamw'], default='adam')
    parser.add_argument('--scheduler_type', type=str, choices=['step'], default='step')

    # Loss parameters
    parser.add_argument('--loss_alpha', type=float, default=0.1, help='Weight for depth BerHu loss')
    parser.add_argument('--loss_beta', type=float, default=1.0, help='Weight for SSIM loss')
    parser.add_argument('--loss_gamma', type=float, default=1.0, help='Weight for gradient loss')
    parser.add_argument('--depth_loss_type', type=str, choices=['l1', 'berhu'], default='berhu')
    parser.add_argument('--berhu_threshold', type=float, default=0.2, help='Threshold for BerHu loss')

    # Active auxiliary techniques
    parser.add_argument('--deep_supervision_enable', type=str2bool, default=True, help='Enable Multi-Scale Deep Supervision')
    parser.add_argument('--deep_supervision_weight', type=float, default=0.1, help='Weight for deep supervision losses')
    parser.add_argument('--localbins_enable', type=str2bool, default=True, help='Enable LocalBins-lite auxiliary head')
    parser.add_argument('--localbins_weight', type=float, default=0.1, help='Weight for LocalBins-lite auxiliary loss')
    parser.add_argument('--localbins_nbins', type=int, default=16, help='Number of local depth bins')

    # CutDepth Data Augmentation
    parser.add_argument('--cutdepth_enable', type=str2bool, default=True, help='Enable CutDepth augmentation')
    parser.add_argument('--cutdepth_mode', type=str, choices=['rect', 'vertical'], default='rect')
    parser.add_argument('--cutdepth_probability', type=float, default=0.75)
    parser.add_argument('--cutdepth_max_area_ratio', type=float, default=0.25)

    # System
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--confirm_test_eval', type=str2bool, default=True)
    parser.add_argument('--eval_val_test_protocol_at_end', type=str2bool, default=False)
    parser.add_argument('--run_id', type=str, default='baseline42+skip_single+berhu0.2+cutdepth_rect0.75+localbins0.1+deep_supervision0.1')
    parser.add_argument('--resume_dir', type=str, default='')

    args = parser.parse_args()
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        parser.set_defaults(**yaml_config)
        args = parser.parse_args()

    return args


def log_combined_result(args, trainer, test_avg):
    import csv
    os.makedirs('./results', exist_ok=True)
    path = './results/combined_summary.csv'
    exists = os.path.isfile(path)

    val = trainer.best_val_metrics or {}
    row = {
        'run_id': getattr(args, 'run_id', 'unnamed'),
        'val_rmse': val.get('rmse', ''), 'val_mae': val.get('mae', ''),
        'val_delta1': val.get('delta1', ''), 'val_delta2': val.get('delta2', ''),
        'val_delta3': val.get('delta3', ''), 'val_rel': val.get('absrel', ''),
        'val_lg10': val.get('lg10', ''), 'val_t_gpu': val.get('gpu_time', ''),
        'test_rmse': '', 'test_mae': '', 'test_delta1': '', 'test_delta2': '',
        'test_delta3': '', 'test_rel': '', 'test_lg10': '', 'test_t_gpu': '',
        'results_dir': trainer.results_pth,
    }
    if test_avg is not None:
        row.update({
            'test_rmse': test_avg.rmse, 'test_mae': test_avg.mae,
            'test_delta1': test_avg.delta1, 'test_delta2': test_avg.delta2,
            'test_delta3': test_avg.delta3, 'test_rel': test_avg.absrel,
            'test_lg10': test_avg.lg10, 'test_t_gpu': test_avg.gpu_time,
        })

    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[Log] Da ghi ket qua run_id='{row['run_id']}' vao {path}")


def _append_test_only(run_id, test_avg):
    import csv
    path = './results/combined_summary.csv'
    if not os.path.exists(path):
        print("Chua co file combined_summary.csv, khong the cap nhat.")
        return
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    updated = False
    for row in rows:
        if row['run_id'] == run_id:
            row.update({
                'test_rmse': test_avg.rmse, 'test_mae': test_avg.mae,
                'test_delta1': test_avg.delta1, 'test_delta2': test_avg.delta2,
                'test_delta3': test_avg.delta3, 'test_rel': test_avg.absrel,
                'test_lg10': test_avg.lg10, 'test_t_gpu': test_avg.gpu_time,
            })
            updated = True
    if updated:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[Log] Da cap nhat test_* cho run_id='{run_id}'")
    else:
        print(f"[Log] Khong tim thay run_id='{run_id}' trong CSV de cap nhat")


def main():
    args = get_args()
    set_seed(args.seed)
    print(args)

    if args.train:
        model_trainer = Trainer(args)
        model_trainer.train()
        
        args.weights_path = os.path.join(model_trainer.results_pth, 'models', 'best_model.pth')
        args.save_results = model_trainer.results_pth
        
        if getattr(args, 'eval_val_test_protocol_at_end', False):
            print("\n--- Starting M0 Test-Aligned Validation Evaluation ---")
            from scripts.eval_val_test_protocol import main as run_val_protocol
            sys_argv_backup = sys.argv
            sys.argv = [
                'eval_val_test_protocol.py',
                '--data_path', args.data_path,
                '--weights_path', args.weights_path,
                '--model', args.model,
                '--resolution', args.resolution,
                '--dataset', args.dataset,
                '--seed', str(args.seed),
                '--val_ratio', str(args.val_ratio),
                '--spatial_attention', str(getattr(args, 'spatial_attention', False)),
                '--skip_connection', str(args.skip_connection),
                '--deep_supervision_enable', str(getattr(args, 'deep_supervision_enable', True)),
                '--localbins_enable', str(getattr(args, 'localbins_enable', True)),
                '--localbins_nbins', str(getattr(args, 'localbins_nbins', 16)),
            ]
            try:
                run_val_protocol()
            except Exception as e:
                print(f"Warning: Auto-run of eval_val_test_protocol failed: {e}")
            finally:
                sys.argv = sys_argv_backup

        test_avg = None
        if getattr(args, 'confirm_test_eval', False):
            print("\n--- Starting Evaluation on Test Set ---")
            evaluation_module = Evaluater(args)
            test_avg = evaluation_module.evaluate()
        else:
            print("\n--- Skip auto-run of Test Set Evaluation ---")

        log_combined_result(args, model_trainer, test_avg)

    elif args.evaluate:
        evaluation_module = Evaluater(args)
        test_avg = evaluation_module.evaluate()
        _append_test_only(args.run_id, test_avg)

if __name__ == '__main__':
    main()
