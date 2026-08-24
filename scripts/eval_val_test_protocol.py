import os
import sys
import argparse
import time

import torch
import torchvision
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.nyu_reduced import depthDatasetDirectory
from model import loader
from metrics import AverageMeter, Result

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Validation Set using strict Test Set Protocol (M0)")
    parser.add_argument('--data_path', type=str, default='./dataset/train', help='Path to NYU dataset directory')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to model weights checkpoint')
    parser.add_argument('--model', type=str, default='GuideDepth', help='Model architecture')
    parser.add_argument('--resolution', type=str, default='half', choices=['full', 'half'], help='Model operating resolution')
    parser.add_argument('--dataset', type=str, default='nyu_reduced', help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Val split random seed')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Validation split ratio')
    
    # Model plumbing parameters
    parser.add_argument('--spatial_attention', type=str2bool, default=False)
    parser.add_argument('--skip_connection', type=str, default='single')
    parser.add_argument('--deep_supervision_enable', type=str2bool, default=True)
    parser.add_argument('--localbins_enable', type=str2bool, default=True)
    parser.add_argument('--localbins_nbins', type=int, default=16)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"=== M0 DIAGNOSTIC: VAL SET EVALUATION UNDER TEST PROTOCOL ===")
    print(f"Model weights: {args.weights_path}")
    print(f"Resolution: {args.resolution}")
    print(f"Seed: {args.seed}, Val Ratio: {args.val_ratio}")

    # 1. Load Model
    model = loader.load_model(
        args.model, args.weights_path,
        spatial_attention=args.spatial_attention,
        skip_connection=args.skip_connection,
        deep_supervision_enable=args.deep_supervision_enable,
        localbins_enable=args.localbins_enable,
        localbins_nbins=args.localbins_nbins
    ).to(device)
    model.eval()

    # 2. Load Validation Dataset WITHOUT transform (keeps full resolution 480x640)
    val_dataset = depthDatasetDirectory(
        root=args.data_path,
        split='val',
        seed=args.seed,
        val_ratio=args.val_ratio,
        transform=None
    )
    print(f"Total Validation Samples: {len(val_dataset)}")

    res_dict = {'full': (480, 640), 'half': (240, 320)}
    model_res = res_dict[args.resolution]
    downscale_fn = torchvision.transforms.Resize(model_res)

    average_meter = AverageMeter()

    for idx in range(len(val_dataset)):
        t0 = time.time()
        sample = val_dataset[idx]
        image_np, gt_np = sample['image'], sample['depth']

        image_pil = Image.fromarray(np.uint8(image_np))
        image_resized_pil = downscale_fn(image_pil)
        
        image_t = torch.from_numpy(np.array(image_resized_pil).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).to(device)

        data_time = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            inv_prediction = model(image_t)
            if isinstance(inv_prediction, tuple):
                inv_prediction = inv_prediction[0]

        # Inverse depth norm without extra clamping (exact paper logic)
        pred_metric = 10.0 / inv_prediction

        prediction_full = torch.nn.functional.interpolate(pred_metric, size=(480, 640), mode='bilinear', align_corners=False)
        gpu_time = time.time() - t0

        pred_eval = prediction_full.data
        gt_eval = gt_t.data

        # NYU Crop
        crop = [20, 460, 24, 616]
        pred_eval = pred_eval[:, :, crop[0]:crop[1], crop[2]:crop[3]]
        gt_eval = gt_eval[:, :, crop[0]:crop[1], crop[2]:crop[3]]

        result = Result()
        result.evaluate(pred_eval, gt_eval)
        average_meter.update(result, gpu_time, data_time, 1)

    avg = average_meter.average()
    print("\n" + "=" * 50)
    print(f"M0 VAL EVALUATION RESULT (Test Protocol Aligned):")
    print(f"RMSE:   {avg.rmse:.4f}")
    print(f"MAE:    {avg.mae:.4f}")
    print(f"Delta1: {avg.delta1:.4f}")
    print(f"Delta2: {avg.delta2:.4f}")
    print(f"Delta3: {avg.delta3:.4f}")
    print(f"REL:    {avg.absrel:.4f}")
    print(f"Lg10:   {avg.lg10:.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
