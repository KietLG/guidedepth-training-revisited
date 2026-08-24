"""
Evaluates test set performance WITHOUT horizontal flip Test-Time Augmentation (TTA).
Useful for measuring pure single-pass performance.
"""
import time
import os
import argparse
import torch
import torchvision
from data import datasets
from model import loader
from metrics import AverageMeter, Result
from data import transforms

max_depths = {
    'kitti': 80.0,
    'nyu': 10.0,
    'nyu_reduced': 10.0,
}
nyu_res = {
    'full': (480, 640),
    'half': (240, 320),
    'mini': (224, 224)}
kitti_res = {
    'full': (384, 1280),
    'half': (192, 640)}
resolutions = {
    'nyu': nyu_res,
    'nyu_reduced': nyu_res,
    'kitti': kitti_res}
crops = {
    'kitti': [128, 381, 45, 1196],
    'nyu': [20, 460, 24, 616],
    'nyu_reduced': [20, 460, 24, 616]}

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('yes', 'true', 't', 'y', '1')

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate depth model without horizontal flip TTA')
    parser.add_argument('--test_path', type=str, default='./dataset/test/official', help='Path to test data')
    parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu', 'nyu_reduced'], default='nyu_reduced')
    parser.add_argument('--resolution', type=str, choices=['full', 'half'], default='half')
    parser.add_argument('--model', type=str, default='GuideDepth')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to model weights checkpoint')
    parser.add_argument('--spatial_attention', type=str2bool, default=False)
    parser.add_argument('--skip_connection', type=str, choices=['none', 'single'], default='single')
    parser.add_argument('--deep_supervision_enable', type=str2bool, default=True)
    parser.add_argument('--localbins_enable', type=str2bool, default=True)
    parser.add_argument('--localbins_nbins', type=int, default=16)
    parser.add_argument('--save_results', type=str, default='./results')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        parser.set_defaults(**yaml_config)
        args = parser.parse_args()
    return args

def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_depth = max_depths[args.dataset]
    res = resolutions[args.dataset][args.resolution]
    crop = crops[args.dataset]

    model = loader.load_model(
        args.model, args.weights_path,
        spatial_attention=args.spatial_attention,
        skip_connection=args.skip_connection,
        deep_supervision_enable=args.deep_supervision_enable,
        localbins_enable=args.localbins_enable,
        localbins_nbins=args.localbins_nbins
    ).to(device).eval()

    test_loader = datasets.get_dataloader(
        args.dataset,
        path=args.test_path,
        split='test',
        batch_size=1,
        resolution=args.resolution,
        workers=args.num_workers
    )

    downscale_image = torchvision.transforms.Resize(res)
    to_tensor = transforms.ToTensor(test=True, maxDepth=max_depth)
    average_meter = AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            t0 = time.time()
            image, gt = data
            packed_data = {'image': image[0], 'depth': gt[0]}
            data = to_tensor(packed_data)
            image = data['image'].unsqueeze(0).to(device)
            gt = data['depth'].unsqueeze(0).to(device)

            image_down = downscale_image(image)
            data_time = time.time() - t0

            t0 = time.time()
            inv_prediction = model(image_down)
            if isinstance(inv_prediction, tuple):
                inv_prediction = inv_prediction[0]

            prediction = max_depth / torch.clamp(inv_prediction, min=1e-6)
            prediction = torch.clamp(prediction, max_depth / 100, max_depth)
            gpu_time = time.time() - t0

            prediction_full = torch.nn.functional.interpolate(prediction, size=gt.shape[-2:], mode='bilinear', align_corners=False)

            pred_crop = prediction_full[:, :, crop[0]:crop[1], crop[2]:crop[3]]
            gt_crop = gt[:, :, crop[0]:crop[1], crop[2]:crop[3]]

            result = Result()
            result.evaluate(pred_crop.data, gt_crop.data)
            average_meter.update(result, gpu_time, data_time, image.size(0))

    avg = average_meter.average()
    print("\n=== Single-Pass Evaluation (No Flip TTA) ===")
    print(f"RMSE:   {avg.rmse:.4f}")
    print(f"MAE:    {avg.mae:.4f}")
    print(f"Delta1: {avg.delta1:.4f}")
    print(f"Delta2: {avg.delta2:.4f}")
    print(f"Delta3: {avg.delta3:.4f}")
    print(f"REL:    {avg.absrel:.4f}")
    print(f"Lg10:   {avg.lg10:.4f}")

if __name__ == '__main__':
    main()
