"""
Generates qualitative comparison figures (RGB, Ground Truth, Predicted Depth Map, Error Heatmap)
for paper figures (e.g. Figure 5).
"""
import os
import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from data import datasets
from model import loader
from data import transforms

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('yes', 'true', 't', 'y', '1')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Qualitative Comparison Figures for Paper")
    parser.add_argument('--weights_path', type=str, required=True, help='Path to trained model weights checkpoint')
    parser.add_argument('--test_path', type=str, default='./dataset/test/official', help='Path to test dataset')
    parser.add_argument('--dataset', type=str, default='nyu_reduced', choices=['nyu_reduced', 'nyu', 'kitti'])
    parser.add_argument('--resolution', type=str, default='half', choices=['full', 'half'])
    parser.add_argument('--output_dir', type=str, default='./qualitative_results', help='Output directory for generated figures')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of test samples to visualize')
    parser.add_argument('--spatial_attention', type=str2bool, default=False)
    parser.add_argument('--skip_connection', type=str, default='single')
    parser.add_argument('--deep_supervision_enable', type=str2bool, default=True)
    parser.add_argument('--localbins_enable', type=str2bool, default=True)
    parser.add_argument('--localbins_nbins', type=int, default=16)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    max_depth = 10.0 if 'nyu' in args.dataset else 80.0
    res_dict = {'full': (480, 640), 'half': (240, 320)}
    model_res = res_dict[args.resolution]

    model = loader.load_model(
        'GuideDepth', args.weights_path,
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
        workers=1
    )

    downscale_fn = torchvision.transforms.Resize(model_res)
    to_tensor = transforms.ToTensor(test=True, maxDepth=max_depth)

    crop = [20, 460, 24, 616] if 'nyu' in args.dataset else [128, 381, 45, 1196]

    print(f"Generating qualitative figures for {min(args.num_samples, len(test_loader.dataset))} samples in '{args.output_dir}'...")

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= args.num_samples:
                break

            image, gt = data
            packed_data = {'image': image[0], 'depth': gt[0]}
            data_t = to_tensor(packed_data)
            image_t = data_t['image'].unsqueeze(0).to(device)
            gt_t = data_t['depth'].unsqueeze(0).to(device)

            image_down = downscale_fn(image_t)
            inv_pred = model(image_down)
            if isinstance(inv_pred, tuple):
                inv_pred = inv_pred[0]

            pred_metric = max_depth / torch.clamp(inv_pred, min=1e-6)
            pred_metric = torch.clamp(pred_metric, max_depth / 100.0, max_depth)

            pred_full = torch.nn.functional.interpolate(pred_metric, size=gt_t.shape[-2:], mode='bilinear', align_corners=False)

            img_np = image_t[0].permute(1, 2, 0).cpu().numpy()
            gt_np = gt_t[0, 0].cpu().numpy()
            pred_np = pred_full[0, 0].cpu().numpy()

            # Apply crop
            img_crop = img_np[crop[0]:crop[1], crop[2]:crop[3]]
            gt_crop = gt_np[crop[0]:crop[1], crop[2]:crop[3]]
            pred_crop = pred_np[crop[0]:crop[1], crop[2]:crop[3]]
            error_crop = np.abs(gt_crop - pred_crop)

            # Save qualitative figure (4 subplots)
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_crop)
            axes[0].set_title("RGB Input")
            axes[0].axis('off')

            axes[1].imshow(gt_crop, cmap='viridis')
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')

            axes[2].imshow(pred_crop, cmap='viridis')
            axes[2].set_title("GuideDepth Output")
            axes[2].axis('off')

            im_err = axes[3].imshow(error_crop, cmap='Reds', vmin=0.0, vmax=1.0)
            axes[3].set_title("Error Heatmap (|GT - Pred|)")
            axes[3].axis('off')

            plt.tight_layout()
            save_path = os.path.join(args.output_dir, f"qualitative_sample_{i+1:03d}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            print(f" Saved: {save_path}")

    print("Qualitative figure generation completed successfully.")

if __name__ == '__main__':
    main()
