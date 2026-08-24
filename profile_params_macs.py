"""
Profiles Parameters and MACs (FLOPs) for the GuideDepth model.
By default, auxiliary heads (deep supervision, localbins) are set to False
to measure the exact inference model deployed at test time.
"""
import argparse
import torch

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from model.loader import load_model

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "y", "1")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    p = argparse.ArgumentParser(description="Profile GuideDepth Params and MACs")
    p.add_argument("--model", type=str, default="GuideDepth")
    p.add_argument("--resolution", type=int, nargs=2, default=[240, 320], help="H W model input resolution (default: 240 320 for half resolution)")
    p.add_argument("--spatial_attention", type=str2bool, default=False)
    p.add_argument("--skip_connection", type=str, default="single", choices=["none", "single"])
    p.add_argument("--deep_supervision_enable", type=str2bool, default=False, help="Include auxiliary deep supervision heads in parameter count")
    p.add_argument("--localbins_enable", type=str2bool, default=False, help="Include auxiliary LocalBins head in parameter count")
    p.add_argument("--localbins_nbins", type=int, default=16)
    p.add_argument("--weights_path", type=str, default="")
    args = p.parse_args()

    model = load_model(
        args.model,
        args.weights_path,
        spatial_attention=args.spatial_attention,
        skip_connection=args.skip_connection,
        deep_supervision_enable=args.deep_supervision_enable,
        localbins_enable=args.localbins_enable,
        localbins_nbins=args.localbins_nbins
    )
    model.eval()

    H, W = args.resolution
    x = torch.randn(1, 3, H, W)

    params = count_parameters(model)

    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: skip_connection={args.skip_connection}, spatial_attention={args.spatial_attention}")
    print(f"Auxiliary Heads (Eval Detached): deep_supervision={args.deep_supervision_enable}, localbins={args.localbins_enable}")
    print(f"Input resolution: {H}x{W}")
    print(f"Parameters: {params:,}  (~{params/1e6:.4f} M)")

    if THOP_AVAILABLE:
        macs, _ = profile(model, inputs=(x,), verbose=False)
        print(f"MACs:       {int(macs):,}  (~{macs/1e9:.4f} G)")
    else:
        print("Note: 'thop' package is not installed. Install via 'pip install thop' to compute MACs.")
    print("=" * 60)

if __name__ == "__main__":
    main()
