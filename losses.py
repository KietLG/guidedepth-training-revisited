""" 
Depth Loss based on Alhashim et al.:

Ibraheem Alhashim, High Quality Monocular Depth Estimation via
Transfer Learning, https://arxiv.org/abs/1812.11941, 2018
"""

import torch
import torch.nn.functional as F
from math import exp


def berhu_loss(pred, target, valid_mask, c_ratio=0.2):
    """
    Computes the reverse Huber (BerHu) loss map.
    """
    diff = torch.abs(pred - target)
    valid_diff = diff[valid_mask]
    if len(valid_diff) == 0:
        return torch.zeros_like(diff)
    c = c_ratio * torch.max(valid_diff).item()
    if c == 0:
        return torch.zeros_like(diff)
    
    loss_map = torch.where(diff <= c, diff, (diff.pow(2) + c**2) / (2 * c))
    return loss_map


class Depth_Loss():
    def __init__(self, alpha=0.1, beta=1.0, gamma=1.0, maxDepth=10.0,
                 depth_loss_type='berhu', berhu_threshold=0.2, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.maxDepth = maxDepth
        self.depth_loss_type = depth_loss_type
        self.berhu_threshold = berhu_threshold

    def compute_depth_term(self, output, depth):
        valid_mask = depth > 0.0
        if self.depth_loss_type == 'berhu':
            loss_map = berhu_loss(output, depth, valid_mask, self.berhu_threshold)
        else:
            loss_map = torch.abs(output - depth)

        l_depth = loss_map[valid_mask].mean() if valid_mask.any() else loss_map.mean()
        return l_depth

    def compute_ssim_term(self, output, depth):
        return torch.clamp((1 - self.ssim(output, depth, self.maxDepth)) * 0.5, 0, 1)

    def compute_grad_term(self, output, depth):
        return self.gradient_loss(output, depth)

    def __call__(self, output, depth, image=None, **kwargs):
        l_depth = self.compute_depth_term(output, depth)
        l_ssim = self.compute_ssim_term(output, depth)
        l_grad = self.compute_grad_term(output, depth)

        loss = self.alpha * l_depth + self.beta * l_ssim + self.gamma * l_grad
        return loss

    def ssim(self, img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
        L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)
            padd = window_size // 2

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret

    def gradient_loss(self, gen_frames, gt_frames, alpha=1):
        dx_g, dy_g = self.gradient(gen_frames)
        dx_t, dy_t = self.gradient(gt_frames)

        grad_diff_x = torch.abs(dx_t - dx_g)
        grad_diff_y = torch.abs(dy_t - dy_g)

        grad_comb = grad_diff_x ** alpha + grad_diff_y ** alpha
        return torch.mean(grad_comb)

    def gradient(self, x):
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
