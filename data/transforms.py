import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random

def _is_pil_image(img):
    return isinstance(img, Image.Image)


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


class ToTensor(object):
    def __init__(self, test=False, maxDepth=10.0):
        self.test = test
        self.maxDepth = maxDepth

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        transformation = transforms.ToTensor()

        if self.test:
            image = np.array(image).astype(np.float32) / 255.0
            depth = np.array(depth).astype(np.float32)
            image, depth = transformation(image), transformation(depth)
        else:
            image = np.array(image).astype(np.float32) / 255.0
            depth = np.array(depth).astype(np.float32)

            zero_mask = depth == 0.0
            image, depth = transformation(image), transformation(depth)
            depth = torch.clamp(depth, self.maxDepth / 100.0, self.maxDepth)
            depth = self.maxDepth / depth
            depth[:, zero_mask] = 0.0

        image = torch.clamp(image, 0.0, 1.0)
        return {'image': image, 'depth': depth}


class Resize(object):
    def __init__(self, output_resolution):
        self.resize = transforms.Resize(output_resolution)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth)

        image = self.resize(image)
        depth = self.resize(depth)

        return {'image': image, 'depth': depth}


class ResizeRGBOnly(object):
    def __init__(self, output_resolution):
        self.resize = transforms.Resize(output_resolution)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth)

        image = self.resize(image)
        return {'image': image, 'depth': depth}


class CutDepth(object):
    """
    CutDepth (Ishii & Yamashita, arXiv:2107.07684)
    Dán vùng lấy từ CHÍNH depth map thật đè lên ảnh RGB đầu vào.
    GT depth GIỮ NGUYÊN.
    """
    def __init__(self, probability=0.75, max_area_ratio=0.25, mode='rect'):
        self.probability = probability
        self.max_area_ratio = max_area_ratio
        self.mode = mode

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if random.random() > self.probability:
            return sample

        img_np = np.array(image).astype(np.float32).copy()
        depth_np = np.array(depth).astype(np.float32)
        h, w = depth_np.shape[:2]

        if self.mode == 'vertical':
            strip_w = max(int(w * random.uniform(0.05, self.max_area_ratio)), 1)
            x0 = random.randint(0, max(w - strip_w, 0))
            y0, cut_h, cut_w = 0, h, strip_w
        else:
            area_ratio = random.uniform(0.02, self.max_area_ratio)
            cut_h = max(int(h * (area_ratio ** 0.5)), 1)
            cut_w = max(int(w * (area_ratio ** 0.5)), 1)
            y0 = random.randint(0, max(h - cut_h, 0))
            x0 = random.randint(0, max(w - cut_w, 0))

        d_patch = depth_np[y0:y0+cut_h, x0:x0+cut_w]
        d_min, d_max = d_patch.min(), d_patch.max()
        d_norm = ((d_patch - d_min) / (d_max - d_min) * 255.0) if d_max > d_min else np.zeros_like(d_patch)
        d_rgb_patch = np.repeat(d_norm[..., None], 3, axis=2)

        img_np[y0:y0+cut_h, x0:x0+cut_w] = d_rgb_patch
        image = Image.fromarray(np.uint8(img_np))

        return {'image': image, 'depth': depth}


def DepthNorm(depth, maxDepth=10.0):
    return maxDepth / depth
