import pandas as pd
import numpy as np
import torch
import os
from zipfile import ZipFile
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from data.transforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor

resolution_dict = {
    'full': (480, 640),
    'half': (240, 320),
    'mini': (224, 224)}

class depthDatasetMemory(Dataset):
    def __init__(self, data, split, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform
        self.split = split

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        image = np.array(image).astype(np.float32)
        depth = np.array(depth).astype(np.float32)

        if self.split == 'train':
            depth = depth / 255.0 * 10.0
        elif self.split == 'val':
            depth = depth * 0.001

        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class NYU_Testset_Extracted(Dataset):
    def __init__(self, root, resolution='full'):
        self.root = root
        if isinstance(resolution, str):
            self.resolution = resolution_dict[resolution]
        else:
            self.resolution = resolution

        self.files = os.listdir(self.root)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.files[index])

        data = np.load(image_path)
        depth, image = data['depth'], data['image']
        depth = np.expand_dims(depth, axis=2)

        image, depth = data['image'], data['depth']
        image = np.array(image)
        depth = np.array(depth)
        return image, depth

    def __len__(self):
        return len(self.files)


class NYU_Testset(Dataset):
    def __init__(self, zip_path):
        input_zip = ZipFile(zip_path)
        data = {name: input_zip.read(name) for name in input_zip.namelist()}
        
        self.rgb = torch.from_numpy(np.load(BytesIO(data['eigen_test_rgb.npy']))).type(torch.float32)
        self.depth = torch.from_numpy(np.load(BytesIO(data['eigen_test_depth.npy']))).type(torch.float32)

    def __getitem__(self, idx):
        image = self.rgb[idx]
        depth = self.depth[idx]
        return image, depth

    def __len__(self):
        return len(self.rgb)


def loadZipToMem(zip_file):
    print('Loading dataset zip file...', end='')
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    print('Loaded (Train Images: {0}, Test Images: {1}).'.format(len(nyu2_train), len(nyu2_test)))
    return data, nyu2_train, nyu2_test


def train_transform(resolution, cutdepth_enable=True, cutdepth_mode='rect', cutdepth_probability=0.75, cutdepth_max_area_ratio=0.25):
    from data.transforms import CutDepth
    tf_list = [
        Resize(resolution),
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5)
    ]
    if cutdepth_enable:
        tf_list.append(CutDepth(probability=cutdepth_probability, max_area_ratio=cutdepth_max_area_ratio, mode=cutdepth_mode))
    tf_list.append(ToTensor(test=False, maxDepth=10.0))
    transform = transforms.Compose(tf_list)
    return transform

def val_transform(resolution):
    from data.transforms import ResizeRGBOnly, ToTensor
    transform = transforms.Compose([
        ResizeRGBOnly(resolution),
        ToTensor(test=True, maxDepth=10.0)
    ])
    return transform


class depthDatasetDirectory(Dataset):
    def __init__(self, root, split, transform=None, val_ratio=0.05, seed=42):
        self.root = root
        self.transform = transform
        self.split = split
        
        selected_scenes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        
        import random
        rng = random.Random(seed)
        shuffled = selected_scenes.copy()
        rng.shuffle(shuffled)
        n_val_scenes = max(int(len(shuffled) * val_ratio), 1)
        val_scenes = set(shuffled[:n_val_scenes])
        train_scenes = [s for s in selected_scenes if s not in val_scenes]

        chosen_scenes = val_scenes if split == 'val' else train_scenes
            
        self.samples = []
        for scene in chosen_scenes:
            scene_dir = os.path.join(root, scene)
            rgb_files = sorted([f for f in os.listdir(scene_dir) if f.startswith('rgb_')])
            for rgb_name in rgb_files:
                depth_name = rgb_name.replace('rgb_', 'depth_')
                self.samples.append((
                    os.path.join(scene_dir, rgb_name),
                    os.path.join(scene_dir, depth_name),
                    scene,
                    rgb_name
                ))

    def __getitem__(self, idx):
        rgb_path, depth_path, scene, rgb_name = self.samples[idx]
        image = Image.open(rgb_path)
        depth = Image.open(depth_path)

        image = np.array(image).astype(np.float32)
        depth = np.array(depth).astype(np.float32)

        depth = depth / 6553.5

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)


class NYU_Testset_PNG(Dataset):
    def __init__(self, root, resolution='full'):
        self.root = root
        if isinstance(resolution, str):
            self.resolution = resolution_dict[resolution]
        else:
            self.resolution = resolution
        self.rgb_files = sorted([f for f in os.listdir(root) if f.startswith('rgb_')])

    def __getitem__(self, index):
        rgb_name = self.rgb_files[index]
        depth_name = rgb_name.replace('rgb_', 'depth_')
        
        rgb_path = os.path.join(self.root, rgb_name)
        depth_path = os.path.join(self.root, depth_name)
        
        image = Image.open(rgb_path)
        depth = Image.open(depth_path)
        
        image = np.array(image).astype(np.float32)
        depth = np.array(depth).astype(np.float32)
        
        depth = depth / 6553.5
        
        return image, depth

    def __len__(self):
        return len(self.rgb_files)


def get_NYU_dataset(zip_path, split, resolution='full', uncompressed=True, **kwargs):
    if isinstance(resolution, str):
        resolution_tuple = resolution_dict[resolution]
        resolution_str = resolution
    else:
        resolution_tuple = resolution
        resolution_str = 'full'
        for k, v in resolution_dict.items():
            if v == resolution:
                resolution_str = k
                break
    
    if os.path.isdir(zip_path):
        if split == 'train':
            transform = train_transform(
                resolution_tuple,
                cutdepth_enable=kwargs.get('cutdepth_enable', True),
                cutdepth_mode=kwargs.get('cutdepth_mode', 'rect'),
                cutdepth_probability=kwargs.get('cutdepth_probability', 0.75),
                cutdepth_max_area_ratio=kwargs.get('cutdepth_max_area_ratio', 0.25)
            )
            dataset = depthDatasetDirectory(
                zip_path, split, transform=transform,
                val_ratio=kwargs.get('val_ratio', 0.05),
                seed=kwargs.get('seed', 42)
            )
        elif split == 'val':
            transform = val_transform(resolution_tuple)
            dataset = depthDatasetDirectory(
                zip_path, split, transform=transform,
                val_ratio=kwargs.get('val_ratio', 0.05),
                seed=kwargs.get('seed', 42)
            )
        elif split == 'test':
            dataset = NYU_Testset_PNG(zip_path, resolution=resolution_str)
        return dataset

    if split == 'train':
        data, nyu2_train, nyu2_test = loadZipToMem(zip_path)

        transform = train_transform(
            resolution_tuple,
            cutdepth_enable=kwargs.get('cutdepth_enable', True),
            cutdepth_mode=kwargs.get('cutdepth_mode', 'rect'),
            cutdepth_probability=kwargs.get('cutdepth_probability', 0.75),
            cutdepth_max_area_ratio=kwargs.get('cutdepth_max_area_ratio', 0.25)
        )
        dataset = depthDatasetMemory(data, split, nyu2_train, transform=transform)
    elif split == 'val':
        data, nyu2_train, nyu2_test = loadZipToMem(zip_path)

        transform = val_transform(resolution_tuple)
        dataset = depthDatasetMemory(data, split, nyu2_test, transform=transform)
    elif split == 'test':
        if uncompressed:
            dataset = NYU_Testset_Extracted(zip_path, resolution=resolution_str)
        else:
            dataset = NYU_Testset(zip_path)

    return dataset
