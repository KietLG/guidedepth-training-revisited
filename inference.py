import time
import os
import argparse

import torch
import torchvision
import matplotlib.pyplot as plt

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
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description='Inference and Speed Benchmark for Monocular Depth Estimation')

    # Mode
    parser.set_defaults(evaluate=False)
    parser.add_argument('--eval', dest='evaluate', action='store_true')

    # Data
    parser.add_argument('--test_path', type=str, help='path to test data', default='./dataset/test/official')
    parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu', 'nyu_reduced'], default='nyu_reduced')
    parser.add_argument('--resolution', type=str, choices=['full', 'half'], default='half')

    # Model
    parser.add_argument('--model', type=str, default='GuideDepth')
    parser.add_argument('--weights_path', type=str, default='', help='path to model weights')
    parser.add_argument('--spatial_attention', type=str2bool, default=False)
    parser.add_argument('--skip_connection', type=str, choices=['none', 'single'], default='single')
    parser.add_argument('--deep_supervision_enable', type=str2bool, default=True)
    parser.add_argument('--localbins_enable', type=str2bool, default=True)
    parser.add_argument('--localbins_nbins', type=int, default=16)

    # Output & System
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


class Inference_Engine():
    def __init__(self, args):
        self.dataset = args.dataset
        self.maxDepth = max_depths[args.dataset]
        self.res_dict = resolutions[args.dataset]
        self.resolution = self.res_dict[args.resolution]
        self.resolution_keyword = args.resolution
        print('Resolution for Eval: {}'.format(self.resolution))
        print('Maximum Depth of Dataset: {}'.format(self.maxDepth))
        self.crop = crops[args.dataset]

        self.result_dir = args.save_results
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir, exist_ok=True)
        self.results_filename = '{}_{}_{}'.format(args.dataset, args.resolution, args.model)

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
        ).to(self.device).eval()

        if args.evaluate:
            self.test_loader = datasets.get_dataloader(
                args.dataset,
                path=args.test_path,
                split='test',
                batch_size=1,
                resolution=args.resolution,
                uncompressed=True,
                workers=args.num_workers
            )

        if args.resolution == 'half':
            self.upscale_depth = torchvision.transforms.Resize(self.res_dict['full'])
            self.downscale_image = torchvision.transforms.Resize(self.resolution)

        self.to_tensor = transforms.ToTensor(test=True, maxDepth=self.maxDepth)
        self.visualize_images = [0, 1, 2, 3, 4, 5]

        # TensorRT integration if available
        self.trt_model = None
        try:
            from torch2trt import torch2trt
            import tensorrt as trt
            self.trt_model, _ = self.convert_PyTorch_to_TensorRT()
        except ImportError:
            print("[Warning] torch2trt or TensorRT not installed. Skipping TensorRT conversion.")

        if args.evaluate:
            self.run_evaluation()

    def run_evaluation(self):
        speed_pyTorch = self.pyTorch_speedtest()
        speed_tensorRT = self.tensorRT_speedtest() if self.trt_model is not None else 0.0
        average = self.evaluate_model()
        self.save_results(average, speed_tensorRT, speed_pyTorch)

    def pyTorch_speedtest(self, num_test_runs=200):
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        x = torch.randn([1, 3, *self.resolution], device=self.device)
        
        with torch.no_grad():
            for i in range(num_test_runs + warm_up_runs):
                if i == warm_up_runs:
                    times = 0.0

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                t0 = time.time()
                result = self.model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times if times > 0 else 0
        print('[PyTorch] Runtime: {:.4f}s | FPS: {:.2f}'.format(times, fps))
        return times

    def tensorRT_speedtest(self, num_test_runs=200):
        if self.trt_model is None:
            return 0.0
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        x = torch.randn([1, 3, *self.resolution], device=self.device)

        with torch.no_grad():
            for i in range(num_test_runs + warm_up_runs):
                if i == warm_up_runs:
                    times = 0.0

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                t0 = time.time()
                result = self.trt_model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times if times > 0 else 0
        print('[tensorRT] Runtime: {:.4f}s | FPS: {:.2f}'.format(times, fps))
        return times

    def convert_PyTorch_to_TensorRT(self):
        from torch2trt import torch2trt
        import tensorrt as trt
        x = torch.ones([1, 3, *self.resolution], device=self.device)
        print('[tensorRT] Starting TensorRT conversion...')
        model_trt = torch2trt(self.model, [x], fp16_mode=True)
        print("[tensorRT] Model converted to TensorRT successfully.")

        TRT_LOGGER = trt.Logger()
        file_path = os.path.join(self.result_dir, '{}.engine'.format(self.results_filename))
        with open(file_path, 'wb') as f:
            f.write(model_trt.engine.serialize())

        with open(file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        print('[tensorRT] Engine serialized\n')
        return model_trt, engine

    def evaluate_model(self):
        torch.cuda.empty_cache()
        average_meter = AverageMeter()
        active_model = self.trt_model if self.trt_model is not None else self.model

        dataset = self.test_loader.dataset
        with torch.no_grad():
            for i, data in enumerate(dataset):
                t0 = time.time()
                image, gt = data
                packed_data = {'image': image, 'depth': gt}
                data = self.to_tensor(packed_data)
                image, gt = self.unpack_and_move(data)
                image = image.unsqueeze(0)
                gt = gt.unsqueeze(0)

                image_flip = torch.flip(image, [3])
                gt_flip = torch.flip(gt, [3])
                if self.resolution_keyword == 'half':
                    image = self.downscale_image(image)
                    image_flip = self.downscale_image(image_flip)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                data_time = time.time() - t0

                t0 = time.time()
                inv_pred = active_model(image)
                if isinstance(inv_pred, tuple):
                    inv_pred = inv_pred[0]
                prediction = self.inverse_depth_norm(inv_pred)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_time0 = time.time() - t0

                t1 = time.time()
                inv_pred_flip = active_model(image_flip)
                if isinstance(inv_pred_flip, tuple):
                    inv_pred_flip = inv_pred_flip[0]
                prediction_flip = self.inverse_depth_norm(inv_pred_flip)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_time1 = time.time() - t1

                if self.resolution_keyword == 'half':
                    prediction = self.upscale_depth(prediction)
                    prediction_flip = self.upscale_depth(prediction_flip)

                if i in self.visualize_images:
                    self.save_image_results(image, gt, prediction, i)

                gt_crop = gt[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
                gt_flip_crop = gt_flip[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
                pred_crop = prediction[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
                pred_flip_crop = prediction_flip[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]

                result = Result()
                result.evaluate(pred_crop.data, gt_crop.data)
                average_meter.update(result, gpu_time0, data_time, image.size(0))

                result_flip = Result()
                result_flip.evaluate(pred_flip_crop.data, gt_flip_crop.data)
                average_meter.update(result_flip, gpu_time1, data_time, image.size(0))

        avg = average_meter.average()
        print('\n*\n'
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'Delta1={average.delta1:.3f}\n'
              'Delta2={average.delta2:.3f}\n'
              'Delta3={average.delta3:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'Lg10={average.lg10:.3f}\n'
              't_GPU={time:.3f}\n'.format(
                  average=avg, time=avg.gpu_time))
        return avg

    def save_results(self, average, trt_speed, pyTorch_speed):
        file_path = os.path.join(self.result_dir, '{}.txt'.format(self.results_filename))
        with open(file_path, 'w') as f:
            f.write('s[PyTorch], s[tensorRT], RMSE,MAE,REL,Lg10,Delta1,Delta2,Delta3\n')
            f.write('{pyTorch_speed:.3f},{trt_speed:.3f},{average.rmse:.3f},{average.mae:.3f},{average.absrel:.3f},{average.lg10:.3f},{average.delta1:.3f},{average.delta2:.3f},{average.delta3:.3f}\n'.format(
                average=average, trt_speed=trt_speed, pyTorch_speed=pyTorch_speed))

    def inverse_depth_norm(self, depth):
        zero_mask = depth == 0.0
        depth_safe = torch.clamp(depth, min=1e-6)
        depth_safe = self.maxDepth / depth_safe
        depth_safe = torch.clamp(depth_safe, self.maxDepth / 100, self.maxDepth)
        depth_safe[zero_mask] = 0.0
        return depth_safe

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

    def save_image_results(self, image, gt, prediction, image_id):
        img = image[0].permute(1, 2, 0).cpu()
        gt_map = gt[0, 0].cpu()
        pred_map = prediction[0, 0].detach().cpu()
        error_map = torch.abs(gt_map - pred_map)

        save_to_dir = os.path.join(self.result_dir, 'image_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        fig.savefig(save_to_dir)
        plt.close(fig)

        save_to_dir = os.path.join(self.result_dir, 'depth_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(pred_map, cmap='viridis')
        fig.savefig(save_to_dir)
        plt.close(fig)


if __name__ == '__main__':
    args = get_args()
    print(args)
    engine = Inference_Engine(args)
