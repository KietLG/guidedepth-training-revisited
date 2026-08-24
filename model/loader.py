import torch.nn as nn
import torch
import os
import re

from model.GuideDepth import GuideDepth

def get_latest_weights(results_dir='./results'):
    if not os.path.exists(results_dir):
        return None
        
    pattern = re.compile(r'^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$')
    run_folders = [f for f in os.listdir(results_dir) if pattern.match(f)]
    run_folders = sorted(run_folders, reverse=True)
    
    for folder in run_folders:
        run_path = os.path.join(results_dir, folder)
        models_dir = os.path.join(run_path, 'models')
        if not os.path.isdir(models_dir):
            continue
            
        best_model_path = os.path.join(models_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            return best_model_path
            
        checkpoints = [f for f in os.listdir(models_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
        if len(checkpoints) > 0:
            def get_epoch(filename):
                try:
                    return int(filename.split('_')[1].split('.')[0])
                except:
                    return -1
            checkpoints = sorted(checkpoints, key=get_epoch, reverse=True)
            return os.path.join(models_dir, checkpoints[0])
            
    return None

def load_model(model_name, weights_pth, spatial_attention=False, skip_connection='single',
               deep_supervision_enable=True, localbins_enable=True, localbins_nbins=16,
               **kwargs):
    model = model_builder(model_name, spatial_attention=spatial_attention,
                          skip_connection=skip_connection,
                          deep_supervision_enable=deep_supervision_enable,
                          localbins_enable=localbins_enable,
                          localbins_nbins=localbins_nbins,
                          **kwargs)

    if weights_pth == 'latest':
        latest_path = get_latest_weights()
        if latest_path is not None:
            print(f"Automatically loading latest weights from: {latest_path}")
            weights_pth = latest_path
        else:
            print("Warning: weights_pth is set to 'latest' but no valid checkpoints were found in './results'.")
            weights_pth = None

    if weights_pth is not None and weights_pth != "":
        state_dict = torch.load(weights_pth, map_location='cpu')
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)

    return model

def model_builder(model_name, spatial_attention=False, skip_connection='single',
                  deep_supervision_enable=True, localbins_enable=True, localbins_nbins=16,
                  **kwargs):
    pretrained_path = './model/weights/DDRNet23s_imagenet.pth'
    pretrained = os.path.exists(pretrained_path)
    if not pretrained:
        print(f"Warning: '{pretrained_path}' not found. Initializing model with pretrained=False.")

    if model_name == 'GuideDepth':
        return GuideDepth(pretrained, spatial_attention=spatial_attention,
                          skip_connection=skip_connection,
                          deep_supervision_enable=deep_supervision_enable,
                          localbins_enable=localbins_enable,
                          localbins_nbins=localbins_nbins)

    print("Invalid model")
    exit(0)
