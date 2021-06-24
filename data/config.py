# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    # prior
    'min_sizes': [[16, 32]],
    'ratios': [1, 1.5, 2], # width/height
    'steps': [8],
    # 'stride_prior': 1
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'ngpu': 1,
    'max_epochs': 250,
    'warmup_epochs': 5,
    'decay_steps': [0.55, 0.78],
    'image_size': 96,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32]],
    'steps': [8],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 96,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

cfg_re34 = {
    'name': 'Resnet34',
    # 'min_sizes': [[16, 32], [48, 64], [80, 96]],
    'min_sizes': [[16, 32]],
    'ratios': [1], # width/height
    'steps': [8],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'max_epochs': 250,
    'warmup_epochs': 5,
    'decay_steps': [0.55, 0.78],
    'image_size': 96,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 128,
    'out_channel': 256
}
