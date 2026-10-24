"""
Experiment configuration for 1-way 1-shot landslide segmentation (RGB PNG, binary mask).

Use by exporting DSPNET_CONFIG=landslide before running training.py / validation.py:
    $env:DSPNET_CONFIG = 'landslide'
    python training.py
    python validation.py with reload_model_path=<runs/.../snapshots/N.pth> eval_split=val
"""
import os
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('landslideSSL')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    """Default configurations"""
    seed = 1234
    gpu_id = 0
    mode = 'train'
    num_workers = 4  # 0 for debugging

    dataset = 'Landslide'
    use_coco_init = True

    ### Training
    n_steps = 50000
    batch_size = 1  # the model asserts a batch size of 1
    lr_milestones = [(ii + 1) * 1000 for ii in range(n_steps // 1000 - 1)]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 100
    save_snapshot_every = 25000
    max_iters_per_load = 1000  # episodes per "epoch" (random episodes are drawn from the train split)
    scan_per_load = -1  # unused for 2D patches, must stay < 0
    which_aug = 'sabs_aug'  # unused by the landslide loader (it has its own augmentation)
    input_size = (256, 256)  # patches are resized to this; keep coupled with feature_hw (stride 8)
    min_fg_data = 64  # min landslide pixels (native resolution) for a patch to be used as support/query
    label_sets = 0
    exclude_cls_list = []
    usealign = True
    use_wce = True  # class weights [0.05, 1.0]; revisit once the landslide fg/bg ratio is known

    ### Validation
    eval_split = 'val'  # 'val' or 'test'
    n_eval_episodes = None  # None: one episode per query patch of the split
    support_idx = [-1]
    val_wsize = 2

    # Network (unchanged)
    modelname = 'dlfcn_res101'
    clsname = "grid_proto"
    reload_model_path = None  # snapshot to evaluate (validation.py)
    proto_grid_size = 8
    feature_hw = [32, 32]  # = input_size / 8

    model = {
        'align': usealign,
        'use_coco_init': use_coco_init,
        'which_model': modelname,
        'cls_name': clsname,
        'proto_grid_size': proto_grid_size,
        'feature_hw': feature_hw,
        'reload_model_path': reload_model_path
    }

    task = {
        'n_ways': 1,
        'n_shots': 1,
        'n_queries': 1,
        'npart': 1
    }

    optim_type = 'sgd'
    optim = {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0005,
    }

    exp_prefix = ''
    exp_str = '_'.join([exp_prefix] + [dataset] + [f'{task["n_ways"]}way_{task["n_shots"]}shot'])

    path = {
        'log_dir': './runs',
        'Landslide': {
            'data_dir': "./datasets/landslide",  # root with train/ val/ test/ (used when a split dir is None)
            'train_dir': None,  # explicit dir of the train split (image/ + label/), overrides data_dir/train
            'val_dir': None,
            'test_dir': None,
            'blacklist': 'default',  # 'default' = <data_dir>/black_list.txt, 'none' = disabled, or a file path
        },
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook function to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
