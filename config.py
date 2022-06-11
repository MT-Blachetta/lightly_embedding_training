
import os
import yaml
from easydict import EasyDict
#import numpy as np
import errno


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def create_config(root_dir, config_file_exp, prefix):
    # Config for environment path

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    #nfix =  cfg['neighbor_prefix']
    #nfix_v = cfg['neighbor_prefix_val']
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, prefix)
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['result_save_path'] = os.path.join(pretext_dir,prefix+'_model.pth')

    #cfg['pretext_dir'] = pretext_dir
    cfg['checkpoint_path'] = os.path.join(pretext_dir,prefix+'_checkpoint.pth.tar')

    return cfg

"""
    cfg['pretext_model'] = os.path.join(pretext_dir,prefix+'_model.pth.tar')
    #os.path.join('topk',prefix+'_topk-train-neighbors.npy')
    cfg['topk_neighbors_train_path'] = os.path.join(base_dir, 'topk/'+nfix+'_topk-train-neighbors.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(base_dir, 'topk/'+nfix_v+'_topk-val-neighbors.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    #if cfg['setup'] in ['scan', 'selflabel','multidouble','multitwist']:
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    scan_dir = os.path.join(base_dir, cfg['setup'])
    selflabel_dir = os.path.join(base_dir, 'selflabel')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(scan_dir)
    mkdir_if_missing(selflabel_dir)
    cfg['scan_dir'] = scan_dir
    cfg['scan_checkpoint'] = os.path.join(scan_dir,prefix+'_checkpoint.pth.tar')
    cfg['evaluation_dir'] = os.path.join(scan_dir,prefix+'_measurements')
    cfg['scan_model'] = os.path.join(scan_dir,prefix+'_model.pth.tar')
    cfg['selflabel_dir'] = selflabel_dir
    cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir,prefix+'_checkpoint.pth.tar')
    cfg['selflabel_model'] = os.path.join(selflabel_dir,prefix+'_model.pth.tar')
"""


