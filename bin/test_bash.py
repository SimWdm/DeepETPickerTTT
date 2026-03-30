import json
import sys
import os
import importlib
from os.path import dirname, abspath
import numpy as np
import pandas as pd
import torch

DeepETPickerHome = dirname(abspath(__file__))
DeepETPickerHome = os.path.split(DeepETPickerHome)[0]
sys.path.append(DeepETPickerHome)
sys.path.append(os.path.split(DeepETPickerHome)[0])
test = importlib.import_module(".test", package=os.path.split(DeepETPickerHome)[1])
option = importlib.import_module(f".options.option", package=os.path.split(DeepETPickerHome)[1])


def _infer_f_maps_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    f_maps = []
    i = 0
    while True:
        conv_key = f"model.encoders.{i}.basic_module.conv1.conv.weight"
        lw_key = f"model.encoders.{i}.basic_module.conv1.pwconv2.weight"
        if conv_key in state_dict:
            f_maps.append(int(state_dict[conv_key].shape[0]))
            i += 1
            continue
        if lw_key in state_dict:
            f_maps.append(int(state_dict[lw_key].shape[0]))
            i += 1
            continue
        break

    return f_maps if len(f_maps) > 0 else None


if __name__ == '__main__':
    options = option.BaseOptions()
    args = options.gather_options()

    # cofig
    with open(args.train_configs, 'r') as f:
        cfg = json.loads(''.join(f.readlines()).lstrip('train_configs='))

    # parameters
    args.use_bg = True
    args.use_IP = True
    args.use_coord = True
    args.test_use_pad = True
    args.use_seg = True
    args.meanPool_NMS = True
    if args.f_maps is None:
        inferred_f_maps = _infer_f_maps_from_checkpoint(args.checkpoints)
        if inferred_f_maps is not None:
            args.f_maps = inferred_f_maps
        else:
            args.f_maps = [24, 48, 72, 108]
    args.num_classes = cfg['num_cls']
    train_cls_num = cfg['num_cls']
    if args.num_classes == 1:
        args.use_sigmoid = True
        args.use_softmax = False
    else:
        train_cls_num = train_cls_num + 1
        args.use_sigmoid = False
        args.use_softmax = True
    args.denoising = cfg['denoising']
    args.batch_size = cfg['batch_size']
    args.block_size = cfg['patch_size']
    args.val_batch_size = args.batch_size
    args.val_block_size = args.block_size
    args.pad_size = [cfg['padding_size']]
    args.learning_rate = cfg['lr']
    args.max_epoch = cfg['max_epochs']
    args.threshold = cfg['seg_thresh']
    args.gpu_id = [int(i) for i in cfg['gpu_ids'].split(',')]
    args.test_mode = 'test_only'
    args.out_name = 'PredictedLabels'
    args.de_duplication = True
    args.de_dup_fmt = 'fmt4'
    args.mini_dist = sorted([int(i) // 2 + 1 for i in cfg['ocp_diameter'].split(',')])[0]
    args.data_split = [0, 1, 0, 1, 0, 1]
    args.configs = args.train_configs
    args.num_classes = train_cls_num

    # test_idxs
    dset_list = np.array(
        [i[:-(len(i.split('.')[-1]) + 1)] for i in os.listdir(cfg['tomo_path']) if cfg['tomo_format'] in i])
    dset_num = dset_list.shape[0]
    num_name = np.concatenate([np.arange(dset_num).reshape(-1, 1), dset_list.reshape(-1, 1)], axis=1)
    np.savetxt(os.path.join(cfg['tomo_path'], 'num_name.csv'),
               num_name,
               delimiter='\t',
               fmt='%s',
               newline='\n')

    #tomo_list = [i for i in os.listdir(cfg[f"{cfg['base_path']}/data_std"]) if cfg['tomo_format'] in i]
    # tomo_list = np.loadtxt(f"{cfg['base_path']}/data_std/num_name.csv",
    #                        delimiter='\t',
    #                        dtype=str)
    tomo_list = pd.read_csv(os.path.join(cfg['tomo_path'], 'num_name.csv'),
                            delimiter='\t',
                            header=None,
                            dtype=str)
    args.test_idxs = np.arange(len(tomo_list))

    for k, v in sorted(vars(args).items()):
        print(k, '=', v)

    # modification: this is not elegant but needed to avoid CUDA out of memory
    for id in  np.arange(len(tomo_list)):
        torch.cuda.empty_cache()
        args.test_idxs = [id]
        # Testing
        test = None
        test = importlib.import_module(".test", package=os.path.split(DeepETPickerHome)[1])
        print(f"Processing tomogram {tomo_list.iloc[id, 1]} ({id+1}/{len(tomo_list)})")
        test.test_func(args, stdout=None)
