import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from Rep_count import Rep_count
from video_mae_cross import SupervisedMAE
from slowfast.utils.parser import load_config
import argparse

import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--num_gpus', default=4, type=int)
    parser.add_argument('--pretrained_encoder', default ='pretrained_models/VIT_B_16x4_MAE_PT.pyth', type=str)
    parser.add_argument('--save_example_encodings', default=True, type=bool)
    return parser

def save_exemplar(dataloaders, model):
    for split in ['train','val', 'test']:
        for i, item in enumerate(dataloaders[split]):
            video = item[0].squeeze(0)
            starts = item[-3]
            ends = item[-2]
            video_name = item[-1][0]
            print(video_name)
            C, T, H, W = video.shape

            clip_list = []
            num_examples = len(starts)
            # if split == 'train':
            for j in range(num_examples):
                idx = np.linspace(starts[j].item(), ends[j].item(), 17)[:16].astype(int)
                clips = video[:, idx]
                clip_list.append(clips)
            # else:
            #     idx = np.linspace(starts[0].item(), ends[0].item(), 16)
            #     clips = video[:, idx]
            #     clip_list.append(clips)
            
            data = torch.stack(clip_list).cuda()
            print(data.shape)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    encoded = model(data)
            # print(encoded.shape)
            
            np.savez('examplar_tokens/{}/{}.npz'.format(split, video_name), encoded.cpu().numpy())

def save_tokens(dataloaders, model):
    for split in ['train', 'val']:
        for i, item in enumerate(dataloaders[split]):
            video = item[0].squeeze(0)
            video_name = item[-1][0]
            print(video_name)
            C, T, H, W = video.shape

            clip_list = []
            for j in range(0, T-64, 16): #### 75% overlap
                idx = np.linspace(j, j+64, 17)[:16]
                clips = video[:,idx]
                clip_list.append(clips)
            data = torch.stack(clip_list).cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    encoded = model(data)
            # print(encoded.shape)
            
            # np.savez('saved_tokens/{}/{}.npz'.format(split, video_name), encoded.cpu().numpy())


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.opts = None
    args.save_video_encodings = not args.save_example_encodings
    args.data_path = '/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/LLSP/'

    

    cfg = load_config(args, path_to_config='pretrain_config.yaml')
    model = SupervisedMAE(cfg=cfg, just_encode=True).cuda()
    model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
    model.eval()

    dataset_train = Rep_count(cfg=cfg,split="train",data_dir=args.data_path,sampling_interval=1,encode_only=True)
    dataset_val = Rep_count(cfg=cfg,split="valid",data_dir=args.data_path,sampling_interval=1,encode_only=True)
    dataset_test = Rep_count(cfg=cfg,split="test",data_dir=args.data_path)

    dataloaders = {'train':torch.utils.data.DataLoader(dataset_train,batch_size=args.batch_size,
                                                       num_workers=10,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=False),
                   'val':torch.utils.data.DataLoader(dataset_val,
                                                     batch_size=args.batch_size,
                                                     num_workers=10,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     drop_last=False),
                   'test':torch.utils.data.DataLoader(dataset_test,
                                                     batch_size=args.batch_size,
                                                     num_workers=10,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     drop_last=False)}
    if args.pretrained_encoder:
        state_dict = torch.load(args.pretrained_encoder)['model_state']
    else:
        state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth')['model_state']
    for name, param in state_dict.items():
        if name in model.state_dict().keys():
            if 'decoder' not in name:
                print(name)
                # new_name = name.replace('quantizer.', '')
                model.state_dict()[name].copy_(param)
    if args.save_video_encodings:
        save_tokens(dataloaders, model)
    elif args.save_example_encodings:
        save_exemplar(dataloaders, model)

    

if __name__ == '__main__':
    main()
    


