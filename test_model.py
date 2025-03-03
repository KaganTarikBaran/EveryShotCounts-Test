import torch
torch.cuda.empty_cache()
import numpy as np
import os
import einops
import cv2
import argparse
import av
import math
import gdown
import warnings
from models.video_mae_cross_full_attention import SupervisedMAE
from slowfast.utils.parser import load_config
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import create_video_transform
from itertools import islice
from torch.utils.data import DataLoader
from util.Rep_Count import Rep_Count
from util.UCF_Rep import UCFRep
from tqdm import tqdm
warnings.simplefilter("ignore")


SCALE_COUNTS = 100
ENCODINGS = "mae"
def get_args_parser():
    parser = argparse.ArgumentParser('MAE encoding', add_help=True)
    parser.add_argument('--resource', default='cpu', type=str, help='choose compute resource to use, e.g `cpu`,`cuda:0`,etc')
    parser.add_argument('--pool_tokens', default=0.4, type=float)
    parser.add_argument('--dataset', default='RepCount', help='choose from [RepCount, UCFRep]', type=str)
    return parser

def extract_tokens(video, model, args, num_frames=16):
    C, T, H, W = video.shape
    video = torch.cat([video, torch.zeros([C, 64, H, W])], 1)  
    clips = [video[:, np.linspace(j, j+64, num_frames+1).astype(int)] for j in range(0, T, 16)]
    data = torch.stack(clips).to(args.resource)

    dtype = 'cuda' if 'cuda' in args.resource else 'cpu'
    with torch.autocast(enabled='cuda' in args.resource, device_type=dtype), torch.no_grad():
            encoded, thw = model(data)
            encoded = encoded.transpose(1,2).reshape(encoded.shape[0], encoded.shape[-1], *thw)
    return encoded

def read_video_timestamps(video_filename, timestamps):
    """ 
    summary

    Args:
        video_filename (string): full filepath of the video
        timestamps (list): list of ints for the temporal points to load from the file

    Returns:
        video_frames: tensor of shape C x T x H x W
        last_frame_index: last frame index from the timestamps
    """

    if not os.path.isfile(video_filename):
        print(f"Error: Video file '{video_filename}' does not exist.")
        return None, None

    frames = []
    container = av.open(video_filename)

    min_t, max_t = min(timestamps), max(timestamps)

    for i, frame in enumerate(islice(container.decode(video=0), min_t, max_t + 1)):
        frame_index = i + min_t  # Adjust index offset
        if frame_index in timestamps:
            frames.extend([frame] * timestamps.tolist().count(frame_index))  # Handle multiple occurrences

    container.close()

    if not frames:
        print(f"Warning: No frames extracted from '{video_filename}'")
        return None, None

    video_frames = torch.stack([torch.from_numpy(f.to_ndarray(format='rgb24')) for f in frames])
    video_frames = thwc_to_cthw(video_frames.to(torch.float32))

    return video_frames, timestamps[-1]  


def load_encoder(args):
    cfg = load_config(args, path_to_config='configs/pretrain_config.yaml')
    encoder = SupervisedMAE(cfg=cfg, just_encode=True, use_precomputed=False, encodings=ENCODINGS).to(args.resource)

    model_path = "models/pretrained_models/VIT_B_16x4_MAE_PT.pyth"
    url = 'https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth'

    if not os.path.isfile(model_path):
        state_dict = torch.hub.load_state_dict_from_url(url, weights_only=True, map_location=args.resource)['model_state']
    else:
        state_dict = torch.load(model_path,  weights_only=True, map_location=args.resource)['model_state']

    for name, param in encoder.state_dict().items():
        if 'decode' in name:
            continue
        if name in state_dict:
            param.copy_(state_dict[name])
        elif '.qkv.' in name and 'blocks' in name:
            q_name, k_name, v_name = (name.replace('.qkv.', f'.{x}.') for x in ['q', 'k', 'v'])
            param.copy_(torch.cat([state_dict[q_name], state_dict[k_name], state_dict[v_name]]))

    return encoder


def load_decoder(args):
    cfg = load_config(args, path_to_config='configs/pretrain_config.yaml')
    decoder = SupervisedMAE(cfg=cfg,use_precomputed=True, token_pool_ratio=0.4, iterative_shots=True, encodings=ENCODINGS, no_exemplars=False, window_size=(4,7,7)).to(args.resource)

    model_path = 'models/pretrained_models/repcount_trained.pyth'

    url = 'https://drive.google.com/uc?id=1cwUtgUM0XotOx5fM4v4ZU29hlKUxze48'

    if not os.path.isfile(model_path):
        gdown.download(url, model_path, quiet=False)

    decoder.load_state_dict(torch.load(model_path, weights_only=True, map_location=args.resource)['model_state_dict'])
    return decoder


def process_video(video_path, encoder, decoder, args):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    transform =  create_video_transform(mode="test",
                                        convert_to_float=False,
                                        min_size = 224,
                                        crop_size = 224,
                                        num_samples = None,
                                        video_mean = [0.485,0.456,0.406], 
                                        video_std = [0.229,0.224,0.225])    
    frame_idx = np.arange(0, num_frames)
    vid_frames, _ = read_video_timestamps(video_path, frame_idx)
    vid_frames = transform(vid_frames / 255.0)

    if vid_frames is None:
        return None

    encoder.eval()
    encoded = extract_tokens(vid_frames, encoder, args)[0::4]
    encoded = einops.rearrange(encoded, 'S C T H W -> C (S T) H W')

    if args.pool_tokens < 1.0:
        factor = math.ceil(encoded.shape[-1] * args.pool_tokens)
        tokens = torch.nn.functional.adaptive_avg_pool3d(encoded, (encoded.shape[-3], factor, factor))
    tokens = tokens.unsqueeze(0)
    shapes = tokens.shape[-3:]
    tokens = einops.rearrange(tokens, 'B C T H W -> B (T H W) C')
    dtype = 'cuda' if 'cuda' in args.resource else 'cpu'

    with torch.autocast(enabled='cuda' in args.resource, device_type=dtype), torch.no_grad():
        predicted_density_map = decoder(tokens, thw=[shapes,], shot_num=0)


    return round(predicted_density_map.sum().item() / SCALE_COUNTS)

def main():
    args = get_args_parser().parse_args()
    args.opts = None
    encoder = load_encoder(args)
    decoder = load_decoder(args)

    dataset_test = Rep_Count() if args.dataset == 'RepCount' else UCFRep(split="val")
    dataloader = DataLoader(dataset_test, num_workers=1, shuffle=False, pin_memory=True, drop_last=False)

    gt_counts, predictions, predict_mae = [], [], []

    for _, item in enumerate(tqdm(dataloader)):
        video_path, actual_count = item[0][0], int(item[1][0])
        predicted_count = process_video(video_path, encoder, decoder, args)

        if predicted_count is not None:
            print(f'The number of repetitions is {predicted_count}')
            predictions.append(predicted_count)
            gt_counts.append(actual_count)
            predict_mae.append(abs(predicted_count - actual_count) / (actual_count + 1e-1))

    predictions, gt_counts, predict_mae = map(np.array, [predictions, gt_counts, predict_mae])
    mae = predict_mae.mean()
    obo = (np.abs(predictions - gt_counts) <= 1).mean()

    print(f'Overall MAE: {mae}') 
    print(f'OBO: {obo}')   

if __name__ == '__main__':
    main()

