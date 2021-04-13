import argparse
import numpy as np
import torch
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--videos_file', type=str, required=True)
parser.add_argument('-o', '--output_file', type=str, required=True)
parser.add_argument('-s', '--n_skip_frame', type=int, default=1,
                    help="number of frames to skip (default: 1 is no skip)")
parser.add_argument('-n', '--n_examples', type=int, default=2)
parser.add_argument('-i', '--start_idx', type=int, default=0)
args = parser.parse_args()
videos = np.load(args.videos_file) # (n, t, h, w, c) in {0, ..., 255}
print('Loaded samples', videos.shape, videos.min(), videos.max())

videos = torch.FloatTensor(videos).permute(0, 1, 4, 2, 3) / 255. # (n, t, c, h, w), [0, 1]
videos = videos[args.start_idx:args.start_idx + args.n_examples, ::args.n_skip_frame]
T = videos.shape[1]
videos = videos.flatten(end_dim=1)

save_image(videos, args.output_file, nrow=T)

