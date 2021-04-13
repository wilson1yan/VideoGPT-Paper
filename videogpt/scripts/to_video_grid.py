import sys
import math
import os.path as osp
import numpy as np
import skvideo.io
import argparse

MAX_N_VIDEOS = 100

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fname', type=str, required=True)
parser.add_argument('-s', '--start', type=int, default=None)
parser.add_argument('-e', '--end', type=int, default=None)
args = parser.parse_args()
fname = args.fname
folder = osp.dirname(fname)

# (b, t, h, w, c) in {0, .., 255}
slc = slice(args.start, args.end)
videos = np.load(fname)[slc]
b, t, h, w, c = videos.shape

grid_size = math.ceil(math.sqrt(b))
padding = 1
video_grid = np.zeros((t, (padding + h) * grid_size + padding,
                       (padding + w) * grid_size + padding, c), dtype='uint8')
for i in range(b):
    r = i // grid_size
    c = i % grid_size

    start_r = (padding + h) * r
    start_c = (padding + w) * c
    video_grid[:, start_r:start_r + h, start_c:start_c + w] = videos[i]

out_fname = osp.join(folder, f'video_grid_{args.start}-{args.end}.mp4')
skvideo.io.vwrite(out_fname, video_grid, inputdict={'-r': '5'})
print('outputted mp4 at', out_fname)

