import argparse
import json
import os
import requests
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
import io
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--download_dir", type=str, default=os.path.join(SRC_DIR, "downloads"))

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args


def main(args):
    if not os.path.exists(args.download_dir):
        os.makedirs(args.download_dir)

    # run the download for moving mnist
    if args.dataset == "moving_mnist":
        url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
        print(f"downloading {args.dataset} dataset..")
        r = requests.get(url)
        r.raise_for_status()
        data = np.load(io.BytesIO(r.content))
        del r
        print("download completed")
        data = data.transpose((1, 0, 2, 3))[..., np.newaxis]  # B, T, H, W, C
        tr_data_split = int(len(data) * 0.7)
        va_data_split = int(len(data) * 0.8)
        te_data_split = int(len(data) * 1.0)
        data_split = dict(
            trX=data[:tr_data_split],
            trY=np.ones(tr_data_split),
            vaX=data[tr_data_split:va_data_split],
            vaY=np.ones(va_data_split-tr_data_split),
            teX=data[va_data_split:te_data_split],
            teY=np.ones(te_data_split-va_data_split),
        )
        for split, chunk in data_split.items():
            filename = f"{args.dataset}_{split}"
            np.save(f"{args.download_dir}/{filename}", chunk)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
