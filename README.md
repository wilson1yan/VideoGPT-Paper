# VideoGPT - Paper

This repo is to reproduce the results of the original paper. For a cleaner version that include pretrained models on BAIR, UCF-101, and Kinetics, please refer to [this repo](https://github.com/wilson1yan/VideoGPT)

## Installation

Install the docker file in `docker/Dockerfile`. Commands below are run in a Docker container using [Singularity](https://singularity.lbl.gov/archive/docs/v2-3/docs-docker). Once inside the docker environment, install the repo as an editable package using `pip install -e VideoGPT-Paper`. 

## Datasets
Note that you will need to edit the `DATA_DIR` variable in `videogpt/dataset.py` to correctly access you datasets.

We include functionality for two of the main datasets: BAIR Robot Pushing dataset and UCF-101. To install the BAIR dataset, run the script `./preprocess_bair/create_bair_dataset.sh`. Then, create a `bair_pushing` directory in your datasets directory and copy the two `.hdf5` files into the `bair_pushing directory`.

In order to install UCF-101, you can download the necessary files [here](https://www.crcv.ucf.edu/data/UCF101.php). The code assumes a `ucf101` directory with the following structure
```
ucf101/
    UCF-101/
        ApplyEyeMakeup/
            v1.avi
            ...
        ...
        YoYo/
            v1.avi
            ...
    ucfTrainTestlist
```

## Training VQ-VQVAE
Execute `python train_vqvae.py -h` for all the tunable parameters. An example vq-vae training command is as follows
```
python train_vqvae.py -o <output_dir> --cfg vae_res64_ds422 --dataset bair_pushing
```

## Training VideoGPT
Execute `python train_videogpt.py -h` for all the tunable parameters. An example VideoGPT training command is as follows
```
python train_videogpt.py -o <output_dir> --cfg gpt_small --vqvae_ckpt /path/to/vqvae/ckpt --dataset bair_pushing
```
You can add the `--amp` flag for automatic mixed precision training if you have the support hardware.

## Misc
Other relevant scripts to compute FVD can be found in `scripts/`