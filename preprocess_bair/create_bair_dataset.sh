#! /bin/sh

mkdir -p ~/.cache/bair
wget -P ~/.cache/bair/ bair_robot_pushing_dataset_v0.tar
tar -xf ~/.cache/bair/bair_robot_pushing_dataset_v0.tar

python bair_extract_images.py --data_dir ~/.cache/bair
python bair_image_to_hdf5.py --data_dir ~/.cache/bair
