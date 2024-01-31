#!/bin/bash

mkdir data
cd data
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq-256.npy

cd ../scripts
# x4
python3 prepare_ffhq256.py ../data/ffhq-256.npy --scale 4 --save_HR

# x8
python3 prepare_ffhq256.py ../data/ffhq-256.npy --scale 8 --save_HR
