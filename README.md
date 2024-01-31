# Efficicient posterior sampling for diverse super-resolution with HVAE prior

This code is the official implementation of the paper "Efficicient posterior sampling for diverse super-resolution with HVAE prior" ([arxiv](https://arxiv.org/abs/2205.10347)).

It trains an encoder on top a pretrained VDVAE model in order to perform super-resolution. Due to the probabilistic nature of VDVAE, it enable us to sample efficiently diverse solutions to the super-resolution problem.

## Setup
Initiate a python virtual environemnt and install the necessary packages
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt 
```

## VDVAE checkpoints
Download pretrained VDVAE checkpoint:
```
mkdir checkpoints
mkdir checkpoints/vdvae
cd checkpoints/vdvae
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model-ema.th
```

## Data preparation
To download ffhq256 dataset and prepare the dataset:
```
./prepare_data.sh
```

## Training
The training code is implemented with [pytorch-lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

For help, run
```
python3 train.py --help
```

To print the default training configuration, run:
```
python3 train.py --print_config fit
```

To start a training run:
```
python3 train.py  fit -c configs/ffhq256_x8_v0.yaml
```




## Test
Pretrained checkpoints will be made available.


For instance to test the x4 model: 
```
python3 sample_sr.py --model_path checkpoints/SR_x4_FFHQ256_bic_000
```
## Acknowledgements

This code use parts from the following project
- https://github.com/YuvalBahat/Explorable-Super-Resolution/tree/master/codes
- https://github.com/openai/vdvae


Dataset:
- https://github.com/NVlabs/ffhq-dataset

Please refer to the associated licenses.