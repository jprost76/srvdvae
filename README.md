# Efficicient posterior sampling for diverse super-resolution with HVAE prior

This code is the official implementation of the paper.
It implements a method to perform diverse image super-resolution by 

## Setup
Initiate a python virtual environemnt and install the necessary packages
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt 
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

## Test
Download pretrained checkpoints:

Download VDVAE pretrained model:



```
python3 sample_sr.py
```
## Acknowledgements

This code borrow elements from several works
- The consistency enforcement module (CEM) implementation is 
- VDVAE implementation
- Dataset ...