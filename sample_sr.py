import argparse
import os
import yaml

import torch
import numpy as np
from torchvision.io import write_png
from compute_metrics import super_resolve
import CEM.CEMnet as CEMnet

from utils import find_best_ckpt_and_config
from train import CondVAE
from data import LRHR_PKLDataset

def quantize(x, range=255):
    if range != 255:
        x = x * (255/range)
    x = torch.clamp(x, 0, 255).round().type(torch.uint8)
    return x
    
def info(x):
    print(x.shape, x.min(), x.max())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="checkpoints/SR_x4_FFHQ256_bic_000")
    parser.add_argument("--nmax", type=int, default=20, help="number max of tested images")
    parser.add_argument("--samples_per_image", type=int, default=5)
    parser.add_argument("--temp", type=float, nargs='+', default=[0.8])
    parser.add_argument("--cem", action="store_true")
    args = parser.parse_args()
    
    # load model
    best_ckpt_path, config_path = find_best_ckpt_and_config(args.model_path)
    print(best_ckpt_path)
    print(config_path)
    module = CondVAE.load_from_checkpoint(best_ckpt_path)
    cvae = module.cvae.cuda()
    with open(config_path) as f:
        conf = yaml.safe_load(f)
    sf = conf['model']['image_size_y'] / 256

    # load CEM
    if args.cem:
        scale = 256// module.y_size
        CEM_conf = CEMnet.Get_CEM_Conf(scale)
        CEM_conf.input_range = np.array([0, 1])
        CEM_conf.decomposed_output = False
        CEM_conf.lower_magnitude_bound = 0.000
        cem_net = CEMnet.CEMnet(CEM_conf, upscale_kernel='cubic')
    else:
        cem_net = None

    # init dataloaders
    #if hr_
    dataset = LRHR_PKLDataset({
        "dataroot_GT" : conf['data']['path_val_HR'],
        "dataroot_LQ" : conf['data']['path_val_LR'],
        "use_flip" : False,
        "use_rot" : False,
        "use_crop" : False
    })

    ckpt_name = os.path.split(best_ckpt_path)[1]

    cem_str = 'cem' if args.cem else 'simple'
    save_dir = os.path.join(args.model_path, 'samples', ckpt_name, cem_str)
    for t in args.temp:
        os.makedirs(os.path.join(save_dir, 't={}'.format(t)), exist_ok=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers=2, shuffle=False)
    # iterate over images
    for img_index, (x, y) in enumerate(loader):
        lr_path = os.path.join(save_dir, 't={}'.format(t), '{:04d}_LR.png'.format(img_index))
        gt_path = os.path.join(save_dir, 't={}'.format(t), '{:04d}_GT.png'.format(img_index))
        write_png(quantize(x).squeeze(0), gt_path)
        write_png(quantize(y).squeeze(0), lr_path)
        for t in args.temp:
            #print('batch : {}, t : {}'.format(img_index, t))
            for sample_index in range(args.samples_per_image):
                with torch.no_grad():
                    xsr = super_resolve(cvae, y, t=t, cem=cem_net, range=255)
                path = os.path.join(save_dir, 't={}'.format(t), '{:04d}_sample_{:02d}.png'.format(img_index, sample_index))
                xsr = quantize(xsr, range=1).cpu().squeeze(0)
                write_png(xsr, path)
        if img_index + 1 >= args.nmax:
            break
    print('samples saved in {}'.format(save_dir))