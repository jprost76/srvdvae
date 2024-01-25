import glob
import os
from pathlib import Path
import torch
import numpy as np
import yaml
from PIL import Image
from cvdvae import CVAE
from kornia.filters import get_gaussian_kernel2d

class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


def load_cvdvae(conf_path, conf_name, state_dict_path=None, map_location=None):
    H = Hyperparams()
    with open(conf_path) as file:
        H.update(yaml.safe_load(file)[conf_name])
    vae = CVAE(H)
    if state_dict_path:
        if map_location:
            state_dict = torch.load(state_dict_path, map_location=map_location)
        else:
            state_dict = torch.load(state_dict_path)
        new_state_dict = vae.state_dict()
        new_state_dict.update(state_dict)
        vae.load_state_dict(new_state_dict)
    return vae


def filter_state_dict(state_dict, pattern):
    """
    filter_state_dict({'encoder.block_enc1' : ...,'encoder.block_enc2' : ..., 'decoder.block_dec'}, 'encoder') = {'block_enc1' : ..., 'block_enc2' : ..., }
    """
    # remove pattern + the next character '.' to the key
    return {k[len(pattern)+1:] : v for k, v in state_dict.items() if k.startswith(pattern)}


def load_image_tensor(path):
    """
    load a image as a torch tensor of type uint8 and shape (1,C,H,W)
    """
    img = Image.open(path)
    t = torch.tensor(np.asarray(img)).permute(2, 0, 1).unsqueeze(0)
    return t


def save_image_tensor(t, path):
    """
    save a uint8 torch image tensor to path
    """
    im = Image.fromarray(t.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    im.save(path)


def load_low_pass_kernel(ds_factor):
    """
    load the gaussian filter applied to the image before sub-sampling
    scale : hr / lr 
    """
    sigma = max(ds_factor, 1.0) 
    ks = int(2.0 * 2 * sigma)
    if (ks % 2) == 0:
        ks = ks + 1
    kernel_size = (ks, ks)
    # gaussian kernel
    g_kernel = get_gaussian_kernel2d(kernel_size, (sigma, sigma))
    g_kernel_np = g_kernel.numpy() / g_kernel.numpy().sum()
    return g_kernel_np 

def find_best_ckpt_and_config(model_path):
    def get_val_loss(path):
        b = os.path.basename(path)
        val_loss = b.split("loss_val=")[1].split('.ckpt')[0].split('-')[0]
        if val_loss == "nan":
            return np.inf
        else:
            return float(val_loss)
    ckpt_list = glob.glob('{}/**/*.ckpt'.format(model_path), recursive=True)
    if not ckpt_list == []:
        ckpt_list = list(filter(os.path.isfile, ckpt_list))
        val_losses = [get_val_loss(p) for p in ckpt_list]
        best_ckpt = ckpt_list[np.argmin(val_losses)]
        p = Path(best_ckpt)
        config = os.path.join(*p.parts[:-2],'config.yaml')
        if not os.path.isfile(config):
            config = os.path.join(*p.parts[:-1],'config.yaml')
        return best_ckpt, config
    else: 
        return None, None
    
def super_resolve(cvae, y, t, range=255, cem=None):
    """_summary_

    Args:
        cvae (_type_): _description_
        ynp (_type_): _description_
        t (_type_): _description_
        range (int, optional): _description_. Defaults to 255.
        cem (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    ycuda = y.cuda()
    if range != 255:
        print('correct range') # input range should be [0, 255]
        ycuda = (ycuda / range) * 255
    xsr, _ = cvae.forward_samples(ycuda, t=t, quantize=False)
    xsr = (xsr + 1) / 2
    if cem:
        ynp = y.squeeze(0).numpy()
        xsr_np = xsr.squeeze(0).permute(1, 2, 0).cpu().numpy()
        xsr_np = cem.Enforce_DT_on_Image_Pair(ynp*1., xsr_np).clip(0, 1)
        out = torch.tensor(xsr_np).permute(2, 0, 1)
    else:
         out = xsr.squeeze()
    return out

if __name__ == "__main__":
    path = "/home/jprost/cvdvae/saved_models/SR_x16_FFHQ256_bic_002"
    print(find_best_ckpt_and_config(path))

