import argparse
import numpy as np
import imresize
import imresize_CEM
from prepare_data_pkl import to_pklv4, to_pklv4_1pct
from tqdm import tqdm
import os
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", default="")
    parser.add_argument("--scale", type=int, default=4, help="downscaling factor (HR / LR)")
    parser.add_argument("--CEM", action="store_true")
    parser.add_argument("--save_HR", action="store_true")
    parser.add_argument("--outdir", default="datasets")
    args = parser.parse_args()

    # string description of resize method used
    desc_resize = 'CEMBic' if args.CEM else 'MBic'
    os.makedirs(args.outdir, exist_ok=True)
    X = np.load(args.npy_path, mmap_mode='r+')
    # use the same split as in vdvae original implementation
    np.random.seed(5)
    tr_va_split_indices = np.random.permutation(X.shape[0])
    X_train = X[tr_va_split_indices[:-7000]]
    X_valid = X[tr_va_split_indices[-7000:]]

    # validation
    # list of HR images
    hrs = []
    # list of LR images
    lrs = []
    # resize
    for x in tqdm(X_valid):
        if args.save_HR:
            hrs.append(x)
        if args.CEM:
            y = imresize_CEM.imresize(x*1., scale_factor=[1./args.scale], kernel='cubic')
            y = np.around(y.clip(0, 255)).astype(np.uint8)
        else:
            y = imresize.imresize(x, scalar_scale = 1./args.scale)
        lrs.append(y)

    # save to pickle
    if args.save_HR:
        hrs_path = os.path.join(args.outdir, "FFHQ256", 'FFHQ256_{}_va.pklv4'.format(desc_resize))
        to_pklv4(hrs, hrs_path, vebose=True)
        to_pklv4_1pct(hrs, hrs_path, vebose=True)

    lrs_path = os.path.join(args.outdir, "FFHQ256", 'FFHQ256_{}_va_X{}.pklv4'.format(desc_resize, args.scale))
    to_pklv4(lrs, lrs_path, vebose=True)
    to_pklv4_1pct(lrs, lrs_path, vebose=True)

    # list of HR images
    hrs = []
    # list of LR images
    lrs = []
    # resize
    for x in tqdm(X_train):
        if args.save_HR:
            hrs.append(x)
        if args.CEM:
            y = imresize_CEM.imresize(x*1., scale_factor=[1./args.scale], kernel='cubic')
            y = np.around(y.clip(0, 255)).astype(np.uint8)
        else:
            y = imresize.imresize(x, scalar_scale = 1./args.scale)
        lrs.append(y)

    # save to pickle
    if args.save_HR:
        hrs_path = os.path.join(args.outdir, "FFHQ256", 'FFHQ256_{}_tr.pklv4'.format(desc_resize))
        to_pklv4(hrs, hrs_path, vebose=True)
        to_pklv4_1pct(hrs, hrs_path, vebose=True)

    lrs_path = os.path.join(args.outdir, "FFHQ256", 'FFHQ256_{}_tr_X{}.pklv4'.format(desc_resize, args.scale))
    to_pklv4(lrs, lrs_path, vebose=True)
    to_pklv4_1pct(lrs, lrs_path, vebose=True)