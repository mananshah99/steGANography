#!/usr/bin/env python3
import argparse
import json
import os
from time import time

import torch

from models import *
from loader import DataLoader


def main():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=os.path.abspath)
    parser.add_argument('--dataset', default="div2k", type=str)
    parser.add_argument('--transform', default=None, type=str)
    parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    validation = DataLoader(os.path.join("data", args.dataset, "val"), shuffle=False, num_workers=0)
    model = SteganoGAN.load(args.path, args.gpu)
    metrics = {}
    model._validate(validation, metrics, transform=args.transform)
    print(metrics)

if __name__ == '__main__':
    main()
