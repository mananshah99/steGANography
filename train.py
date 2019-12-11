#!/usr/bin/env python3
import argparse
import json
import os
from time import time

import torch

from models import *
from critics import BasicCritic, DenseCritic, ResidualCritic
from decoders import DenseDecoder, BasicDecoder
from encoders import BasicEncoder, DenseEncoder, ResidualEncoder
from loader import DataLoader


def main():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=str(int(time())), type=str)

    parser.add_argument('--encoder', default="dense", type=str)
    parser.add_argument('--decoder', default="dense", type=str)
    parser.add_argument('--critic', default="basic", type=str)
    parser.add_argument('--perceptual', default=False, action="store_true")

    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--data_depth', default=1, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--dataset', default="div2k", type=str)
    parser.add_argument('--output', default=False, type=str)
    parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    train = DataLoader(os.path.join("data", args.dataset, "train"), shuffle=True, num_workers=0)
    validation = DataLoader(os.path.join("data", args.dataset, "val"), shuffle=False, num_workers=0)

    encoder = {
        "basic": BasicEncoder,
        "residual": ResidualEncoder,
        "dense": DenseEncoder,
    }[args.encoder]

    decoder = {
        "basic": BasicDecoder,
        "dense": DenseDecoder,
    }[args.decoder]

    critic = {
        "basic": BasicCritic,
        "residual": ResidualCritic,
        "dense": DenseCritic,
    }[args.critic]

    model = SteganoGAN(
        perceptual_loss=args.perceptual,
        data_depth=args.data_depth,
        encoder=encoder,
        decoder=decoder,
        critic=critic,
        hidden_size=args.hidden_size,
        cuda=True,
        verbose=True,
        log_dir=os.path.join('models', args.name)
    )

    with open(os.path.join("models", args.name, "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    model.fit(train, validation, epochs=args.epochs)
    model.save(os.path.join("models", args.name, "weights.bin"))
    if args.output:
        model.save(args.output)

if __name__ == '__main__':
    main()
