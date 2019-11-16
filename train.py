#!/usr/bin/env python3
import argparse
import json
import os
from time import time

import torch

from models import SteganoGAN
from critics import BasicCritic
from decoders import DenseDecoder
from encoders import BasicEncoder, DenseEncoder, ResidualEncoder
from loader import DataLoader


def main():
    torch.manual_seed(42)
    timestamp = int(time())

    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder', default="basic", type=str)
    parser.add_argument('--decoder', default="basic", type=str)
    parser.add_argument('--critic', default="basic", type=str)

    parser.add_argument('--epochs', default=4, type=int)

    parser.add_argument('--data_depth', default=1, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--dataset', default="div2k", type=str)
    parser.add_argument('--output', default=False, type=str)
    args = parser.parse_args()

    train = DataLoader(os.path.join("data", args.dataset, "train"), shuffle=True)
    validation = DataLoader(os.path.join("data", args.dataset, "val"), shuffle=False)

    encoder = {
        "basic": BasicEncoder,
        "residual": ResidualEncoder,
        "dense": DenseEncoder,
    }[args.encoder]

    decoder = {
        "basic": BasicDecoder,
        "dense:": DenseDecoder,
    }[args.decoder]

    critic = {
        "basic": BasicCritic
    }[args.critic]

    model = SteganoGAN(
        data_depth=args.data_depth,
        encoder=encoder,
        decoder=decoder,
        critic=critic,
        hidden_size=args.hidden_size,
        cuda=False,
        verbose=True,
        log_dir=os.path.join('models', timestamp)
    )

    with open(os.path.join("models", timestamp, "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    model.fit(train, validation, epochs=args.epochs)
    model.save(os.path.join("models", timestamp, "weights.bin"))
    if args.output:
        model.save(args.output)

if __name__ == '__main__':
    main()
