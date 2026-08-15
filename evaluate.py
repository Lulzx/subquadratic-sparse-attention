import argparse
import json

import torch

from ssa.experiment import device_name, evaluate
from ssa.model import TinyLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--lengths", default="1024,2048,4096,8192,16384")
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = device_name(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyLM(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    train_len = checkpoint["train_len"]
    for length in map(int, args.lengths.split(",")):
        result = evaluate(model, length, train_len, args.batches, args.batch, args.seed + length, device)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
