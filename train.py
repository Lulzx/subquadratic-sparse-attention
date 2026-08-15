import argparse
import json
import pathlib
import time

import numpy as np
import torch

from ssa.experiment import answer_metrics, device_name, evaluate, model_config
from ssa.model import TinyLM
from ssa.tasks import mqar_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attn", choices=["ssa", "window", "dense"], default="ssa")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--select-buckets", type=int, default=4)
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--capacity", type=int, default=512)
    p.add_argument("--codes", type=int, default=16)
    p.add_argument("--aux-window", type=int, default=512)
    p.add_argument("--chunk-q", type=int, default=1024)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--aux-every", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="runs/model.pt")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = device_name(args.device)
    config = model_config(args)
    model = TinyLM(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        x, answers = mqar_batch(args.batch, args.seq_len, rng)
        x = x.to(device)
        logits = model(x)
        task_loss, accuracy = answer_metrics(logits, answers)
        aux = torch.zeros((), device=device)
        if args.attn == "ssa" and args.aux_weight and step % args.aux_every == 0:
            aux = model.aux_loss(x)
        loss = task_loss + args.aux_weight * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        if step == 1 or step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "step": step,
                "task_loss": round(float(task_loss.detach().cpu()), 4),
                "aux_loss": round(float(aux.detach().cpu()), 4),
                "accuracy": round(float(accuracy.detach().cpu()), 4),
                "steps_per_second": round(step / elapsed, 3),
            }), flush=True)
    result = evaluate(model, args.seq_len, args.seq_len, 8, args.batch, args.seed + 1, device)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "train_len": args.seq_len, "args": vars(args)}, output)
    print(json.dumps({"checkpoint": str(output), "validation": result}, indent=2))


if __name__ == "__main__":
    main()
