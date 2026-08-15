# Reproduction guide

## MLX environment

MLX requires Apple silicon. From a clean checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mlx.txt
```

The recorded environment used Python 3.13, MLX 0.30.0, and NumPy 2.2.6.

## Correctness

```bash
python3 mlx_tests.py
python3 mlx_selector.py --lengths 1024,4096,16384 --repeats 2
```

Expected properties:

- selected attention matches NumPy within `2e-5` and typically below `9e-7`;
- chunked attention matches unchunked attention;
- exact-needle selector recall is 100% in the seeded benchmark;
- `selected_per_query` is 32.

## Training

```bash
python3 mlx_train.py \
  --steps 300 \
  --seq-len 128 \
  --batch 16 \
  --width 64 \
  --layers 2 \
  --heads 4 \
  --window 32 \
  --output runs/mlx-ssa-128.safetensors
```

The command writes weights and a matching JSON configuration under `runs/`. Generated runs are ignored by Git.

The seeded reference run reached 100% current-batch accuracy by step 250 and 99.92% held-out accuracy after step 300. Exact timings and optimization trajectories can vary by hardware and MLX version.

## Length extrapolation

```bash
python3 mlx_evaluate.py \
  runs/mlx-ssa-128.safetensors \
  --lengths 128,256,512,1024 \
  --batch 4 \
  --batches 16

python3 mlx_evaluate.py \
  runs/mlx-ssa-128.safetensors \
  --lengths 2048,4096 \
  --batch 1 \
  --batches 8
```

The split commands keep memory conservative at longer contexts.

## Benchmarks

Selector only:

```bash
python3 mlx_selector.py
```

Projected selected attention:

```bash
python3 mlx_attention_bench.py
```

Both commands default to a maximum of 16K. Do not override the cap without reading [memory safety](memory-safety.md).

## Model-card arithmetic

```bash
python3 replicate.py
```

This requires NumPy but does not require MLX or the model weights.

## Optional PyTorch reference

```bash
pip install -r requirements-pytorch.txt
python3 tests.py
python3 bench.py --lengths 256,512,1024
```

The PyTorch benchmark skips dense attention above 8K by default and SSA above 16K. The shorter command above is the recommended smoke test.
