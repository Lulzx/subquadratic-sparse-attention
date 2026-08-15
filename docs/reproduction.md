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
  --tables 4 \
  --members 4 \
  --probes 1 \
  --batch 16 \
  --width 64 \
  --layers 2 \
  --heads 4 \
  --window 32 \
  --output runs/mlx-ssa-128.safetensors
```

The command writes weights and a matching JSON configuration under `runs/`. Generated runs are ignored by Git.

The seeded reference run reached 100% current-batch accuracy by step 250 and 99.92% held-out accuracy after step 300. Exact timings and optimization trajectories can vary by hardware and MLX version.

To reproduce one seed of the recommended eight-table curriculum configuration:

```bash
python3 mlx_train.py \
  --steps 1200 \
  --train-lengths 128,256,512,1024 \
  --batch 16 \
  --width 64 \
  --layers 2 \
  --heads 4 \
  --window 32 \
  --tables 8 \
  --bits 16 \
  --members 2 \
  --probes 1 \
  --seed 0 \
  --output runs/mlx-ssa-8table-curriculum-seed0.safetensors
```

Repeat with seeds 1 and 2, then evaluate each checkpoint with the same held-out protocol:

```bash
python3 mlx_evaluate.py \
  runs/mlx-ssa-8table-curriculum-seed0.safetensors \
  --lengths 4096 \
  --batch 4 \
  --batches 16
```

For the fixed-budget multiprobe comparison, use four tables, two probes, and two members. For the curriculum-only control, use four tables, one probe, and four members. All three configurations have `K=32`.

The exploratory 8K curriculum frontier uses the same model with seven 300-step stages:

```bash
python3 mlx_train.py \
  --steps 2100 \
  --train-lengths 128,256,512,1024,2048,4096,8192 \
  --batch 16 \
  --width 64 \
  --layers 2 \
  --heads 4 \
  --window 32 \
  --tables 8 \
  --bits 16 \
  --members 2 \
  --probes 1 \
  --seed 0 \
  --output runs/mlx-ssa-8table-curriculum-8192-seed0.safetensors
```

Follow the memory-safety checklist before overriding the evaluator cap. The recorded 32K run first used batch 1 and one batch, then increased only the batch count after observing a safe peak.

To continue from a shorter curriculum checkpoint without retraining its earlier stages, pass `--resume`. This restores model weights and intentionally starts a fresh optimizer:

```bash
python3 mlx_train.py \
  --resume runs/mlx-ssa-8table-curriculum-4096-seed0.safetensors \
  --steps 300 \
  --train-lengths 8192 \
  --batch 1 \
  --tables 8 \
  --members 2 \
  --probes 1 \
  --output runs/mlx-ssa-8table-finetune-8192-seed0.safetensors
```

Architecture arguments must match the source checkpoint. Shape mismatches fail during weight loading rather than silently reinitializing parameters.

## Semantic-router and compressed-global experiment

The research variant is opt-in so existing checkpoints and baseline commands remain
unchanged:

```bash
python3 mlx_train.py \
  --steps 300 \
  --seq-len 128 \
  --batch 16 \
  --width 64 \
  --layers 2 \
  --heads 4 \
  --window 32 \
  --tables 8 \
  --bits 16 \
  --members 2 \
  --probes 1 \
  --semantic-router \
  --router-teacher-tokens 128 \
  --semantic-loss-weight 1.0 \
  --global-slots 4 \
  --seed 0 \
  --output runs/mlx-semantic-global-smoke-seed0.safetensors
```

This is a mechanism smoke test, not a valid comparative result. A reported ablation
must compare semantic-router-only, global-only, combined, and unchanged baseline runs
over at least three seeds with the same curriculum and selected-token budget. See the
[replication roadmap](replication-roadmap.md#stage-a-router-and-global-path-ablations).

## Pretrained donor-router distillation

The first natural-language routing experiment downloads the approximately 257 MB BF16
SmolLM2-135M base checkpoint on first use. It freezes the donor, caches layer-15 hidden
states and distant attention distributions, and trains only the hash projections:

```bash
python3 mlx_donor_router.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --layer 15 \
  --seq-len 256 \
  --stride 128 \
  --window 32 \
  --sink-tokens 4 \
  --train-segments 32 \
  --eval-segments 4 \
  --tables 8 \
  --bits 8 \
  --members 4 \
  --probes 1 \
  --steps 2000 \
  --alignment-weight 1.0 \
  --balance-weight 10 \
  --seed 0 \
  --output runs/smollm2-layer15-router-seed0.safetensors
```

Repeat with seeds 1 and 2. The JSON beside each checkpoint records corpus files, donor
configuration, and before/after hard and continuous metrics. Generated weights and
metadata under `runs/` remain ignored by Git.

The hard metrics use bucket lookup with `min_distance=window`, so local-window tokens
cannot consume the distant sparse budget. Continuous top-k is reported only as a
diagnostic ceiling and must not be confused with subquadratic hash lookup.

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
