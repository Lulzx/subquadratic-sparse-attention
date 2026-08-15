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

### Current LFM2.5 causal donor

The default is the instruction-tuned
[LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M). Its hybrid decoder has six
full-attention layers interleaved with ten convolution layers; the default target is
the last attention layer, index 14. Run a minimal local smoke test with:

```bash
python3 mlx_donor_router.py \
  --seq-len 64 --window 8 --sink-tokens 2 \
  --train-segments 1 --eval-segments 1 --steps 1 \
  --output runs/lfm2.5-causal-donor-smoke.safetensors
```

Use `--model LiquidAI/LFM2.5-350M-Base` for the base-model ablation. The historical
SmolLM2 protocol below remains the reproducible three-seed baseline.

### One-layer LFM2.5 sparse conversion

First produce router checkpoints for seeds 0, 1, and 2 with the causal-donor command
above using the full 200-step protocol. Then align the sparse layer replacement:

```bash
python3 mlx_lfm_replacement.py \
  --layer 14 --window 32 --tables 8 --bits 8 --members 4 --probes 1 \
  --alignment-dataset wikitext2 --train-segments 32 --eval-segments 8 \
  --quality-dataset wikitext2 --quality-segments 16 \
  --steps 1000 --lr 1e-5 --seed 0 \
  --output runs/lfm2.5-layer14-replacement-wikitext-seed0.safetensors
```

Repeat with seeds 1 and 2. The runner uses disjoint
[WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext) train,
validation, and test splits. At replacement gate zero it executes the original dense
attention and asserts exact loss equality. At gate one it skips the dense branch and
executes the sparse replacement. The current path evaluates parallel causal prefill;
incremental sparse decoding remains unimplemented.

### Two-layer composition and joint recovery

Train layer 12 with the same router and replacement commands by adding `--layer 12`
and layer-specific output paths. Measure both independently converted layers together:

```bash
python3 mlx_lfm_multilayer_eval.py --layers 12,14 --seed 0
```

The historical recovery run trained only their sparse branches against cached dense
final hidden states:

```bash
python3 mlx_lfm_joint_recovery.py \
  --layers 12,14 --steps 500 --lr 1e-6 --seed 0
```

Repeat both commands for seeds 1 and 2. The joint stage freezes the complete donor,
including dense fallback branches, and unfreezes only the copied sparse Q/K/V/O
projections. Its zero-gate check must remain exactly equal to dense loss.

The larger audit showed that final-hidden recovery was insufficient. The current
quality-recovery command distills the dense model's top-64 next-token distribution on
an equal mixture of WikiText and PG-19 segments:

```bash
python3 mlx_lfm_joint_recovery.py \
  --layers 12,14 --objective kl --teacher-topk 64 \
  --train-segments 256 --eval-segments 64 \
  --pg19-train-segments 256 --pg19-eval-segments 64 \
  --steps 500 --lr 1e-6 --seed 0 \
  --checkpoint-template \
    'runs/lfm2.5-layer{layer}-joint-12-14-seed{seed}.safetensors' \
  --output-template \
    'runs/lfm2.5-layer{layer}-joint-kl-12-14-seed{seed}.safetensors'
```

Repeat for seeds 1 and 2. Teacher probabilities are cached on the host in float16, so
the run peaks near 1.61 GB instead of retaining a full-vocabulary device cache.

Run the larger paired quality gate before converting another layer:

```bash
python3 mlx_lfm_quality_eval.py \
  --layers 12,14 --tokens-per-corpus 65536 --bootstrap-samples 10000 \
  --batch-size 4 --seed 0 \
  --checkpoint-template \
    'runs/lfm2.5-layer{layer}-joint-kl-12-14-seed{seed}.safetensors'
```

Repeat for seeds 1 and 2. The command evaluates WikiText-2 test and the script-free
`emozilla/pg19` validation mirror, reports paired bootstrap intervals, and uses jointly
recovered checkpoints by default. A smaller result must not override this audit.

### Paired behavior and retrieval-router gate

Measure dense and sparse generation on exact, lexical-mismatch, and variable-length
retrieval prompts:

```bash
python3 mlx_lfm_behavior_eval.py \
  --checkpoint-template \
    'runs/lfm2.5-layer{layer}-joint-kl-12-14-seed{seed}.safetensors' \
  --lengths 256,512,1024 --positions 0.1,0.5,0.9 \
  --tasks exact,lexical,variable --max-new-tokens 16 --seed 0
```

Train the hash routers with streamed source-position supervision:

```bash
python3 mlx_lfm_retrieval_router.py \
  --checkpoint-template \
    'runs/lfm2.5-layer{layer}-joint-kl-12-14-seed{seed}.safetensors' \
  --output-template \
    'runs/lfm2.5-layer{layer}-retrieval-router-12-14-seed{seed}.safetensors' \
  --lengths 256,512 --positions 0.1,0.5 \
  --steps 300 --lr 3e-4 --memory-limit-mb 1400 \
  --cache-limit-mb 128 --seed 0
```

Rerun the behavior command with the retrieval-router checkpoint template, then rerun
the 65,536-token quality gate. The trainer streams hidden states rather than caching
them; the reference seed-0 run peaks at 1.24 GB MLX memory.

### Historical SmolLM2 baseline

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

### LFM2.5 semantic embedding router

Install the separate PyTorch/MPS environment dependencies, then distill the frozen
bidirectional embedding geometry into shared binary projections:

```bash
python3 -m pip install -r requirements-embedding.txt
python3 lfm_embedding_router.py \
  --tables 8 --bits 8 --radius 2 --steps 300 --seed 0 \
  --output runs/lfm2.5-embedding-router-seed0.json
```

The runner pins the model's remote-code revision and includes a narrow compatibility
shim for the `seq_idx` keyword introduced by Transformers 5. It uses headings as
queries and their Markdown section bodies as positive document blocks. This is a
small block-routing plumbing benchmark; it is not an LM benchmark or a general
semantic-retrieval result. Repeat with seeds 1 and 2 before drawing comparative
conclusions.

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
