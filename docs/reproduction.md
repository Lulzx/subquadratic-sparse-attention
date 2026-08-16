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

Continue each original router with the inference-aligned objective. The default
teacher target is normalized attention probability times value-vector L2 norm; the
top 32 targets are assigned four per table and trained from straight-through hard
Hamming distance:

```bash
for seed in 0 1 2; do
  python3 mlx_donor_router.py \
    --init-checkpoint "runs/lfm2.5-layer14-router-seed${seed}.safetensors" \
    --layer 14 --seq-len 256 --stride 192 \
    --window 32 --sink-tokens 4 \
    --tables 8 --bits 8 --members 4 --probes 1 \
    --steps 200 --lr 0.003 \
    --retrieval-weight 1 --retrieval-topk 32 \
    --retrieval-positive-weight 10 \
    --train-segments 8 --eval-segments 4 --seed "$seed" \
    --output "runs/lfm2.5-layer14-hard-hamming-seed${seed}.safetensors"
done
```

Audit agreement, occupancy, table dependence, distance-conditioned recall, candidate
counts, and retained teacher contribution for the resulting routers with:

```bash
python3 mlx_router_audit.py \
  runs/lfm2.5-layer14-hard-hamming-seed0.safetensors \
  runs/lfm2.5-layer14-hard-hamming-seed1.safetensors \
  runs/lfm2.5-layer14-hard-hamming-seed2.safetensors \
  --eval-segments 4 \
  --output runs/lfm2.5-layer14-learned-router-audit.json \
  --markdown-output runs/lfm2.5-layer14-learned-router-audit.md
```

The trainer records SHA-256 hashes for new training and evaluation corpora. The auditor
also hashes checkpoints, metadata, and evaluation files, requires compatible router
configurations, and evaluates every seed on the same held-out segments. Its default
1,400 MB working-set limit peaked at 838 MB for this three-seed audit.

To reproduce the fixed-budget bucket-history comparison, audit the inference-aligned
checkpoints once with the default recent tail and once with the opt-in midpoint-history
slot:

```bash
python3 mlx_router_audit.py \
  runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed0.safetensors \
  runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed1.safetensors \
  runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed2.safetensors \
  --eval-segments 4 \
  --output runs/lfm2.5-layer14-research-aligned-audit.json \
  --markdown-output runs/lfm2.5-layer14-research-aligned-audit.md

python3 mlx_router_audit.py \
  runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed0.safetensors \
  runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed1.safetensors \
  runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed2.safetensors \
  --eval-segments 4 \
  --member-policy hybrid --history-fraction 0.5 \
  --output runs/lfm2.5-layer14-history-0.5-audit.json \
  --markdown-output runs/lfm2.5-layer14-history-0.5-audit.md
```

Compare the two policies in the collision-heavy exact-needle selector benchmark. Both
commands select exactly 32 anchors per query:

```bash
python3 mlx_selector.py \
  --lengths 1024,2048,4096,8192,16384 \
  --tables 8 --bits 8 --members 4 --probes 1 \
  --anchors-only --repeats 9 \
  --member-policy recent

python3 mlx_selector.py \
  --lengths 1024,2048,4096,8192,16384 \
  --tables 8 --bits 8 --members 4 --probes 1 \
  --anchors-only --repeats 9 \
  --member-policy hybrid --history-fraction 0.5
```

The same `--member-policy hybrid --history-fraction 0.5` flags are available on the
LFM replacement, multilayer, recovery, quality, behavior, and retrieval-router
commands. They affect token-bucket selection only; `--block-size` keeps the existing
completed-block policy.

Evaluate the post-hoc policy transfer on the three existing token-router checkpoints:

```bash
for seed in 0 1 2; do
  python3 mlx_lfm_behavior_eval.py \
    --checkpoint-template \
      'runs/lfm2.5-layer{layer}-retrieval-router-12-14-seed{seed}.safetensors' \
    --lengths 256,512,1024 --positions 0.1,0.5,0.9 \
    --tasks exact,lexical,variable --max-new-tokens 16 \
    --member-policy hybrid --history-fraction 0.5 \
    --seed "$seed" \
    --output \
      "runs/lfm2.5-behavior-retrieval-router-hybrid05-12-14-seed${seed}.json"
done
```

This comparison uses checkpoints trained under recent-tail selection. It tests an
inference-time policy change, not hybrid-aware router training or sparse recovery.

Retrain the routers with hybrid selection active, then repeat the behavior and paired
quality gates:

```bash
for seed in 0 1 2; do
  python3 mlx_lfm_retrieval_router.py \
    --checkpoint-template \
      'runs/lfm2.5-layer{layer}-joint-kl-12-14-seed{seed}.safetensors' \
    --output-template \
      'runs/lfm2.5-layer{layer}-hybrid05-retrieval-router-12-14-seed{seed}.safetensors' \
    --lengths 256,512 --positions 0.1,0.5 \
    --steps 300 --lr 3e-4 \
    --member-policy hybrid --history-fraction 0.5 \
    --memory-limit-mb 1400 --cache-limit-mb 128 \
    --seed "$seed" \
    --output \
      "runs/lfm2.5-retrieval-router-hybrid05-trained-12-14-seed${seed}.json"

  python3 mlx_lfm_behavior_eval.py \
    --checkpoint-template \
      'runs/lfm2.5-layer{layer}-hybrid05-retrieval-router-12-14-seed{seed}.safetensors' \
    --lengths 256,512,1024 --positions 0.1,0.5,0.9 \
    --tasks exact,lexical,variable --max-new-tokens 16 \
    --member-policy hybrid --history-fraction 0.5 \
    --seed "$seed" \
    --output \
      "runs/lfm2.5-behavior-hybrid05-trained-retrieval-router-12-14-seed${seed}.json"

  python3 mlx_lfm_quality_eval.py \
    --layers 12,14 --tokens-per-corpus 65536 \
    --bootstrap-samples 10000 --batch-size 4 \
    --checkpoint-template \
      'runs/lfm2.5-layer{layer}-hybrid05-retrieval-router-12-14-seed{seed}.safetensors' \
    --member-policy hybrid --history-fraction 0.5 \
    --seed "$seed" \
    --output \
      "runs/lfm2.5-quality-hybrid05-trained-retrieval-router-65k-seed${seed}.json"
done
```

This retrains only the hash routers. The parent sparse Q/K/V/O branches are the
existing joint-KL checkpoints recovered under recent-tail selection.

### Expanded retrieval-generalization gate

Create the tokenizer-derived manifest. It contains all four intended lengths even
though the local memory probe approves only the 1K slice:

```bash
python3 mlx_lfm_retrieval_generalization.py \
  --make-manifest runs/lfm2.5-retrieval-generalization-manifest.json \
  --lengths 1024,4096,8192,16384 --positions 0.1,0.5,0.9
```

Evaluate dense once, then each sparse checkpoint seed on the same 72 1K cases. Reports
are rewritten after every case and can be continued with `--resume` after an
interruption.

```bash
python3 mlx_lfm_retrieval_generalization.py \
  --manifest runs/lfm2.5-retrieval-generalization-manifest.json \
  --variant dense --mode dense --only-lengths 1024 \
  --output runs/lfm2.5-generalization-dense-1k.json

for seed in 0 1 2; do
  python3 mlx_lfm_retrieval_generalization.py \
    --manifest runs/lfm2.5-retrieval-generalization-manifest.json \
    --variant recent_k32 --mode sparse \
    --checkpoint-template \
      'runs/lfm2.5-layer{layer}-retrieval-router-12-14-seed{seed}.safetensors' \
    --members 4 --member-policy recent --seed "$seed" --only-lengths 1024 \
    --output "runs/lfm2.5-generalization-recent-k32-seed${seed}-1k.json"

  python3 mlx_lfm_retrieval_generalization.py \
    --manifest runs/lfm2.5-retrieval-generalization-manifest.json \
    --variant hybrid_k32 --mode sparse \
    --checkpoint-template \
      'runs/lfm2.5-layer{layer}-hybrid05-retrieval-router-12-14-seed{seed}.safetensors' \
    --members 4 --member-policy hybrid --history-fraction 0.5 \
    --seed "$seed" --only-lengths 1024 \
    --output "runs/lfm2.5-generalization-hybrid-k32-seed${seed}-1k.json"

  python3 mlx_lfm_retrieval_generalization.py \
    --manifest runs/lfm2.5-retrieval-generalization-manifest.json \
    --variant block_k64 --mode sparse \
    --checkpoint-template \
      'runs/lfm2.5-layer{layer}-block4-retrieval-sft-12-14-seed{seed}.safetensors' \
    --members 2 --block-size 4 --seed "$seed" --only-lengths 1024 \
    --output "runs/lfm2.5-generalization-block-k64-seed${seed}-1k.json"
done
```

Aggregate dense-pass preservation, Wilson intervals, seed variation, and task/value/
template/distance slices:

```bash
python3 mlx_lfm_retrieval_generalization_report.py \
  --manifest runs/lfm2.5-retrieval-generalization-manifest.json \
  runs/lfm2.5-generalization-dense-1k.json \
  runs/lfm2.5-generalization-recent-k32-seed{0,1,2}-1k.json \
  runs/lfm2.5-generalization-hybrid-k32-seed{0,1,2}-1k.json \
  runs/lfm2.5-generalization-block-k64-seed{0,1,2}-1k.json \
  --skip 'dense:4096:single-case probe peaked at 2819.22 MB, above the 1792 MB operating limit' \
  --skip 'recent_k32:4096:not attempted after the matched K32 hybrid probe peaked at 2482.98 MB with the same four remaining dense layers' \
  --skip 'hybrid_k32:4096:single-case probe peaked at 2482.98 MB, above the 1792 MB operating limit' \
  --skip 'block_k64:4096:not attempted because its 1K peak exceeds token routing and the same four dense layers already fail the 4K limit' \
  --skip 'dense:8192:4K already exceeded the operating limit' \
  --skip 'recent_k32:8192:4K matched architecture already exceeded the operating limit' \
  --skip 'hybrid_k32:8192:4K already exceeded the operating limit' \
  --skip 'block_k64:8192:4K matched architecture already exceeded the operating limit' \
  --skip 'dense:16384:4K already exceeded the operating limit' \
  --skip 'recent_k32:16384:4K matched architecture already exceeded the operating limit' \
  --skip 'hybrid_k32:16384:4K already exceeded the operating limit' \
  --skip 'block_k64:16384:4K matched architecture already exceeded the operating limit' \
  --output runs/lfm2.5-retrieval-generalization-1k.json \
  --markdown-output runs/lfm2.5-retrieval-generalization-1k.md
```

The final recorded report also includes explicit `--skip VARIANT:LENGTH:REASON`
metadata for 4K, 8K, and 16K. A one-case 4K dense probe peaked at 2.82 GB and a hybrid
probe at 2.48 GB, above the 1.792 GB operating limit. Do not rerun or extend these
probes on the reference 24 GB laptop; use a higher-memory machine or first convert the
remaining dense attention layers.

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

For the experimental completed-block path, train a four-token block router with one
block per table, then run retrieval SFT and evaluate with two blocks per table:

```bash
python3 mlx_lfm_retrieval_router.py \
  --checkpoint-template \
    'runs/lfm2.5-layer{layer}-retrieval-router-12-14-seed{seed}.safetensors' \
  --output-template \
    'runs/lfm2.5-layer{layer}-block4-router-12-14-seed{seed}.safetensors' \
  --block-size 4 --members 1 --steps 300 --seed 0

python3 mlx_lfm_retrieval_recovery.py \
  --checkpoint-template \
    'runs/lfm2.5-layer{layer}-block4-router-12-14-seed{seed}.safetensors' \
  --output-template \
    'runs/lfm2.5-layer{layer}-block4-retrieval-sft-12-14-seed{seed}.safetensors' \
  --objective lm --variable-values 32 --span-size 1 --block-size 4 \
  --members 1 --steps 300 --seed 0

python3 mlx_lfm_behavior_eval.py \
  --checkpoint-template \
    'runs/lfm2.5-layer{layer}-block4-retrieval-sft-12-14-seed{seed}.safetensors' \
  --block-size 4 --members 2 --max-new-tokens 16 --seed 0
```

The router is trained at one block per table (`K=32`) while evaluation reads two
(`K=64`). Report both the block budget and the expanded token budget.

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

Routing-only scan versus persistent addressing, with one-query timings and a fixed
`K=32` final budget:

```bash
python3 mlx_routing_scan_bench.py \
  --lengths 16384,32768,65536,131072,262144,524288,1048576,2097152 \
  --queries 64 --k 32 \
  --tables 4 --bits 16 --probes 2 --bucket-capacity 16 \
  --retention-policy tail \
  --warmups 4 --repeats 7 --recall-batch 1 \
  --output runs/routing-scan-addressed-c16-16k-2m.json

python3 mlx_routing_scan_report.py \
  runs/routing-scan-addressed-c16-16k-2m.json \
  --markdown-output runs/routing-scan-addressed-c16-16k-2m.md \
  --plot-output docs/assets/routing-scan-scaling.svg
```

The benchmark times query encoding plus routing and selection. It deliberately
excludes offline index construction, KV gather, attention, and model execution. The
reported bytes/query are logical index payloads, not measured DRAM traffic. Run it
alone: the full sweep peaks near 432 MB on the reference laptop. The 2M row is the
documented ceiling; larger contexts are intentionally outside this protocol.

Run the matched bucket-retention ablation sequentially. Every command uses the same
seed-stable keys, projections, and per-length query set:

```bash
for spec in '16 tail' '16 reservoir' '32 tail' '32 reservoir' '64 tail'; do
  set -- $spec
  python3 mlx_routing_scan_bench.py \
    --lengths 262144,1048576,2097152 --queries 64 --k 32 \
    --tables 4 --bits 16 --probes 2 --bucket-capacity "$1" \
    --retention-policy "$2" --warmups 4 --repeats 7 --recall-batch 1 \
    --output "runs/routing-retention-c${1}-${2}-256k-2m.json"
done

python3 mlx_bucket_retention_report.py \
  runs/routing-retention-c16-tail-256k-2m.json \
  runs/routing-retention-c16-reservoir-256k-2m.json \
  runs/routing-retention-c32-tail-256k-2m.json \
  runs/routing-retention-c32-reservoir-256k-2m.json \
  runs/routing-retention-c64-tail-256k-2m.json \
  --markdown-output runs/routing-retention-256k-2m.md \
  --plot-output docs/assets/bucket-retention-tradeoff.svg
```

The fingerprint-policy follow-ups use the same benchmark command with the
corresponding `--bucket-capacity` and `--retention-policy` values. They are recorded
as negative follow-up evidence rather than included in the five-way plot.

Projected selected attention:

```bash
python3 mlx_attention_bench.py
```

Both commands default to a maximum of 16K. Do not override the cap without reading [memory safety](memory-safety.md).

Run the unified finite-range validation harness to benchmark the selector, gathered
sparse attention, a safely capped dense reference, the real selector under matched
collision pressure, and the mutation-based causality suite:

```bash
python3 validate_scaling.py
```

It writes machine-readable JSON and a Markdown report under `runs/`. The latency
report fits `n`, `n log n`, and `n^2` models and reports doubling ratios; these are
finite-range diagnostics rather than a proof of Big-O. The capacity sweep uses exact
copied query/key vectors and multiple independent seeds, so it tests collision-tail
behavior in the implemented selector but does not substitute for learned-router
agreement, occupancy, attention-mass, or language-quality evaluation.

## Idealized hash-capacity calculation

Run the constant-memory mathematical baseline:

```bash
python3 capacity_scaling.py --self-test
python3 capacity_scaling.py --lengths 1024,4096,16384,65536,1048576
```

This calculates collision-tail survival under balanced, independent hashes. It does
not load MLX or any model weights and is an optimistic capacity baseline, not measured
learned-router recall. See [Complexity and capacity](complexity-and-capacity.md) for the
assumptions and formula.

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
