# SubQ model-card claim audit

Source: [SubQ-1.1-Small model card](https://subq.ai/docs/subq-1-1-small-model-card.pdf).

## Scope

The model weights, public API, SSA implementation, layer dimensions used for absolute FLOP counts, and benchmark harness are not publicly available. This prevents independent reruns of the report's model-quality and wall-clock claims.

The repository therefore separates:

1. arithmetic that can be reconstructed from published tables;
2. internal consistency checks;
3. empirical claims that require inaccessible artifacts.

Run the audit with:

```bash
python3 replicate.py
```

## Reconstructed successfully

- Figure 3's dense operation counts equal `n² / 2` token-pair comparisons: about 8.6B at 128K, 549B at 1M, and 2.2T at 2M.
- Table 1 is consistent, up to reported rounding, with quadratic dense scaling and linear SSA scaling.
- The 64.5× headline at 1M is `252 / 3.9 ≈ 64.6`.
- The Table 2 indexer/attention crossover solves to roughly 51K, consistent with the report's approximately 52K statement.
- Table 3's DSA layer totals equal indexer cost plus sparse-attention cost.
- The stated 0.13% attended-pair fraction at 12M follows from the Table 1 scaling.
- The reported 56× timing ratio is `54,164 ms / 966 ms ≈ 56.07`.

The script currently reports 67 of 76 checks within its configured tolerances.

## Important qualifications

### “Nearly 1,000×”

An attended fraction of 0.13% corresponds to approximately `1 / 0.0013 ≈ 769×` fewer pairs, or about 775× using the table-derived fraction. Calling this “nearly 1,000×” is generous marketing language.

### Indexer scaling

The published Lightning Indexer column is not a pure `c n²` curve anchored at 128K. It fits `c n² + b n` closely, which is compatible with quadratic scoring plus linear overhead. The prose is asymptotically correct, while the displayed numbers include more than a pure quadratic term.

### NIAH inconsistency

The report text and abstract state 98% at the longest 6M/12M settings, while a figure caption states 100% at those lengths. Those statements cannot all be correct. With 50 examples, 98% corresponds to 49 successful retrievals.

### Absolute FLOP constants

Scaling can be checked, but the absolute Table 1 FLOP constants cannot be independently derived without the exact layer dimensions and counting convention. Figure 3 counts token pairs; Table 1 counts FLOPs, so they are not directly interchangeable.

## Not independently reproducible

- RULER 99.12 at 128K;
- NIAH accuracy at 1M–12M;
- GPQA 85.4;
- LiveCodeBench 89.7 pass@4;
- AutomationBench Finance 13%;
- 966 ms SSA versus 54,164 ms FlashAttention-2;
- the claim that model quality is retained after replacing the donor model's dense attention.

These may be valid results; they are simply not independently testable from the released report.

## Relationship to this project

This repository does not label its synthetic MQAR results as a replication of the inaccessible model benchmarks. It independently tests whether one disclosed requirement—content-dependent, bounded-cost, causal retrieval—can be achieved with a small inspectable architecture.
