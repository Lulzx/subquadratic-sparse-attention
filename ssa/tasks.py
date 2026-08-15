import numpy as np

KEYS = 512
VALS = 1024
FILLER = 1536
VOCAB = 2048
QUERY = VOCAB - 1


def mqar_numpy_batch(batch, seq_len, rng, query_frac=0.35):
    seqs, ans = [], []
    max_pairs = min(KEYS, max(2, seq_len // 8))
    for _ in range(batch):
        n_pairs = max(2, min(max_pairs, int(seq_len // 8)))
        keys = rng.choice(KEYS, size=n_pairs, replace=False)
        vals = rng.integers(VALS, size=n_pairs) + KEYS
        n_fill = max(1, (seq_len - 2 * n_pairs) // (2 * n_pairs))
        body = []
        for i in range(n_pairs):
            body += [int(keys[i]), int(vals[i])]
            body += list(rng.integers(FILLER, QUERY, size=int(rng.integers(0, 2 * n_fill))))
        n_query = max(1, int(n_pairs * query_frac))
        q_idx = rng.permutation(n_pairs)[:n_query]
        queries = []
        for i in q_idx:
            queries += [QUERY, int(keys[i]), int(vals[i])]
        sep = list(rng.integers(FILLER, QUERY, size=max(0, seq_len - len(body) - len(queries))))
        full = (body + sep + queries)[:seq_len]
        q_start = len(body) + len(sep)
        batch_ans = []
        for j in range(q_start + 2, len(full), 3):
            batch_ans.append((j, int(full[j])))
        seqs.append(full)
        ans.append(batch_ans)
    return np.array(seqs, dtype=np.int64), ans


def mqar_batch(batch, seq_len, rng, query_frac=0.35):
    import torch

    sequences, answers = mqar_numpy_batch(batch, seq_len, rng, query_frac)
    return torch.from_numpy(sequences), answers
