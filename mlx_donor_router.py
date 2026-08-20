import argparse
import hashlib
import json
import math
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from ssa.mlx_selector import hash_codes, probe_codes, select_indices_qk


DEFAULT_TRAIN_FILES = [
    "README.md",
    "docs/architecture.md",
    "docs/design-history.md",
    "docs/experiments.md",
    "docs/reproduction.md",
]
DEFAULT_EVAL_FILES = ["docs/limitations.md", "docs/model-card-audit.md"]


def language_body(model):
    """Return the text decoder body for text-only and multimodal MLX-LM models."""
    if hasattr(model, "language_model"):
        return model.language_model.model
    return model.model


class DonorHashRouter(nn.Module):
    def __init__(self, width, tables, bits):
        super().__init__()
        scale = 1.0 / math.sqrt(width)
        projection = mx.random.normal((width, tables * bits)) * scale
        self.query_projection = projection
        self.key_projection = projection + mx.zeros_like(projection)
        self.tables = tables
        self.bits = bits

    def logits(self, x):
        shape = (*x.shape[:-1], self.tables, self.bits)
        return (x @ self.query_projection).reshape(shape), (x @ self.key_projection).reshape(shape)

    @staticmethod
    def straight_through_sign(logits):
        continuous = mx.tanh(logits)
        binary = mx.where(logits >= 0, mx.ones_like(logits), -mx.ones_like(logits))
        return mx.stop_gradient(binary - continuous) + continuous

    def loss(self, x, teacher_probability, query_start, window, sink_tokens,
             alignment_weight, balance_weight, decorrelation_weight=0.0,
             retrieval_weight=1.0, retrieval_topk=32,
             retrieval_positive_weight=10.0):
        query_logits, key_logits = self.logits(x)
        query_logits = query_logits[:, query_start:]
        # Forward values are the exact +/-1 bits used by Hamming lookup while
        # backward gradients flow through tanh.
        query_code = self.straight_through_sign(query_logits)
        key_code = self.straight_through_sign(key_logits)
        student_by_table = mx.einsum("bqtd,bktd->bqkt", query_code, key_code)
        student_scores = mx.max(student_by_table, axis=-1) / math.sqrt(self.bits)
        query_positions = mx.arange(query_start, x.shape[1]).reshape(-1, 1)
        key_positions = mx.arange(x.shape[1]).reshape(1, -1)
        eligible = (key_positions < query_positions - window) & (key_positions >= sink_tokens)
        student_scores = mx.where(eligible, student_scores, mx.array(-1e9, student_scores.dtype))
        student_log_probability = student_scores - mx.logsumexp(
            student_scores, axis=-1, keepdims=True
        )
        cross_entropy = -mx.mean(
            mx.sum(teacher_probability * student_log_probability, axis=-1)
        )

        # The deployed selector succeeds only when at least one complete table
        # address agrees. Train the probability of that discrete event instead
        # of relying only on a smooth Hamming-similarity distribution.
        hamming_distance = (self.bits - student_by_table) / 2.0
        # Positive only for an exact hard-code match, with a differentiable
        # straight-through distance. A one-bit mismatch is already negative.
        table_match_logit = 2.0 * (0.5 - hamming_distance)
        label_count = min(retrieval_topk, teacher_probability.shape[-1])
        # Allocate teacher targets across tables in decreasing-importance order.
        # With topk == tables * members, each table learns exactly the number of
        # positives its bounded bucket tail can retain instead of all tables
        # collapsing onto the same popular keys.
        teacher_order = mx.argsort(-teacher_probability, axis=-1)
        teacher_rank = mx.argsort(teacher_order, axis=-1)
        table_index = mx.arange(self.tables).reshape(1, 1, 1, -1)
        retrieval_labels = (
            (teacher_probability[..., None] > 0.0)
            & (teacher_rank[..., None] < label_count)
            & ((teacher_rank[..., None] % self.tables) == table_index)
        ).astype(table_match_logit.dtype)
        retrieval_weights = 1.0 + (
            retrieval_positive_weight - 1.0
        ) * retrieval_labels
        retrieval_terms = retrieval_weights * (
            mx.logaddexp(mx.zeros_like(table_match_logit), table_match_logit)
            - retrieval_labels * table_match_logit
        )
        eligible_float = eligible[..., None].astype(retrieval_terms.dtype)
        retrieval_bce = mx.sum(retrieval_terms * eligible_float) / mx.maximum(
            mx.sum(eligible_float) * self.tables,
            mx.array(1.0, retrieval_terms.dtype),
        )

        top_key = mx.argmax(teacher_probability, axis=-1)
        batch_index = mx.arange(x.shape[0]).reshape(-1, 1)
        query_probability = mx.sigmoid(query_logits)
        key_probability = mx.sigmoid(key_logits)[batch_index, top_key]
        same_bit = (
            query_probability * key_probability
            + (1.0 - query_probability) * (1.0 - key_probability)
        )
        log_table_match = mx.sum(
            mx.log(mx.maximum(same_bit, mx.array(1e-6, same_bit.dtype))), axis=-1
        )
        alignment = -mx.mean(mx.max(log_table_match, axis=-1))
        probabilities = [mx.sigmoid(query_logits), mx.sigmoid(key_logits)]
        balance = mx.mean(mx.stack([
            mx.mean(mx.square(mx.mean(probability, axis=(0, 1)) - 0.5))
            for probability in probabilities
        ]))
        confidence = mx.mean(mx.stack([
            mx.mean(probability * (1.0 - probability)) for probability in probabilities
        ]))
        decorrelation_terms = []
        for logits in (query_logits, key_logits):
            dimensions = logits.shape[-2] * logits.shape[-1]
            values = mx.tanh(logits).reshape(-1, dimensions)
            values = values - mx.mean(values, axis=0, keepdims=True)
            covariance = (values.T @ values) / max(values.shape[0], 1)
            variance = mx.diag(covariance)
            scale = mx.sqrt(mx.maximum(
                variance[:, None] * variance[None, :],
                mx.array(1e-6, covariance.dtype),
            ))
            correlation = covariance / scale
            off_diagonal = correlation * (
                1.0 - mx.eye(correlation.shape[0], dtype=correlation.dtype)
            )
            dimensions = correlation.shape[0]
            decorrelation_terms.append(
                mx.sum(mx.square(off_diagonal))
                / max(dimensions * (dimensions - 1), 1)
            )
        decorrelation = mx.mean(mx.stack(decorrelation_terms))
        total = (
            cross_entropy + alignment_weight * alignment
            + balance_weight * balance + 0.01 * confidence
            + decorrelation_weight * decorrelation
            + retrieval_weight * retrieval_bce
        )
        return total, (
            cross_entropy,
            retrieval_bce,
            alignment,
            balance,
            confidence,
            decorrelation,
        )


class HierarchicalAttentionRouter(DonorHashRouter):
    """Use separate 64-bit codes for addressing and bounded candidate reranking."""

    def __init__(self, width, tables, bits, rerank_bytes=8):
        super().__init__(width, tables, bits)
        scale = 1.0 / math.sqrt(width)
        self.rerank_tables = rerank_bytes
        self.rerank_query_projection = mx.random.normal(
            (width, rerank_bytes * 8)
        ) * scale
        self.rerank_key_projection = self.rerank_query_projection + mx.zeros_like(
            self.rerank_query_projection
        )
        self.rerank_bit_weights = mx.zeros((rerank_bytes * 8,))
        self.rerank_bilinear = mx.eye(rerank_bytes * 8) / 2.0
        byte_values = np.arange(256, dtype=np.uint16)
        byte_hamming = np.unpackbits(
            np.bitwise_xor(byte_values[:, None], byte_values[None, :])
            .astype(np.uint8)[..., None],
            axis=-1,
        ).sum(axis=-1).astype(np.float32)
        self.rerank_lookup = mx.array(
            np.repeat((-byte_hamming)[None, :, :], rerank_bytes, axis=0)
        )
        decoder_width = max(128, rerank_bytes * 8)
        decoder_extra = mx.random.normal(
            (width, decoder_width - rerank_bytes * 8)
        ) * scale
        self.rerank_decoder_query = mx.concatenate(
            [self.rerank_query_projection, decoder_extra], axis=-1
        )
        decoder_embeddings = np.zeros(
            (rerank_bytes, 256, decoder_width), dtype=np.float32
        )
        byte_bits = np.unpackbits(
            np.arange(256, dtype=np.uint8)[:, None], axis=-1,
            bitorder="little",
        ).astype(np.float32) * 2.0 - 1.0
        for table in range(rerank_bytes):
            decoder_embeddings[table, :, table * 8:(table + 1) * 8] = byte_bits
        self.rerank_decoder_keys = mx.array(decoder_embeddings)
        self.rerank_distance_bias = mx.zeros((16,))
        self.rerank_query_lookup_weight = mx.random.normal(
            (width, rerank_bytes * 256)
        ) * scale
        self.rerank_query_lookup_bias = mx.zeros((rerank_bytes * 256,))
        self.retention_projection = mx.random.normal((width,)) * scale

    def rerank_logits(self, x):
        shape = (*x.shape[:-1], self.rerank_tables, 8)
        return (
            (x @ self.rerank_query_projection).reshape(shape),
            (x @ self.rerank_key_projection).reshape(shape),
        )

    def attention_rerank_loss(
        self, x, teacher_probability, query_start, window, sink_tokens,
        temperature=4.0, mass_gamma=1.0, balance_weight=10.0,
        decorrelation_weight=1.0, pairwise_weight=1.0,
        retrieval_topk=32, hard_negatives=32, pairwise_margin=2.0,
        candidate_mask=None,
        confidence_weighted=False, confidence_power=1.0,
        confidence_mix=1.0,
        bilinear=False,
        lookup=False,
        decoder=False,
        distance_bias=False,
        query_lookup=False,
    ):
        query_logits, key_logits = self.rerank_logits(x)
        query_logits = query_logits[:, query_start:]
        query_code = self.straight_through_sign(query_logits)
        key_code = self.straight_through_sign(key_logits)
        bit_weights = None
        if confidence_weighted:
            bit_weights = mx.power(mx.abs(query_logits), confidence_power)
            global_weights = mx.logaddexp(
                mx.zeros_like(self.rerank_bit_weights), self.rerank_bit_weights
            ).reshape(1, 1, self.rerank_tables, 8)
            bit_weights = bit_weights * global_weights
            bit_weights = bit_weights / mx.maximum(
                mx.mean(bit_weights, axis=(-1, -2), keepdims=True),
                mx.array(1e-6, bit_weights.dtype),
            )
            bit_weights = (
                (1.0 - confidence_mix) + confidence_mix * bit_weights
            )
        if query_lookup:
            powers = mx.power(
                mx.array(2, dtype=mx.int32), mx.arange(8, dtype=mx.int32)
            )
            key_indices = mx.sum(
                ((key_code + 1.0) / 2.0).astype(mx.int32)
                * powers.reshape(1, 1, 1, 8),
                axis=-1,
            )
            lookup = (
                x[:, query_start:] @ self.rerank_query_lookup_weight
                + self.rerank_query_lookup_bias
            ).reshape(
                x.shape[0], x.shape[1] - query_start,
                self.rerank_tables, 256,
            )
            batch_scores = []
            for batch in range(x.shape[0]):
                table_scores = []
                for table in range(self.rerank_tables):
                    table_scores.append(
                        lookup[
                            batch, :, table, key_indices[batch, :, table]
                        ].T
                    )
                batch_scores.append(mx.sum(mx.stack(table_scores, axis=-1), axis=-1))
            distance = -mx.stack(batch_scores, axis=0)
        elif decoder:
            powers = mx.power(
                mx.array(2, dtype=mx.int32), mx.arange(8, dtype=mx.int32)
            )
            key_indices = mx.sum(
                ((key_code + 1.0) / 2.0).astype(mx.int32)
                * powers.reshape(1, 1, 1, 8),
                axis=-1,
            )
            decoded_keys = mx.sum(mx.stack([
                self.rerank_decoder_keys[table, key_indices[:, :, table]]
                for table in range(self.rerank_tables)
            ], axis=-2), axis=-2)
            decoded_queries = x[:, query_start:] @ self.rerank_decoder_query
            distance = -mx.einsum(
                "bqd,bkd->bqk", decoded_queries, decoded_keys
            )
        elif lookup:
            powers = mx.power(
                mx.array(2, dtype=mx.int32), mx.arange(8, dtype=mx.int32)
            )
            query_indices = mx.sum(
                ((query_code + 1.0) / 2.0).astype(mx.int32)
                * powers.reshape(1, 1, 1, 8),
                axis=-1,
            )
            key_indices = mx.sum(
                ((key_code + 1.0) / 2.0).astype(mx.int32)
                * powers.reshape(1, 1, 1, 8),
                axis=-1,
            )
            table_scores = []
            for table in range(self.rerank_tables):
                table_scores.append(
                    self.rerank_lookup[
                        table,
                        query_indices[:, :, None, table],
                        key_indices[:, None, :, table],
                    ]
                )
            distance = -mx.sum(mx.stack(table_scores, axis=-1), axis=-1)
        elif bilinear:
            query_flat = query_code.reshape(
                query_code.shape[0], query_code.shape[1], -1
            )
            if bit_weights is not None:
                query_flat = query_flat * bit_weights.reshape(
                    bit_weights.shape[0], bit_weights.shape[1], -1
                )
            key_flat = key_code.reshape(key_code.shape[0], key_code.shape[1], -1)
            bilinear_score = mx.einsum(
                "bqd,df,bkf->bqk",
                query_flat, self.rerank_bilinear, key_flat,
            )
            distance = -bilinear_score
        else:
            bit_distance = (
                1.0
                - query_code[:, :, None, :, :] * key_code[:, None, :, :, :]
            ) / 2.0
            if bit_weights is not None:
                distance = mx.sum(
                    bit_distance * bit_weights[:, :, None, :, :],
                    axis=(-1, -2),
                )
            else:
                distance = mx.sum(bit_distance, axis=(-1, -2))
        query_positions = mx.arange(query_start, x.shape[1]).reshape(-1, 1)
        key_positions = mx.arange(x.shape[1]).reshape(1, -1)
        if distance_bias:
            relative = mx.maximum(query_positions - key_positions, 1)
            buckets = mx.minimum(
                mx.floor(mx.log2(relative.astype(mx.float32))).astype(mx.int32),
                self.rerank_distance_bias.shape[0] - 1,
            )
            distance = distance - self.rerank_distance_bias[buckets][None, :, :]
        scores = -distance / temperature
        eligible = (key_positions < query_positions - window) & (
            key_positions >= sink_tokens
        )
        if candidate_mask is not None:
            eligible = eligible & candidate_mask.astype(mx.bool_)
        scores = mx.where(eligible, scores, mx.array(-1e9, scores.dtype))
        log_probability = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
        masked_teacher = mx.where(
            eligible, teacher_probability, mx.zeros_like(teacher_probability)
        )
        target = mx.power(
            mx.maximum(masked_teacher, mx.array(0.0, masked_teacher.dtype)),
            mass_gamma,
        )
        target = target / mx.maximum(
            mx.sum(target, axis=-1, keepdims=True),
            mx.array(1e-12, target.dtype),
        )
        cross_entropy = -mx.mean(mx.sum(target * log_probability, axis=-1))

        label_count = min(retrieval_topk, teacher_probability.shape[-1])
        teacher_order = mx.argsort(
            -mx.where(
                eligible, teacher_probability,
                mx.array(-1.0, teacher_probability.dtype),
            ),
            axis=-1,
        )
        positive_indices = teacher_order[..., :label_count]
        positive_mass = mx.take_along_axis(
            masked_teacher, positive_indices, axis=-1
        )
        positive_weights = mx.power(
            mx.maximum(positive_mass, mx.array(0.0, positive_mass.dtype)),
            mass_gamma,
        )
        positive_weights = positive_weights / mx.maximum(
            mx.sum(positive_weights, axis=-1, keepdims=True),
            mx.array(1e-12, positive_weights.dtype),
        )
        teacher_rank = mx.argsort(teacher_order, axis=-1)
        negative_mask = eligible & (teacher_rank >= label_count)
        negative_order_score = mx.where(
            negative_mask,
            -mx.stop_gradient(distance),
            mx.array(-1e9, distance.dtype),
        )
        negative_count = min(hard_negatives, teacher_probability.shape[-1])
        negative_indices = mx.argsort(-negative_order_score, axis=-1)[
            ..., :negative_count
        ]
        negative_valid = mx.take_along_axis(
            negative_mask, negative_indices, axis=-1
        )
        negative_weights = negative_valid.astype(distance.dtype)
        negative_weights = negative_weights / mx.maximum(
            mx.sum(negative_weights, axis=-1, keepdims=True),
            mx.array(1.0, negative_weights.dtype),
        )
        positive_distance = mx.take_along_axis(
            distance, positive_indices, axis=-1
        )
        negative_distance = mx.take_along_axis(
            distance, negative_indices, axis=-1
        )
        pairwise_terms = mx.logaddexp(
            mx.zeros_like(
                positive_distance[..., :, None] - negative_distance[..., None, :]
            ),
            pairwise_margin
            + positive_distance[..., :, None]
            - negative_distance[..., None, :],
        )
        pairwise_weights = (
            positive_weights[..., :, None] * negative_weights[..., None, :]
        )
        pairwise_loss = mx.sum(pairwise_terms * pairwise_weights) / mx.maximum(
            mx.sum(pairwise_weights), mx.array(1.0, pairwise_weights.dtype)
        )

        probabilities = [mx.sigmoid(query_logits), mx.sigmoid(key_logits)]
        balance = mx.mean(mx.stack([
            mx.mean(mx.square(mx.mean(probability, axis=(0, 1)) - 0.5))
            for probability in probabilities
        ]))
        confidence = mx.mean(mx.stack([
            mx.mean(probability * (1.0 - probability))
            for probability in probabilities
        ]))
        decorrelation_terms = []
        for logits in (query_logits, key_logits):
            rerank_dimensions = logits.shape[-2] * logits.shape[-1]
            values = mx.tanh(logits).reshape(-1, rerank_dimensions)
            values = values - mx.mean(values, axis=0, keepdims=True)
            covariance = (values.T @ values) / max(values.shape[0], 1)
            variance = mx.diag(covariance)
            covariance_scale = mx.sqrt(mx.maximum(
                variance[:, None] * variance[None, :],
                mx.array(1e-6, covariance.dtype),
            ))
            correlation = covariance / covariance_scale
            off_diagonal = correlation * (
                1.0 - mx.eye(correlation.shape[0], dtype=correlation.dtype)
            )
            dimensions = correlation.shape[0]
            decorrelation_terms.append(
                mx.sum(mx.square(off_diagonal))
                / max(dimensions * (dimensions - 1), 1)
            )
        decorrelation = mx.mean(mx.stack(decorrelation_terms))
        total = (
            cross_entropy
            + pairwise_weight * pairwise_loss
            + balance_weight * balance
            + decorrelation_weight * decorrelation
            + 0.01 * confidence
        )
        return total, (
            cross_entropy, pairwise_loss, balance, confidence, decorrelation
        )


class ProductQuantizedAttentionRouter(HierarchicalAttentionRouter):
    """Four-byte differentiable product-quantized attention reranker."""

    def __init__(self, width, tables, bits, rerank_bytes=4, centroids=256):
        super().__init__(width, tables, bits, rerank_bytes=rerank_bytes)
        subspace_width = self.rerank_query_projection.shape[-1] // rerank_bytes
        initial = mx.random.normal((rerank_bytes, centroids, subspace_width))
        self.pq_centroids = initial / mx.maximum(
            mx.sqrt(mx.sum(mx.square(initial), axis=-1, keepdims=True)),
            mx.array(1e-6, initial.dtype),
        )

    @staticmethod
    def _normalized_subspaces(logits):
        return logits / mx.maximum(
            mx.sqrt(mx.sum(mx.square(logits), axis=-1, keepdims=True)),
            mx.array(1e-6, logits.dtype),
        )

    def pq_rerank_loss(
        self, x, teacher_probability, query_start, window, sink_tokens,
        candidate_mask=None, retrieval_topk=32, hard_negatives=32,
        score_temperature=0.25, assignment_temperature=0.1,
        pairwise_weight=2.0, pairwise_margin=0.2,
        balance_weight=0.1, quantization_weight=0.1,
    ):
        query_logits, key_logits = self.rerank_logits(x)
        query = self._normalized_subspaces(query_logits[:, query_start:])
        key = self._normalized_subspaces(key_logits)
        centroids = self._normalized_subspaces(self.pq_centroids)
        assignment_scores = mx.einsum("bktd,tcd->bktc", key, centroids)
        soft_assignment = mx.softmax(
            assignment_scores / assignment_temperature, axis=-1
        )
        hard_index = mx.argmax(assignment_scores, axis=-1)
        hard_assignment = (
            hard_index[..., None]
            == mx.arange(centroids.shape[1]).reshape(1, 1, 1, -1)
        ).astype(soft_assignment.dtype)
        assignment = soft_assignment + mx.stop_gradient(
            hard_assignment - soft_assignment
        )
        quantized_key = mx.einsum(
            "bktc,tcd->bktd", assignment, centroids
        )
        scores = mx.einsum("bqtd,bktd->bqk", query, quantized_key)
        query_positions = mx.arange(query_start, x.shape[1]).reshape(-1, 1)
        key_positions = mx.arange(x.shape[1]).reshape(1, -1)
        eligible = (key_positions < query_positions - window) & (
            key_positions >= sink_tokens
        )
        if candidate_mask is not None:
            eligible = eligible & candidate_mask.astype(mx.bool_)
        masked_scores = mx.where(
            eligible, scores / score_temperature,
            mx.array(-1e9, scores.dtype),
        )
        log_probability = masked_scores - mx.logsumexp(
            masked_scores, axis=-1, keepdims=True
        )
        target = mx.where(
            eligible, teacher_probability, mx.zeros_like(teacher_probability)
        )
        target = target / mx.maximum(
            mx.sum(target, axis=-1, keepdims=True),
            mx.array(1e-12, target.dtype),
        )
        cross_entropy = -mx.mean(mx.sum(target * log_probability, axis=-1))

        label_count = min(retrieval_topk, teacher_probability.shape[-1])
        teacher_order = mx.argsort(
            -mx.where(
                eligible, teacher_probability,
                mx.array(-1.0, teacher_probability.dtype),
            ), axis=-1,
        )
        positive_indices = teacher_order[..., :label_count]
        positive_mass = mx.take_along_axis(target, positive_indices, axis=-1)
        positive_weights = positive_mass / mx.maximum(
            mx.sum(positive_mass, axis=-1, keepdims=True),
            mx.array(1e-12, positive_mass.dtype),
        )
        teacher_rank = mx.argsort(teacher_order, axis=-1)
        negative_mask = eligible & (teacher_rank >= label_count)
        negative_order = mx.where(
            negative_mask, mx.stop_gradient(scores),
            mx.array(-1e9, scores.dtype),
        )
        negative_count = min(hard_negatives, teacher_probability.shape[-1])
        negative_indices = mx.argsort(-negative_order, axis=-1)[
            ..., :negative_count
        ]
        negative_valid = mx.take_along_axis(
            negative_mask, negative_indices, axis=-1
        ).astype(scores.dtype)
        negative_weights = negative_valid / mx.maximum(
            mx.sum(negative_valid, axis=-1, keepdims=True),
            mx.array(1.0, scores.dtype),
        )
        positive_score = mx.take_along_axis(scores, positive_indices, axis=-1)
        negative_score = mx.take_along_axis(scores, negative_indices, axis=-1)
        pairwise_terms = mx.logaddexp(
            mx.zeros_like(
                positive_score[..., :, None] - negative_score[..., None, :]
            ),
            pairwise_margin - positive_score[..., :, None]
            + negative_score[..., None, :],
        )
        pairwise_weights = (
            positive_weights[..., :, None] * negative_weights[..., None, :]
        )
        pairwise = mx.sum(pairwise_terms * pairwise_weights) / mx.maximum(
            mx.sum(pairwise_weights), mx.array(1.0, pairwise_weights.dtype)
        )
        usage = mx.mean(soft_assignment, axis=(0, 1))
        balance = mx.mean(mx.square(usage - 1.0 / centroids.shape[1]))
        quantization = mx.mean(mx.square(key - quantized_key))
        total = (
            cross_entropy + pairwise_weight * pairwise
            + balance_weight * balance
            + quantization_weight * quantization
        )
        return total, (cross_entropy, pairwise, balance, quantization)

    def retention_scores(self, x):
        return x @ self.retention_projection

    def attention_retention_loss(
        self, x, teacher_probability, retrieval_topk=32,
        pairwise_weight=1.0, pairwise_margin=1.0,
        leaf_pairwise_weight=0.0, leaf_storage_capacity=0,
    ):
        salience = mx.sum(teacher_probability, axis=1)
        target = salience / mx.maximum(
            mx.sum(salience, axis=-1, keepdims=True),
            mx.array(1e-12, salience.dtype),
        )
        scores = self.retention_scores(x)
        log_probability = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
        cross_entropy = -mx.mean(mx.sum(target * log_probability, axis=-1))

        keep = min(retrieval_topk, x.shape[1])
        order = mx.argsort(-salience, axis=-1)
        positive_indices = order[..., :keep]
        rank = mx.argsort(order, axis=-1)
        negative_mask = rank >= keep
        negative_order = mx.argsort(
            -mx.where(
                negative_mask, mx.stop_gradient(scores),
                mx.array(-1e9, scores.dtype),
            ),
            axis=-1,
        )[..., :keep]
        positive_score = mx.take_along_axis(scores, positive_indices, axis=-1)
        negative_score = mx.take_along_axis(scores, negative_order, axis=-1)
        positive_mass = mx.take_along_axis(salience, positive_indices, axis=-1)
        positive_weight = positive_mass / mx.maximum(
            mx.sum(positive_mass, axis=-1, keepdims=True),
            mx.array(1e-12, positive_mass.dtype),
        )
        pairwise = mx.logaddexp(
            mx.zeros_like(positive_score[..., :, None] - negative_score[..., None, :]),
            pairwise_margin
            - positive_score[..., :, None]
            + negative_score[..., None, :],
        )
        pairwise_loss = mx.mean(mx.sum(
            pairwise * positive_weight[..., :, None], axis=-2
        ))
        leaf_pairwise_loss = mx.array(0.0, scores.dtype)
        if leaf_pairwise_weight != 0.0:
            if leaf_storage_capacity < 1:
                raise ValueError(
                    "leaf_storage_capacity must be positive for leaf retention"
                )
            _, key_logits = self.logits(x)
            key_bits = (key_logits >= 0.0).astype(mx.int32)
            powers = mx.power(
                mx.array(2, dtype=mx.int32), mx.arange(self.bits, dtype=mx.int32)
            )
            key_codes = mx.sum(key_bits * powers.reshape(1, 1, 1, -1), axis=-1)
            score_difference = scores[:, :, None] - scores[:, None, :]
            leaf_terms = []
            for table in range(self.tables):
                adjacent = (table + 1) % self.tables
                composite = (
                    key_codes[:, :, table] * (1 << self.bits)
                    + key_codes[:, :, adjacent]
                )
                same_leaf = composite[:, :, None] == composite[:, None, :]
                higher_salience = salience[:, None, :] > salience[:, :, None]
                within_leaf_rank = mx.sum(
                    (same_leaf & higher_salience).astype(mx.int32), axis=-1
                )
                should_retain = within_leaf_rank < leaf_storage_capacity
                boundary_pairs = (
                    same_leaf
                    & should_retain[:, :, None]
                    & (~should_retain[:, None, :])
                )
                weights = (
                    boundary_pairs.astype(scores.dtype)
                    * salience[:, :, None]
                )
                terms = mx.logaddexp(
                    mx.zeros_like(score_difference),
                    pairwise_margin - score_difference,
                )
                leaf_terms.append(
                    mx.sum(terms * weights)
                    / mx.maximum(mx.sum(weights), mx.array(1.0, weights.dtype))
                )
            leaf_pairwise_loss = mx.mean(mx.stack(leaf_terms))
        total = (
            cross_entropy + pairwise_weight * pairwise_loss
            + leaf_pairwise_weight * leaf_pairwise_loss
        )
        return total, (cross_entropy, pairwise_loss, leaf_pairwise_loss)

    def hierarchical_loss(
        self, x, teacher_probability, query_start, window, sink_tokens,
        mass_cover=0.95, mass_gamma=0.5, max_positives=56,
        positive_weight=1.0, hard_negative_weight=1.0,
        hard_negative_temperature=1.0, rerank_weight=1.0,
        rerank_temperature=1.0, balance_weight=10.0,
        decorrelation_weight=0.0, address_entropy_weight=0.0,
        leaf_overflow_weight=0.0, leaf_storage_capacity=0,
        candidate_set_weight=0.0, candidate_set_temperature=16.0,
        candidate_set_query_stride=1, secondary_probes=1,
        retrieval_topk=32,
        deployed_candidate_mask=None, exact_boundary_weight=0.0,
        exact_boundary_negative_weight=0.0,
        exact_boundary_query_stride=1,
        reranker="path-hamming",
    ):
        """Train the exact adjacent-table path used by hierarchical lookup.

        A target assigned to path t must match both primary table t and secondary
        table t+1. Positives cover a declared amount of distant teacher mass, while
        the negative term concentrates on low-mass keys closest to each current path.
        The reranking term uses minimum adjacent-pair Hamming distance, matching the
        path-aware selector without adding fingerprint bytes.
        """
        query_logits, key_logits = self.logits(x)
        query_logits = query_logits[:, query_start:]
        query_code = self.straight_through_sign(query_logits)
        key_code = self.straight_through_sign(key_logits)
        dot_by_table = mx.einsum("bqtd,bktd->bqkt", query_code, key_code)
        distance_by_table = (self.bits - dot_by_table) / 2.0
        secondary_distance = mx.concatenate(
            [distance_by_table[..., 1:], distance_by_table[..., :1]], axis=-1
        )
        path_distance = distance_by_table + secondary_distance
        # A positive margin means an exact 16-bit primary/secondary path match.
        path_match_logit = 2.0 * (0.5 - path_distance)

        query_positions = mx.arange(query_start, x.shape[1]).reshape(-1, 1)
        key_positions = mx.arange(x.shape[1]).reshape(1, -1)
        eligible = (key_positions < query_positions - window) & (
            key_positions >= sink_tokens
        )

        teacher_order = mx.argsort(-teacher_probability, axis=-1)
        sorted_mass = mx.take_along_axis(
            teacher_probability, teacher_order, axis=-1
        )
        cumulative_before = mx.cumsum(sorted_mass, axis=-1) - sorted_mass
        sorted_rank = mx.arange(teacher_probability.shape[-1]).reshape(1, 1, -1)
        selected_sorted = (
            (cumulative_before < mass_cover)
            & (sorted_rank < max_positives)
            & (sorted_mass > 0.0)
        )
        teacher_rank = mx.argsort(teacher_order, axis=-1)
        selected_positive = mx.take_along_axis(
            selected_sorted, teacher_rank, axis=-1
        ) & eligible
        table_index = mx.arange(self.tables).reshape(1, 1, 1, -1)
        assigned_positive = selected_positive[..., None] & (
            (teacher_rank[..., None] % self.tables) == table_index
        )

        powered_mass = mx.power(
            mx.maximum(teacher_probability, mx.array(0.0, teacher_probability.dtype)),
            mass_gamma,
        ) * selected_positive.astype(teacher_probability.dtype)
        powered_mass = powered_mass / mx.maximum(
            mx.sum(powered_mass, axis=-1, keepdims=True),
            mx.array(1e-12, powered_mass.dtype),
        )
        positive_weights = powered_mass[..., None] * assigned_positive.astype(
            powered_mass.dtype
        )
        positive_loss = mx.sum(
            positive_weights
            * mx.logaddexp(mx.zeros_like(path_match_logit), -path_match_logit)
        ) / mx.maximum(
            mx.sum(positive_weights), mx.array(1.0, positive_weights.dtype)
        )

        negative_mask = eligible[..., None] & (~selected_positive[..., None])
        negative_logits = -mx.stop_gradient(path_distance) / hard_negative_temperature
        negative_logits = mx.where(
            negative_mask,
            negative_logits,
            mx.array(-1e9, negative_logits.dtype),
        )
        negative_weights = mx.softmax(negative_logits, axis=2) * negative_mask.astype(
            negative_logits.dtype
        )
        negative_weights = negative_weights / mx.maximum(
            mx.sum(negative_weights, axis=2, keepdims=True),
            mx.array(1e-12, negative_weights.dtype),
        )
        negative_weights = mx.stop_gradient(negative_weights)
        hard_negative_loss = mx.mean(mx.sum(
            negative_weights
            * mx.logaddexp(mx.zeros_like(path_match_logit), path_match_logit),
            axis=2,
        ))

        if reranker == "path-hamming":
            rerank_distance = mx.min(path_distance, axis=-1)
        elif reranker == "full-hamming":
            rerank_distance = mx.sum(distance_by_table, axis=-1)
        else:
            raise ValueError("reranker must be path-hamming or full-hamming")
        rerank_scores = -rerank_distance / rerank_temperature
        rerank_scores = mx.where(
            eligible, rerank_scores, mx.array(-1e9, rerank_scores.dtype)
        )
        rerank_log_probability = rerank_scores - mx.logsumexp(
            rerank_scores, axis=-1, keepdims=True
        )
        rerank_target = mx.power(
            mx.maximum(teacher_probability, mx.array(0.0, teacher_probability.dtype)),
            mass_gamma,
        )
        rerank_target = rerank_target / mx.maximum(
            mx.sum(rerank_target, axis=-1, keepdims=True),
            mx.array(1e-12, rerank_target.dtype),
        )
        rerank_cross_entropy = -mx.mean(mx.sum(
            rerank_target * rerank_log_probability, axis=-1
        ))

        probabilities = [mx.sigmoid(query_logits), mx.sigmoid(key_logits)]
        balance = mx.mean(mx.stack([
            mx.mean(mx.square(mx.mean(probability, axis=(0, 1)) - 0.5))
            for probability in probabilities
        ]))
        address_values = mx.arange(1 << self.bits, dtype=mx.int32)
        bit_offsets = mx.arange(self.bits, dtype=mx.int32)
        address_patterns = (
            mx.right_shift(address_values[:, None], bit_offsets[None, :])
            & mx.array(1, dtype=mx.int32)
        ).astype(query_logits.dtype)
        address_entropy_terms = []
        for probability in probabilities:
            safe_probability = mx.clip(probability, 1e-6, 1.0 - 1e-6)
            log_probability = mx.log(safe_probability)
            log_inverse = mx.log(1.0 - safe_probability)
            code_log_probability = mx.sum(
                log_probability[..., None, :] * address_patterns
                + log_inverse[..., None, :] * (1.0 - address_patterns),
                axis=-1,
            )
            mean_code_probability = mx.mean(
                mx.exp(code_log_probability), axis=(0, 1)
            )
            normalized_kl = mx.sum(
                mean_code_probability * mx.log(mx.maximum(
                    mean_code_probability * float(1 << self.bits),
                    mx.array(1e-12, mean_code_probability.dtype),
                )),
                axis=-1,
            ) / math.log(float(1 << self.bits))
            address_entropy_terms.append(mx.mean(normalized_kl))
        address_entropy = mx.mean(mx.stack(address_entropy_terms))
        leaf_overflow = mx.array(0.0, query_logits.dtype)
        if leaf_overflow_weight != 0.0:
            if leaf_storage_capacity < 1:
                raise ValueError(
                    "leaf_storage_capacity must be positive when leaf overflow "
                    "regularization is enabled"
                )
            # Match the deployed primary+adjacent-secondary leaf exactly in the
            # forward pass, while differentiating through the probability that
            # two keys share every bit.  Unlike global entropy, this penalizes
            # only keys that currently collide in an actually over-capacity leaf.
            key_probability = probabilities[1]
            hard_key_bits = ((key_code + 1.0) / 2.0)
            overflow_terms = []

            def exact_table_match(probability, hard_bits):
                soft_bit_match = (
                    probability[:, :, None, :] * probability[:, None, :, :]
                    + (1.0 - probability[:, :, None, :])
                    * (1.0 - probability[:, None, :, :])
                )
                hard_bit_match = (
                    hard_bits[:, :, None, :] == hard_bits[:, None, :, :]
                ).astype(soft_bit_match.dtype)
                bit_match = soft_bit_match + mx.stop_gradient(
                    hard_bit_match - soft_bit_match
                )
                return mx.prod(bit_match, axis=-1)

            for table in range(self.tables):
                adjacent = (table + 1) % self.tables
                exact_leaf_match = exact_table_match(
                    key_probability[:, :, table], hard_key_bits[:, :, table]
                ) * exact_table_match(
                    key_probability[:, :, adjacent], hard_key_bits[:, :, adjacent]
                )
                leaf_load = mx.sum(exact_leaf_match, axis=-1)
                overflow = mx.maximum(
                    leaf_load - float(leaf_storage_capacity),
                    mx.array(0.0, leaf_load.dtype),
                ) / float(leaf_storage_capacity)
                overloaded = mx.stop_gradient((overflow > 0.0).astype(overflow.dtype))
                overflow_terms.append(
                    mx.sum(mx.square(overflow) * overloaded)
                    / mx.maximum(
                        mx.sum(overloaded), mx.array(1.0, overflow.dtype)
                    )
                )
            leaf_overflow = mx.mean(mx.stack(overflow_terms))
        candidate_set_mass = mx.array(0.0, query_logits.dtype)
        if candidate_set_weight != 0.0:
            if leaf_storage_capacity < 1:
                raise ValueError(
                    "leaf_storage_capacity must be positive when candidate-set "
                    "training is enabled"
                )
            if secondary_probes < 1 or secondary_probes > self.bits + 1:
                raise ValueError("secondary_probes must be in [1, bits + 1]")
            if candidate_set_temperature <= 0.0:
                raise ValueError("candidate_set_temperature must be positive")
            if candidate_set_query_stride < 1:
                raise ValueError("candidate_set_query_stride must be positive")

            # Approximate the deployed bounded shortlist rather than optimizing
            # address agreement in isolation.  Each primary+secondary path uses
            # the same exact hard bits as lookup in the forward pass.  Gradients
            # flow through bit-agreement probabilities.  A posting in a leaf of
            # load L receives the reservoir survival estimate min(1, C/L), so
            # useful mass in a hot leaf is worth less than useful mass in a leaf
            # that can actually retain it.
            key_probability = probabilities[1]
            hard_key_bits = (key_code + 1.0) / 2.0
            query_probability = probabilities[0][
                :, ::candidate_set_query_stride
            ]
            hard_query_bits = ((query_code + 1.0) / 2.0)[
                :, ::candidate_set_query_stride
            ]
            candidate_teacher = teacher_probability[
                :, ::candidate_set_query_stride
            ]
            candidate_query_logits = query_logits[
                :, ::candidate_set_query_stride
            ]

            def pair_distance(left_probability, left_bits, right_probability,
                              right_bits):
                soft_bit_match = (
                    left_probability[:, :, None, :]
                    * right_probability[:, None, :, :]
                    + (1.0 - left_probability[:, :, None, :])
                    * (1.0 - right_probability[:, None, :, :])
                )
                soft_distance = mx.sum(1.0 - soft_bit_match, axis=-1)
                hard_distance = mx.sum(
                    (left_bits[:, :, None, :] != right_bits[:, None, :, :])
                    .astype(soft_distance.dtype),
                    axis=-1,
                )
                return soft_distance + mx.stop_gradient(
                    hard_distance - soft_distance
                )

            retained_probability = None
            for table in range(self.tables):
                adjacent = (table + 1) % self.tables
                primary_distance = pair_distance(
                    query_probability[:, :, table],
                    hard_query_bits[:, :, table],
                    key_probability[:, :, table],
                    hard_key_bits[:, :, table],
                )
                uncertain = mx.argsort(
                    mx.abs(candidate_query_logits[:, :, adjacent]), axis=-1
                )[..., :max(secondary_probes - 1, 0)]
                secondary_distances = [pair_distance(
                    query_probability[:, :, adjacent],
                    hard_query_bits[:, :, adjacent],
                    key_probability[:, :, adjacent],
                    hard_key_bits[:, :, adjacent],
                )]
                bit_indices = mx.arange(self.bits).reshape(1, 1, -1)
                for probe in range(secondary_probes - 1):
                    flip = bit_indices == uncertain[..., probe, None]
                    flipped_probability = mx.where(
                        flip, 1.0 - query_probability[:, :, adjacent],
                        query_probability[:, :, adjacent],
                    )
                    flipped_bits = mx.where(
                        flip, 1.0 - hard_query_bits[:, :, adjacent],
                        hard_query_bits[:, :, adjacent],
                    )
                    secondary_distances.append(pair_distance(
                        flipped_probability, flipped_bits,
                        key_probability[:, :, adjacent],
                        hard_key_bits[:, :, adjacent],
                    ))
                secondary_distance = mx.min(
                    mx.stack(secondary_distances, axis=-1), axis=-1
                )

                # The load is exact in the forward pass and differentiable with
                # respect to the same leaf bits.  Compute it one table at a time
                # to preserve the existing bounded-memory behavior.
                def exact_key_match(probability, hard_bits):
                    soft_bit_match = (
                        probability[:, :, None, :]
                        * probability[:, None, :, :]
                        + (1.0 - probability[:, :, None, :])
                        * (1.0 - probability[:, None, :, :])
                    )
                    hard_bit_match = (
                        hard_bits[:, :, None, :]
                        == hard_bits[:, None, :, :]
                    ).astype(soft_bit_match.dtype)
                    bit_match = soft_bit_match + mx.stop_gradient(
                        hard_bit_match - soft_bit_match
                    )
                    return mx.prod(bit_match, axis=-1)

                same_leaf = exact_key_match(
                    key_probability[:, :, table],
                    hard_key_bits[:, :, table],
                ) * exact_key_match(
                    key_probability[:, :, adjacent],
                    hard_key_bits[:, :, adjacent],
                )
                leaf_load = mx.sum(same_leaf, axis=-1)
                survival = mx.minimum(
                    mx.ones_like(leaf_load),
                    float(leaf_storage_capacity) / mx.maximum(
                        leaf_load, mx.array(1.0, leaf_load.dtype)
                    ),
                )
                path_probability = mx.sigmoid(
                    candidate_set_temperature
                    * (0.5 - primary_distance - secondary_distance)
                ) * survival[:, None, :]
                path_probability = mx.clip(path_probability, 0.0, 1.0)
                retained_probability = (
                    path_probability
                    if retained_probability is None else
                    retained_probability + path_probability
                    - retained_probability * path_probability
                )
            candidate_positions = (
                query_start
                + mx.arange(retained_probability.shape[1])
                * candidate_set_query_stride
            ).reshape(-1, 1)
            candidate_keys = mx.arange(x.shape[1]).reshape(1, -1)
            candidate_eligible = (
                (candidate_keys < candidate_positions - window)
                & (candidate_keys >= sink_tokens)
            )
            candidate_order = mx.argsort(
                -mx.where(
                    candidate_eligible,
                    candidate_teacher,
                    mx.array(-1.0, candidate_teacher.dtype),
                ),
                axis=-1,
            )
            candidate_rank = mx.argsort(candidate_order, axis=-1)
            target_mask = candidate_eligible & (
                candidate_rank < min(retrieval_topk, x.shape[1])
            )
            target_mass = mx.where(
                target_mask, candidate_teacher,
                mx.zeros_like(candidate_teacher),
            )
            normalized_target = target_mass / mx.maximum(
                mx.sum(target_mass, axis=-1, keepdims=True),
                mx.array(1e-12, target_mass.dtype),
            )
            captured_mass = mx.sum(
                normalized_target * retained_probability, axis=-1
            )
            candidate_set_mass = -mx.mean(mx.log(mx.maximum(
                captured_mass, mx.array(1e-8, captured_mass.dtype)
            )))
        exact_boundary_positive = mx.array(0.0, query_logits.dtype)
        exact_boundary_negative = mx.array(0.0, query_logits.dtype)
        if exact_boundary_weight != 0.0 or exact_boundary_negative_weight != 0.0:
            if deployed_candidate_mask is None:
                raise ValueError(
                    "exact boundary training requires a deployed candidate mask"
                )
            if exact_boundary_query_stride < 1:
                raise ValueError("exact_boundary_query_stride must be positive")
            boundary_slice = slice(None, None, exact_boundary_query_stride)
            boundary_path_distance = path_distance[:, boundary_slice]
            boundary_path_match_logit = path_match_logit[:, boundary_slice]
            boundary_eligible = eligible[boundary_slice]
            boundary_teacher = teacher_probability[:, boundary_slice]
            boundary_teacher_rank = teacher_rank[:, boundary_slice]
            deployed = (
                deployed_candidate_mask[:, boundary_slice].astype(mx.bool_)
                & boundary_eligible
            )
            boundary_topk = boundary_teacher_rank < min(
                retrieval_topk, teacher_probability.shape[-1]
            )
            target_topk = boundary_topk & boundary_eligible
            missing = target_topk & (~deployed)
            missing_mass = mx.where(
                missing, boundary_teacher,
                mx.zeros_like(boundary_teacher),
            )
            missing_weights = missing_mass / mx.maximum(
                mx.sum(missing_mass, axis=-1, keepdims=True),
                mx.array(1e-12, missing_mass.dtype),
            )
            # Assign each exact deployed miss to its currently nearest complete
            # primary+secondary path.  The straight-through path logits then
            # pull the missed key across the actual hard membership boundary.
            best_positive_table = mx.argmin(
                mx.stop_gradient(boundary_path_distance), axis=-1
            )
            positive_table_mask = (
                best_positive_table[..., None] == table_index
            ).astype(boundary_path_match_logit.dtype)
            positive_terms = mx.logaddexp(
                mx.zeros_like(boundary_path_match_logit),
                -boundary_path_match_logit,
            )
            positive_boundary_weights = (
                missing_weights[..., None] * positive_table_mask
            )
            exact_boundary_positive = mx.sum(
                positive_terms * positive_boundary_weights
            ) / mx.maximum(
                mx.sum(positive_boundary_weights),
                mx.array(1.0, positive_boundary_weights.dtype),
            )

            # Push only actual false-positive occupants away, and only on paths
            # they match exactly in the hard forward pass.  This targets the
            # finite-capacity boundary without penalizing unrelated history.
            false_retained = deployed & (~target_topk)
            exact_path = mx.stop_gradient(
                (boundary_path_distance < 0.5).astype(
                    boundary_path_match_logit.dtype
                )
            )
            negative_boundary_weights = (
                false_retained[..., None].astype(
                    boundary_path_match_logit.dtype
                )
                * exact_path
            )
            negative_boundary_weights = negative_boundary_weights / mx.maximum(
                mx.sum(negative_boundary_weights, axis=2, keepdims=True),
                mx.array(1.0, negative_boundary_weights.dtype),
            )
            negative_terms = mx.logaddexp(
                mx.zeros_like(boundary_path_match_logit),
                boundary_path_match_logit,
            )
            exact_boundary_negative = mx.mean(mx.sum(
                negative_terms * negative_boundary_weights, axis=2
            ))
        confidence = mx.mean(mx.stack([
            mx.mean(probability * (1.0 - probability))
            for probability in probabilities
        ]))
        decorrelation_terms = []
        for logits in (query_logits, key_logits):
            values = mx.tanh(logits).reshape(-1, self.tables * self.bits)
            values = values - mx.mean(values, axis=0, keepdims=True)
            covariance = (values.T @ values) / max(values.shape[0], 1)
            variance = mx.diag(covariance)
            covariance_scale = mx.sqrt(mx.maximum(
                variance[:, None] * variance[None, :],
                mx.array(1e-6, covariance.dtype),
            ))
            correlation = covariance / covariance_scale
            off_diagonal = correlation * (
                1.0 - mx.eye(correlation.shape[0], dtype=correlation.dtype)
            )
            dimensions = correlation.shape[0]
            decorrelation_terms.append(
                mx.sum(mx.square(off_diagonal))
                / max(dimensions * (dimensions - 1), 1)
            )
        decorrelation = mx.mean(mx.stack(decorrelation_terms))
        total = (
            positive_weight * positive_loss
            + hard_negative_weight * hard_negative_loss
            + rerank_weight * rerank_cross_entropy
            + balance_weight * balance
            + decorrelation_weight * decorrelation
            + address_entropy_weight * address_entropy
            + leaf_overflow_weight * leaf_overflow
            + candidate_set_weight * candidate_set_mass
            + exact_boundary_weight * exact_boundary_positive
            + exact_boundary_negative_weight * exact_boundary_negative
            + 0.01 * confidence
        )
        selected_count = mx.mean(mx.sum(
            selected_positive.astype(mx.float32), axis=-1
        ))
        selected_mass = mx.mean(mx.sum(
            teacher_probability * selected_positive.astype(teacher_probability.dtype),
            axis=-1,
        ))
        return total, (
            positive_loss,
            hard_negative_loss,
            rerank_cross_entropy,
            balance,
            confidence,
            selected_count,
            selected_mass,
            decorrelation,
            address_entropy,
            leaf_overflow,
            candidate_set_mass,
            exact_boundary_positive,
            exact_boundary_negative,
        )


# The PQ subclass is declared before these pre-existing hierarchy methods so it
# can reuse them. Keep the non-PQ router API intact for existing checkpoints.
HierarchicalAttentionRouter.retention_scores = (
    ProductQuantizedAttentionRouter.retention_scores
)
HierarchicalAttentionRouter.attention_retention_loss = (
    ProductQuantizedAttentionRouter.attention_retention_loss
)
HierarchicalAttentionRouter.hierarchical_loss = (
    ProductQuantizedAttentionRouter.hierarchical_loss
)


def parse_paths(spec, defaults):
    values = [value.strip() for value in spec.split(",") if value.strip()] if spec else defaults
    paths = [pathlib.Path(value) for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing corpus files: {', '.join(missing)}")
    return paths


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_segments(tokenizer, paths, seq_len, stride, limit):
    segments = []
    for path in paths:
        token_ids = tokenizer.encode(path.read_text())
        for start in range(0, max(0, len(token_ids) - seq_len + 1), stride):
            segments.append(mx.array([token_ids[start:start + seq_len]], dtype=mx.int32))
            if len(segments) == limit:
                return segments
    if not segments:
        raise ValueError(f"corpus contains fewer than {seq_len} tokens")
    return segments


def donor_example(
    model, tokens, layer_index, window, sink_tokens,
    teacher_target="contribution",
):
    body = language_body(model)
    h = body.embed_tokens(tokens)
    attention_mask = create_attention_mask(h)
    state_mask = create_ssm_mask(h)
    for layer in body.layers[:layer_index]:
        layer_mask = attention_mask if hasattr(layer, "self_attn") else state_mask
        h = layer(h, layer_mask)
    layer = body.layers[layer_index]
    if getattr(layer, "is_linear", False) or not hasattr(layer, "self_attn"):
        raise ValueError(
            f"layer {layer_index} is not a full-attention layer; choose one with self_attn"
        )
    layer_norm = getattr(layer, "input_layernorm", None)
    if layer_norm is None:
        layer_norm = getattr(layer, "operator_norm", None)
    if layer_norm is None:
        raise ValueError(f"cannot find the input normalization for layer {layer_index}")
    x = layer_norm(h)
    attention = layer.self_attn
    batch, length, _ = x.shape
    heads = getattr(attention, "n_heads", getattr(attention, "num_attention_heads", None))
    kv_heads = getattr(
        attention, "n_kv_heads", getattr(attention, "num_key_value_heads", None)
    )
    head_dim = getattr(attention, "head_dim", x.shape[-1] // heads)
    query_projection = attention.q_proj(x)
    if query_projection.shape[-1] == heads * head_dim * 2:
        queries, _ = mx.split(
            query_projection.reshape(batch, length, heads, -1), 2, axis=-1
        )
    else:
        queries = query_projection.reshape(batch, length, heads, head_dim)
    keys = attention.k_proj(x).reshape(batch, length, kv_heads, head_dim)
    values = attention.v_proj(x).reshape(batch, length, kv_heads, head_dim)
    query_norm = getattr(attention, "q_norm", None)
    if query_norm is None:
        query_norm = getattr(attention, "q_layernorm", None)
    key_norm = getattr(attention, "k_norm", None)
    if key_norm is None:
        key_norm = getattr(attention, "k_layernorm", None)
    if query_norm is not None:
        queries = query_norm(queries)
    if key_norm is not None:
        keys = key_norm(keys)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)
    queries = attention.rope(queries)
    keys = attention.rope(keys)
    if heads != kv_heads:
        keys = mx.repeat(keys, heads // kv_heads, axis=1)
        values = mx.repeat(values, heads // kv_heads, axis=1)
    query_start = window + sink_tokens + 1
    queries = queries[:, :, query_start:]
    scores = mx.einsum("bhqd,bhkd->bhqk", queries, keys) * attention.scale
    query_positions = mx.arange(query_start, length).reshape(-1, 1)
    key_positions = mx.arange(length).reshape(1, -1)
    eligible = (key_positions < query_positions - window) & (key_positions >= sink_tokens)
    scores = mx.where(eligible, scores, mx.array(-1e9, scores.dtype))
    attention_probability = mx.softmax(scores.astype(mx.float32), axis=-1)
    if teacher_target == "attention":
        teacher_probability = mx.mean(attention_probability, axis=1)
    elif teacher_target == "contribution":
        value_norm = mx.sqrt(mx.sum(mx.square(values.astype(mx.float32)), axis=-1))
        contribution = attention_probability * value_norm[:, :, None, :]
        teacher_probability = mx.mean(contribution, axis=1)
    else:
        raise ValueError("teacher_target must be attention or contribution")
    teacher_probability = teacher_probability / mx.maximum(
        mx.sum(teacher_probability, axis=-1, keepdims=True),
        mx.array(1e-12, teacher_probability.dtype),
    )
    x = mx.stop_gradient(x)
    teacher_probability = mx.stop_gradient(teacher_probability)
    mx.eval(x, teacher_probability)
    return x, teacher_probability, query_start


def _mean_pairwise_correlation(values):
    values = np.asarray(values, dtype=np.float64)
    variable = values[:, np.std(values, axis=0) > 0.0]
    if variable.shape[1] < 2:
        return None
    correlation = np.corrcoef(variable, rowvar=False)
    upper = correlation[np.triu_indices(correlation.shape[0], 1)]
    finite = upper[np.isfinite(upper)]
    return float(np.mean(finite)) if finite.size else None


def _occupancy_metrics(code_batches, tables, bits):
    bucket_count = 1 << bits
    histogram = np.zeros(1, dtype=np.int64)
    per_table = [[] for _ in range(tables)]
    for codes in code_batches:
        for batch in range(codes.shape[0]):
            for table in range(tables):
                counts = np.bincount(
                    codes[batch, :, table], minlength=bucket_count
                )
                load_histogram = np.bincount(counts)
                if load_histogram.size > histogram.size:
                    histogram = np.pad(
                        histogram, (0, load_histogram.size - histogram.size)
                    )
                histogram[: load_histogram.size] += load_histogram
                occupied = counts[counts > 0]
                probabilities = occupied / max(int(counts.sum()), 1)
                entropy = -float(
                    np.sum(probabilities * np.log2(probabilities))
                )
                pair_denominator = int(counts.sum()) * (int(counts.sum()) - 1)
                pair_collision = (
                    float(np.sum(counts * (counts - 1)) / pair_denominator)
                    if pair_denominator
                    else 0.0
                )
                per_table[table].append({
                    "empty_bucket_fraction": float(np.mean(counts == 0)),
                    "maximum_load": int(occupied.max()) if occupied.size else 0,
                    "p95_occupied_load": float(np.percentile(occupied, 95))
                    if occupied.size else 0.0,
                    "p99_occupied_load": float(np.percentile(occupied, 99))
                    if occupied.size else 0.0,
                    "normalized_entropy": entropy / bits,
                    "pair_collision_probability": pair_collision,
                    "collision_inflation_vs_balanced": pair_collision * bucket_count,
                })

    def summarize(rows):
        return {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }

    by_table = [summarize(rows) for rows in per_table]
    all_rows = [row for rows in per_table for row in rows]
    return {
        "bucket_load_histogram": {
            str(load): int(count)
            for load, count in enumerate(histogram)
            if count
        },
        "mean": summarize(all_rows),
        "by_table": by_table,
    }


def _distance_metrics(distances, recall, retained_mass, candidates):
    ranges = [
        ("1-64", 1, 64),
        ("65-128", 65, 128),
        ("129-256", 129, 256),
        ("257-512", 257, 512),
        ("513+", 513, None),
    ]
    result = {}
    distances = np.asarray(distances)
    recall = np.asarray(recall, dtype=np.float64)
    retained_mass = np.asarray(retained_mass, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    for label, lower, upper in ranges:
        selected = distances >= lower
        if upper is not None:
            selected &= distances <= upper
        if not np.any(selected):
            continue
        result[label] = {
            "queries": int(np.sum(selected)),
            "teacher_top1_recall": float(np.mean(recall[selected])),
            "retained_teacher_mass": float(np.mean(retained_mass[selected])),
            "mean_unique_candidates": float(np.mean(candidates[selected])),
        }
    return result


def hard_metrics(router, examples, members, probes, window, sink_tokens,
                 member_policy="recent", history_fraction=0.5):
    retained_mass = []
    top_one = []
    candidates = []
    soft_retained_mass = []
    soft_top_one = []
    teacher_top_positions = []
    exact_agreement_rows = []
    probed_agreement_rows = []
    table_success_rows = []
    key_code_batches = []
    retrieval_distances = []
    for x, teacher_probability, query_start in examples:
        query_logits, key_logits = router.logits(x)
        query_code = mx.tanh(query_logits[:, query_start:])
        key_code = mx.tanh(key_logits)
        soft_scores = mx.max(
            mx.einsum("bqtd,bktd->bqkt", query_code, key_code), axis=-1
        )
        selected = select_indices_qk(
            x, x, router.query_projection, router.key_projection,
            tables=router.tables, bits=router.bits, members=members, probes=probes, block=False,
            min_distance=window, member_policy=member_policy,
            history_fraction=history_fraction,
        )
        query_codes = probe_codes(
            x, router.query_projection, router.tables, router.bits, probes
        )
        key_codes = hash_codes(
            x, router.key_projection, router.tables, router.bits
        )
        mx.eval(
            selected, teacher_probability, soft_scores, query_codes, key_codes
        )
        selected_np = np.array(selected)
        teacher_np = np.array(teacher_probability)
        soft_np = np.array(soft_scores)
        query_codes_np = np.array(query_codes)
        key_codes_np = np.array(key_codes)
        key_code_batches.append(key_codes_np)
        selected_by_table = selected_np.reshape(
            *selected_np.shape[:2], router.tables, probes, members
        )
        soft_budget = router.tables * members * probes
        for batch in range(selected_np.shape[0]):
            for offset, position in enumerate(range(query_start, selected_np.shape[1])):
                valid = np.unique(selected_np[batch, position])
                valid = valid[(valid >= sink_tokens) & (valid < position - window)]
                candidates.append(len(valid))
                teacher_top = int(np.argmax(teacher_np[batch, offset]))
                teacher_top_positions.append(teacher_top)
                retrieval_distances.append(position - teacher_top)
                target_codes = key_codes_np[batch, teacher_top]
                query_table_codes = query_codes_np[batch, position]
                exact_agreement_rows.append(
                    query_table_codes[:, 0] == target_codes
                )
                probed_agreement_rows.append(
                    np.any(query_table_codes == target_codes[:, None], axis=-1)
                )
                table_success_rows.append(
                    np.any(
                        selected_by_table[batch, position]
                        == teacher_top,
                        axis=(1, 2),
                    )
                )
                eligible_positions = np.arange(sink_tokens, position - window)
                count = min(soft_budget, len(eligible_positions))
                soft_order = np.argsort(soft_np[batch, offset, eligible_positions])[-count:]
                soft_selected = eligible_positions[soft_order]
                soft_retained_mass.append(float(teacher_np[batch, offset, soft_selected].sum()))
                soft_top_one.append(teacher_top in soft_selected)
                if len(valid):
                    retained_mass.append(float(teacher_np[batch, offset, valid].sum()))
                    top_one.append(teacher_top in valid)
                else:
                    retained_mass.append(0.0)
                    top_one.append(False)
    exact_agreement = np.asarray(exact_agreement_rows, dtype=bool)
    probed_agreement = np.asarray(probed_agreement_rows, dtype=bool)
    table_success = np.asarray(table_success_rows, dtype=bool)
    actual_recall = np.asarray(top_one, dtype=bool)
    table_success_rate = np.mean(table_success, axis=0)
    independent_recall = 1.0 - float(np.prod(1.0 - table_success_rate))
    agreement_any = np.any(probed_agreement, axis=1)
    selected_any = np.any(table_success, axis=1)
    attribution = {
        "no_probed_address_agreement": int(np.sum(~agreement_any)),
        "agreement_without_selection": int(
            np.sum(agreement_any & ~selected_any)
        ),
        "selected": int(np.sum(selected_any)),
    }
    total_queries = len(actual_recall)
    attribution["fractions"] = {
        key: value / total_queries
        for key, value in attribution.items()
        if key != "fractions"
    }
    candidate_array = np.asarray(candidates, dtype=np.float64)
    retained_array = np.asarray(retained_mass, dtype=np.float64)
    return {
        "retained_teacher_mass": float(np.mean(retained_mass)),
        "teacher_top1_recall": float(np.mean(top_one)),
        "unique_candidates": float(np.mean(candidates)),
        "soft_topk_retained_teacher_mass": float(np.mean(soft_retained_mass)),
        "soft_topk_teacher_top1_recall": float(np.mean(soft_top_one)),
        "teacher_top1_unique_positions": len(set(teacher_top_positions)),
        "teacher_top1_mode_fraction": float(
            np.max(np.bincount(teacher_top_positions)) / len(teacher_top_positions)
        ),
        "queries": len(retained_mass),
        "query_key_agreement": {
            "exact_any_table": float(np.mean(np.any(exact_agreement, axis=1))),
            "probed_any_table": float(np.mean(agreement_any)),
            "exact_by_table": np.mean(exact_agreement, axis=0).tolist(),
            "probed_by_table": np.mean(probed_agreement, axis=0).tolist(),
        },
        "bucket_occupancy": _occupancy_metrics(
            key_code_batches, router.tables, router.bits
        ),
        "table_retrieval": {
            "success_by_table": table_success_rate.tolist(),
            "mean_pairwise_success_correlation": _mean_pairwise_correlation(
                table_success
            ),
            "independence_predicted_top1_recall": independent_recall,
            "actual_top1_recall": float(np.mean(actual_recall)),
            "prediction_gap": float(np.mean(actual_recall)) - independent_recall,
        },
        "failure_attribution": attribution,
        "distance": _distance_metrics(
            retrieval_distances, top_one, retained_mass, candidates
        ),
        "candidate_distribution": {
            "mean": float(np.mean(candidate_array)),
            "p05": float(np.percentile(candidate_array, 5)),
            "p50": float(np.percentile(candidate_array, 50)),
            "p95": float(np.percentile(candidate_array, 95)),
        },
        "retained_mass_distribution": {
            "mean": float(np.mean(retained_array)),
            "p05": float(np.percentile(retained_array, 5)),
            "p50": float(np.percentile(retained_array, 50)),
            "p95": float(np.percentile(retained_array, 95)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--balance-weight", type=float, default=10.0)
    parser.add_argument("--decorrelation-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-weight", type=float, default=1.0)
    parser.add_argument("--retrieval-topk", type=int, default=32)
    parser.add_argument("--retrieval-positive-weight", type=float, default=10.0)
    parser.add_argument("--train-segments", type=int, default=8)
    parser.add_argument("--eval-segments", type=int, default=2)
    parser.add_argument("--train-files", default="")
    parser.add_argument("--eval-files", default="")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--output", default="runs/lfm2.5-router.safetensors")
    args = parser.parse_args()
    if args.seq_len <= args.window + 1:
        parser.error("--seq-len must exceed --window + 1")
    if args.sink_tokens < 0 or args.sink_tokens >= args.seq_len - args.window - 1:
        parser.error("--sink-tokens leaves no eligible distant keys")
    if args.bits < 1 or args.bits > 30:
        parser.error("--bits must be between 1 and 30")
    if args.probes < 1 or args.probes > args.bits + 1:
        parser.error("--probes must be between 1 and --bits + 1")
    if args.retrieval_topk < 1 or args.retrieval_topk > args.seq_len:
        parser.error("--retrieval-topk must be between 1 and --seq-len")
    if args.retrieval_positive_weight < 1.0:
        parser.error("--retrieval-positive-weight must be at least 1")
    mx.random.seed(args.seed)

    train_paths = parse_paths(args.train_files, DEFAULT_TRAIN_FILES)
    eval_paths = parse_paths(args.eval_files, DEFAULT_EVAL_FILES)
    donor, tokenizer, config = load(args.model, lazy=True, return_config=True)
    body = language_body(donor)
    if args.layer < 0 or args.layer >= len(body.layers):
        parser.error(f"--layer must be between 0 and {len(body.layers) - 1}")
    if not hasattr(body.layers[args.layer], "self_attn"):
        full_layers = [
            index for index, layer in enumerate(body.layers)
            if hasattr(layer, "self_attn")
        ]
        parser.error(
            f"--layer {args.layer} is not full attention; full-attention layers: {full_layers}"
        )
    train_tokens = token_segments(
        tokenizer, train_paths, args.seq_len, args.stride, args.train_segments
    )
    eval_tokens = token_segments(
        tokenizer, eval_paths, args.seq_len, args.stride, args.eval_segments
    )
    train_examples = [
        donor_example(donor, tokens, args.layer, args.window, args.sink_tokens)
        for tokens in train_tokens
    ]
    eval_examples = [
        donor_example(donor, tokens, args.layer, args.window, args.sink_tokens)
        for tokens in eval_tokens
    ]
    width = train_examples[0][0].shape[-1]
    del donor
    mx.clear_cache()

    router = DonorHashRouter(width, args.tables, args.bits)
    if args.init_checkpoint:
        checkpoint = pathlib.Path(args.init_checkpoint)
        if not checkpoint.is_file():
            parser.error(f"--init-checkpoint does not exist: {checkpoint}")
        router.load_weights(str(checkpoint))
        mx.eval(router.parameters())
    before = hard_metrics(
        router, eval_examples, args.members, args.probes, args.window, args.sink_tokens
    )
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(model, x, teacher, query_start):
        return model.loss(
            x, teacher, query_start, args.window, args.sink_tokens,
            args.alignment_weight, args.balance_weight,
            args.decorrelation_weight,
            args.retrieval_weight,
            args.retrieval_topk,
            args.retrieval_positive_weight,
        )

    loss_and_grad = nn.value_and_grad(router, loss_fn)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, teacher, query_start = train_examples[(step - 1) % len(train_examples)]
        (loss, parts), gradients = loss_and_grad(router, x, teacher, query_start)
        optimizer.update(router, gradients)
        mx.eval(router.parameters(), optimizer.state, loss, parts)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            (
                cross_entropy,
                retrieval_bce,
                alignment,
                balance,
                confidence,
                decorrelation,
            ) = map(float, parts)
            print(json.dumps({
                "step": step,
                "loss": round(float(loss), 5),
                "cross_entropy": round(cross_entropy, 5),
                "retrieval_bce": round(retrieval_bce, 5),
                "alignment": round(alignment, 5),
                "balance": round(balance, 5),
                "confidence": round(confidence, 5),
                "decorrelation": round(decorrelation, 5),
                "steps_per_second": round(step / (time.perf_counter() - started), 3),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
            }), flush=True)

    after = hard_metrics(
        router, eval_examples, args.members, args.probes, args.window, args.sink_tokens
    )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    router.save_weights(str(output))
    donor_config = config.get("text_config", config)
    metadata = vars(args) | {
        "teacher_target": "normalized_mean_attention_probability_times_value_l2_norm",
        "donor_config": {
            key: donor_config.get(key) for key in [
                "model_type", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
            ]
        },
        "train_files_resolved": [str(path) for path in train_paths],
        "eval_files_resolved": [str(path) for path in eval_paths],
        "train_corpus": [
            {"path": str(path), "sha256": sha256(path)} for path in train_paths
        ],
        "eval_corpus": [
            {"path": str(path), "sha256": sha256(path)} for path in eval_paths
        ],
        "before": before,
        "after": after,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"before": before, "after": after, "checkpoint": str(output)}), flush=True)


if __name__ == "__main__":
    main()
