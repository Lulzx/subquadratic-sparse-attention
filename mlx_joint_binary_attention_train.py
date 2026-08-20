"""Jointly train a 32-bit key code and RoPE-aware multi-head attention decoder."""

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

from mlx_attention_key_decoder_train import apply_rope
from mlx_donor_router import HierarchicalAttentionRouter, language_body
from mlx_hierarchical_router_train import corpus_segments, parse_corpora
from mlx_lfm_hierarchical_eval import (
    binary_fingerprint_bytes,
    capture_layer,
    categorical_address_codes,
    causal_hierarchical_candidates,
    primary_conditioned_categorical_address_codes,
    router_codes,
)


class ThresholdedHierarchicalAttentionRouter(HierarchicalAttentionRouter):
    """Hierarchical sign codes with learned per-table decision thresholds."""

    def __init__(self, width, tables, bits, rerank_bytes=8):
        super().__init__(width, tables, bits, rerank_bytes=rerank_bytes)
        self.address_query_bias = mx.zeros((tables, bits))
        self.address_key_bias = mx.zeros((tables, bits))

    def logits(self, x):
        query, key = super().logits(x)
        return (
            query + self.address_query_bias,
            key + self.address_key_bias,
        )


class CategoricalHierarchicalAddressRouter(nn.Module):
    """Direct 256-way byte assignments for every hierarchical address table."""

    def __init__(
        self, query_projection, key_projection, tables=8, bits=8,
        temperature=1.0,
    ):
        super().__init__()
        if bits != 8:
            raise ValueError("categorical byte addresses require eight bits")
        byte_bits = np.unpackbits(
            np.arange(256, dtype=np.uint8)[:, None], axis=-1,
            bitorder="little",
        ).astype(np.float32) * 2.0 - 1.0
        pattern = mx.array(byte_bits.T)
        width = query_projection.shape[0]
        query_tables = query_projection.reshape(width, tables, bits)
        key_tables = key_projection.reshape(width, tables, bits)
        self.address_query_assignment_weight = mx.einsum(
            "dtb,bc->dtc", query_tables, pattern
        )
        self.address_key_assignment_weight = mx.einsum(
            "dtb,bc->dtc", key_tables, pattern
        )
        self.address_query_assignment_bias = mx.zeros((tables, 256))
        self.address_key_assignment_bias = mx.zeros((tables, 256))
        self.tables = tables
        self.categories = 256
        self.temperature = temperature

    def logits(self, x):
        return (
            mx.einsum(
                "bnd,dtc->bntc", x,
                self.address_query_assignment_weight,
            ) + self.address_query_assignment_bias,
            mx.einsum(
                "bnd,dtc->bntc", x,
                self.address_key_assignment_weight,
            ) + self.address_key_assignment_bias,
        )

    def assignments(self, logits):
        probability = mx.softmax(logits / self.temperature, axis=-1)
        index = mx.argmax(logits, axis=-1)
        hard = (
            index[..., None]
            == mx.arange(self.categories, dtype=index.dtype)
        ).astype(probability.dtype)
        return probability + mx.stop_gradient(hard - probability), probability

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
        retrieval_topk=32, deployed_candidate_mask=None,
        exact_boundary_weight=0.0, exact_boundary_negative_weight=0.0,
        exact_boundary_query_stride=1, reranker="full-hamming",
    ):
        del (
            candidate_set_temperature, candidate_set_query_stride,
            secondary_probes, reranker,
        )
        if candidate_set_weight != 0.0:
            raise ValueError(
                "categorical address training does not implement the rejected "
                "expected-survival surrogate"
            )
        query_logits, key_logits = self.logits(x)
        query_logits = query_logits[:, query_start:]
        query_code, query_probability = self.assignments(query_logits)
        key_code, key_probability = self.assignments(key_logits)
        table_match = mx.einsum(
            "bqtc,bktc->bqkt", query_code, key_code
        )
        secondary_match = mx.concatenate(
            [table_match[..., 1:], table_match[..., :1]], axis=-1
        )
        path_distance = 2.0 - table_match - secondary_match
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
            mx.maximum(
                teacher_probability,
                mx.array(0.0, teacher_probability.dtype),
            ), mass_gamma,
        ) * selected_positive.astype(teacher_probability.dtype)
        powered_mass = powered_mass / mx.maximum(
            mx.sum(powered_mass, axis=-1, keepdims=True),
            mx.array(1e-12, powered_mass.dtype),
        )
        positive_weights = powered_mass[..., None] * assigned_positive.astype(
            powered_mass.dtype
        )
        positive_loss = mx.sum(
            positive_weights * mx.logaddexp(
                mx.zeros_like(path_match_logit), -path_match_logit
            )
        ) / mx.maximum(
            mx.sum(positive_weights), mx.array(1.0, positive_weights.dtype)
        )

        negative_mask = eligible[..., None] & (~selected_positive[..., None])
        negative_logits = (
            -mx.stop_gradient(path_distance) / hard_negative_temperature
        )
        negative_logits = mx.where(
            negative_mask, negative_logits,
            mx.array(-1e9, negative_logits.dtype),
        )
        negative_weights = mx.softmax(negative_logits, axis=2) * (
            negative_mask.astype(negative_logits.dtype)
        )
        negative_weights = mx.stop_gradient(
            negative_weights / mx.maximum(
                mx.sum(negative_weights, axis=2, keepdims=True),
                mx.array(1e-12, negative_weights.dtype),
            )
        )
        hard_negative_loss = mx.mean(mx.sum(
            negative_weights * mx.logaddexp(
                mx.zeros_like(path_match_logit), path_match_logit
            ), axis=2,
        ))

        rerank_distance = mx.sum(1.0 - table_match, axis=-1)
        rerank_scores = mx.where(
            eligible, -rerank_distance / rerank_temperature,
            mx.array(-1e9, rerank_distance.dtype),
        )
        rerank_log_probability = rerank_scores - mx.logsumexp(
            rerank_scores, axis=-1, keepdims=True
        )
        rerank_target = mx.power(
            mx.maximum(
                teacher_probability,
                mx.array(0.0, teacher_probability.dtype),
            ), mass_gamma,
        )
        rerank_target = rerank_target / mx.maximum(
            mx.sum(rerank_target, axis=-1, keepdims=True),
            mx.array(1e-12, rerank_target.dtype),
        )
        rerank_cross_entropy = -mx.mean(mx.sum(
            rerank_target * rerank_log_probability, axis=-1
        ))

        usage = [
            mx.mean(query_probability, axis=(0, 1)),
            mx.mean(key_probability, axis=(0, 1)),
        ]
        balance = self.categories * mx.mean(mx.stack([
            mx.mean(mx.square(value - 1.0 / self.categories))
            for value in usage
        ]))
        address_entropy = mx.mean(mx.stack([
            mx.mean(mx.sum(
                value * mx.log(mx.maximum(
                    value * self.categories,
                    mx.array(1e-12, value.dtype),
                )), axis=-1
            )) / math.log(self.categories)
            for value in usage
        ]))
        confidence = mx.mean(mx.stack([
            mx.mean(1.0 - mx.max(query_probability, axis=-1)),
            mx.mean(1.0 - mx.max(key_probability, axis=-1)),
        ]))
        decorrelation = mx.array(0.0, query_logits.dtype)

        leaf_overflow = mx.array(0.0, query_logits.dtype)
        if leaf_overflow_weight != 0.0:
            if leaf_storage_capacity < 1:
                raise ValueError("leaf_storage_capacity must be positive")
            overflow_terms = []
            for table in range(self.tables):
                adjacent = (table + 1) % self.tables
                same_primary = mx.einsum(
                    "bkc,bjc->bkj",
                    key_code[:, :, table], key_code[:, :, table],
                )
                same_secondary = mx.einsum(
                    "bkc,bjc->bkj",
                    key_code[:, :, adjacent], key_code[:, :, adjacent],
                )
                same_leaf = same_primary * same_secondary
                leaf_load = mx.sum(same_leaf, axis=-1)
                overflow = mx.maximum(
                    leaf_load - float(leaf_storage_capacity),
                    mx.array(0.0, leaf_load.dtype),
                ) / float(leaf_storage_capacity)
                overloaded = mx.stop_gradient(
                    (overflow > 0.0).astype(overflow.dtype)
                )
                overflow_terms.append(
                    mx.sum(mx.square(overflow) * overloaded)
                    / mx.maximum(
                        mx.sum(overloaded),
                        mx.array(1.0, overflow.dtype),
                    )
                )
            leaf_overflow = mx.mean(mx.stack(overflow_terms))

        candidate_set_mass = mx.array(0.0, query_logits.dtype)
        exact_boundary_positive = mx.array(0.0, query_logits.dtype)
        exact_boundary_negative = mx.array(0.0, query_logits.dtype)
        if exact_boundary_weight != 0.0 or exact_boundary_negative_weight != 0.0:
            if deployed_candidate_mask is None:
                raise ValueError(
                    "exact boundary training requires a deployed candidate mask"
                )
            boundary_slice = slice(None, None, exact_boundary_query_stride)
            boundary_distance = path_distance[:, boundary_slice]
            boundary_logit = path_match_logit[:, boundary_slice]
            boundary_eligible = eligible[boundary_slice]
            boundary_teacher = teacher_probability[:, boundary_slice]
            boundary_rank = teacher_rank[:, boundary_slice]
            deployed = (
                deployed_candidate_mask[:, boundary_slice].astype(mx.bool_)
                & boundary_eligible
            )
            target_topk = (
                boundary_rank < min(retrieval_topk, x.shape[1])
            ) & boundary_eligible
            missing = target_topk & (~deployed)
            missing_mass = mx.where(
                missing, boundary_teacher, mx.zeros_like(boundary_teacher)
            )
            missing_weights = missing_mass / mx.maximum(
                mx.sum(missing_mass, axis=-1, keepdims=True),
                mx.array(1e-12, missing_mass.dtype),
            )
            best_table = mx.argmin(
                mx.stop_gradient(boundary_distance), axis=-1
            )
            positive_table = (
                best_table[..., None] == table_index
            ).astype(boundary_logit.dtype)
            positive_boundary_weights = missing_weights[..., None] * positive_table
            exact_boundary_positive = mx.sum(
                mx.logaddexp(mx.zeros_like(boundary_logit), -boundary_logit)
                * positive_boundary_weights
            ) / mx.maximum(
                mx.sum(positive_boundary_weights),
                mx.array(1.0, positive_boundary_weights.dtype),
            )
            false_retained = deployed & (~target_topk)
            exact_path = mx.stop_gradient(
                (boundary_distance < 0.5).astype(boundary_logit.dtype)
            )
            negative_weights = (
                false_retained[..., None].astype(boundary_logit.dtype)
                * exact_path
            )
            negative_weights = negative_weights / mx.maximum(
                mx.sum(negative_weights, axis=2, keepdims=True),
                mx.array(1.0, negative_weights.dtype),
            )
            exact_boundary_negative = mx.mean(mx.sum(
                mx.logaddexp(mx.zeros_like(boundary_logit), boundary_logit)
                * negative_weights,
                axis=2,
            ))

        total = (
            positive_weight * positive_loss
            + hard_negative_weight * hard_negative_loss
            + rerank_weight * rerank_cross_entropy
            + balance_weight * balance
            + decorrelation_weight * decorrelation
            + address_entropy_weight * address_entropy
            + leaf_overflow_weight * leaf_overflow
            + exact_boundary_weight * exact_boundary_positive
            + exact_boundary_negative_weight * exact_boundary_negative
            + 0.01 * confidence
        )
        selected_count = mx.mean(mx.sum(
            selected_positive.astype(mx.float32), axis=-1
        ))
        selected_mass = mx.mean(mx.sum(
            teacher_probability
            * selected_positive.astype(teacher_probability.dtype), axis=-1
        ))
        return total, (
            positive_loss, hard_negative_loss, rerank_cross_entropy,
            balance, confidence, selected_count, selected_mass,
            decorrelation, address_entropy, leaf_overflow,
            candidate_set_mass, exact_boundary_positive,
            exact_boundary_negative,
        )


class ResidualCategoricalSecondaryRouter(nn.Module):
    """Frozen binary primary discovery plus learned local secondary bytes."""

    def __init__(
        self, query_projection, key_projection, tables=8, bits=8,
        temperature=1.0,
    ):
        super().__init__()
        if bits != 8:
            raise ValueError("residual categorical secondary codes require 8 bits")
        self.query_projection = query_projection
        self.key_projection = key_projection
        byte_bits = np.unpackbits(
            np.arange(256, dtype=np.uint8)[:, None], axis=-1,
            bitorder="little",
        ).astype(np.float32) * 2.0 - 1.0
        pattern = mx.array(byte_bits.T)
        width = query_projection.shape[0]
        query_tables = query_projection.reshape(width, tables, bits)
        key_tables = key_projection.reshape(width, tables, bits)
        query_adjacent = mx.concatenate(
            [query_tables[:, 1:], query_tables[:, :1]], axis=1
        )
        key_adjacent = mx.concatenate(
            [key_tables[:, 1:], key_tables[:, :1]], axis=1
        )
        self.secondary_query_assignment_weight = mx.einsum(
            "dtb,bc->dtc", query_adjacent, pattern
        )
        self.secondary_key_assignment_weight = mx.einsum(
            "dtb,bc->dtc", key_adjacent, pattern
        )
        self.secondary_query_assignment_bias = mx.zeros((tables, 256))
        self.secondary_key_assignment_bias = mx.zeros((tables, 256))
        self.tables = tables
        self.bits = bits
        self.categories = 256
        self.temperature = temperature
        self.freeze(keys=["query_projection", "key_projection"], strict=True)

    def secondary_logits(self, x):
        return (
            mx.einsum(
                "bnd,dtc->bntc", x,
                self.secondary_query_assignment_weight,
            ) + self.secondary_query_assignment_bias,
            mx.einsum(
                "bnd,dtc->bntc", x,
                self.secondary_key_assignment_weight,
            ) + self.secondary_key_assignment_bias,
        )

    def assignments(self, logits):
        probability = mx.softmax(logits / self.temperature, axis=-1)
        index = mx.argmax(logits, axis=-1)
        hard = (
            index[..., None]
            == mx.arange(self.categories, dtype=index.dtype)
        ).astype(probability.dtype)
        return probability + mx.stop_gradient(hard - probability), probability

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
        retrieval_topk=32, deployed_candidate_mask=None,
        exact_boundary_weight=0.0, exact_boundary_negative_weight=0.0,
        exact_boundary_query_stride=1, reranker="full-hamming",
    ):
        del (
            rerank_weight, rerank_temperature, balance_weight,
            decorrelation_weight, address_entropy_weight,
            candidate_set_temperature, candidate_set_query_stride,
            secondary_probes, reranker,
        )
        if candidate_set_weight != 0.0:
            raise ValueError("residual secondary training uses exact retained masks")
        shape = (*x.shape[:-1], self.tables, self.bits)
        primary_query_logits = (x @ self.query_projection).reshape(shape)[
            :, query_start:
        ]
        primary_key_logits = (x @ self.key_projection).reshape(shape)
        primary_query = mx.stop_gradient(mx.where(
            primary_query_logits >= 0.0,
            mx.ones_like(primary_query_logits),
            -mx.ones_like(primary_query_logits),
        ))
        primary_key = mx.stop_gradient(mx.where(
            primary_key_logits >= 0.0,
            mx.ones_like(primary_key_logits),
            -mx.ones_like(primary_key_logits),
        ))
        primary_distance = (
            self.bits - mx.einsum(
                "bqtd,bktd->bqkt", primary_query, primary_key
            )
        ) / 2.0
        primary_match = mx.stop_gradient(
            (primary_distance < 0.5).astype(primary_distance.dtype)
        )

        secondary_query_logits, secondary_key_logits = self.secondary_logits(x)
        secondary_query_logits = secondary_query_logits[:, query_start:]
        secondary_query, query_probability = self.assignments(
            secondary_query_logits
        )
        secondary_key, key_probability = self.assignments(secondary_key_logits)
        secondary_match = mx.einsum(
            "bqtc,bktc->bqkt", secondary_query, secondary_key
        )
        path_distance = 2.0 - primary_match - secondary_match
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
        selected_sorted = (
            (cumulative_before < mass_cover)
            & (mx.arange(teacher_probability.shape[-1]).reshape(1, 1, -1)
               < max_positives)
            & (sorted_mass > 0.0)
        )
        teacher_rank = mx.argsort(teacher_order, axis=-1)
        selected_positive = mx.take_along_axis(
            selected_sorted, teacher_rank, axis=-1
        ) & eligible
        has_primary = mx.any(primary_match > 0.5, axis=-1)
        addressable_positive = selected_positive & has_primary
        best_primary_table = mx.argmax(primary_match, axis=-1)
        table_index = mx.arange(self.tables).reshape(1, 1, 1, -1)
        assigned_positive = addressable_positive[..., None] & (
            best_primary_table[..., None] == table_index
        )
        powered_mass = mx.power(
            mx.maximum(
                teacher_probability,
                mx.array(0.0, teacher_probability.dtype),
            ), mass_gamma,
        ) * addressable_positive.astype(teacher_probability.dtype)
        powered_mass = powered_mass / mx.maximum(
            mx.sum(powered_mass, axis=-1, keepdims=True),
            mx.array(1e-12, powered_mass.dtype),
        )
        positive_weights = powered_mass[..., None] * assigned_positive.astype(
            powered_mass.dtype
        )
        positive_loss = mx.sum(
            positive_weights * mx.logaddexp(
                mx.zeros_like(path_match_logit), -path_match_logit
            )
        ) / mx.maximum(
            mx.sum(positive_weights), mx.array(1.0, positive_weights.dtype)
        )

        negative_mask = (
            eligible[..., None]
            & (~selected_positive[..., None])
            & (primary_match > 0.5)
        )
        negative_logits = mx.where(
            negative_mask,
            -mx.stop_gradient(path_distance) / hard_negative_temperature,
            mx.array(-1e9, path_distance.dtype),
        )
        negative_weights = mx.softmax(negative_logits, axis=2) * (
            negative_mask.astype(path_distance.dtype)
        )
        negative_weights = mx.stop_gradient(
            negative_weights / mx.maximum(
                mx.sum(negative_weights, axis=2, keepdims=True),
                mx.array(1e-12, negative_weights.dtype),
            )
        )
        hard_negative_loss = mx.mean(mx.sum(
            negative_weights * mx.logaddexp(
                mx.zeros_like(path_match_logit), path_match_logit
            ), axis=2,
        ))

        rerank_cross_entropy = mx.array(0.0, path_distance.dtype)
        balance = mx.array(0.0, path_distance.dtype)
        decorrelation = mx.array(0.0, path_distance.dtype)
        address_entropy = mx.array(0.0, path_distance.dtype)
        confidence = mx.mean(mx.stack([
            mx.mean(1.0 - mx.max(query_probability, axis=-1)),
            mx.mean(1.0 - mx.max(key_probability, axis=-1)),
        ]))

        leaf_overflow = mx.array(0.0, path_distance.dtype)
        if leaf_overflow_weight != 0.0:
            if leaf_storage_capacity < 1:
                raise ValueError("leaf_storage_capacity must be positive")
            key_primary_match = mx.stop_gradient(
                (primary_key[:, :, None] == primary_key[:, None, :])
                .all(axis=-1).astype(path_distance.dtype)
            )
            overflow_terms = []
            for table in range(self.tables):
                secondary_same = mx.einsum(
                    "bkc,bjc->bkj",
                    secondary_key[:, :, table],
                    secondary_key[:, :, table],
                )
                same_leaf = key_primary_match[..., table] * secondary_same
                leaf_load = mx.sum(same_leaf, axis=-1)
                overflow = mx.maximum(
                    leaf_load - float(leaf_storage_capacity),
                    mx.array(0.0, leaf_load.dtype),
                ) / float(leaf_storage_capacity)
                overloaded = mx.stop_gradient(
                    (overflow > 0.0).astype(overflow.dtype)
                )
                overflow_terms.append(
                    mx.sum(mx.square(overflow) * overloaded)
                    / mx.maximum(
                        mx.sum(overloaded), mx.array(1.0, overflow.dtype)
                    )
                )
            leaf_overflow = mx.mean(mx.stack(overflow_terms))

        candidate_set_mass = mx.array(0.0, path_distance.dtype)
        exact_boundary_positive = mx.array(0.0, path_distance.dtype)
        exact_boundary_negative = mx.array(0.0, path_distance.dtype)
        if exact_boundary_weight != 0.0 or exact_boundary_negative_weight != 0.0:
            if deployed_candidate_mask is None:
                raise ValueError("exact boundary training requires deployed masks")
            boundary_slice = slice(None, None, exact_boundary_query_stride)
            boundary_distance = path_distance[:, boundary_slice]
            boundary_logit = path_match_logit[:, boundary_slice]
            boundary_primary = primary_match[:, boundary_slice]
            boundary_eligible = eligible[boundary_slice]
            boundary_teacher = teacher_probability[:, boundary_slice]
            boundary_rank = teacher_rank[:, boundary_slice]
            deployed = (
                deployed_candidate_mask[:, boundary_slice].astype(mx.bool_)
                & boundary_eligible
            )
            target_topk = (
                boundary_rank < min(retrieval_topk, x.shape[1])
            ) & boundary_eligible
            addressable = mx.any(boundary_primary > 0.5, axis=-1)
            missing = target_topk & (~deployed) & addressable
            missing_mass = mx.where(
                missing, boundary_teacher, mx.zeros_like(boundary_teacher)
            )
            missing_weights = missing_mass / mx.maximum(
                mx.sum(missing_mass, axis=-1, keepdims=True),
                mx.array(1e-12, missing_mass.dtype),
            )
            masked_distance = mx.where(
                boundary_primary > 0.5, boundary_distance,
                mx.array(1e9, boundary_distance.dtype),
            )
            best_table = mx.argmin(mx.stop_gradient(masked_distance), axis=-1)
            positive_table = (
                best_table[..., None] == table_index
            ).astype(boundary_logit.dtype)
            positive_weights_boundary = missing_weights[..., None] * positive_table
            exact_boundary_positive = mx.sum(
                mx.logaddexp(mx.zeros_like(boundary_logit), -boundary_logit)
                * positive_weights_boundary
            ) / mx.maximum(
                mx.sum(positive_weights_boundary),
                mx.array(1.0, positive_weights_boundary.dtype),
            )
            false_retained = deployed & (~target_topk)
            exact_path = mx.stop_gradient(
                (boundary_distance < 0.5).astype(boundary_logit.dtype)
            )
            negative_weights_boundary = (
                false_retained[..., None].astype(boundary_logit.dtype)
                * exact_path
            )
            negative_weights_boundary = negative_weights_boundary / mx.maximum(
                mx.sum(negative_weights_boundary, axis=2, keepdims=True),
                mx.array(1.0, negative_weights_boundary.dtype),
            )
            exact_boundary_negative = mx.mean(mx.sum(
                mx.logaddexp(mx.zeros_like(boundary_logit), boundary_logit)
                * negative_weights_boundary,
                axis=2,
            ))

        total = (
            positive_weight * positive_loss
            + hard_negative_weight * hard_negative_loss
            + leaf_overflow_weight * leaf_overflow
            + exact_boundary_weight * exact_boundary_positive
            + exact_boundary_negative_weight * exact_boundary_negative
            + 0.01 * confidence
        )
        selected_count = mx.mean(mx.sum(
            addressable_positive.astype(mx.float32), axis=-1
        ))
        selected_mass = mx.mean(mx.sum(
            teacher_probability
            * addressable_positive.astype(teacher_probability.dtype), axis=-1
        ))
        return total, (
            positive_loss, hard_negative_loss, rerank_cross_entropy,
            balance, confidence, selected_count, selected_mass,
            decorrelation, address_entropy, leaf_overflow,
            candidate_set_mass, exact_boundary_positive,
            exact_boundary_negative,
        )


class PrimaryConditionedResidualSecondaryRouter(
    ResidualCategoricalSecondaryRouter
):
    """Frozen primary/local projections plus per-primary secondary biases."""

    def __init__(
        self, query_projection, key_projection, tables=8, bits=8,
        temperature=1.0,
    ):
        super().__init__(
            query_projection, key_projection, tables=tables, bits=bits,
            temperature=temperature,
        )
        self.secondary_query_primary_bias = mx.zeros(
            (tables, 256, self.categories)
        )
        self.secondary_key_primary_bias = mx.zeros(
            (tables, 256, self.categories)
        )
        self.freeze(keys=[
            "secondary_query_assignment_weight",
            "secondary_query_assignment_bias",
            "secondary_key_assignment_weight",
            "secondary_key_assignment_bias",
        ], strict=True)

    def primary_codes(self, x, projection):
        logits = (x @ projection).reshape(
            *x.shape[:-1], self.tables, self.bits
        )
        powers = mx.array(
            1 << np.arange(self.bits, dtype=np.uint16), dtype=mx.uint16
        )
        return mx.sum(
            (logits >= 0.0).astype(mx.uint16) * powers, axis=-1
        ).astype(mx.int32)

    def conditioned_bias(self, bias, primary_codes):
        table_offsets = (
            mx.arange(self.tables, dtype=mx.int32)
            .reshape(*([1] * (primary_codes.ndim - 1)), self.tables)
            * 256
        )
        return mx.take(
            bias.reshape(self.tables * 256, self.categories),
            table_offsets + primary_codes,
            axis=0,
        )

    def secondary_logits(self, x):
        query_logits, key_logits = super().secondary_logits(x)
        query_primary = self.primary_codes(x, self.query_projection)
        key_primary = self.primary_codes(x, self.key_projection)
        return (
            query_logits + self.conditioned_bias(
                self.secondary_query_primary_bias, query_primary
            ),
            key_logits + self.conditioned_bias(
                self.secondary_key_primary_bias, key_primary
            ),
        )


class JointBinaryAttentionDecoder(nn.Module):
    def __init__(
        self, key_projection, kv_heads, query_heads, head_dim,
        query_weight, query_norm,
    ):
        super().__init__()
        self.key_projection = key_projection
        bits = key_projection.shape[-1]
        self.decoder = mx.random.normal((bits, kv_heads * head_dim)) / np.sqrt(bits)
        self.decoder_hidden_weight = mx.random.normal((bits, 128)) / np.sqrt(bits)
        self.decoder_hidden_bias = mx.zeros((128,))
        self.decoder_output_weight = mx.zeros((128, kv_heads * head_dim))
        self.decoder_output_bias = mx.zeros((kv_heads * head_dim,))
        self.head_bias_weight = mx.zeros((key_projection.shape[0], query_heads))
        self.head_bias = mx.zeros((query_heads,))
        self.query_weight = query_weight
        self.query_norm = query_norm

    @staticmethod
    def straight_through_sign(logits):
        soft = mx.tanh(logits)
        hard = mx.where(logits >= 0, 1.0, -1.0)
        return soft + mx.stop_gradient(hard - soft)

    def __call__(self, hidden, kv_heads, head_dim):
        logits = hidden @ self.key_projection
        code = self.straight_through_sign(logits)
        decoder_hidden = mx.tanh(
            code @ self.decoder_hidden_weight + self.decoder_hidden_bias
        )
        decoded_flat = (
            code @ self.decoder
            + decoder_hidden @ self.decoder_output_weight
            + self.decoder_output_bias
        )
        decoded = decoded_flat.reshape(
            hidden.shape[0], kv_heads, head_dim
        )
        return decoded, logits


class JointVQAttentionDecoder(nn.Module):
    """Learned byte assignments with additive decoded-key codebooks."""

    def __init__(
        self, key_projection, decoder, kv_heads, query_heads, head_dim,
        temperature,
    ):
        super().__init__()
        width, bits = key_projection.shape
        if bits % 8:
            raise ValueError("VQ initialization requires whole-byte source codes")
        tables = bits // 8
        byte_bits = np.unpackbits(
            np.arange(256, dtype=np.uint8)[:, None], axis=-1,
            bitorder="little",
        ).astype(np.float32) * 2.0 - 1.0
        assignment = []
        codebooks = []
        for table in range(tables):
            bit_slice = slice(table * 8, (table + 1) * 8)
            assignment.append(key_projection[:, bit_slice] @ mx.array(byte_bits.T))
            codebooks.append(mx.array(byte_bits) @ decoder[bit_slice])
        self.assignment_weight = mx.stack(assignment, axis=1)
        self.assignment_bias = mx.zeros((tables, 256))
        self.codebooks = mx.stack(codebooks, axis=0)
        self.temperature = temperature
        self.head_bias_weight = mx.zeros((width, query_heads))
        self.head_bias = mx.zeros((query_heads,))

    def __call__(self, hidden, kv_heads, head_dim):
        logits = mx.einsum("nd,dtc->ntc", hidden, self.assignment_weight)
        logits = logits + self.assignment_bias
        probability = mx.softmax(logits / self.temperature, axis=-1)
        indices = mx.argmax(logits, axis=-1)
        hard = (
            indices[..., None] == mx.arange(256, dtype=indices.dtype)
        ).astype(probability.dtype)
        assignment = probability + mx.stop_gradient(hard - probability)
        decoded_flat = mx.einsum("ntc,tcf->nf", assignment, self.codebooks)
        return decoded_flat.reshape(hidden.shape[0], kv_heads, head_dim), logits


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--router", required=True, type=pathlib.Path)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--fingerprint-bytes", type=int, default=4)
    parser.add_argument("--code-type", choices=("binary", "vq"), default="binary")
    parser.add_argument("--vq-temperature", type=float, default=0.5)
    parser.add_argument("--secondary-probes", type=int, default=3)
    parser.add_argument("--leaf-capacity", type=int, default=14)
    parser.add_argument("--storage-capacity", type=int, default=32)
    parser.add_argument("--corpora", default="wikitext2,pg19")
    parser.add_argument("--train-segments-per-corpus", type=int, default=32)
    parser.add_argument("--eval-segments-per-corpus", type=int, default=4)
    parser.add_argument(
        "--objective", choices=("reconstruction", "attention"),
        default="attention",
    )
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--confidence-weight", type=float, default=0.001)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.2)
    parser.add_argument("--hard-negatives", type=int, default=64)
    parser.add_argument(
        "--boundary-shortlist", type=int, default=0,
        help=(
            "restrict the listwise teacher loss to the scorer's current top-N "
            "candidates (0 uses the full deployed pool)"
        ),
    )
    parser.add_argument(
        "--teacher-topk", type=int, default=0,
        help=(
            "restrict the listwise target to the teacher's highest-mass K "
            "retained candidates (0 preserves the full-mass objective)"
        ),
    )
    parser.add_argument(
        "--normalize-heads", action="store_true",
        help="normalize each attention head across the deployed candidate pool",
    )
    parser.add_argument(
        "--linear-only", action="store_true",
        help="train the binary projection and linear decoder only",
    )
    parser.add_argument(
        "--nonlinear-only", action="store_true",
        help=(
            "freeze the binary projection and linear decoder, training only "
            "the zero-initialized nonlinear residual"
        ),
    )
    parser.add_argument(
        "--head-bias-only", action="store_true",
        help=(
            "freeze the key code and decoders, training only a query-conditioned "
            "per-head calibration term"
        ),
    )
    parser.add_argument(
        "--query-only", action="store_true",
        help="freeze the key code/decoder and train only the attention query map",
    )
    parser.add_argument(
        "--train-query", action="store_true",
        help="jointly train the attention query map with the key code/decoder",
    )
    parser.add_argument(
        "--joint-address", action="store_true",
        help=(
            "interleave exact retained-boundary address updates with scorer "
            "updates against the same mined causal pool"
        ),
    )
    parser.add_argument(
        "--joint-address-thresholds", action="store_true",
        help=(
            "learn per-table query/key sign thresholds while preserving the "
            "fixed 64-bit address projections"
        ),
    )
    parser.add_argument(
        "--joint-address-categorical", action="store_true",
        help=(
            "replace correlated sign-bit bytes with direct learned 256-way "
            "assignments for every address table"
        ),
    )
    parser.add_argument(
        "--joint-address-residual-secondary", action="store_true",
        help=(
            "freeze binary primary discovery and learn only a categorical "
            "secondary partition inside each primary table"
        ),
    )
    parser.add_argument(
        "--joint-address-primary-conditioned-secondary", action="store_true",
        help=(
            "freeze binary primary and shared secondary projections, learning "
            "only per-primary-region categorical secondary biases"
        ),
    )
    parser.add_argument(
        "--address-categorical-temperature", type=float, default=1.0
    )
    parser.add_argument("--address-lr", type=float, default=3e-4)
    parser.add_argument("--address-overflow-weight", type=float, default=0.3)
    parser.add_argument("--address-boundary-weight", type=float, default=1.0)
    parser.add_argument(
        "--address-boundary-negative-weight", type=float, default=0.5
    )
    parser.add_argument("--address-boundary-query-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--group-dro-beta", type=float, default=0.0,
        help="softmax strength for worst-corpus loss upweighting (0 disables)",
    )
    parser.add_argument(
        "--group-dro-ema", type=float, default=0.99,
        help="EMA decay for per-corpus training losses",
    )
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=64)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if not args.router.is_file():
        parser.error("--router does not exist")
    if args.fingerprint_bytes not in (4, 5, 6, 7):
        parser.error("--fingerprint-bytes must be 4, 5, 6, or 7")
    if args.boundary_shortlist and args.boundary_shortlist < 32:
        parser.error("--boundary-shortlist must be zero or at least 32")
    if args.teacher_topk < 0:
        parser.error("--teacher-topk must be non-negative")
    if args.group_dro_beta < 0:
        parser.error("--group-dro-beta must be nonnegative")
    if not 0.0 <= args.group_dro_ema < 1.0:
        parser.error("--group-dro-ema must be in [0, 1)")
    if args.joint_address and (
        args.code_type != "binary" or args.objective != "attention"
    ):
        parser.error("--joint-address requires binary attention training")
    if args.joint_address_thresholds and not args.joint_address:
        parser.error("--joint-address-thresholds requires --joint-address")
    if args.joint_address_categorical and not args.joint_address:
        parser.error("--joint-address-categorical requires --joint-address")
    if args.joint_address_residual_secondary and not args.joint_address:
        parser.error(
            "--joint-address-residual-secondary requires --joint-address"
        )
    if (
        args.joint_address_primary_conditioned_secondary
        and not args.joint_address
    ):
        parser.error(
            "--joint-address-primary-conditioned-secondary requires "
            "--joint-address"
        )
    if sum((
        args.joint_address_thresholds,
        args.joint_address_categorical,
        args.joint_address_residual_secondary,
        args.joint_address_primary_conditioned_secondary,
    )) > 1:
        parser.error(
            "joint address representation flags are mutually exclusive"
        )
    if args.address_categorical_temperature <= 0:
        parser.error("--address-categorical-temperature must be positive")
    if args.address_lr <= 0:
        parser.error("--address-lr must be positive")
    if min(
        args.address_overflow_weight,
        args.address_boundary_weight,
        args.address_boundary_negative_weight,
    ) < 0:
        parser.error("joint address loss weights must be nonnegative")
    if args.address_boundary_query_stride < 1:
        parser.error("--address-boundary-query-stride must be positive")
    corpora = parse_corpora(args.corpora)
    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    mx.random.seed(args.seed)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    layer = language_body(model).layers[args.layer]
    attention = layer.self_attn
    weights = mx.load(str(args.router))
    rerank_width = args.fingerprint_bytes * 8
    address_router = None
    if args.joint_address:
        if (
            args.joint_address_residual_secondary
            or args.joint_address_primary_conditioned_secondary
        ):
            residual_router_class = (
                PrimaryConditionedResidualSecondaryRouter
                if args.joint_address_primary_conditioned_secondary
                else ResidualCategoricalSecondaryRouter
            )
            address_router = residual_router_class(
                weights["query_projection"], weights["key_projection"],
                temperature=args.address_categorical_temperature,
            )
            if "secondary_query_assignment_weight" in weights:
                for name in (
                    "secondary_query_assignment_weight",
                    "secondary_query_assignment_bias",
                    "secondary_key_assignment_weight",
                    "secondary_key_assignment_bias",
                ):
                    setattr(address_router, name, weights[name])
            if args.joint_address_primary_conditioned_secondary:
                for name in (
                    "secondary_query_primary_bias",
                    "secondary_key_primary_bias",
                ):
                    if name in weights:
                        setattr(address_router, name, weights[name])
        elif args.joint_address_categorical:
            address_router = CategoricalHierarchicalAddressRouter(
                weights["query_projection"], weights["key_projection"],
                temperature=args.address_categorical_temperature,
            )
            if "address_query_assignment_weight" in weights:
                for name in (
                    "address_query_assignment_weight",
                    "address_query_assignment_bias",
                    "address_key_assignment_weight",
                    "address_key_assignment_bias",
                ):
                    setattr(address_router, name, weights[name])
        else:
            address_router_class = (
                ThresholdedHierarchicalAttentionRouter
                if args.joint_address_thresholds else HierarchicalAttentionRouter
            )
            address_router = address_router_class(
                weights["query_projection"].shape[0], 8, 8,
                rerank_bytes=args.fingerprint_bytes,
            )
            address_router.query_projection = weights["query_projection"]
            address_router.key_projection = weights["key_projection"]
            if args.joint_address_thresholds:
                if "address_query_bias" in weights:
                    address_router.address_query_bias = weights[
                        "address_query_bias"
                    ]
                    address_router.address_key_bias = weights[
                        "address_key_bias"
                    ]
                address_router.freeze(
                    keys=["query_projection", "key_projection"], strict=True
                )
            address_router.freeze(keys=[
                "rerank_query_projection", "rerank_key_projection",
                "rerank_bit_weights", "rerank_bilinear", "rerank_lookup",
                "rerank_decoder_query", "rerank_decoder_keys",
                "rerank_distance_bias", "rerank_query_lookup_weight",
                "rerank_query_lookup_bias", "retention_projection",
            ], strict=True)
        mx.eval(address_router.parameters())
    source_key_projection = weights["rerank_key_projection"][:, :rerank_width]
    if source_key_projection.shape[-1] < rerank_width:
        source_key_projection = mx.concatenate([
            source_key_projection,
            weights["key_projection"][
                :, source_key_projection.shape[-1]:rerank_width
            ],
        ], axis=-1)
    source_decoder = weights.get(
        "joint_binary_attention_decoder",
        mx.random.normal((rerank_width, attention.n_kv_heads * attention.head_dim))
        / np.sqrt(rerank_width),
    )
    if source_decoder.shape[0] < rerank_width:
        source_decoder = mx.concatenate([
            source_decoder,
            mx.zeros((
                rerank_width - source_decoder.shape[0],
                source_decoder.shape[1],
            )),
        ], axis=0)
    if args.code_type == "vq":
        module = JointVQAttentionDecoder(
            source_key_projection, source_decoder, attention.n_kv_heads,
            attention.n_heads, attention.head_dim, args.vq_temperature,
        )
        if "joint_vq_assignment_weight" in weights:
            module.assignment_weight = weights["joint_vq_assignment_weight"]
            module.assignment_bias = weights["joint_vq_assignment_bias"]
            module.codebooks = weights["joint_vq_attention_decoder"]
            module.head_bias_weight = weights[
                "joint_binary_attention_head_bias_weight"
            ]
            module.head_bias = weights["joint_binary_attention_head_bias"]
    else:
        module = JointBinaryAttentionDecoder(
            source_key_projection, attention.n_kv_heads, attention.n_heads,
            attention.head_dim, attention.q_proj.weight,
            attention.q_layernorm.weight,
        )
        module.decoder = source_decoder
        for name in (
            "decoder_hidden_weight", "decoder_hidden_bias",
            "decoder_output_weight", "decoder_output_bias",
        ):
            checkpoint_name = f"joint_binary_attention_{name}"
            if checkpoint_name in weights and (
                "hidden_weight" not in name
                or weights[checkpoint_name].shape[0] == rerank_width
            ):
                setattr(module, name, weights[checkpoint_name])
        if "joint_binary_attention_head_bias_weight" in weights:
            module.head_bias_weight = weights[
                "joint_binary_attention_head_bias_weight"
            ]
            module.head_bias = weights["joint_binary_attention_head_bias"]
    if sum((
        args.nonlinear_only, args.head_bias_only, args.linear_only,
        args.query_only,
    )) > 1:
        parser.error(
            "--nonlinear-only, --head-bias-only, --linear-only, and "
            "--query-only are "
            "mutually exclusive"
        )
    if args.code_type == "vq" and (
        args.nonlinear_only or args.head_bias_only or args.linear_only
        or args.query_only or args.train_query
    ):
        parser.error("binary-only freezing modes cannot be used with --code-type vq")
    if args.query_only and args.train_query:
        parser.error("--query-only and --train-query are mutually exclusive")
    if args.nonlinear_only:
        module.freeze(keys=["key_projection", "decoder"], strict=True)
        module.freeze(keys=["head_bias_weight", "head_bias"], strict=True)
    if args.head_bias_only:
        module.freeze(
            keys=[
                "key_projection", "decoder", "decoder_hidden_weight",
                "decoder_hidden_bias", "decoder_output_weight",
                "decoder_output_bias",
            ],
            strict=True,
        )
    if args.query_only:
        module.freeze(
            keys=[
                "key_projection", "decoder", "decoder_hidden_weight",
                "decoder_hidden_bias", "decoder_output_weight",
                "decoder_output_bias", "head_bias_weight", "head_bias",
            ],
            strict=True,
        )
    elif args.code_type == "binary" and not args.train_query:
        module.freeze(keys=["query_weight", "query_norm"], strict=True)
    if args.linear_only:
        module.freeze(
            keys=[
                "decoder_hidden_weight", "decoder_hidden_bias",
                "decoder_output_weight", "decoder_output_bias",
                "head_bias_weight", "head_bias",
            ],
            strict=True,
        )

    query_projection = np.array(
        weights["query_projection"].astype(mx.float16).astype(mx.float32)
    )
    address_key_projection = np.array(
        weights["key_projection"].astype(mx.float16).astype(mx.float32)
    )
    address_query_bias = np.array(
        weights.get("address_query_bias", mx.zeros((8, 8))).astype(mx.float32)
    )
    address_key_bias = np.array(
        weights.get("address_key_bias", mx.zeros((8, 8))).astype(mx.float32)
    )
    categorical_query_weight = (
        np.array(weights["address_query_assignment_weight"].astype(mx.float32))
        if "address_query_assignment_weight" in weights else None
    )
    categorical_query_bias = (
        np.array(weights["address_query_assignment_bias"].astype(mx.float32))
        if "address_query_assignment_bias" in weights else None
    )
    categorical_key_weight = (
        np.array(weights["address_key_assignment_weight"].astype(mx.float32))
        if "address_key_assignment_weight" in weights else None
    )
    categorical_key_bias = (
        np.array(weights["address_key_assignment_bias"].astype(mx.float32))
        if "address_key_assignment_bias" in weights else None
    )
    residual_query_weight = (
        np.array(weights["secondary_query_assignment_weight"].astype(mx.float32))
        if "secondary_query_assignment_weight" in weights else None
    )
    residual_query_bias = (
        np.array(weights["secondary_query_assignment_bias"].astype(mx.float32))
        if "secondary_query_assignment_bias" in weights else None
    )
    residual_key_weight = (
        np.array(weights["secondary_key_assignment_weight"].astype(mx.float32))
        if "secondary_key_assignment_weight" in weights else None
    )
    residual_key_bias = (
        np.array(weights["secondary_key_assignment_bias"].astype(mx.float32))
        if "secondary_key_assignment_bias" in weights else None
    )
    residual_query_primary_bias = (
        np.array(weights["secondary_query_primary_bias"].astype(mx.float32))
        if "secondary_query_primary_bias" in weights else None
    )
    residual_key_primary_bias = (
        np.array(weights["secondary_key_primary_bias"].astype(mx.float32))
        if "secondary_key_primary_bias" in weights else None
    )
    source_query_projection = weights[
        "rerank_query_projection"
    ][:, :rerank_width]
    if source_query_projection.shape[-1] < rerank_width:
        source_query_projection = mx.concatenate([
            source_query_projection,
            weights["query_projection"][
                :, source_query_projection.shape[-1]:rerank_width
            ],
        ], axis=-1)
    rerank_query_projection = np.array(
        source_query_projection.astype(mx.float16).astype(mx.float32)
    )
    initial_key_projection = np.array(
        source_key_projection.astype(mx.float16).astype(mx.float32)
    )

    train_rows = corpus_segments(
        tokenizer, corpora, args.seq_len,
        args.train_segments_per_corpus, skip=0,
    )
    eval_rows = corpus_segments(
        tokenizer, corpora, args.seq_len,
        args.eval_segments_per_corpus, skip=args.train_segments_per_corpus,
    )

    def capture_example(corpus, tokens):
        capture = capture_layer(model, tokens, args.layer)
        hidden_np = np.array(capture["x"][0].astype(mx.float32)).copy()
        if args.objective == "reconstruction":
            target = attention.k_proj(capture["x"]).reshape(
                1, args.seq_len, attention.n_kv_heads, attention.head_dim
            )
            target = attention.k_layernorm(target)[0].astype(mx.float16)
            query_heads = mx.zeros((1, 1, 1), dtype=mx.float16)
            teacher = mx.zeros((1, 1), dtype=mx.float16)
            candidate_mask = mx.zeros((1, 1), dtype=mx.bool_)
        else:
            query_start = args.window + args.sink_tokens + 1
            query_heads = capture["queries"][0, :, query_start:].transpose(
                1, 0, 2
            ).astype(mx.float16)
            exact_keys = capture["keys"][0].astype(mx.float32)
            exact_logits = mx.einsum(
                "qhd,hkd->qhk", query_heads.astype(mx.float32), exact_keys
            ) * attention.scale
            query_positions = mx.arange(query_start, args.seq_len).reshape(-1, 1)
            key_positions = mx.arange(args.seq_len).reshape(1, -1)
            eligible = (key_positions < query_positions - args.window) & (
                key_positions >= args.sink_tokens
            )
            exact_logits = mx.where(
                eligible[:, None, :], exact_logits,
                mx.array(-1e9, exact_logits.dtype),
            )
            teacher = mx.mean(
                mx.softmax(exact_logits, axis=-1), axis=1
            ).astype(mx.float16)
            target = mx.zeros((1, 1, 1), dtype=mx.float16)

            if categorical_query_weight is not None:
                query_logits, query_codes, _ = categorical_address_codes(
                    hidden_np, categorical_query_weight,
                    categorical_query_bias,
                )
                _, key_codes, _ = categorical_address_codes(
                    hidden_np, categorical_key_weight, categorical_key_bias,
                )
            else:
                query_logits, query_codes, _ = router_codes(
                    hidden_np, query_projection, 8, 8,
                    bias=address_query_bias,
                )
                _, key_codes, _ = router_codes(
                    hidden_np, address_key_projection, 8, 8,
                    bias=address_key_bias,
                )
            residual_query_logits = None
            residual_key_codes = None
            if residual_query_weight is not None:
                if residual_query_primary_bias is None:
                    residual_query_logits, _, _ = categorical_address_codes(
                        hidden_np, residual_query_weight, residual_query_bias,
                    )
                    _, residual_key_codes, _ = categorical_address_codes(
                        hidden_np, residual_key_weight, residual_key_bias,
                    )
                else:
                    residual_query_logits, _, _ = (
                        primary_conditioned_categorical_address_codes(
                            hidden_np, residual_query_weight,
                            residual_query_bias, query_projection,
                            residual_query_primary_bias,
                        )
                    )
                    _, residual_key_codes, _ = (
                        primary_conditioned_categorical_address_codes(
                            hidden_np, residual_key_weight,
                            residual_key_bias, address_key_projection,
                            residual_key_primary_bias,
                        )
                    )
            query_bytes = binary_fingerprint_bytes(
                hidden_np, rerank_query_projection
            )
            key_bytes = binary_fingerprint_bytes(
                hidden_np, initial_key_projection
            )
            _, _, _, retained, _ = causal_hierarchical_candidates(
                query_logits, query_codes, query_bytes, key_codes, key_bytes,
                args.window, args.sink_tokens, args.secondary_probes,
                args.leaf_capacity, 32, "full-hamming", "reservoir",
                storage_capacity=args.storage_capacity,
                candidate_budget=8 * args.secondary_probes * args.leaf_capacity,
                secondary_query_logits=residual_query_logits,
                secondary_key_codes=residual_key_codes,
            )
            mask_np = np.zeros(
                (args.seq_len - query_start, args.seq_len), dtype=np.bool_
            )
            for offset, positions in enumerate(retained[query_start:]):
                valid = positions[positions >= 0]
                mask_np[offset, valid] = True
            candidate_mask = mx.array(mask_np)

        hidden = mx.stop_gradient(capture["x"][0].astype(mx.float16))
        target = mx.stop_gradient(target)
        query_heads = mx.stop_gradient(query_heads)
        teacher = mx.stop_gradient(teacher)
        candidate_mask = mx.stop_gradient(candidate_mask)
        mx.eval(hidden, target, query_heads, teacher, candidate_mask)
        del capture
        mx.clear_cache()
        if args.joint_address:
            packed_mask = np.packbits(
                np.array(candidate_mask), axis=-1, bitorder="little"
            )
            return (
                corpus,
                np.array(hidden.astype(mx.float16)).copy(),
                np.array(target.astype(mx.float16)).copy(),
                np.array(query_heads.astype(mx.float16)).copy(),
                np.array(teacher.astype(mx.float16)).copy(),
                (packed_mask, candidate_mask.shape[-1]),
            )
        return corpus, hidden, target, query_heads, teacher, candidate_mask

    train_examples = [capture_example(*row) for row in train_rows]
    eval_examples = [capture_example(*row) for row in eval_rows]
    del model
    mx.clear_cache()

    def materialize_example(row):
        corpus, hidden, target, queries, teacher, mask = row
        if isinstance(hidden, np.ndarray):
            hidden = mx.array(hidden)
            target = mx.array(target)
            queries = mx.array(queries)
            teacher = mx.array(teacher)
        if isinstance(mask, tuple):
            packed, width = mask
            mask = mx.array(np.unpackbits(
                packed, axis=-1, count=width, bitorder="little"
            ).astype(np.bool_))
        return corpus, hidden, target, queries, teacher, mask

    def loss_fn(
        current, hidden, target, query_heads, teacher, candidate_mask,
        example_weight=1.0,
    ):
        decoded, logits = current(
            hidden.astype(mx.float32), attention.n_kv_heads, attention.head_dim
        )
        if args.objective == "reconstruction":
            cross_entropy = mx.mean(mx.square(
                decoded - target.astype(mx.float32)
            ))
            pairwise = mx.array(0.0, cross_entropy.dtype)
        else:
            decoded_keys = apply_rope(decoded)
            decoded_keys = mx.repeat(
                decoded_keys,
                query_heads.shape[1] // decoded_keys.shape[1], axis=1,
            )
            if args.query_only or args.train_query:
                learned_queries = (
                    hidden.astype(mx.float32) @ current.query_weight.T
                ).reshape(
                    hidden.shape[0], attention.n_heads, attention.head_dim
                )
                learned_queries = learned_queries * mx.rsqrt(
                    mx.mean(mx.square(learned_queries), axis=-1, keepdims=True)
                    + 1e-5
                ) * current.query_norm.reshape(1, 1, -1)
                learned_queries = apply_rope(learned_queries)[
                    -query_heads.shape[0]:
                ]
            else:
                learned_queries = query_heads.astype(mx.float32)
            scores_by_head = mx.einsum(
                "qhd,khd->qkh", learned_queries, decoded_keys
            ) * attention.scale
            query_hidden = hidden[-query_heads.shape[0]:].astype(mx.float32)
            head_bias = (
                query_hidden @ current.head_bias_weight + current.head_bias
            )
            scores_by_head = scores_by_head + head_bias[:, None, :]
            if args.normalize_heads:
                masked_head_scores = mx.where(
                    candidate_mask[..., None], scores_by_head,
                    mx.array(-1e9, scores_by_head.dtype),
                )
                scores_by_head = scores_by_head - mx.logsumexp(
                    masked_head_scores, axis=1, keepdims=True
                )
            scores = mx.logsumexp(scores_by_head, axis=-1)
            scores = mx.where(
                candidate_mask, scores, mx.array(-1e9, scores.dtype)
            )
            training_mask = candidate_mask
            if args.boundary_shortlist:
                score_order = mx.argsort(-mx.stop_gradient(scores), axis=-1)
                score_rank = mx.argsort(score_order, axis=-1)
                training_mask = candidate_mask & (
                    score_rank < args.boundary_shortlist
                )
                scores = mx.where(
                    training_mask, scores, mx.array(-1e9, scores.dtype)
                )
            target_pool = mx.where(
                training_mask, teacher.astype(mx.float32),
                mx.zeros_like(teacher.astype(mx.float32)),
            )
            if args.teacher_topk:
                teacher_order = mx.argsort(-target_pool, axis=-1)
                teacher_rank = mx.argsort(teacher_order, axis=-1)
                target_pool = mx.where(
                    teacher_rank < args.teacher_topk,
                    target_pool,
                    mx.zeros_like(target_pool),
                )
            target_pool = target_pool / mx.maximum(
                mx.sum(target_pool, axis=-1, keepdims=True),
                mx.array(1e-12, target_pool.dtype),
            )
            log_probability = scores - mx.logsumexp(
                scores, axis=-1, keepdims=True
            )
            cross_entropy = -mx.mean(mx.sum(
                target_pool * log_probability, axis=-1
            ))
            positive_count = min(32, scores.shape[-1])
            teacher_order = mx.argsort(-target_pool, axis=-1)
            positive_indices = teacher_order[..., :positive_count]
            positive_mass = mx.take_along_axis(
                target_pool, positive_indices, axis=-1
            )
            positive_weights = positive_mass / mx.maximum(
                mx.sum(positive_mass, axis=-1, keepdims=True),
                mx.array(1e-12, positive_mass.dtype),
            )
            teacher_rank = mx.argsort(teacher_order, axis=-1)
            negative_mask = training_mask & (teacher_rank >= positive_count)
            negative_scores = mx.where(
                negative_mask, mx.stop_gradient(scores),
                mx.array(-1e9, scores.dtype),
            )
            negative_count = min(args.hard_negatives, scores.shape[-1])
            negative_indices = mx.argsort(-negative_scores, axis=-1)[
                ..., :negative_count
            ]
            negative_valid = mx.take_along_axis(
                negative_mask, negative_indices, axis=-1
            ).astype(scores.dtype)
            negative_weights = negative_valid / mx.maximum(
                mx.sum(negative_valid, axis=-1, keepdims=True),
                mx.array(1.0, scores.dtype),
            )
            positive_score = mx.take_along_axis(
                scores, positive_indices, axis=-1
            )
            negative_score = mx.take_along_axis(
                scores, negative_indices, axis=-1
            )
            pairwise_terms = mx.logaddexp(
                mx.zeros_like(
                    positive_score[..., :, None]
                    - negative_score[..., None, :]
                ),
                args.pairwise_margin
                - positive_score[..., :, None]
                + negative_score[..., None, :],
            )
            pairwise_weights = (
                positive_weights[..., :, None]
                * negative_weights[..., None, :]
            )
            pairwise = mx.sum(
                pairwise_terms * pairwise_weights
            ) / mx.maximum(
                mx.sum(pairwise_weights),
                mx.array(1.0, pairwise_weights.dtype),
            )
        task_loss = cross_entropy + args.pairwise_weight * pairwise
        if args.code_type == "vq":
            probability = mx.softmax(logits / args.vq_temperature, axis=-1)
            balance = 256.0 * mx.mean(mx.square(
                mx.mean(probability, axis=0) - 1.0 / 256.0
            ))
            confidence = mx.mean(1.0 - mx.max(probability, axis=-1))
        else:
            probability = mx.sigmoid(logits)
            balance = mx.mean(mx.square(mx.mean(probability, axis=0) - 0.5))
            confidence = mx.mean(probability * (1.0 - probability))
        total = (
            task_loss + args.balance_weight * balance
            + args.confidence_weight * confidence
        )
        return (
            total * example_weight,
            (task_loss, cross_entropy, pairwise, balance, confidence),
        )

    loss_and_grad = nn.value_and_grad(module, loss_fn)
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)

    def address_loss_fn(
        current, hidden, teacher, candidate_mask, example_weight=1.0,
    ):
        address_loss, address_parts = current.hierarchical_loss(
            hidden[None].astype(mx.float32),
            teacher[None].astype(mx.float32),
            args.window + args.sink_tokens + 1,
            args.window,
            args.sink_tokens,
            mass_cover=0.95,
            mass_gamma=0.5,
            max_positives=56,
            positive_weight=1.0,
            hard_negative_weight=1.0,
            hard_negative_temperature=1.0,
            rerank_weight=1.0,
            rerank_temperature=1.0,
            balance_weight=10.0,
            decorrelation_weight=0.0,
            address_entropy_weight=0.0,
            leaf_overflow_weight=args.address_overflow_weight,
            leaf_storage_capacity=args.storage_capacity,
            secondary_probes=args.secondary_probes,
            retrieval_topk=32,
            deployed_candidate_mask=candidate_mask[None],
            exact_boundary_weight=args.address_boundary_weight,
            exact_boundary_negative_weight=(
                args.address_boundary_negative_weight
            ),
            exact_boundary_query_stride=args.address_boundary_query_stride,
            reranker="full-hamming",
        )
        return address_loss * example_weight, address_parts

    address_loss_and_grad = (
        nn.value_and_grad(address_router, address_loss_fn)
        if address_router is not None else None
    )
    address_optimizer = (
        optim.AdamW(learning_rate=args.address_lr, weight_decay=0.01)
        if address_router is not None else None
    )

    def mean_loss(rows, return_by_corpus=False):
        values = []
        values_by_corpus = {}
        for packed_row in rows:
            corpus, hidden, target, queries, teacher, mask = (
                materialize_example(packed_row)
            )
            loss, parts = loss_fn(module, hidden, target, queries, teacher, mask)
            mx.eval(loss, parts)
            row = (float(loss), *(float(value) for value in parts))
            values.append(row)
            values_by_corpus.setdefault(corpus, []).append(row)

        def summarize(group):
            array = np.asarray(group, dtype=np.float64)
            return dict(zip(
                ("loss", "task", "cross_entropy", "pairwise", "balance", "confidence"),
                map(float, np.mean(array, axis=0)),
            ))

        overall = summarize(values)
        if not return_by_corpus:
            return overall
        return overall, {
            corpus: summarize(corpus_values)
            for corpus, corpus_values in values_by_corpus.items()
        }

    before, before_by_corpus = mean_loss(
        eval_examples, return_by_corpus=True
    )

    def mean_address_loss(rows, return_by_corpus=False):
        if address_router is None:
            return (None, None) if return_by_corpus else None
        values = []
        values_by_corpus = {}
        for packed_row in rows:
            corpus, hidden, _, _, teacher, mask = materialize_example(packed_row)
            loss, parts = address_loss_fn(
                address_router, hidden, teacher, mask
            )
            mx.eval(loss, parts)
            row = (float(loss), *(float(value) for value in parts))
            values.append(row)
            values_by_corpus.setdefault(corpus, []).append(row)

        names = (
            "loss", "mass_positive", "hard_negative", "rerank_cross_entropy",
            "balance", "confidence", "selected_positive_count",
            "selected_positive_mass", "decorrelation", "address_entropy",
            "leaf_overflow", "candidate_set_mass", "exact_boundary_positive",
            "exact_boundary_negative",
        )

        def summarize(group):
            return dict(zip(
                names,
                map(float, np.mean(np.asarray(group, dtype=np.float64), axis=0)),
            ))

        overall = summarize(values)
        if not return_by_corpus:
            return overall
        return overall, {
            corpus: summarize(corpus_values)
            for corpus, corpus_values in values_by_corpus.items()
        }

    address_before, address_before_by_corpus = mean_address_loss(
        eval_examples, return_by_corpus=True
    )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_examples))
    group_loss_ema = {
        corpus: metrics["loss"] for corpus, metrics in before_by_corpus.items()
    }

    def group_weights():
        if args.group_dro_beta == 0.0:
            return {corpus: 1.0 for corpus in group_loss_ema}
        corpus_order = sorted(group_loss_ema)
        logits = args.group_dro_beta * np.asarray(
            [group_loss_ema[corpus] for corpus in corpus_order], dtype=np.float64
        )
        probability = np.exp(logits - np.max(logits))
        probability /= np.sum(probability)
        return {
            corpus: float(value * len(corpus_order))
            for corpus, value in zip(corpus_order, probability)
        }

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(order) == 0 and step > 1:
            order = rng.permutation(len(train_examples))
        corpus, hidden, target, queries, teacher, mask = materialize_example(
            train_examples[order[(step - 1) % len(order)]]
        )
        example_weight = group_weights()[corpus]
        (loss, parts), gradients = loss_and_grad(
            module, hidden, target, queries, teacher, mask, example_weight
        )
        gradients, gradient_norm = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(module, gradients)
        mx.eval(module.parameters(), optimizer.state, loss, parts, gradient_norm)
        address_loss = None
        address_parts = None
        address_gradient_norm = None
        if address_router is not None:
            mx.clear_cache()
            (address_loss, address_parts), address_gradients = (
                address_loss_and_grad(
                    address_router, hidden, teacher, mask, example_weight
                )
            )
            address_gradients, address_gradient_norm = optim.clip_grad_norm(
                address_gradients, 1.0
            )
            address_optimizer.update(address_router, address_gradients)
            mx.eval(
                address_router.parameters(), address_optimizer.state,
                address_loss, address_parts, address_gradient_norm,
            )
        raw_loss = float(loss) / max(example_weight, 1e-12)
        group_loss_ema[corpus] = (
            args.group_dro_ema * group_loss_ema[corpus]
            + (1.0 - args.group_dro_ema) * raw_loss
        )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            payload = {
                "step": step,
                "loss": float(loss),
                "task": float(parts[0]),
                "cross_entropy": float(parts[1]),
                "pairwise": float(parts[2]),
                "balance": float(parts[3]),
                "confidence": float(parts[4]),
                "gradient_norm": float(gradient_norm),
                "steps_per_second": step / (time.perf_counter() - started),
                "peak_memory_mb": mx.get_peak_memory() / 2**20,
                "corpus": corpus,
                "group_weight": example_weight,
                "group_loss_ema": dict(group_loss_ema),
            }
            if address_router is not None:
                payload.update({
                    "address_loss": float(address_loss),
                    "address_gradient_norm": float(address_gradient_norm),
                    "address_leaf_overflow": float(address_parts[9]),
                    "address_exact_boundary_positive": float(address_parts[11]),
                    "address_exact_boundary_negative": float(address_parts[12]),
                })
            print(json.dumps(payload), flush=True)

    after, after_by_corpus = mean_loss(
        eval_examples, return_by_corpus=True
    )
    address_after, address_after_by_corpus = mean_address_loss(
        eval_examples, return_by_corpus=True
    )
    output_weights = dict(weights)
    if address_router is not None:
        if (
            args.joint_address_residual_secondary
            or args.joint_address_primary_conditioned_secondary
        ):
            for name in (
                "secondary_query_assignment_weight",
                "secondary_query_assignment_bias",
                "secondary_key_assignment_weight",
                "secondary_key_assignment_bias",
            ):
                output_weights[name] = getattr(address_router, name)
            if args.joint_address_primary_conditioned_secondary:
                for name in (
                    "secondary_query_primary_bias",
                    "secondary_key_primary_bias",
                ):
                    output_weights[name] = getattr(address_router, name)
        elif args.joint_address_categorical:
            for name in (
                "address_query_assignment_weight",
                "address_query_assignment_bias",
                "address_key_assignment_weight",
                "address_key_assignment_bias",
            ):
                output_weights[name] = getattr(address_router, name)
        else:
            output_weights["query_projection"] = address_router.query_projection
            output_weights["key_projection"] = address_router.key_projection
        if args.joint_address_thresholds:
            output_weights["address_query_bias"] = (
                address_router.address_query_bias
            )
            output_weights["address_key_bias"] = address_router.address_key_bias
    # Query fingerprints still participate in the shared sparse lookup path,
    # including for VQ rerankers whose final score ignores Hamming distance.
    # Save the width-matched projection for every code type.
    output_weights["rerank_query_projection"] = mx.array(
        rerank_query_projection
    )
    if args.code_type == "vq":
        output_weights["joint_vq_assignment_weight"] = module.assignment_weight
        output_weights["joint_vq_assignment_bias"] = module.assignment_bias
        output_weights["joint_vq_attention_decoder"] = module.codebooks
    else:
        # The evaluator uses the rerank query projection to construct query
        # fingerprints before applying the learned attention decoder.  Preserve
        # the expanded projection when training wider-than-32-bit checkpoints;
        # otherwise the saved key and query fingerprints have different widths.
        output_weights["rerank_key_projection"] = module.key_projection
        output_weights["joint_binary_attention_decoder"] = module.decoder
        output_weights[
            "joint_binary_attention_decoder_hidden_weight"
        ] = module.decoder_hidden_weight
        output_weights[
            "joint_binary_attention_decoder_hidden_bias"
        ] = module.decoder_hidden_bias
        output_weights[
            "joint_binary_attention_decoder_output_weight"
        ] = module.decoder_output_weight
        output_weights[
            "joint_binary_attention_decoder_output_bias"
        ] = module.decoder_output_bias
    output_weights[
        "joint_binary_attention_head_bias_weight"
    ] = module.head_bias_weight
    output_weights["joint_binary_attention_head_bias"] = module.head_bias
    output_weights["attention_query_weight"] = (
        module.query_weight if args.code_type == "binary"
        else attention.q_proj.weight
    )
    output_weights["attention_query_norm"] = (
        module.query_norm if args.code_type == "binary"
        else attention.q_layernorm.weight
    )
    if args.query_only or args.train_query:
        output_weights["attention_query_trained"] = mx.array([1], mx.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), output_weights)
    metadata = vars(args) | {
        "router": str(args.router),
        "router_sha256": sha256(args.router),
        "before": before,
        "before_by_corpus": before_by_corpus,
        "after": after,
        "after_by_corpus": after_by_corpus,
        "address_before": address_before,
        "address_before_by_corpus": address_before_by_corpus,
        "address_after": address_after,
        "address_after_by_corpus": address_after_by_corpus,
        "final_group_loss_ema": group_loss_ema,
        "kv_heads": attention.n_kv_heads,
        "query_heads": attention.n_heads,
        "head_dim": attention.head_dim,
        "rope_base": attention.rope.base,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
