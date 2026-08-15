#!/usr/bin/env python3
"""Idealized collision-capacity analysis for bounded-tail hash retrieval.

This is an analytical model, not a learned-router benchmark. It assumes balanced,
independent tables and conditions on a query/key address agreement probability.
Memory use is constant in context length.
"""

import argparse
import math


def binomial_tail_survival(later_keys, bucket_probability, members):
    """Return P[Binomial(later_keys, bucket_probability) < members]."""
    if later_keys < 0:
        raise ValueError("later_keys must be non-negative")
    if members < 1:
        raise ValueError("members must be positive")
    if not 0.0 <= bucket_probability <= 1.0:
        raise ValueError("bucket_probability must be in [0, 1]")
    if members > later_keys:
        return 1.0
    if bucket_probability == 0.0:
        return 1.0
    if bucket_probability == 1.0:
        return float(later_keys < members)

    upper = min(members - 1, later_keys)
    log_q = math.log(bucket_probability)
    log_one_minus_q = math.log1p(-bucket_probability)
    log_pmf = later_keys * log_one_minus_q
    log_terms = [log_pmf]
    for collisions in range(upper):
        log_pmf += (
            math.log(later_keys - collisions)
            - math.log(collisions + 1)
            + log_q
            - log_one_minus_q
        )
        log_terms.append(log_pmf)
    maximum = max(log_terms)
    if maximum < -745.0:
        return 0.0
    return min(1.0, math.exp(maximum) * sum(math.exp(x - maximum) for x in log_terms))


def retrieval_probability(later_keys, bits, tables, members, agreement=1.0):
    """Probability that at least one independent table retrieves the target."""
    if bits < 1:
        raise ValueError("bits must be positive")
    if tables < 1:
        raise ValueError("tables must be positive")
    if not 0.0 <= agreement <= 1.0:
        raise ValueError("agreement must be in [0, 1]")
    survival = binomial_tail_survival(later_keys, 2.0**-bits, members)
    table_success = agreement * survival
    return -math.expm1(tables * math.log1p(-table_success)) if table_success < 1.0 else 1.0


def minimum_bits(later_keys, tables, members, target_recall, agreement=1.0, max_bits=63):
    """Find the smallest address width reaching the idealized recall target."""
    if not 0.0 < target_recall < 1.0:
        raise ValueError("target_recall must be strictly between zero and one")
    for bits in range(1, max_bits + 1):
        if retrieval_probability(later_keys, bits, tables, members, agreement) >= target_recall:
            return bits
    return None


def parse_lengths(value):
    lengths = [int(item) for item in value.split(",")]
    if not lengths or any(length < 2 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers >= 2")
    return lengths


def self_test():
    assert binomial_tail_survival(0, 0.5, 1) == 1.0
    assert math.isclose(binomial_tail_survival(1, 0.5, 1), 0.5)
    assert math.isclose(binomial_tail_survival(2, 0.5, 2), 0.75)
    one = retrieval_probability(1, 1, 1, 1)
    two = retrieval_probability(1, 1, 2, 1)
    assert math.isclose(one, 0.5)
    assert math.isclose(two, 0.75)
    assert minimum_bits(1_000, 4, 4, 0.95) is not None
    print("PASS idealized collision-capacity analysis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("1024,4096,16384,65536,1048576"))
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--tables", type=int, default=4)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--agreement", type=float, default=1.0)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.0,
        help="target position divided by length; 0 is the oldest possible target",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 0.0 <= args.target_fraction < 1.0:
        parser.error("--target-fraction must be in [0, 1)")

    print("length\tlater_keys\texpected_bucket_tail\trecall\tminimum_bits")
    for length in args.lengths:
        target_position = int(length * args.target_fraction)
        later_keys = length - target_position - 1
        occupancy = later_keys * (2.0**-args.bits)
        recall = retrieval_probability(
            later_keys, args.bits, args.tables, args.members, args.agreement
        )
        required = minimum_bits(
            later_keys,
            args.tables,
            args.members,
            args.target_recall,
            args.agreement,
        )
        print(
            f"{length}\t{later_keys}\t{occupancy:.6f}\t{recall:.8f}\t"
            f"{required if required is not None else '>' + str(63)}"
        )


if __name__ == "__main__":
    main()
