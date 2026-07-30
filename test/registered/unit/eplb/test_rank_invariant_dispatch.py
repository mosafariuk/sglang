"""Unit tests for the rank-invariant (`round_robin`) EPLB dispatch.

Without an a2a backend every EP rank runs the MoE over the *same* tokens and only
owns a slice of the physical experts, so the partial outputs are summed. A
logical->physical choice that differs per rank therefore makes a replicated
logical expert run on several ranks and be counted several times: a silent
accuracy loss (DeepSeek-V2-Lite GSM8K 0.665 -> 0.348 with
`--ep-size 2 --ep-num-redundant-experts 32 --enable-eplb`).

The `round_robin` algorithm exists to keep that choice identical on every rank.
Two things have to hold for that, and both are covered below: the candidate map
must not be collapsed to the rank-local replica, and the replica pick must be a
pure function of the token row.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.eplb.expert_location import _compute_logical_to_all_physical_map
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    _topk_ids_logical_to_physical_round_robin,
)
from sglang.test.test_utils import CustomTestCase

EP_SIZE = 2
NUM_LOGICAL = 4
# 6 physical slots over 2 ranks: rank 0 owns 0-2, rank 1 owns 3-5.
NUM_PHYSICAL = 6
NUM_LAYERS = 1


def _make_server_args(ep_dispatch_algorithm):
    return types.SimpleNamespace(
        ep_size=EP_SIZE,
        nnodes=1,
        ep_join_mode=None,
        ep_dispatch_algorithm=ep_dispatch_algorithm,
    )


def _physical_to_logical_map():
    """Trivial placement with 2 redundant slots, as `init_trivial` builds it.

    physical: 0 1 2 3 4 5
    logical:  0 1 2 3 0 1   -> logical 0 lives on both ranks (0 and 4),
                               logical 1 too (1 and 5).
    """
    return torch.tensor(
        [[0, 1, 2, 3, 0, 1]] * NUM_LAYERS,
        dtype=torch.int64,
    )


def _logical_to_all_physical(ep_dispatch_algorithm, moe_ep_rank):
    return _compute_logical_to_all_physical_map(
        server_args=_make_server_args(ep_dispatch_algorithm),
        physical_to_logical_map=_physical_to_logical_map(),
        num_logical_experts=NUM_LOGICAL,
        ep_size=EP_SIZE,
        moe_ep_rank=moe_ep_rank,
    )


def _make_info(logical_to_all_physical):
    partial = logical_to_all_physical[0]
    return ExpertLocationDispatchInfo(
        ep_dispatch_algorithm="round_robin",
        partial_logical_to_rank_dispatch_physical_map=None,
        partial_logical_to_all_physical_map=partial,
        partial_logical_to_all_physical_map_num_valid=torch.count_nonzero(
            partial != -1, dim=-1
        ),
        num_physical_experts=NUM_PHYSICAL,
    )


class TestRankInvariantCandidateMap(CustomTestCase):
    """`_compute_logical_to_all_physical_map` must stay rank-invariant for round_robin."""

    def test_round_robin_map_is_identical_across_ranks(self):
        """Regression: the rank-local collapse used to run for every algorithm, so
        rank 0 saw logical 0 -> [0] while rank 1 saw logical 0 -> [4]. round_robin
        then picked a different physical expert per rank and the MoE all-reduce
        double-counted logical 0."""
        maps = [
            _logical_to_all_physical("round_robin", moe_ep_rank=rank)
            for rank in range(EP_SIZE)
        ]
        self.assertTrue(
            torch.equal(maps[0], maps[1]),
            f"round_robin candidate map differs across ranks:\n{maps[0]}\n{maps[1]}",
        )

    def test_round_robin_map_keeps_every_replica(self):
        """Both replicas must survive, otherwise redundant experts are dead weight
        and round_robin cannot spread a hot expert's tokens."""
        candidates = _logical_to_all_physical("round_robin", moe_ep_rank=0)[0]
        self.assertEqual(candidates[0].tolist(), [0, 4])
        self.assertEqual(candidates[1].tolist(), [1, 5])

    def test_static_map_is_rank_dependent(self):
        """The contrast case that motivates round_robin: `static` deliberately
        collapses to the rank-local replica, which is correct only when an a2a
        backend routes each token to a single rank."""
        maps = [
            _logical_to_all_physical("static", moe_ep_rank=rank)
            for rank in range(EP_SIZE)
        ]
        self.assertEqual(maps[0][0, 0].tolist(), [0])
        self.assertEqual(maps[1][0, 0].tolist(), [4])


class TestRoundRobinDispatch(CustomTestCase):
    """Tests for `_topk_ids_logical_to_physical_round_robin`."""

    def setUp(self):
        self.info = _make_info(_logical_to_all_physical("round_robin", moe_ep_rank=0))

    def test_is_deterministic(self):
        """The replica pick must not use randomness: two ranks running the same
        batch have to agree, and `torch.randint` (as `dynamic` uses) would not."""
        topk_ids = torch.randint(0, NUM_LOGICAL, (16, 3), dtype=torch.int32)
        first = _topk_ids_logical_to_physical_round_robin(topk_ids, self.info)
        second = _topk_ids_logical_to_physical_round_robin(topk_ids, self.info)
        self.assertTrue(torch.equal(first, second))

    def test_alternates_replicas_by_token_row(self):
        """Row parity selects the replica, so a replicated logical expert splits
        its tokens evenly over both ranks -- the load split EPLB assumes."""
        topk_ids = torch.zeros((4, 1), dtype=torch.int32)  # every token -> logical 0
        result = _topk_ids_logical_to_physical_round_robin(topk_ids, self.info)
        self.assertEqual(result.flatten().tolist(), [0, 4, 0, 4])

    def test_single_replica_experts_are_unchanged(self):
        """Logical experts with one candidate ignore the row index."""
        topk_ids = torch.full((4, 1), 2, dtype=torch.int32)
        result = _topk_ids_logical_to_physical_round_robin(topk_ids, self.info)
        self.assertEqual(result.flatten().tolist(), [2, 2, 2, 2])

    def test_preserves_dtype_and_shape(self):
        """int32 topk_ids must stay int32 -- backends read them via data_ptr()."""
        topk_ids = torch.randint(0, NUM_LOGICAL, (8, 3), dtype=torch.int32)
        result = _topk_ids_logical_to_physical_round_robin(topk_ids, self.info)
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.shape, (8, 3))


if __name__ == "__main__":
    unittest.main()
