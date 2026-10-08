# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Interleaving model of A2/A3 progress and completion mailboxes.

Individual peer stores/polls, strided AIV ownership, local barriers and changing
chunk boundaries are modeled. Hardware pipeline ordering needs device tests.
"""

import random
import unittest


class Protocol:
    def __init__(self, ranks, workers, calls, seed):
        self.ranks, self.workers, self.calls = ranks, workers, calls
        self.ready = [[0] * ranks for _ in range(ranks)]
        self.done = [[0] * ranks for _ in range(ranks)]
        self.counts = [-1] * ranks
        self.payload = {}
        self.arrived = [0] * ranks
        self.phase = [0] * ranks
        self.rng = random.Random(seed)
        self.seed = seed

    def barrier(self, rank):
        phase = self.phase[rank]
        self.arrived[rank] += 1
        if self.arrived[rank] == self.workers:
            self.arrived[rank] = 0
            self.phase[rank] += 1
        yield lambda: self.phase[rank] > phase

    @staticmethod
    def active(generation, source, destination, expert):
        # Entirely empty invocations, asymmetric zeros, self and remote routes.
        return (
            generation % 5 != 0
            and (source + destination + expert + generation) % 3 != 0
        )

    def worker(self, rank, worker):
        peers = range(worker, self.ranks, self.workers)
        for generation in range(self.calls):
            experts = (1, 7, 2, 5)[generation % 4]
            if worker == 0:
                self.counts[rank] = generation
                yield None
            yield from self.barrier(rank)
            for peer in peers:
                self.ready[peer][rank] = 1
                yield None
            yield from self.barrier(rank)
            for peer in peers:
                yield lambda peer=peer: self.ready[rank][peer] >= 1
                assert self.counts[peer] == generation, "count table reused too early"
            yield from self.barrier(rank)
            start = 0
            while start < experts:
                # Rank-local chunk endpoints; not a shared chunk counter.
                end = min(experts, start + 1 + (rank + generation + start) % 4)
                for expert in range(start + worker, end, self.workers):
                    self.payload[rank, expert] = generation
                    yield None
                yield from self.barrier(rank)
                for peer in peers:
                    self.ready[peer][rank] = end + 1
                    yield None
                yield from self.barrier(rank)
                for peer in peers:
                    needed = [
                        e
                        for e in range(start, end)
                        if self.active(generation, peer, rank, e)
                    ]
                    if needed:
                        expected = needed[-1] + 2
                        yield (
                            lambda peer=peer, expected=expected: self.ready[rank][peer]
                            >= expected
                        )
                yield from self.barrier(rank)
                for peer in peers:
                    for expert in range(start, end):
                        if self.active(generation, peer, rank, expert):
                            assert self.payload[peer, expert] == generation, (
                                "stale/overwritten payload"
                            )
                            yield None
                yield from self.barrier(rank)
                start = end
            # Final pull barrier also orders all publications and payload reads.
            for peer in peers:
                self.done[peer][rank] = 1
                yield None
            yield from self.barrier(rank)
            for peer in peers:
                yield lambda peer=peer: self.done[rank][peer] == 1
                self.ready[peer][rank] = 0
                yield None
            yield from self.barrier(rank)
            for peer in peers:
                yield lambda peer=peer: self.ready[rank][peer] == 0
                self.done[peer][rank] = 0
                yield None
            yield from self.barrier(rank)
            for peer in peers:
                yield lambda peer=peer: self.done[rank][peer] == 0
            yield from self.barrier(rank)

    def run(self):
        workers = [
            self.worker(rank, worker)
            for rank in range(self.ranks)
            for worker in range(self.workers)
        ]
        pending = [None] * len(workers)
        live = set(range(len(workers)))
        while live:
            enabled = [idx for idx in live if pending[idx] is None or pending[idx]()]
            assert enabled, (
                f"deadlock: ranks={self.ranks} workers={self.workers} seed={self.seed}"
            )
            fast = self.seed % self.ranks
            idx = self.rng.choices(
                enabled,
                weights=[100 if x // self.workers == fast else 1 for x in enabled],
            )[0]
            try:
                pending[idx] = next(workers[idx])
            except StopIteration:
                live.remove(idx)
        assert not any(map(any, self.ready))
        assert not any(map(any, self.done))


class ReturnToZeroTest(unittest.TestCase):
    def test_skewed_calls_chunks_empty_routes_and_worker_ownership(self):
        for ranks, workers in ((2, 1), (3, 2), (4, 3), (8, 3), (2, 4)):
            for seed in range(40):
                with self.subTest(ranks=ranks, workers=workers, seed=seed):
                    Protocol(ranks, workers, calls=20, seed=seed).run()

    def test_progress_may_skip_requested_value(self):
        # A producer with chunk [0,4) must satisfy consumers waiting for
        # count-ready (1) and expert 1 (3), even if they miss earlier values.
        observed = 5
        self.assertGreaterEqual(observed, 1)
        self.assertGreaterEqual(observed, 3)
        self.assertNotEqual(observed, 3)


if __name__ == "__main__":
    unittest.main()
