# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
NPU precision test for minimax_sparse_attention_split_kv.

Style follows kdb/msa sparse_attention_score_prefill NPU tests:
CPU KV-centric two-phase golden vs NPU kernel. Covers:
  * innerPrecise 0 (fp32 softmax) / 4 (bf16 softmax) / 1 (bf16 O_partial)
  * TND paged, TND contiguous, BNSD / BSND contiguous
  * softmax LSE output
  * q_len / kv_len = 0 dummy requests
plus production-config sweeps (seed / q-boundary / kv-blockcount / uneven batch).
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = int(os.environ.get("ASCEND_DEVICE_ID", os.environ.get("DEVICE_ID", "0")))
torch_npu.npu.set_device(int(DEVICE_ID))

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TORCH_EXT = _REPO_ROOT / "torch_extension"
_GOLDEN_TEST = (
    Path(__file__).resolve().parent / "test_minimax_sparse_attention_split_kv_golden.py"
)
sys.path.insert(0, str(_TORCH_EXT))
sys.path.insert(0, str(_GOLDEN_TEST.parent))

_golden_spec = importlib.util.spec_from_file_location(
    "msa_split_kv_golden", _GOLDEN_TEST
)
prefill_golden = importlib.util.module_from_spec(_golden_spec)
_golden_spec.loader.exec_module(prefill_golden)

make_case = prefill_golden.make_case
cpu_golden_prefill_bf16 = prefill_golden.cpu_golden_prefill_bf16
prepare_npu_case = prefill_golden.prepare_npu_case
layout_golden_attn = prefill_golden.layout_golden_attn
layout_golden_lse = prefill_golden.layout_golden_lse


def _load_op():
    import cann_ops_transformer.ops.minimax_sparse_attention_split_kv  # noqa: F401


_load_op()

_PREC = {0: 1e-2, 1: 3e-2, 4: 2e-2}


def _has_valid_tokens(q_seqlens, kv_seqlens):
    return any(q > 0 and kv > 0 for q, kv in zip(q_seqlens, kv_seqlens))


class TestMinimaxSparseAttentionSplitKvNPU(TestCase):
    """NPU vs CPU KV-centric golden for minimax_sparse_attention_split_kv."""

    def _max_abs_diff(self, a, b):
        return (a.float() - b.float()).abs().max().item()

    def _run_npu_test(
        self,
        batch,
        q_seqlens,
        kv_seqlens,
        q_heads,
        kv_heads,
        head_dim=128,
        block_size=128,
        top_k=4,
        seed=42,
        prec=None,
        inner_precise=4,
        layout="TND",
        paged=True,
        softmax_lse_flag=False,
        tag="",
    ):
        if layout != "TND":
            paged = False
        if layout == "TND" and sum(q_seqlens) == 0:
            self.skipTest("TND packed query T=0")
        if layout == "TND" and (not paged) and sum(kv_seqlens) == 0:
            self.skipTest("TND contiguous packed KV T=0")
        if prec is None:
            prec = _PREC.get(inner_precise, 2e-2)

        torch.npu.synchronize()
        torch.npu.empty_cache()

        data = make_case(
            batch,
            q_seqlens,
            kv_seqlens,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            top_k,
            seed,
        )

        golden_attn, golden_lse = cpu_golden_prefill_bf16(
            data["query"],
            data["key"],
            data["value"],
            data["k2q_row_ptr"],
            data["k2q_q_indices"],
            data["k2q_slot_indices"],
            data["block_table"],
            data["actual_seq_lengths"],
            data["actual_seq_lengths_kv"],
            data["block_size"],
            data["top_k"],
            data["scale_value"],
            inner_precise,
        )

        npu_data = prepare_npu_case(data, layout=layout, paged=paged)
        s_q = npu_data["s_q"]
        golden_attn_cmp = layout_golden_attn(golden_attn, q_seqlens, layout, s_q)
        golden_lse_cmp = layout_golden_lse(golden_lse, q_seqlens, layout, s_q)

        bt = npu_data["block_table"]
        npu_attn, npu_lse = (
            torch.ops.cann_ops_transformer.minimax_sparse_attention_split_kv(
                npu_data["query"].npu(),
                npu_data["key"].npu(),
                npu_data["value"].npu(),
                None if bt is None else bt.npu(),
                npu_data["k2q_row_ptr"].npu(),
                npu_data["k2q_q_indices"].npu(),
                npu_data["k2q_slot_indices"].npu(),
                npu_data["actual_seq_lengths"].npu(),
                npu_data["actual_seq_lengths_kv"].npu(),
                kv_heads,
                data["scale_value"],
                block_size,
                top_k,
                inner_precise,
                softmax_lse_flag,
                layout,
            )
        )

        npu_attn_cpu = npu_attn.cpu()
        max_diff = self._max_abs_diff(golden_attn_cmp, npu_attn_cpu)
        mean_diff = (golden_attn_cmp.float() - npu_attn_cpu.float()).abs().mean().item()
        prefix = f"[npu {tag}]" if tag else "[npu]"
        print(
            f"{prefix} layout={layout} paged={paged} ip={inner_precise} lse={int(softmax_lse_flag)} "
            f"batch={batch} q={q_seqlens} kv={kv_seqlens} top_k={top_k} "
            f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}"
        )

        self.assertFalse(
            torch.any(torch.isnan(npu_attn_cpu.float())), "NPU output contains NaN"
        )
        self.assertFalse(
            torch.any(torch.isinf(npu_attn_cpu.float())), "NPU output contains Inf"
        )
        if _has_valid_tokens(q_seqlens, kv_seqlens):
            self.assertFalse(
                torch.all(npu_attn_cpu.float() == 0),
                "NPU output is all zeros (suspect Phase2 combine / padding skip)",
            )
        self.assertRtolEqual(
            golden_attn_cmp.float().numpy(), npu_attn_cpu.float().numpy(), prec=prec
        )

        if softmax_lse_flag:
            npu_lse_cpu = npu_lse.cpu()
            lse_prec = max(prec, 2e-2)
            lse_max = self._max_abs_diff(golden_lse_cmp, npu_lse_cpu)
            print(f"  lse max_diff={lse_max:.6f}")
            self.assertFalse(torch.any(torch.isnan(npu_lse_cpu)))
            self.assertFalse(torch.any(torch.isinf(npu_lse_cpu)))
            self.assertRtolEqual(
                golden_lse_cmp.float().numpy(),
                npu_lse_cpu.float().numpy(),
                prec=lse_prec,
            )

    # ---------------------------------------------------------------------------
    # Basic TND paged, innerPrecise 0 / 4 (kdb-style shapes)
    # ---------------------------------------------------------------------------

    def test_bf16_single_batch_small(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[4],
                    kv_seqlens=[512],
                    q_heads=8,
                    kv_heads=2,
                    top_k=3,
                    inner_precise=ip,
                    tag=f"small/ip{ip}",
                )

    def test_bf16_single_batch_medium(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[32],
                    kv_seqlens=[2048],
                    q_heads=16,
                    kv_heads=4,
                    top_k=5,
                    seed=123,
                    inner_precise=ip,
                    tag=f"medium/ip{ip}",
                )

    def test_bf16_multi_batch(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=3,
                    q_seqlens=[8, 16, 4],
                    kv_seqlens=[1024, 2048, 512],
                    q_heads=8,
                    kv_heads=2,
                    top_k=4,
                    seed=456,
                    inner_precise=ip,
                    tag=f"multi/ip{ip}",
                )

    def test_bf16_gqa_large_group(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[16],
                    kv_seqlens=[1024],
                    q_heads=32,
                    kv_heads=4,
                    top_k=6,
                    seed=789,
                    inner_precise=ip,
                    tag=f"gqa/ip{ip}",
                )

    def test_bf16_partial_last_block(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[8],
                    kv_seqlens=[300],
                    q_heads=8,
                    kv_heads=2,
                    top_k=3,
                    seed=101,
                    inner_precise=ip,
                    tag=f"partial/ip{ip}",
                )

    def test_bf16_top_k_1(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[16],
                    kv_seqlens=[1024],
                    q_heads=8,
                    kv_heads=2,
                    top_k=1,
                    seed=202,
                    inner_precise=ip,
                    tag=f"topk1/ip{ip}",
                )

    def test_bf16_long_prefill(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[128],
                    kv_seqlens=[4096],
                    q_heads=8,
                    kv_heads=2,
                    top_k=8,
                    seed=303,
                    inner_precise=ip,
                    tag=f"long/ip{ip}",
                )

    def test_bf16_minimal_debug(self):
        for ip in (0, 4, 1):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[1],
                    kv_seqlens=[128],
                    q_heads=1,
                    kv_heads=1,
                    top_k=1,
                    seed=0,
                    inner_precise=ip,
                    tag=f"min1/ip{ip}",
                )

    def test_bf16_minimal_debug_2(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[2],
                    kv_seqlens=[128],
                    q_heads=2,
                    kv_heads=2,
                    top_k=1,
                    seed=1,
                    inner_precise=ip,
                    tag=f"min2/ip{ip}",
                )

    def test_bf16_minimal_debug_gqa(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[1],
                    kv_seqlens=[128],
                    q_heads=2,
                    kv_heads=1,
                    top_k=1,
                    seed=2,
                    inner_precise=ip,
                    tag=f"min_gqa/ip{ip}",
                )

    def test_bf16_prod_heads_small(self):
        for ip in (0, 4, 1):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=1,
                    q_seqlens=[16],
                    kv_seqlens=[16],
                    q_heads=64,
                    kv_heads=4,
                    top_k=16,
                    seed=42,
                    inner_precise=ip,
                    tag=f"prod_small/ip{ip}",
                )

    # ---------------------------------------------------------------------------
    # Layout × innerPrecise × LSE × paged/contiguous (moderate shapes)
    # ---------------------------------------------------------------------------

    def test_layout_inner_precise_lse_matrix(self):
        shapes = [
            (1, [8], [256], 8, 2, 2, 11, "s8_k256"),
            (2, [4, 8], [128, 256], 8, 2, 3, 22, "2b"),
            (1, [16], [300], 16, 2, 4, 33, "partial"),
        ]
        modes = [
            ("TND", True),
            ("TND", False),
            ("BNSD", False),
            ("BSND", False),
        ]
        for batch, q_sl, kv_sl, qh, kvh, top_k, seed, stag in shapes:
            for layout, paged in modes:
                for ip in (0, 4, 1):
                    for lse in (False, True):
                        tag = (
                            f"matrix/{stag}/{layout}/p{int(paged)}/ip{ip}/lse{int(lse)}"
                        )
                        with self.subTest(tag=tag):
                            self._run_npu_test(
                                batch=batch,
                                q_seqlens=q_sl,
                                kv_seqlens=kv_sl,
                                q_heads=qh,
                                kv_heads=kvh,
                                top_k=top_k,
                                seed=seed,
                                inner_precise=ip,
                                layout=layout,
                                paged=paged,
                                softmax_lse_flag=lse,
                                tag=tag,
                            )

    def test_tnd_contiguous_basic(self):
        for ip in (0, 4):
            with self.subTest(inner_precise=ip):
                self._run_npu_test(
                    batch=2,
                    q_seqlens=[8, 16],
                    kv_seqlens=[256, 512],
                    q_heads=16,
                    kv_heads=2,
                    top_k=4,
                    seed=77,
                    inner_precise=ip,
                    layout="TND",
                    paged=False,
                    softmax_lse_flag=True,
                    tag=f"tnd_cont/ip{ip}",
                )

    def test_bnsd_bsnd_gqa(self):
        for layout in ("BNSD", "BSND"):
            for ip in (0, 4):
                with self.subTest(layout=layout, inner_precise=ip):
                    self._run_npu_test(
                        batch=1,
                        q_seqlens=[32],
                        kv_seqlens=[1024],
                        q_heads=32,
                        kv_heads=4,
                        top_k=6,
                        seed=88,
                        inner_precise=ip,
                        layout=layout,
                        paged=False,
                        softmax_lse_flag=True,
                        tag=f"{layout}/gqa/ip{ip}",
                    )

    # ---------------------------------------------------------------------------
    # q_len / kv_len = 0 padding
    # ---------------------------------------------------------------------------

    _PAD_CASES = [
        (2, [0, 8], [0, 256], "lead_dummy"),
        (2, [8, 0], [256, 0], "tail_dummy"),
        (3, [8, 0, 4], [256, 0, 128], "mid_dummy"),
        (4, [0, 8, 0, 4], [0, 256, 0, 128], "lead_mid"),
        (3, [0, 0, 8], [0, 0, 256], "two_lead"),
        (3, [8, 0, 0], [256, 0, 0], "two_tail"),
        (1, [8], [0], "kv0_only"),
        (2, [0, 16], [128, 512], "q0_with_kv"),
        (2, [16, 0], [512, 256], "q_with_kv0"),
        (3, [0, 1, 8], [0, 128, 256], "mix_q1"),
        (4, [0, 4, 0, 0], [0, 256, 0, 0], "one_valid"),
        (2, [0, 32], [0, 300], "lead_partial_kv"),
    ]

    def test_padding_tnd_paged(self):
        for batch, q_sl, kv_sl, stag in self._PAD_CASES:
            for ip in (0, 4):
                for lse in (False, True):
                    tag = f"pad_tnd/{stag}/ip{ip}/lse{int(lse)}"
                    with self.subTest(tag=tag):
                        self._run_npu_test(
                            batch=batch,
                            q_seqlens=q_sl,
                            kv_seqlens=kv_sl,
                            q_heads=8,
                            kv_heads=2,
                            top_k=3,
                            seed=200 + batch,
                            inner_precise=ip,
                            layout="TND",
                            paged=True,
                            softmax_lse_flag=lse,
                            tag=tag,
                        )

    def test_padding_tnd_contiguous(self):
        for batch, q_sl, kv_sl, stag in self._PAD_CASES:
            for ip in (0, 4):
                tag = f"pad_tnd_c/{stag}/ip{ip}"
                with self.subTest(tag=tag):
                    self._run_npu_test(
                        batch=batch,
                        q_seqlens=q_sl,
                        kv_seqlens=kv_sl,
                        q_heads=8,
                        kv_heads=2,
                        top_k=2,
                        seed=300 + batch,
                        inner_precise=ip,
                        layout="TND",
                        paged=False,
                        softmax_lse_flag=True,
                        tag=tag,
                    )

    def test_padding_bnsd_bsnd(self):
        for layout in ("BNSD", "BSND"):
            for batch, q_sl, kv_sl, stag in self._PAD_CASES:
                for ip in (0, 4):
                    tag = f"pad_{layout}/{stag}/ip{ip}"
                    with self.subTest(tag=tag):
                        self._run_npu_test(
                            batch=batch,
                            q_seqlens=q_sl,
                            kv_seqlens=kv_sl,
                            q_heads=16,
                            kv_heads=2,
                            top_k=3,
                            seed=400 + batch,
                            inner_precise=ip,
                            layout=layout,
                            paged=False,
                            softmax_lse_flag=True,
                            tag=tag,
                        )

    def test_padding_prod_heads(self):
        """Dummy requests at production GQA (Hq=64, Hkv=4, topK=16)."""
        for layout, paged in (("TND", True), ("BNSD", False), ("BSND", False)):
            for ip in (0, 4):
                tag = f"pad_prod/{layout}/ip{ip}"
                with self.subTest(tag=tag):
                    self._run_npu_test(
                        batch=4,
                        q_seqlens=[0, 16, 0, 8],
                        kv_seqlens=[0, 2048, 0, 512],
                        q_heads=64,
                        kv_heads=4,
                        top_k=16,
                        seed=501,
                        inner_precise=ip,
                        layout=layout,
                        paged=paged,
                        softmax_lse_flag=True,
                        tag=tag,
                    )

    # ---------------------------------------------------------------------------
    # Production-config precision sweep. FIXED dims (do NOT vary):
    #   head_dim=128, block_size=128, top_k=16,
    #   q_heads=64, kv_heads=4, group_size=16 (batchGroupsMax=8).
    # innerPrecise 0 and 4 are both required (high-prec softmax vs default).
    # ---------------------------------------------------------------------------

    _HEAD_DIM = 128
    _BLOCK_SIZE = 128
    _TOP_K = 16
    _Q_HEADS = 64
    _KV_HEADS = 4

    def _run_prod_config(
        self,
        batch,
        q_seqlens,
        kv_seqlens,
        seed,
        prec=1e-2,
        tag="",
        inner_precise=4,
        layout="TND",
        paged=True,
        softmax_lse_flag=False,
    ):
        if tag:
            print(f"[prod:{tag}]")
        self._run_npu_test(
            batch,
            q_seqlens,
            kv_seqlens,
            self._Q_HEADS,
            self._KV_HEADS,
            head_dim=self._HEAD_DIM,
            block_size=self._BLOCK_SIZE,
            top_k=self._TOP_K,
            seed=seed,
            prec=prec,
            inner_precise=inner_precise,
            layout=layout,
            paged=paged,
            softmax_lse_flag=softmax_lse_flag,
            tag=tag,
        )

    def test_prod_topk16_seed_sweep(self):
        """Same representative shape, many seeds; both innerPrecise 0/4."""
        for ip in (0, 4):
            for seed in range(40):
                with self.subTest(inner_precise=ip, seed=seed):
                    self._run_prod_config(
                        batch=3,
                        q_seqlens=[1, 256, 8],
                        kv_seqlens=[512, 2048, 1024],
                        seed=1000 + seed,
                        prec=1.5e-2,
                        inner_precise=ip,
                        tag=f"seed_sweep/ip{ip}/{seed}",
                    )

    def test_prod_topk16_edge_shapes(self):
        edge_configs = [
            (1, [2048], [2048], 4101, 1.5e-2, "pure_prefill_q_eq_kv"),
            (1, [1], [2048], 4102, 1e-2, "q1_decode_like"),
            (1, [4], [128], 4103, 1e-2, "kv1block"),
            (1, [32], [512], 4104, 1e-2, "kv4block_partial_slots"),
            (1, [16], [2176], 4105, 1e-2, "partial_last_block"),
            (1, [128], [8192], 4106, 1.5e-2, "kv8192_long"),
            (8, [1] * 8, [256] * 8, 4107, 1e-2, "8batch_tiny"),
            (2, [1, 512], [256, 2048], 4108, 1.5e-2, "uneven_hot_batch"),
            (2, [2, 2], [128, 128], 4109, 1e-2, "2batch_kv1block"),
            (1, [256], [4096], 4110, 1e-2, "mid_prefill_kv4096"),
            (3, [8, 64, 4], [1024, 8192, 512], 4111, 1.5e-2, "3batch_mid_long"),
            (1, [1024], [8192], 4112, 1.5e-2, "extreme_q1024_kv8192"),
            (1, [64], [1920], 4113, 1e-2, "kv15block"),
            (1, [1], [64], 4114, 1e-2, "kv_subblock"),
            (1, [256], [4096], 4115, 1.5e-2, "long_prefill_kv4096"),
            (2, [1, 512], [2048, 4096], 4116, 1.5e-2, "mixed_decode_prefill"),
        ]
        for ip in (0, 4):
            for cfg in edge_configs:
                batch, q_sl, kv_sl, seed, prec, stag = cfg
                with self.subTest(inner_precise=ip, tag=stag):
                    self._run_prod_config(
                        batch=batch,
                        q_seqlens=q_sl,
                        kv_seqlens=kv_sl,
                        seed=seed,
                        prec=max(prec, _PREC[ip]),
                        inner_precise=ip,
                        softmax_lse_flag=(
                            stag
                            in (
                                "q1_decode_like",
                                "kv1block",
                                "uneven_hot_batch",
                                "8batch_tiny",
                            )
                        ),
                        tag=f"edge/{stag}/ip{ip}",
                    )

    def test_prod_topk16_q_boundary_sweep(self):
        q_vals = [
            1,
            2,
            3,
            7,
            8,
            9,
            15,
            16,
            17,
            31,
            33,
            64,
            127,
            128,
            129,
            255,
            256,
            257,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            2047,
            2048,
        ]
        for ip in (0, 4):
            for idx, q in enumerate(q_vals):
                with self.subTest(inner_precise=ip, q=q):
                    self._run_prod_config(
                        batch=1,
                        q_seqlens=[q],
                        kv_seqlens=[2048],
                        seed=8000 + idx,
                        prec=(1.5e-2 if q >= 1024 else max(1e-2, _PREC[ip])),
                        inner_precise=ip,
                        tag=f"q_boundary/q{q}/ip{ip}",
                    )

    def test_prod_topk16_kv_blockcount_sweep(self):
        block_counts = [1, 2, 4, 8, 15, 16, 17, 32, 64]
        for ip in (0, 4):
            for idx, n in enumerate(block_counts):
                kv = n * 128
                with self.subTest(inner_precise=ip, n_blocks=n):
                    self._run_prod_config(
                        batch=1,
                        q_seqlens=[64],
                        kv_seqlens=[kv],
                        seed=9000 + idx,
                        prec=(1.5e-2 if n >= 32 else max(1e-2, _PREC[ip])),
                        inner_precise=ip,
                        tag=f"kv_blocks/n{n}/ip{ip}",
                    )
            for idx, n in enumerate([1, 4, 16, 17, 32]):
                kv = n * 128 + 1
                with self.subTest(inner_precise=ip, n_blocks=n, tail=1):
                    self._run_prod_config(
                        batch=1,
                        q_seqlens=[64],
                        kv_seqlens=[kv],
                        seed=9100 + idx,
                        prec=max(1e-2, _PREC[ip]),
                        inner_precise=ip,
                        tag=f"kv_blocks/n{n}_tail1/ip{ip}",
                    )

    def test_prod_topk16_batch_count_sweep(self):
        for ip in (0, 4):
            for b in [1, 2, 3, 4, 6, 8]:
                q_sl = [4] * (b - 1) + [128]
                kv_sl = [2048] * (b - 1) + [4096]
                with self.subTest(inner_precise=ip, batch=b):
                    self._run_prod_config(
                        batch=b,
                        q_seqlens=q_sl,
                        kv_seqlens=kv_sl,
                        seed=9300 + b,
                        prec=1.5e-2,
                        inner_precise=ip,
                        tag=f"batch_count/b{b}/ip{ip}",
                    )

    def test_prod_topk16_history_heavy(self):
        for ip in (0, 4):
            for idx, (q, kv) in enumerate(
                [(1, 8192), (2, 8192), (4, 4096), (8, 8192), (4, 16384), (16, 16384)]
            ):
                with self.subTest(inner_precise=ip, q=q, kv=kv):
                    self._run_prod_config(
                        batch=1,
                        q_seqlens=[q],
                        kv_seqlens=[kv],
                        seed=9400 + idx,
                        prec=1.5e-2,
                        inner_precise=ip,
                        tag=f"history/q{q}_kv{kv}/ip{ip}",
                    )

    def test_prod_topk16_pure_prefill_seed_sweep(self):
        for ip in (0, 4):
            for seed in range(25):
                with self.subTest(inner_precise=ip, seed=seed):
                    self._run_prod_config(
                        batch=1,
                        q_seqlens=[2048],
                        kv_seqlens=[2048],
                        seed=9500 + seed,
                        prec=1.5e-2,
                        inner_precise=ip,
                        tag=f"pure_prefill_sweep/ip{ip}/{seed}",
                    )

    def test_prod_topk16_decode_seed_sweep(self):
        for ip in (0, 4):
            for seed in range(25):
                with self.subTest(inner_precise=ip, seed=seed):
                    self._run_prod_config(
                        batch=1,
                        q_seqlens=[1],
                        kv_seqlens=[2048],
                        seed=9600 + seed,
                        prec=max(1e-2, _PREC[ip]),
                        inner_precise=ip,
                        softmax_lse_flag=(seed % 5 == 0),
                        tag=f"decode_sweep/ip{ip}/{seed}",
                    )

    def test_prod_topk16_uneven_multibatch_matrix(self):
        uneven_configs = [
            (2, [1, 1024], [256, 2048], 9801, 1.5e-2, "2b_hot1024"),
            (2, [1, 2048], [256, 4096], 9802, 1.5e-2, "2b_hot2048"),
            (2, [2, 512], [128, 4096], 9803, 1.5e-2, "2b_hot512_kv1short"),
            (3, [1, 512, 1], [256, 2048, 256], 9804, 1.5e-2, "3b_mid_hot"),
            (3, [4, 4, 1024], [512, 512, 2048], 9805, 1.5e-2, "3b_tail_hot"),
            (4, [1, 256, 1, 256], [256, 2048, 256, 2048], 9806, 1.5e-2, "4b_alt_hot"),
            (2, [8, 2048], [512, 8192], 9807, 1.5e-2, "2b_hot2048_kv8192"),
            (3, [1, 1, 2048], [128, 128, 4096], 9808, 1.5e-2, "3b_two_cold_one_hot"),
            (2, [128, 1], [2048, 256], 9809, 1.5e-2, "2b_hot_first_then_cold"),
            (
                5,
                [1, 1, 1, 1, 512],
                [256, 256, 256, 256, 4096],
                9810,
                1.5e-2,
                "5b_cold_then_hot",
            ),
        ]
        for ip in (0, 4):
            for cfg in uneven_configs:
                batch, q_sl, kv_sl, seed, prec, stag = cfg
                with self.subTest(inner_precise=ip, tag=stag):
                    self._run_prod_config(
                        batch=batch,
                        q_seqlens=q_sl,
                        kv_seqlens=kv_sl,
                        seed=seed,
                        prec=prec,
                        inner_precise=ip,
                        tag=f"uneven/{stag}/ip{ip}",
                    )

    def test_prod_topk16_uneven_seed_sweep(self):
        for ip in (0, 4):
            for seed in range(25):
                with self.subTest(inner_precise=ip, seed=seed):
                    self._run_prod_config(
                        batch=2,
                        q_seqlens=[1, 1024],
                        kv_seqlens=[256, 4096],
                        seed=9900 + seed,
                        prec=1.5e-2,
                        inner_precise=ip,
                        tag=f"uneven_sweep/ip{ip}/{seed}",
                    )

    def test_prod_topk16_edge_multi_seed(self):
        base_edges = [
            (1, [16], [2176], "partial_last_block"),
            (1, [1], [128], "kv1block"),
            (1, [32], [512], "kv4block_partial_slots"),
            (2, [1, 512], [256, 2048], "uneven_hot_batch"),
            (1, [1024], [8192], "extreme_q1024_kv8192"),
            (1, [256], [4096], "long_prefill_kv4096"),
        ]
        for ip in (0, 4):
            for (batch, q_sl, kv_sl, tag0), s in (
                (cfg, s) for cfg in base_edges for s in range(3)
            ):
                with self.subTest(inner_precise=ip, tag=tag0, seed_off=s):
                    self._run_prod_config(
                        batch=batch,
                        q_seqlens=q_sl,
                        kv_seqlens=kv_sl,
                        seed=9700 + s * 37,
                        prec=1.5e-2,
                        inner_precise=ip,
                        tag=f"edge_ms/{tag0}/s{s}/ip{ip}",
                    )

    # ---------------------------------------------------------------------------
    # BNSD / BSND production-like (contiguous KV; paged is TND-only)
    # ---------------------------------------------------------------------------

    def test_prod_layout_bnsd_bsnd_edge(self):
        layout_edges = [
            (1, [16], [512], 5101, 1.5e-2, "q16_kv512"),
            (1, [1], [2048], 5102, 1e-2, "q1_decode"),
            (1, [64], [128], 5103, 1e-2, "kv1block"),
            (2, [0, 32], [0, 1024], 5104, 1.5e-2, "pad_lead"),
            (3, [8, 0, 16], [256, 0, 512], 5105, 1.5e-2, "pad_mid"),
            (2, [1, 128], [256, 2048], 5106, 1.5e-2, "uneven"),
            (1, [128], [2048], 5107, 1.5e-2, "q128_kv2048"),
            (1, [256], [4096], 5108, 1.5e-2, "q256_kv4096"),
            (4, [0, 4, 0, 8], [0, 256, 0, 512], 5109, 1.5e-2, "pad_4b"),
            (1, [32], [300], 5110, 1e-2, "partial_kv"),
        ]
        for layout in ("BNSD", "BSND"):
            for ip in (0, 4):
                for cfg in layout_edges:
                    batch, q_sl, kv_sl, seed, prec, stag = cfg
                    with self.subTest(layout=layout, inner_precise=ip, tag=stag):
                        self._run_prod_config(
                            batch=batch,
                            q_seqlens=q_sl,
                            kv_seqlens=kv_sl,
                            seed=seed,
                            prec=max(prec, _PREC[ip]),
                            inner_precise=ip,
                            layout=layout,
                            paged=False,
                            softmax_lse_flag=True,
                            tag=f"{layout}/{stag}/ip{ip}",
                        )

    def test_prod_layout_bnsd_bsnd_q_boundary(self):
        q_vals = [1, 8, 16, 32, 64, 127, 128, 129, 256, 512]
        for layout in ("BNSD", "BSND"):
            for ip in (0, 4):
                for idx, q in enumerate(q_vals):
                    with self.subTest(layout=layout, inner_precise=ip, q=q):
                        self._run_prod_config(
                            batch=1,
                            q_seqlens=[q],
                            kv_seqlens=[1024],
                            seed=5200 + idx,
                            prec=max(1e-2, _PREC[ip]),
                            inner_precise=ip,
                            layout=layout,
                            paged=False,
                            softmax_lse_flag=(q in (1, 128)),
                            tag=f"{layout}/q{q}/ip{ip}",
                        )

    def test_prod_layout_bnsd_bsnd_kv_blocks(self):
        for layout in ("BNSD", "BSND"):
            for ip in (0, 4):
                for idx, n in enumerate([1, 2, 8, 16, 17, 32]):
                    with self.subTest(layout=layout, inner_precise=ip, n_blocks=n):
                        self._run_prod_config(
                            batch=1,
                            q_seqlens=[32],
                            kv_seqlens=[n * 128],
                            seed=5300 + idx,
                            prec=max(1e-2, _PREC[ip]),
                            inner_precise=ip,
                            layout=layout,
                            paged=False,
                            softmax_lse_flag=True,
                            tag=f"{layout}/kv_n{n}/ip{ip}",
                        )

    def test_prod_layout_bnsd_bsnd_seed_sweep(self):
        for layout in ("BNSD", "BSND"):
            for ip in (0, 4):
                for seed in range(15):
                    with self.subTest(layout=layout, inner_precise=ip, seed=seed):
                        self._run_prod_config(
                            batch=3,
                            q_seqlens=[0, 64, 8],
                            kv_seqlens=[0, 1024, 256],
                            seed=5400 + seed,
                            prec=1.5e-2,
                            inner_precise=ip,
                            layout=layout,
                            paged=False,
                            softmax_lse_flag=True,
                            tag=f"{layout}/pad_sweep/{seed}/ip{ip}",
                        )

    def test_inner_precise_1_prod_like(self):
        self._run_prod_config(
            batch=1,
            q_seqlens=[32],
            kv_seqlens=[1024],
            seed=5601,
            prec=_PREC[1],
            inner_precise=1,
            softmax_lse_flag=True,
            tag="ip1/tnd",
        )
        for layout in ("BNSD", "BSND"):
            with self.subTest(layout=layout):
                self._run_prod_config(
                    batch=2,
                    q_seqlens=[0, 16],
                    kv_seqlens=[0, 512],
                    seed=5602,
                    prec=_PREC[1],
                    inner_precise=1,
                    layout=layout,
                    paged=False,
                    softmax_lse_flag=True,
                    tag=f"ip1/{layout}/pad",
                )


if __name__ == "__main__":
    run_tests()
