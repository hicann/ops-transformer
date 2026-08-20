#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TurboQuantSparseFlashAttention 非法参数用例。

覆盖算子声明与 host 校验的支持边界：属性取值、layout 组合、dtype 与 shape。
每条用例都断言调用被拒绝，避免「声明接受但实现不支持」的组合被静默放行。
合法基线同时被断言可正常执行，防止校验收紧过头把正常用法一并拦掉。
"""

import unittest

import numpy as np
import torch
import torch_npu  # noqa: F401

import custom_ops  # noqa: F401  register npu_turbo_quant_sparse_flash_attention

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    TestCase = unittest.TestCase
    run_tests = unittest.main

from test_npu_turbo_quant_sparse_flash_attention import (  # noqa: E402
    BLOCK,
    DN,
    NHEAD,
    build_case,
)

INT64_MAX = 9223372036854775807


class TestTurboQuantSparseFlashAttentionInvalid(TestCase):
    """非法参数应被 host 校验拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.c = build_case(1, 512, 256, 9, torch.bfloat16, "normal")

    def _call(self, q=None, k=None, v=None, **over):
        return self._call_on(self.c, q=q, k=k, v=v, **over)

    def _call_on(self, c, q=None, k=None, v=None, **over):
        kw = dict(
            key_dequant_scale=c["ks"],
            value_dequant_scale=c["ks"],
            block_table=c["bt"],
            actual_seq_lengths_query=c["asq"],
            actual_seq_lengths_kv=c["ask"],
            scale_value=c["scale_value"],
            key_quant_mode=3,
            value_quant_mode=3,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            pre_tokens=INT64_MAX,
            next_tokens=INT64_MAX,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
            return_softmax_lse=False,
        )
        kw.update(over)
        out = torch.ops.custom.npu_turbo_quant_sparse_flash_attention(
            c["q"] if q is None else q,
            c["kv"] if k is None else k,
            c["kv"] if v is None else v,
            c["si"],
            **kw,
        )
        torch.npu.synchronize()
        return out

    def _assert_rejected(self, tag, **kw):
        """断言被拒绝，且必须是 RuntimeError。

        只捕获 RuntimeError：host 参数校验失败与 TORCH_CHECK 均抛该类型；用例自身写错
        参数名会抛 TypeError，若一并吞掉就会把测试 bug 误判成「正确拒绝」。
        """
        with self.subTest(case=tag):
            try:
                self._call(**kw)
            except RuntimeError as e:
                msg = str(e).replace("\n", " ")
                self.assertTrue(msg, "%s 抛出了空错误信息" % tag)
                print("  [PASS] %-38s 已拒绝: %s" % (tag, msg[:70]), flush=True)
                return
            self.fail("%s 未被拒绝" % tag)

    def test_baseline_is_valid(self):
        """合法基线必须能跑通，否则「全部拒绝」会让下列断言失去意义。"""
        out, _, _ = self._call()
        self.assertEqual(list(out.shape), [1, NHEAD, DN])
        print("  [PASS] 合法基线 out.shape=%s" % list(out.shape), flush=True)

    def test_invalid_attr_value(self):
        """属性取值：当前实现对每个属性只有唯一合法值。"""
        self._assert_rejected("tile_size=64", tile_size=64)
        self._assert_rejected("tile_size=256", tile_size=256)
        self._assert_rejected("rope_head_dim=32", rope_head_dim=32)
        self._assert_rejected("key_quant_mode=1", key_quant_mode=1)
        self._assert_rejected("value_quant_mode=1", value_quant_mode=1)
        self._assert_rejected("attention_mode=0", attention_mode=0)
        self._assert_rejected("quant_scale_repo_mode=0", quant_scale_repo_mode=0)

    def test_invalid_layout(self):
        """layout：query 仅 TND，KV 仅 PA_BSND / BSND。"""
        self._assert_rejected("layout_query=BSND", layout_query="BSND")
        self._assert_rejected("layout_query=BNSD", layout_query="BNSD")
        self._assert_rejected("layout_kv=TND", layout_kv="TND")
        self._assert_rejected("layout_kv=BSND", layout_kv="BSND")

    def test_invalid_dtype(self):
        """dtype：query 仅 BFLOAT16；actual_seq_lengths 仅 INT32。"""
        c = self.c
        self._assert_rejected("query=float16", q=c["q"].to(torch.float16))
        self._assert_rejected("query=float32", q=c["q"].float())
        self._assert_rejected(
            "actual_seq_lengths_kv=int64", actual_seq_lengths_kv=c["ask"].long()
        )

    def test_invalid_shape(self):
        """shape：value 须与 key 完全一致；block_table 首维须等于 batch。"""
        c = self.c
        half = c["kv"][: max(1, c["kv"].shape[0] // 2)]
        self._assert_rejected("value block 数减半", v=half)
        self._assert_rejected(
            "block_table 首维不等于 batch",
            block_table=torch.cat([c["bt"], c["bt"]], dim=0),
        )

    def test_zero_kv_len_outputs_initialized(self):
        """某 batch 的 KV 长度为 0 时，三个输出都必须被写入。

        上层把 batch 补齐到固定尺寸时，填充槽位的 actual_seq_lengths_kv 即为 0。
        attention_out 应为全 0；softmax_max / softmax_sum 由调用方 at::empty 分配，
        算子必须落值（max = -2e38、sum = 0），否则返回未初始化内存。

        用毒化分配器验证：先用哨兵值填满同形张量再释放，缓存分配器很可能把同一块
        内存交给算子输出；若算子不写，哨兵值会原样返回。
        """
        c = build_case(4, 512, 256, 8000, torch.bfloat16, "normal")
        ask_zero = torch.zeros(4, dtype=torch.int32).npu()
        lse_shape = (1, c["q"].shape[0], c["q"].shape[1])
        for poison in (-12345.0, 777.0):
            with self.subTest(poison=poison):
                junk = [
                    torch.full(lse_shape, poison, dtype=torch.float32).npu()
                    for _ in range(64)
                ]
                junk += [
                    torch.full(
                        list(c["q"].shape[:2]) + [DN], poison, dtype=torch.bfloat16
                    ).npu()
                    for _ in range(16)
                ]
                torch.npu.synchronize()
                del junk
                out, smax, ssum = self._call_on(
                    c, actual_seq_lengths_kv=ask_zero, return_softmax_lse=True
                )
                out_np = out.float().cpu().numpy()
                smax_np = smax.float().cpu().numpy()
                ssum_np = ssum.float().cpu().numpy()
                self.assertFalse(
                    np.isclose(smax_np, poison).any(), "softmax_max 未被写入"
                )
                self.assertFalse(
                    np.isclose(ssum_np, poison).any(), "softmax_sum 未被写入"
                )
                self.assertTrue(np.allclose(out_np, 0.0), "attention_out 应为全 0")
                self.assertTrue(np.allclose(ssum_np, 0.0), "softmax_sum 应为 0")
                self.assertTrue(
                    (smax_np < -1e38).all(), "softmax_max 应为 SOFTMAX_MIN_NUM"
                )
                print(
                    "  [PASS] 零 KV 长度：out=0, ssum=0, smax=%.1e（毒化值 %s 未泄漏）"
                    % (float(smax_np.ravel()[0]), poison),
                    flush=True,
                )

    def test_meta_matches_runtime(self):
        """Meta（图模式 / FakeTensor）与 runtime 的输出 shape 及入参校验必须一致。

        Meta 走的是 InferShape 同一套 shape 推导。历史上 Meta 只校验 query 维数，
        非法的 rope_head_dim 会算出错误甚至为负的尾维，仅由 at::empty 兜住报错，
        且报错信息与 runtime 不同；正数但错误的取值更会静默返回错误 shape。
        """
        from torch._subclasses.fake_tensor import FakeTensorMode

        c = self.c
        real = [list(t.shape) for t in self._call_on(c)]
        with FakeTensorMode(allow_non_fake_inputs=True):
            meta = [list(t.shape) for t in self._call_on(c)]
        self.assertEqual(meta, real)
        print("  [PASS] Meta 输出 shape 与 runtime 一致: %s" % meta, flush=True)

        # Meta 负责 shape 推导，只校验能否算出合法 shape；参数是否被算子支持由 host
        # tiling 判定。故分两类断言，不把 host 的取值约束在 Meta 里再复制一份。
        # layout 与 rank 在 Meta 层即应拒绝，避免图推导先得到看似合法的 shape。
        with FakeTensorMode(allow_non_fake_inputs=True):
            for tag, over in [
                ("layout_query=BSND", dict(layout_query="BSND")),
                ("layout_kv=BSND", dict(layout_kv="BSND")),
            ]:
                with self.subTest(case=tag):
                    with self.assertRaises(RuntimeError):
                        self._call_on(c, **over)
                    print("  [PASS] Meta 拒绝 %s" % tag, flush=True)

        with FakeTensorMode(allow_non_fake_inputs=True):
            for rhd in (4096, -1):  # 尾维为负 / 取值为负：Meta 自身即无法算出合法 shape
                with self.subTest(rope_head_dim=rhd):
                    with self.assertRaises(RuntimeError):
                        self._call_on(c, rope_head_dim=rhd)
                    print("  [PASS] Meta 拒绝 rope_head_dim=%d" % rhd, flush=True)

        # 取值合法但算子不支持（仅支持 64）：Meta 能算出 shape，runtime 必须明确拒绝，
        # 不能静默返回与 Meta 不同的结果。
        with FakeTensorMode(allow_non_fake_inputs=True):
            meta_shape = list(self._call_on(c, rope_head_dim=100)[0].shape)
        self.assertEqual(meta_shape[2], c["q"].shape[2] - 100)
        with self.assertRaises(RuntimeError):
            self._call_on(c, rope_head_dim=100)
        print(
            "  [PASS] rope_head_dim=100: Meta 得 %s，runtime 拒绝" % meta_shape,
            flush=True,
        )

    def test_softmax_lse_shape(self):
        """softmax_max / softmax_sum 的 shape 随 return_softmax_lse 变化。

        为 False 时算子不产出 LSE，两个输出为空张量；为 True 时 shape 为
        [N2, T, N1 / N2]——注意首维是 KV 头数而非 T。当前 N2 恒为 1，若只用 T=1 的
        用例，[N2, T, G] 与 [T, N2, G] 无法区分，故必须同时覆盖 T > 1。
        """
        for bsz in (1, 4):
            c = build_case(bsz, 512, 256, 4100 + bsz, torch.bfloat16, "normal")
            t_size, n1 = c["q"].shape[0], c["q"].shape[1]
            n2 = c["kv"].shape[2]  # PA_BSND: [blockNum, blockSize, N2, D]
            for flag in (False, True):
                with self.subTest(bsz=bsz, return_softmax_lse=flag):
                    _, smax, ssum = self._call_on(c, return_softmax_lse=flag)
                    self.assertEqual(smax.dtype, torch.float32)
                    self.assertEqual(ssum.dtype, torch.float32)
                    self.assertEqual(list(smax.shape), list(ssum.shape))
                    if flag:
                        self.assertEqual(list(smax.shape), [n2, t_size, n1 // n2])
                    else:
                        self.assertEqual(smax.numel(), 0)
                    print(
                        "  [PASS] T=%d return_softmax_lse=%-5s smax.shape=%s"
                        % (t_size, flag, list(smax.shape)),
                        flush=True,
                    )


if __name__ == "__main__":
    run_tests()
