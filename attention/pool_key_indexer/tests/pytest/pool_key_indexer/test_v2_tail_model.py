#!/usr/bin/python
# -*- coding: utf-8 -*-
# fmt: off
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# -----------------------------------------------------------------------------------------------------------
"""V2 接口(sparse_indices_mode) + tail 语义 + GLM53 模型场景冒烟套。

场景覆盖(合并原 test_version_v2 / recheck_tail_and_compact / archive_recheck /
model_glm53_tail / c8_features F1):
  * V2 compact 布局: 三条展开路径(ps%8==0 vecPath / ps%8!=0 gatherPath)短序列 +
    长序列, 紧凑性不变量 + 双布局有效集合一致;
  * mode=0 经 V2 与 V1 逐比特一致(BSND + TND/PA); mode=2 非法 host 拒绝;
  * tail bug 回归(交付修复原始场景): ps=4/topk=2048/L=1025(256 pool+ptk=1),
    query@1024 对 tail token 1024 可见并追加, x fp16/int8 x legacy/compact;
  * 预算充足(pc>sc) V1≡V2 逐比特 / 预算不足(pc<sc) legacy 空洞合同 + compact 前缀合同;
  * 多 query prefill 逐行因果(早期行无未来尾 token, 末期行全尾可见)。
"""

import numpy as np
import pytest
import torch

import pkicase
from pkicase import D, spec, run_golden_compare, rows_of


# ---- V2 compact 三条展开路径 + 长序列 ----
_V2 = [
    ("vec_ps4", 4, (1, 3, 1, D), (1, 3, 1), 3),  # 短序列 3 pool < sc=4
    ("vec_ps16", 16, (1, 1, 1, D), (1, 1, 1), 15),  # 单 pool + 尾 15
    ("gather_ps3", 3, (1, 3, 1, D), (1, 3, 1), 2),  # ps%8!=0 gatherPath
    ("long_ps4", 4, (1, 8, 1, D), (1, 8, 1), 3),  # 长序列 8 pool >= sc=4
]


@pytest.mark.pki_npu
class TestV2Compact:
    @pytest.mark.parametrize("tag,ps,k_shape,kd_shape,ptk", _V2, ids=[v[0] for v in _V2])
    def test_compact(self, npu_backend, tag, ps, k_shape, kd_shape, ptk):
        sp = spec(f"V2c_{tag}", seed=20260200 + ps, layout_q="BSND", layout_k="BSND", q_shape=(1, 4, 2, D), k_shape=k_shape, w_shape=(1, 4, 2), kd_shape=kd_shape, topk=16 if ps != 3 else 12, ps=ps, mask=0, rv=True, ptk=[ptk], quant_mode=2, version=1,)
        idx, _, _ = run_golden_compare(npu_backend, sp, index_version=2)
        rows = rows_of(idx, sp)
        for r in range(rows.shape[0]):
            n_valid = int((rows[r] >= 0).sum())
            assert not np.any(rows[r][:n_valid] == -1) and not np.any(
                rows[r][n_valid:] != -1
            ), f"{tag} row{r}: 紧凑布局存在中间空洞"
        # 双布局有效集合一致
        legacy = spec(sp["name"] + "_lg", sp["seed"], layout_q="BSND", layout_k="BSND", q_shape=sp["q_shape"], k_shape=k_shape, w_shape=(1, 4, 2), kd_shape=kd_shape, topk=sp["topk"], ps=ps, mask=0, rv=True, ptk=[ptk], quant_mode=2, version=None,)
        _, _, g1 = run_golden_compare(npu_backend, legacy, index_version=1)
        g_idx = np.asarray(g1.indices).reshape(-1, sp["topk"] + ps - 1)
        for r in range(rows.shape[0]):
            assert sorted(g_idx[r][g_idx[r] >= 0].tolist()) == sorted(
                rows[r][rows[r] >= 0].tolist()
            ), f"{tag} row{r}: 双布局有效集合不一致"

    @pytest.mark.parametrize("lq,lk,asq,ask,bt,ptk", [ ("BSND", "BSND", None, None, None, [0, 0]), ("TND", "PA_BBND", [1, 2], [16, 32], [[0, 1], [2, 3]], [0, 3]), ], ids=["bsnd", "tnd_pa"],)
    def test_mode0_equals_v1(self, npu_backend, lq, lk, asq, ask, bt, ptk):
        common = dict(layout_q=lq, layout_k=lk, q_shape=(2, 8, D) if lq == "TND" else (2, 4, 2, D), k_shape=(4, 16, 1, D) if lk == "PA_BBND" else (2, 8, 1, D), w_shape=(2, 8) if lq == "TND" else (2, 4, 2), kd_shape=(4, 16, 1) if lk == "PA_BBND" else (2, 8, 1), topk=32, ps=4, mask=3, rv=True, ptk=ptk, asq=asq, ask=ask, bt=bt, quant_mode=2,)
        m0 = spec("V2m0", 20260210, version=0, **common)
        v1 = spec("V1d", 20260210, version=None, **common)
        idx_m0, _, _ = run_golden_compare(npu_backend, m0, index_version=1)
        idx_v1, _, _ = run_golden_compare(npu_backend, v1, index_version=1)
        assert np.array_equal(idx_v1, idx_m0), "V2(mode=0) 与 V1 输出应逐比特一致"

    def test_mode2_rejected(self, npu_backend):
        sp = spec("V2m2", 20260211, layout_q="TND", layout_k="PA_BBND", q_shape=(1, 8, D), k_shape=(4, 16, 1, D), w_shape=(1, 8), kd_shape=(4, 16, 1), topk=32, ps=4, mask=3, rv=True, ptk=[3], asq=[1], ask=[16], bt=[[0, 1]], quant_mode=2, version=2,)
        result = npu_backend.run_case(sp, pkicase.make_inputs(sp))
        assert not isinstance(result, tuple), "sparse_indices_mode=2 应被 host 拒绝"
        assert "sparse_indices_mode" in (result.stdout or ""), (
            f"报错应含 sparse_indices_mode 校验说明:\n{result.stdout[-400:]}"
        )


# ---- tail bug 回归(交付修复原始场景) + 预算两态 + GLM53 逐行因果 ----
def _bug_spec(tag, quant_mode, version):
    import torch

    return spec(f"BUG_{tag}", seed=20260220, layout_q="TND", layout_k="PA_BBND", q_shape=(1, 8, D), k_shape=(16, 32, 1, D), w_shape=(1, 8), kd_shape=(16, 32, 1), topk=2048, ps=4, mask=3, rv=True, ptk=[1], asq=[1], ask=[256], bt=[[0, 1, 2, 3, 4, 5, 6, 7]], quant_mode=quant_mode, version=version, q_dtype=torch.int8 if quant_mode == 2 else torch.float16,)


@pytest.mark.pki_npu
class TestTailSemantics:
    @pytest.mark.parametrize("quant_mode,version", [(-1, None), (-1, 1), (2, None), (2, 1)], ids=["fp16_lg", "fp16_cp", "int8_lg", "int8_cp"],)
    def test_bug_tail_1024(self, npu_backend, quant_mode, version):
        """tail bug 回归: 修复前以 topk 为尾池起点, Max(0, Min(1, 1024-2048+1))=0 漏 tail;
        修复后以 seqTailStart=L-ptk=1024 为起点, query@1024 可见并追加 token 1024。"""
        sp = _bug_spec(f"q{quant_mode}_m{version or 0}", quant_mode, version)
        idx, _, _ = run_golden_compare(
            npu_backend, sp, index_version=(version + 1) if version is not None else 1
        )
        row = rows_of(idx, sp)[0]
        assert 1024 in row.tolist(), (
            f"{sp['name']}: 合法 tail token 1024 未追加(bug 回归)"
        )
        assert int((row == 1024).sum()) == 1 and not np.any(row > 1024), (
            f"{sp['name']}: tail 语义异常(重复/未来 token)"
        )
        if version == 1:
            n_valid = int((row >= 0).sum())
            assert not np.any(row[:n_valid] == -1), "compact 行存在中间空洞"

    def test_budget_regimes(self, npu_backend):
        """预算充足(640>512) V1≡V2 逐比特; 预算不足(256<512) legacy 空洞 + compact 前缀含 tail。"""
        common = dict(layout_q="TND", layout_k="TND", q_shape=(1, 8, D), w_shape=(1, 8), topk=2048, ps=4, mask=3, rv=True, ptk=[1], asq=[1], quant_mode=-1, q_dtype=torch.float16,)
        for pools in (640, 256):
            sp_lg = spec(f"BG_l{pools}", 20260221, k_shape=(pools, 1, D), kd_shape=(pools, 1, D), ask=[pools], version=None, **common,)
            sp_cp = spec(f"BG_c{pools}", 20260221, k_shape=(pools, 1, D), kd_shape=(pools, 1, D), ask=[pools], version=1, **common,)
            idx_lg, _, _ = run_golden_compare(npu_backend, sp_lg, index_version=1)
            idx_cp, _, _ = run_golden_compare(npu_backend, sp_cp, index_version=2)
            row_lg, row_cp = rows_of(idx_lg, sp_lg)[0], rows_of(idx_cp, sp_cp)[0]
            if pools > 512:  # 预算充足: 展开区铺满, 两布局逐比特一致
                assert np.array_equal(row_lg, row_cp), "预算充足 V1/V2 应逐比特一致"
            else:  # 预算不足: legacy [expandTotal, topk) 为 -1 空洞 + tail@topk; compact 前缀含 tail
                assert bool((row_lg[1024:2048] == -1).all()), "legacy 空洞合同破坏"
                assert row_lg[2048] == 1024, "legacy tail 应固定偏移 topk"
                n_valid = int((row_cp >= 0).sum())
                assert not np.any(row_cp[:n_valid] == -1) and row_cp[1024] == 1024, (
                    "compact 前缀应紧跟 tail(无空洞)"
                )

    def test_glm53_causal_rows(self, npu_backend):
        """多 query prefill 逐行因果: 早期行无未来 tail, 末期行全尾可见(300 pool/ptk=1)。"""
        sp = spec("G_rows", 20260222, layout_q="TND", layout_k="TND", q_shape=(8, 8, D), k_shape=(300, 1, D), w_shape=(8, 8), kd_shape=(300, 1, D), topk=2048, ps=4, mask=3, rv=True, ptk=[1], asq=[8], ask=[300], quant_mode=-1, q_dtype=torch.float16,)
        idx, _, _ = run_golden_compare(npu_backend, sp, index_version=1)
        rows = rows_of(idx, sp)
        L, tk, s1 = 300 * 4 + 1, 1, 8
        for i, row in enumerate(rows):
            gpos = L - s1 + i  # query 全局位置(L_k - S1 + 行内偏移)
            expect_tail = list(range(L - tk, L)) if gpos >= L - tk else []
            got_tail = sorted(int(t) for t in row if t >= L - tk)
            assert got_tail == expect_tail, (
                f"gpos={gpos}: 尾可见违约 期望{expect_tail} 实际{got_tail}"
            )
            assert not np.any(row > L - 1), f"gpos={gpos}: 出现未来 token"

    def test_tail_index_domain_bsnd(self, npu_backend):
        """poolTailK>0 索引域回归(#6636): poolKey 仅含完整池, 尾 token 起点=完整池数×ps。

        npool=8 完整池 + ptk=2 → 合法域 [0, 34), 尾区槽为 32,33(尾池 token 的
        原始 KV 索引, 非越界); ptk=0 对照组同位置为 -1, 展开区两态逐比特一致。
        """
        common = dict(layout_q="BSND", layout_k="BSND", q_shape=(1, 2, 1, D), k_shape=(1, 8, 1, D), w_shape=(1, 2, 1), kd_shape=(1, 8, 1, D), topk=16, ps=4, mask=0, rv=True, quant_mode=-1, q_dtype=torch.float16,)
        sp_t2 = spec("TD_6636_t2", 20260223, ptk=[2], version=None, **common)
        sp_t0 = spec("TD_6636_t0", 20260223, ptk=[0], version=None, **common)
        idx_t2, _, _ = run_golden_compare(npu_backend, sp_t2, index_version=1)
        idx_t0, _, _ = run_golden_compare(npu_backend, sp_t0, index_version=1)
        row_t2, row_t0 = rows_of(idx_t2, sp_t2)[0], rows_of(idx_t0, sp_t0)[0]
        pool_tokens = 8 * 4  # poolKey 完整池覆盖的 token 数
        assert np.array_equal(row_t2[:16], row_t0[:16]), "展开区不应随 poolTailK 变化"
        assert row_t2[16:].tolist() == [pool_tokens, pool_tokens + 1, -1], (
            f"尾区槽应为 [{pool_tokens}, {pool_tokens + 1}, -1], 实际 {row_t2[16:].tolist()}"
        )
        assert row_t0[16:].tolist() == [-1, -1, -1], "ptk=0 尾区槽应全 -1"
        valid_t2 = row_t2[row_t2 >= 0]
        assert int(valid_t2.max()) < pool_tokens + 2, "ptk=2 索引应 < npool*ps+ptk"
        valid_t0 = row_t0[row_t0 >= 0]
        assert int(valid_t0.max()) < pool_tokens, "ptk=0 索引应 < npool*ps"
