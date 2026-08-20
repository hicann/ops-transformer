# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

"""TurboQuantSparseFlashAttention 精度用例。

判据取自 experimental 精度标准
(opbase/docs/zh/ops_precision_standard/experimental_standard.md) 的双条件：
    1) 混合容差 |actual - golden| <= atol + rtol * |golden| 的元素占比 >= 0.99
    2) 最大绝对误差 <= 该数据类型的硬上限

标杆的选取：本算子为 4bit 量化算子，其输出与全精度 Attention 之间存在量化本身引入
的固有偏差。若直接对标全精度实现，量化误差会被计入算子误差，无法反映实现是否正确。
因此 golden 采用**与算子相同量化方案的 float64 CPU 实现**——量化/反量化、码本查找、
fp16 scale 舍入均按算子语义复刻，只是全程用 float64 计算，残差中只剩实现误差。
"""

import unittest

import numpy as np
import torch
import torch_npu
import custom_ops  # noqa: F401  register npu_turbo_quant_sparse_flash_attention

# 优先用仓内惯例的 torch_npu TestCase；该模块依赖 expecttest，缺失时退回标准 unittest，
# 使用例在未装测试依赖的环境下同样可跑。
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    TestCase = unittest.TestCase
    run_tests = unittest.main

DEVICE_ID = 0
torch_npu.npu.set_device(DEVICE_ID)

DN = 512  # latent(nope) 维度
DR = 64  # rope 维度
D = DN + DR  # query 最后一维（含 rope）
TILE = 128
NHEAD = 8
BLOCK = 128  # PageAttention block size
INT64_MAX = 9223372036854775807

# 4bit 码本，升序。相邻中心的中点构成 15 条单调边界，最近邻搜索即"数边界"。
CENT = np.array(
    [
        -0.12091285,
        -0.09111122,
        -0.07112455,
        -0.05513602,
        -0.04132067,
        -0.02874970,
        -0.01700489,
        -0.00568677,
        0.00547294,
        0.01680406,
        0.02857605,
        0.04108622,
        0.05492980,
        0.07101817,
        0.09115373,
        0.12037795,
    ],
    dtype=np.float32,
)

# 规范表格：FLOAT16 -> rtol=atol=2^-9，匹配率 0.99，绝对误差上限 1e-1 或 32*ULP。
# 输出为 BFLOAT16 时尾数位更少（8 bit vs 11 bit），沿用 FLOAT16 阈值属于偏严的取法。
TOL = {
    torch.bfloat16: dict(rtol=2**-9, atol=2**-9, ratio=0.99, max_abs=1e-1),
}


def pack_slot(nope_f32, rope_f32):
    """按算子的存储格式打出 KV slot: [256B nibble | 128B rope(bf16) | 2B fp16 scale]。

    rope 按契约存放 rope / s_t：kernel 把 s_t 作用在整列 score 上（含 rope 项），
    乘回 s_t 后才还原真实 rope 贡献。

    同时返回 golden 需要的 yhat、fp16 scale 与**实际存入**的 rope（已预除且按 bf16
    舍入），避免两侧各算一遍量化或舍入引入不一致。
    """
    ntok = nope_f32.shape[0]
    vecnorm = np.sqrt((nope_f32.astype(np.float64) ** 2).sum(1) + 1e-16)
    u = nope_f32 / vecnorm[:, None].astype(np.float32)

    bnd = ((CENT[:-1] + CENT[1:]) / 2.0).astype(np.float32)
    nib = (
        np.searchsorted(bnd, u.ravel(), side="right").reshape(ntok, DN).astype(np.uint8)
    )
    yhat = CENT[nib]
    scale = (vecnorm / np.sqrt((yhat.astype(np.float64) ** 2).sum(1) + 1e-16)).astype(
        np.float16
    )

    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)  # 低位 = 偶数维
    # 契约：存入 rope / s_t
    rope_store = (rope_f32 / scale.astype(np.float32)[:, None]).astype(np.float32)
    rope_t = torch.from_numpy(rope_store).to(torch.bfloat16).contiguous()
    rope_b16 = rope_t.view(torch.uint8).numpy()
    rope_eff = rope_t.float().numpy().astype(np.float64)  # golden 用实际存入值

    slot = np.zeros((ntok, DN // 2 + DR * 2 + 2), dtype=np.uint8)
    slot[:, : DN // 2] = packed
    slot[:, DN // 2 : DN // 2 + DR * 2] = rope_b16.reshape(ntok, DR * 2)
    slot[:, -2:] = scale.view(np.uint8).reshape(ntok, 2)
    return slot, yhat.astype(np.float64), scale, rope_eff


def build_case(bsz, s2, topk, seed, q_dtype, dist, sigma=None, rope_amp=None, q_len=1):
    """构造一条用例。

    s2 为每个 batch 的**有效** KV 长度，允许非 block 对齐——此时按 ceil 分配 block，
    尾块只有前若干 token 有效，正好覆盖尾部处理路径。
    dist: 'normal' 或 'uniform'，对应规范中 50/50 的数据分布要求。
    """
    rng = np.random.default_rng(seed)
    # s2 可以是标量（各 batch 等长）或长度为 bsz 的序列（各 batch 不等长）。
    # 不等长时按最长者统一分配 block，短 batch 的尾部 block 不被 sparse_indices 选中。
    s2_list = [int(s2)] * bsz if np.isscalar(s2) else [int(v) for v in s2]
    assert len(s2_list) == bsz
    # q_len 为标量或长度 bsz 的序列；> 1 时进入 causal 路径（sparse_mode=3 下
    # 每个 query 只可见前缀 KV），此前所有用例都是每 batch 单 query 的 decode 形状。
    q_list = [int(q_len)] * bsz if np.isscalar(q_len) else [int(v) for v in q_len]
    assert len(q_list) == bsz and all(0 < q <= s for q, s in zip(q_list, s2_list))
    t_total = sum(q_list)
    blk_per_b = (max(s2_list) + BLOCK - 1) // BLOCK
    nblk = bsz * blk_per_b
    ntok = nblk * BLOCK

    # sigma 决定归一化系数 s_t 的量级（nope 按 sqrt(DN) 归一后，||nope|| ≈ sigma）；
    # rope_amp 决定 rope 项在 score 中的占比。两者都需远离退化点才能检出 s_t 语义错误。
    if dist == "uniform":
        amp = 0.02 if rope_amp is None else rope_amp
        nope = (rng.uniform(-5.0, 5.0, (ntok, DN)) / np.sqrt(DN)).astype(np.float32)
        rope = (rng.uniform(-5.0, 5.0, (ntok, DR)) * amp).astype(np.float32)
        qraw = (rng.uniform(-5.0, 5.0, (t_total, NHEAD, D)) * 0.02).astype(np.float32)
    else:
        sig = float(rng.uniform(0.1, 2.0)) if sigma is None else sigma
        amp = 0.1 if rope_amp is None else rope_amp
        nope = (rng.normal(0.0, sig, (ntok, DN)) / np.sqrt(DN)).astype(np.float32)
        rope = (rng.normal(0.0, sig, (ntok, DR)) * amp).astype(np.float32)
        qraw = (rng.normal(0.0, sig, (t_total, NHEAD, D)) * 0.1).astype(np.float32)

    slot, yhat, scale, rope_eff = pack_slot(nope, rope)
    kv = torch.from_numpy(slot.view(np.int8)).reshape(nblk, BLOCK, 1, -1).npu()

    # sparse_indices: 每个 query 从本 batch 的有效 KV 里选。topk 超过实际 KV 长度时，
    # 尾部以 -1 填充（生产语义：topk 缓冲宽度固定而 KV 长度可变）。
    # 索引是 **每 batch 局部** 的 KV 位置，全局落点由 block_table 解析；
    # golden 侧需自行加上 batch 的块偏移。
    # sparse_mode=3 为右对齐因果：batch 内第 r 个 query 可见 KV [0, s2 - q_len + r]。
    # kernel 只读取 sparse_indices 的前 min(topk, 可见长度) 个槽位，故此处按可见范围
    # 取样并前置紧排，尾部以 -1 填充——与 README 的契约一致。
    nvalid = []
    idx = np.full((t_total, topk), -1, dtype=np.int32)
    t = 0
    for b in range(bsz):
        for r in range(q_list[b]):
            limit = s2_list[b] - q_list[b] + r + 1  # 该 query 的因果可见 KV 长度
            nv = min(topk, limit)
            idx[t, :nv] = rng.choice(limit, size=nv, replace=False).astype(np.int32)
            nvalid.append(nv)
            t += 1
    si = torch.from_numpy(idx).reshape(t_total, 1, -1).npu()

    q = torch.from_numpy(qraw).to(q_dtype).npu()
    bt = torch.arange(nblk, dtype=torch.int32).reshape(bsz, -1).npu()
    # TND 布局下 actual_seq_lengths 是【累积和】，不是每 batch 的长度
    asq = torch.tensor(np.cumsum(q_list), dtype=torch.int32).npu()
    # actual_seq_lengths_query 为 TND，kernel 按相邻差分还原每 batch 长度，故传累加和；
    # actual_seq_lengths_kv 走 PA_BSND 分支，kernel 直接取 [bIdx]，故传每 batch 实际长度。
    ask = torch.tensor(s2_list, dtype=torch.int32).npu()
    ks = torch.ones(nblk, BLOCK, 1, DN // TILE, dtype=torch.float32).npu()

    return dict(
        q=q,
        kv=kv,
        si=si,
        bt=bt,
        asq=asq,
        ask=ask,
        ks=ks,
        idx=idx,
        nvalid=nvalid,
        q_list=q_list,
        t_total=t_total,
        blk_per_b=blk_per_b,
        yhat=yhat,
        scale=scale,
        rope_eff=rope_eff,
        qraw=qraw,
        q_dtype=q_dtype,
        scale_value=1.0 / (DN**0.5),
    )


def golden(c):
    """float64 CPU 标杆，复刻算子语义。

    s_t 作用在**整列 score** 上（含 rope 项），与 kernel 的按列缩放一致；
    rope 取 KV slot 中**实际存入**的值（已预除 s_t 并按 bf16 舍入）。

    q 必须按算子实际收到的 dtype 做舍入：fp16 有 11 位尾数、bf16 只有 8 位，
    若一律用 bf16 建模，fp16 用例会拿一份更粗的 q 去比对，差异被误算成算子误差。
    rope 则固定按 bf16 —— 那是 KV slot 的存储格式，与 query dtype 无关。
    """
    s_t = c["scale"].astype(np.float64)
    Yhat = c["yhat"]  # 未乘 s_t 的码本重建值
    Kr = c["rope_eff"]  # slot 中实际存入的 rope（已预除 s_t）
    q = torch.from_numpy(c["qraw"]).to(c["q_dtype"]).float().numpy().astype(np.float64)
    qn, qr = q[..., :DN], q[..., DN:]

    # 按**全局 query 下标**遍历：每个 query 有自己的 sparse_indices 与可见长度，
    # 所属 batch 决定 KV 的 block 偏移。
    out = np.empty((c["t_total"], NHEAD, DN), dtype=np.float64)
    t = 0
    for b in range(len(c["q_list"])):
        for _ in range(c["q_list"][b]):
            # 局部索引 -> 全局 token 下标；-1 填充位不参与计算
            sel = (
                c["idx"][t][: c["nvalid"][t]].astype(np.int64)
                + b * c["blk_per_b"] * BLOCK
            )
            yh, kr, sb = Yhat[sel], Kr[sel], s_t[sel]
            kn = yh * sb[:, None]  # 反量化后的 latent，V 侧直接用它
            # s_t 按列作用在整条 score 上，rope 项因存放时已预除而被还原
            sc = sb[None, :] * (qn[t] @ yh.T + qr[t] @ kr.T) * c["scale_value"]
            sc = sc - sc.max(axis=1, keepdims=True)
            p = np.exp(sc)
            p = p / p.sum(axis=1, keepdims=True)
            out[t] = p @ kn
            t += 1
    return out


def check(actual, expect, dtype):
    """规范的双条件判定。返回 (是否通过, 描述)。"""
    t = TOL[dtype]
    err = np.abs(actual.astype(np.float64) - expect)
    tol = t["atol"] + t["rtol"] * np.abs(expect)
    matched = float((err <= tol).mean())
    max_abs = float(err.max())
    ok = matched >= t["ratio"] and max_abs <= t["max_abs"]
    desc = "匹配率 %.4f (>=%.2f)  最大绝对误差 %.3e (<=%.1e)" % (
        matched,
        t["ratio"],
        max_abs,
        t["max_abs"],
    )
    return ok, desc


class TestTurboQuantSparseFlashAttention(TestCase):
    """覆盖：2^n 与 2^n-1 两类尺寸、均匀与正态两种分布。

    算子仅声明支持 BFLOAT16（FLOAT16 已从支持列表移除），故不再覆盖 fp16。
    """

    CASES = [
        # (bsz, s2, topk, dist)；s2 取 2^n 与 2^n-1 两类，后者非 block 对齐
        (1, 1024, 256, "normal"),
        (1, 1023, 256, "uniform"),
        (4, 2048, 512, "normal"),
        (4, 2047, 512, "uniform"),
        (8, 512, 128, "normal"),
        (8, 511, 128, "uniform"),
        # T=25：非 2 的幂且为奇数，触及分核尾部；与算子实测采集的 query shape 对齐
        (25, 1024, 256, "normal"),
        (25, 1023, 256, "uniform"),
    ]

    @staticmethod
    def _call(c, with_scale=True):
        """with_scale=False 时不传两个 dequant scale 可选输入。"""
        scale = (
            {"key_dequant_scale": c["ks"], "value_dequant_scale": c["ks"]}
            if with_scale
            else {}
        )
        out, _, _ = torch.ops.custom.npu_turbo_quant_sparse_flash_attention(
            c["q"],
            c["kv"],
            c["kv"],
            c["si"],
            **scale,
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
            tile_size=TILE,
            rope_head_dim=DR,
            return_softmax_lse=False,
        )
        torch.npu.synchronize()
        return out

    def _run_one(
        self, bsz, s2, topk, dist, dtype, seed, sigma=None, rope_amp=None, q_len=1
    ):
        c = build_case(
            bsz,
            s2,
            topk,
            seed,
            dtype,
            dist,
            sigma=sigma,
            rope_amp=rope_amp,
            q_len=q_len,
        )
        out = self._call(c)
        actual = out.float().cpu().numpy().reshape(c["t_total"], NHEAD, DN)
        ok, desc = check(actual, golden(c), dtype)
        tag = "bsz=%d s2=%-11s topk=%-4d q_len=%-7s %-7s %s" % (
            bsz,
            s2 if np.isscalar(s2) else ",".join(str(v) for v in s2),
            topk,
            q_len if np.isscalar(q_len) else ",".join(str(v) for v in q_len),
            dist,
            str(dtype).split(".")[-1],
        )
        print("  [%s] %s  %s" % ("PASS" if ok else "FAIL", tag, desc), flush=True)
        self.assertTrue(ok, "%s: %s" % (tag, desc))

    def test_precision_bf16(self):
        for i, (bsz, s2, topk, dist) in enumerate(self.CASES):
            with self.subTest(case=i):
                self._run_one(bsz, s2, topk, dist, torch.bfloat16, seed=1000 + i)

    def test_topk_exceeds_actual_kv(self):
        """topk 缓冲宽度 >= 实际 KV 长度的场景，sparse_indices 尾部为 -1 填充。

        该区间历史上触发过尾部处理缺陷（非 32B 对齐的向量写），生产中短 prompt 与
        图捕获 warmup 必经，需单独覆盖。
        """
        for i, (s2, topk) in enumerate([(128, 256), (256, 256), (100, 512)]):
            with self.subTest(case=i):
                self._run_one(1, s2, topk, "normal", torch.bfloat16, seed=3000 + i)

    def test_varying_kv_lengths(self):
        """各 batch KV 长度不等：覆盖 actual_seq_lengths_kv 这一维。

        PA_BSND 下 kernel 直接取 actual_seq_lengths_kv[bIdx]（只有 TND 分支才对累加和
        做差分）。等长用例无法区分该语义，且实测表明传入大于实际长度的值是惰性的，
        因此必须用不等长且含短 batch 的组合才能真正覆盖。
        """
        for i, s2 in enumerate(
            [[512, 256, 128, 64], [1024, 512, 100, 900], [128, 1023]]
        ):
            with self.subTest(case=i):
                self._run_one(len(s2), s2, 256, "normal", torch.bfloat16, seed=6000 + i)

    def test_multi_query_causal(self):
        """每 batch 多个 query，走 sparse_mode=3 的因果路径。

        此前所有精度用例都是每 batch 单 query 的 decode 形状（T=25 是 25 个 batch
        各 1 个 query），因果分支完全没有覆盖。这里覆盖等长与不等长两种 q_len，
        并让 q_len 跨越 block 边界。
        """
        cases = [
            (1, 512, 256, 4),  # 单 batch 4 个 query
            (2, 512, 256, [3, 8]),  # 不等长 q_len
            (2, [256, 1023], 256, [4, 4]),  # 不等长 KV + 多 query
            (1, 512, 128, 129),  # q_len 跨 block 边界（>128）
            # 下面两条让因果可见长度 < topk，使 kernel 的 causal 裁剪真正生效：
            # 上面几条 topk 均小于可见长度，裁剪不起作用，测不到该分支。
            (1, 8, 256, 8),  # 可见长度 1..8，全部 < topk
            (2, [16, 40], 64, [16, 40]),  # 可见长度最小为 1
        ]
        for i, (bsz, s2, topk, q_len) in enumerate(cases):
            with self.subTest(case=i):
                self._run_one(
                    bsz, s2, topk, "normal", torch.bfloat16, seed=7000 + i, q_len=q_len
                )

    def test_rope_scale_contract(self):
        """rope 主导且 s_t 远离 1：检出 s_t 未作用于 rope 项一类的语义错误。

        kernel 把 s_t 作用在整列 score 上，故 rope 按 rope / s_t 存放。若该契约被
        破坏，误差为 (s_t - 1)·q_rope·rope；rope 幅值小或 s_t ≈ 1 时不可见，因此
        必须显式取 sigma 远离 1（s_t ≈ sigma）且 rope 幅值显著的组合。
        """
        for i, (sigma, amp) in enumerate([(2.0, 5.0), (2.0, 0.5), (0.1, 5.0)]):
            with self.subTest(case=i):
                self._run_one(
                    4,
                    512,
                    256,
                    "normal",
                    torch.bfloat16,
                    seed=5000 + i,
                    sigma=sigma,
                    rope_amp=amp,
                )

    def test_dequant_scale_not_consumed(self):
        """quant_scale_repo_mode=1（COMBINE）下反量化系数存放于 KV slot 内。

        此时 key_dequant_scale 与 value_dequant_scale 两个可选输入不被 kernel 消费，
        不传应当成功，且结果与传入时逐位相同。
        """
        c = build_case(4, 512, 256, 4000, torch.bfloat16, "normal")
        with_scale = self._call(c, with_scale=True).cpu()
        without_scale = self._call(c, with_scale=False).cpu()
        same = torch.equal(with_scale, without_scale)
        print(
            "  [%s] 不传 dequant scale 结果逐位相同" % ("PASS" if same else "FAIL"),
            flush=True,
        )
        self.assertTrue(same, "不传 dequant scale 时输出发生变化")


if __name__ == "__main__":
    run_tests()
