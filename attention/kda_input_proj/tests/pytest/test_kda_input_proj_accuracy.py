# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KdaInputProj 四个输出的端到端精度测试。

一次算子调用同时校验四路输出，golden 全部从 ``x`` 与权重在 CPU 上重算：

    qkv  = MX 量化矩阵乘(mx_quant(x), w_qkv, weight_qkv_scale)   BF16
    beta = sigmoid(x @ w_beta)                            FP32
    gate = x @ w_gate                                     BF16
    g    = x @ w_g                                        BF16

Stage1 的量化中间量 quantX / scaleX 留在 workspace 内部、不对外暴露，所以 qkv 这一路
是先在 CPU 上按 bit-exact 的 OCP 语义量化、再做 MX 矩阵乘得到 golden。量化若出错，qkv
会随之偏离而被抓到，代价是无法定位到具体哪一块 scale。

BF16 输出（qkv/gate/g）的判据：
  1. 与 CPU bf16 golden 逐 bit 相同的元素占比 >= 99.9%
  2. 相对 float64 参考的平均相对误差不超过 bf16 golden 自身基线的 1.05 倍
  3. >2ULP 的离群点数量受限，且每一个都必须落在发生抵消、幅值可忽略的位置

FP32 输出（beta）不要求逐 bit：设备侧 sigmoid 用向量 exp/div 实现，与 torch.sigmoid
存在几个 ULP 的差异，因此只要求相对 float64 的误差不显著劣于 FP32 golden 的基线。
"""

from __future__ import annotations

import os

import pytest
import torch
import torch_npu

try:
    from cann_ops_transformer_custom.ops import kda_input_proj as kda_input_proj_op  # noqa: F401
except ImportError:
    from cann_ops_transformer.ops import kda_input_proj as kda_input_proj_op  # noqa: F401

from kda_input_proj_bgg_golden import bgg_golden, bgg_golden_hp
from kda_input_proj_mx_quant_golden import mx_quant_golden
from kda_input_proj_qkv_golden import (
    bf16_ulp_diff,
    mean_rel_error,
    qkv_golden,
    qkv_golden_hp,
)

HIDDEN_SIZE = 7168
QKV_SIZE = 4608
BETA_SIZE = 12
GATE_SIZE = 1536
G_SIZE = 1536
MX_SCALE_GROUP_K = 64


_picked_device_id = None


def _pick_device_id() -> int:
    global _picked_device_id
    if _picked_device_id is not None:
        return _picked_device_id
    env = os.environ.get("TEST_DEVICE_ID")
    if env is not None:
        _picked_device_id = int(env)
        return _picked_device_id
    # 共享机上 device0 常被占满，TsdOpen 失败；按 0,5,6,... 探测第一个能 set_device 的卡。
    candidates = [0, 5, 6, 4, 3, 2, 1, 7]
    count = 0
    try:
        count = int(torch.npu.device_count())
    except Exception:
        count = 8
    for idx in candidates:
        if idx >= count:
            continue
        try:
            torch_npu.npu.set_device(idx)
            _picked_device_id = idx
            return idx
        except Exception:
            continue
    _picked_device_id = 0
    return _picked_device_id


def _device():
    return torch.device(f"npu:{_pick_device_id()}")


def _is_ascend950() -> bool:
    # get_device_name 不依赖 set_device / TsdOpen，避免占用卡导致整份 ST 被 skip。
    try:
        idx = int(os.environ.get("TEST_DEVICE_ID", "0"))
        return "Ascend950" in str(torch.npu.get_device_name(idx))
    except Exception:  # pragma: no cover
        try:
            return "Ascend950" in str(torch.npu.get_device_name(0))
        except Exception:
            return False


pytestmark = pytest.mark.skipif(
    not _is_ascend950(), reason="KdaInputProj only supports Ascend 950PR/950DT"
)


def _make_weights(
    scale_kind: str = "vary", trans_qkv: bool = False, trans_bgg: bool = False
):
    """默认权重按 [K, N] 连续存储（trans_weight_*=false），qkv scale 为 [ceil(K/64), N, 2]。

    trans_qkv/trans_bgg=True 时改为 Linear 的 [N, K] 存储、公开 view [K, N]
    （stride[-2]==1），由 aclnn 推断 trans=true。qkv scale 相应地变成 [N, ceil(K/64), 2]。
    """
    torch.manual_seed(1234)
    kg = HIDDEN_SIZE // MX_SCALE_GROUP_K
    if trans_qkv:
        weight_qkv = (
            (torch.randn(QKV_SIZE, HIDDEN_SIZE) * 0.3).to(torch.float8_e4m3fn).t()
        )
        scale_shape = (QKV_SIZE, kg, 2)
    else:
        weight_qkv = (torch.randn(HIDDEN_SIZE, QKV_SIZE) * 0.3).to(torch.float8_e4m3fn)
        scale_shape = (kg, QKV_SIZE, 2)
    if scale_kind == "const":
        weight_qkv_scale = torch.full(scale_shape, 1.0)
    else:
        exp = torch.randint(120, 132, scale_shape).float()
        weight_qkv_scale = torch.pow(2.0, exp - 127.0)
    # beta/gate/g 权重量级压小，避免 sigmoid 全部饱和到 0/1 而失去分辨力
    if trans_bgg:
        weight_beta = (
            (torch.randn(BETA_SIZE, HIDDEN_SIZE) * 0.02).to(torch.bfloat16).t()
        )
        weight_gate = (
            (torch.randn(GATE_SIZE, HIDDEN_SIZE) * 0.02).to(torch.bfloat16).t()
        )
        weight_g = (torch.randn(G_SIZE, HIDDEN_SIZE) * 0.02).to(torch.bfloat16).t()
    else:
        weight_beta = (torch.randn(HIDDEN_SIZE, BETA_SIZE) * 0.02).to(torch.bfloat16)
        weight_gate = (torch.randn(HIDDEN_SIZE, GATE_SIZE) * 0.02).to(torch.bfloat16)
        weight_g = (torch.randn(HIDDEN_SIZE, G_SIZE) * 0.02).to(torch.bfloat16)
    return (
        weight_qkv,
        weight_qkv_scale.to(torch.float8_e8m0fnu),
        weight_beta,
        weight_gate,
        weight_g,
    )


def _run(x_cpu, weights):
    weight_qkv, weight_qkv_scale, weight_beta, weight_gate, weight_g = weights
    device = _device()
    torch_npu.npu.set_device(device)
    outs = torch.ops.cann_ops_transformer.kda_input_proj(
        x_cpu.to(device),
        weight_qkv.to(device),
        weight_beta.to(device),
        weight_gate.to(device),
        weight_g.to(device),
        weight_qkv_scale=weight_qkv_scale.to(device),
    )
    torch.npu.synchronize()
    return [o.cpu() for o in outs]


def _assert_bf16_output(got, golden, golden_hp, name):
    """BF16 输出的共用判据（qkv / gate / g）。"""
    ulp = bf16_ulp_diff(got, golden)
    exact = (ulp == 0).float().mean().item()
    npu_err = mean_rel_error(got.float(), golden_hp)
    ref_err = mean_rel_error(golden.float(), golden_hp)

    assert exact >= 0.999, f"[{name}] 逐 bit 相同占比过低: {exact:.5f}"
    assert npu_err <= ref_err * 1.05, (
        f"[{name}] 平均相对误差 {npu_err:.4e} 超过 golden 基线 {ref_err:.4e} 的 1.05 倍"
    )

    outlier = ulp > 2
    n_outlier = int(outlier.sum())
    budget = max(4, int(got.numel() * 1e-4))
    assert n_outlier <= budget, f"[{name}] >2ULP 离群点过多: {n_outlier} > {budget}"
    if n_outlier:
        # 参考尺度必须按行取：输入分布可能让各行幅值相差几十个二进制量级（见 mixed 用例），
        # 用全局中位数会把大幅值行里的正常离群点误判成异常。
        mag = golden_hp.abs()
        row_scale = mag.median(dim=1, keepdim=True).values.expand_as(mag)
        assert (mag[outlier] < 0.01 * row_scale[outlier]).all(), (
            f"[{name}] 存在幅值不可忽略的 >2ULP 离群点"
        )
    print(
        f"  [{name}] exact={exact:.5f} outliers={n_outlier}/{budget} "
        f"rel_mean npu={npu_err:.4e} golden={ref_err:.4e}"
    )


def _assert_fp32_output(got, golden, golden_hp, name):
    """beta 走 FP32，sigmoid 的向量实现与 torch 有几个 ULP 差异，只比误差量级。"""
    npu_err = mean_rel_error(got.float(), golden_hp)
    ref_err = mean_rel_error(golden.float(), golden_hp)
    max_abs = (got.float() - golden.float()).abs().max().item()

    # beta 是 sigmoid 输出，落在 (0,1)，绝对误差和相对误差都该在 FP32 舍入量级
    assert max_abs <= 1e-5, f"[{name}] 与 FP32 golden 的最大绝对偏差过大: {max_abs:.3e}"
    assert npu_err <= max(ref_err * 4.0, 1e-6), (
        f"[{name}] 平均相对误差 {npu_err:.4e} 显著劣于 golden 基线 {ref_err:.4e}"
    )
    print(
        f"  [{name}] max_abs={max_abs:.3e} rel_mean npu={npu_err:.4e} golden={ref_err:.4e}"
    )


def _check_all(x, weights, tag, trans_qkv: bool = False):
    qkv, beta, gate, g = _run(x, weights)
    weight_qkv, weight_qkv_scale, weight_beta, weight_gate, weight_g = weights

    # qkv：CPU 上先按 bit-exact 语义量化，再做 MX 矩阵乘
    quant_bytes, scale_bytes = mx_quant_golden(x)
    quant_x = quant_bytes.view(torch.float8_e4m3fn)
    x_scale = scale_bytes.view(torch.float8_e8m0fnu)
    scale_kn = (
        weight_qkv_scale.permute(1, 0, 2).contiguous()
        if trans_qkv
        else weight_qkv_scale
    )
    qkv_g = qkv_golden(quant_x, x_scale, weight_qkv, scale_kn)
    qkv_hp = qkv_golden_hp(quant_x, x_scale, weight_qkv, scale_kn)
    _assert_bf16_output(qkv, qkv_g, qkv_hp, f"qkv[{tag}]")

    beta_g, gate_g, g_g = bgg_golden(x, weight_beta, weight_gate, weight_g)
    beta_hp, gate_hp, g_hp = bgg_golden_hp(x, weight_beta, weight_gate, weight_g)
    _assert_fp32_output(beta, beta_g, beta_hp, f"beta[{tag}]")
    _assert_bf16_output(gate, gate_g, gate_hp, f"gate[{tag}]")
    _assert_bf16_output(g, g_g, g_hp, f"g[{tag}]")


@pytest.mark.ci
def test_transposed_bgg_weights():
    """beta/gate/g 走 Linear [N,K] 存储，公开 [K,N] 转置 view。"""
    weights = _make_weights(trans_bgg=True)
    torch.manual_seed(16)
    x = torch.randn(16, HIDDEN_SIZE).to(torch.bfloat16)
    _check_all(x, weights, "trans_bgg")


@pytest.mark.ci
def test_transposed_qkv_weight():
    """qkv 权重 [N,K] 存储 + scale [N, K/64, 2]，覆盖 trans_weight_qkv=true。"""
    weights = _make_weights(trans_qkv=True)
    torch.manual_seed(17)
    x = torch.randn(16, HIDDEN_SIZE).to(torch.bfloat16)
    _check_all(x, weights, "trans_qkv", trans_qkv=True)


@pytest.mark.ci
def test_all_transposed_linear_nk_storage():
    """四路权重全部按 Linear [N,K] 存储，对应 tiling key 全 1。"""
    weights = _make_weights(trans_qkv=True, trans_bgg=True)
    torch.manual_seed(18)
    x = torch.randn(16, HIDDEN_SIZE).to(torch.bfloat16)
    _check_all(x, weights, "trans_all", trans_qkv=True)


# 225/240/256 曾因 QMM L1 回退失效触发 LOAD2D 越界，作为回归点固定下来。
@pytest.mark.ci
@pytest.mark.parametrize(
    "t_size", [1, 2, 7, 16, 64, 128, 129, 224, 225, 240, 256, 257, 512, 1000]
)
def test_all_outputs_shapes(t_size):
    """扫 T，四路输出同时校验。"""
    weights = _make_weights()
    torch.manual_seed(t_size * 7 + 1)
    x = torch.randn(t_size, HIDDEN_SIZE).to(torch.bfloat16)
    _check_all(x, weights, f"T={t_size}")


@pytest.mark.ci
@pytest.mark.parametrize("scale_kind", ["const", "vary"])
def test_qkv_weight_scale(scale_kind):
    """分别用恒定/变化的权重 scale，隔离验证 qkv 路 B 侧 MX scale 的排布。"""
    weights = _make_weights(scale_kind)
    torch.manual_seed(7)
    x = torch.randn(16, HIDDEN_SIZE).to(torch.bfloat16)
    _check_all(x, weights, f"wscale={scale_kind}")


def _make_weights_sized(k_size, n_qkv=32, n_beta=8, n_gate=16, n_g=16, seed=1234):
    """任意 K。scale 的 K 轴按 Ceil(K/64) 取整，权重本身不补零。"""
    torch.manual_seed(seed)
    kg = (k_size + MX_SCALE_GROUP_K - 1) // MX_SCALE_GROUP_K
    weight_qkv = (torch.randn(k_size, n_qkv) * 0.3).to(torch.float8_e4m3fn)
    exp = torch.randint(120, 132, (kg, n_qkv, 2)).float()
    weight_qkv_scale = torch.pow(2.0, exp - 127.0).to(torch.float8_e8m0fnu)
    weight_beta = (torch.randn(k_size, n_beta) * 0.02).to(torch.bfloat16)
    weight_gate = (torch.randn(k_size, n_gate) * 0.02).to(torch.bfloat16)
    weight_g = (torch.randn(k_size, n_g) * 0.02).to(torch.bfloat16)
    return weight_qkv, weight_qkv_scale, weight_beta, weight_gate, weight_g


@pytest.mark.ci
@pytest.mark.parametrize("t_size", [1, 4])
@pytest.mark.parametrize(
    "k_size", [1, 16, 31, 32, 33, 48, 63, 65, 96, 97, 127, 129, 256]
)
def test_arbitrary_k(t_size, k_size):
    """K 不是 32 的倍数时，尾块在 UB/L1 内补零，对外仍是真实 K。T>1 覆盖奇数 block 的行间 stride。"""
    weights = _make_weights_sized(k_size, seed=k_size * 17 + t_size)
    torch.manual_seed(k_size * 3 + t_size)
    x = torch.randn(t_size, k_size).to(torch.bfloat16)
    _check_all(x, weights, f"T={t_size},K={k_size}")


@pytest.mark.ci
@pytest.mark.parametrize("dist", ["normal", "tiny", "huge", "mixed"])
def test_input_distribution(dist):
    """不同量级的输入分布，覆盖量化 scale 的指数范围。"""
    weights = _make_weights()
    torch.manual_seed(4321)
    t_size = 64
    if dist == "normal":
        x = torch.randn(t_size, HIDDEN_SIZE)
    elif dist == "tiny":
        x = torch.randn(t_size, HIDDEN_SIZE) * 2.0**-30
    elif dist == "huge":
        x = torch.randn(t_size, HIDDEN_SIZE) * 2.0**30
    else:
        x = torch.randn(t_size, HIDDEN_SIZE) * torch.pow(
            2.0, torch.randint(-30, 30, (t_size, 1)).float()
        )
    _check_all(x.to(torch.bfloat16), weights, dist)
