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
MhcSinkhornBackward 算子 Golden 实现

算子原型：MhcSinkhornBackward（MhcSinkhorn 的反向算子）
接口：aclnnMhcSinkhornBackward(gradOutput, normOut, sumOut, out)
平台：Ascend 950PR&950DT (arch35)

================================================================================
数学语义（对应 docs/aclnnMhcSinkhornBackward.md 计算公式 与
op_kernel/arch35/mhc_sinkhorn_backward_simd.h 的 CalcColBackward / CalcRowBackward /
CalcSoftmaxBackward）
================================================================================

设 grad_curr 为当前梯度（n×n 矩阵，对 totalLength 个矩阵逐个独立计算），
norm_out[k] 为正向第 k 次归一化保存的方向矩阵，sum_out[k] 为对应缩放系数。
反向按 k 从大到小回溯，与 kernel Process() 的执行顺序一致：

  主循环 iter = num_iters-1 .. 1（每轮先列反向 k=2*iter+1、再行反向 k=2*iter）:
    k = 2*iter+1（列反向, CalcColBackward）:
        dot[c]   = Σ_r  grad_curr[r][c] * norm_out[k][r][c]        # sum over dim=-2
        grad_curr = (grad_curr - dot) / sum_out[k][c]
    k = 2*iter（行反向, CalcRowBackward）:
        dot[r]   = Σ_c  grad_curr[r][c] * norm_out[k][r][c]        # sum over dim=-1
        grad_curr = (grad_curr - dot) / sum_out[k][r]

  收尾处理（主循环结束后，对应 kernel 的 CalcCol(offset=1) + CalcSoftmax，
  即 k=1 / k=0，不计入主循环 iter）:
    k = 1（列反向）:
        dot[c]   = Σ_r  grad_curr[r][c] * norm_out[1][r][c]
        grad_curr = (grad_curr - dot) / sum_out[1][c]
    k = 0（softmax 行反向）:
        dot[r]   = Σ_c  grad_curr[r][c] * norm_out[0][r][c]
        grad_input = (grad_curr - dot) * norm_out[0]               # 乘 norm, 不再除 sum

================================================================================
norm / sum 一维内存布局（从 kernel 偏移公式推导）
================================================================================

op_host/mhc_sinkhorn_backward_tiling.cpp 通过 mhc_sinkhorn_base_tiling.cpp 的
BaseSplitCores 与正向对齐分核（取 forward/backward 较小 tUbFactor），
norm/sum 的布局与 tiling（核切分/循环切分）相关：

  - 当 totalLength(=B*S 或 T) ≤ AIV 核数时，每核处理 1 个 T、核内单循环
    （perloopSize=1），布局唯一确定为：
        norm_flat[(t * 2*num_iters + k) * n * align_n + r * align_n + c]
            = norm_out[k][t][r][c]        # 即 [T][k][r][c]
        sum_flat[(t * 2*num_iters + k) * align_n + c]
            = sum_out[k][t][c]            # 即 [T][k][c]
    其中 align_n 固定为 8（n≤8 按 FP32 32B 块对齐），c ∈ [0, align_n)，
    仅 c < n 的列参与计算（kernel 中通过 MASK_4/6/8 屏蔽 padding 列）。
  - 本 golden 通过断言 totalLength ≤ MAX_TOTAL_LENGTH_FOR_LAYOUT 保证该前提。
    ST 用例（tests/st/arch35/ttk_kernel_mhc_sinkhorn_backward_st.csv）中所有
    totalLength 均 ≤ 5，满足前提。

num_iters 无独立属性，由 norm 的 shape 反推（与 op_host tiling 一致）：
    num_iters = norm.numel() / (2 * totalLength * n * align_n)

约束（op_host/mhc_sinkhorn_backward_tiling.cpp CheckShape/CheckDtype）：
    - grad_y 为 3 维 (T,n,n) 或 4 维 (B,S,n,n)，末两维相等
    - n ∈ {4, 6, 8}
    - num_iters ∈ [1, 100]
    - grad_input 形状与 grad_y 完全一致
    - 全部张量 dtype 为 float32
================================================================================
"""

import numpy as np
import torch

__golden__ = {
    "kernel": {"mhc_sinkhorn_backward": "mhc_sinkhorn_backward_golden"},
    "aclnn": {"aclnnMhcSinkhornBackward": "aclnn_mhc_sinkhorn_backward_golden"},
}

# align_n：n 按 FP32 32 字节块对齐后的值，固定为 8（n 仅支持 4/6/8）
ALIGN_N = 8
# n 的合法取值（op_host/mhc_sinkhorn_backward_tiling.cpp）
VALID_N = (4, 6, 8)
# norm/sum 布局前提：totalLength 不超过该值时，每核 1 个 T、核内单循环，
# 布局为 [T][k][r][c] / [T][k][c]（950 系列 AIV 核数远大于该值）
MAX_TOTAL_LENGTH_FOR_LAYOUT = 32


def _to_torch(t):
    """numpy.ndarray / torch.Tensor -> torch.Tensor，保留浮点 dtype（自验可用 float64 对拍）"""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach()
    return torch.from_numpy(np.ascontiguousarray(t))


def _to_numpy(t):
    """torch.Tensor / numpy.ndarray -> numpy.ndarray（float32）"""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _check_and_parse(grad_y, norm, sum_):
    """校验输入约束并解析 totalLength / n / num_iters（与 op_host tiling 校验一致）"""
    if grad_y.dim() not in (3, 4):
        raise ValueError(
            "grad_y must be 3-dim (T,n,n) or 4-dim (B,S,n,n), got {}-dim".format(
                grad_y.dim()
            )
        )
    n = grad_y.shape[-1]
    if grad_y.shape[-2] != n:
        raise ValueError(
            "last two dims of grad_y must be equal, got {}".format(tuple(grad_y.shape))
        )
    if n not in VALID_N:
        raise ValueError(
            "n (last dim of grad_y) must be one of {}, got {}".format(VALID_N, n)
        )

    total_length = int(grad_y.numel() // (n * n))
    if total_length > MAX_TOTAL_LENGTH_FOR_LAYOUT:
        raise ValueError(
            "totalLength={} exceeds MAX_TOTAL_LENGTH_FOR_LAYOUT={}: the [T][k][r][c] "
            "layout of norm/sum is only guaranteed when totalLength <= AIV core num "
            "(1 T per core, single loop per core)".format(
                total_length, MAX_TOTAL_LENGTH_FOR_LAYOUT
            )
        )

    # norm 仅支持 1 维（对齐 op_host tiling CheckShape: NORM_INPUT_DIMNUM == 1）
    if norm.dim() != 1:
        raise ValueError(
            "norm must be 1-dim (2*num_iters*n*align_n*totalLength), got {}-dim".format(
                norm.dim()
            )
        )
    norm_numel = int(norm.numel())
    if norm_numel % 2 != 0:
        raise ValueError("numel of norm must be even, got {}".format(norm_numel))
    denom = 2 * total_length * n * ALIGN_N
    if norm_numel % denom != 0:
        raise ValueError(
            "numel of norm ({}) must be a multiple of 2*totalLength*n*align_n ({})".format(
                norm_numel, denom
            )
        )
    num_iters = norm_numel // denom
    if num_iters < 1 or num_iters > 100:
        raise ValueError("num_iters must be in [1, 100], got {}".format(num_iters))

    # sum 仅支持 1 维（对齐 op_host tiling CheckShape: SUM_INPUT_DIMNUM == 1）
    if sum_.dim() != 1:
        raise ValueError(
            "sum must be 1-dim (2*num_iters*align_n*totalLength), got {}-dim".format(
                sum_.dim()
            )
        )
    sum_numel = int(sum_.numel())
    if sum_numel != 2 * num_iters * total_length * ALIGN_N:
        raise ValueError(
            "numel of sum ({}) must be 2*num_iters*align_n*totalLength ({})".format(
                sum_numel, 2 * num_iters * total_length * ALIGN_N
            )
        )
    return total_length, n, num_iters


def _sinkhorn_backward_core(grad_y, norm, sum_):
    """MhcSinkhornBackward 数学核心（torch, float32）。

    Args:
        grad_y: (T, n, n) / (B, S, n, n) 初始梯度
        norm:   1-D, 布局 [T][k][r][c]，k ∈ [0, 2*num_iters)
        sum_:   1-D, 布局 [T][k][c]
    Returns:
        grad_input: 与 grad_y 同形状
    """
    total_length, n, num_iters = _check_and_parse(grad_y, norm, sum_)

    norm_v = norm.reshape(total_length, 2 * num_iters, n, ALIGN_N)
    sum_v = sum_.reshape(total_length, 2 * num_iters, ALIGN_N)
    g = grad_y.reshape(total_length, n, n).clone()

    def _col_backward(cur, k):
        # 列反向: dot[c] = Σ_r g[r][c]*norm[k][r][c]; g ← (g - dot) / sum[k][c]
        dot = (cur * norm_v[:, k, :, :n]).sum(dim=1, keepdim=True)  # (T,1,n)
        return (cur - dot) / sum_v[:, k, :n].reshape(-1, 1, n)

    def _row_backward(cur, k, use_sum=True):
        # 行反向: dot[r] = Σ_c g[r][c]*norm[k][r][c]
        dot = (cur * norm_v[:, k, :, :n]).sum(dim=2, keepdim=True)  # (T,n,1)
        if use_sum:
            # g ← (g - dot) / sum[k][r]
            return (cur - dot) / sum_v[:, k, :n].reshape(-1, n, 1)
        # 最后一次迭代(softmax): g ← (g - dot) * norm[0][r][c]
        return (cur - dot) * norm_v[:, k, :, :n]

    # 主循环 iter = num_iters-1 .. 1，每轮先列反向 k=2*i+1、再行反向 k=2*i（kernel Process）
    for i in range(num_iters - 1, 0, -1):
        g = _col_backward(g, 2 * i + 1)
        g = _row_backward(g, 2 * i)

    # 收尾处理（主循环之后，不计入 iter）：列反向 k=1 + softmax 行反向 k=0（乘 norm 不除 sum）
    g = _col_backward(g, 1)
    g = _row_backward(g, 0, use_sum=False)

    return g.reshape(grad_y.shape)


def mhc_sinkhorn_backward_golden(grad_y, norm, sum_, **kwargs):
    """MhcSinkhornBackward kernel 级标杆函数。

    参数顺序与算子输入一一对应：grad_y, norm, sum；返回 grad_input。
    所有输入/输出均为 numpy.ndarray（float32）。
    """
    grad_y_t = _to_torch(grad_y)
    norm_t = _to_torch(norm)
    sum_t = _to_torch(sum_)
    grad_input = _sinkhorn_backward_core(grad_y_t, norm_t, sum_t)
    return _to_numpy(grad_input)


def aclnn_mhc_sinkhorn_backward_golden(gradOutput, normOut, sumOut, out, **kwargs):
    """MhcSinkhornBackward aclnn 级标杆函数。

    参数名与顺序对应 aclnnMhcSinkhornBackwardGetWorkspaceSize
    （aclnn_mhc_sinkhorn_backward.h）中的 gradOutput/normOut/sumOut/out，
    不含 workspaceSize 与 executor。返回输出梯度 grad_input（torch.Tensor）。
    """
    grad_input_np = mhc_sinkhorn_backward_golden(gradOutput, normOut, sumOut)
    return torch.from_numpy(grad_input_np)


def _naive_reference(grad_y, norm, sum_):
    """朴素逐步实现（纯 Python 标量循环），用于对拍验证向量化核心的正确性。

    与 _sinkhorn_backward_core 相互独立，逐矩阵、逐元素实现同一递推公式。
    """
    grad_y = _to_numpy(grad_y).astype(np.float64)
    norm = _to_numpy(norm).astype(np.float64)
    sum_ = _to_numpy(sum_).astype(np.float64)
    total_length, n, num_iters = _check_and_parse(
        torch.as_tensor(grad_y), torch.as_tensor(norm), torch.as_tensor(sum_)
    )
    norm_v = norm.reshape(total_length, 2 * num_iters, n, ALIGN_N)
    sum_v = sum_.reshape(total_length, 2 * num_iters, ALIGN_N)
    out = np.zeros_like(grad_y).reshape(total_length, n, n)
    for t in range(total_length):
        g = [
            [float(grad_y.reshape(total_length, n, n)[t][r][c]) for c in range(n)]
            for r in range(n)
        ]
        for i in range(num_iters - 1, 0, -1):
            k = 2 * i + 1  # 列反向
            for c in range(n):
                dot = sum(g[r][c] * norm_v[t][k][r][c] for r in range(n))
                for r in range(n):
                    g[r][c] = (g[r][c] - dot) / float(sum_v[t][k][c])
            k = 2 * i  # 行反向
            for r in range(n):
                dot = sum(g[r][c] * norm_v[t][k][r][c] for c in range(n))
                for c in range(n):
                    g[r][c] = (g[r][c] - dot) / float(sum_v[t][k][r])
        k = 1  # 最后一次迭代：列反向
        for c in range(n):
            dot = sum(g[r][c] * norm_v[t][k][r][c] for r in range(n))
            for r in range(n):
                g[r][c] = (g[r][c] - dot) / float(sum_v[t][k][c])
        k = 0  # 最后一次迭代：softmax 行反向（乘 norm）
        for r in range(n):
            dot = sum(g[r][c] * norm_v[t][k][r][c] for c in range(n))
            for c in range(n):
                out[t][r][c] = (g[r][c] - dot) * float(norm_v[t][k][r][c])
    return out.reshape(grad_y.shape)


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # 自验：
    #   1) 遍历 ST CSV 全部用例配置（shape / num_iters 组合），float64 下
    #      向量化核心与朴素标量参考实现逐元素对拍（验证递推逻辑/轴序一致，
    #      阈值 1e-9）；float32 路径（真实 golden 输出）对 float64 结果的
    #      偏差按 ops-precision-standard 的 FLOAT32 max_abs_error_limit(1e-2) 校验；
    #   2) 线性性检查（sinkhorn 反向对 grad_y 为线性映射，float64 下 1e-9）；
    #   3) num_iters=1 边界与 n=4/6/8 全覆盖。
    # ---------------------------------------------------------------------------
    torch.manual_seed(0)
    np.random.seed(0)

    # (名称, grad_y shape, num_iters) —— 与 ttk_kernel_mhc_sinkhorn_backward_st.csv 一致
    cases = [
        ("test_mhc_sinkhorn_backward_0001", (4, 4, 4), 2),
        ("test_mhc_sinkhorn_backward_0002", (1, 3, 6, 6), 3),
        ("test_mhc_sinkhorn_backward_0003", (2, 8, 8), 1),
        ("test_mhc_sinkhorn_backward_0004", (2, 2, 4, 4), 4),
        ("test_mhc_sinkhorn_backward_0005", (5, 6, 6), 2),
        ("test_mhc_sinkhorn_backward_0006", (1, 1, 8, 8), 3),
    ]

    all_pass = True
    for name, shape, iters in cases:
        gy32 = torch.empty(shape).uniform_(-10.0, 10.0)
        total_length = int(gy32.numel() // (shape[-1] * shape[-1]))
        n = shape[-1]
        norm32 = torch.empty(2 * iters * total_length * n * ALIGN_N).uniform_(0.0, 1.0)
        sm32 = torch.empty(2 * iters * total_length * ALIGN_N).uniform_(0.5, 2.0)

        # float64 逻辑对拍：向量化 vs 朴素标量
        gy64, norm64, sm64 = gy32.double(), norm32.double(), sm32.double()
        out_vec64 = _sinkhorn_backward_core(gy64, norm64, sm64)
        out_naive64 = torch.from_numpy(
            _naive_reference(gy64.numpy(), norm64.numpy(), sm64.numpy())
        )
        logic_err = (out_vec64 - out_naive64).abs().max().item()

        # float32 输出 vs float64 参考（舍入噪声，上限按 FLOAT32 max_abs_error_limit）
        out_vec32 = _sinkhorn_backward_core(gy32, norm32, sm32)
        noise_err = (out_vec32.double() - out_vec64).abs().max().item()

        # 线性性(float64): F(a*g1 + b*g2) = a*F(g1) + b*F(g2)
        g1 = torch.empty(shape).uniform_(-10.0, 10.0).double()
        g2 = torch.empty(shape).uniform_(-10.0, 10.0).double()
        lin_err = (
            (
                _sinkhorn_backward_core(0.3 * g1 + 1.7 * g2, norm64, sm64)
                - (
                    0.3 * _sinkhorn_backward_core(g1, norm64, sm64)
                    + 1.7 * _sinkhorn_backward_core(g2, norm64, sm64)
                )
            )
            .abs()
            .max()
            .item()
        )

        ok = (
            logic_err < 1e-9
            and lin_err < 1e-9
            and noise_err < 1e-2
            and out_vec32.shape == gy32.shape
        )
        all_pass = all_pass and ok
        print(
            "[{}] shape={} num_iters={} totalLength={} "
            "| logic_err={:.3e} linear_err={:.3e} fp32_noise={:.3e} -> {}".format(
                name,
                tuple(shape),
                iters,
                total_length,
                logic_err,
                lin_err,
                noise_err,
                "PASS" if ok else "FAIL",
            )
        )

    # aclnn 级入口自验（签名与 aclnnMhcSinkhornBackward 一一对应）
    gy = torch.empty(1, 2, 4, 4).uniform_(-10.0, 10.0)
    norm = torch.empty(2 * 2 * 2 * 4 * ALIGN_N).uniform_(0.0, 1.0)
    sm = torch.empty(2 * 2 * 2 * ALIGN_N).uniform_(0.5, 2.0)
    out_aclnn = aclnn_mhc_sinkhorn_backward_golden(gy, norm, sm, torch.empty_like(gy))
    assert isinstance(out_aclnn, torch.Tensor) and out_aclnn.shape == gy.shape
    print("[aclnn entry] signature ok, out shape = {}".format(tuple(out_aclnn.shape)))

    print("SELF-TEST {}".format("PASSED" if all_pass else "FAILED"))
    raise SystemExit(0 if all_pass else 1)
