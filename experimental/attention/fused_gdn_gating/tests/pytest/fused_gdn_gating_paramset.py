# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools

import torch

# 测试参数取值范围（覆盖numHeads/batch/dtype/beta/threshold全组合）
NUM_HEADS_VALUES = [4, 6, 8, 12, 16, 24, 32, 48, 64, 128]
BATCH_SIZES = [1, 7, 37, 128, 512, 4096, 16384]
DTYPES = [torch.bfloat16, torch.float16]
PARAM_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
DTYPE_COMBINATIONS = [
    (dtype, param_dtype) for dtype in DTYPES for param_dtype in PARAM_DTYPES
]

_DTYPE_NAME = {
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
}


def _dtype_name(dtype):
    return _DTYPE_NAME.get(dtype, str(dtype))


def _make_case(
    case_id,
    testcase_name,
    num_heads,
    batch,
    dtype,
    param_dtype,
    beta=1.0,
    threshold=20.0,
    force_threshold=False,
    soc="all",
):
    return {
        "case_id": case_id,
        "testcase_name": testcase_name,
        "num_heads": num_heads,
        "batch": batch,
        "dtype": dtype,
        "param_dtype": param_dtype,
        "beta": beta,
        "threshold": threshold,
        "force_threshold": force_threshold,
        "soc": soc,
        "expected_result": "PASS",
    }


# ---------------------------------------------------------------------------
# Group 1: 核心正确性 — num_heads × batch × dtype 组合（对齐 910B 用例）
#   - BF16 dtype 标记 soc="910b"，在 310P 上自动跳过
#   - FP16 dtype 标记 soc="all"，310P 和 910B 均可运行
# ---------------------------------------------------------------------------
GROUP_CORE = [
    _make_case(
        f"FGG_CORE_{i:03d}",
        f"core_nh{nh}_b{b}_{_dtype_name(dt)}",
        nh,
        b,
        dt,
        torch.float32,
        soc="910b" if dt == torch.bfloat16 else "all",
    )
    for i, (nh, b, dt) in enumerate(
        itertools.product(NUM_HEADS_VALUES, BATCH_SIZES, DTYPES)
    )
]

# ---------------------------------------------------------------------------
# Group 2: 非默认参数 — beta=0.5, threshold=1.0, 注入边界值
# ---------------------------------------------------------------------------
GROUP_NONDEF = [
    _make_case(
        f"FGG_NONDEF_{i:03d}",
        f"nondef_nh{nh}_b{b}",
        nh,
        b,
        torch.bfloat16,
        torch.float32,
        beta=0.5,
        threshold=1.0,
        force_threshold=True,
        soc="910b",
    )
    for i, (nh, b) in enumerate(itertools.product([16, 32, 64], [1, 37]))
]

# ---------------------------------------------------------------------------
# Group 3: dtype 矩阵 — (a/b dtype) × (A_log/dt_bias dtype) 组合
# ---------------------------------------------------------------------------
GROUP_DTYPE = [
    _make_case(
        f"FGG_DTYPE_{i:03d}",
        f"dtype_{_dtype_name(dt)}_{_dtype_name(pdt)}",
        32,
        37,
        dt,
        pdt,
        threshold=2.0,
        force_threshold=True,
        soc="910b" if dt == torch.bfloat16 or pdt == torch.bfloat16 else "all",
    )
    for i, (dt, pdt) in enumerate(DTYPE_COMBINATIONS)
]

# ---------------------------------------------------------------------------
# Group 4: 大 batch 多行处理
# ---------------------------------------------------------------------------
GROUP_LARGE = [
    _make_case(
        f"FGG_LARGE_{i:03d}",
        f"large_nh{nh}_b{b}",
        nh,
        b,
        torch.bfloat16,
        torch.float32,
        soc="910b",
    )
    for i, (nh, b) in enumerate(
        itertools.product([8, 16, 32, 64], [64, 256, 1024, 4096])
    )
]

# ---------------------------------------------------------------------------
# Group 5: Bulk DMA 对齐（nh % 16 == 0 快速路径）
# ---------------------------------------------------------------------------
GROUP_BULK = [
    _make_case(
        f"FGG_BULK_{i:03d}",
        f"bulk_nh{nh}",
        nh,
        512,
        torch.bfloat16,
        torch.float32,
        soc="910b",
    )
    for i, nh in enumerate([16, 32, 48])
]

# ---------------------------------------------------------------------------
# Group 6: 非 Bulk DMA 回退（nh % 16 != 0 逐行回退路径）
# ---------------------------------------------------------------------------
GROUP_FALLBACK = [
    _make_case(
        f"FGG_FALLBACK_{i:03d}",
        f"fallback_nh{nh}",
        nh,
        256,
        torch.bfloat16,
        torch.float32,
        soc="910b",
    )
    for i, nh in enumerate([6, 12, 24])
]

# ---------------------------------------------------------------------------
# Group 7: 小 batch 优化（batch < rows_per_iter，自适应 UB 预算）
# ---------------------------------------------------------------------------
GROUP_SMALL = [
    _make_case(
        "FGG_SMALL_000",
        "small_batch",
        32,
        8,
        torch.bfloat16,
        torch.float32,
        soc="910b",
    )
]

# ---------------------------------------------------------------------------
# Group 8: 极端大 batch 压力测试
# ---------------------------------------------------------------------------
GROUP_EXTREME = [
    _make_case(
        "FGG_EXTREME_000",
        "extreme_large",
        32,
        65536,
        torch.bfloat16,
        torch.float32,
        soc="910b",
    )
]

# ---------------------------------------------------------------------------
# Group 9: 310P 专用 — FP16 only, 多种 beta/threshold 组合（对齐 310P 用例）
#   - 在 310P 上验证 310P 内核路径
#   - 在 910B 上同样可运行，验证 910B 内核路径
# ---------------------------------------------------------------------------
GROUP_310P = [
    _make_case(
        f"FGG_310P_{i:03d}",
        f"310p_b{b}_nh{nh}_beta{beta}_thr{thr}",
        nh,
        b,
        torch.float16,
        torch.float16,
        beta=beta,
        threshold=thr,
        force_threshold=True,
        soc="all",
    )
    for i, (b, nh, beta, thr) in enumerate(
        itertools.product(
            [1, 7, 37, 128, 512], [4, 8, 16, 32, 64], [0.5, 1.0], [1.0, 20.0]
        )
    )
]

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
ENABLED_PARAMS = (
    GROUP_CORE
    + GROUP_NONDEF
    + GROUP_DTYPE
    + GROUP_LARGE
    + GROUP_BULK
    + GROUP_FALLBACK
    + GROUP_SMALL
    + GROUP_EXTREME
    + GROUP_310P
)
