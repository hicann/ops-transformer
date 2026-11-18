# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttnGrad PyPTO Pro JIT entry.

Layout:
  pipeline/ -- setup, preload loops and synchronization
  compute/ -- Cube and Vector block computations
  memory/ -- on-chip addresses and tile declarations
  schedule/ -- task selection and RunInfo construction
  schedule/deterministic/ -- state, shared formulas and family decoders
  schedule/deterministic/tnd/ -- device-side TND metadata consumers

Keep exactly one @pl.jit kernel visible to pypto_codegen.py.
FlashAttnGradTilingData/Key and the ping-pong arrays remain exported here.
The entry selects preload3 for BN2GS1S2 with d_align <= 192, else preload2.
TND planning remains in the metadata operator; these modules only consume it.
"""

import os
import sys

import pypto_pro.language as pl

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)

from pipeline.fag_kernel import (
    flash_attn_grad_preload2,
    flash_attn_grad_preload3,
)
from fag_tiling import FlashAttnGradTilingData, FlashAttnGradTilingKey, Tmpl

# 这四个 flag 是**按 taskId 奇偶下标取**的，故必须在运行期真的存在一个数组。
# codegen 只把「入口函数闭包里」的 list 提升成一个具名全局数组；若只在
# fag_kernel 的模块作用域里可见，就会退化成每次用到时在循环体内现场构造一个
# 匿名数组。功能一样，但白白多出一段循环内的初始化，故在这里显式引入。
# 其余标量 flag 会被折成立即数，不需要这样。
from fag_common import (
    SYNC_C1_TO_V2_FLAG,
    SYNC_C2_TO_V2_FLAG,
    SYNC_V2_TO_C1_FLAG,
    SYNC_V2_TO_C2_FLAG,
)

# re-export：kernel 自身不用，但 JIT 脚本按老路径 import。
from fag_common import CUBE_BASEM, CUBE_BASEN, TILE_D, TILE_M, TILE_N


@pl.jit(
    auto_mutex=True,
    tiling_key=FlashAttnGradTilingKey,
    datatype={
        "q": "input_dtype",
    },
)
def flash_attn_grad(
    # 13 个输入，顺序严格对齐 op_host/flash_attn_grad_def.cpp 的 Input() 声明
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    dsink: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
):
    """入口只做编译期分发：template / d_align 来自 tilingkey。

    preload3：仅 ``Tmpl.BN2GS1S2`` 且 D<=192，``runInfos[3]``。
    preload2：其余（``Tmpl.BN2`` 与 D=256/768 GS1S2），1-ahead，``runInfos[2]``。
    """
    if template == Tmpl.BN2GS1S2 and d_align <= 192:
        flash_attn_grad_preload3(
            q,
            k,
            v,
            dout,
            attn_out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
            sinks,
            attn_mask,
            metadata,
            dq,
            dk,
            dv,
            dsink,
            workspace,
            tiling,
            d_align,
            dv_align,
            swizzle,
            deter,
            sink,
        )
    else:
        flash_attn_grad_preload2(
            q,
            k,
            v,
            dout,
            attn_out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
            attn_mask,
            metadata,
            sinks,
            dq,
            dk,
            dv,
            dsink,
            workspace,
            tiling,
            d_align,
            dv_align,
            swizzle,
            deter,
            sink,
        )
