# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttnGrad kernel 入口 (PyPTO Pro)。

只做**编译期分发**：把 tilingkey 的 ``template`` 位翻成一个模板主循环。
实现按职责拆在同目录的 ``fag_*`` 模块：

===================  ============================================
模块                  职责
===================  ============================================
``fag_common``       基本块尺寸、同步 flag、ConstInfo/RunInfo
``fag_mem``          L1/L0/UB 地址与 mutex id
``fag_tiling``       host 契约（TilingData / TilingKey）
``fag_schedule``     块有效性、swizzle、RunInfo
``fag_buffers``      片上 tile 声明
``fag_block_cube``   C1..C5 五个 matmul
``fag_block_vec``    V1..V6 与 pre/post 编排
``fag_vector_api``   寄存器级 VF 微内核
``attenmask``        mask 搬入与 softmax VF
``fag_kernel``       两个模板的主循环
===================  ============================================

本文件必须保持「有且仅有一个 ``@pl.jit``」：codegen
(``cmake/scripts/pypto_codegen.py``) 靠这一点在模块里找到 kernel。
:class:`FlashAttnGradTilingData` / :class:`FlashAttnGradTilingKey` 在这里
re-export，供 JIT 脚本按老路径 import。模板 / 对齐 / swizzle 判据只在
host（``op_host/plan/*``）里实现，不再提供 Python 镜像。
"""

import os
import sys

import pypto_pro.language as pl

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)

from fag_kernel import flash_attn_grad_bn2, flash_attn_grad_bn2gs1s2
from fag_tiling import FlashAttnGradTilingData, FlashAttnGradTilingKey

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
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
):
    """入口只做编译期分发：template 来自 tilingkey。

    0=BN2GS1S2，1=BN2（含 MultiBlk）。3 预留给 BN2S2，本轮 is_valid 裁掉。
    BN2 不接 GQA / D=192。
    """
    if template == 1:
        flash_attn_grad_bn2(
            q,
            k,
            v,
            dout,
            attn_out,
            softmax_lse,
            attn_mask,
            metadata,
            dq,
            dk,
            dv,
            workspace,
            tiling,
            d_align,
        )
    else:
        flash_attn_grad_bn2gs1s2(
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
            workspace,
            tiling,
            d_align,
            dv_align,
            swizzle,
        )
