/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_scatter_tiling_data.h
 * \brief MRS 统一 tiling data, urma 与 hcomm 两条通信通路的 tiling 并列定义
 *
 * urma 通路: 先 matmul 计算, 再 AllToAll PUT 和归约 (QuantMatmulReduceScatterTilingData)
 * hcomm 通路: HCCL/CCU 通信 (hcommMatmulReduceScatterV2TilingData)
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "../../../tiling/matmul_tiling/quant_matmul_tiling_data.h"
#include "../../../tiling/tiling_data_struct_define.h"
#include "../../../core/aiv_comm/collective_comm_context.h"

// 封装 UDMA 和 UBMEM 上下文, 使 kernel 只需接收一个指针
namespace Apace {
namespace AivComm {
struct CommContext {
    CommUdmaContext udmaCtx;
    CommUbmemContext ubmemCtx;
};
} // namespace AivComm
} // namespace Apace

using Apace::AivComm::CommContext;

// urma 通路的 tiling: 通信数据类型为 CType (bfloat16), 不通信 ScaleA.
// 头块和尾块分别保存 QBMM tiling, blockDim 使用头块 tiling 的 usedCoreNum.
struct QuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData; // C 矩阵 AllToAll 通信切分(M 轴), splitAxisTailCnt 仅支持 0 或 1
    QuantMatmulTilingData tileQbmmTilingData; // 头块 QBMM tiling, m = rankSize * headMSize
    // 尾块 QBMM tiling, m = rankSize * paddedTailMSize; 无尾块时全零, 通信和归约仍使用真实 tailMSize
    QuantMatmulTilingData tailQbmmTilingData;
};

// hcomm 通路 tiling: 头/尾两份 QBMM tiling 描述单个通信片的组装逻辑 M (sliceM * rankDim).
// QBMM kernel 在 rank 边界切开调度块, 每个子块映射到各 rank 段内相同的 local-M 区间.
// CommTilingData 描述通信片如何覆盖各 rank 的输出 M 范围, nonSplitAxisSize 即输出 N 维.
// 尾块单独保存 tiling, 使 Matmul 与 ReduceScatter 可以流水执行,
// 不必先算完整 [rankM, N] 的 Matmul 再开始通信.
// 临时 C 按 [slice][rank][sliceM][N] 布局, 保证每份 ReduceScatter 输入连续; 算子输入仍按 rank 主序.
struct hcommMatmulReduceScatterV2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    CommTilingData commTilingData;
    QuantMatmulTilingData tileQbmmTilingData;
    QuantMatmulTilingData tailQbmmTilingData;
};
