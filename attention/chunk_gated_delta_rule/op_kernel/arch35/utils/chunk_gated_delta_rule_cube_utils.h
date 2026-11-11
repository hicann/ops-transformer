/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "chunk_gated_delta_rule_utils_arch35.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// FIX 写 UB 的行主序配置；第二个配置位 isToUB=true，表示目标为 UB。
static constexpr FixpipeConfig CGDR_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

// 无状态的 Cube 指令辅助函数：只描述 tensor、shape 和 stride，不依赖 head/chunk 或缓冲槽。
// 调用方负责 Lock/Unlock、跨核握手和缓冲生命周期；这里不隐含任何同步。

// MTE2：GM 行主序 [rows,cols] -> L1 NZ；两个行跨度以元素/逻辑行数传入。
__aicore__ inline void CopyGmToL1(GlobalTensor<bfloat16_t> gm, LocalTensor<bfloat16_t> l1, uint32_t rows, uint32_t cols,
                                  uint32_t srcRowStride, uint32_t dstRows)
{
    Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = rows;
    params.dValue = cols;
    params.srcDValue = srcRowStride;
    params.dstNzC0Stride = CeilAlign(dstRows, MM_BLOCK_CUBE);
    params.dstNzNStride = 1;
    params.srcNdMatrixStride = 0;
    params.dstNzMatrixStride = 0;
    DataCopy(l1, gm, params);
}

// MTE1：从 L1 NZ 的 (rowStart,colStart) 装入 [rows,cols]，按模板选择 L0 装载转置。
template <bool ifTranspose>
__aicore__ inline void CopyL1ToL0(LocalTensor<bfloat16_t> l0, LocalTensor<bfloat16_t> l1, uint32_t rows, uint32_t cols,
                                  uint32_t fullRows, uint32_t rowStart = 0, uint32_t colStart = 0)
{
    LoadData2DParamsV2 params;
    params.mStartPosition = rowStart / MM_BLOCK_CUBE;
    params.kStartPosition = colStart / MM_NZ_FRACTAL;
    params.mStep = CeilAlign(rows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
    params.kStep = CeilAlign(cols, MM_NZ_FRACTAL) / MM_NZ_FRACTAL;
    params.srcStride = CeilAlign(fullRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
    params.dstStride = ifTranspose ? params.kStep : params.mStep;
    params.ifTranspose = ifTranspose;
    LoadData(l0, l1, params);
}

// MTE1：L1 NZ 的 [rows,cols] 行片 -> L0A 指定行片；行跨度及起点均以元素传入。
__aicore__ inline void CopyRowsL1ToL0A(LocalTensor<bfloat16_t> l0a, LocalTensor<bfloat16_t> l1, uint32_t rows,
                                       uint32_t cols, uint32_t srcFullRows, uint32_t srcRowStart, uint32_t dstFullRows,
                                       uint32_t dstRowStart)
{
    LoadData2DParamsV2 params;
    params.mStartPosition = srcRowStart / MM_BLOCK_CUBE;
    params.kStartPosition = 0;
    params.mStep = CeilAlign(rows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
    params.kStep = CeilAlign(cols, MM_NZ_FRACTAL) / MM_NZ_FRACTAL;
    params.srcStride = CeilAlign(srcFullRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
    params.dstStride = CeilAlign(dstFullRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
    params.ifTranspose = false;
    LoadData(l0a[dstRowStart * MM_NZ_FRACTAL], l1, params);
}

// M：C[m,n] = A[m,k] @ B[k,n]；init=false 时加到已有 FP32 C，不插入量化。
__aicore__ inline void Mmad(LocalTensor<float> l0c, LocalTensor<bfloat16_t> l0a, LocalTensor<bfloat16_t> l0b,
                            uint32_t m, uint32_t n, uint32_t k, bool init = true)
{
    MmadParams params;
    params.m = m < MM_BLOCK_CUBE ? MM_BLOCK_CUBE : m;
    params.n = n;
    params.k = k;
    params.cmatrixInitVal = init;
    params.cmatrixSource = false;
    AscendC::Mmad(l0c, l0a, l0b, params);
}

template <typename outputType>
__aicore__ inline void CopyResultRowsL0CToUb(LocalTensor<outputType> dst, LocalTensor<float> src, uint32_t rows,
                                             uint32_t cols)
{
    // [rows,cols] -> 两个 AIV 的 [rows/2,cols] row-major UB；保持两条 FIX 的先后顺序。
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> copy;
    copy.nSize = cols;
    copy.mSize = rows / 2;
    copy.srcStride = CeilAlign(rows, MM_BLOCK_CUBE);
    copy.dstStride = cols;
    copy.quantPre = std::is_same_v<outputType, float> ? QuantMode_t::NoQuant : QuantMode_t::F322BF16;
    copy.dualDstCtl = 0;
    copy.params.ndNum = 1;
    copy.params.srcNdStride = 0;
    copy.params.dstNdStride = 0;
    copy.subBlockId = 0;
    Fixpipe<outputType, float, CGDR_CFG_ROW_MAJOR_UB>(dst, src, copy);
    copy.subBlockId = 1;
    Fixpipe<outputType, float, CGDR_CFG_ROW_MAJOR_UB>(dst, src[rows / 2 * MM_NZ_FRACTAL], copy);
}

// FIX：FP32 L0C [m,n] -> BF16 L1 NZ；dst 已指向目标子块，fullRows 为目标矩阵行跨度。
__aicore__ inline void CopyTileL0CToL1(LocalTensor<bfloat16_t> dst, LocalTensor<float> l0c, uint32_t m, uint32_t n,
                                       uint32_t fullRows)
{
    FixpipeParamsC310<CO2Layout::NZ> params;
    params.nSize = n;
    params.mSize = m;
    params.srcStride = CeilAlign(m, MM_BLOCK_CUBE);
    params.dstStride = CeilAlign(fullRows, MM_BLOCK_CUBE) * MM_NZ_FRACTAL;
    params.quantPre = QuantMode_t::F322BF16;
    Fixpipe<bfloat16_t, float, CFG_NZ>(dst, l0c, params);
}

// FIX：L0C 有效 [rows,cols] -> GM 行主序；srcRows/dstRowStride 均以元素计。
__aicore__ inline void CopyRowsL0CToGm(GlobalTensor<bfloat16_t> dst, LocalTensor<float> l0c, uint32_t rows,
                                       uint32_t cols, uint32_t srcRows, uint32_t dstRowStride)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
    params.nSize = cols;
    params.mSize = rows;
    params.srcStride = CeilAlign(srcRows, MM_BLOCK_CUBE);
    params.dstStride = dstRowStride;
    params.quantPre = QuantMode_t::F322BF16;
    params.dualDstCtl = 0;
    params.params.ndNum = 1;
    params.params.srcNdStride = 0;
    params.params.dstNdStride = 0;
    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(dst, l0c, params);
}

// FIX：FP32 [rows,cols] 按行平分到两个 AIV 的 UB，各保存 NZ [cols/16,rows/2,16]。
// 优化方案：一条 dual-destination FIX 完成分发，保留 FP32 精度和 16 列分块布局。
// 配置通过 const 引用模板参数传入 Fixpipe，必须具有静态存储期。
static constexpr FixpipeConfig CGDR_CFG_NZ_UB = {CO2Layout::NZ, true};
__aicore__ inline void CopyResultNzL0CToUb(LocalTensor<float> dst, LocalTensor<float> src, uint32_t rows, uint32_t cols)
{
    FixpipeParamsC310<CO2Layout::NZ> fix;
    fix.nSize = cols;
    fix.mSize = rows;
    fix.srcStride = rows;
    fix.dstStride = (rows / 2) * MM_NZ_FRACTAL;
    fix.quantPre = QuantMode_t::NoQuant;
    fix.dualDstCtl = 1;
    fix.subBlockId = 0;
    Fixpipe<float, float, CGDR_CFG_NZ_UB>(dst, src, fix);
}

} // namespace ChunkGatedDeltaRule
