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
 * \file all_to_all_matmul_tiling_data.h
 * \brief 融合算子 tiling——聚合结构体（通信 + 计算切分）+ host 侧构造入口
 *
 * QuantMatmulTilingData 为 apace 仓内共享件，原位引用不拷入。
 */

#pragma once

#include "ccu_all_to_all.asc" // CcuCommTilingData（通信契约）
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h" // QuantMatmulTilingData
#include "apace/tiling/quant_matmul_tiling_swat.h"                                      // host 侧 SWAT tiling 引擎

// 统一 tiling：通信切分 + 计算切分
#pragma pack(push, 8)
struct alignas(8) ccuAllToAllMatmulTilingData {
    CcuCommTilingData comm;
    QuantMatmulTilingData matmul;
};
#pragma pack(pop)

// comm 仅 tileM（其余由 Validate/Prepare 填充）；matmul 由 SWAT 引擎按 shape 现算
inline void FillAllToAllMatmulTilingData(uint64_t blockRows, uint64_t N, uint64_t K, uint64_t tileM,
                                         ccuAllToAllMatmulTilingData& tilingData)
{
    tilingData.comm.tileM = static_cast<uint32_t>(tileM);
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN, mm::BiasDataType::DT_FLOAT>
        tilingEngine;
    tilingEngine.GetTilingData(blockRows, N, K, tilingData.matmul);
}
