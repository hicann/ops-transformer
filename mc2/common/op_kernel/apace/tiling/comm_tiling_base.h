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
 * \file comm_tiling_base.h
 * \brief 通信 Tiling 基类：根据 matmul tiling 结果推导 AllToAll/AllGather 通信切分方案
 *
 * 通信切分跟随 matmul 的最优分核结果：在 M 维度上与 matmul tile 粒度对齐，
 * 确保每个 tile 的通信数据量匹配一份核算力覆盖的 M-range，实现通信与计算流水并行。
 *
 * 用法：
 *   1. 先由 QuantMatmulTilingSwat / QuantMatmulTilingBase 计算 matmul tiling
 *   2. 将 mm tiling 的 baseM/baseN/usedCoreNum + 算子维度 (m, n, ka) 传入
 *   3. 得到 data 和 scale 两组 CommTilingData
 */

#pragma once

#include <cstdint>
#include "comm_tiling_data.h"
#include "quant_matmul_tiling_data.h"
#include "../utils/apace_constant.h"

namespace apace {

class CommTilingBase {
public:
    /*!
     * \brief 计算数据通道的通信 tiling
     *
     * \param m         M_per_rank（单卡输出 M）
     * \param n         N
     * \param ka        per-rank K
     * \param mmTiling  已完成 matmul tiling 的结果（需填充 baseM/baseN/usedCoreNum）
     * \param dataTiling   [out] 数据通信切分
     * \param scaleTiling  [out] Scale 通信切分（nonSplitAxisSize = ka / 32）
     */
    static void GetCommTilingData(uint64_t m, uint64_t n, uint64_t ka, const QuantMatmulTilingData &mmTiling,
                                  CommTilingData &dataTiling, CommTilingData &scaleTiling)
    {
        uint64_t baseM = mmTiling.baseM;
        uint64_t baseN = mmTiling.baseN;
        uint64_t usedCoreNum = mmTiling.usedCoreNum;

        // 防御性零值守卫：baseM/baseN/usedCoreNum 为 0 说明 matmul tiling 异常，
        // 此时无法推导通信切分，直接返回零值 tiling 避免除零。
        if (baseM == 0UL || baseN == 0UL || usedCoreNum == 0UL) {
            dataTiling = {};
            scaleTiling = {};
            return;
        }

        auto nTile = (n + baseN - 1) / baseN;
        uint64_t headMSize = ((usedCoreNum + nTile - 1) / nTile) * baseM;
        if (headMSize == 0UL) {
            dataTiling = {};
            scaleTiling = {};
            return;
        }
        uint64_t headTileCnt = m / headMSize;
        uint64_t tailMSize = m % headMSize;
        uint64_t tailTileCnt = (tailMSize > 0) ? 1 : 0;

        dataTiling.splitAxisTileSize = headMSize;
        dataTiling.splitAxisTileCnt = headTileCnt;
        dataTiling.splitAxisTailSize = tailMSize;
        dataTiling.splitAxisTailCnt = tailTileCnt;
        dataTiling.nonSplitAxisSize = ka;

        scaleTiling = dataTiling;
        scaleTiling.nonSplitAxisSize = ka / MX_GROUP_SIZE;
    }
};

} // namespace apace
