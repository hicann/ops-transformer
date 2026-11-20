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
 * \file allto_all_hccl_utils.h
 * \brief HCCL/CCU limit adjustment utilities shared by MatmulAlltoAll and AllToAllMatmul.
 */

#ifndef ALLTO_ALL_HCCL_UTILS_H
#define ALLTO_ALL_HCCL_UTILS_H

#include <cstdint>
#include <string>

#include "graph/ge_error_codes.h"
#include "op_host/op_tiling/formulaic_tiling_datatype.h"

namespace MC2Tiling {

constexpr uint64_t HCCL_MEM_LIMIT = 256UL * 1024UL * 1024UL; // 256MB
constexpr uint64_t HCCL_MAX_TOTAL_TILES = 63;
constexpr uint64_t HCCL_MAX_TOTAL_TILES_MX = 62;

struct CommSizeInfo {
    // per-peer 单行通信内轴：MatmulAlltoAll=N/rank；AllToAllMatmul=H（fp4/HiF4 为 ceil(H/2)）
    uint64_t innerDim;
    uint64_t dtypeSize;
};

/*!
 * \brief CCU 总量可行性前置检查：总通信量超过 maxTotalTiles × 256MB 时提前失败
 *
 * \note 必须在公式化 tiling 前调用；失败时无需进入后续 tiling 和调整逻辑
 * \param maxTotalTiles 通信任务数上限：MatmulAlltoAll 恒为 63；AllToAllMatmul FP/KC 为 63，MX 为 62
 */
ge::graphStatus CheckCcuTotalCommSize(uint64_t mValue, const CommSizeInfo& commSize, uint64_t maxTotalTiles,
                                      const std::string& opName);

/*!
 * \brief CCU 单次通信量约束调整：未超限时保持原切分，超限时按 256MB 上限重切 M 轴
 *
 * \note 调用前需先通过 CheckCcuTotalCommSize 的总量检查
 * \param maxTotalTiles 通信任务数上限：MatmulAlltoAll 恒为 63；AllToAllMatmul FP/KC 为 63，MX 为 62
 */
ge::graphStatus AdjustCutResultForCCU(CutResult& cutRes, uint64_t mValue, const CommSizeInfo& commSize,
                                      const std::string& opName, uint64_t maxTotalTiles = HCCL_MAX_TOTAL_TILES);

} // namespace MC2Tiling

#endif // ALLTO_ALL_HCCL_UTILS_H
