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
 * \file all_gather_hccl_utils.h
 * \brief HCCL limit adjustment utilities for all_gather_matmul_v2 (arch35)
 */

#ifndef ALL_GATHER_HCCL_UTILS_H
#define ALL_GATHER_HCCL_UTILS_H

#include <cstdint>
#include <string>
#include "op_host/op_tiling/formulaic_tiling_datatype.h"

namespace optiling {

constexpr uint64_t HCCL_MEM_LIMIT = 268435456; // 256MB
constexpr uint64_t HCCL_MAX_TOTAL_TILES = 62; // 总通信次数上限63, tile+tail最大62, 1个保留给x1Scale通信的情况
constexpr uint64_t ALIGN_LEN = 256;           // tileM对齐长度
constexpr uint64_t TILE_M_CANDIDATES[] = {2048, 1024, 512, 256}; // 优选候选值

constexpr uint64_t HCCL_NO_ADJUSTMENT = 0;        // 不需要调整
constexpr uint64_t HCCL_UNSUPPORTED = UINT64_MAX; // 无法处理

constexpr uint64_t AICPU_TARGET_TILE_CNT = 4; // AICPU 目标 tile 数

struct CommSizeInfo {
    uint64_t kValue;
    uint64_t dtypeSize;
    uint64_t rankDim;
};

uint64_t CalcMaxTileMFromHcclLimit(const CutResult &cutRes, const CommSizeInfo &commSize, const std::string &opName);

uint64_t SelectOptimalCandidateTileM(uint64_t maxTileM);

uint64_t DetermineFinalTileMWithLimit(uint64_t mValue, uint64_t candidateTileM, uint64_t maxTileM,
                                      const CommSizeInfo &commSize, const std::string &opName);

void ApplyTileSplit(CutResult &cutRes, uint64_t mValue, uint64_t tileM, const CommSizeInfo &commSize,
                    const std::string &opName);

void AdjustCutResultForCCU(CutResult &cutRes, uint64_t mValue, const CommSizeInfo &commSize, const std::string &opName);

void AdjustCutResultForAICPU(CutResult &cutRes, uint64_t mValue, uint64_t baseM, const std::string &opName);
} // namespace optiling

#endif // ALL_GATHER_HCCL_UTILS_H
