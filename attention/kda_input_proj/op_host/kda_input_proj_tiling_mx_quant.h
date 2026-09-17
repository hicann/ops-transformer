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
 * \file kda_input_proj_tiling_mx_quant.h
 * \brief Stage1 AIV: DynamicMxQuant tiling
 *
 * 对齐 kernel 侧 KdaInputProjMxQuant（arch35 尾轴优化模板，blockSize=32，scaleAlg=0 OCP，
 * bf16 -> fp8_e4m3fn）。量化轴固定为 x 的尾轴（hiddenSize），因此这里只实现
 * ops-nn DynamicMxQuant 的 tail-axis 分支，不需要 axis / 非尾轴的分支判断。
 */

#ifndef KDA_INPUT_PROJ_TILING_MX_QUANT_H
#define KDA_INPUT_PROJ_TILING_MX_QUANT_H

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

#include "kda_input_proj_tiling.h"

namespace optiling {
namespace kda_input_proj_mx_quant {
constexpr int64_t MX_BLOCK_SIZE = 32L; // MX 量化块大小，与 kernel blockSize 一致
constexpr int64_t SLICE_SIZE_256 = 256L;
constexpr int64_t SLICE_SIZE_512 = 512L;
constexpr int64_t BLOCKS_PER_SLICE = SLICE_SIZE_256 / MX_BLOCK_SIZE; // 一个 256 切片含 8 个 32-block
constexpr int64_t DB_BUFFER = 2L;
constexpr int64_t VF_LEN_16 = 128L; // 与 kernel 的 scale buffer 对齐粒度一致
constexpr int64_t RESERVED_UB_SIZE = 2048L;
constexpr int64_t ROUND_MODE_RINT = 4L; // 与 kernel Cast 的 RoundMode::CAST_RINT 对应
constexpr float LOAD_BALANCE_THRESHOLD = 0.75F;

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return b == 0L ? 0L : (a + b - 1L) / b;
}

inline int64_t CeilAlign(int64_t a, int64_t b)
{
    return CeilDiv(a, b) * b;
}

// 每个 32-block 的 UB 开销，忽略 scale buffer 的 VF_LEN_16 对齐余量，用于估初值
constexpr int64_t UB_BYTES_PER_BLOCK =
    DB_BUFFER * (MX_BLOCK_SIZE * static_cast<int64_t>(sizeof(uint16_t)) +
                 MX_BLOCK_SIZE * static_cast<int64_t>(sizeof(uint8_t)) + static_cast<int64_t>(sizeof(uint16_t)) +
                 2L * static_cast<int64_t>(sizeof(uint16_t)) + 2L * static_cast<int64_t>(sizeof(uint16_t)));

// kernel InitUbLayout 的精确 UB 占用：x(bf16) / y(fp8) / scale(uint16) 各 DB 份，
// 外加 maxExp、recipScale 两块 uint16 暂存（每 32-block 2 个元素）。
inline int64_t UbBytesForBlockNum(int64_t blockNum)
{
    const int64_t xSlotBytes = blockNum * MX_BLOCK_SIZE * static_cast<int64_t>(sizeof(uint16_t));
    const int64_t ySlotBytes = blockNum * MX_BLOCK_SIZE * static_cast<int64_t>(sizeof(uint8_t));
    const int64_t scaleSlotBytes = CeilAlign(blockNum, VF_LEN_16) * static_cast<int64_t>(sizeof(uint16_t));
    const int64_t maxExpSlotBytes = blockNum * 2L * static_cast<int64_t>(sizeof(uint16_t));
    const int64_t recipScaleSlotBytes = blockNum * 2L * static_cast<int64_t>(sizeof(uint16_t));
    return DB_BUFFER * (xSlotBytes + ySlotBytes + scaleSlotBytes + maxExpSlotBytes + recipScaleSlotBytes);
}
} // namespace kda_input_proj_mx_quant

class KdaInputProjMxQuantTiling {
public:
    explicit KdaInputProjMxQuantTiling(const KdaInputProjTilingInfo &tilingInfo)
        : tilingInfo_(tilingInfo)
    {}

    ge::graphStatus CalcTiling(KdaInputProjMxQuantParams &params) const
    {
        using namespace kda_input_proj_mx_quant;

        const char *opName = tilingInfo_.opName != nullptr ? tilingInfo_.opName : "KdaInputProj";
        const int64_t rowNum = static_cast<int64_t>(tilingInfo_.baseParams.tSize);
        const int64_t colNum = static_cast<int64_t>(tilingInfo_.baseParams.hiddenSize);
        const int64_t totalCoreNum = static_cast<int64_t>(tilingInfo_.aivNum);
        const int64_t ubSize = static_cast<int64_t>(tilingInfo_.ubSize);

        OP_CHECK_IF(rowNum <= 0L || colNum <= 0L,
                    OP_LOGE(opName, "MxQuant got invalid x shape [%ld, %ld].", rowNum, colNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(totalCoreNum <= 0L, OP_LOGE(opName, "MxQuant got aivNum=%ld.", totalCoreNum),
                    return ge::GRAPH_FAILED);
        // kernel 的 CopyIn/CopyOut 按 32 元素块搬运，量化轴必须整块划分
        OP_CHECK_IF(colNum % MX_BLOCK_SIZE != 0L,
                    OP_LOGE(opName, "MxQuant requires hiddenSize=%ld to be a multiple of %ld.", colNum, MX_BLOCK_SIZE),
                    return ge::GRAPH_FAILED);

        int64_t maxUbBlockNum = 0L;
        OP_CHECK_IF(CalcMaxUbBlockNum(ubSize, maxUbBlockNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName, "MxQuant failed to fit UB, ubSize=%ld.", ubSize), return ge::GRAPH_FAILED);

        const int64_t colBlocks = colNum / MX_BLOCK_SIZE;
        // 一行能进 UB 时只按行切核，列方向连续搬。T=8 每核 1 行；T=128 大约 43 核、每核 3 行
        if (colBlocks > 0L && colBlocks <= maxUbBlockNum) {
            const int64_t rowNormalBlockNum = CeilDiv(rowNum, std::min(rowNum, totalCoreNum));
            const int64_t rowTileNum = CeilDiv(rowNum, rowNormalBlockNum);
            const int64_t colTileNum = 1L;
            const int64_t colNormalBlockNum = CeilDiv(colNum, SLICE_SIZE_256);
            params.tilingKey = 0UL;
            params.ubSize = ubSize;
            params.roundMode = ROUND_MODE_RINT;
            params.blockSize = MX_BLOCK_SIZE;
            params.totalCoreNum = totalCoreNum;
            params.usedCoreNum = rowTileNum * colTileNum;
            params.rowTileNum = rowTileNum;
            params.colTileNum = colTileNum;
            params.rowNum = rowNum;
            params.colNum = colNum;
            params.colNormalBlockNum = colNormalBlockNum;
            params.colTailLen = colNum;
            params.rowNormalBlockNum = rowNormalBlockNum;
            params.rowTailLen = rowNum - rowNormalBlockNum * (rowTileNum - 1L);
            OP_CHECK_IF(params.rowTailLen <= 0L || params.usedCoreNum > totalCoreNum,
                        OP_LOGE(opName, "MxQuant full-row split invalid: used=%ld aiv=%ld rowNorm=%ld rowTail=%ld.",
                                params.usedCoreNum, totalCoreNum, params.rowNormalBlockNum, params.rowTailLen),
                        return ge::GRAPH_FAILED);
            params.maxUbBlockNum = maxUbBlockNum;
            params.dstTypeMax = 0.0F;
            params.invDstTypeMax = 0.0F;
            params.maxLowBound = 0.0F;
            OP_LOGI(opName,
                    "KdaInputProj MxQuant tiling: row=%ld col=%ld core=%ld/%ld tile=[%ld, %ld] normal=[%ld, %ld] "
                    "tail=[%ld, %ld] maxUbBlockNum=%ld (full-row).",
                    params.rowNum, params.colNum, params.usedCoreNum, params.totalCoreNum, params.rowTileNum,
                    params.colTileNum, params.rowNormalBlockNum, params.colNormalBlockNum, params.rowTailLen,
                    params.colTailLen, params.maxUbBlockNum);
            return ge::GRAPH_SUCCESS;
        }

        // 列方向按 256 或 512 元素切片，行数不足时用更小的切片换取更好的负载均衡
        int64_t sliceSize = SLICE_SIZE_512;
        if (static_cast<float>(rowNum * CeilDiv(colNum, sliceSize)) <
            static_cast<float>(totalCoreNum) * LOAD_BALANCE_THRESHOLD) {
            sliceSize = SLICE_SIZE_256;
        }
        const int64_t rowBlockLoopNum = rowNum;
        const int64_t colBlockLoopNum = CeilDiv(colNum, sliceSize);

        int64_t rowTileNum = 1L;
        int64_t colTileNum = 1L;
        OP_CHECK_IF(
            SplitCores(totalCoreNum, rowBlockLoopNum, colBlockLoopNum, rowTileNum, colTileNum) != ge::GRAPH_SUCCESS,
            OP_LOGE(opName, "MxQuant failed to split cores for [%ld, %ld].", rowNum, colNum), return ge::GRAPH_FAILED);

        int64_t rowNormalBlockNum = CeilDiv(rowBlockLoopNum, rowTileNum);
        int64_t colNormalBlockNum = CeilDiv(colBlockLoopNum, colTileNum);
        rowTileNum = CeilDiv(rowBlockLoopNum, rowNormalBlockNum);
        colTileNum = CeilDiv(colBlockLoopNum, colNormalBlockNum);
        // kernel 侧 colNormalBlockNum 的单位是 256 元素，需要把切片单位折算回去
        colNormalBlockNum = colNormalBlockNum * sliceSize / SLICE_SIZE_256;

        const int64_t usedCoreNum = rowTileNum * colTileNum;
        OP_CHECK_IF(usedCoreNum > totalCoreNum,
                    OP_LOGE(opName, "MxQuant usedCoreNum=%ld exceeds aivNum=%ld.", usedCoreNum, totalCoreNum),
                    return ge::GRAPH_FAILED);

        params.tilingKey = 0UL; // 算子级 tilingKey 由 trans_weight_* 决定，模块内不再分支
        params.ubSize = ubSize;
        params.roundMode = ROUND_MODE_RINT;
        params.blockSize = MX_BLOCK_SIZE;
        params.totalCoreNum = totalCoreNum;
        params.usedCoreNum = usedCoreNum;
        params.rowTileNum = rowTileNum;
        params.colTileNum = colTileNum;
        params.rowNum = rowNum;
        params.colNum = colNum;
        params.colNormalBlockNum = colNormalBlockNum;
        params.colTailLen = colNum - colNormalBlockNum * SLICE_SIZE_256 * (colTileNum - 1L);
        params.rowNormalBlockNum = rowNormalBlockNum;
        params.rowTailLen = rowNum - rowNormalBlockNum * (rowTileNum - 1L);
        params.maxUbBlockNum = maxUbBlockNum;
        // scaleAlg=0 (OCP) 下 kernel 用 FP8_E4M3_MAX_EXP 常量推导 shared exponent，不读这三个字段
        params.dstTypeMax = 0.0F;
        params.invDstTypeMax = 0.0F;
        params.maxLowBound = 0.0F;

        OP_LOGI(opName,
                "KdaInputProj MxQuant tiling: row=%ld col=%ld core=%ld/%ld tile=[%ld, %ld] normal=[%ld, %ld] "
                "tail=[%ld, %ld] maxUbBlockNum=%ld.",
                params.rowNum, params.colNum, params.usedCoreNum, params.totalCoreNum, params.rowTileNum,
                params.colTileNum, params.rowNormalBlockNum, params.colNormalBlockNum, params.rowTailLen,
                params.colTailLen, params.maxUbBlockNum);
        return ge::GRAPH_SUCCESS;
    }

private:
    // UB 能容纳的 32-block 数量，按 8 的倍数对齐（kernel 以 8 个 block 为一次 VF 迭代）
    static ge::graphStatus CalcMaxUbBlockNum(int64_t ubSize, int64_t &maxUbBlockNum)
    {
        using namespace kda_input_proj_mx_quant;

        const int64_t budget = ubSize - RESERVED_UB_SIZE;
        if (budget <= 0L) {
            return ge::GRAPH_FAILED;
        }
        // 先按忽略 scale 对齐的近似值估一个上界，再逐步收缩到精确容量之内
        int64_t blockNum = budget / UB_BYTES_PER_BLOCK / BLOCKS_PER_SLICE * BLOCKS_PER_SLICE;
        while (blockNum > 0L && UbBytesForBlockNum(blockNum) > budget) {
            blockNum -= BLOCKS_PER_SLICE;
        }
        if (blockNum <= 0L) {
            return ge::GRAPH_FAILED;
        }
        maxUbBlockNum = blockNum;
        return ge::GRAPH_SUCCESS;
    }

    // 枚举 rowTileNum * colTileNum 的组合，取尾块浪费最小、优先切行的方案
    static ge::graphStatus SplitCores(int64_t totalCoreNum, int64_t rowBlockLoopNum, int64_t colBlockLoopNum,
                                      int64_t &rowTileNum, int64_t &colTileNum)
    {
        using namespace kda_input_proj_mx_quant;

        int64_t usedCoreNum = std::min(totalCoreNum, rowBlockLoopNum * colBlockLoopNum);
        usedCoreNum = usedCoreNum <= 0L ? 1L : usedCoreNum;

        std::set<int64_t> cutSet;
        const int64_t upBound = static_cast<int64_t>(std::ceil(std::sqrt(static_cast<double>(usedCoreNum)))) + 1L;
        for (int64_t m = 1L; m < upBound; ++m) {
            cutSet.insert(m);
            cutSet.insert(usedCoreNum / m);
        }

        // {rowTileNum, colTileNum, delta}，delta 越小表示尾块空转越少
        std::vector<std::vector<int64_t>> candidates;
        for (int64_t m : cutSet) {
            if (m <= 0L || m > rowBlockLoopNum) {
                continue;
            }
            int64_t n = usedCoreNum / m;
            n = n < 1L ? 1L : n;
            if (n > colBlockLoopNum) {
                continue;
            }

            const int64_t rowNormal = CeilDiv(rowBlockLoopNum, m);
            const int64_t colNormal = CeilDiv(colBlockLoopNum, n);
            int64_t delta = rowNormal * colNormal;
            if (m * n == usedCoreNum) {
                if (rowBlockLoopNum % m == 0L && colBlockLoopNum % n == 0L) {
                    delta = 0L;
                } else if (rowBlockLoopNum % m == 0L) {
                    delta -= rowNormal * (colBlockLoopNum % colNormal);
                } else if (colBlockLoopNum % n == 0L) {
                    delta -= (rowBlockLoopNum % rowNormal) * colNormal;
                } else {
                    delta -= (rowBlockLoopNum % rowNormal) * (colBlockLoopNum % colNormal);
                }
            }
            candidates.push_back({m, n, delta});
        }
        if (candidates.empty()) {
            return ge::GRAPH_FAILED;
        }

        constexpr size_t COL_TILE_IDX = 1U;
        constexpr size_t DELTA_IDX = 2U;
        std::sort(
            candidates.begin(), candidates.end(), [](const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
                // delta 相同时优先切行（colTileNum 更小），减少 scale 尾块拼接
                return std::make_pair(a[DELTA_IDX], a[COL_TILE_IDX]) < std::make_pair(b[DELTA_IDX], b[COL_TILE_IDX]);
            });

        rowTileNum = candidates[0][0];
        colTileNum = candidates[0][COL_TILE_IDX];
        return ge::GRAPH_SUCCESS;
    }

    const KdaInputProjTilingInfo &tilingInfo_;
};
} // namespace optiling

#endif // KDA_INPUT_PROJ_TILING_MX_QUANT_H
