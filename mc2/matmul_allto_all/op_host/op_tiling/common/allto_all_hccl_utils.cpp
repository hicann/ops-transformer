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
 * \file allto_all_hccl_utils.cpp
 * \brief HCCL/CCU limit adjustment utilities implementation for AllToAll-family operators.
 */

#include "allto_all_hccl_utils.h"

#include <algorithm>

#include "mc2_log.h"

namespace MC2Tiling {

// 前置总量约束：总通信量超过任务数上限对应的总容量时，任何切分方式都无法满足，直接提前报错。
ge::graphStatus CheckCcuTotalCommSize(uint64_t mValue, const CommSizeInfo& commSize, uint64_t maxTotalTiles,
                                      const std::string& opName)
{
    if (commSize.innerDim == 0 || commSize.dtypeSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName.c_str(), "commSize", "0",
                                              "innerDim and dtypeSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    // 单行 M 数据在单个 peer 上的通信字节数：innerDim × dtypeSize
    const uint64_t perPeerRowBytes = commSize.innerDim * commSize.dtypeSize;
    // 总通信量上限为 maxTotalTiles × 256MB；用除法避免 mValue × perPeerRowBytes 溢出
    const uint64_t maxTotalBytes = maxTotalTiles * HCCL_MEM_LIMIT;
    if (mValue > maxTotalBytes / perPeerRowBytes) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName.c_str(), "mValue", std::to_string(mValue).c_str(),
            "The total AllToAll communication volume must not exceed maxTotalTiles * 256MB");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(opName.c_str(), "CCU total comm size check passed: mValue=%lu, perPeerRowBytes=%lu, maxTotalTiles=%lu",
            mValue, perPeerRowBytes, maxTotalTiles);
    return ge::GRAPH_SUCCESS;
}

namespace {
constexpr uint64_t ALIGN_LEN = 256;
constexpr double BYTES_PER_MB = 1024.0 * 1024.0;

// 取主块和尾块中的较大长度；没有尾块时只看主块。
uint64_t GetMaxTileLen(const CutResult& cutRes)
{
    uint64_t shortTileLen = (cutRes.numShortTile > 0) ? cutRes.shortTileLen : 0;
    return std::max(cutRes.longTileLen, shortTileLen);
}

// 计算单个切块在当前 per-peer 口径下的通信字节数。
uint64_t CalcCommBytes(uint64_t tileLen, const CommSizeInfo& commSize)
{
    // 上游已校验 x1/x2 轴长，tileLen × innerDim × dtypeSize 不会溢出 uint64
    return tileLen * commSize.innerDim * commSize.dtypeSize;
}

// 判断当前切分是否需要调整：当前最大块的 per-peer 通信量是否超过 256MB。
bool IfExceedCcuLimit(const CutResult& cutRes, const CommSizeInfo& commSize, const std::string& opName)
{
    const uint64_t curMaxTileLen = GetMaxTileLen(cutRes);
    if (curMaxTileLen == 0) {
        return false;
    }

    const uint64_t originalCommSize = CalcCommBytes(curMaxTileLen, commSize);
    OP_LOGD(opName.c_str(), "AllToAll original cut: tileM=%lu, tileCnt=%lu, tailM=%lu, tailCnt=%lu, commSize=%.2fMB",
            cutRes.longTileLen, cutRes.numLongTile, cutRes.shortTileLen, cutRes.numShortTile,
            static_cast<double>(originalCommSize) / BYTES_PER_MB);
    return originalCommSize > HCCL_MEM_LIMIT;
}

// 前置总量检查已保证 perPeerRowBytes > 0；此处只做计算，能否按 256 对齐由调用方校验。
uint64_t CalcMaxTileMFromHcclLimit(const CommSizeInfo& commSize)
{
    return HCCL_MEM_LIMIT / (commSize.innerDim * commSize.dtypeSize);
}

// 按 tileM 重切 M 轴：主块为 tileM，尾块为余数，并置 shortTileAtBack。
void ApplyTileSplit(CutResult& cutRes, uint64_t mValue, uint64_t tileM, const CommSizeInfo& commSize,
                    const std::string& opName)
{
    if (tileM == 0) {
        return;
    }

    uint64_t numLongTile = mValue / tileM;
    uint64_t tailM = mValue - numLongTile * tileM;
    if (numLongTile == 0) {
        numLongTile = 1;
        tileM = mValue;
        tailM = 0;
    }

    const uint64_t commBytes = CalcCommBytes(tileM, commSize);
    cutRes.longTileLen = tileM;
    cutRes.numLongTile = numLongTile;
    cutRes.shortTileLen = tailM;
    cutRes.numShortTile = (tailM > 0) ? 1 : 0;
    cutRes.totalTileCnt = numLongTile + cutRes.numShortTile;
    cutRes.shortTileAtBack = true; // 主块=长块、尾块最后，满足两个算子的 canonical 映射

    OP_LOGD(opName.c_str(),
            "HCCL adjusted cut: tileM=%lu, numLongTile=%lu, tailM=%lu, tailCnt=%lu, totalTileCnt=%lu, "
            "commSize=%.2fMB",
            tileM, numLongTile, tailM, cutRes.numShortTile, cutRes.totalTileCnt,
            static_cast<double>(commBytes) / BYTES_PER_MB);
}
} // namespace

// CCU 256MB 约束入口：调用前需先通过 CheckCcuTotalCommSize；未超限保持原切分，超限时重切并校验任务数与最终通信量。
ge::graphStatus AdjustCutResultForCCU(CutResult& cutRes, uint64_t mValue, const CommSizeInfo& commSize,
                                      const std::string& opName, uint64_t maxTotalTiles)
{
    if (maxTotalTiles == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName.c_str(), "maxTotalTiles", "0",
                                              "maxTotalTiles must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    // 防御性校验：即使绕过前置 CheckCcuTotalCommSize，也避免后续除 0
    if (commSize.innerDim == 0 || commSize.dtypeSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName.c_str(), "commSize", "0",
                                              "innerDim and dtypeSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    if (!IfExceedCcuLimit(cutRes, commSize, opName)) {
        OP_LOGD(opName.c_str(), "HCCL limit satisfied, no adjustment needed");
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t maxTileM = CalcMaxTileMFromHcclLimit(commSize);
    if (maxTileM < ALIGN_LEN) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName.c_str(), "maxTileM", std::to_string(maxTileM).c_str(),
                                              "The value of maxTileM must satisfy the HCCL 256MB limit after 256 "
                                              "alignment");
        return ge::GRAPH_FAILED;
    }
    const uint64_t tileM = (maxTileM / ALIGN_LEN) * ALIGN_LEN;
    OP_LOGD(opName.c_str(), "HCCL limit requires adjustment: maxTileM=%lu, tileM=%lu", maxTileM, tileM);
    ApplyTileSplit(cutRes, mValue, tileM, commSize, opName);
    if (cutRes.totalTileCnt > maxTotalTiles) {
        OP_LOGE(opName.c_str(), "Adjusted totalTileCnt=%lu exceeds task limit=%lu", cutRes.totalTileCnt, maxTotalTiles);
        return ge::GRAPH_FAILED;
    }

    const uint64_t finalMaxTileLen = GetMaxTileLen(cutRes);
    const uint64_t finalCommBytes = CalcCommBytes(finalMaxTileLen, commSize);
    if (finalCommBytes > HCCL_MEM_LIMIT) {
        OP_LOGE(opName.c_str(), "Adjusted comm size exceeds HCCL limit: tileLen=%lu, commSize=%.2fMB", finalMaxTileLen,
                static_cast<double>(finalCommBytes) / BYTES_PER_MB);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace MC2Tiling
