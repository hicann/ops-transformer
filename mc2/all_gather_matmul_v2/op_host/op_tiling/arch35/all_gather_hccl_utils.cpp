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
 * \file all_gather_hccl_utils.cpp
 * \brief HCCL limit adjustment utilities implementation
 */

#include "all_gather_hccl_utils.h"
#include "mc2_log.h"
#include <algorithm>
#include <cmath>

namespace optiling {

/*!
 * \brief 计算满足HCCL 256MB单次通信限制的最大tileM，并判断是否需要调整
 *
 * \param [in] cutRes      公式化切分的原始结果
 * \param [in] kValue      K维大小
 * \param [in] dtypeSize   数据类型大小（字节）
 * \param [in] rankDim     卡数
 * \param [in] opName      算子名称（用于日志）
 *
 * \return HCCL_NO_ADJUSTMENT (0)  - 不需要调整（原始切分已满足256MB限制）
 * \return HCCL_UNSUPPORTED       - 无法支持（单行通信量过大，无法进行有效切分）
 * \return 其他值                 - 需要调整，返回满足256MB限制的最大tileM
 *
 * \details 算法步骤：
 *   1. 计算原始切分的单次通信量：tileM × K × dtypeSize × rankDim
 *   2. 若通信量 ≤ 256MB，返回 HCCL_NO_ADJUSTMENT
 *   3. 计算满足256MB的最大tileM = 256MB / (K × dtypeSize × rankDim)
 *   4. 若 maxTileM < 256（最小对齐长度），返回 HCCL_UNSUPPORTED
 *   5. 返回 maxTileM，表示需要调整
 */
uint64_t CalcMaxTileMFromHcclLimit(const CutResult &cutRes, const CommSizeInfo &commSize, const std::string &opName)
{
    uint64_t singleRowSize = commSize.kValue * commSize.dtypeSize * commSize.rankDim;
    if (singleRowSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName.c_str(), "singleRowSize", std::to_string(singleRowSize).c_str(),
                                              "The value of singleRowSize must not be zero");
        return HCCL_UNSUPPORTED;
    }

    uint64_t tileMKBytes = cutRes.longTileLen * commSize.kValue;
    if (cutRes.longTileLen != 0 && tileMKBytes / cutRes.longTileLen != commSize.kValue) {
        OP_LOGE(opName.c_str(), "Multiplication overflow: longTileLen * kValue = %lu * %lu = %lu", cutRes.longTileLen,
                commSize.kValue, tileMKBytes);
        return HCCL_UNSUPPORTED;
    }

    uint64_t tileBytesPerRank = tileMKBytes * commSize.dtypeSize;
    if (tileMKBytes != 0 && tileBytesPerRank / tileMKBytes != commSize.dtypeSize) {
        OP_LOGE(opName.c_str(), "Multiplication overflow: tileMKBytes * dtypeSize = %lu * %lu = %lu", tileMKBytes,
                commSize.dtypeSize, tileBytesPerRank);
        return HCCL_UNSUPPORTED;
    }

    uint64_t originalCommSize = tileBytesPerRank * commSize.rankDim;
    if (tileBytesPerRank != 0 && originalCommSize / tileBytesPerRank != commSize.rankDim) {
        OP_LOGE(opName.c_str(), "Multiplication overflow: tileBytesPerRank * rankDim = %lu * %lu = %lu",
                tileBytesPerRank, commSize.rankDim, originalCommSize);
        return HCCL_UNSUPPORTED;
    }

    double originalCommMB = static_cast<double>(originalCommSize) / (1024.0 * 1024.0);
    OP_LOGD(opName.c_str(),
            "Formulaic original cut: tileM=%lu, tileCnt=%lu, tailM=%lu, "
            "shortTileAtBack=%d, commSize=%.2fMB",
            cutRes.longTileLen, cutRes.numLongTile, cutRes.shortTileLen, cutRes.shortTileAtBack, originalCommMB);

    if (originalCommSize <= HCCL_MEM_LIMIT) {
        OP_LOGD(opName.c_str(), "HCCL limit satisfied, no adjustment needed");
        return HCCL_NO_ADJUSTMENT;
    }

    uint64_t maxTileM = HCCL_MEM_LIMIT / singleRowSize;
    if (maxTileM < ALIGN_LEN) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName.c_str(), "maxTileM", std::to_string(maxTileM).c_str(),
                                              "The value of maxTileM must satisfy the HCCL memory limit");
        return HCCL_UNSUPPORTED;
    }

    OP_LOGD(opName.c_str(), "HCCL limit requires adjustment: maxTileM=%lu", maxTileM);
    return maxTileM;
}

/*!
 * \brief 从候选列表中优选满足256MB限制的tileM
 *
 * \param [in] maxTileM    满足256MB限制的最大tileM
 *
 * \return 优选的候选tileM（2048、1024、512、256或256对齐的计算值）
 *
 * \details 优选策略：
 *   1. 候选列表：[2048, 1024, 512, 256]（优先大值，性能更好）
 *   2. 从大到小遍历，找第一个 ≤ maxTileM 的值
 *   3. 若候选值都超过 maxTileM，返回向下256对齐的计算值
 *
 * \note 候选列表中的值有利于：
 *   - 减少通信次数（大tileM → 少切分）
 *   - 提升计算效率（整齐的block划分）
 */
uint64_t SelectOptimalCandidateTileM(uint64_t maxTileM)
{
    for (uint64_t candidate : TILE_M_CANDIDATES) {
        if (candidate <= maxTileM) {
            return candidate;
        }
    }

    return (maxTileM / ALIGN_LEN) * ALIGN_LEN;
}

/*!
 * \brief 确定满足HCCL_MAX_TOTAL_TILES限制的最终tileM，必要时反向计算
 *
 * \param [in] mValue         M维大小
 * \param [in] candidateTileM 优选的候选tileM
 * \param [in] maxTileM       满足256MB限制的最大tileM
 * \param [in] kValue         K维大小
 * \param [in] dtypeSize      数据类型大小（字节）
 * \param [in] rankDim        卡数
 * \param [in] opName         算子名称（用于日志）
 *
 * \return HCCL_UNSUPPORTED   - 无法同时满足256MB和HCCL_MAX_TOTAL_TILES限制
 * \return 其他值             - 最终的tileM
 *
 * \details 算法步骤：
 *   1. 预估总切分份数：totalTileCnt = ceil(M / candidateTileM)
 *   2. 若 totalTileCnt ≤ HCCL_MAX_TOTAL_TILES，直接返回 candidateTileM（优选值已满足）
 *   3. 若 totalTileCnt > HCCL_MAX_TOTAL_TILES，反向计算满足HCCL_MAX_TOTAL_TILES次的最小tileM：
 *      - minTileM = ceil(M / HCCL_MAX_TOTAL_TILES)
 *      - 向上256对齐 → minTileM_align
 *      - 在候选列表中找 ≥ minTileM_align 的最小值（仍优先候选列表值）
 *   4. 验证256MB限制：若 finalTileM > maxTileM，返回 HCCL_UNSUPPORTED
 *   5. 返回 finalTileM
 *
 * \note 这种策略确保：
 *   - 正常数据量：优先性能（大tileM、少通信）
 *   - 大数据量：扩展到最多HCCL_MAX_TOTAL_TILES次通信处理
 *   - 超大数据量：报错提示不支持
 */
uint64_t DetermineFinalTileMWithLimit(uint64_t mValue, uint64_t candidateTileM, uint64_t maxTileM,
                                      const std::string &opName)
{
    uint64_t totalTileCnt = (mValue + candidateTileM - 1) / candidateTileM;
    if (totalTileCnt <= HCCL_MAX_TOTAL_TILES) {
        OP_LOGD(opName.c_str(), "Candidate tileM=%lu satisfies limit: totalTileCnt=%lu ≤ %lu", candidateTileM,
                totalTileCnt, HCCL_MAX_TOTAL_TILES);
        return candidateTileM;
    }

    OP_LOGD(opName.c_str(),
            "Candidate tileM=%lu requires too many tiles: totalTileCnt=%lu > %lu, "
            "recalculating...",
            candidateTileM, totalTileCnt, HCCL_MAX_TOTAL_TILES);

    uint64_t minTileM = (mValue + HCCL_MAX_TOTAL_TILES - 1) / HCCL_MAX_TOTAL_TILES;
    uint64_t minTileM_align = ((minTileM + ALIGN_LEN - 1) / ALIGN_LEN) * ALIGN_LEN;
    uint64_t finalTileM = HCCL_UNSUPPORTED;
    for (size_t i = 0; i < sizeof(TILE_M_CANDIDATES) / sizeof(TILE_M_CANDIDATES[0]); i++) {
        if (TILE_M_CANDIDATES[i] >= minTileM_align) {
            finalTileM = TILE_M_CANDIDATES[i];
            break;
        }
    }

    if (finalTileM == HCCL_UNSUPPORTED) {
        finalTileM = minTileM_align;
    }

    if (finalTileM > maxTileM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName.c_str(), "HCCL tile size", std::to_string(finalTileM).c_str(),
                                              "The value of HCCL tile size must satisfy the HCCL 256MB limit");
        return HCCL_UNSUPPORTED;
    }

    OP_LOGD(opName.c_str(), "Recalculated tileM=%lu to satisfy %lu tile limit", finalTileM, HCCL_MAX_TOTAL_TILES);
    return finalTileM;
}

/*!
 * \brief 应用tileM切分结果，计算主块和尾块，更新cutRes
 *
 * \param [inout] cutRes    切分结果（将被更新）
 * \param [in] mValue       M维大小
 * \param [in] tileM        最终确定的tileM
 * \param [in] kValue       K维大小
 * \param [in] dtypeSize    数据类型大小（字节）
 * \param [in] rankDim      卡数
 * \param [in] opName       算子名称（用于日志）
 *
 * \details 算法步骤：
 *   1. 计算切分：numLongTile = floor(M / tileM)，tailM = M - numLongTile × tileM
 *   2. 边界处理：若 numLongTile == 0（M < tileM），调整为单块切分
 *   3. 验证单块256MB限制（极端小shape）
 *   4. 更新 cutRes 各字段：
 *      - longTileLen = tileM（主块长度）
 *      - numLongTile（主块数量）
 *      - shortTileLen = tailM（尾块长度）
 *      - numShortTile = (tailM > 0) ? 1 : 0
 *      - totalTileCnt = numLongTile + numShortTile
 *      - shortTileAtBack 保持原始值
 *   5. 打印调整后的切分结果
 */
void ApplyTileSplit(CutResult &cutRes, uint64_t mValue, uint64_t tileM, const CommSizeInfo &commSize,
                    const std::string &opName)
{
    uint64_t numLongTile = mValue / tileM;
    uint64_t tailM = mValue - numLongTile * tileM;
    if (numLongTile == 0) {
        numLongTile = 1;
        tileM = mValue;
        tailM = 0;

        uint64_t newCommSize = tileM * commSize.kValue * commSize.dtypeSize * commSize.rankDim;
        if (newCommSize > HCCL_MEM_LIMIT) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName.c_str(), "newCommSize", std::to_string(newCommSize).c_str(),
                                                  "The value of newCommSize must not exceed the HCCL limit");
            return;
        }
    }

    cutRes.longTileLen = tileM;
    cutRes.numLongTile = numLongTile;
    cutRes.shortTileLen = tailM;
    cutRes.numShortTile = (tailM > 0) ? 1 : 0;
    cutRes.totalTileCnt = numLongTile + cutRes.numShortTile;

    uint64_t newCommSize = tileM * commSize.kValue * commSize.dtypeSize * commSize.rankDim;
    double newCommMB = static_cast<double>(newCommSize) / (1024.0 * 1024.0);
    OP_LOGD(opName.c_str(),
            "HCCL adjusted cut: tileM=%lu, numLongTile=%lu, tailM=%lu, "
            "totalTileCnt=%lu, shortTileAtBack=%d, commSize=%.2fMB",
            tileM, numLongTile, tailM, cutRes.totalTileCnt, cutRes.shortTileAtBack, newCommMB);
    if (tailM == 0) {
        OP_LOGD(opName.c_str(), "Perfect split: no tail block, all tiles aligned");
    }
}

/*!
 * \brief HCCL限制调整主函数，协调各子函数完成切分调整
 *
 * \param [inout] cutRes    公式化切分的原始结果（将被调整）
 * \param [in] mValue       M维大小
 * \param [in] kValue       K维大小
 * \param [in] dtypeSize    数据类型大小（字节）
 * \param [in] rankDim      卡数
 * \param [in] opName       算子名称（用于日志）
 *
 * \details 整体流程：
 *   1. CalcMaxTileMFromHcclLimit()  - 判断是否需要调整，计算maxTileM
 *   2. SelectOptimalCandidateTileM() - 优选tileM候选值（2048优先）
 *   3. DetermineFinalTileMWithLimit() - 确定满足HCCL_MAX_TOTAL_TILES次限制的最终tileM
 *   4. ApplyTileSplit()              - 应用切分结果，更新cutRes
 *
 * \note 调整策略：
 *   - 正常数据量：优先性能，8-16次通信内完成，使用大tileM（2048/1024）
 *   - 大数据量：扩展到最多HCCL_MAX_TOTAL_TILES次通信，仍优先使用优选候选值
 *   - 超大数据量：报错，无法同时满足256MB单次限制和HCCL_MAX_TOTAL_TILES次总限制
 */
void AdjustCutResultForCCU(CutResult &cutRes, uint64_t mValue, const CommSizeInfo &commSize, const std::string &opName)
{
    uint64_t maxTileM = CalcMaxTileMFromHcclLimit(cutRes, commSize, opName);
    if (maxTileM == HCCL_NO_ADJUSTMENT || maxTileM == HCCL_UNSUPPORTED) {
        return;
    }

    uint64_t candidateTileM = SelectOptimalCandidateTileM(maxTileM);
    OP_LOGD(opName.c_str(), "Selected optimal candidate tileM=%lu", candidateTileM);

    uint64_t tileM = DetermineFinalTileMWithLimit(mValue, candidateTileM, maxTileM, opName);
    if (tileM == HCCL_UNSUPPORTED) {
        return;
    }

    ApplyTileSplit(cutRes, mValue, tileM, commSize, opName);
}

/*!
 * \brief AICPU通信模式减少切分，降低通信固定开销
 *
 * \param [inout] cutRes         公式化切分的原始结果（将被调整）
 * \param [in] mValue            M维大小
 * \param [in] baseM             Matmul基本块M，用于tileM对齐
 * \param [in] opName            算子名称（用于日志）
 *
 * \details 整体流程：
 *   1. 判断当前tile数是否已满足AICPU_TARGET_TILE_CNT，若是则直接返回
 *   2. 计算目标tileM = ceil(M / AICPU_TARGET_TILE_CNT)，向上对齐到baseM
 *   3. 取max(对齐后tileM, 原始longTileLen)，确保不减少切分粒度
 *   4. 重新切分并更新cutRes
 *
 */
void AdjustCutResultForAICPU(CutResult &cutRes, uint64_t mValue, uint64_t baseM, const std::string &opName)
{
    uint64_t currentTileCnt = cutRes.numLongTile + cutRes.numShortTile;
    if (currentTileCnt <= AICPU_TARGET_TILE_CNT) {
        OP_LOGD(opName.c_str(), "AICPU: currentTileCnt=%lu <= target=%lu, no adjustment needed", currentTileCnt,
                AICPU_TARGET_TILE_CNT);
        return;
    }

    uint64_t idealTileM = (mValue + AICPU_TARGET_TILE_CNT - 1) / AICPU_TARGET_TILE_CNT;
    uint64_t alignedTileM = idealTileM;
    if (baseM > 0) {
        alignedTileM = ((idealTileM + baseM - 1) / baseM) * baseM;
    }

    uint64_t newTileM = std::max(alignedTileM, cutRes.longTileLen);
    uint64_t numLongTile = mValue / newTileM;
    uint64_t tailM = mValue - numLongTile * newTileM;
    if (numLongTile == 0) {
        numLongTile = 1;
        newTileM = mValue;
        tailM = 0;
    }

    cutRes.longTileLen = newTileM;
    cutRes.numLongTile = numLongTile;
    cutRes.shortTileLen = tailM;
    cutRes.numShortTile = (tailM > 0) ? 1 : 0;
    cutRes.totalTileCnt = numLongTile + cutRes.numShortTile;

    OP_LOGD(opName.c_str(),
            "AICPU adjusted: tileM=%lu, numLongTile=%lu, tailM=%lu, "
            "totalTileCnt=%lu (was %lu), shortTileAtBack=%d",
            newTileM, numLongTile, tailM, cutRes.totalTileCnt, currentTileCnt, cutRes.shortTileAtBack);
}

} // namespace optiling
