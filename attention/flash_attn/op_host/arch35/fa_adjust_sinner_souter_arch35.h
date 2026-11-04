/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_ARCH35_H
#define FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_ARCH35_H

#include <cstdint>
#include <initializer_list>
#include "../../op_kernel/utils/flash_attn_common_def.h"

namespace optiling {
namespace flash_attn {
namespace fa_tiling_util {

constexpr int64_t MAX_SEQ_LEN_DEFAULT = 2147483647;

// 路由开关: 打开后由 GetFlashAttnTemplateId 按条件路由到不合轴 DN 模板
// (kernel/block *_dn_unmerged.h 文件, templateId=2), host tiling 与 flash_attn_metadata(AICPU)
// 共享该开关, 保证两侧分核选择严格一致; 关闭时模板路由维持原逻辑(ND/合轴DN)。
constexpr bool DN_UNMERGED_ROUTE_ENABLED = true;

/**
 * @brief 根据算子参数决定 sOuter / sInner 切块大小，纯函数，不依赖任何类。
 *
 * @param vHeadDim   V 的 head dim
 * @param gSize      GQA 的 group 数（n1/n2）
 * @param maxSeqQ    Q 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maxSeqKv   KV 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maskMode   mask 模式（0/3/4）
 * @param winLeft    左侧窗口
 * @param winRight   右侧窗口
 * @param sOuterFactor [out] sOuter 切块大小
 * @param sInnerFactor [out] sInner 切块大小
 */
inline void AdjustSinnerAndSouter(uint32_t vHeadDim, uint32_t gSize, int64_t maxSeqQ, int64_t maxSeqKv,
                                  int32_t maskMode, int64_t winLeft, int64_t winRight, uint32_t& sOuterFactor,
                                  uint32_t& sInnerFactor)
{
    if (maxSeqQ == -1) {
        maxSeqQ = MAX_SEQ_LEN_DEFAULT;
    }
    if (maxSeqKv == -1) {
        maxSeqKv = MAX_SEQ_LEN_DEFAULT;
    }
    sOuterFactor = 64;
    sInnerFactor = 128;

    bool checkQueryAndValueS = maxSeqQ <= 64 && maxSeqKv > 128;

    if (vHeadDim <= 128) {
        int64_t winLeftTmp = winLeft;
        int64_t winRightTmp = winRight;
        if (maskMode == 0) {
            winLeftTmp = (winLeftTmp > 0) ? 0 : winLeftTmp;
        } else if (maskMode == 4) {
            winRightTmp = (winRightTmp > 0) ? 0 : winRightTmp;
        }
        bool checkMaskMode = (maskMode != 2 && winLeftTmp + winRightTmp > 128);
        if (checkQueryAndValueS && checkMaskMode) {
            sOuterFactor = 32;
            sInnerFactor = 256;
        }
    }
    if (vHeadDim == 256) {
        if (gSize * maxSeqQ < 64) {
            sOuterFactor = 32;
            sInnerFactor = 256;
        } else {
            sOuterFactor = 64;
            sInnerFactor = 128;
        }
    }
}

// Both tiling and metadata must use the same static bounds, without reading sequence tensors.
// Count dense (B, N2, ceil(Q * G / M), ceil(KV / N)) blocks to retain split-KV parallelism.
// Unknown/empty dimensions keep the platform limit, including the existing empty-output path.
inline uint32_t GetMaxUsedAicCores(uint32_t aicNum, uint32_t batchSize, uint32_t kvHeads, uint32_t groupSize,
                                   int64_t maxSeqQ, int64_t maxSeqKv, uint32_t mBaseSize, uint32_t s2BaseSize)
{
    if (aicNum == 0 || batchSize == 0 || kvHeads == 0 || groupSize == 0 || maxSeqQ <= 0 || maxSeqKv <= 0 ||
        mBaseSize == 0 || s2BaseSize == 0) {
        return aicNum;
    }
    // Cap before multiplying Q by G, so even very large static bounds cannot overflow.
    if (static_cast<uint64_t>(maxSeqQ) > static_cast<uint64_t>(aicNum) * mBaseSize / groupSize) {
        return aicNum;
    }
    const uint64_t mSize = static_cast<uint64_t>(maxSeqQ) * groupSize;
    const uint64_t mBlocks = (mSize - 1) / mBaseSize + 1;
    const uint64_t s2Blocks = (static_cast<uint64_t>(maxSeqKv) - 1) / s2BaseSize + 1;
    uint32_t cores = 1;
    for (uint64_t factor : {static_cast<uint64_t>(batchSize), static_cast<uint64_t>(kvHeads), mBlocks, s2Blocks}) {
        if (factor >= (static_cast<uint64_t>(aicNum) - 1) / cores + 1) {
            return aicNum;
        }
        cores *= static_cast<uint32_t>(factor);
    }
    return cores;
}

// Select the final arch35 template: unmerged DN first, then merged DN, otherwise ND.
// Host tiling and metadata pass the same per-AIV sOuterFactor before applying the CV ratio.
inline uint32_t GetFlashAttnTemplateId(uint32_t qkHeadDim, uint32_t vHeadDim, int64_t maskMode, int64_t maxSeqlenQ,
                                       uint32_t sOuterFactor)
{
    if (DN_UNMERGED_ROUTE_ENABLED && vHeadDim <= 128U && (maskMode == 0 || maskMode == 3) && maxSeqlenQ >= 256) {
        return FA_Template_DN_Unmerged;
    }
    // No-mask arch35 config=0/2/6/8 with sOuter=64 selects merged DN (D=72 uses config=8).
    bool mergedDnDims = (qkHeadDim == vHeadDim && (qkHeadDim == 64 || qkHeadDim == 72 || qkHeadDim == 128)) ||
                        (qkHeadDim == 192 && vHeadDim == 128);
    if (maskMode == 0 && sOuterFactor == 64U && mergedDnDims) {
        return FA_Template_DN;
    }
    return FA_Template_ND;
}

} // namespace fa_tiling_util
} // namespace flash_attn
} // namespace optiling

#endif // FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_ARCH35_H
