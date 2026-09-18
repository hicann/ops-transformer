/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_H
#define FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_H

#include <cstdint>
#include <initializer_list>

namespace optiling {
namespace flash_attn {
namespace fa_tiling_util {

constexpr int64_t MAX_SEQ_LEN_DEFAULT = 2147483647;

// layout 枚举值，与 FiaLayout 一致，供外部算子使用
constexpr uint32_t LAYOUT_BSH = 0;
constexpr uint32_t LAYOUT_BSND = 1;
constexpr uint32_t LAYOUT_BNSD = 2;
constexpr uint32_t LAYOUT_TND = 4;

// tiling 切块常量
constexpr uint32_t SOUTER_32 = 32;
constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SINNER_128 = 128;
constexpr uint32_t SINNER_256 = 256;
constexpr uint32_t DSIZE_128 = 128;
constexpr uint32_t DSIZE_256 = 256;

/**
 * @brief 根据算子参数决定 sOuter / sInner 切块大小，纯函数，不依赖任何类。
 *
 * @param vHeadDim   V 的 head dim
 * @param gSize      GQA 的 group 数（n1/n2）
 * @param maxSeqQ    Q 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maxSeqKv   KV 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maskMode   mask 模式（0/2/4 等）
 * @param winLeft    左侧窗口
 * @param winRight   右侧窗口
 * @param qLayout    Q 的 layout（使用 LAYOUT_BSH / LAYOUT_BSND / LAYOUT_TND 等）
 * @param sOuterFactor [out] sOuter 切块大小
 * @param sInnerFactor [out] sInner 切块大小
 */
inline void AdjustSinnerAndSouter(uint32_t vHeadDim, uint32_t gSize, int64_t maxSeqQ, int64_t maxSeqKv,
                                  int32_t maskMode, int64_t winLeft, int64_t winRight, uint32_t qLayout,
                                  uint32_t &sOuterFactor, uint32_t &sInnerFactor)
{
    if (maxSeqQ == -1) {
        maxSeqQ = MAX_SEQ_LEN_DEFAULT;
    }
    if (maxSeqKv == -1) {
        maxSeqKv = MAX_SEQ_LEN_DEFAULT;
    }
    sOuterFactor = SOUTER_64;
    sInnerFactor = SINNER_128;

    bool checkQueryAndValueS = maxSeqQ <= SOUTER_64 && maxSeqKv > SINNER_128;

    if (vHeadDim <= DSIZE_128) {
        int64_t winLeftTmp = winLeft;
        int64_t winRightTmp = winRight;
        if (maskMode == 0) {
            winLeftTmp = (winLeftTmp > 0) ? 0 : winLeftTmp;
        } else if (maskMode == 4) {
            winRightTmp = (winRightTmp > 0) ? 0 : winRightTmp;
        }
        bool checkSparseMode = (maskMode != 2 && winLeftTmp + winRightTmp > 128);
        if (checkQueryAndValueS && checkSparseMode) {
            sOuterFactor = SOUTER_32;
            sInnerFactor = SINNER_256;
        }
    }
    if (vHeadDim == DSIZE_256) {
        if (gSize * maxSeqQ < SOUTER_64) {
            sOuterFactor = SOUTER_32;
            sInnerFactor = SINNER_256;
        } else {
            sOuterFactor = SOUTER_64;
            sInnerFactor = SINNER_128;
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

constexpr uint32_t FA_TEMPLATE_ND = 0;
constexpr uint32_t FA_TEMPLATE_DN = 1;
constexpr uint32_t FA_TEMPLATE_DN_UNMERGED = 2; // 不合轴 DN(templateId=2), 预埋路由入口默认关闭

// 预埋开关: false 时模板路由维持原逻辑(ND/合轴DN), 不影响现网行为;
// 打开后所有 case 路由到不合轴 DN 新模板(kernel/block *_dn_unmerged.h 文件, templateId=2),
// host tiling 与 flash_attn_metadata(AICPU) 共享该开关, 保证两侧分核选择严格一致。
constexpr bool DN_UNMERGED_ROUTE_ENABLED = false;

// DN 模板路由的共享判定: host tiling 与 flash_attn_metadata(AICPU) 必须使用同一份逻辑,
// 保证 metadata 的分核 outputLayout 选择与 kernel 模板选择严格一致。
// 全量路由: 所有 case 均走 DN 模板; 接口层已限定 D/DV 注册组合, 此处注册表仅作
// 未注册维度的防御性回退, mask/layout/S1 大小不设任何路由限制。
inline uint32_t GetFlashAttnTemplateId(uint32_t qkHeadDim, uint32_t vHeadDim)
{
    bool supportedEqualDim =
        (qkHeadDim == vHeadDim) && (qkHeadDim == 64 || qkHeadDim == 72 || qkHeadDim == 128 || qkHeadDim == 256);
    bool supportedMlaDim = (qkHeadDim == 192) && (vHeadDim == 128);
    return (supportedEqualDim || supportedMlaDim) ? FA_TEMPLATE_DN_UNMERGED : FA_TEMPLATE_ND;
}

} // namespace fa_tiling_util
} // namespace flash_attn
} // namespace optiling

#endif // FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_H
