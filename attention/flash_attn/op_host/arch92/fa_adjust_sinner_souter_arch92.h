/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_ARCH92_H_
#define FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_ARCH92_H_

#include <cstdint>
#include <initializer_list>
#include "../../op_kernel/utils/flash_attn_common_def.h"

namespace optiling {
namespace flash_attn {
namespace fa_tiling_util {

inline void SetMBaseSize(uint32_t vHeadDim, uint32_t& mBaseSize)
{
    if (vHeadDim == 128) {
        mBaseSize = 128;
    } else {
        mBaseSize = 64;
    }
}

inline void SetKernelTemplateId(uint32_t vHeadDim, int32_t maxSeqQ, uint32_t mBaseSize, uint32_t maskMode,
                                uint32_t& templateId)
{
    if (maxSeqQ == -1) { // 默认走ND模板，GS1合轴
        templateId = FA_Template_ND;
        return;
    }

    if (vHeadDim == 128) {
        if (maxSeqQ < mBaseSize) { // maxSeqQ < mBaseSize时，走ND模板，GS1合轴打满mBase，达成cube bound
            templateId = FA_Template_ND;
        } else if (maskMode == 4U) { // MaskMode::BAND: DN模板S2切256，band模式ub不足，走ND模板
            templateId = FA_Template_ND;
        } else {
            templateId = FA_Template_DN_Unmerged;
        }
    } else if (vHeadDim == 64) {
        templateId = FA_Template_ND;
    } else { // D=256
        templateId = FA_Template_ND;
    }
}

inline void SetS2BaseSize(uint32_t templateId, uint32_t vHeadDim, uint32_t& s2BaseSize)
{
    s2BaseSize = 128;
    if (templateId == FA_Template_ND) {
        if (vHeadDim == 256 || vHeadDim == 64) {
            s2BaseSize = 256;
        }
    } else {
        if (vHeadDim == 128) {
            s2BaseSize = 256;
        }
    }
}

/**
 * @brief 根据算子参数决定模板类型及切块大小，纯函数，不依赖任何类。
 *
 * @param vHeadDim    V 的 head dim
 * @param maskMode    mask 模式（0/3/4）
 * @param maxSeqQ     Q 的最大序列长度，-1 表示默认值
 * @param templateId  [out] 模板类型（FA_Template_ND / FA_Template_DN）
 * @param mBaseSize   [out] mBase 切块大小
 * @param s2BaseSize  [out] s2 切块大小
 */
inline void GetConfigInfo(uint32_t vHeadDim, int32_t maskMode, int32_t maxSeqQ, uint32_t& templateId,
                          uint32_t& mBaseSize, uint32_t& s2BaseSize)
{
    SetMBaseSize(vHeadDim, mBaseSize);
    SetKernelTemplateId(vHeadDim, maxSeqQ, mBaseSize, maskMode, templateId);
    SetS2BaseSize(templateId, vHeadDim, s2BaseSize);
}

} // namespace fa_tiling_util
} // namespace flash_attn
} // namespace optiling

#endif // FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_ARCH92_H_
