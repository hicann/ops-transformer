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
 * \file flash_attn_tiling.cpp
 * \brief
 */

#include "flash_attn_tiling_arch35.h"
#include "../flash_attn_tiling.h"
#include "fa_adjust_sinner_souter_arch35.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../flash_attn_tiling_utils.h"
#include "../../op_kernel/flash_attn_template_tiling_key.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace flash_attn {

void FlashAttnTilingImplArch35::SplitPolicy()
{
    int64_t winLeft = faInfo_->winLeft;
    int64_t winRight = faInfo_->winRight;
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK) ||
        faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        winLeft = MASK_MODE_INT_MAX;
        winRight = MASK_MODE_INT_MAX;
    }
    if (winLeft == -1) {
        winLeft = MASK_MODE_INT_MAX;
    }
    if (winRight == -1) {
        winRight = MASK_MODE_INT_MAX;
    }
    fa_tiling_util::AdjustSinnerAndSouter(
        static_cast<uint32_t>(faInfo_->vHeadDim), static_cast<uint32_t>(faInfo_->gSize), faInfo_->maxSeqQ,
        faInfo_->maxSeqKv, static_cast<int32_t>(faInfo_->maskMode), winLeft, winRight, sOuterFactor_, sInnerFactor_);
    uint32_t maxUsedAicCores = fa_tiling_util::GetMaxUsedAicCores(platformInfo_.aicNum, faInfo_->bSize, faInfo_->n2Size,
                                                                  faInfo_->gSize, faInfo_->maxSeqQ, faInfo_->maxSeqKv,
                                                                  sOuterFactor_ * platformInfo_.cvRatio, sInnerFactor_);
    CalcNumBlocks(maxUsedAicCores);
    flashDecodeFlag_ = true;
}

void FlashAttnTilingImplArch35::UpdateTilingKeyConfig()
{
    if (faInfo_->qkHeadDim == faInfo_->vHeadDim) {
        // 等长场景：原始 D=qkHeadDim、原始 DV=vHeadDim；模板 D/DV 为 ConfigParams.d/dv。
        // originalDV 单独保留原始 V/输出维度，Q/K 的原始 D 仍由运行时 qkHeadDim 提供。
        // config | 原始 D | 原始 DV | 模板 D | 模板 DV | originalDV | sOuter | sInner
        //      0 |     64 |      64 |     64 |      64 |         64 |     64 |    128
        //      1 |     64 |      64 |     64 |      64 |         64 |     32 |    256
        //      2 |    128 |     128 |    128 |     128 |        128 |     64 |    128
        //      3 |    128 |     128 |    128 |     128 |        128 |     32 |    256
        //      4 |    256 |     256 |    256 |     256 |        256 |     64 |    128
        //      5 |    256 |     256 |    256 |     256 |        256 |     32 |    256
        //      8 |     72 |      72 |    128 |     128 |         72 |     64 |    128
        //      9 |     72 |      72 |    128 |     128 |         72 |     32 |    256
        if (faInfo_->qkHeadDim == 64) {
            tilingKeyInfo_.config = (sOuterFactor_ == 64) ? 0 : 1;
        } else if (faInfo_->qkHeadDim == 128) {
            tilingKeyInfo_.config = (sOuterFactor_ == 64) ? 2 : 3;
        } else if (faInfo_->qkHeadDim == 72) {
            // 真实 Dk=Dv=72；模板 D=DV=128，originalDV=72，sOuter/sInner 切分逻辑沿用原 128 档
            tilingKeyInfo_.config = (sOuterFactor_ == 64) ? 8 : 9;
        } else if (faInfo_->qkHeadDim == 256) {
            tilingKeyInfo_.config = (sOuterFactor_ == 64) ? 4 : 5;
        }
    } else {
        // 不等长 (qkHeadDim != vHeadDim) 按 (QK D, V DV) 组合映射:
        if (faInfo_->qkHeadDim == 192 && faInfo_->vHeadDim == 128) {
            // MLA 非吸收场景：原始 D=192、原始 DV=128；模板 D=192、模板 DV=128、originalDV=128。
            // config | 原始 D | 原始 DV | 模板 D | 模板 DV | originalDV | sOuter | sInner
            //      6 |    192 |     128 |    192 |     128 |        128 |     64 |    128
            //      7 |    192 |     128 |    192 |     128 |        128 |     32 |    256
            tilingKeyInfo_.config = (sOuterFactor_ == 64) ? 6 : 7;
        }
    }
}

void FlashAttnTilingImplArch35::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    UpdateTilingKeyKvLayout();
    tilingKeyInfo_.hasAttenMask = (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) ? 0 : 1;
    UpdateTilingKeyTemplateId();
}

void FlashAttnTilingImplArch35::UpdateTilingKeyTemplateId()
{
    // 与 metadata 共享三路选择：不合轴 DN 优先，其次合轴 DN，最后 ND；D=72 已由 config 8/9 区分。
    // dnFlag_ 仅表示合轴 DN，不合轴 DN 使用独立的 metadata 分核模式。
    tilingKeyInfo_.templateId = fa_tiling_util::GetFlashAttnTemplateId(
        static_cast<uint32_t>(faInfo_->qkHeadDim), static_cast<uint32_t>(faInfo_->vHeadDim), faInfo_->maskMode,
        faInfo_->maxSeqQ, sOuterFactor_);
    dnFlag_ = tilingKeyInfo_.templateId == FA_Template_DN;
}

} // namespace flash_attn

using flash_attn::FlashAttnTilingImplArch35;

// 值越小表示优先级越高
REGISTER_TILING_TEMPLATE_FIA(FlashAttn, FlashAttnTilingImplArch35,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 1);
} // namespace optiling
