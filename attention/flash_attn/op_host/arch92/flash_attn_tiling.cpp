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

#include "flash_attn_tiling_arch92.h"
#include "../flash_attn_tiling.h"
#include "fa_adjust_sinner_souter_arch92.h"
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

void FlashAttnTilingImplArch92::SplitPolicy()
{
    int64_t winLeft = faInfo_->winLeft;
    int64_t winRight = faInfo_->winRight;
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK) ||
        faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        winLeft = MASK_MODE_INT_MAX;
        winRight = MASK_MODE_INT_MAX;
    }
    fa_tiling_util::GetConfigInfo(static_cast<uint32_t>(faInfo_->vHeadDim), static_cast<int32_t>(faInfo_->maskMode),
                                  faInfo_->maxSeqQ, templateId_, sOuterFactor_, sInnerFactor_);

    OP_LOGI(faInfo_->opName, "vHeadDim:%d, maskMode:%d, maxSeqQ:%d, templateId_:%d, sOuterFactor_:%d, sInnerFactor_:%d",
            faInfo_->vHeadDim, faInfo_->maskMode, faInfo_->maxSeqQ, templateId_, sOuterFactor_, sInnerFactor_);

    CalcNumBlocks(platformInfo_.aicNum);
    flashDecodeFlag_ = true;
}

void FlashAttnTilingImplArch92::UpdateTilingKeyConfig()
{
    if (tilingKeyInfo_.templateId == FA_Template_ND) {
        if (faInfo_->qkHeadDim == 64) {
            tilingKeyInfo_.config = 1; // (M=64, S2=256, D=64)
        } else if (faInfo_->qkHeadDim == 128) {
            tilingKeyInfo_.config = 2; // (M=128, S2=128, D=128)
        } else if (faInfo_->qkHeadDim == 256) {
            tilingKeyInfo_.config = 4; // (M=64, S2=256, D=256)
        } else {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(faInfo_->opName, "qkHeadDim(Head num of Q/K)",
                                                   std::to_string(faInfo_->qkHeadDim).c_str(),
                                                   "The value of qkHeadDim(Head num of Q/K) can only be 64/128/256");
        }
    } else { // FA_Template_DN
        if (faInfo_->qkHeadDim == 64) {
            tilingKeyInfo_.config = 1; // (M=64, S2=256, D=64)
        } else if (faInfo_->qkHeadDim == 128) {
            tilingKeyInfo_.config = 3; // (M=128, S2=256, D=128)
        } else if (faInfo_->qkHeadDim == 256) {
            tilingKeyInfo_.config = 4; // (M=64, S2=256, D=256)
        } else {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(faInfo_->opName, "qkHeadDim(Head num of Q/K)",
                                                   std::to_string(faInfo_->qkHeadDim).c_str(),
                                                   "The value of qkHeadDim(Head num of Q/K) can only be 64/128/256");
        }
    }
}

void FlashAttnTilingImplArch92::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyKvLayout();
    tilingKeyInfo_.hasAttenMask = (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) ? 0 : 1;
    tilingKeyInfo_.templateId = templateId_;

    UpdateTilingKeyConfig();
}

} // namespace flash_attn

using flash_attn::FlashAttnTilingImplArch92;

// 值越小表示优先级越高
REGISTER_TILING_TEMPLATE_FIA(FlashAttn, FlashAttnTilingImplArch92,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_9201)}), 1);
} // namespace optiling
