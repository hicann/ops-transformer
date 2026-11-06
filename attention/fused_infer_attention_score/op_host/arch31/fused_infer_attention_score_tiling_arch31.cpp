/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file fused_infer_attention_score_tiling_arch31.cpp
 * \brief arch31 (KirinX90 / Kirin9030) Tiling 入口实现
 */

#include "fused_infer_attention_score_tiling_arch31.h"
#include "fia_tiling_nonquant_arch31.h"
#include "fused_infer_attention_score_tiling_check_arch31.h"
#include "../fused_infer_attention_score_tiling.h"
#include "../fia_tiling_info.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

// TilingKey 注册
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000000000, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000000256, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000000512, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000000768, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000001024, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000002304, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000008960, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000010496, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000017152, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000018688, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000024576, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000026880, FusedInferAttentionScoreTilingData)

// HighPerformance
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000000257, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000000258, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000000259, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_6000000000000000260, FusedInferAttentionScoreTilingData)

// arch31 Tiling 入口
ge::graphStatus TilingFusedInferAttentionScoreArch31(gert::TilingContext* context)
{
    FusedInferAttentionScoreTilingArch31 tilingArch31(context);
    return tilingArch31.DoOpTiling();
}

ge::graphStatus FusedInferAttentionScoreTilingArch31::DoOpTiling()
{
    // 1. 平台信息获取（Host 侧）
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context_->GetNodeName(), "GetPlatformInfo returned nullptr"})),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    // 2. arch31 专属 tiling 检查
    OP_CHECK_IF(
        CheckArch31Inputs(context_) != ge::GRAPH_SUCCESS,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context_->GetNodeName(), "arch31 tiling check failed"})),
        return ge::GRAPH_FAILED);
    // 3. 单核 BLOCK_DIM=1
    context_->SetBlockDim(1);

    // 4. 调用arch31 non-quant tiling 实现
    FiaTilingNonQuantArch31 fiaTiling(context_);
    OP_CHECK_IF(fiaTiling.DoTiling() != ge::GRAPH_SUCCESS,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context_->GetNodeName(), "FiaTilingNonQuantArch31 failed"})),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
} //  namespace optiling
