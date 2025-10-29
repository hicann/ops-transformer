/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_v3.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling_v3.h"
#include "fused_infer_attention_score_tiling_check.h"
#include "fused_infer_attention_score_tiling_info_parser.h"
#include "../../common/op_host/arch32/fia_tiling_nonquant_mla.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"

using namespace AscendC;
namespace optiling {
FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context)
{
    FiaTilingInfo fiaInfo;
    FiaInfoParser fiaInfoParser(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check函数只做校验，不能修改fiaInfo中的信息
    if (TilingCheck::Check(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo);
}

bool RouteToFia(gert::TilingContext *context, IncreFlashAttentionContext &ifaContext)
{
    if ((context == nullptr) || (ifaContext.query.desc == nullptr) || (ifaContext.key.desc == nullptr)) {
        return false;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        return false;
    }

    ge::DataType qDataType = ifaContext.query.desc->GetDataType();
    ge::DataType kDataType = ifaContext.key.desc->GetDataType();
    bool isMla = (ifaContext.keyRope.tensor != nullptr && ifaContext.queryRope.tensor != nullptr);
    if (isMla) {
        // MLA非量化
        if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
            return true;
        }
        // MLA BF16伪量化
        if ((qDataType == ge::DT_BF16) && (kDataType == ge::DT_INT8)) {
            return false;
        }
        // MLA FP16伪量化
        if ((qDataType == ge::DT_FLOAT16) && (kDataType == ge::DT_INT8)) {
            return false;
        }
        // MLA全量化
        if ((qDataType == ge::DT_INT8) && (kDataType == ge::DT_INT8)) {
            return false;
        }
    }
    return false;
}

} // namespace optiling
