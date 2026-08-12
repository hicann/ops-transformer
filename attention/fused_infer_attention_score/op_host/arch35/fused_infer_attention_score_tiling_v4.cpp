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
 * \file fused_infer_attention_score_tiling_v4.cpp
 * \brief
 */

#include "../../../common/op_host/fia_tiling_templates_registry.h"
#include "fused_infer_attention_score_tiling_v4.h"
#include "../fused_infer_attention_score_tiling_info_parser.h"
#include "../checkers/fia_checker.h"

namespace optiling {
ge::graphStatus TilingFusedInferAttentionScoreV4(gert::TilingContext *context) {
    // Parse -> Check -> DoOpTiling
    FiaTilingInfo fiaInfo;
    FiaInfoParser fiaInfoParser(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FIAChecker fiaChecker;
    fiaChecker.Init(fiaInfo);
    // Check函数只做校验，不能修改fiaInfo中的信息
    if (fiaChecker.Process(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo);
}

} // namespace optiling