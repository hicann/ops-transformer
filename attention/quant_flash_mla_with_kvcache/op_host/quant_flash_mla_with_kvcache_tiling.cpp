/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_with_kvcache_tiling.cpp
 * \brief QuantFlashMlaWithKvcache Tiling主入口
 */

#include <cmath>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "quant_flash_mla_with_kvcache_tiling.h"
#include "qmla_tiling_info_parser.h"
#include "checkers/qmla_checker.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
using namespace quant_flash_mla_with_kvcache;

ASCENDC_EXTERN_C ge::graphStatus TilingQuantFlashMlaWithKvcache(gert::TilingContext* context)
{
    OP_LOGI(context, "QuantFlashMlaWithKvcache TilingQuantFlashMlaWithKvcache start.");

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    QmlaTilingInfo qmlaInfo;
    QmlaInfoParser qmlaInfoParser(context);
    if (qmlaInfoParser.Parse(qmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QmlaChecker qmlaChecker;
    qmlaChecker.Init(qmlaInfo);
    if (qmlaChecker.Process(qmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &qmlaInfo);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForQuantFlashMlaWithKvcache(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<QuantFlashMlaWithKvcacheCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantFlashMlaWithKvcache)
    .Tiling(TilingQuantFlashMlaWithKvcache)
    .TilingParse<QuantFlashMlaWithKvcacheCompileInfo>(TilingPrepareForQuantFlashMlaWithKvcache);

} // namespace optiling
