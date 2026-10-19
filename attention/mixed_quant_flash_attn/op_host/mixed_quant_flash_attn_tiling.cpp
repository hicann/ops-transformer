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
 * \file mixed_quant_flash_attn_tiling.cpp
 * \brief MixedQuantFlashAttn Tiling主入口
 */

#include <cmath>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "mixed_quant_flash_attn_tiling.h"
#include "mixed_quant_flash_attn_tiling_common.h"
#include "mixed_quant_flash_attn_tiling_info_parser.h"
#include "checkers/mqfa_fa_checker.h"
#include "checkers/sinks_checker.h"
#include "../op_kernel/mixed_quant_flash_attn_template_tiling_key.h"
#include "../op_kernel/mixed_quant_flash_attn_tiling_data.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
using namespace mixed_quant_flash_attn;

static ge::graphStatus TilingEmptyInput(gert::TilingContext* context, const FaTilingInfo& faInfo,
                                        platform_ascendc::PlatformAscendC& platform)
{
    uint32_t aivNum = platform.GetCoreNumAiv();
    uint32_t aicNum = platform.GetCoreNumAic();
    OP_CHECK_IF(aivNum == 0 || aicNum == 0, OP_LOGE(faInfo.opName, "num of core obtained is 0."),
                return ge::GRAPH_FAILED);
    uint32_t blockDim = platform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    MixedQuantFlashAttnTilingData tilingData{};
    auto& baseParams = tilingData.baseTiling.mixedQuantFlashAttnBaseParams;
    baseParams.bSize = static_cast<uint32_t>(faInfo.bSize);
    baseParams.t1Size = static_cast<uint32_t>(faInfo.qTSize);
    baseParams.n2Size = static_cast<uint32_t>(faInfo.n1Size);
    baseParams.gSize = 1;
    baseParams.s1Size = static_cast<uint32_t>(faInfo.s1Size);
    baseParams.seqUsedQSize = static_cast<uint32_t>(faInfo.seqUsedQSize);
    baseParams.seqUsedKvSize = static_cast<uint32_t>(faInfo.seqUsedKvSize);
    baseParams.cuSeqLensQSize = static_cast<uint32_t>(faInfo.cuSeqLensQSize);
    baseParams.isSoftMaxLseEnable = faInfo.softmaxLseFlag;
    baseParams.coreNum = blockDim;
    baseParams.outputLayout =
        faInfo.qLayout == FaLayout::TND ? static_cast<uint32_t>(FA_LAYOUT::TND) : static_cast<uint32_t>(faInfo.qLayout);
    baseParams.needInitOutput = true;
    baseParams.emptyTensor = true;
    baseParams.hasSink = faInfo.sinksFlag;
    baseParams.totalOutputSize = faInfo.totalOutputSize;
    baseParams.totalLseSize = faInfo.totalLseSize;

    auto tiling = context->GetTilingData<MixedQuantFlashAttnTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(faInfo.opName, "The tiling data is nullptr."), return ge::GRAPH_FAILED);
    *tiling = tilingData;

    uint64_t tilingKey;
    if (platform.GetCurNpuArch() == NpuArch::DAV_3510) {
        tilingKey =
            GET_TPL_TILING_KEY(InOutLayoutType_BSND, KvLayoutType_PA_BBH, false,
                               Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128, A16C4_KV_MXFP4_SOFTMAX_FP32);
    } else {
        tilingKey =
            GET_TPL_TILING_KEY(InOutLayoutType_BSND, KvLayoutType_NO_PA, false,
                               Config_S1Aligned32_S2Aligned512_DAligned128_DVAligned128, A16C4_KV_MXFP4_SOFTMAX_FP32);
    }
    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr, OP_LOGE(faInfo.opName, "Workspace size buffer is nullptr."),
                return ge::GRAPH_FAILED);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(context->SetScheduleMode(static_cast<uint32_t>(ScheduleMode::BATCH_MODE)) != ge::GRAPH_SUCCESS,
                OP_LOGE(faInfo.opName, "Set schedule mode failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingMixedQuantFlashAttn(gert::TilingContext* context)
{
    OP_LOGW(context, "MixedQuantFlashAttn TilingMixedQuantFlashAttn start.");

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    FaTilingInfo faInfo;
    FaInfoParser faInfoParser(context);
    if (faInfoParser.Parse(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (faInfo.emptyTensorFlag) {
        return TilingEmptyInput(context, faInfo, ascendcPlatform);
    }

    FAChecker faChecker;
    faChecker.Init(faInfo);
    if (faChecker.Process(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &faInfo);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMixedQuantFlashAttn(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<MixedQuantFlashAttnCompileInfo>();
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

// 注册tiling函数：
IMPL_OP_OPTILING(MixedQuantFlashAttn)
    .Tiling(TilingMixedQuantFlashAttn)

    .TilingParse<MixedQuantFlashAttnCompileInfo>(TilingPrepareForMixedQuantFlashAttn);

} // namespace optiling
