/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_grad_tiling.cpp
 * \brief
 */

#include "sparse_flash_attention_grad_tiling.h"
#include <register/op_impl_registry.h>
#include "op_host/data_copy_transpose_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/arch35/sparse_flash_attention_grad_tiling_data_regbase.h"
#include "../op_kernel/arch35/sparse_flash_attention_grad_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
namespace sfag {

ASCENDC_EXTERN_C ge::graphStatus TilingSparseFlashAttentionGrad(gert::TilingContext* context)
{
    auto platform = context->GetPlatformInfo();
    auto sfagPlatform = platform_ascendc::PlatformAscendC(platform);
    if (sfagPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGW(context, "Current npu arch is dav-3510.");
    } else {
        OP_LOGW(context, "Current npu arch is not dav-3510.");
    }
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForSparseFlashAttentionGrad(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttentionGrad", "context is null."), return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "platformInfoPtr is null."),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<SparseFlashAttentionGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    OP_LOGI(context,
            "parse TilingParseContext succ. aivNum:%u, aicNum:%u, "
            "ubSize:%lu, l1Size:%lu, l0aSize:%lu, l0bSize:%lu, l0cSize:%lu, l2CacheSize:%lu",
            compileInfoPtr->aivNum, compileInfoPtr->aicNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
            compileInfoPtr->l0aSize, compileInfoPtr->l0bSize, compileInfoPtr->l0cSize, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseFlashAttentionGrad)
    .Tiling(TilingSparseFlashAttentionGrad)
    .TilingParse<SparseFlashAttentionGradCompileInfo>(TilingPrepareForSparseFlashAttentionGrad); // 向框架注册入口函数
} // namespace sfag
} // namespace optiling
