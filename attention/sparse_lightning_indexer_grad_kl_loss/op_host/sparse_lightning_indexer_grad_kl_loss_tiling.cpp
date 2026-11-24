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
 * \file sparse_lightning_indexer_grad_kl_loss_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "sparse_lightning_indexer_grad_kl_loss_tiling_common.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;

namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

static ge::graphStatus CheckRopeDtypesAndSoftmaxRanks(gert::TilingContext* context)
{
    const uint32_t ropeIndices[] = {QUERY_ROPE_INPUT_INDEX, KEY_ROPE_INPUT_INDEX};
    const char* ropeNames[] = {"query_rope", "key_rope"};
    for (size_t index = 0; index < 2; ++index) {
        if (context->GetOptionalInputShape(ropeIndices[index]) == nullptr) {
            continue;
        }
        const auto* desc = context->GetOptionalInputDesc(ropeIndices[index]);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        const auto dtype = desc->GetDataType();
        OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), ropeNames[index],
                                                          ge::TypeUtils::DataTypeToSerialString(dtype).c_str(),
                                                          "RoPE inputs must have FLOAT16 or BFLOAT16 dtype"),
                    return ge::GRAPH_FAILED);
    }

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* layout = attrs->GetAttrPointer<char>(LAYOUT_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layout);
    const size_t expectedRank = std::string(layout) == "TND" ? DIM_NUM_THREE : DIM_NUM_FOUR;
    const uint32_t softmaxIndices[] = {SOFTMAX_MAX_INPUT_INDEX, SOFTMAX_SUM_INPUT_INDEX};
    const char* softmaxNames[] = {"softmax_max", "softmax_sum"};
    for (size_t index = 0; index < 2; ++index) {
        const auto* shape = context->GetInputShape(softmaxIndices[index]);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        const auto rank = shape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(rank != expectedRank,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), softmaxNames[index], std::to_string(rank).c_str(),
                        "Softmax inputs must be rank 3 for TND or rank 4 for BSND"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSparseLightningIndexerGradKLLoss(gert::TilingContext* context)
{
    OP_CHECK_IF(CheckRopeDtypesAndSoftmaxRanks(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context, "Invalid RoPE dtype or softmax rank."), return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    auto sligPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto npuArch = sligPlatform.GetCurNpuArch();
    if (sligPlatform.GetCurNpuArch() == Ops::Base::DAV_3510) {
        OP_LOGW(context, "Current npu arch is dav-3510.");
    } else if (npuArch == Ops::Base::DAV_9201 || npuArch == Ops::Base::DAV_9202) {
        OP_LOGW(context, "Current npu arch is dav-9201 or dav-9202.");
    } else {
        OP_LOGW(context, "Current npu arch is not dav3510 or dav9201 or dav9202.");
    }
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForSparseLightningIndexerGradKLLoss(gert::TilingParseContext* context)
{
    OP_LOGW(context, "Start registering tiling.");
    auto compileInfoPtr = context->GetCompiledInfo<SparseLightningIndexerGradKLLossCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(), "compileInfoPtr is null"})),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfoPtr == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context->GetNodeName(), "platformInfoPtr is null"})),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseLightningIndexerGradKLLoss)
    .Tiling(TilingSparseLightningIndexerGradKLLoss)
    .TilingParse<SparseLightningIndexerGradKLLossCompileInfo>(TilingPrepareForSparseLightningIndexerGradKLLoss);
} // namespace optiling
