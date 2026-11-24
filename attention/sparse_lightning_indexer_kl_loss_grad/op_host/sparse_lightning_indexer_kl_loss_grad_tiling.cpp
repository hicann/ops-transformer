#include "log/log.h"
/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file sparse_lightning_indexer_kl_loss_grad_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "register/op_def_registry.h"
#include "../op_kernel/sparse_lightning_indexer_kl_loss_grad_template_tiling_key.h"
#include "../op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_tiling.h"
#include "sparse_lightning_indexer_kl_loss_grad_tiling_general.h"
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "sparse_lightning_indexer_kl_loss_grad_tiling_common.h"

using std::map;
using std::pair;
using std::string;

using namespace ge;
using namespace AscendC;

namespace optiling {
constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string BLOCK_TABLE_NAME = "block_table";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string QUERY_ROPE_NAME = "query_rope";
static const std::string KEY_ROPE_NAME = "key_rope";
static const std::string ATTEN_OUT_NAME = "attention_out";

static ge::graphStatus CheckRequiredInputsNotEmpty(gert::TilingContext* context)
{
    constexpr uint32_t requiredInputIndices[] = {QUERY_INPUT_INDEX, KEY_INPUT_INDEX, WEIGHT_INPUT_INDEX,
                                                 SPARSE_INDICES_INPUT_INDEX, ATTN_SOFTMAX_L1_NORM_INPUT_INDEX};
    constexpr const char* requiredInputNames[] = {"q", "k", "w", "sparse_indices", "attn_softmax_l1_norm"};

    for (size_t inputIdx = 0; inputIdx < sizeof(requiredInputIndices) / sizeof(requiredInputIndices[0]); ++inputIdx) {
        const gert::StorageShape* inputShape = context->GetInputShape(requiredInputIndices[inputIdx]);
        OP_CHECK_IF(inputShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), requiredInputNames[inputIdx],
                                                             "required input shape is missing"),
                    return ge::GRAPH_FAILED);
        const gert::Shape& storageShape = inputShape->GetStorageShape();
        for (size_t axis = 0; axis < storageShape.GetDimNum(); ++axis) {
            OP_CHECK_IF(storageShape.GetDim(axis) == 0,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context->GetNodeName(), requiredInputNames[inputIdx],
                            std::to_string(storageShape.GetDim(axis)).c_str(),
                            ("empty dimension at axis " + std::to_string(axis) + " is not supported").c_str()),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSeqUsedShapes(gert::TilingContext* context)
{
    if (context->GetOptionalInputShape(SEQUSED_QUERY_INPUT_INDEX) == nullptr &&
        context->GetOptionalInputShape(SEQUSED_KEY_INPUT_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const auto* queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const auto& qShape = queryShape->GetStorageShape();
    const auto queryRank = qShape.GetDimNum();
    OP_CHECK_IF(queryRank != DIM_NUM_THREE && queryRank != DIM_NUM_FOUR,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "q", std::to_string(queryRank).c_str(),
                                                         "Query must be rank 3 for TND or rank 4 for BSND"),
                return ge::GRAPH_FAILED);

    int64_t batchSize = qShape.GetDim(0);
    if (queryRank == DIM_NUM_THREE) {
        const auto* cuSeqShape = context->GetOptionalInputShape(CU_SEQLENS_QUERY_INPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqShape);
        const auto& cuShape = cuSeqShape->GetStorageShape();
        OP_CHECK_IF(cuShape.GetDimNum() != 1 || cuShape.GetDim(0) <= 1,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "cu_seqlens_q",
                        "TND requires a one-dimensional cumulative sequence length tensor with B+1 elements and B > 0"),
                    return ge::GRAPH_FAILED);
        batchSize = cuShape.GetDim(0) - 1;
    }

    const uint32_t inputIndices[] = {SEQUSED_QUERY_INPUT_INDEX, SEQUSED_KEY_INPUT_INDEX};
    const char* inputNames[] = {"seqused_q", "seqused_k"};
    for (size_t index = 0; index < 2; ++index) {
        const auto* shape = context->GetOptionalInputShape(inputIndices[index]);
        if (shape == nullptr) {
            continue;
        }
        const auto& storageShape = shape->GetStorageShape();
        const std::string shapeInfo = "rank=" + std::to_string(storageShape.GetDimNum()) +
                                      ", size=" + std::to_string(storageShape.GetShapeSize());
        const std::string reason =
            "Sequence lengths must be one-dimensional tensors with shape (B,), where B=" + std::to_string(batchSize);
        OP_CHECK_IF(storageShape.GetDimNum() != 1 || storageShape.GetDim(0) != batchSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), inputNames[index], shapeInfo.c_str(),
                                                          reason.c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSparseLightningIndexerKLLossGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        CheckRequiredInputsNotEmpty(context) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context, "SparseLightningIndexerKLLossGrad does not support empty input tensors."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckSeqUsedShapes(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context, "Invalid seqused_q or seqused_k shape."), return ge::GRAPH_FAILED);

    auto platformInfoPtr = context->GetPlatformInfo();
    auto sligPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (sligPlatform.GetCurNpuArch() == Ops::Base::DAV_3510) {
        OP_LOGW(context, "Current npu arch is dav-3510.");
        return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
    } else if (sligPlatform.GetCurNpuArch() == Ops::Base::DAV_9201 ||
               sligPlatform.GetCurNpuArch() == Ops::Base::DAV_9202) {
        OP_LOGW(context, "Current npu arch is dav-9201 or dav-9202.");
        return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
    } else {
        OP_LOGW(context, "Current npu arch is not dav-3510.");
        SparseLightningIndexerKLLossGradTilingBase sligKLLossTiling(context);
        return sligKLLossTiling.DoTiling();
    }
}

ge::graphStatus TilingPrepareForSparseLightningIndexerKLLossGrad(gert::TilingParseContext* context)
{
    OP_LOGW(context, "Start registering tiling.");
    auto compileInfoPtr = context->GetCompiledInfo<SparseLightningIndexerKLLossGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(), "compileInfoPtr is null"})),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseLightningIndexerKLLossGrad)
    .Tiling(TilingSparseLightningIndexerKLLossGrad)
    .TilingParse<SparseLightningIndexerKLLossGradCompileInfo>(TilingPrepareForSparseLightningIndexerKLLossGrad);
} // namespace optiling
