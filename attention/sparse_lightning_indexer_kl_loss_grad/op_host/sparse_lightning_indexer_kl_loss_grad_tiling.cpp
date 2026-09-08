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

static ge::graphStatus CheckRequiredInputsNotEmpty(gert::TilingContext *context)
{
    constexpr uint32_t requiredInputIndices[] = {QUERY_INPUT_INDEX, KEY_INPUT_INDEX, WEIGHT_INPUT_INDEX,
                                                 SPARSE_INDICES_INPUT_INDEX, ATTN_SOFTMAX_L1_NORM_INPUT_INDEX};
    constexpr const char *requiredInputNames[] = {"q", "k", "w", "sparse_indices", "attn_softmax_l1_norm"};

    for (size_t inputIdx = 0; inputIdx < sizeof(requiredInputIndices) / sizeof(requiredInputIndices[0]); ++inputIdx) {
        const gert::StorageShape *inputShape = context->GetInputShape(requiredInputIndices[inputIdx]);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
        const gert::Shape &storageShape = inputShape->GetStorageShape();
        for (size_t axis = 0; axis < storageShape.GetDimNum(); ++axis) {
            OP_CHECK_IF(storageShape.GetDim(axis) == 0,
                        OP_LOGE(context, "Input %s has an empty dimension at axis %zu, which is not supported.",
                                requiredInputNames[inputIdx], axis),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSparseLightningIndexerKLLossGrad(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(CheckRequiredInputsNotEmpty(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SparseLightningIndexerKLLossGrad does not support empty input tensors."),
                return ge::GRAPH_FAILED);

    auto platformInfoPtr = context->GetPlatformInfo();
    auto sligPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (sligPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGW(context, "Current npu arch is dav-3510.");
        return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
    } else {
        OP_LOGW(context, "Current npu arch is not dav-3510.");
        SparseLightningIndexerKLLossGradTilingBase sligKLLossTiling(context);
        sligKLLossTiling.DoTiling();
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus TilingPrepareForSparseLightningIndexerKLLossGrad(gert::TilingParseContext *context)
{
    OP_LOGW(context, "Start registering tiling.");
    auto compileInfoPtr = context->GetCompiledInfo<SparseLightningIndexerKLLossGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseLightningIndexerKLLossGrad)
    .Tiling(TilingSparseLightningIndexerKLLossGrad)
    .TilingParse<SparseLightningIndexerKLLossGradCompileInfo>(TilingPrepareForSparseLightningIndexerKLLossGrad);
} // namespace optiling
