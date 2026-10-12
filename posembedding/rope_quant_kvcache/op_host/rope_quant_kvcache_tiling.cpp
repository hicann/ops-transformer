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
 * \file rope_quant_kvcache_tiling.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "register/op_def_registry.h"
#include "rope_quant_kvcache_tiling.h"
using namespace ge;

namespace {
constexpr int64_t TASK_NUM = 2;
constexpr int SIZE_SPLIT_Q = 0;
constexpr int SIZE_SPLIT_K = 1;
constexpr int SIZE_SPLIT_V = 2;
constexpr int KVCACHE_INPUT_INDEX = 5;
constexpr int KVCACHE_HEAD_DIM_INDEX = 2;
constexpr int KVCACHE_HIDDEN_DIM_INDEX = 3;
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
} // namespace

namespace optiling {

static ge::graphStatus TilingForRopeQuantKvcache(gert::TilingContext* context)
{
    const gert::StorageShape* cacheShape = context->GetInputShape(KVCACHE_INPUT_INDEX);

    auto attrs = context->GetAttrs();
    auto attr = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t* attrData = reinterpret_cast<const int64_t*>(attr->GetData());

    int64_t batch = cacheShape->GetStorageShape().GetDim(0);
    int64_t cacheSeqlen = cacheShape->GetStorageShape().GetDim(1);
    int64_t kvHeadNum = cacheShape->GetStorageShape().GetDim(KVCACHE_HEAD_DIM_INDEX);
    int64_t hiddenSize = cacheShape->GetStorageShape().GetDim(KVCACHE_HIDDEN_DIM_INDEX);
    int64_t qHeadNum = attrData[0] / hiddenSize;

    int64_t coreNumUsed = batch;
    RopeQuantKvcacheTilingData tiling;
    tiling.set_cacheSeqlen(cacheSeqlen);
    tiling.set_qHeadNum(qHeadNum);
    tiling.set_kvHeadNum(kvHeadNum);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_qHiddenSize(attrData[SIZE_SPLIT_Q]);
    tiling.set_kHiddenSize(attrData[SIZE_SPLIT_K]);
    tiling.set_vHiddenSize(attrData[SIZE_SPLIT_V]);

    context->SetBlockDim(coreNumUsed * TASK_NUM);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForRopeQuantKvcache(gert::TilingParseContext* context)
{
    (void)context;
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RopeQuantKvcache)
    .Tiling(TilingForRopeQuantKvcache)
    .TilingParse<RopeQuantKvcacheCompileInfo>(TilingPrepareForRopeQuantKvcache);
} // namespace optiling
