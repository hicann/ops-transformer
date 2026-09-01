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
 * \file decode_update_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "util/shape_util.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_type.h"
#include "op_host/tiling_util.h"
#include "decode_update_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "attention_update_tiling.h"

namespace {
constexpr uint64_t TILING_KEY_GO_FP32_WITHOUT_MAX_OUT = 10010;
constexpr uint64_t TILING_KEY_GO_FP16_WITHOUT_MAX_OUT = 10020;
constexpr uint64_t TILING_KEY_GO_BF16_WITHOUT_MAX_OUT = 10030;

constexpr uint64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024UL;
constexpr uint32_t LSE_TOTAL_LENGTH_DIM = 0;
constexpr uint32_t IN_HD_DIM = 1;
constexpr uint32_t LOCAL_OUT_INDEX = 1;
constexpr uint32_t ATTR_UPDATETYPE_INDEX = 0;
constexpr uint32_t ATTR_SP_INDEX = 1;

//  hDim = 128， spAligned = 8时，单核b*hc最大7，所以取3
constexpr uint32_t MIN_BLOCK_LENGTH = 3;
} // namespace

namespace optiling {

uint64_t GetTilingKey(uint32_t sp, ge::DataType goType_)
{
    uint64_t tiling_key = 0UL;
    if (goType_ == ge::DataType::DT_FLOAT) {
        tiling_key = TILING_KEY_GO_FP32_WITHOUT_MAX_OUT;
    } else if (goType_ == ge::DataType::DT_FLOAT16) {
        tiling_key = TILING_KEY_GO_FP16_WITHOUT_MAX_OUT;
    } else if (goType_ == ge::DataType::DT_BF16) {
        tiling_key = TILING_KEY_GO_BF16_WITHOUT_MAX_OUT;
    }
    return tiling_key;
}

ge::graphStatus DecodeUpdateTiling(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("AttentionUpdate", "context is null"),
                return ge::GRAPH_FAILED);
    auto compileInfo = reinterpret_cast<const DecodeUpdateCompileInfo *>(context->GetCompileInfo());
    if (compileInfo->is_ascendc) {
        return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling initializing");

    auto attrs = context->GetAttrs();
    const int64_t *updateTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_UPDATETYPE_INDEX);
    const int64_t *spPtr = attrs->GetAttrPointer<int64_t>(ATTR_SP_INDEX);

    auto updateType = *updateTypePtr;
    auto sp = *spPtr;

    ge::DataType goType_ = context->GetInputDesc(LOCAL_OUT_INDEX * sp)->GetDataType();
    auto tilingKey = GetTilingKey(sp, goType_);
    context->SetTilingKey(tilingKey);

    auto lseShape = context->GetInputShape(0)->GetStorageShape();
    auto inShape = context->GetInputShape(sp)->GetStorageShape();

    // sp, B*s*hc in1
    // sp, B*s*hc, hd in2
    const uint32_t totalLength = lseShape.GetDim(LSE_TOTAL_LENGTH_DIM);
    const uint32_t hd = inShape.GetDim(IN_HD_DIM);

    OP_LOGD(nodeName, "TotalLength of b*s*hc is %u, sp is %ld, hd is %u",
            totalLength, sp, hd);

    uint32_t blockDims = 0;
    uint32_t coreNum = compileInfo->coreNum;
    blockDims = std::min<uint32_t>((totalLength + MIN_BLOCK_LENGTH - 1UL) / MIN_BLOCK_LENGTH, coreNum);

    DecodeUpdateTilingData tilingData;
    tilingData.set_formerNum(static_cast<uint32_t>(totalLength % blockDims));
    tilingData.set_tailNum(static_cast<uint32_t>(blockDims - tilingData.get_formerNum()));
    tilingData.set_tailLength(static_cast<uint32_t>(totalLength / blockDims));
    tilingData.set_formerLength(static_cast<uint32_t>(tilingData.get_formerNum() == 0 ?
                                                          0 :
                                                          tilingData.get_tailLength() + 1));
    tilingData.set_hDim(static_cast<uint32_t>(hd));
    tilingData.set_sp(static_cast<uint32_t>(sp));
    tilingData.set_updateType(static_cast<uint32_t>(updateType));
    tilingData.set_totalLength(static_cast<uint32_t>(totalLength));

    OP_LOGD(nodeName, "BlockDims is %u", blockDims);
    OP_LOGD(nodeName, "formerLength is %u", tilingData.get_formerLength());
    OP_LOGD(nodeName, "tailLength is %u", tilingData.get_tailLength());
    OP_LOGD(nodeName, "formerNum is %u", tilingData.get_formerNum());
    OP_LOGD(nodeName, "tailNum is %u", tilingData.get_tailNum());
    OP_LOGD(nodeName, "hDim is %u", tilingData.get_hDim());
    OP_LOGD(nodeName, "sp is %u", tilingData.get_sp());
    OP_LOGD(nodeName, "totalLength is %u", tilingData.get_totalLength());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SYS_WORKSPACE_SIZE;

    context->SetBlockDim(blockDims);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4DecodeUpdate(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<DecodeUpdateCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                OP_LOGE(context, "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->ubSize = ubSize;
    compileInfo->is_ascendc = Ops::Transformer::OpTiling::IsRegbaseSocVersion(context);

    OP_CHECK_IF(compileInfo->coreNum <= 0,
                OP_LOGE(context->GetNodeName(),
                        "AttentionUpdate GetHardwareInfo Failed, vectorCoreNum: %u",
                        compileInfo->coreNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compileInfo->ubSize <= 0,
                OP_LOGE(context->GetNodeName(),
                        "AttentionUpdate GetHardwareInfo Failed, ubSize: %lu",
                        compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionUpdate)
    .Tiling(DecodeUpdateTiling)
    .TilingParse<DecodeUpdateCompileInfo>(TilingPrepare4DecodeUpdate);
} // namespace optiling
