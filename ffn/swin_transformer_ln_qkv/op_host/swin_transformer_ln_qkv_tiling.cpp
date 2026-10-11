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
 * \file swin_transformer_ln_qkv_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "swin_transformer_ln_qkv_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"

using namespace std;
using namespace ge;
using namespace matmul_tiling;
namespace optiling {
struct SwinTransformerLnQKVCompileInfo {
    uint32_t coreNum = 0;
};
class SwinTransformerLnQKVTilingCompute {
public:
    SwinTransformerLnQKVTilingData tilingData;
    ge::graphStatus SwinTransformerLnQKVTilingMainProc(gert::TilingContext* context);

protected:
    ge::graphStatus SwinTransformerLnQKVSaveTilingData(gert::TilingContext* context);
    ge::graphStatus SwinTransformerLnQKVSetLayernormTilingData(
        gert::TilingContext* context, SwinTransformerLnQKVLayernormTilingData& layernormTilingData);
    ge::graphStatus SwinTransformerLnQKVSetMatmulTilingData(gert::TilingContext* context, TCubeTiling mmtilingData);
}; // class SwinTransformerLnQKVTilingCompute

ge::graphStatus SwinTransformerLnQKVTilingCompute::SwinTransformerLnQKVSaveTilingData(gert::TilingContext* context)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwinTransformerLnQKVTilingCompute::SwinTransformerLnQKVSetLayernormTilingData(
    gert::TilingContext* context, SwinTransformerLnQKVLayernormTilingData& layernormTilingData)
{
    uint32_t bLength = 8;   // 8 is for ln
    uint32_t sLength = 256; // 256 is for ln
    uint32_t hLength = 128; // 128 is for ln
    layernormTilingData.set_bLength(bLength);
    layernormTilingData.set_sLength(sLength);
    layernormTilingData.set_hLength(hLength);
    uint32_t loopSize = bLength * sLength * hLength;
    layernormTilingData.set_loopSize(loopSize);
    layernormTilingData.set_bsLength(bLength * sLength);
    layernormTilingData.set_shLength(sLength * hLength);
    uint32_t loopSum = tilingData.get_inputSizeSum() / loopSize;
    uint32_t vectorNum = tilingData.get_maxCoreNum();
    if (vectorNum == 0) {
        vectorNum = 1; // 1 is for a division by zero error
    }
    uint32_t loopNumForAllBlock = loopSum / vectorNum;
    uint32_t remainderBlock = loopSum % vectorNum;
    if (loopSum < tilingData.get_maxCoreNum()) {
        context->SetBlockDim(loopSum);
        tilingData.opBaseInfo.set_baseLoopNum(1);
        tilingData.opBaseInfo.set_remainderBlockNum(0);
        tilingData.set_useVectorNum(loopSum);
    } else {
        context->SetBlockDim(tilingData.get_maxCoreNum());
        tilingData.opBaseInfo.set_baseLoopNum(loopNumForAllBlock);
        tilingData.opBaseInfo.set_remainderBlockNum(remainderBlock);
        tilingData.set_useVectorNum(vectorNum);
    }

    layernormTilingData.set_elementPerBlock(loopSize * tilingData.opBaseInfo.get_baseLoopNum());
    layernormTilingData.set_remainderElementPerBlock(loopSize * (tilingData.opBaseInfo.get_baseLoopNum() + 1));
    layernormTilingData.set_normalBlockElementOffset(loopSize * (tilingData.opBaseInfo.get_baseLoopNum() + 1) *
                                                     tilingData.opBaseInfo.get_remainderBlockNum());

    uint32_t innerLoopSize = sLength * hLength; // 32K
    layernormTilingData.set_innerLoopLength(innerLoopSize);
    uint32_t innerLoopNum = bLength; // b * s *h / innerLoopSize
    layernormTilingData.set_innerLoopNum(innerLoopNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwinTransformerLnQKVTilingCompute::SwinTransformerLnQKVSetMatmulTilingData(gert::TilingContext* context,
                                                                                           TCubeTiling mmtilingData)
{
    uint32_t seqLength = 128; // 128 is for ln
    uint32_t loopTime = 8;
    uint32_t matmulSize = 2;
    uint32_t matmulHeight = 3;
    MatmulApiTiling tiling;
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetShape(loopTime * seqLength * matmulSize, seqLength * matmulHeight, seqLength);
    tiling.SetOrgShape(loopTime * seqLength * matmulSize, seqLength * matmulHeight, seqLength);
    tiling.SetBias(true);
    tiling.SetBufferSpace(-1, -1, -1);
    tiling.SetTraverse(MatrixTraverse::FIRSTN);
    int ret = tiling.GetTiling(mmtilingData); // if ret = -1, gen tiling failed
    if (ret == -1) {
        OP_LOGW(context, "SwinTransformerLnQKV get tiling fail.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwinTransformerLnQKVTilingCompute::SwinTransformerLnQKVTilingMainProc(gert::TilingContext* context)
{
    const gert::StorageShape* x1Shape =
        context->GetInputShape(0); // [B,S, H]   [8,65536, 128] -> [1, 65536,128] -> 32 * [8,256,128]
    int32_t dataSize = 1;          // 8(batch) *32(out loop) * [32, 64, 128] ->  [8192, 64 ,128]
    OP_CHECK_IF(x1Shape == nullptr, OP_LOGE("SwinTransformerLnQkv", "The INPUT x1Shape param is null!"),
                return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < x1Shape->GetStorageShape().GetDimNum(); i++) {
        dataSize *= x1Shape->GetStorageShape().GetDim(i);
    }
    tilingData.set_inputSizeSum(dataSize);

    auto compileInfo = reinterpret_cast<const SwinTransformerLnQKVCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    tilingData.set_maxCoreNum(uint32_t(compileInfo->coreNum));

    SwinTransformerLnQKVSetLayernormTilingData(context, tilingData.layernormTilingParams);
    SwinTransformerLnQKVSetMatmulTilingData(context, tilingData.mmTilingParams);

    tilingData.opBaseInfo.set_inputsize(dataSize);
    tilingData.opBaseInfo.set_hSize(256); // The default value is 256.
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE("SwinTransformerLnQkv", "failed to get workspace size"),
                return ge::GRAPH_FAILED);

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::KIRINX90 ||
        ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::KIRIN9030) {
        OP_LOGD(context->GetNodeName(), "Current soc version is KIRIN.");
        const gert::StorageShape* qShape = context->GetOutputShape(0);
        OP_CHECK_IF(qShape == nullptr, OP_LOGE(context->GetNodeName(), "qShape is null"), return ge::GRAPH_FAILED);
        int32_t outSize = 1;
        for (uint32_t i = 0; i < qShape->GetStorageShape().GetDimNum(); i++) {
            outSize *= qShape->GetStorageShape().GetDim(i);
        }
        workspaces[0] = dataSize * sizeof(uint16_t) + outSize * sizeof(uint16_t) + outSize * sizeof(uint16_t) +
                        outSize * sizeof(uint16_t);
    } else {
        workspaces[0] =
            1200 * 1024 *
            1024; // To use the workspace, set the workspace size to 1200 x 1024 x 1024 MB by default. Memory on GM
    }
    tilingData.set_workspaceSize(workspaces[0]);

    SwinTransformerLnQKVSaveTilingData(context);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFuncForSwinTransformerLnQKV(gert::TilingContext* context)
{
    SwinTransformerLnQKVTilingCompute tilingCompute;
    auto ret = tilingCompute.SwinTransformerLnQKVTilingMainProc(context);

    OP_LOGI(context->GetNodeName(), "SwinTransformerLnQKV tiling end.");
    return ret;
} // TilingFunc

static ge::graphStatus TilingPrepareForSwinTransformerLnQKV(gert::TilingParseContext* context)
{
    // auto compileInfo = GetCompileInfoPtr<SwinTransformerLnQKVCompileInfo>(context);
    auto compileInfo = context->GetCompiledInfo<SwinTransformerLnQKVCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = (uint32_t)platformInfo.GetCoreNum();
    uint64_t ubSizePlatForm;
    uint64_t l1SizePlatForm;
    uint64_t l0CSizePlatForm;
    uint64_t l0ASizePlatForm;

    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1SizePlatForm);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSizePlatForm);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASizePlatForm);
    compileInfo->coreNum = coreNum;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwinTransformerLnQKV)
    .Tiling(TilingFuncForSwinTransformerLnQKV)
    .TilingParse<SwinTransformerLnQKVCompileInfo>(TilingPrepareForSwinTransformerLnQKV);
} // namespace optiling
