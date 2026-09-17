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
 * \file incre_flash_attention_tiling_arch38.cpp
 * \brief
 */

#include "incre_flash_attention_tiling_arch38.h"
#include "incre_flash_attention_tiling_base.h"
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling_arch38.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace arch38 {

const int64_t tokenDefault = 2147483647; // for token default value
const int32_t sparseDefault = 0;

// Validated scenario boundary of this platform, out-of-range cases are rejected instead of silently mis-computed.
constexpr int64_t ARCH38_IFA_SUPPORTED_MAX_N = 128;
constexpr int64_t ARCH38_IFA_SUPPORTED_MAX_SKV = 4101;
constexpr size_t BNSD_DIM_NUM = 4U;
constexpr size_t BNSD_DIM_S = 2U;
constexpr size_t BNSD_DIM_D = 3U;

ge::graphStatus PFAConvertContext(ContextParamsForPFATiling &contextKeyParams, gert::TilingContext *context)
{
    contextKeyParams.opName = context->GetNodeName();
    contextKeyParams.isKvContinuous = 1U;
    contextKeyParams.emptyTensor = 0U;
    contextKeyParams.fromTilingSink = 0U;
    contextKeyParams.fromFused = 71U; // 71 for default value
    contextKeyParams.pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INPUT_INDEX);
    contextKeyParams.attentionMask = context->GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);
    contextKeyParams.actualSequenceLengthKV = context->GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
    contextKeyParams.antiquantScale = context->GetOptionalInputTensor(ANTIQUANT_SCALE_INPUT_INDEX);
    contextKeyParams.antiquantOffset = context->GetOptionalInputTensor(ANTIQUANT_OFFSET_INPUT_INDEX);
    contextKeyParams.blockTable = context->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    contextKeyParams.kvPaddingSize = context->GetOptionalInputTensor(KV_PADDING_SIZE_INPUT_INDEX);
    contextKeyParams.actualSequenceLengthQ = nullptr;
    contextKeyParams.keySharedPrefix = (nullptr);
    contextKeyParams.valueSharedPrefix = (nullptr);
    contextKeyParams.actualSharedPrefixLen = (nullptr);
    contextKeyParams.inputDataType = context->GetInputDesc(QUERY_INPUT_INDEX)->GetDataType();
    contextKeyParams.kDataType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
    contextKeyParams.vDataType = context->GetInputDesc(VALUE_INPUT_INDEX)->GetDataType();
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
                                            context->GetOptionalInputDesc(PSE_SHIFT_INPUT_INDEX)->GetDataType() :
                                            contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
                                        context->GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX)->GetDataType() :
                                        contextKeyParams.inputDataType;
    contextKeyParams.blockTableType = (context->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX) != nullptr) ?
                                          context->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX)->GetDataType() :
                                          ge::DT_INT32;
    contextKeyParams.outputDataType = context->GetOutputDesc(OUTPUT_INDEX)->GetDataType();
    contextKeyParams.deqScaleType = (context->GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX) != nullptr) ?
                                        context->GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX)->GetDataType() :
                                        contextKeyParams.inputDataType;
    contextKeyParams.deqScale2Type = (context->GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX) != nullptr) ?
                                         context->GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX)->GetDataType() :
                                         contextKeyParams.inputDataType;
    contextKeyParams.quantScale2Type = (context->GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX) != nullptr) ?
                                           context->GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX)->GetDataType() :
                                           ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context->GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX) != nullptr) ?
                                            context->GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX)->GetDataType() :
                                            ge::DT_FLOAT;
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INPUT_INDEX);
    contextKeyParams.keyInputShape = context->GetInputShape(KEY_INPUT_INDEX);
    contextKeyParams.valueInputShape = context->GetInputShape(VALUE_INPUT_INDEX);
    contextKeyParams.pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INPUT_INDEX);
    contextKeyParams.attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INPUT_INDEX);
    contextKeyParams.deqScale1Shape = context->GetOptionalInputShape(DEQUANT_SCALE_1_INPUT_INDEX);
    contextKeyParams.scale1Shape = context->GetOptionalInputShape(QUANT_SCALE_1_INPUT_INDEX);
    contextKeyParams.deqScale2Shape = context->GetOptionalInputShape(DEQUANT_SCALE_2_INPUT_INDEX);
    contextKeyParams.scale2Shape = context->GetOptionalInputShape(QUANT_SCALE_2_INPUT_INDEX);
    contextKeyParams.offset2Shape = context->GetOptionalInputShape(QUANT_OFFSET_2_INPUT_INDEX);
    contextKeyParams.antiquantScaleShape = context->GetOptionalInputShape(ANTIQUANT_SCALE_INPUT_INDEX);
    contextKeyParams.antiquantOffsetShape = context->GetOptionalInputShape(ANTIQUANT_OFFSET_INPUT_INDEX);
    contextKeyParams.blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INPUT_INDEX);
    contextKeyParams.outputShape = context->GetOutputShape(OUTPUT_INDEX);
    auto attrs = context->GetAttrs();
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(INNER_PRECISE_ATTR_INDEX);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int64_t>(NUM_HEADS_ATTR_INDEX);
    contextKeyParams.blockSize = attrs->GetAttrPointer<int32_t>(BLOCK_SIZE_ATTR_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(LAYOUT_ATTR_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int64_t>(KV_NUM_HEADS_ATTR_INDEX);
    contextKeyParams.sparseMode = &sparseDefault;
    contextKeyParams.preToken = &tokenDefault;
    contextKeyParams.nextToken = &tokenDefault;
    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);
    contextKeyParams.compileInfoPtr =
        reinterpret_cast<const PromptFlashAttentionCompileInfo *>(context->GetCompileInfo());
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1U : 0U;

    return ge::GRAPH_SUCCESS;
}

void TilingGetTempCompileInfo(platform_ascendc::PlatformAscendC &ascendcPlatform,
                              PromptFlashAttentionCompileInfo &compileInfo)
{
    compileInfo.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo.aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo.l0BSize);
    compileInfo.defaultSysWorkspaceSize = 0U;
}

ge::graphStatus IFATilingArch38::ConvertContext(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    OP_CHECK_IF(context.GetNodeName() == nullptr,
                OP_LOGE("IncreFlashAttention", "OpName got from TilingContext is null."), return ge::GRAPH_FAILED);
    ifaContext.opName = context.GetNodeName();
    ifaContext.platformInfo = context.GetPlatformInfo();
    ifaContext.query.desc = context.GetInputDesc(QUERY_INPUT_INDEX);
    ifaContext.query.shape = context.GetInputShape(QUERY_INPUT_INDEX);
    ifaContext.key.desc = context.GetInputDesc(KEY_INPUT_INDEX);
    ifaContext.key.shape = context.GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
                OP_LOGE(context.GetNodeName(), "Shape of query or shape of key is null."), return ge::GRAPH_FAILED);
    auto batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(NUM0);
    auto batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(NUM0);
    if (batchOfQuery != batchOfKey) {
        ifaContext.kCache.resize(batchOfQuery);
        ifaContext.vCache.resize(batchOfQuery);
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            ifaContext.kCache[size] =
                const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, size));
            ifaContext.vCache[size] =
                const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, size));
        }
    } else {
        ifaContext.kCache.resize(NUM1);
        ifaContext.vCache.resize(NUM1);
        ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, 0));
        ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, 0));
    }

    ifaContext.value.desc = context.GetInputDesc(VALUE_INPUT_INDEX);
    ifaContext.value.shape = context.GetInputShape(VALUE_INPUT_INDEX);
    ifaContext.pseShift.desc = context.GetOptionalInputDesc(PSE_SHIFT_INPUT_INDEX);
    ifaContext.pseShift.tensor = context.GetOptionalInputTensor(PSE_SHIFT_INPUT_INDEX);
    ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX);
    ifaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);
    ifaContext.attenOut.desc = context.GetOutputDesc(OUTPUT_INDEX);
    ifaContext.attenOut.shape = context.GetOutputShape(OUTPUT_INDEX);

    ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
    ifaContext.deqScale1.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_1_INPUT_INDEX);
    ifaContext.quantScale1.tensor = context.GetOptionalInputTensor(QUANT_SCALE_1_INPUT_INDEX);
    ifaContext.deqScale2.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantScale2.tensor = context.GetOptionalInputTensor(QUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantOffset2.tensor = context.GetOptionalInputTensor(QUANT_OFFSET_2_INPUT_INDEX);
    ifaContext.deqScale1.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX);
    ifaContext.quantScale1.desc = context.GetOptionalInputDesc(QUANT_SCALE_1_INPUT_INDEX);
    ifaContext.deqScale2.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantScale2.desc = context.GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantOffset2.desc = context.GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX);
    ifaContext.antiquantScale.tensor = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INPUT_INDEX);
    ifaContext.antiquantOffset.tensor = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INPUT_INDEX);
    ifaContext.antiquantScale.desc = context.GetOptionalInputDesc(ANTIQUANT_SCALE_INPUT_INDEX);
    ifaContext.antiquantOffset.desc = context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INPUT_INDEX);
    ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    ifaContext.kvPaddingSize.tensor = context.GetOptionalInputTensor(KV_PADDING_SIZE_INPUT_INDEX);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "Attrs got from ge is null."),
                return ge::GRAPH_FAILED);

    ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(NUM_HEADS_ATTR_INDEX);
    ifaContext.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    ifaContext.layOut = attrs->GetStr(LAYOUT_ATTR_INDEX);
    ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(KV_NUM_HEADS_ATTR_INDEX);
    ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(BLOCK_SIZE_ATTR_INDEX);
    ifaContext.innerPrecise = attrs->GetAttrPointer<uint32_t>(INNER_PRECISE_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "WorkSpaceSize got from ge is null."),
                return ge::GRAPH_FAILED);
    ifaContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATilingArch38::DoOpTiling()
{
    IncreFlashAttentionContext ifaContext;
    auto ret = ConvertContext(*context_, ifaContext);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "fail to convert to IFAParams"),
                return ge::GRAPH_FAILED);
    ret = DoSubOpTiling(ifaContext);
    return ret;
}

bool IFATilingArch38::CheckArch38ScenarioSupported(const IncreFlashAttentionContext &ifaContext) const
{
    const char *opName = context_->GetNodeName();
    // PagedAttention lays key out by block, its shape carries no real kv seq length for the check below
    OP_CHECK_IF((ifaContext.blockTable.tensor != nullptr),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "PagedAttention is not supported on this platform."), return false);

    // BNSD_BSND shares the BNSD input layout, it only transposes the output, same as PFA's SetInputLayout
    const std::string layoutStr = (ifaContext.layOut == nullptr) ? "" : std::string(ifaContext.layOut);
    OP_CHECK_IF((layoutStr != "BNSD"),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "only BNSD layout is supported on this platform."), return false);

    OP_CHECK_IF(((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr) ||
                 (ifaContext.value.shape == nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "shape of query, key or value is null."), return false);

    const auto &queryShape = ifaContext.query.shape->GetStorageShape();
    const auto &keyShape = ifaContext.key.shape->GetStorageShape();
    const auto &valueShape = ifaContext.value.shape->GetStorageShape();
    OP_CHECK_IF(((queryShape.GetDimNum() != BNSD_DIM_NUM) || (keyShape.GetDimNum() != BNSD_DIM_NUM) ||
                 (valueShape.GetDimNum() != BNSD_DIM_NUM)),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "BNSD layout requires 4-dim query, key and value shapes."),
                return false);

    const int64_t sQ = queryShape.GetDim(BNSD_DIM_S);
    OP_CHECK_IF((sQ != 1), OPS_REPORT_VECTOR_INNER_ERR(opName, "query seq length must be 1, but qs = %ld.", sQ),
                return false);

    const int64_t sKV = keyShape.GetDim(BNSD_DIM_S);
    OP_CHECK_IF(((sKV <= 1) || (sKV > ARCH38_IFA_SUPPORTED_MAX_SKV)),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "kv seq length must be in (1, %ld], but kvs = %ld.",
                                            ARCH38_IFA_SUPPORTED_MAX_SKV, sKV),
                return false);

    OP_CHECK_IF((ifaContext.numHeads == nullptr), OPS_REPORT_VECTOR_INNER_ERR(opName, "numHeads got from ge is null."),
                return false);
    const int64_t nQ = static_cast<int64_t>(*ifaContext.numHeads);
    // numKeyValueHeads defaults to numHeads when it is absent or left as 0
    const int64_t nKV = ((ifaContext.kvHeadNums != nullptr) && (*ifaContext.kvHeadNums != 0U)) ?
                            static_cast<int64_t>(*ifaContext.kvHeadNums) :
                            nQ;
    OP_CHECK_IF(((nQ <= 0) || (nQ > ARCH38_IFA_SUPPORTED_MAX_N) || (nKV <= 0) || (nKV > ARCH38_IFA_SUPPORTED_MAX_N)),
                OPS_REPORT_VECTOR_INNER_ERR(opName,
                                            "head num must be in (0, %ld], but numHeads = %ld, "
                                            "numKeyValueHeads = %ld.",
                                            ARCH38_IFA_SUPPORTED_MAX_N, nQ, nKV),
                return false);

    const int64_t dQ = queryShape.GetDim(BNSD_DIM_D);
    const int64_t dV = valueShape.GetDim(BNSD_DIM_D);
    OP_CHECK_IF((!IsArch38SupportedHeadSize(dQ) || !IsArch38SupportedHeadSize(dV)),
                OPS_REPORT_VECTOR_INNER_ERR(opName,
                                            "head size only supports 64/80/128, but query d = %ld, "
                                            "value d = %ld.",
                                            dQ, dV),
                return false);

    return true;
}

ge::graphStatus IFATilingArch38::DoSubOpTiling(IncreFlashAttentionContext &ifaContext)
{
    OP_CHECK_IF(!CheckArch38ScenarioSupported(ifaContext),
                OP_LOGE(context_->GetNodeName(), "unsupported scenario on this platform."), return ge::GRAPH_FAILED);

    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    PromptFlashAttentionCompileInfo compileInfo = {0, 0, 0, 0, 0, 0, 0, 0, ascendcPlatform.GetSocVersion()};
    TilingGetTempCompileInfo(ascendcPlatform, compileInfo);
    PromptFlashAttentionTilingArch38 flashTilingArch38(context_);
    ContextParamsForPFATiling contextParamsForPFATiling;
    auto ret = PFAConvertContext(contextParamsForPFATiling, context_);
    contextParamsForPFATiling.compileInfoPtr = &compileInfo;
    OP_CHECK_IF(ret == ge::GRAPH_FAILED,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "fail to convert to PFAParams"),
                return ge::GRAPH_FAILED);
    PromptFlashAttentionTilingData tilingData;
    ret = flashTilingArch38.DoSubOpTiling(tilingData, contextParamsForPFATiling);
    return ret;
}

REGISTER_TILING_TEMPLATE_FIA(IncreFlashAttention, IFATilingArch38, std::vector<int32_t>({(int32_t)NpuArch::DAV_5102}),
                             92);

} // namespace arch38
} // namespace optiling
