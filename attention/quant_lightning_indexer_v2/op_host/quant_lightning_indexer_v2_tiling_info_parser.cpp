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
 * \file quant_lightning_indexer_v2_tiling_info_parser.cpp
 * \brief QuantLightningIndexerV2 tiling info parser
 */

#include "quant_lightning_indexer_v2_tiling_info_parser.h"

#include "../../lightning_indexer_v2/op_host/checkers/checker_adapter_lightning_indexer_v2.h"
#include "checkers/qliv2_checker.h"

namespace optiling {
namespace {
constexpr uint32_t BLOCK_TABLE_RANK = 2U;
constexpr char OP_NAME[] = "QuantLightningIndexerV2";

ge::graphStatus InitCheckerInfo(gert::TilingContext *context, QLIV2TilingInfo &tilingInfo,
                                lightning_indexer_v2_checker::LightningIndexerV2CheckerInfo &checkerInfo)
{
    const char *opName = context->GetNodeName();
    opName = opName == nullptr ? OP_NAME : opName;
    tilingInfo.platformInfo = context->GetPlatformInfo();
    if (tilingInfo.platformInfo == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "platform_info", "Platform information must be provided");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    if (ascendcPlatform.GetCoreNumAic() == 0 || ascendcPlatform.GetCoreNumAiv() == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "core_num", "0",
                                              "AIC and AIV core counts must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (context->GetWorkspaceSizes(1) == nullptr || context->GetRawTilingData() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "tiling_buffer",
                                                 "Workspace sizes and raw tiling data must be provided");
        return ge::GRAPH_FAILED;
    }
    if (ascendcPlatform.GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "npu_arch", "non-DAV_3510", "DAV_3510");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.opName = opName;
    tilingInfo.socVersion = ascendcPlatform.GetSocVersion();
    tilingInfo.npuArch = NpuArch::DAV_3510;
    checkerInfo.opName = opName;
    checkerInfo.quantized = true;
    return ge::GRAPH_SUCCESS;
}

void PopulateCheckerTensors(const gert::TilingContext *context,
                            lightning_indexer_v2_checker::LightningIndexerV2CheckerInfo &info)
{
    info.query = lightning_indexer_v2_checker::MakeRequiredTensor(context, QUERY_INDEX);
    info.key = lightning_indexer_v2_checker::MakeRequiredTensor(context, KEY_INDEX);
    info.weights = lightning_indexer_v2_checker::MakeRequiredTensor(context, WEIGTHS_INDEX);
    info.queryDescale = lightning_indexer_v2_checker::MakeRequiredTensor(context, QUERY_DEQUANT_SCALE_INDEX);
    info.keyDescale = lightning_indexer_v2_checker::MakeRequiredTensor(context, KEY_DEQUANT_SCALE_INDEX);
    info.cuSeqlensQ = lightning_indexer_v2_checker::MakeOptionalTensor(context, CU_SEQLENS_Q_INDEX);
    info.cuSeqlensK = lightning_indexer_v2_checker::MakeOptionalTensor(context, CU_SEQLENS_K_INDEX);
    info.sequsedQ = lightning_indexer_v2_checker::MakeOptionalTensor(context, SEQUSED_Q_INDEX);
    info.sequsedK = lightning_indexer_v2_checker::MakeOptionalTensor(context, SEQUSED_K_INDEX);
    info.cmpResidualK = lightning_indexer_v2_checker::MakeOptionalTensor(context, CMP_RESIDUAL_K_INDEX);
    info.blockTable = lightning_indexer_v2_checker::MakeOptionalTensor(context, BLOCK_TABLE_INDEX);
    info.outputIdxOffset = lightning_indexer_v2_checker::MakeOptionalTensor(context, OUTPUT_IDX_OFFSET_INDEX);
    info.metadata = lightning_indexer_v2_checker::MakeOptionalTensor(context, METADATA_INDEX);
    info.sparseIndices = lightning_indexer_v2_checker::MakeOutputTensor(context, SPARSE_INDICES_INDEX);
    info.sparseValues = lightning_indexer_v2_checker::MakeOutputTensor(context, SPARSE_VALUES_INDEX);
}

ge::graphStatus ParseCheckerAttrs(const gert::TilingContext *context,
                                  lightning_indexer_v2_checker::LightningIndexerV2CheckerInfo &info)
{
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attrs", "Operator attributes must be provided");
        return ge::GRAPH_FAILED;
    }
    const int64_t *topk = attrs->GetAttrPointer<int64_t>(ATTR_TOPK_INDEX);
    const int64_t *quantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    const int64_t *maxSeqlenQ = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);
    const char *layoutQ = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    const char *layoutK = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    const int64_t *maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);
    const int64_t *cmpRatio = attrs->GetAttrPointer<int64_t>(ATTR_CMP_RATIO_INDEX);
    const int64_t *returnValue = attrs->GetAttrPointer<int64_t>(ATTR_RETURN_VALUE_INDEX);
    if (topk == nullptr || quantMode == nullptr || maxSeqlenQ == nullptr || layoutQ == nullptr || layoutK == nullptr ||
        maskMode == nullptr || cmpRatio == nullptr || returnValue == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attrs", "Required/defaulted attributes are missing");
        return ge::GRAPH_FAILED;
    }
    info.topk = *topk;
    info.quantMode = *quantMode;
    info.maxSeqlenQ = *maxSeqlenQ;
    info.layoutQText = layoutQ;
    info.layoutKText = layoutK;
    info.maskMode = *maskMode;
    info.cmpRatio = *cmpRatio;
    info.returnValue = *returnValue;
    lightning_indexer_v2_checker::PopulateDerivedInfo(info);
    return ge::GRAPH_SUCCESS;
}

void SetDataLayouts(const lightning_indexer_v2_checker::LightningIndexerV2CheckerInfo &info,
                    QLIV2TilingInfo &tilingInfo)
{
    if (info.layoutQ == lightning_indexer_v2_checker::CheckerLayout::TND) {
        tilingInfo.inputQLayout = DataLayout::TND;
    } else {
        tilingInfo.inputQLayout = DataLayout::BSND;
    }
    if (info.layoutK == lightning_indexer_v2_checker::CheckerLayout::TND) {
        tilingInfo.inputKLayout = DataLayout::TND;
    } else if (info.layoutK == lightning_indexer_v2_checker::CheckerLayout::PA_BBND) {
        tilingInfo.inputKLayout = DataLayout::PA_BBND;
    } else {
        tilingInfo.inputKLayout = DataLayout::BSND;
    }
}

void PopulateStrideInfo(const lightning_indexer_v2_checker::LightningIndexerV2CheckerInfo &info,
                        QLIV2TilingInfo &tilingInfo)
{
    if (info.key.stride != nullptr && info.key.stride->GetDimNum() > 0) {
        tilingInfo.keyStride0 = static_cast<uint32_t>(info.key.stride->GetStride(0));
        if (info.quantMode == QUANT_MODE_MXFP4) {
            tilingInfo.keyStride0 /= MXFP4_PACK_NUM;
        }
    }
    if (info.keyDescale.stride != nullptr && info.keyDescale.stride->GetDimNum() > 0) {
        tilingInfo.keyDequantScaleStride0 = static_cast<uint32_t>(info.keyDescale.stride->GetStride(0));
    } else if (info.quantMode == QUANT_MODE_MXFP8 || info.quantMode == QUANT_MODE_MXFP4) {
        tilingInfo.keyDequantScaleStride0 =
            static_cast<uint32_t>(tilingInfo.blockSize) * (tilingInfo.qkHeadDim / MX_SCALE_GROUP_SIZE);
    }
}

void PopulateTilingInfo(const lightning_indexer_v2_checker::LightningIndexerV2CheckerInfo &info,
                        QLIV2TilingInfo &tilingInfo)
{
    tilingInfo.bSize = static_cast<uint32_t>(info.batch);
    tilingInfo.n1Size = static_cast<uint32_t>(info.qHeads);
    tilingInfo.n2Size = static_cast<uint32_t>(info.kHeads);
    tilingInfo.s1Size =
        info.layoutQ == lightning_indexer_v2_checker::CheckerLayout::BSND ? static_cast<uint32_t>(info.qSeq) : 0U;
    tilingInfo.s2Size = info.kSeq;
    tilingInfo.qkHeadDim = static_cast<uint32_t>(info.headDim);
    tilingInfo.gSize = info.kHeads == 0 ? 0U : static_cast<uint32_t>(info.qHeads / info.kHeads);
    tilingInfo.pageAttentionFlag = info.layoutK == lightning_indexer_v2_checker::CheckerLayout::PA_BBND;
    tilingInfo.blockSize = static_cast<int32_t>(info.blockSize);
    const gert::Shape *blockShape = info.blockTable.GetShape();
    if (tilingInfo.pageAttentionFlag && blockShape != nullptr && blockShape->GetDimNum() >= BLOCK_TABLE_RANK) {
        tilingInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape->GetDim(DIM_IDX_ONE));
    }
    tilingInfo.inputQType = info.query.desc->GetDataType();
    tilingInfo.inputKType = info.key.desc->GetDataType();
    tilingInfo.outputType = info.sparseIndices.desc->GetDataType();
    tilingInfo.sparseMode = static_cast<int32_t>(info.maskMode);
    tilingInfo.sparseCount = static_cast<uint32_t>(info.topk);
    tilingInfo.cmpRatio = static_cast<uint32_t>(info.cmpRatio);
    tilingInfo.returnValue = info.returnValue != 0;
    tilingInfo.maxSeqlenQ = static_cast<int32_t>(info.maxSeqlenQ);
    SetDataLayouts(info, tilingInfo);
    PopulateStrideInfo(info, tilingInfo);
}
} // namespace

ge::graphStatus ParseAndCheckQLIV2Arch35(gert::TilingContext *context, QLIV2TilingInfo &tilingInfo)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "context", "Tiling context must not be null");
        return ge::GRAPH_FAILED;
    }
    lightning_indexer_v2_checker::LightningIndexerV2CheckerInfo checkerInfo;
    if (InitCheckerInfo(context, tilingInfo, checkerInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PopulateCheckerTensors(context, checkerInfo);
    if (ParseCheckerAttrs(context, checkerInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const lightning_indexer_v2_checker::QLIV2Checker checker(checkerInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingInfo.quantMode = static_cast<int32_t>(checkerInfo.quantMode);
    tilingInfo.opParamInfo.quantMode = &tilingInfo.quantMode;
    PopulateTilingInfo(checkerInfo, tilingInfo);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
