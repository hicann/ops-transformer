/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_compress_grad.cpp
 * \brief
 */
#include "nsa_compress_grad_tiling.h"
#include <queue>
#include <cmath>
#include <cfloat>
#include "log/log.h"
#include "err/ops_err.h"
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
namespace {
    constexpr uint64_t WORKSIZE = static_cast<uint64_t>(16) * 1024 * 1024;
    constexpr uint32_t BLOCK_SIZE_16 = 16;
    constexpr uint32_t ONE_KILO = 1024;
    constexpr uint32_t INDEXZERO = 0;
    constexpr uint32_t INDEXONE = 1;
    constexpr uint32_t INDEXTWO = 2;
    constexpr uint32_t INDEXTHREE = 3;
} // namespace 

namespace optiling {

void NsaCompressGradTiling::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
}

ge::graphStatus NsaCompressGradTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool NsaCompressGradTiling::PreparePlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGI(context_, "NsaCompressGrad get platformInfo is null.");
        auto compileInfoPtr = context_->GetCompileInfo<NsaCompileInfo>();
        OP_CHECK_IF(
            compileInfoPtr == nullptr, 
            OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "NsaCompressGrad CompileInfoPtr is null."), 
            return false
        );
        aicNum_ = compileInfoPtr->aicNum;
        aivNum_ = compileInfoPtr->aivNum;
        ubMaxSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicNum_ = ascendcPlatform.GetCoreNumAic();
        aivNum_ = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubMaxSize_);
    }
    OP_LOGI(
        context_,
        "NsaCompressGrad Platform info: aicNum(%u) aivNum(%u) ubSize(%lu).",
        aicNum_,
        aivNum_,
        ubMaxSize_
    );
    return true;
}

ge::graphStatus NsaCompressGradTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool NsaCompressGradTiling::IsCapable()
{
    return true;
}

ge::graphStatus NsaCompressGradTiling::DoOpTiling()
{
    OP_CHECK_IF(!PreparePlatformInfo(), 
               OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "NsaCompressGrad SetPlatformInfoForTiling fail."), 
               return ge::GRAPH_FAILED
    );

    const auto& outGradShape = GetShapeOfInput(INDEXZERO);
    auto nBlocks = static_cast<uint32_t>(outGradShape.GetDim(INDEXZERO));
    auto nHeads = static_cast<uint32_t>(outGradShape.GetDim(INDEXONE));
    auto dimHead = static_cast<uint32_t>(outGradShape.GetDim(INDEXTWO));

    const auto& inputKvShape = GetShapeOfInput(INDEXONE);
    tSeqLen_ = static_cast<uint32_t>(inputKvShape.GetDim(INDEXZERO));
    auto headToProcessPerCore = tSeqLen_ * nHeads / aivNum_;
    auto headRemainder = (tSeqLen_ * nHeads) % aivNum_;

    auto batchSize = 1;
    const auto& actSeqLenShape = context_->GetOptionalInputShape(INDEXTHREE);
    if (actSeqLenShape != nullptr) {
        batchSize = static_cast<uint32_t>(actSeqLenShape->GetStorageShape().GetDim(INDEXZERO));
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const auto* blockSizePtr = attrs->GetAttrPointer<uint32_t>(INDEXZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockSizePtr);
    uint32_t blockSize = *blockSizePtr;

    const auto* blockStridePtr = attrs->GetAttrPointer<uint32_t>(INDEXONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockStridePtr);
    uint32_t blockStride = *blockStridePtr;

    const auto* seqLenTypePtr = attrs->GetAttrPointer<uint32_t>(INDEXTWO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqLenTypePtr);
    uint32_t seqLenType = *seqLenTypePtr;

    tilingData_.set_batchSize(batchSize);
    tilingData_.set_numOfBlock(nBlocks);
    tilingData_.set_numOfHead(nHeads);
    tilingData_.set_dimOfHead(dimHead);
    tilingData_.set_blockSize(blockSize);
    tilingData_.set_blockStride(blockStride);
    tilingData_.set_seqLenType(seqLenType);
    tilingData_.set_headToProcess(headToProcessPerCore);
    tilingData_.set_headRemainder(headRemainder);
    tilingData_.set_ubMaxSize(ubMaxSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaCompressGradTiling::DoLibApiTiling() {
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaCompressGradTiling::GetWorkspaceSize() {
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    size_t sysWorkspaceSize = WORKSIZE;
    /* usrWorkspaceSize = workspace for other gm */
    size_t usrWorkspaceSize = tilingData_.get_blockSize() * tilingData_.get_numOfHead() * sizeof(uint32_t) * aivNum_ +
    tSeqLen_ * tilingData_.get_numOfHead() * tilingData_.get_dimOfHead() * sizeof(uint32_t) + ONE_KILO * ONE_KILO;
    workspaces[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

uint64_t NsaCompressGradTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus NsaCompressGradTiling::PostTiling()
{
    OP_CHECK_IF(!CheckTilingData(), 
               OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Tiling data not support."), 
               return ge::GRAPH_FAILED
    );
    
    OP_CHECK_IF(aivNum_ == 0, 
               OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "Get block dim is 0."), 
               return ge::GRAPH_FAILED
    );
    context_->SetBlockDim(aivNum_);

    context_->SetTilingKey(GetTilingKey());

    tilingData_.SaveToBuffer(
        context_->GetRawTilingData()->GetData(),
        context_->GetRawTilingData()->GetCapacity()
    );
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("NsaCompressGrad", NsaCompressGradTiling, 0);

const gert::Shape NsaCompressGradTiling::GetShapeOfInput(const size_t index)
{
    return context_->GetInputShape(index)->GetStorageShape();
}

ASCENDC_EXTERN_C ge::graphStatus TilingNsaCompressGrad(gert::TilingContext* context) {
    NsaCompressGradTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForNsaCompressGrad(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<NsaCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    OP_LOGI(
        context->GetNodeName(),
        "Parse compile info success, soc: %d", static_cast<int>(compileInfoPtr->socVersion)
    );
    return ge::GRAPH_SUCCESS;
}

bool NsaCompressGradTiling::CheckTilingData()
{
    OP_CHECK_IF(
        tilingData_.get_blockSize() % BLOCK_SIZE_16 != 0,
        OP_LOGE(context_->GetNodeName(), "Compress block size must be align to 16."),
        return false);

    OP_CHECK_IF(
        tilingData_.get_blockStride() % BLOCK_SIZE_16 != 0,
        OP_LOGE(context_->GetNodeName(), "Compress block stride must be align to 16."),
        return false);

    OP_CHECK_IF(
        tilingData_.get_dimOfHead() % BLOCK_SIZE_16 != 0,
        OP_LOGE(context_->GetNodeName(), "Head dim must be align to 16."),
        return false);

    return true;
}

IMPL_OP_OPTILING(NsaCompressGrad)
    .Tiling(TilingNsaCompressGrad)
    .TilingParse<NsaCompileInfo>(TilingPrepareForNsaCompressGrad);
} // namespace optiling
