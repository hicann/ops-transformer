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
 * \file sparse_flash_mla_grad_tiling_bs1_regbase.cpp
 * \brief
 */

#include "sparse_flash_mla_grad_tiling_bs1_regbase.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"

namespace optiling {
namespace smlag {
constexpr uint32_t WORKSPACE_BASE_CAL = 32 * 1024 * 1024; // 100MB系统预留
constexpr uint32_t BLOCK = 32;                            // 32B
constexpr uint32_t B32 = 4;                               // 4B
constexpr uint32_t B16 = 2;
constexpr uint32_t BASE_LEN_256 = 256;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t PING_PONG_BUFFER = 2;
constexpr uint32_t D_SIZE = 512;
constexpr uint32_t DROPE_SIZE = 64;
constexpr uint32_t SELECTED_BLOCK_SIZE = 1;
static const std::string METADATA_NAME = "metadata";

ge::graphStatus CheckAttentionInShape(gert::TilingContext *context)
{
    auto attentionInShape = context->GetInputShape(static_cast<size_t>(InputIndex::ATTENTION_OUT));
    if (attentionInShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        OP_LOGE(context, "The dimnum of attentionIn %zu should be equal to query %zu", attentionInShapeDim,
                queryShapeDim);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < queryShapeDim - 1; i++) {
        if (attentionInShape->GetStorageShape().GetDim(i) != queryShape->GetStorageShape().GetDim(i)) {
            OP_LOGE(context, "The dim %zu of attentionIn shape is invalid, got %ld, should be %ld", i,
                    attentionInShape->GetStorageShape().GetDim(i), queryShape->GetStorageShape().GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }

    auto oriKVShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::ORI_KV));
    auto cmpKVShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::CMP_KV));
    if (oriKVShape != nullptr) {
        auto oriKVShapeDim = oriKVShape->GetStorageShape().GetDimNum();
        if (attentionInShapeDim != oriKVShapeDim) {
            OP_LOGE(context, "The dimnum of attentionIn %zu should be equal to ori_kv %zu", attentionInShapeDim,
                    oriKVShapeDim);
            return ge::GRAPH_FAILED;
        }
        // check head_dim, attention D == value D
        if (attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1) !=
            oriKVShape->GetStorageShape().GetDim(queryShapeDim - 1)) {
            OP_LOGE(context, "The head_dim of attentionIn shape is invalid, got %ld, should be %ld",
                    attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1),
                    oriKVShape->GetStorageShape().GetDim(queryShapeDim - 1));
            return ge::GRAPH_FAILED;
        }
    }
    if (cmpKVShape != nullptr) {
        auto cmpKVShapeDim = cmpKVShape->GetStorageShape().GetDimNum();
        if (attentionInShapeDim != cmpKVShapeDim) {
            OP_LOGE(context, "The dimnum of attentionIn %zu should be equal to cmp_kv %zu", attentionInShapeDim,
                    cmpKVShapeDim);
            return ge::GRAPH_FAILED;
        }
        // check head_dim, attention D == value D
        if (attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1) !=
            cmpKVShape->GetStorageShape().GetDim(queryShapeDim - 1)) {
            OP_LOGE(context, "The head_dim of attentionIn shape is invalid, got %ld, should be %ld",
                    attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1),
                    cmpKVShape->GetStorageShape().GetDim(queryShapeDim - 1));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInDtype(gert::TilingContext *context)
{
    auto query = context->GetInputDesc(static_cast<size_t>(InputIndex::QUERY));
    auto attentionIn = context->GetInputDesc(static_cast<size_t>(InputIndex::ATTENTION_OUT));
    OP_CHECK_IF(query == nullptr || attentionIn == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query or attentionIn is nullptr."),
                return ge::GRAPH_FAILED);

    auto queryType = static_cast<uint32_t>(query->GetDataType());
    auto attentionInType = static_cast<uint32_t>(attentionIn->GetDataType());

    OP_CHECK_IF(queryType != attentionInType,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "invalid attentionIn dtype should be same with query's dtype"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndShapeValid(gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d, int64_t d2,
                                   int64_t n2)
{
    if (context == nullptr) {
        OP_LOGE(context, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto isShapeInValid = (t1 == 0 || n1 == 0 || d == 0 || d2 == 0 || n2 == 0);
    OP_CHECK_IF(
        isShapeInValid,
        OP_LOGE(context, "input shape error, got 0 in tnd(%ld,%ld,%ld) or tnd(%ld,%ld,%ld)", t1, n1, d, t1, n1, d2),
        return ge::GRAPH_FAILED);

    auto g = n1 / n2;
    auto ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeValid(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE(context, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto ret = CheckAttentionInDtype(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

bool IsSameShape(const gert::StorageShape *aShape, const gert::StorageShape *bShape)
{
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr), OP_LOGW("aShape or bShape is nullptr."), return false);
    uint32_t dimSizeA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimSizeB = bShape->GetStorageShape().GetDimNum();
    if (dimSizeA != dimSizeB) {
        return false;
    }

    for (uint32_t i = 0; i < dimSizeA; i++) {
        auto dimA = aShape->GetStorageShape().GetDim(i);
        auto dimB = bShape->GetStorageShape().GetDim(i);
        if (dimA != dimB) {
            return false;
        }
    }
    return true;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::GetShapeAttrsInfo()
{
    /*
    Get all shape info and attr
    */
    opName = context_->GetNodeName();
    OP_CHECK_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "context is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetAttrs() == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "GetAttrs is nullptr."),
                return ge::GRAPH_FAILED);

    auto status = GetBaseShapeInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    OP_LOGI(context_, "SparseFlashMlaGrad with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] s3[%d] d[%ld] d1[%ld]!",
            baseParams_->get_b(), baseParams_->get_n2(), baseParams_->get_g(), baseParams_->get_s1(),
            baseParams_->get_s2(), baseParams_->get_s3(), baseParams_->get_d(), baseParams_->get_d1());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    uint64_t l2CacheSize;
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const SparseFlashMlaGradCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compile_info is null."),
                    return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->aivNum;
        aicoreParams_.aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0aSize = compileInfoPtr->l0aSize;
        aicoreParams_.l0bSize = compileInfoPtr->l0bSize;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }

    OP_CHECK_IF((aicoreParams_.numBlocks == 0) || (aicoreParams_.aicNum == 0),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "num of coreNum(aivNum) is %lu, num of aicNum is %lu.",
                                            aicoreParams_.numBlocks, aicoreParams_.aicNum),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(aicoreParams_.ubSize <= 0 || l2CacheSize <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ubSize or l2CacheSize is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


bool SparseFlashMlaGradTilingBs1Regbase::IsCapable()
{
    OP_LOGI(context_, "SparseFlashMlaGrad basic template hit.");
    return true;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::DoOpTiling()
{
    OP_LOGI(context_, "SparseFlashMlaGrad DoTiling start");

    // Init
    tmpData.singleM = 128; // G方向上的固定切分大小
    tmpData.singleN = 128; // s2方向上的固定切分大小

    auto status = DoBlockTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = DoCastTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::GetWorkspaceSize()
{
    int64_t coreNum = aicoreParams_.aicNum;
    int64_t launchBlockDims = aicoreParams_.numBlocks;
    int64_t inputDtypeSize = B16;
    int64_t selectedS2 = tmpData.singleN;

    // Tiling传递的内存大小、起始地址，统一为字节数，单位为B
    auto blockdim = CalcTschBlockDim(launchBlockDims, aicoreParams_.aicNum, aicoreParams_.numBlocks);
    OP_CHECK_IF(blockdim == 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "blockdim is 0, aicNum is %lu, aivNum is %lu.",
                                            aicoreParams_.aicNum, aicoreParams_.numBlocks),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);

    // 使用SyncAll，需要设置为batch mode模式，所有核同时启动，否则在多流方式下执行可能会卡死
    context_->SetScheduleMode(1);

    // 系统预留
    int64_t sysLen = WORKSPACE_BASE_CAL;

    // Gather/Scatter
    int64_t selectedKWorkspaceLen = 128 * tmpData.d * B16;

    selectedKWorkspaceLen = AlignData(selectedKWorkspaceLen, GM_ALIGN) * 3;

    int64_t selectBlockCount = tmpData.deterministic ? 4096 : 128;

    int64_t mm4WorkspaceLen = selectBlockCount * tmpData.d * B32;
    int64_t mm5WorkspaceLen = selectBlockCount * tmpData.d1 * B32;
    int64_t dSinkWorkspaceLen = tmpData.g * tmpData.n2 * B32;
    mm4WorkspaceLen = AlignData(mm4WorkspaceLen, GM_ALIGN) * PING_PONG_BUFFER;
    mm5WorkspaceLen = AlignData(mm5WorkspaceLen, GM_ALIGN) * PING_PONG_BUFFER;

    int64_t dqWorkspaceLen = tmpData.dqWorkspaceLen;
    int64_t doriKVWorkspaceLen = tmpData.doriKVWorkspaceLen;
    int64_t dcmpKVWorkspaceLen = tmpData.dcmpKVWorkspaceLen;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = sysLen;
    workspaces[0] += selectedKWorkspaceLen * coreNum;
    workspaces[0] += (mm4WorkspaceLen + mm5WorkspaceLen + dSinkWorkspaceLen) * coreNum;
    workspaces[0] += dqWorkspaceLen + doriKVWorkspaceLen + dcmpKVWorkspaceLen;

    baseParams_->set_selectedKWorkSpaceOffset(0);
    int64_t workspaceOffsets = selectedKWorkspaceLen * coreNum;
    baseParams_->set_mm4ResWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets += mm4WorkspaceLen * coreNum;
    baseParams_->set_mm5ResWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets += mm5WorkspaceLen * coreNum;
    baseParams_->set_dSinkWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets += dSinkWorkspaceLen * coreNum;
    postTilingData_->set_dqWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + dqWorkspaceLen;
    postTilingData_->set_dOriKVWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + doriKVWorkspaceLen;
    postTilingData_->set_dCmpKVWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + dcmpKVWorkspaceLen;

    return ge::GRAPH_SUCCESS;
}

uint64_t SparseFlashMlaGradTilingBs1Regbase::GetTilingKey() const
{
    int64_t inputDtypeSize = tmpData.queryType == ge::DT_FLOAT16 ? 0 : 1;
    int64_t isTnd = tmpData.layout == static_cast<uint32_t>(InputLayout::TND) ? 1 : 0;
    uint64_t tilingKey = 0;

    baseParams_->set_hasUsedSeqQ(tmpData.usedSeqQ);
    baseParams_->set_hasUsedSeqOriKV(tmpData.usedSeqOriKV);
    baseParams_->set_hasUsedSeqCmpKV(tmpData.usedSeqCmpKV);
    baseParams_->set_hasOriTopK(tmpData.oriTopK);
    baseParams_->set_hasCmpTopK(tmpData.cmpTopK);
    baseParams_->set_isSink(tmpData.sinks);

    OP_LOGI(
        context_,
        "SparseFlashMlaGrad get tilingkey, InputDType[%ld], IsTnd[%ld], GTemplateNum[%ld], S2TemplateNum[%ld], \
        DTemplateNum[%ld], IsOriKVExist[%d], IsCmpKVExist[%d], IsOriKVSparse[%d], IsCmpKVSparse[%d], \
        HasUsedSeqQ[%d], HasUsedSeqOriKV[%d], HasUsedSeqCmpKV[%d], HasOriTopK[%d], HasCmpTopK[%d], IsSink[%d], \
        Deterministic[%d]",
        inputDtypeSize, isTnd, tmpData.singleM, tmpData.singleN, tmpData.d, static_cast<uint8_t>(tmpData.oriKV),
        static_cast<uint8_t>(tmpData.cmpKV), static_cast<uint8_t>(tmpData.oriKVSparse),
        static_cast<uint8_t>(tmpData.cmpKVSparse), static_cast<uint8_t>(tmpData.usedSeqQ),
        static_cast<uint8_t>(tmpData.usedSeqOriKV), static_cast<uint8_t>(tmpData.usedSeqCmpKV),
        static_cast<uint8_t>(tmpData.oriTopK), static_cast<uint8_t>(tmpData.cmpTopK),
        static_cast<uint8_t>(tmpData.sinks), static_cast<uint8_t>(tmpData.deterministic));
    // tmpData.singleM 为G方向上固定切分大小 tmpData.singleN为S2方向上固定切分大小
    tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint8_t>(inputDtypeSize), static_cast<uint8_t>(isTnd), static_cast<uint16_t>(tmpData.singleM),
        static_cast<uint16_t>(tmpData.singleN), static_cast<uint16_t>(tmpData.d), static_cast<uint8_t>(tmpData.oriKV),
        static_cast<uint8_t>(tmpData.cmpKV), static_cast<uint8_t>(tmpData.oriKVSparse),
        static_cast<uint8_t>(tmpData.cmpKVSparse), static_cast<uint8_t>(tmpData.deterministic));

    OP_LOGI(context_,
            "SparseFlashMlaGrad DoTiling success, tilingkey is"
            " %lu.",
            tilingKey);
    return tilingKey;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::DoBlockTiling()
{
    /*
     * 分核策略
     */
    int32_t nNums = tmpData.layout == static_cast<uint32_t>(InputLayout::TND) ? tmpData.t1 : tmpData.b * tmpData.s1;
    int32_t aicNum = aicoreParams_.aicNum;
    int32_t usedCoreNum = nNums < aicNum ? nNums : aicNum;
    int64_t formerCoreProcessNNums = CeilCommon(nNums, aicNum);
    int64_t remainCoreNum = formerCoreProcessNNums * aicNum - nNums;

    // 使用总核数
    baseParams_->set_usedCoreNum(usedCoreNum);
    baseParams_->set_formerCoreNum(aicNum - remainCoreNum);
    baseParams_->set_formerCoreProcessNNum(formerCoreProcessNNums);
    baseParams_->set_remainCoreProcessNNum(formerCoreProcessNNums - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::PostTiling()
{
    OP_LOGI(context_, "baseParams_.b is: [%ld]", baseParams_->get_b());
    OP_LOGI(context_, "baseParams_.n2 is: [%ld]", baseParams_->get_n2());
    OP_LOGI(context_, "baseParams_.g is: [%ld]", baseParams_->get_g());
    OP_LOGI(context_, "baseParams_.s1 is: [%ld]", baseParams_->get_s1());
    OP_LOGI(context_, "baseParams_.s2 is: [%ld]", baseParams_->get_s2());
    OP_LOGI(context_, "baseParams_.s3 is: [%ld]", baseParams_->get_s3());
    OP_LOGI(context_, "baseParams_.d is: [%ld]", baseParams_->get_d());
    OP_LOGI(context_, "baseParams_.d1 is: [%ld]", baseParams_->get_d1());
    OP_LOGI(context_, "baseParams_.oriselectedBlockCount is: [%ld]", baseParams_->get_oriselectedBlockCount());
    OP_LOGI(context_, "baseParams_.cmpselectedBlockCount is: [%ld]", baseParams_->get_cmpselectedBlockCount());
    OP_LOGI(context_, "baseParams_.usedCoreNum is: [%ld]", baseParams_->get_usedCoreNum());
    OP_LOGI(context_, "baseParams_.formerCoreNum is: [%ld]", baseParams_->get_formerCoreNum());
    OP_LOGI(context_, "baseParams_.formerCoreProcessNNum is: [%ld]", baseParams_->get_formerCoreProcessNNum());
    OP_LOGI(context_, "baseParams_.remainCoreProcessNNum is: [%ld]", baseParams_->get_remainCoreProcessNNum());
    OP_LOGI(context_, "baseParams_.layout is: [%ld]", baseParams_->get_layout());
    OP_LOGI(context_, "baseParams_.cmpSparseMode is: [%ld]", baseParams_->get_cmpSparseMode());
    OP_LOGI(context_, "baseParams_.oriSparseMode is: [%ld]", baseParams_->get_oriSparseMode());
    OP_LOGI(context_, "baseParams_.oriWinLeft is: [%ld]", baseParams_->get_oriWinLeft());
    OP_LOGI(context_, "baseParams_.oriWinRight is: [%ld]", baseParams_->get_oriWinRight());
    OP_LOGI(context_, "baseParams_.scaleValue is: [%f]", baseParams_->get_scaleValue());
    OP_LOGI(context_, "baseParams_.selectedKWorkSpaceOffset is: [%ld]", baseParams_->get_selectedKWorkSpaceOffset());
    OP_LOGI(context_, "baseParams_.mm4ResWorkSpaceOffset is: [%ld]", baseParams_->get_mm4ResWorkSpaceOffset());
    OP_LOGI(context_, "baseParams_.mm5ResWorkSpaceOffset is: [%ld]", baseParams_->get_mm5ResWorkSpaceOffset());
    OP_LOGI(context_, "baseParams_.dSinkWorkSpaceOffset is: [%ld]", baseParams_->get_dSinkWorkSpaceOffset());

    OP_LOGI(context_, "preTilingData_.qPreBlockFactor is: [%ld]", preTilingData_->get_qPreBlockFactor());
    OP_LOGI(context_, "preTilingData_.qPreBlockTotal is: [%ld]", preTilingData_->get_qPreBlockTotal());
    OP_LOGI(context_, "preTilingData_.qPreBlockTail is: [%ld]", preTilingData_->get_qPreBlockTail());
    OP_LOGI(context_, "preTilingData_.oriKVPreBlockFactor is: [%ld]", preTilingData_->get_oriKVPreBlockFactor());
    OP_LOGI(context_, "preTilingData_.oriKVPreBlockTotal is: [%ld]", preTilingData_->get_oriKVPreBlockTotal());
    OP_LOGI(context_, "preTilingData_.oriKVPreBlockTail is: [%ld]", preTilingData_->get_oriKVPreBlockTail());
    OP_LOGI(context_, "preTilingData_.cmpKVPreBlockFactor is: [%ld]", preTilingData_->get_cmpKVPreBlockFactor());
    OP_LOGI(context_, "preTilingData_.cmpKVPreBlockTotal is: [%ld]", preTilingData_->get_cmpKVPreBlockTotal());
    OP_LOGI(context_, "preTilingData_.cmpKVPreBlockTail is: [%ld]", preTilingData_->get_cmpKVPreBlockTail());
    OP_LOGI(context_, "preTilingData_.dSinksPreBlockFactor is: [%ld]", preTilingData_->get_dSinksPreBlockFactor());
    OP_LOGI(context_, "preTilingData_.dSinksPreBlockTotal is: [%ld]", preTilingData_->get_dSinksPreBlockTotal());
    OP_LOGI(context_, "preTilingData_.dSinksPreBlockTail is: [%ld]", preTilingData_->get_dSinksPreBlockTail());
    OP_LOGI(context_, "preTilingData_.oriSoftmaxL1PreBlockFactor is: [%ld]",
            preTilingData_->get_oriSoftmaxL1PreBlockFactor());
    OP_LOGI(context_, "preTilingData_.oriSoftmaxL1PreBlockTotal is: [%ld]",
            preTilingData_->get_oriSoftmaxL1PreBlockTotal());
    OP_LOGI(context_, "preTilingData_.oriSoftmaxL1PreBlockTail is: [%ld]",
            preTilingData_->get_oriSoftmaxL1PreBlockTail());
    OP_LOGI(context_, "preTilingData_.cmpSoftmaxL1PreBlockFactor is: [%ld]",
            preTilingData_->get_cmpSoftmaxL1PreBlockFactor());
    OP_LOGI(context_, "preTilingData_.cmpSoftmaxL1PreBlockTotal is: [%ld]",
            preTilingData_->get_cmpSoftmaxL1PreBlockTotal());
    OP_LOGI(context_, "preTilingData_.cmpSoftmaxL1PreBlockTail is: [%ld]",
            preTilingData_->get_cmpSoftmaxL1PreBlockTail());

    OP_LOGI(context_, "postTilingData_.postUbBaseSize is: [%ld]", postTilingData_->get_postUbBaseSize());
    OP_LOGI(context_, "postTilingData_.qPostBlockFactor is: [%ld]", postTilingData_->get_qPostBlockFactor());
    OP_LOGI(context_, "postTilingData_.qPostBlockTotal is: [%ld]", postTilingData_->get_qPostBlockTotal());
    OP_LOGI(context_, "postTilingData_.qPostBaseNum is: [%ld]", postTilingData_->get_qPostBaseNum());
    OP_LOGI(context_, "postTilingData_.qPostTailNum is: [%ld]", postTilingData_->get_qPostTailNum());
    OP_LOGI(context_, "postTilingData_.oriKVPostBlockFactor is: [%ld]", postTilingData_->get_oriKVPostBlockFactor());
    OP_LOGI(context_, "postTilingData_.oriKVPostBlockTotal is: [%ld]", postTilingData_->get_oriKVPostBlockTotal());
    OP_LOGI(context_, "postTilingData_.oriKVPostBaseNum is: [%ld]", postTilingData_->get_oriKVPostBaseNum());
    OP_LOGI(context_, "postTilingData_.oriKVPostTailNum is: [%ld]", postTilingData_->get_oriKVPostTailNum());
    OP_LOGI(context_, "postTilingData_.cmpKVPostBlockFactor is: [%ld]", postTilingData_->get_cmpKVPostBlockFactor());
    OP_LOGI(context_, "postTilingData_.cmpKVPostBlockTotal is: [%ld]", postTilingData_->get_cmpKVPostBlockTotal());
    OP_LOGI(context_, "postTilingData_.cmpKVPostBaseNum is: [%ld]", postTilingData_->get_cmpKVPostBaseNum());
    OP_LOGI(context_, "postTilingData_.cmpKVPostTailNum is: [%ld]", postTilingData_->get_cmpKVPostTailNum());
    OP_LOGI(context_, "postTilingData_.dqWorkSpaceOffset is: [%ld]", postTilingData_->get_dqWorkSpaceOffset());
    OP_LOGI(context_, "postTilingData_.dOriKVWorkSpaceOffset is: [%ld]", postTilingData_->get_dOriKVWorkSpaceOffset());
    OP_LOGI(context_, "postTilingData_.dCmpKVWorkSpaceOffset is: [%ld]", postTilingData_->get_dCmpKVWorkSpaceOffset());
    OP_LOGI(context_, "postTilingData_.sinkReduceAxis is: [%ld]", postTilingData_->get_sinkReduceAxis());
    OP_LOGI(context_, "postTilingData_.sinkPostBlockTotal is: [%ld]", postTilingData_->get_sinkPostBlockTotal());
    OP_LOGI(context_, "postTilingData_.sinkPostBlockFactor is: [%ld]", postTilingData_->get_sinkPostBlockFactor());
    OP_LOGI(context_, "postTilingData_.sinkPostTailNum is: [%ld]", postTilingData_->get_sinkPostTailNum());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::DoCastTiling()
{
    int64_t dAlign = (baseParams_->get_d() + 15) / 16 * 16;
    int64_t d1Align = (baseParams_->get_d1() + 15) / 16 * 16;
    // query
    int64_t allNumQuery =
        baseParams_->get_b() * baseParams_->get_n2() * baseParams_->get_g() * baseParams_->get_s1() * dAlign;
    // TND时候要按照真实的query的num数计算
    if (baseParams_->get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumQuery = tmpData.t1 * baseParams_->get_n2() * baseParams_->get_g() * dAlign;
    }

    // ori_kv
    int64_t allNumOriKV = baseParams_->get_b() * baseParams_->get_n2() * baseParams_->get_s2() * dAlign;
    // TND时候要按照真实的ori_kv的num数计算
    if (baseParams_->get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumOriKV = tmpData.t2 * baseParams_->get_n2() * 1 * dAlign;
    }

    // cmp_kv
    int64_t allNumCmpKV = baseParams_->get_b() * baseParams_->get_n2() * baseParams_->get_s3() * d1Align;
    // TND时候要按照真实的cmp_kv的num数计算
    if (baseParams_->get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumCmpKV = tmpData.t3 * baseParams_->get_n2() * 1 * baseParams_->get_d1();
    }

    // sinks
    int64_t allNumSinks = baseParams_->get_n2() * baseParams_->get_g();
    // TND时候要按照真实的sinks的num数计算
    if (baseParams_->get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumSinks = baseParams_->get_n2() * 1 * baseParams_->get_g();
    }

    // ori_softmax_l1
    int64_t allNumOriSoftmaxL1 =
        baseParams_->get_b() * baseParams_->get_n2() * baseParams_->get_s1() * tmpData.ori_selected_block_count;
    // TND时候要按照真实的ori_softmax_l1的num数计算
    if (baseParams_->get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumOriSoftmaxL1 = tmpData.t1 * baseParams_->get_n2() * 1 * tmpData.ori_selected_block_count;
    }

    // cmp_softmax_l1
    int64_t allNumCmpSoftmaxL1 =
        baseParams_->get_b() * baseParams_->get_n2() * baseParams_->get_s1() * tmpData.cmp_selected_block_count;
    // TND时候要按照真实的cmp_softmax_l1的num数计算
    if (baseParams_->get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumCmpSoftmaxL1 = tmpData.t1 * baseParams_->get_n2() * 1 * tmpData.cmp_selected_block_count;
    }

    uint32_t typeSize = B16;
    uint32_t usedCoreNum = aicoreParams_.numBlocks;
    uint32_t coreNum = aicoreParams_.aicNum;
    constexpr uint32_t postNzCoexNode = 12;
    constexpr uint32_t blockSize = 32;
    constexpr uint32_t postNzReservedN = 1;

    uint32_t postUbBaseSize = 0;
    uint32_t qPostBaseNum = 0;
    int64_t nzReservedSize = 0;
    int64_t curPostCoexNode = postNzCoexNode;
    postUbBaseSize = (aicoreParams_.ubSize - 2 * nzReservedSize) / curPostCoexNode / // 开DB预留2份nzReservedSize
                     BASE_LEN_256 * BASE_LEN_256;
    OP_LOGI(context_, "DoCastTiling postUbBaseSize: %ld.", postUbBaseSize);
    qPostBaseNum = 128 * 128;
    OP_LOGI(context_, "DoCastTiling qPostBaseNum: %ld.", qPostBaseNum);

    OP_CHECK_IF(qPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "qPostBaseNum is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "castUsedCoreNum is 0."),
                return ge::GRAPH_FAILED);

    int64_t qPreBlockFactor = (allNumQuery + usedCoreNum - 1) / usedCoreNum;
    int64_t qPreBlockTotal = (allNumQuery + qPreBlockFactor - 1) / qPreBlockFactor;
    int64_t qPreBlockTailTmp = allNumQuery % qPreBlockFactor;
    int64_t qPreBlockTail = qPreBlockTailTmp == 0 ? qPreBlockFactor : qPreBlockTailTmp;

    int64_t oriKVPreBlockFactor = 0;
    int64_t oriKVPreBlockTotal = 0;
    int64_t oriKVPreBlockTailTmp = 0;
    int64_t oriKVPreBlockTail = 0;
    if (allNumOriKV != 0) {
        oriKVPreBlockFactor = (allNumOriKV + usedCoreNum - 1) / usedCoreNum;
        oriKVPreBlockTotal = (allNumOriKV + oriKVPreBlockFactor - 1) / oriKVPreBlockFactor;
        oriKVPreBlockTailTmp = allNumOriKV % oriKVPreBlockFactor;
        oriKVPreBlockTail = oriKVPreBlockTailTmp == 0 ? oriKVPreBlockFactor : oriKVPreBlockTailTmp;
    }

    int64_t cmpKVPreBlockFactor = 0;
    int64_t cmpKVPreBlockTotal = 0;
    int64_t cmpKVPreBlockTailTmp = 0;
    int64_t cmpKVPreBlockTail = 0;
    if (allNumCmpKV != 0) {
        cmpKVPreBlockFactor = (allNumCmpKV + usedCoreNum - 1) / usedCoreNum;
        cmpKVPreBlockTotal = (allNumCmpKV + cmpKVPreBlockFactor - 1) / cmpKVPreBlockFactor;
        cmpKVPreBlockTailTmp = allNumCmpKV % cmpKVPreBlockFactor;
        cmpKVPreBlockTail = cmpKVPreBlockTailTmp == 0 ? cmpKVPreBlockFactor : cmpKVPreBlockTailTmp;
    }

    int64_t oriSoftmaxL1PreBlockFactor = 0;
    int64_t oriSoftmaxL1PreBlockTotal = 0;
    int64_t oriSoftmaxL1PreBlockTailTmp = 0;
    int64_t oriSoftmaxL1PreBlockTail = 0;
    if (allNumOriSoftmaxL1 != 0) {
        oriSoftmaxL1PreBlockFactor = (allNumOriSoftmaxL1 + usedCoreNum - 1) / usedCoreNum;
        oriSoftmaxL1PreBlockTotal = (allNumOriSoftmaxL1 + oriSoftmaxL1PreBlockFactor - 1) / oriSoftmaxL1PreBlockFactor;
        oriSoftmaxL1PreBlockTailTmp = allNumOriSoftmaxL1 % oriSoftmaxL1PreBlockFactor;
        oriSoftmaxL1PreBlockTail =
            oriSoftmaxL1PreBlockTailTmp == 0 ? oriSoftmaxL1PreBlockFactor : oriSoftmaxL1PreBlockTailTmp;
    }

    int64_t cmpSoftmaxL1PreBlockFactor = 0;
    int64_t cmpSoftmaxL1PreBlockTotal = 0;
    int64_t cmpSoftmaxL1PreBlockTailTmp = 0;
    int64_t cmpSoftmaxL1PreBlockTail = 0;
    if (allNumCmpSoftmaxL1 != 0) {
        cmpSoftmaxL1PreBlockFactor = (allNumCmpSoftmaxL1 + usedCoreNum - 1) / usedCoreNum;
        cmpSoftmaxL1PreBlockTotal = (allNumCmpSoftmaxL1 + cmpSoftmaxL1PreBlockFactor - 1) / cmpSoftmaxL1PreBlockFactor;
        cmpSoftmaxL1PreBlockTailTmp = allNumCmpSoftmaxL1 % cmpSoftmaxL1PreBlockFactor;
        cmpSoftmaxL1PreBlockTail =
            cmpSoftmaxL1PreBlockTailTmp == 0 ? cmpSoftmaxL1PreBlockFactor : cmpSoftmaxL1PreBlockTailTmp;
    }

    preTilingData_->set_qPreBlockFactor(qPreBlockFactor);
    preTilingData_->set_qPreBlockTotal(qPreBlockTotal);
    preTilingData_->set_qPreBlockTail(qPreBlockTail);
    preTilingData_->set_oriKVPreBlockFactor(oriKVPreBlockFactor);
    preTilingData_->set_oriKVPreBlockTotal(oriKVPreBlockTotal);
    preTilingData_->set_oriKVPreBlockTail(oriKVPreBlockTail);
    preTilingData_->set_cmpKVPreBlockFactor(cmpKVPreBlockFactor);
    preTilingData_->set_cmpKVPreBlockTotal(cmpKVPreBlockTotal);
    preTilingData_->set_cmpKVPreBlockTail(cmpKVPreBlockTail);
    preTilingData_->set_oriSoftmaxL1PreBlockFactor(oriSoftmaxL1PreBlockFactor);
    preTilingData_->set_oriSoftmaxL1PreBlockTotal(oriSoftmaxL1PreBlockTotal);
    preTilingData_->set_oriSoftmaxL1PreBlockTail(oriSoftmaxL1PreBlockTail);
    preTilingData_->set_cmpSoftmaxL1PreBlockFactor(cmpSoftmaxL1PreBlockFactor);
    preTilingData_->set_cmpSoftmaxL1PreBlockTotal(cmpSoftmaxL1PreBlockTotal);
    preTilingData_->set_cmpSoftmaxL1PreBlockTail(cmpSoftmaxL1PreBlockTail);

    int64_t qPostBlockTotal = allNumQuery / dAlign * (baseParams_->get_d());
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t oriKVPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(oriKVPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "oriKVPostBaseNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t oriKVPostBlockTotal = allNumOriKV / dAlign * (baseParams_->get_d());
    int64_t oriKVPostTailNumTmp = oriKVPostBlockTotal % oriKVPostBaseNum;
    int64_t oriKVPostTailNum = oriKVPostTailNumTmp == 0 ? oriKVPostBaseNum : oriKVPostTailNumTmp;
    int64_t oriKVPostBlockOuterTotal = (oriKVPostBlockTotal + oriKVPostBaseNum - 1) / oriKVPostBaseNum;
    int64_t oriKVPostBlockFactor = (oriKVPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t cmpKVPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(cmpKVPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "cmpKVPostBaseNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t cmpKVPostBlockTotal = allNumCmpKV / d1Align * baseParams_->get_d1();
    int64_t cmpKVPostTailNumTmp = cmpKVPostBlockTotal % cmpKVPostBaseNum;
    int64_t cmpKVPostTailNum = cmpKVPostTailNumTmp == 0 ? cmpKVPostBaseNum : cmpKVPostTailNumTmp;
    int64_t cmpKVPostBlockOuterTotal = (cmpKVPostBlockTotal + cmpKVPostBaseNum - 1) / cmpKVPostBaseNum;
    int64_t cmpKVPostBlockFactor = (cmpKVPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    postTilingData_->set_postUbBaseSize(postUbBaseSize);

    postTilingData_->set_qPostBlockFactor(qPostBlockFactor);
    postTilingData_->set_qPostBlockTotal(qPostBlockTotal);
    postTilingData_->set_qPostBaseNum(qPostBaseNum);
    postTilingData_->set_qPostTailNum(qPostTailNum);

    postTilingData_->set_oriKVPostBlockFactor(oriKVPostBlockFactor);
    postTilingData_->set_oriKVPostBlockTotal(oriKVPostBlockTotal);
    postTilingData_->set_oriKVPostBaseNum(oriKVPostBaseNum);
    postTilingData_->set_oriKVPostTailNum(oriKVPostTailNum);

    postTilingData_->set_cmpKVPostBlockFactor(cmpKVPostBlockFactor);
    postTilingData_->set_cmpKVPostBlockTotal(cmpKVPostBlockTotal);
    postTilingData_->set_cmpKVPostBaseNum(cmpKVPostBaseNum);
    postTilingData_->set_cmpKVPostTailNum(cmpKVPostTailNum);

    tmpData.dqWorkspaceLen = (allNumQuery * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    tmpData.doriKVWorkspaceLen = (allNumOriKV * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    tmpData.dcmpKVWorkspaceLen = (allNumCmpKV * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::CheckOutShapeInfo(const gert::Shape &inputshape,
                                                                      const char *inputName,
                                                                      const gert::Shape &outputshape,
                                                                      const char *inputLayout)
{
    if (strcmp(inputLayout, TND_STR) == 0) {
        if (inputshape.GetDim(DIM_0) != outputshape.GetDim(DIM_0) ||
            inputshape.GetDim(DIM_1) != outputshape.GetDim(DIM_1) ||
            inputshape.GetDim(DIM_2) != outputshape.GetDim(DIM_2)) {
            OP_LOGE(context_, "SparseFlashMlaGrad Input %s [%ld, %ld, %ld] is not equal to Output d_%s [%ld, %ld, %ld]",
                    inputName, inputshape.GetDim(DIM_0), inputshape.GetDim(DIM_1), inputshape.GetDim(DIM_2), inputName,
                    outputshape.GetDim(DIM_0), outputshape.GetDim(DIM_1), outputshape.GetDim(DIM_2));
            return ge::GRAPH_FAILED;
        }
    } else {
        if (inputshape.GetDim(DIM_0) != outputshape.GetDim(DIM_0) ||
            inputshape.GetDim(DIM_1) != outputshape.GetDim(DIM_1) ||
            inputshape.GetDim(DIM_2) != outputshape.GetDim(DIM_2) ||
            inputshape.GetDim(DIM_3) != outputshape.GetDim(DIM_3)) {
            OP_LOGE(context_,
                    "SparseFlashMlaGrad Input %s [%ld, %ld, %ld, %ld] is not equal to Output d_%s [%ld, %ld, %ld, %ld]",
                    inputName, inputshape.GetDim(DIM_0), inputshape.GetDim(DIM_1), inputshape.GetDim(DIM_2),
                    inputshape.GetDim(DIM_3), inputName, outputshape.GetDim(DIM_0), outputshape.GetDim(DIM_1),
                    outputshape.GetDim(DIM_2), outputshape.GetDim(DIM_3));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::GetOriWinToken(int64_t oriMaskMode)
{
    if (oriMaskMode != 4) {
        return ge::GRAPH_SUCCESS;
    }

    auto oriWinLeft = *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::ORI_WIN_LEFT));
    auto oriWinRight = *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::ORI_WIN_RIGHT));
    oriWinLeft = oriWinLeft == -1 ? INT32_MAX : oriWinLeft;
    oriWinRight = oriWinRight == -1 ? INT32_MAX : oriWinRight;
    if (oriWinLeft < 0 || oriWinRight < 0) {
        OP_LOGE(context_,
                "ori_win_left and ori_win_right should be greater than or equal to 0 or equal to -1, but got "
                "ori_win_left: [%ld], ori_win_right: [%ld].",
                oriWinLeft, oriWinRight);
        return ge::GRAPH_FAILED;
    }
    baseParams_->set_oriWinLeft(oriWinLeft);
    baseParams_->set_oriWinRight(oriWinRight);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradTilingBs1Regbase::GetBaseShapeInfo()
{
    OP_CHECK_IF((context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY)) == nullptr),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "InputShape of q is nullptr."), return ge::GRAPH_FAILED);

    // attrs
    const char *layoutQ = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::LAYOUT_Q));
    const char *layoutKV = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::LAYOUT_KV));

    // input
    const gert::Shape &qShape = context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY))->GetStorageShape();
    const gert::Shape &dqueryShape = context_->GetOutputShape(static_cast<size_t>(OutputIndex::DQ))->GetStorageShape();
    const gert::Shape &doriKVShape =
        context_->GetOutputShape(static_cast<size_t>(OutputIndex::DORI_KV))->GetStorageShape();
    const gert::Shape &dcmpKVShape =
        context_->GetOutputShape(static_cast<size_t>(OutputIndex::DCMP_KV))->GetStorageShape();
    const gert::Shape &dsinksShape =
        context_->GetOutputShape(static_cast<size_t>(OutputIndex::DSINKS))->GetStorageShape();
    const gert::Shape &oriSoftmaxL1Shape =
        context_->GetOutputShape(static_cast<size_t>(OutputIndex::ORI_SOFTMAX_L1_NORM))->GetStorageShape();
    const gert::Shape &cmpSoftmaxL1Shape =
        context_->GetOutputShape(static_cast<size_t>(OutputIndex::CMP_SOFTMAX_L1_NORM))->GetStorageShape();
    uint32_t dimSize = qShape.GetDimNum();
    int64_t dimDq = qShape.GetDim(dimSize - 1);

    auto status = CheckOutShapeInfo(qShape, "query", dqueryShape, layoutQ);
    if (status == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (dimDq != D_SIZE) {
        OP_LOGE(context_, "head_dim of Query[%ld] should be equal to 512.", dimDq);
        return ge::GRAPH_FAILED;
    }

    auto oriKV = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::ORI_KV));
    if (oriKV != nullptr) {
        tmpData.oriKV = true;
        const gert::Shape &oriKVShape = oriKV->GetStorageShape();
        status = CheckOutShapeInfo(oriKVShape, "ori_kv", doriKVShape, layoutKV);
        if (status == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
        int64_t dimDorikv = oriKVShape.GetDim(dimSize - 1);
        if (dimDorikv != D_SIZE) {
            OP_LOGE(context_, "head_dim of ori_kv[%ld] should be equal to 512.", dimDorikv);
            return ge::GRAPH_FAILED;
        }
        if (dimDq < dimDorikv) {
            OP_LOGE(context_, "head_dim of Query[%ld] can not less than head_dim of ori_kv[%ld].", dimDq, dimDorikv);
            return ge::GRAPH_FAILED;
        }
        if (strcmp(layoutQ, TND_STR) == 0) {
            tmpData.t2 = oriKVShape.GetDim(DIM_0);
            tmpData.n2 = oriKVShape.GetDim(DIM_1);
            tmpData.s2 = tmpData.t2;
        } else {
            tmpData.s2 = oriKVShape.GetDim(DIM_1);
            tmpData.n2 = oriKVShape.GetDim(DIM_2);
        }
    } else {
        tmpData.s2 = 0;
    }

    auto cmpKV = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::CMP_KV));
    if (cmpKV != nullptr) {
        tmpData.cmpKV = true;
        const gert::Shape &cmpKVShape = cmpKV->GetStorageShape();
        status = CheckOutShapeInfo(cmpKVShape, "cmp_kv", dcmpKVShape, layoutKV);
        if (status == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
        int64_t dimDcmpkv = cmpKVShape.GetDim(dimSize - 1);
        if (dimDcmpkv != D_SIZE) {
            OP_LOGE(context_, "head_dim of cmp_kv[%ld] should be equal to 512.", dimDcmpkv);
            return ge::GRAPH_FAILED;
        }
        if (dimDq < dimDcmpkv) {
            OP_LOGE(context_, "head_dim of Query[%ld] can not less than head_dim of cmp_kv[%ld].", dimDq, dimDcmpkv);
            return ge::GRAPH_FAILED;
        }
        if (strcmp(layoutQ, TND_STR) == 0) {
            tmpData.t3 = cmpKVShape.GetDim(DIM_0);
            tmpData.n2 = cmpKVShape.GetDim(DIM_1);
            tmpData.s3 = tmpData.t3;
        } else {
            tmpData.s3 = cmpKVShape.GetDim(DIM_1);
            tmpData.n2 = cmpKVShape.GetDim(DIM_2);
        }
    } else {
        tmpData.s3 = 0;
    }

    auto dout = context_->GetInputShape(static_cast<size_t>(InputIndex::ATTENTION_OUT_GRAD))->GetStorageShape();
    auto out = context_->GetInputShape(static_cast<size_t>(InputIndex::ATTENTION_OUT))->GetStorageShape();

    auto oriSparseIndices = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::ORI_SPARSE_INDICES));
    if (oriSparseIndices != nullptr) {
        tmpData.oriKVSparse = true;
        const gert::Shape &oriSparseIndicesShape = oriSparseIndices->GetStorageShape();
        auto ori_selected_block_count = oriSparseIndicesShape.GetDim(dimSize - 1);
        baseParams_->set_oriselectedBlockCount(ori_selected_block_count);
        tmpData.ori_selected_block_count = ori_selected_block_count;
    } else {
        int64_t ori_selected_block_count = 0;
        baseParams_->set_oriselectedBlockCount(ori_selected_block_count);
        tmpData.ori_selected_block_count = ori_selected_block_count;
    }

    auto cmpSparseIndices = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::CMP_SPARSE_INDICES));
    if (cmpSparseIndices != nullptr) {
        tmpData.cmpKVSparse = true;
        const gert::Shape &cmpSparseIndicesShape = cmpSparseIndices->GetStorageShape();
        auto cmp_selected_block_count = cmpSparseIndicesShape.GetDim(dimSize - 1);
        baseParams_->set_cmpselectedBlockCount(cmp_selected_block_count);
        tmpData.cmp_selected_block_count = cmp_selected_block_count;
    } else {
        int64_t cmp_selected_block_count = 0;
        baseParams_->set_cmpselectedBlockCount(cmp_selected_block_count);
        tmpData.cmp_selected_block_count = cmp_selected_block_count;
    }

    auto usedSeqQ = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SEQUSED_Q));
    if (usedSeqQ != nullptr) {
        tmpData.usedSeqQ = true;
        const gert::Shape &usedSeqQShape = usedSeqQ->GetStorageShape();
    }

    auto usedSeqOriKV = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SEQUSED_ORI_KV));
    if (usedSeqOriKV != nullptr) {
        tmpData.usedSeqOriKV = true;
        const gert::Shape &usedSeqOriKVShape = usedSeqOriKV->GetStorageShape();
    }

    auto oriMaskMode = *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::ORI_MASK_MODE));
    baseParams_->set_oriSparseMode(oriMaskMode);
    auto cmpMaskMode = *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::CMP_MASK_MODE));
    baseParams_->set_cmpSparseMode(cmpMaskMode);
    if (GetOriWinToken(oriMaskMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto cmpRatio = *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::CMP_RATIO));
    if (cmpRatio < 1 || cmpRatio > 128) {
        OP_LOGE(context_, "cmp_ratio must be range [1,128], but now get input cmp_ratio [%ld]!", cmpRatio);
        return ge::GRAPH_FAILED;
    }
    auto usedSeqCmpKV = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SEQUSED_CMP_KV));
    if (usedSeqCmpKV != nullptr) {
        tmpData.usedSeqCmpKV = true;
        const gert::Shape &usedSeqCmpKVShape = usedSeqCmpKV->GetStorageShape();
    }
    auto cmpResidualKV = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::CMP_RESIDUAL_KV));
    if (cmpMaskMode == 3 && cmpRatio != 1 && cmpResidualKV == nullptr) {
        OP_LOGE(context_, "cmp_residual_kv is required when cmp_mask_mode is 3 with cmp_ratio not equal to 1!");
        return ge::GRAPH_FAILED;
    }
    auto oriTopkLength = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::ORI_TOPK_LENGTH));
    if (oriSparseIndices != nullptr && oriMaskMode == 0 && oriTopkLength == nullptr) {
        OP_LOGE(context_, "ori_topk_length is required when ori_sparse_indices exist with ori_mask_mode equal to 0!");
        return ge::GRAPH_FAILED;
    }
    if (oriTopkLength != nullptr) {
        tmpData.oriTopK = true;
        const gert::Shape &oriTopkLengthShape = oriTopkLength->GetStorageShape();
    }

    auto cmpTopkLength = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::CMP_TOPK_LENGTH));
    if (cmpSparseIndices != nullptr && cmpMaskMode == 0 && cmpTopkLength == nullptr) {
        OP_LOGE(context_, "cmp_topk_length is required when cmp_sparse_indices exist with cmp_mask_mode equal to 0!");
        return ge::GRAPH_FAILED;
    }
    if (cmpTopkLength != nullptr) {
        tmpData.cmpTopK = true;
        const gert::Shape &cmpTopkLengthShape = cmpTopkLength->GetStorageShape();
    }

    auto softmaxLseShape = context_->GetInputShape(static_cast<size_t>(InputIndex::LSE))->GetStorageShape();
    if (softmaxLseShape.GetDimNum() == 0) {
        OP_LOGE(context_, "The shape of softmax_lse cannot be zero! ");
        return ge::GRAPH_FAILED;
    }

    auto sinks = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SINKS));
    if (sinks != nullptr) {
        tmpData.sinks = true;
        const gert::Shape &sinksShape = sinks->GetStorageShape();
    }

    auto metadataTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::METADATA));
    if (metadataTensor == nullptr) {
        OP_LOGE(context_, "%s must be provided!", METADATA_NAME.c_str());
        return ge::GRAPH_FAILED;
    }

    auto selected_block_size = SELECTED_BLOCK_SIZE;
    if (selected_block_size != 1) {
        OP_LOGE(context_, "SparseFlashMlaGrad only support sparse_block_size [1] now, but got sparse_block_size=%ld.",
                selected_block_size);
        return ge::GRAPH_FAILED;
    }

    tmpData.deterministic = (context_->GetDeterministic() == 1);

    if (strcmp(layoutQ, TND_STR) != 0 && strcmp(layoutQ, BSND_STR) != 0 && strcmp(layoutKV, TND_STR) != 0 &&
        strcmp(layoutKV, BSND_STR) != 0 && strcmp(layoutQ, layoutKV) != 0) {
        OP_LOGE(context_,
                "SparseFlashMlaGrad only support TND or BSND layout, and layoutQ must equal to layoutKV, but now "
                "layoutQ is %s and layoutKV is %s.",
                layoutQ, layoutKV);
        return ge::GRAPH_FAILED;
    }

    if (static_cast<size_t>(dimSize) != strlen(layoutQ)) {
        OP_LOGE(context_, "SparseFlashMlaGrad layout dims is not equal to the input's dim, now the query of dim is %u.",
                dimSize);
        return ge::GRAPH_FAILED;
    }

    if (strcmp(layoutQ, TND_STR) == 0) {
        const gert::Shape &actSeqQLenShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CU_SEQLENS_Q))->GetStorageShape();
        tmpData.b = actSeqQLenShape.GetDim(DIM_0) - 1;
        tmpData.t1 = qShape.GetDim(DIM_0);
        tmpData.s1 = tmpData.t1;
        tmpData.s2 = tmpData.t2;
        tmpData.s3 = tmpData.t3;
        OP_CHECK_IF(tmpData.n2 == 0, OP_LOGE(context_, "headnum of key and value is 0"), return ge::GRAPH_FAILED);
        tmpData.g = qShape.GetDim(DIM_1) / tmpData.n2;
        tmpData.layout = static_cast<uint32_t>(InputLayout::TND);
    } else { // BSND
        tmpData.b = qShape.GetDim(DIM_0);
        tmpData.s1 = qShape.GetDim(DIM_1);
        OP_CHECK_IF(tmpData.n2 == 0, OP_LOGE(context_, "headnum of key and value is 0"), return ge::GRAPH_FAILED);
        tmpData.g = qShape.GetDim(DIM_2) / tmpData.n2;
        tmpData.layout = static_cast<uint32_t>(InputLayout::BSND);
    }

    if (tmpData.g <= 0) {
        OP_LOGE(context_, "g (N1 / N2) should be larger than 0, but got g=%ld.", tmpData.g);
        return ge::GRAPH_FAILED;
    }

    int64_t n1 = tmpData.n2 * tmpData.g;
    if (tmpData.n2 != 1 && (n1 != 128 || n1 != 64)) {
        OP_LOGE(context_, "SparseFlashMlaGrad only support n2=1 and n1=64/128, but got n2=%ld n1=%ld.", tmpData.n2, n1);
    }

    baseParams_->set_b(tmpData.b);
    baseParams_->set_g(tmpData.g);
    OP_CHECK_IF(baseParams_->get_g() == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);
    baseParams_->set_n2(tmpData.n2);
    baseParams_->set_s1(tmpData.s1);
    baseParams_->set_s2(tmpData.s2);
    baseParams_->set_s3(tmpData.s3);
    // d=576 dq+drope /  d1=512 dq
    baseParams_->set_d(dimDq);
    baseParams_->set_d1(dimDq);
    baseParams_->set_layout(tmpData.layout);
    baseParams_->set_scaleValue(
        *context_->GetAttrs()->GetAttrPointer<float>(static_cast<size_t>(AttrIndex::SCALE_VALUE)));
    baseParams_->set_cmpRatio(cmpRatio);

    tmpData.d = baseParams_->get_d();
    tmpData.d1 = baseParams_->get_d1();
    tmpData.dataTypeSize = B32;
    tmpData.queryType =
        static_cast<uint32_t>(context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType());
    tmpData.selected_block_size = selected_block_size;

    auto ret = CheckDtypeValid(context_);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SparseFlashMlaGrad the dtype of input is invalid.");
        return ge::GRAPH_FAILED;
    }

    if (tmpData.layout == static_cast<uint32_t>(InputLayout::TND)) {
        ret = CheckTndShapeValid(context_, tmpData.t1, tmpData.n2 * tmpData.g, tmpData.d, tmpData.d1, tmpData.n2);
    }

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SparseFlashMlaGrad the input shpae of TND Layout is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (tmpData.layout == static_cast<uint32_t>(InputLayout::TND)) {
        ret = CheckTndShapeValid(context_, tmpData.t1, tmpData.n2 * tmpData.g, tmpData.d, tmpData.d1, tmpData.n2);
    }

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SparseFlashMlaGrad the input shpae of TND Layout is invalid.");
        return ge::GRAPH_FAILED;
    }
    int32_t nNums = tmpData.layout == static_cast<uint32_t>(InputLayout::TND) ? tmpData.t1 : tmpData.b * tmpData.s1;
    baseParams_->set_totalSize(nNums);

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(SparseFlashMlaGrad, SparseFlashMlaGradTilingBs1Regbase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace smlag
} // namespace optiling
