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
 * \file sparse_flash_mla_grad_tiling_general.cpp
 * \brief
 */

#include "sparse_flash_mla_grad_tiling_general.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/sparse_flash_mla_grad_template_tiling_key.h"

namespace optiling {
namespace smlag {
constexpr uint32_t WORKSPACE_BASE_CAL = 32 * 1024 * 1024; // 32MB系统预留
constexpr uint32_t BLOCK = 32;                            // 32B
constexpr uint32_t B32 = 4;                               // 4B
constexpr uint32_t B16 = 2;
constexpr uint32_t SINGLE_N_512 = 512;
constexpr uint32_t SINGLE_N_256 = 256;
constexpr uint32_t SINGLE_N_128 = 128;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t PING_PONG_BUFFER = 2;
constexpr uint32_t SINGLE_M = 128;

ge::graphStatus SparseFlashMlaGradBasicTiling::GetShapeAttrsInfo()
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

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradBasicTiling::GetPlatformInfo()
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

bool SparseFlashMlaGradBasicTiling::IsCapable()
{
    OP_LOGI(context_, "SparseFlashMlaGrad basic template hit.");
    return true;
}

ge::graphStatus SparseFlashMlaGradBasicTiling::DoOpTiling()
{
    OP_LOGI(context_, "SparseFlashMlaGrad DoTiling start");

    // Init
    int32_t selBlkCntAlign = CeilCommon(tilingData.opInfo.get_selectedBlockCount(), 8) * 8;
    tmpData.singleM = tmpData.mode == SMLAG_SCFA_MODE ? tilingData.opInfo.get_G() : SINGLE_M;
    if (tmpData.mode == SMLAG_SCFA_MODE && selBlkCntAlign >= SINGLE_N_512) {
        tmpData.singleN = tilingData.opInfo.get_selectedBlockCount() >= 1024 &&
                                  tilingData.opInfo.get_selectedBlockCount() % SINGLE_N_512 == 0 &&
                                  tilingData.opInfo.get_G() <= 64 ?
                              SINGLE_N_256 :
                              SINGLE_N_512;
    } else {
        tmpData.singleN = SINGLE_N_128;
    }
    tmpData.s1BasicSize = tmpData.mode == SMLAG_SCFA_MODE ? 1 : tmpData.singleM / tilingData.opInfo.get_G();

    // setTilingData
    tilingData.splitCoreParams.set_singleM(tmpData.singleM);
    tilingData.splitCoreParams.set_singleN(tmpData.singleN);
    tilingData.splitCoreParams.set_s1BasicSize(tmpData.s1BasicSize);

    auto status = DoSftTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = DoBlockTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = DoCastTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradBasicTiling::DoLibApiTiling()
{
    // calc for simpleSoftMax which dstShape is as same as srcShape
    auto simpleSoftMaxShape = ge::Shape({tmpData.singleM, tmpData.singleN});
    auto helpLenA = tmpData.singleM * tmpData.singleN * tmpData.dataTypeSize; // UB内数据类型
    AscendC::SoftMaxTilingFunc(simpleSoftMaxShape, sizeof(float), helpLenA, tilingData.softmaxTilingData);

    // calc for softmaxGrad
    auto softmaxGradShape = ge::Shape({tmpData.singleM, BLOCK / tmpData.dataTypeSize});
    auto helpLenB = 2 * tmpData.singleM * tmpData.singleN * tmpData.dataTypeSize; // UB内数据类型 64KB
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, tmpData.dataTypeSize, helpLenB, tilingData.softmaxGradTilingData,
                                   true);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradBasicTiling::GetWorkspaceSize()
{
    int64_t currentUseCoreNum = tilingData.opInfo.get_usedCoreNum();
    int64_t launchBlockDims = aicoreParams_.numBlocks;
    int64_t inputDtypeSize = tmpData.queryType == ge::DT_FLOAT ? B32 : B16;
    int64_t selectedS2 = tmpData.singleN;

    // Tiling传递的内存大小、起始地址，统一为字节数，单位为B
    auto blockdim = CalcTschBlockDim(launchBlockDims, aicoreParams_.aicNum, aicoreParams_.numBlocks);
    OP_CHECK_IF(blockdim == 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "blockdim is 0, aicNum is %lu, aivNum is %lu.",
                                            aicoreParams_.aicNum, aicoreParams_.numBlocks),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);

    // 系统预留
    int64_t sysLen = WORKSPACE_BASE_CAL;
    int64_t mm12WorkspaceLen = tmpData.singleM * tmpData.singleN * B32;
    mm12WorkspaceLen = AlignData(mm12WorkspaceLen, GM_ALIGN) * PING_PONG_BUFFER;

    int64_t dqWorkspaceLen = tilingData.opInfo.get_dqWorkspaceLen();
    int64_t dkWorkspaceLen = tilingData.opInfo.get_dkWorkspaceLen(); // 包括ori+cmp

    int64_t selectedKWorkspaceLen = 0;
    if (tmpData.mode == SMLAG_SCFA_MODE) {
        // Gather/Scatter
        selectedKWorkspaceLen = selectedS2 * tmpData.d * inputDtypeSize;
        selectedKWorkspaceLen = AlignData(selectedKWorkspaceLen, GM_ALIGN);
        selectedKWorkspaceLen *= 4;
    }

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = sysLen;
    workspaces[0] += selectedKWorkspaceLen * currentUseCoreNum;
    workspaces[0] += mm12WorkspaceLen * 4 * currentUseCoreNum;
    workspaces[0] += dqWorkspaceLen + dkWorkspaceLen;

    if (tmpData.mode == SMLAG_SCFA_MODE) {
        int64_t dAlign = (tilingData.opInfo.get_D() + 15) / 16 * 16;
        // 每个s1做完，做scatter add累加，workspace开DB
        workspaces[0] += 24 * PING_PONG_BUFFER * tmpData.selected_block_count * (dAlign + dAlign) * B32;
        workspaces[0] += tilingData.opInfo.get_additionalWorkspaceLen(); // 用于保存原先dk，在post阶段muls(scale)
    } else {
        workspaces[0] += dkWorkspaceLen; // 用于保存原先dk，在post阶段muls(scale)
    }

    tilingData.opInfo.set_mm12WorkspaceLen(mm12WorkspaceLen);
    tilingData.opInfo.set_selectedKWorkspaceLen(selectedKWorkspaceLen);

    int64_t workspaceOffsets = selectedKWorkspaceLen * currentUseCoreNum;
    workspaceOffsets += mm12WorkspaceLen * 4 * currentUseCoreNum;
    tilingData.postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + dqWorkspaceLen;
    tilingData.postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradBasicTiling::PostTiling()
{
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t SparseFlashMlaGradBasicTiling::GetTilingKey() const
{
    // -------------set tilingkey-----------------
    // LAYOUT: 0(BSND); 1(TND)
    // CMP_MODE: 0(SWA); 1(CFA); 2(SCFA)
    uint32_t layout = tmpData.layout == static_cast<uint32_t>(InputLayout::TND) ? 1U : 0U;
    return GET_TPL_TILING_KEY(layout, tmpData.mode);
}

ge::graphStatus SparseFlashMlaGradBasicTiling::DoBlockTiling()
{
    /*
     * 分核策略
     */
    int32_t nNums = tmpData.layout == static_cast<uint32_t>(InputLayout::TND) ?
                        CeilCommon(tmpData.t1, tmpData.s1BasicSize) :
                        CeilCommon(tmpData.b * tmpData.s1, tmpData.s1BasicSize);
    int32_t aicNum = aicoreParams_.aicNum;
    int32_t usedCoreNum = nNums < aicNum ? nNums : aicNum;
    int64_t formerCoreProcessNNums = CeilCommon(nNums, aicNum);
    int64_t remainCoreNum = formerCoreProcessNNums * aicNum - nNums;

    tilingData.opInfo.set_usedCoreNum(usedCoreNum);
    tilingData.opInfo.set_formerCoreNum(aicNum - remainCoreNum);
    tilingData.opInfo.set_formerCoreProcessNNum(formerCoreProcessNNums);
    tilingData.opInfo.set_remainCoreProcessNNum(static_cast<uint32_t>(formerCoreProcessNNums - 1));
    tilingData.opInfo.set_castUsedCoreNum(aicoreParams_.numBlocks);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradBasicTiling::DoSftTiling()
{
    /*
     * softmax tiling切分策略
     */
    constexpr int32_t maxUbSize = 191 * 1024;

    uint32_t sftBaseN = tmpData.singleN;
    uint32_t sftBaseNAlign = (tmpData.singleN + 8 - 1) / 8 * 8;
    uint32_t sftBaseM = (maxUbSize - tilingData.opInfo.get_G() * 32 - sftBaseNAlign * B32) /
                        (3 * 32 + sftBaseN * (B32 * 3 + B16) + tilingData.opInfo.get_D() * (B16 + B32) * 2 + 2 * B32);

    uint32_t actualUsed =
        sftBaseM * (3 * 32 + sftBaseN * (B32 * 3 + B16) + tilingData.opInfo.get_D() * (B16 + B32) * 2) +
        ((sftBaseM + 8 - 1) / 8 * 8) * B32 * 2 + sftBaseNAlign * B32 + tilingData.opInfo.get_G() * 32;
    while (actualUsed > maxUbSize) {
        sftBaseM -= 1;
        actualUsed = sftBaseM * (3 * 32 + sftBaseN * (B32 * 3 + B16) + tilingData.opInfo.get_D() * (B16 + B32) * 2) +
                     ((sftBaseM + 8 - 1) / 8 * 8) * B32 * 2 + sftBaseNAlign * B32 + tilingData.opInfo.get_G() * 32;
    }

    tilingData.splitCoreParams.set_sftBaseM(sftBaseM);
    tilingData.splitCoreParams.set_sftBaseN(sftBaseN);

    uint32_t remainSize = maxUbSize - sftBaseNAlign * B32 - tilingData.opInfo.get_G() * 32;
    int64_t maxGatherSize = remainSize / (tilingData.opInfo.get_D() * B16 * PING_PONG_BUFFER);
    maxGatherSize = maxGatherSize / 2 * 2;
    tilingData.splitCoreParams.set_maxGatherSize(maxGatherSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradBasicTiling::DoCastTiling()
{
    int64_t dAlign = (tilingData.opInfo.get_D() + 15) / 16 * 16;
    // query
    int64_t allNumQuery = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_G() *
                          tilingData.opInfo.get_S1() * dAlign;
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumQuery = tmpData.t1 * tilingData.opInfo.get_N2() * tilingData.opInfo.get_G() * dAlign;
    }

    // ori_kv
    int64_t allNumOriKv = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_S2() * dAlign;
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumOriKv = tmpData.t2 * tilingData.opInfo.get_N2() * 1 * dAlign;
    }

    // cmp_kv
    int64_t allNumCmpKv = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_S3() * dAlign;
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumCmpKv = tmpData.t3 * tilingData.opInfo.get_N2() * 1 * dAlign;
    }

    // ori_kv + cmp_kv
    int64_t allNumKey = allNumOriKv + allNumCmpKv;

    uint32_t typeSize = tmpData.queryType == ge::DT_FLOAT ? B32 : B16;
    uint32_t usedCoreNum = tilingData.opInfo.get_castUsedCoreNum();

    uint32_t postUbBaseSize = 0;
    uint32_t qPostBaseNum = 0;
    int64_t curPostCoexNode = 7;
    postUbBaseSize = aicoreParams_.ubSize / curPostCoexNode;
    qPostBaseNum = postUbBaseSize / typeSize / dAlign * tilingData.opInfo.get_D();

    OP_CHECK_IF(qPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "qPostBaseNum is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "castUsedCoreNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t qPostBlockTotal = allNumQuery / dAlign * tilingData.opInfo.get_D();
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t oriKvPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(oriKvPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "oriKvPostBaseNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t oriKvPostBlockTotal = allNumOriKv / dAlign * tilingData.opInfo.get_D();
    int64_t oriKvPostTailNumTmp = oriKvPostBlockTotal % oriKvPostBaseNum;
    int64_t oriKvPostTailNum = oriKvPostTailNumTmp == 0 ? oriKvPostBaseNum : oriKvPostTailNumTmp;
    int64_t oriKvPostBlockOuterTotal = (oriKvPostBlockTotal + oriKvPostBaseNum - 1) / oriKvPostBaseNum;
    int64_t oriKvPostBlockFactor = (oriKvPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    tilingData.postTilingData.set_coreNum(usedCoreNum);
    tilingData.postTilingData.set_scaleValue(tilingData.opInfo.get_scaleValue());
    tilingData.postTilingData.set_postUbBaseSize(postUbBaseSize);

    tilingData.postTilingData.set_qPostBlockFactor(qPostBlockFactor);
    tilingData.postTilingData.set_qPostBlockTotal(qPostBlockTotal);
    tilingData.postTilingData.set_qPostBaseNum(qPostBaseNum);
    tilingData.postTilingData.set_qPostTailNum(qPostTailNum);

    tilingData.postTilingData.set_oriKvPostBlockFactor(oriKvPostBlockFactor);
    tilingData.postTilingData.set_oriKvPostBlockTotal(oriKvPostBlockTotal);
    tilingData.postTilingData.set_oriKvPostBaseNum(oriKvPostBaseNum);
    tilingData.postTilingData.set_oriKvPostTailNum(oriKvPostTailNum);

    if (tmpData.mode != SMLAG_SWA_MODE) {
        int64_t cmpKvPostBaseNum = qPostBaseNum;
        OP_CHECK_IF(cmpKvPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "cmpKvPostBaseNum is 0."),
                    return ge::GRAPH_FAILED);
        int64_t cmpKvPostBlockTotal = allNumCmpKv / dAlign * tilingData.opInfo.get_D();
        int64_t cmpKvPostTailNumTmp = cmpKvPostBlockTotal % cmpKvPostBaseNum;
        int64_t cmpKvPostTailNum = cmpKvPostTailNumTmp == 0 ? cmpKvPostBaseNum : cmpKvPostTailNumTmp;
        int64_t cmpKvPostBlockOuterTotal = (cmpKvPostBlockTotal + cmpKvPostBaseNum - 1) / cmpKvPostBaseNum;
        int64_t cmpKvPostBlockFactor = (cmpKvPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

        tilingData.postTilingData.set_cmpKvPostBlockFactor(cmpKvPostBlockFactor);
        tilingData.postTilingData.set_cmpKvPostBlockTotal(cmpKvPostBlockTotal);
        tilingData.postTilingData.set_cmpKvPostBaseNum(cmpKvPostBaseNum);
        tilingData.postTilingData.set_cmpKvPostTailNum(cmpKvPostTailNum);
    }

    tilingData.opInfo.set_dqWorkspaceLen((allNumQuery * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    tilingData.opInfo.set_dkWorkspaceLen((allNumKey * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    if (tmpData.mode == SMLAG_SCFA_MODE) {
        tilingData.opInfo.set_additionalWorkspaceLen((allNumOriKv * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradBasicTiling::GetBaseShapeInfo()
{
    OP_CHECK_IF(((context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY)) == nullptr) ||
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::ORI_KV)) == nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "InputShape of query, ori_kv is nullptr."),
                return ge::GRAPH_FAILED);
    // -------------- input ----------------
    // TND: query [t1, n1, d]  ori_kv [t2, n2, d]  cmp_kv [t3, n2, d]   dy/attentionIn [t1, n1, d]
    // BSND: query [b, s1, n1, d]   k [b, s2, n2, d]  v [b, s3, n2, d]   dy/attentionIn [b, s1, n1, d]
    const gert::Shape &queryShape = context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY))->GetStorageShape();
    const gert::Shape &oriKvShape = context_->GetInputShape(static_cast<size_t>(InputIndex::ORI_KV))->GetStorageShape();
    auto cmpKvTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_KV));
    auto cmpIndicesTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_SPARSE_INDICES));

    if (queryShape.GetDimNum() != oriKvShape.GetDimNum()) {
        OP_LOGE(context_, "The dim num of inputs shape should be the same.");
        return ge::GRAPH_FAILED;
    }
    uint32_t dimSize = queryShape.GetDimNum();
    int64_t dimDq = queryShape.GetDim(dimSize - 1);
    int64_t dimOriKv = oriKvShape.GetDim(dimSize - 1);

    // -------------- 当前不支持参数必须传空校验 --------------
    auto oriSparseIndicesTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ORI_SPARSE_INDICES));
    auto seqUsedQTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::SEQUSED_Q));
    auto seqUsedOriKvTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::SEQUSED_ORI_KV));
    auto seqUsedCmpKvTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::SEQUSED_CMP_KV));
    auto oriTopkLenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ORI_TOPK_LENGTH));
    auto cmpTopkLenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_TOPK_LENGTH));
    auto metadataTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::METADATA));
    OP_CHECK_IF(oriSparseIndicesTensor != nullptr,
                OP_LOGE("SparseFlashMlaGrad", "oriSparseIndices should be nullptr now."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqUsedQTensor != nullptr, OP_LOGE("SparseFlashMlaGrad", "seqUsedQ should be nullptr now."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqUsedOriKvTensor != nullptr, OP_LOGE("SparseFlashMlaGrad", "seqUsedOriKv should be nullptr now."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqUsedCmpKvTensor != nullptr, OP_LOGE("SparseFlashMlaGrad", "seqUsedCmpKv should be nullptr now."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriTopkLenTensor != nullptr, OP_LOGE("SparseFlashMlaGrad", "oriTopkLength should be nullptr now."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cmpTopkLenTensor != nullptr, OP_LOGE("SparseFlashMlaGrad", "cmpTopkLength should be nullptr now."),
                return ge::GRAPH_FAILED);

    // -------------- attrs ----------------
    const char *inputLayoutQ = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::LAYOUT_Q));
    const char *inputLayoutKv = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::LAYOUT_KV));
    const int64_t *cmpRatioPtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::CMP_RATIO));
    const int64_t *oriWinLeftPtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::ORI_WIN_LEFT));
    const int64_t *oriWinRightPtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::ORI_WIN_RIGHT));
    const int64_t *oriMaskModePtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::ORI_MASK_MODE));
    const int64_t *cmpMaskModePtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::CMP_MASK_MODE));

    // -------------------------------------
    if (context_->GetDeterministic() == 1) {
        OP_LOGE(context_, "SparseFlashMlaGrad not support deterministic yet.");
        return ge::GRAPH_FAILED;
    }

    if (dimDq != dimOriKv) {
        OP_LOGE(context_, "The headDim of query, ori_kv should be the same. But got dimDq=%ld, dimOriKv=%ld", dimDq,
                dimOriKv);
        return ge::GRAPH_FAILED;
    }
    int64_t N2 = oriKvShape.GetDim(dimSize - 2);
    if (strcmp(inputLayoutQ, inputLayoutKv) != 0) {
        OP_LOGE(context_,
                "SparseFlashMlaGrad layout_q and layout_kv should be same, now layout_q is %s and layout_kv is %s.",
                inputLayoutQ, inputLayoutKv);
        return ge::GRAPH_FAILED;
    }
    if (strcmp(inputLayoutQ, TND_STR) != 0 && strcmp(inputLayoutQ, BSND_STR) != 0) {
        OP_LOGE(context_, "SparseFlashMlaGrad only support TND or BSND layout, now layout is %s.", inputLayoutQ);
        return ge::GRAPH_FAILED;
    }
    if (static_cast<size_t>(dimSize) != strlen(inputLayoutQ)) {
        OP_LOGE(context_, "SparseFlashMlaGrad layout dims is not equal to the input's dim, now the query of dim is %u.",
                dimSize);
        return ge::GRAPH_FAILED;
    }

    if (strcmp(inputLayoutQ, TND_STR) == 0) {
        const gert::Shape &actSeqQLenShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CU_SEQLENS_Q))->GetStorageShape();
        tmpData.b = actSeqQLenShape.GetDim(DIM_0) - 1;
        tmpData.t1 = queryShape.GetDim(DIM_0);
        tmpData.t2 = oriKvShape.GetDim(DIM_0);
        tmpData.t3 = 0;
        tmpData.s1 = tmpData.t1;
        tmpData.s2 = tmpData.t2;
        tmpData.s3 = 0;
        tmpData.n2 = N2;
        OP_CHECK_IF(tmpData.n2 == 0, OP_LOGE(context_, "N2 is 0"), return ge::GRAPH_FAILED);
        tmpData.g = queryShape.GetDim(DIM_1) / tmpData.n2;
        tmpData.layout = static_cast<uint32_t>(InputLayout::TND);
    } else { // BSND
        tmpData.b = queryShape.GetDim(DIM_0);
        tmpData.s1 = queryShape.GetDim(DIM_1);
        tmpData.s2 = oriKvShape.GetDim(DIM_1);
        tmpData.s3 = 0;
        tmpData.n2 = N2;
        OP_CHECK_IF(tmpData.n2 == 0, OP_LOGE(context_, "N2 is 0"), return ge::GRAPH_FAILED);
        tmpData.g = queryShape.GetDim(DIM_2) / tmpData.n2;
        tmpData.layout = static_cast<uint32_t>(InputLayout::BSND);
    }

    if (cmpKvTensor != nullptr) {
        const gert::Shape &cmpKvShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_KV))->GetStorageShape();
        int64_t dimCmpKv = cmpKvShape.GetDim(dimSize - 1);
        if (dimDq != dimCmpKv) {
            OP_LOGE(context_, "The headDim of query, cmp_kv should be the same. But got dimDq=%ld, dimCmpKv=%ld", dimDq,
                    dimCmpKv);
            return ge::GRAPH_FAILED;
        }
        if (strcmp(inputLayoutQ, TND_STR) == 0) {
            tmpData.t3 = cmpKvShape.GetDim(DIM_0);
            tmpData.s3 = tmpData.t3;
        } else {
            tmpData.s3 = cmpKvShape.GetDim(DIM_1);
        }
    }

    OP_CHECK_IF(cmpRatioPtr == nullptr, OP_LOGE("SparseFlashMlaGrad", "cmpRatioPtr is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriWinLeftPtr == nullptr, OP_LOGE("SparseFlashMlaGrad", "oriWinLeftPtr is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriWinRightPtr == nullptr, OP_LOGE("SparseFlashMlaGrad", "oriWinRightPtr is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cmpMaskModePtr == nullptr, OP_LOGE("SparseFlashMlaGrad", "cmpMaskModePtr is null"),
                return ge::GRAPH_FAILED);

    if (tmpData.g <= 0) {
        OP_LOGE(context_, "g (N1 / N2) should be larger than 0, but got g=%ld.", tmpData.g);
        return ge::GRAPH_FAILED;
    }

    int64_t n1 = tmpData.n2 * tmpData.g;
    if (tmpData.n2 != 1 || n1 > 128) {
        OP_LOGE(context_, "SparseFlashMlaGrad only support n2=1 and 1<=n1<=128, but got n2=%ld n1=%ld.", tmpData.n2,
                n1);
        return ge::GRAPH_FAILED;
    }

    int64_t cmpRatio = *cmpRatioPtr;
    if (!(cmpRatio >= 1 && cmpRatio <= 128)) {
        OP_LOGE(context_, "SparseFlashMlaGrad only support cmpRatio >= 1 and <=128, but got cmpRatio=%ld.", cmpRatio);
        return ge::GRAPH_FAILED;
    }

    int64_t oriWinLeft = *oriWinLeftPtr;
    int64_t oriWinRight = *oriWinRightPtr;
    OP_CHECK_IF(oriWinLeft != 127 || oriWinRight != 0,
                OP_LOGE("SparseFlashMlaGrad", "SparseFlashMlaGrad only support oriWinLeft=127 and oriWinRight=0 now."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(*oriMaskModePtr != 4,
                OP_LOGE("SparseFlashMlaGrad", "SparseFlashMlaGrad only support oriMaskMode=4 now, but got %ld.",
                        *oriMaskModePtr),
                return ge::GRAPH_FAILED);

    tilingData.opInfo.set_B(tmpData.b);
    tilingData.opInfo.set_G(tmpData.g);
    tilingData.opInfo.set_N2(tmpData.n2);
    tilingData.opInfo.set_S1(tmpData.s1);
    tilingData.opInfo.set_S2(tmpData.s2);
    tilingData.opInfo.set_S3(tmpData.s3);
    tilingData.opInfo.set_D(dimDq);
    tilingData.opInfo.set_layout(tmpData.layout);
    tilingData.opInfo.set_scaleValue(
        *context_->GetAttrs()->GetAttrPointer<float>(static_cast<size_t>(AttrIndex::SCALE_VALUE)));
    tilingData.opInfo.set_cmpRatio(cmpRatio);
    tilingData.opInfo.set_oriWinLeft(oriWinLeft);
    tilingData.opInfo.set_oriWinRight(oriWinRight);

    tmpData.d = tilingData.opInfo.get_D();
    tmpData.dataTypeSize = B32;
    tmpData.queryType =
        static_cast<uint32_t>(context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType());

    if (cmpIndicesTensor != nullptr) {
        const gert::Shape &cmpIndicesShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_SPARSE_INDICES))->GetStorageShape();
        tmpData.selected_block_count = cmpIndicesShape.GetDim(dimSize - 1);
        tilingData.opInfo.set_selectedBlockCount(tmpData.selected_block_count);
        tmpData.mode = SMLAG_SCFA_MODE;

        auto cmpSoftmaxL1ShapePtr = context_->GetOutputShape(static_cast<size_t>(OutputIndex::CMP_SOFTMAX_L1_NORM));
        OP_CHECK_IF(cmpSoftmaxL1ShapePtr == nullptr,
                    OP_LOGE("SparseFlashMlaGrad", "In SCFA mode, cmpSoftmaxL1Norm cannot be nullptr."),
                    return ge::GRAPH_FAILED);
        const gert::Shape &cmpSoftmaxL1Shape = cmpSoftmaxL1ShapePtr->GetStorageShape();
        OP_CHECK_IF(cmpSoftmaxL1Shape.GetDimNum() != strlen(inputLayoutQ),
                    OP_LOGE("SparseFlashMlaGrad",
                            "In SCFA mode, cmpSoftmaxL1Norm layout should be same with layout_q and layout_kv."),
                    return ge::GRAPH_FAILED);

        if (strcmp(inputLayoutQ, TND_STR) == 0) {
            OP_CHECK_IF(
                cmpSoftmaxL1Shape.GetDim(DIM_0) != cmpIndicesShape.GetDim(DIM_0) ||
                    cmpSoftmaxL1Shape.GetDim(DIM_1) != cmpIndicesShape.GetDim(DIM_1) ||
                    cmpSoftmaxL1Shape.GetDim(DIM_2) != cmpIndicesShape.GetDim(DIM_2),
                OP_LOGE("SparseFlashMlaGrad",
                        "In SCFA mode, the shape of cmpSoftmaxL1Norm should be the same with cmpSparseIndices."),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                cmpSoftmaxL1Shape.GetDim(DIM_0) != cmpIndicesShape.GetDim(DIM_0) ||
                    cmpSoftmaxL1Shape.GetDim(DIM_1) != cmpIndicesShape.GetDim(DIM_1) ||
                    cmpSoftmaxL1Shape.GetDim(DIM_2) != cmpIndicesShape.GetDim(DIM_2) ||
                    cmpSoftmaxL1Shape.GetDim(DIM_3) != cmpIndicesShape.GetDim(DIM_3),
                OP_LOGE("SparseFlashMlaGrad",
                        "In SCFA mode, the shape of cmpSoftmaxL1Norm should be the same with cmpSparseIndices."),
                return ge::GRAPH_FAILED);
        }
    } else if (cmpKvTensor == nullptr) {
        tmpData.mode = SMLAG_SWA_MODE;
    } else {
        tmpData.mode = SMLAG_CFA_MODE;
    }

    if (tmpData.mode == SMLAG_CFA_MODE || tmpData.mode == SMLAG_SCFA_MODE) {
        OP_CHECK_IF(cmpMaskModePtr == nullptr, OP_LOGE("SparseFlashMlaGrad", "cmpMaskModePtr is null"),
                    return ge::GRAPH_FAILED);
        int64_t cmpMaskMode = *cmpMaskModePtr;
        OP_CHECK_IF(cmpMaskMode != 3,
                    OP_LOGE("SparseFlashMlaGrad", "cmpMaskMode only support 3 now, but got %ld.", cmpMaskMode),
                    return ge::GRAPH_FAILED);
        if (cmpMaskMode == 3) {
            // check cmpResidual tensor shape
            auto cmpResidualKvTensor =
                context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_RESIDUAL_KV));
            OP_CHECK_IF(cmpResidualKvTensor == nullptr,
                        OP_LOGE("SparseFlashMlaGrad", "cmpResidual cannot be null when CFA/SCFA and cmpMaskMode=3"),
                        return ge::GRAPH_FAILED);

            const gert::Shape &cmpResidualKvShape =
                context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_RESIDUAL_KV))->GetStorageShape();
            OP_CHECK_IF(cmpResidualKvShape.GetDim(DIM_0) != tmpData.b,
                        OP_LOGE("SparseFlashMlaGrad", "cmpResidual length should be equal to batchNum."),
                        return ge::GRAPH_FAILED);
        }
    }
    tilingData.opInfo.set_selectedBlockSize(1);

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(SparseFlashMlaGrad, SparseFlashMlaGradBasicTiling,
                                   std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 1);

} // namespace smlag
} // namespace optiling
