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

#include "log/log.h"
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
constexpr uint32_t D_SIZE = 512;
constexpr uint32_t SINGLE_N_256 = 256;
constexpr uint32_t SINGLE_N_128 = 128;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t PING_PONG_BUFFER = 2;
constexpr uint32_t SINGLE_M = 128;

namespace {
ge::graphStatus CheckAttentionInDtype(gert::TilingContext* context)
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

ge::graphStatus CheckDtypeValid(gert::TilingContext* context)
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

ge::graphStatus CheckLseShape(gert::TilingContext* context)
{
    auto lseShape = context->GetInputShape(static_cast<size_t>(InputIndex::LSE));
    if (lseShape == nullptr || lseShape->GetStorageShape().GetDimNum() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "lse", "0",
                                              "The shape of softmax_lse cannot be zero");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus SparseFlashMlaGradBasicTiling::GetShapeAttrsInfo()
{
    /*
    Get all shape info and attr
    */
    opName = context_->GetNodeName();
    OP_CHECK_IF(context_ == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{opName, "context is nullptr."})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetAttrs() == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{opName, "GetAttrs is nullptr."})),
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
        auto compileInfoPtr = reinterpret_cast<const SparseFlashMlaGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr,
                    REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                              (std::vector<const char*>{opName, "compile_info is null."})),
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
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{opName, (std::string() + "num of coreNum(aivNum) is " +
                                                       std::to_string(aicoreParams_.numBlocks) + ", num of aicNum is " +
                                                       std::to_string(aicoreParams_.aicNum) + ".")
                                                          .c_str()})),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(aicoreParams_.ubSize <= 0 || l2CacheSize <= 0,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{opName, "ubSize or l2CacheSize is invalid."})),
                return ge::GRAPH_FAILED);

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
    bool sparseSel = tmpData.mode == SMLAG_SCFA_MODE || tmpData.mode == SMLAG_ORI_SPARSE_MODE;
    tmpData.singleM = sparseSel ? tilingData.opInfo.get_G() : SINGLE_M;
    if (sparseSel && selBlkCntAlign >= SINGLE_N_512) {
        tmpData.singleN = tilingData.opInfo.get_selectedBlockCount() >= 1024 &&
                                  tilingData.opInfo.get_selectedBlockCount() % SINGLE_N_512 == 0 &&
                                  tilingData.opInfo.get_G() <= 64 ?
                              SINGLE_N_256 :
                              SINGLE_N_512;
    } else {
        tmpData.singleN = SINGLE_N_128;
    }
    tmpData.s1BasicSize = sparseSel ? 1 : tmpData.singleM / tilingData.opInfo.get_G();

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
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        opName, (std::string() + "blockdim is 0, aicNum is " + std::to_string(aicoreParams_.aicNum) +
                                 ", aivNum is " + std::to_string(aicoreParams_.numBlocks) + ".")
                                    .c_str()})),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);

    // 系统预留
    int64_t sysLen = WORKSPACE_BASE_CAL;
    int64_t mm12WorkspaceLen = tmpData.singleM * tmpData.singleN * B32;
    mm12WorkspaceLen = AlignData(mm12WorkspaceLen, GM_ALIGN) * PING_PONG_BUFFER;

    int64_t dqWorkspaceLen = tilingData.opInfo.get_dqWorkspaceLen();
    int64_t dkWorkspaceLen = tilingData.opInfo.get_dkWorkspaceLen(); // 包括ori+cmp

    int64_t selectedKWorkspaceLen = 0;
    bool sparseGather = tmpData.mode == SMLAG_SCFA_MODE || tmpData.mode == SMLAG_ORI_SPARSE_MODE;
    if (sparseGather) {
        // Gather/Scatter
        selectedKWorkspaceLen = selectedS2 * tmpData.d * inputDtypeSize;
        selectedKWorkspaceLen = AlignData(selectedKWorkspaceLen, GM_ALIGN);
        selectedKWorkspaceLen *= 4;
    }

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = sysLen;
    workspaces[0] += selectedKWorkspaceLen * currentUseCoreNum;
    workspaces[0] += mm12WorkspaceLen * 4 * currentUseCoreNum;
    workspaces[0] += dqWorkspaceLen + dkWorkspaceLen;

    int64_t dAlign = (tilingData.opInfo.get_D() + 15) / 16 * 16;
    if (sparseGather) {
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
    workspaceOffsets = workspaceOffsets + dkWorkspaceLen;
    if (sparseGather) {
        workspaceOffsets += 24 * PING_PONG_BUFFER * tmpData.selected_block_count * (dAlign + dAlign) * B32;
        workspaceOffsets += tilingData.opInfo.get_additionalWorkspaceLen();
    } else {
        workspaceOffsets += dkWorkspaceLen;
    }

    // Deterministic extra workspaces
    int64_t aivNum = static_cast<int64_t>(aicoreParams_.numBlocks);
    int64_t aicNum = static_cast<int64_t>(aicoreParams_.aicNum);
    int64_t deterKvWorkspaceLen = 0;
    int64_t dSinkWorkspaceLen = 0;
    int64_t cmpSoftmaxL1WorkspaceLen = 0;
    if (tmpData.deterministic) {
        // ori & CFA: aicNum * singleN * (Dk+Dv) * DB * sizeof(fp32)
        // SCFA cmp / ori sparse 复用既有 mm4/mm5 staging，按核保序 scatter
        // 末尾追加 AccMeta（UB DataCopy 广播，供全 AIV 均分 Acc）；ori sparse 不走 Acc
        dSinkWorkspaceLen = aivNum * tilingData.opInfo.get_G() * B32;
        dSinkWorkspaceLen = AlignData(dSinkWorkspaceLen, GM_ALIGN);
        if (tmpData.mode != SMLAG_ORI_SPARSE_MODE) {
            constexpr int64_t kDeterAccMetaBytes = ((24 * 8 * static_cast<int64_t>(sizeof(int64_t)) + 511) / 512) * 512;
            deterKvWorkspaceLen = aicNum * tmpData.singleN * (dAlign + dAlign) * PING_PONG_BUFFER * B32;
            deterKvWorkspaceLen = AlignData(deterKvWorkspaceLen, GM_ALIGN) + kDeterAccMetaBytes;
            if (tmpData.mode == SMLAG_SCFA_MODE) {
                cmpSoftmaxL1WorkspaceLen = tmpData.singleN * aivNum * B32;
                cmpSoftmaxL1WorkspaceLen = AlignData(cmpSoftmaxL1WorkspaceLen, GM_ALIGN);
            }
        }
        workspaces[0] += deterKvWorkspaceLen + dSinkWorkspaceLen + cmpSoftmaxL1WorkspaceLen;
    }
    tilingData.opInfo.set_isDeterministic(tmpData.deterministic ? 1U : 0U);
    tilingData.opInfo.set_deterKvWorkspaceLen(deterKvWorkspaceLen);
    tilingData.opInfo.set_dSinkWorkspaceLen(dSinkWorkspaceLen);
    tilingData.opInfo.set_cmpSoftmaxL1WorkspaceLen(cmpSoftmaxL1WorkspaceLen);
    tilingData.opInfo.set_aivNum(static_cast<uint32_t>(aivNum));

    tilingData.postTilingData.set_deterKvWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets += deterKvWorkspaceLen;
    tilingData.postTilingData.set_dSinkWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets += dSinkWorkspaceLen;
    tilingData.postTilingData.set_cmpSoftmaxL1WorkSpaceOffset(workspaceOffsets);

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
    // CMP_MODE: 0(SWA); 1(CFA); 2(SCFA); 3(ori sparse)
    // Deterministic: 0/1
    uint32_t layout = tmpData.layout == static_cast<uint32_t>(InputLayout::TND) ? 1U : 0U;
    bool hasSeqused = tmpData.usedSeqQ || tmpData.usedSeqOriKV || tmpData.usedSeqCmpKV;
    return GET_TPL_TILING_KEY(layout, tmpData.mode, static_cast<uint8_t>(hasSeqused),
                              static_cast<uint8_t>(tmpData.deterministic));
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

    OP_CHECK_IF(qPostBaseNum == 0,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{opName, "qPostBaseNum is 0."})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{opName, "castUsedCoreNum is 0."})),
                return ge::GRAPH_FAILED);
    int64_t qPostBlockTotal = allNumQuery / dAlign * tilingData.opInfo.get_D();
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t oriKvPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(oriKvPostBaseNum == 0,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{opName, "oriKvPostBaseNum is 0."})),
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

    if (tmpData.mode == SMLAG_CFA_MODE || tmpData.mode == SMLAG_SCFA_MODE) {
        int64_t cmpKvPostBaseNum = qPostBaseNum;
        OP_CHECK_IF(cmpKvPostBaseNum == 0,
                    REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                              (std::vector<const char*>{opName, "cmpKvPostBaseNum is 0."})),
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
    if (tmpData.mode == SMLAG_SCFA_MODE || tmpData.mode == SMLAG_ORI_SPARSE_MODE) {
        tilingData.opInfo.set_additionalWorkspaceLen((allNumOriKv * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashMlaGradBasicTiling::GetBaseShapeInfo()
{
    OP_CHECK_IF(
        ((context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY)) == nullptr) ||
         (context_->GetInputShape(static_cast<size_t>(InputIndex::ORI_KV)) == nullptr)),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "query, ori_kv", "InputShape of query, ori_kv is nullptr."),
        return ge::GRAPH_FAILED);
    // -------------- input ----------------
    // TND: query [t1, n1, d]  ori_kv [t2, n2, d]  cmp_kv [t3, n2, d]   dy/attentionIn [t1, n1, d]
    // BSND: query [b, s1, n1, d]   k [b, s2, n2, d]  v [b, s3, n2, d]   dy/attentionIn [b, s1, n1, d]
    const gert::Shape& queryShape = context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY))->GetStorageShape();
    const gert::Shape& oriKvShape = context_->GetInputShape(static_cast<size_t>(InputIndex::ORI_KV))->GetStorageShape();
    auto cmpKvTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_KV));
    auto cmpIndicesTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_SPARSE_INDICES));

    if (queryShape.GetDimNum() != oriKvShape.GetDimNum()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(),
                                                 "query, d_out, out, lse, ori_kv",
                                                 "The dim num of inputs shape should be the same.");
        return ge::GRAPH_FAILED;
    }
    uint32_t dimSize = queryShape.GetDimNum();
    int64_t dimDq = queryShape.GetDim(dimSize - 1);
    int64_t dimOriKv = oriKvShape.GetDim(dimSize - 1);
    if (dimDq != D_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context_).c_str(), "query", std::to_string(dimDq).c_str(),
            (std::string() + "The headDim of query should be equal to 512, but got dimDq=" + std::to_string(dimDq) +
             ".")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    if (dimOriKv != D_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context_).c_str(), "ori_kv", std::to_string(dimOriKv).c_str(),
            (std::string() +
             "The headDim of ori_kv should be equal to 512, but got dimOriKv=" + std::to_string(dimOriKv) + ".")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    // ori_kv + ori_sparse_indices + ori_topk_length，cmp_kv 为空：MODE 3。
    // 未命中时仍要求这三个可选输入为空，SWA/CFA/SCFA 的后续判定不改。
    auto oriSparseIndicesTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ORI_SPARSE_INDICES));
    auto oriTopkLenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ORI_TOPK_LENGTH));
    auto cmpTopkLenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_TOPK_LENGTH));
    auto metadataTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::METADATA));
    (void)metadataTensor;
    bool hitOriSparse = false;
    if (oriSparseIndicesTensor != nullptr) {
        OP_CHECK_IF(cmpKvTensor != nullptr || cmpIndicesTensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        "SparseFlashMlaGrad", "ori_sparse_indices",
                        "ori_sparse_indices cannot be combined with cmp_kv or cmp_sparse_indices."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            oriTopkLenTensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "ori_topk_length",
                                                     "ori_topk_length is required when ori_sparse_indices exist."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(cmpTopkLenTensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "cmp_topk_length",
                                                             "cmp_topk_length must be nullptr in ori sparse mode."),
                    return ge::GRAPH_FAILED);
        auto cmpResidualKvTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_RESIDUAL_KV));
        OP_CHECK_IF(cmpResidualKvTensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "cmp_residual_kv",
                                                             "cmp_residual_kv must be nullptr in ori sparse mode."),
                    return ge::GRAPH_FAILED);
        hitOriSparse = true;
    } else {
        OP_CHECK_IF(oriTopkLenTensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "ori_topk_length",
                                                             "oriTopkLength should be nullptr now."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(cmpTopkLenTensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "cmp_topk_length",
                                                             "cmpTopkLength should be nullptr now."),
                    return ge::GRAPH_FAILED);
    }

    auto usedSeqQ = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SEQUSED_Q));
    if (usedSeqQ != nullptr) {
        tmpData.usedSeqQ = true;
    }
    auto usedSeqOriKV = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SEQUSED_ORI_KV));
    if (usedSeqOriKV != nullptr) {
        tmpData.usedSeqOriKV = true;
    }
    auto usedSeqCmpKV = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SEQUSED_CMP_KV));
    if (usedSeqCmpKV != nullptr) {
        tmpData.usedSeqCmpKV = true;
    }

    // -------------- attrs ----------------
    const char* inputLayoutQ = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::LAYOUT_Q));
    const char* inputLayoutKv = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::LAYOUT_KV));
    const int64_t* cmpRatioPtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::CMP_RATIO));
    const int64_t* oriWinLeftPtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::ORI_WIN_LEFT));
    const int64_t* oriWinRightPtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::ORI_WIN_RIGHT));
    const int64_t* oriMaskModePtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::ORI_MASK_MODE));
    const int64_t* cmpMaskModePtr =
        context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::CMP_MASK_MODE));

    // -------------------------------------
    tmpData.deterministic = (context_->GetDeterministic() == 1);

    if (dimDq != dimOriKv) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context_).c_str(), "query, ori_kv",
            (std::string() + std::to_string(dimDq) + ", " + std::to_string(dimOriKv)).c_str(),
            (std::string() + "The headDim of query, ori_kv should be the same. But got dimDq=" + std::to_string(dimDq) +
             ", dimOriKv=" + std::to_string(dimOriKv))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t N2 = oriKvShape.GetDim(dimSize - 2);
    if (strcmp(inputLayoutQ, inputLayoutKv) != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            Ops::Base::GetOpInfo(context_).c_str(), "attributes",
            (std::string() + std::string(inputLayoutQ) + ", " + std::string(inputLayoutKv)).c_str(),
            (std::string() + "SparseFlashMlaGrad layout_q and layout_kv should be same, now layout_q is " +
             std::string(inputLayoutQ) + " and layout_kv is " + std::string(inputLayoutKv) + ".")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    if (strcmp(inputLayoutQ, TND_STR) != 0 && strcmp(inputLayoutQ, BSND_STR) != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            Ops::Base::GetOpInfo(context_).c_str(), "attributes", (std::string() + std::string(inputLayoutQ)).c_str(),
            (std::string() + "SparseFlashMlaGrad only support TND or BSND layout, now layout is " +
             std::string(inputLayoutQ) + ".")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    if (static_cast<size_t>(dimSize) != strlen(inputLayoutQ)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context_).c_str(), "query", (std::string() + std::to_string(dimSize)).c_str(),
            (std::string() +
             "SparseFlashMlaGrad layout dims is not equal to the input's dim, now the query of dim is " +
             std::to_string(dimSize) + ".")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    if (strcmp(inputLayoutQ, TND_STR) == 0) {
        const gert::Shape& actSeqQLenShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CU_SEQLENS_Q))->GetStorageShape();
        tmpData.b = actSeqQLenShape.GetDim(DIM_0) - 1;
        tmpData.t1 = queryShape.GetDim(DIM_0);
        tmpData.t2 = oriKvShape.GetDim(DIM_0);
        tmpData.t3 = 0;
        tmpData.s1 = tmpData.t1;
        tmpData.s2 = tmpData.t2;
        tmpData.s3 = 0;
        tmpData.n2 = N2;
        OP_CHECK_IF(tmpData.n2 == 0,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(),
                                                             "query, d_out, out, lse, ori_kv", "N2 is 0"),
                    return ge::GRAPH_FAILED);
        tmpData.g = queryShape.GetDim(DIM_1) / tmpData.n2;
        tmpData.layout = static_cast<uint32_t>(InputLayout::TND);
    } else { // BSND
        tmpData.b = queryShape.GetDim(DIM_0);
        tmpData.s1 = queryShape.GetDim(DIM_1);
        tmpData.s2 = oriKvShape.GetDim(DIM_1);
        tmpData.s3 = 0;
        tmpData.n2 = N2;
        OP_CHECK_IF(tmpData.n2 == 0,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(),
                                                             "query, d_out, out, lse, ori_kv", "N2 is 0"),
                    return ge::GRAPH_FAILED);
        tmpData.g = queryShape.GetDim(DIM_2) / tmpData.n2;
        tmpData.layout = static_cast<uint32_t>(InputLayout::BSND);
        // BSND 下 kernel 用 S1 做 t1Idx / curS1 分核，S1 须严格为正
        OP_CHECK_IF(
            tmpData.s1 <= 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "query", std::to_string(tmpData.s1).c_str(),
                                                  "S1 must be greater than 0 in BSND layout"),
            return ge::GRAPH_FAILED);
    }

    if (cmpKvTensor != nullptr) {
        const gert::Shape& cmpKvShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_KV))->GetStorageShape();
        int64_t dimCmpKv = cmpKvShape.GetDim(dimSize - 1);
        if (dimCmpKv != D_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                Ops::Base::GetOpInfo(context_).c_str(), "cmp_kv", std::to_string(dimCmpKv).c_str(),
                (std::string() +
                 "The headDim of cmp_kv should be equal to 512, but got dimCmpKv=" + std::to_string(dimCmpKv) + ".")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        if (dimDq != dimCmpKv) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                Ops::Base::GetOpInfo(context_).c_str(), "query, cmp_kv",
                (std::string() + std::to_string(dimDq) + ", " + std::to_string(dimCmpKv)).c_str(),
                (std::string() + "The headDim of query, cmp_kv should be the same. But got dimDq=" +
                 std::to_string(dimDq) + ", dimCmpKv=" + std::to_string(dimCmpKv))
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        if (strcmp(inputLayoutQ, TND_STR) == 0) {
            tmpData.t3 = cmpKvShape.GetDim(DIM_0);
            tmpData.s3 = tmpData.t3;
        } else {
            tmpData.s3 = cmpKvShape.GetDim(DIM_1);
        }
    }

    if (CheckLseShape(context_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(cmpRatioPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "cmp_ratio", "cmpRatioPtr is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriWinLeftPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "ori_win_left", "oriWinLeftPtr is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        oriWinRightPtr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "ori_win_right", "oriWinRightPtr is null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        cmpMaskModePtr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "cmp_mask_mode", "cmpMaskModePtr is null"),
        return ge::GRAPH_FAILED);

    if (tmpData.g <= 0) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context_).c_str(), "query, d_out, out, lse, ori_kv",
            (std::string() + std::to_string(tmpData.g)).c_str(),
            (std::string() + "g (N1 / N2) should be larger than 0, but got g=" + std::to_string(tmpData.g) + ".")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t n1 = tmpData.n2 * tmpData.g;
    if (tmpData.n2 != 1 || n1 > 128) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context_).c_str(), "query, d_out, out, lse, ori_kv",
            (std::string() + std::to_string(tmpData.n2) + ", " + std::to_string(n1)).c_str(),
            (std::string() + "SparseFlashMlaGrad only support n2=1 and 1<=n1<=128, but got n2=" +
             std::to_string(tmpData.n2) + " n1=" + std::to_string(n1) + ".")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t cmpRatio = *cmpRatioPtr;
    if (!(cmpRatio >= 1 && cmpRatio <= 128)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            Ops::Base::GetOpInfo(context_).c_str(), "attributes", (std::string() + std::to_string(cmpRatio)).c_str(),
            (std::string() + "SparseFlashMlaGrad only support cmpRatio >= 1 and <=128, but got cmpRatio=" +
             std::to_string(cmpRatio) + ".")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t oriWinLeft = *oriWinLeftPtr;
    int64_t oriWinRight = *oriWinRightPtr;
    if (!hitOriSparse) {
        OP_CHECK_IF(oriWinLeft != 127 || oriWinRight != 0,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        "SparseFlashMlaGrad", "attributes",
                        "SparseFlashMlaGrad only support oriWinLeft=127 and oriWinRight=0 now."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(*oriMaskModePtr != 4,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "SparseFlashMlaGrad", "attributes", (std::string() + std::to_string(*oriMaskModePtr)).c_str(),
                        (std::string() + "SparseFlashMlaGrad only support oriMaskMode=4 now, but got " +
                         std::to_string(*oriMaskModePtr) + ".")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(oriMaskModePtr == nullptr || *oriMaskModePtr != 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "SparseFlashMlaGrad", "attributes",
                        oriMaskModePtr == nullptr ? "null" : (std::string() + std::to_string(*oriMaskModePtr)).c_str(),
                        (std::string() + "ori sparse mode only support oriMaskMode=0, but got " +
                         (oriMaskModePtr == nullptr ? std::string("null") : std::to_string(*oriMaskModePtr)) + ".")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }

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

    if (hitOriSparse) {
        const gert::Shape& oriIndicesShape = oriSparseIndicesTensor->GetStorageShape();
        const gert::Shape& oriTopkShape = oriTopkLenTensor->GetStorageShape();
        int64_t selectedCnt = oriIndicesShape.GetDim(dimSize - 1);
        OP_CHECK_IF(selectedCnt <= 0,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "ori_sparse_indices",
                                                             "ori_sparse_indices last dim K must be > 0."),
                    return ge::GRAPH_FAILED);
        if (strcmp(inputLayoutQ, TND_STR) == 0) {
            OP_CHECK_IF(oriIndicesShape.GetDimNum() != 3 || oriIndicesShape.GetDim(DIM_0) != tmpData.t1 ||
                            oriIndicesShape.GetDim(DIM_1) != tmpData.n2,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "ori_sparse_indices",
                                                                 "TND ori_sparse_indices shape must be [T1, N2, K]."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(oriTopkShape.GetDimNum() != 2 || oriTopkShape.GetDim(DIM_0) != tmpData.t1 ||
                            oriTopkShape.GetDim(DIM_1) != tmpData.n2,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "ori_topk_length",
                                                                 "TND ori_topk_length shape must be [T1, N2]."),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                oriIndicesShape.GetDimNum() != 4 || oriIndicesShape.GetDim(DIM_0) != tmpData.b ||
                    oriIndicesShape.GetDim(DIM_1) != tmpData.s1 || oriIndicesShape.GetDim(DIM_2) != tmpData.n2,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "ori_sparse_indices",
                                                         "BSND ori_sparse_indices shape must be [B, S1, N2, K]."),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(oriTopkShape.GetDimNum() != 3 || oriTopkShape.GetDim(DIM_0) != tmpData.b ||
                            oriTopkShape.GetDim(DIM_1) != tmpData.s1 || oriTopkShape.GetDim(DIM_2) != tmpData.n2,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "ori_topk_length",
                                                                 "BSND ori_topk_length shape must be [B, S1, N2]."),
                        return ge::GRAPH_FAILED);
        }
        auto dOriKvShapePtr = context_->GetOutputShape(static_cast<size_t>(OutputIndex::DORI_KV));
        OP_CHECK_IF(dOriKvShapePtr == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "d_ori_kv",
                                                             "d_ori_kv is required in ori sparse mode."),
                    return ge::GRAPH_FAILED);
        const gert::Shape& dOriKvShape = dOriKvShapePtr->GetStorageShape();
        OP_CHECK_IF(dOriKvShape.GetDimNum() != oriKvShape.GetDimNum(),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "d_ori_kv",
                                                             "d_ori_kv rank must match ori_kv."),
                    return ge::GRAPH_FAILED);
        for (size_t dimIdx = 0; dimIdx < oriKvShape.GetDimNum(); ++dimIdx) {
            OP_CHECK_IF(dOriKvShape.GetDim(dimIdx) != oriKvShape.GetDim(dimIdx),
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "d_ori_kv",
                                                                 "d_ori_kv shape must match ori_kv."),
                        return ge::GRAPH_FAILED);
        }
        tmpData.selected_block_count = static_cast<uint32_t>(selectedCnt);
        tilingData.opInfo.set_selectedBlockCount(tmpData.selected_block_count);
        tmpData.mode = SMLAG_ORI_SPARSE_MODE;
    } else if (cmpIndicesTensor != nullptr) {
        const gert::Shape& cmpIndicesShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_SPARSE_INDICES))->GetStorageShape();
        tmpData.selected_block_count = cmpIndicesShape.GetDim(dimSize - 1);
        tilingData.opInfo.set_selectedBlockCount(tmpData.selected_block_count);
        tmpData.mode = SMLAG_SCFA_MODE;

        auto cmpSoftmaxL1ShapePtr = context_->GetOutputShape(static_cast<size_t>(OutputIndex::CMP_SOFTMAX_L1_NORM));
        OP_CHECK_IF(cmpSoftmaxL1ShapePtr == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "cmp_softmax_l1_norm",
                                                             "In SCFA mode, cmpSoftmaxL1Norm cannot be nullptr."),
                    return ge::GRAPH_FAILED);
        const gert::Shape& cmpSoftmaxL1Shape = cmpSoftmaxL1ShapePtr->GetStorageShape();
        OP_CHECK_IF(cmpSoftmaxL1Shape.GetDimNum() != strlen(inputLayoutQ),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        "SparseFlashMlaGrad", "cmp_softmax_l1_norm",
                        "In SCFA mode, cmpSoftmaxL1Norm layout should be same with layout_q and layout_kv."),
                    return ge::GRAPH_FAILED);

        if (strcmp(inputLayoutQ, TND_STR) == 0) {
            OP_CHECK_IF(cmpSoftmaxL1Shape.GetDim(DIM_0) != cmpIndicesShape.GetDim(DIM_0) ||
                            cmpSoftmaxL1Shape.GetDim(DIM_1) != cmpIndicesShape.GetDim(DIM_1) ||
                            cmpSoftmaxL1Shape.GetDim(DIM_2) != cmpIndicesShape.GetDim(DIM_2),
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            "SparseFlashMlaGrad", "cmp_softmax_l1_norm",
                            "In SCFA mode, the shape of cmpSoftmaxL1Norm should be the same with cmpSparseIndices."),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(cmpSoftmaxL1Shape.GetDim(DIM_0) != cmpIndicesShape.GetDim(DIM_0) ||
                            cmpSoftmaxL1Shape.GetDim(DIM_1) != cmpIndicesShape.GetDim(DIM_1) ||
                            cmpSoftmaxL1Shape.GetDim(DIM_2) != cmpIndicesShape.GetDim(DIM_2) ||
                            cmpSoftmaxL1Shape.GetDim(DIM_3) != cmpIndicesShape.GetDim(DIM_3),
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            "SparseFlashMlaGrad", "cmp_softmax_l1_norm",
                            "In SCFA mode, the shape of cmpSoftmaxL1Norm should be the same with cmpSparseIndices."),
                        return ge::GRAPH_FAILED);
        }
    } else if (cmpKvTensor == nullptr) {
        tmpData.mode = SMLAG_SWA_MODE;
    } else {
        tmpData.mode = SMLAG_CFA_MODE;
    }

    if (tmpData.mode == SMLAG_CFA_MODE || tmpData.mode == SMLAG_SCFA_MODE) {
        OP_CHECK_IF(
            cmpMaskModePtr == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "cmp_mask_mode", "cmpMaskModePtr is null"),
            return ge::GRAPH_FAILED);
        int64_t cmpMaskMode = *cmpMaskModePtr;
        OP_CHECK_IF(cmpMaskMode != 3,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "SparseFlashMlaGrad", "attributes", (std::string() + std::to_string(cmpMaskMode)).c_str(),
                        (std::string() + "cmpMaskMode only support 3 now, but got " + std::to_string(cmpMaskMode) + ".")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        if (cmpMaskMode == 3) {
            // check cmpResidual tensor shape
            auto cmpResidualKvTensor =
                context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_RESIDUAL_KV));
            OP_CHECK_IF(
                cmpResidualKvTensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "cmp_residual_kv",
                                                         "cmpResidual cannot be null when CFA/SCFA and cmpMaskMode=3"),
                return ge::GRAPH_FAILED);

            const gert::Shape& cmpResidualKvShape =
                context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::CMP_RESIDUAL_KV))->GetStorageShape();
            OP_CHECK_IF(cmpResidualKvShape.GetDim(DIM_0) != tmpData.b,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("SparseFlashMlaGrad", "cmp_residual_kv",
                                                                 "cmpResidual length should be equal to batchNum."),
                        return ge::GRAPH_FAILED);
        }
    }

    // torch 包装层在 sinks 缺省时分配 shape [0] 的 d_sinks，与 nullptr 一样视为未提供。
    auto sinksTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::SINKS));
    const gert::StorageShape* dSinksStorage = context_->GetOutputShape(static_cast<size_t>(OutputIndex::DSINKS));
    auto isZeroElement = [](const gert::Shape& shape) { return shape.GetDimNum() == 1 && shape.GetDim(0) == 0; };
    bool hasSinks = sinksTensor != nullptr && !isZeroElement(sinksTensor->GetStorageShape());
    bool hasDSinks = dSinksStorage != nullptr && !isZeroElement(dSinksStorage->GetStorageShape());
    if (hasSinks != hasDSinks) {
        OP_LOGE(context_,
                "sinks and d_sinks must both be provided or both be omitted, but got hasSinks=%d, hasDSinks=%d.",
                static_cast<int>(hasSinks), static_cast<int>(hasDSinks));
        return ge::GRAPH_FAILED;
    }
    tmpData.sinks = hasSinks;

    tilingData.opInfo.set_hasUsedSeqQ(tmpData.usedSeqQ);
    tilingData.opInfo.set_hasUsedSeqOriKV(tmpData.usedSeqOriKV);
    tilingData.opInfo.set_hasUsedSeqCmpKV(tmpData.usedSeqCmpKV);
    tilingData.opInfo.set_isSink(static_cast<uint8_t>(tmpData.sinks));
    tilingData.opInfo.set_selectedBlockSize(1);

    auto ret = CheckDtypeValid(context_);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SparseFlashMlaGrad the dtype of input is invalid.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(SparseFlashMlaGrad, SparseFlashMlaGradBasicTiling,
                                   std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 1);

} // namespace smlag
} // namespace optiling
