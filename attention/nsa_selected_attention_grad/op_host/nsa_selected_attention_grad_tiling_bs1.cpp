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
 * \file nsa_selected_attention_grad_bs1.h
 * \brief
 */

#include "nsa_selected_attention_grad_tiling_bs1.h"
#include "op_host/tiling_templates_registry.h"
#include "err/ops_err.h"
using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace nsa {
constexpr uint32_t FLOAT_PRECISION_SIZE = 4;
constexpr uint32_t WORKSPACE_BASE_CAL = 32 * 1024 * 1024; // 100MB系统预留
constexpr uint32_t BLOCK = 32;                            // 32B
constexpr uint32_t B32 = 4;                               // 4B
constexpr uint32_t B16 = 2;
constexpr uint32_t BASE_LEN_256 = 256;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t PING_PONG_BUFFER = 2;
constexpr uint32_t SELECTED_BLOCK_COUNT_MAX = 128;
constexpr uint32_t SELECTED_BLOCK_COUNT_MIN = 1;
constexpr uint32_t KEY_DETERMINISTIC = 2;

bool NsaSelectedAttentionGradTiling::IsCapable()
{
    return true;
}

uint64_t NsaSelectedAttentionGradTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (tmpData.attenEnable) {
        tilingKey += 1;
    }
    tilingKey += tmpData.isDeterministic * KEY_DETERMINISTIC;
    OP_LOGI(context_,
            "NsaSelectedAttentionGrad DoTiling success, tilingkey is"
            " %lu.",
            tilingKey);
    return tilingKey;
}

ge::graphStatus NsaSelectedAttentionGradTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    uint64_t l2CacheSize;
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<NsaSelectedAttentionGradCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "compile_info is null."),
                    return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->aivNum;
        aicoreParams_.aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0aSize = compileInfoPtr->l0aSize;
        aicoreParams_.l0bSize = compileInfoPtr->l0bSize;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize =
            compileInfoPtr->l2CacheSize; // AiCoreParams使用的是cann仓的结构体，l2CacheSize暂时定义成类成员变量
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

    OP_CHECK_IF(
        (aicoreParams_.numBlocks == 0) || (aicoreParams_.aicNum == 0),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "num of coreNum(aivNum) is %lu, num of aicNum is %lu.",
                                    aicoreParams_.numBlocks, aicoreParams_.aicNum),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(aicoreParams_.ubSize <= 0 || l2CacheSize <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "ubSize or l2CacheSize is invalid."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::GetShapeAttrsInfo()
{
    /*
    Get all shape info and attr
    */
    OP_CHECK_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "context is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetAttrs() == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetAttrs is nullptr."), return ge::GRAPH_FAILED);

    auto status = GetLayoutInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = GetBaseShapeInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    auto selected_block_count =
        *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_COUNT));
    auto selected_block_size =
        *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_SIZE));
    auto sparse_mode = *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SPARSE_MODE));
    if (sparse_mode == 0) {
        tmpData.attenEnable = false;
    } else if (sparse_mode == 2) {
        OP_LOGI(context_, "NsaSelectedAttentionGrad AttenMask enable.");
        tmpData.attenEnable = true;
    } else {
        OP_LOGE(context_, "NsaSelectedAttentionGrad only support sparse_mode=2, now sparse_mode=%d.", sparse_mode);
        return ge::GRAPH_FAILED;
    }

    if (selected_block_count > static_cast<int>(SELECTED_BLOCK_COUNT_MAX) ||
        selected_block_count < static_cast<int>(SELECTED_BLOCK_COUNT_MIN)) {
        OP_LOGE(context_,
                "NsaSelectedAttentionGrad only support selected_block_count [1,128], now "
                "selected_block_count=%d.",
                selected_block_count);
        return ge::GRAPH_FAILED;
    }

    /*
     * Due to the limitation of UB space, the selected_block_size must be less than 128 and larger than 16.
     * If you need to support larger sizes, you should split and modify the scatter process in kernel code.
     */
    if (selected_block_size > 128 || selected_block_size < 16 || selected_block_size % 16 != 0) {
        OP_LOGE(context_,
                "NsaSelectedAttentionGrad only support selected_block_size [16,128] and must align to 16, now "
                "selected_block_size=%d.",
                selected_block_size);
        return ge::GRAPH_FAILED;
    }

    tmpData.selected_block_count = selected_block_count;
    tmpData.selected_block_size = selected_block_size;
    tilingData.opInfo.set_scaleValue(
        *context_->GetAttrs()->GetAttrPointer<float>(static_cast<size_t>(AttrIndex::SCALE_VALUE)));
    tilingData.opInfo.set_selectedBlockCount(selected_block_count);
    tilingData.opInfo.set_selectedBlockSize(selected_block_size);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::DoBlockTiling()
{
    int64_t usedCoreNumForProcess = aicoreParams_.numBlocks - tmpData.isDeterministic;
    tilingData.opInfo.set_usedCoreNum(usedCoreNumForProcess);
    tilingData.opInfo.set_castUsedCoreNum(aicoreParams_.numBlocks);
    int64_t nNums = tmpData.t1;
    int64_t formerCoreProcessNNums = CeilCommon(nNums, usedCoreNumForProcess);
    int64_t remainCoreNum = formerCoreProcessNNums * usedCoreNumForProcess - nNums;

    tilingData.opInfo.set_formerCoreNum(usedCoreNumForProcess - remainCoreNum);
    tilingData.opInfo.set_formerCoreProcessNNum(formerCoreProcessNNums);
    tilingData.opInfo.set_remainCoreProcessNNum(static_cast<uint32_t>(formerCoreProcessNNums - 1));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::DoSftTiling()
{
    constexpr int32_t maxProcessDataSize = 8 * 1024;

    uint32_t sftBaseN = singleN;
    uint32_t sftBaseM = maxProcessDataSize / sftBaseN;

    tilingData.splitCoreParams.set_sftBaseM(sftBaseM);
    tilingData.splitCoreParams.set_sftBaseN(sftBaseN);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::DoOpTiling()
{
    OP_LOGI(context_, "NsaSelectedAttentionGrad DoTiling start");

    // Init
    singleM = tilingData.opInfo.get_G();
    singleN = tmpData.selected_block_count * tmpData.selected_block_size;

    // setTilingData
    tilingData.splitCoreParams.set_singleM(singleM);
    tilingData.splitCoreParams.set_singleN(singleN);
    tilingData.splitCoreParams.set_s2OuterOuter(1);

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

void NsaSelectedAttentionGradTiling::PrintShapeInfo()
{
    OP_LOGI(context_, "NsaSelectedAttentionGrad with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] d2[%ld]!",
            tilingData.opInfo.get_B(), tilingData.opInfo.get_N2(), tilingData.opInfo.get_G(),
            tilingData.opInfo.get_S1(), tilingData.opInfo.get_S2(), tilingData.opInfo.get_D(),
            tilingData.opInfo.get_D2());
}

ge::graphStatus NsaSelectedAttentionGradTiling::DoLibApiTiling()
{
    // calc for simpleSoftMax which dstShape is as same as srcShape
    auto simpleSoftMaxShape = ge::Shape({singleM, singleN});
    auto helpLenA = singleM * singleN * tmpData.dataTypeSize; // UB内数据类型
    AscendC::SoftMaxTilingFunc(simpleSoftMaxShape, sizeof(float), helpLenA, tilingData.softmaxTilingData);

    // calc for softmaxGrad
    auto softmaxGradShape = ge::Shape({singleM, BLOCK / tmpData.dataTypeSize});
    auto helpLenB = 2 * singleM * singleN * tmpData.dataTypeSize; // UB内数据类型 64KB
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, tmpData.dataTypeSize, helpLenB, tilingData.softmaxGradTilingData,
                                   true);

    matmul_tiling::DataType inputType = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType outputType = matmul_tiling::DataType::DT_FLOAT;
    auto valueType =
        static_cast<uint32_t>(context_->GetInputDesc(static_cast<size_t>(InputIndex::VALUE))->GetDataType());
    if (valueType == ge::DT_BF16) {
        inputType = matmul_tiling::DataType::DT_BF16;
    }
    // mm1: attentionGrad @ V^T
    auto status = Setmm1TilingData(inputType, outputType);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    // mm2: Q @ K^T --> p
    status = Setmm2TilingData(inputType, outputType);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    // mm3: softmaxGradRes @ K --> dq
    status = Setmm3TilingData(inputType, outputType);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    // mm4: softmaxRes^T @ attentionGrad --> dv
    status = Setmm4TilingData(inputType, outputType);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    // mm5: softmaxGradRes^T @ Q --> dk
    status = Setmm5TilingData(inputType, outputType);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::DoCastTiling()
{
    int64_t dAlign = (tilingData.opInfo.get_D() + 15) / 16 * 16;
    int64_t d2Align = (tilingData.opInfo.get_D2() + 15) / 16 * 16;
    // query
    int64_t allNumQuery = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_G() *
                          tilingData.opInfo.get_S1() * dAlign;
    // TND时候要按照真实的query的num数计算
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumQuery = tmpData.t1 * tilingData.opInfo.get_N2() * tilingData.opInfo.get_G() * tilingData.opInfo.get_D();
    }

    // key
    int64_t allNumKey = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_S2() * dAlign;
    // TND时候要按照真实的key的num数计算
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumKey = tmpData.t2 * tilingData.opInfo.get_N2() * 1 * tilingData.opInfo.get_D();
    }

    // Value
    int64_t allNumValue = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_S2() * d2Align;
    // TND时候要按照真实的value的num数计算
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumValue = tmpData.t2 * tilingData.opInfo.get_N2() * 1 * tilingData.opInfo.get_D2();
    }

    uint32_t typeSize = tmpData.queryType == ge::DT_FLOAT ? B32 : B16;
    uint32_t usedCoreNum = tilingData.opInfo.get_castUsedCoreNum();
    constexpr uint32_t postNzCoexNode = 10;
    constexpr uint32_t blockSize = 32;
    constexpr uint32_t postNzReservedN = 1;

    uint32_t postUbBaseSize = 0;
    uint32_t qPostBaseNum = 0;
    int64_t nzReservedSize = 0;
    int64_t curPostCoexNode = postNzCoexNode;
    nzReservedSize = dAlign / 16 * blockSize * postNzReservedN; // 16为一个单元长度
    postUbBaseSize = (aicoreParams_.ubSize - 2 * nzReservedSize) / curPostCoexNode /
                     BUFFER_NUM / // 开DB预留2份nzReservedSize
                     BASE_LEN_256 * BASE_LEN_256;
    qPostBaseNum = postUbBaseSize / typeSize / dAlign * tilingData.opInfo.get_D();

    OP_CHECK_IF(qPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "qPostBaseNum is 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "castUsedCoreNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t qPostBlockTotal = allNumQuery / dAlign * tilingData.opInfo.get_D();
    int64_t qSizeAlign = (qPostBlockTotal + BASE_LEN_256 - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t kPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(kPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "kPostBaseNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t kPostBlockTotal = allNumKey / dAlign * tilingData.opInfo.get_D();
    int64_t kSizeAlign = (kPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t kPostTailNumTmp = kPostBlockTotal % kPostBaseNum;
    int64_t kPostTailNum = kPostTailNumTmp == 0 ? kPostBaseNum : kPostTailNumTmp;
    int64_t kPostBlockOuterTotal = (kPostBlockTotal + kPostBaseNum - 1) / kPostBaseNum;
    int64_t kPostBlockFactor = (kPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t vPostBaseNum = postUbBaseSize / typeSize / d2Align * tilingData.opInfo.get_D2();
    OP_CHECK_IF(vPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "vPostBaseNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t vPostBlockTotal = allNumValue / d2Align * tilingData.opInfo.get_D2();
    int64_t vSizeAlign = (vPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t vPostTailNumTmp = vPostBlockTotal % vPostBaseNum;
    int64_t vPostTailNum = vPostTailNumTmp == 0 ? vPostBaseNum : vPostTailNumTmp;
    int64_t vPostBlockOuterTotal = (vPostBlockTotal + vPostBaseNum - 1) / vPostBaseNum;
    int64_t vPostBlockFactor = (vPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    tilingData.postTilingData.set_coreNum(usedCoreNum);
    tilingData.postTilingData.set_scaleValue(tilingData.opInfo.get_scaleValue());
    tilingData.postTilingData.set_postUbBaseSize(postUbBaseSize);
    tilingData.postTilingData.set_nzReservedSize(nzReservedSize);

    tilingData.postTilingData.set_qPostBlockFactor(qPostBlockFactor);
    tilingData.postTilingData.set_qPostBlockTotal(qPostBlockTotal);
    tilingData.postTilingData.set_qPostBaseNum(qPostBaseNum);
    tilingData.postTilingData.set_qPostTailNum(qPostTailNum);
    tilingData.postTilingData.set_qSizeAlign(qSizeAlign);

    tilingData.postTilingData.set_kPostBlockFactor(kPostBlockFactor);
    tilingData.postTilingData.set_kPostBlockTotal(kPostBlockTotal);
    tilingData.postTilingData.set_kPostBaseNum(kPostBaseNum);
    tilingData.postTilingData.set_kPostTailNum(kPostTailNum);
    tilingData.postTilingData.set_kSizeAlign(kSizeAlign);

    tilingData.postTilingData.set_vPostBlockFactor(vPostBlockFactor);
    tilingData.postTilingData.set_vPostBlockTotal(vPostBlockTotal);
    tilingData.postTilingData.set_vPostBaseNum(vPostBaseNum);
    tilingData.postTilingData.set_vPostTailNum(vPostTailNum);
    tilingData.postTilingData.set_vSizeAlign(vSizeAlign);

    tilingData.opInfo.set_dqWorkspaceLen((allNumQuery * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    tilingData.opInfo.set_dkWorkspaceLen((allNumKey * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    tilingData.opInfo.set_dvWorkspaceLen((allNumValue * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);

    tilingData.postTilingData.set_b(tilingData.opInfo.get_B());
    tilingData.postTilingData.set_n2(tilingData.opInfo.get_N2());
    tilingData.postTilingData.set_g(tilingData.opInfo.get_G());
    tilingData.postTilingData.set_s1(tilingData.opInfo.get_S1());
    tilingData.postTilingData.set_s2(tilingData.opInfo.get_S2());
    tilingData.postTilingData.set_d(tilingData.opInfo.get_D());
    tilingData.postTilingData.set_d2(tilingData.opInfo.get_D2());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::GetWorkspaceSize()
{
    int64_t currentUseCoreNum = tilingData.opInfo.get_usedCoreNum();
    int64_t launchBlockDims = aicoreParams_.numBlocks;
    int64_t inputDtypeSize = tmpData.queryType == ge::DT_FLOAT ? B32 : B16;
    int64_t selectedS2 = tmpData.selected_block_count * tmpData.selected_block_size;

    // Tiling传递的内存大小、起始地址，统一为字节数，单位为B
    auto blockdim = CalcTschBlockDim(launchBlockDims, aicoreParams_.aicNum, aicoreParams_.numBlocks);
    OP_CHECK_IF(blockdim == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "blockdim is 0, aicNum is %lu, aivNum is %lu.",
                                            aicoreParams_.aicNum, aicoreParams_.numBlocks),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);
    context_->SetScheduleMode(1);

    // 系统预留
    int64_t sysLen = WORKSPACE_BASE_CAL;
    // mm
    int64_t mm1WorkspaceLen = singleM * singleN * B32;
    int64_t mm2WorkspaceLen = singleM * singleN * B32;
    int64_t mm3InputWorkspaceLen = singleM * singleN * inputDtypeSize;
    int64_t mm4InputWorkspaceLen = singleM * singleN * inputDtypeSize;
    mm1WorkspaceLen = AlignData(mm1WorkspaceLen, GM_ALIGN);
    mm2WorkspaceLen = AlignData(mm2WorkspaceLen, GM_ALIGN);
    mm3InputWorkspaceLen = AlignData(mm3InputWorkspaceLen, GM_ALIGN);
    mm4InputWorkspaceLen = AlignData(mm4InputWorkspaceLen, GM_ALIGN);

    mm1WorkspaceLen *= PING_PONG_BUFFER;
    mm2WorkspaceLen *= PING_PONG_BUFFER;
    mm3InputWorkspaceLen *= PING_PONG_BUFFER;
    mm4InputWorkspaceLen *= PING_PONG_BUFFER;

    // Gather/Scatter
    int64_t selectedKWorkspaceLen = selectedS2 * tmpData.d * inputDtypeSize;
    int64_t selectedVWorkspaceLen = selectedS2 * tmpData.d2 * inputDtypeSize;
    int64_t scatterDkWorkspaceLen = selectedS2 * tmpData.d * B32;
    int64_t scatterDvWorkspaceLen = selectedS2 * tmpData.d2 * B32;
    selectedKWorkspaceLen = AlignData(selectedKWorkspaceLen, GM_ALIGN);
    selectedVWorkspaceLen = AlignData(selectedVWorkspaceLen, GM_ALIGN);
    scatterDkWorkspaceLen = AlignData(scatterDkWorkspaceLen, GM_ALIGN);
    scatterDvWorkspaceLen = AlignData(scatterDvWorkspaceLen, GM_ALIGN);
    /**
     * Since the select operation will copy the key data, but when we calculate dq, we also need the key data.
     * However, due to the optimization of scatter process, it will conflict with the select process.
     * Therefore, we need 3 buffers to store the key data.
     */
    selectedKWorkspaceLen *= 3;
    selectedVWorkspaceLen *= PING_PONG_BUFFER;
    scatterDkWorkspaceLen *= PING_PONG_BUFFER;
    scatterDvWorkspaceLen *= PING_PONG_BUFFER;

    int64_t dqWorkspaceLen = tilingData.opInfo.get_dqWorkspaceLen();
    int64_t dkWorkspaceLen = tilingData.opInfo.get_dkWorkspaceLen();
    int64_t dvWorkspaceLen = tilingData.opInfo.get_dvWorkspaceLen();
    // set global workspace
    // 内存顺序排布
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = sysLen;
    workspaces[0] += (selectedKWorkspaceLen + selectedVWorkspaceLen) * currentUseCoreNum;
    workspaces[0] += (mm1WorkspaceLen + mm2WorkspaceLen) * currentUseCoreNum;
    workspaces[0] += (mm3InputWorkspaceLen + mm4InputWorkspaceLen) * currentUseCoreNum;
    workspaces[0] += (scatterDkWorkspaceLen + scatterDvWorkspaceLen) * currentUseCoreNum;
    workspaces[0] += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    tilingData.opInfo.set_mm1WorkspaceLen(mm1WorkspaceLen);
    tilingData.opInfo.set_mm2WorkspaceLen(mm2WorkspaceLen);
    tilingData.opInfo.set_mm3InputWorkspaceLen(mm3InputWorkspaceLen);
    tilingData.opInfo.set_mm4InputWorkspaceLen(mm4InputWorkspaceLen);
    tilingData.opInfo.set_selectedKWorkspaceLen(selectedKWorkspaceLen);
    tilingData.opInfo.set_selectedVWorkspaceLen(selectedVWorkspaceLen);
    tilingData.opInfo.set_scatterDkWorkspaceLen(scatterDkWorkspaceLen);
    tilingData.opInfo.set_scatterDvWorkspaceLen(scatterDvWorkspaceLen);

    int64_t workspaceOffsets = (selectedKWorkspaceLen + selectedVWorkspaceLen) * currentUseCoreNum;
    workspaceOffsets += (mm1WorkspaceLen + mm2WorkspaceLen) * currentUseCoreNum;
    workspaceOffsets += (mm3InputWorkspaceLen + mm4InputWorkspaceLen) * currentUseCoreNum;
    workspaceOffsets += (scatterDkWorkspaceLen + scatterDvWorkspaceLen) * currentUseCoreNum;
    tilingData.postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = workspaceOffsets + tilingData.opInfo.get_dqWorkspaceLen();
    tilingData.postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = workspaceOffsets + tilingData.opInfo.get_dkWorkspaceLen();
    tilingData.postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::PostTiling()
{
    // 判断如果GetDataSize > GetCapacity的异常情况,其中确定性计算下终止流程，非确定性计算流入下一个模板判断
    OP_CHECK_IF(
        (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) && (context_->GetDeterministic() == 1),
        OP_LOGE(context_, "The size of TilingDataSize[%zu] is larger than the size of MaxDataCapacity[%zu].",
                tilingData.GetDataSize(), context_->GetRawTilingData()->GetCapacity()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) && (context_->GetDeterministic() != 1),
        OP_LOGE(context_, "The size of TilingDataSize[%zu] is larger than the size of MaxDataCapacity[%zu].",
                tilingData.GetDataSize(), context_->GetRawTilingData()->GetCapacity()),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::GetLayoutInfo()
{
    const char* inputLayout = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    if (strcmp(inputLayout, TND_STR) != 0) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad only support TND layout, now layout is %s.", inputLayout);
        return ge::GRAPH_FAILED;
    }

    tilingData.opInfo.set_layout(static_cast<uint32_t>(InputLayout::TND));
    tmpData.layout = tilingData.opInfo.get_layout();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::SetBaseInfo(const gert::Shape& queryShape, const gert::Shape& keyShape,
                                                            const gert::Shape& valueShape, int64_t dimN1)
{
    OP_CHECK_IF(dimN1 == 0, OP_LOGW(context_, "headNum is 0."), return ge::GRAPH_FAILED);

    uint32_t dimSize = queryShape.GetDimNum();
    const char* inputLayout = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    OP_CHECK_IF(static_cast<size_t>(dimSize) != strlen(inputLayout),
                OP_LOGW(context_, "layout dims is not inputLayout's length."), return ge::GRAPH_FAILED);

    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        auto actualSeqQlenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_Q_LEN));
        auto actualSeqKvlenTensor =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
        OP_CHECK_IF((actualSeqQlenTensor == nullptr || actualSeqKvlenTensor == nullptr),
                    OP_LOGW(context_, "actualSeqQlenTensor or actualSeqKvlenTensor is nullptr."),
                    return ge::GRAPH_FAILED);

        const size_t seqQShapeSize = static_cast<size_t>(actualSeqQlenTensor->GetShapeSize());
        const size_t kvSeqShapeSize = static_cast<size_t>(actualSeqKvlenTensor->GetShapeSize());
        OP_CHECK_IF((seqQShapeSize != kvSeqShapeSize),
                    OP_LOGW(context_, "actualSeqQlenTensor shapeSize is not equal actualSeqKvlenTensor."),
                    return ge::GRAPH_FAILED);

        std::vector<int64_t> actualSeqQlen;
        std::vector<int64_t> actualSeqKvlen;
        const int64_t* qValue = actualSeqQlenTensor->GetData<int64_t>();
        const int64_t* kvValue = actualSeqKvlenTensor->GetData<int64_t>();
        OP_CHECK_IF((qValue == nullptr || kvValue == nullptr),
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "qValue or kvValue is nullptr."),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < seqQShapeSize; i++) {
            int64_t qSeqLen = (i == 0 ? qValue[i] : (qValue[i] - qValue[i - 1]));
            int64_t kvSeqLen = (i == 0 ? kvValue[i] : (kvValue[i] - kvValue[i - 1]));
            tmpData.actualSeqQlen.push_back(qSeqLen);
            tmpData.actualSeqKvlen.push_back(kvSeqLen);
            if (qSeqLen <= 0 || kvSeqLen <= 0) {
                OP_LOGE(context_, "NSAG get invaild actualSeqlen.");
                return ge::GRAPH_FAILED;
            }
        }

        tmpData.t1 = queryShape.GetDim(DIM_0);
        tmpData.t2 = keyShape.GetDim(DIM_0);
        uint64_t tailZeroCount = 0;
        for (auto i = seqQShapeSize - 1; i >= 1; --i) {
            if (tmpData.actualSeqQlen[i] <= 0) {
                ++tailZeroCount;
            } else {
                break;
            }
        }
        auto realBSize = seqQShapeSize - tailZeroCount;
        tilingData.opInfo.set_B(realBSize);

        int64_t dq = queryShape.GetDim(DIM_2);
        int64_t dk = keyShape.GetDim(DIM_2);
        int64_t dv = valueShape.GetDim(DIM_2);
        OP_CHECK_IF((dq != dk),
                    OP_LOGE(context_, "head_dim of Query[%ld] should be equal to head_dim of Key[%ld].", dq, dk),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((dq < dv),
                    OP_LOGE(context_, "head_dim of Query[%ld] can not less than head_dim of Value[%ld].", dq, dv),
                    return ge::GRAPH_FAILED);

        // query [t1, n1, d]   k [t2, n2, d]  v [t2, n2, d2]   dy/attentionIn [t1, n1, d2]
        OP_CHECK_IF(keyShape.GetDim(DIM_1) == 0, OP_LOGE(context_, "dim 1 of key is 0."), return ge::GRAPH_FAILED);
        tilingData.opInfo.set_G(queryShape.GetDim(DIM_1) / keyShape.GetDim(DIM_1));
        OP_CHECK_IF(tilingData.opInfo.get_G() == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);
        tilingData.opInfo.set_N2(keyShape.GetDim(DIM_1));
        tilingData.opInfo.set_S1(*std::max_element(tmpData.actualSeqQlen.begin(), tmpData.actualSeqQlen.end()));
        tilingData.opInfo.set_S2(*std::max_element(tmpData.actualSeqKvlen.begin(), tmpData.actualSeqKvlen.end()));
        tilingData.opInfo.set_D(dq);
        tilingData.opInfo.set_D2(dv);
    } else {
        OP_LOGE(context_, "inputLayout is invalid");
        return ge::GRAPH_FAILED;
    }

    tmpData.isDeterministic = context_->GetDeterministic();
    tmpData.b = tilingData.opInfo.get_B();
    tmpData.n2 = tilingData.opInfo.get_N2();
    tmpData.g = tilingData.opInfo.get_G();
    tmpData.d = tilingData.opInfo.get_D();
    tmpData.d2 = tilingData.opInfo.get_D2();
    tmpData.s1 = tilingData.opInfo.get_S1();
    tmpData.s2 = tilingData.opInfo.get_S2();

    auto ret = CheckDtypeValid(context_);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGW(context_, "dtype is invalid."), return ret);

    return CheckTndShapeValid(context_, tmpData.t1, dimN1, tmpData.d, tmpData.d2);
}

ge::graphStatus NsaSelectedAttentionGradTiling::GetBaseShapeInfo()
{
    OP_CHECK_IF(((context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY)) == nullptr) ||
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::KEY)) == nullptr) ||
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::VALUE)) == nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "InputShape of query, key or value is nullptr."),
                return ge::GRAPH_FAILED);
    const gert::Shape& queryShape = context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY))->GetStorageShape();
    const gert::Shape& keyShape = context_->GetInputShape(static_cast<size_t>(InputIndex::KEY))->GetStorageShape();
    const gert::Shape& valueShape = context_->GetInputShape(static_cast<size_t>(InputIndex::VALUE))->GetStorageShape();
    int64_t dimN1 = *context_->GetAttrs()->GetAttrPointer<int>(
        static_cast<size_t>(AttrIndex::HEAD_NUM)); // headNum is as same as N1
    auto shapeStatus = SetBaseInfo(queryShape, keyShape, valueShape, dimN1);
    if (shapeStatus != ge::GRAPH_SUCCESS) {
        return shapeStatus;
    }

    tmpData.dataTypeSize = FLOAT_PRECISION_SIZE;
    tmpData.queryType =
        static_cast<uint32_t>(context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::Setmm1TilingData(matmul_tiling::DataType inputDtype,
                                                                 matmul_tiling::DataType outputDtype)
{
    matmul_tiling::MatmulApiTiling mm1;
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, false);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, true);
    mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, outputDtype);
    mm1.SetShape(singleM, singleN, tmpData.d2);
    mm1.SetOrgShape(singleM, singleN, tmpData.d2);
    mm1.SetBias(false);
    mm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);

    if (mm1.GetTiling(tilingData.mm1TilingData) != 0) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad mm1 GetTiling Failed.");
        return ge::GRAPH_FAILED;
    }
    tilingData.mm1TilingData.set_shareMode(0);
    tilingData.mm1TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.mm1TilingData.set_shareL0CSize(aicoreParams_.l0cSize);
    tilingData.mm1TilingData.set_iterateOrder(1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::Setmm2TilingData(matmul_tiling::DataType inputDtype,
                                                                 matmul_tiling::DataType outputDtype)
{
    matmul_tiling::MatmulApiTiling mm2;
    mm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, false);
    mm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, true);
    mm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, outputDtype);
    mm2.SetShape(singleM, singleN, tmpData.d);
    mm2.SetOrgShape(singleM, singleN, tmpData.d);
    mm2.SetBias(false);
    mm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);

    if (mm2.GetTiling(tilingData.mm2TilingData) != 0) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad mm2 GetTiling Failed.");
        return ge::GRAPH_FAILED;
    }
    tilingData.mm2TilingData.set_shareMode(0);
    tilingData.mm2TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.mm2TilingData.set_shareL0CSize(aicoreParams_.l0cSize);
    tilingData.mm2TilingData.set_iterateOrder(1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::Setmm3TilingData(matmul_tiling::DataType inputDtype,
                                                                 matmul_tiling::DataType outputDtype)
{
    matmul_tiling::MatmulApiTiling mm31;
    mm31.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, false);
    mm31.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, false);
    mm31.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, outputDtype);
    mm31.SetShape(singleM, tilingData.opInfo.get_D(), singleN);
    mm31.SetOrgShape(singleM, tilingData.opInfo.get_D(), singleN);
    mm31.SetBias(false);
    mm31.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);

    if (mm31.GetTiling(tilingData.mm31TilingData) != 0) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad mm3 GetTiling Failed.");
        return ge::GRAPH_FAILED;
    }
    tilingData.mm31TilingData.set_shareMode(0);
    tilingData.mm31TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.mm31TilingData.set_shareL0CSize(aicoreParams_.l0cSize);
    tilingData.mm31TilingData.set_iterateOrder(0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::Setmm4TilingData(matmul_tiling::DataType inputDtype,
                                                                 matmul_tiling::DataType outputDtype)
{
    matmul_tiling::MatmulApiTiling mm4;
    mm4.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, true);
    mm4.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, false);
    mm4.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, outputDtype);
    mm4.SetShape(singleN, tilingData.opInfo.get_D2(), singleM);
    mm4.SetOrgShape(singleN, tilingData.opInfo.get_D2(), singleM);
    mm4.SetBias(false);
    mm4.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);

    if (mm4.GetTiling(tilingData.mm4TilingData) != 0) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad mm4 GetTiling Failed.");
        return ge::GRAPH_FAILED;
    }
    tilingData.mm4TilingData.set_shareMode(0);
    tilingData.mm4TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.mm4TilingData.set_shareL0CSize(aicoreParams_.l0cSize);
    tilingData.mm4TilingData.set_iterateOrder(0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradTiling::Setmm5TilingData(matmul_tiling::DataType inputDtype,
                                                                 matmul_tiling::DataType outputDtype)
{
    matmul_tiling::MatmulApiTiling mm5;
    mm5.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, true);
    mm5.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, false);
    mm5.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, outputDtype);
    mm5.SetShape(singleN, tilingData.opInfo.get_D(), singleM);
    mm5.SetOrgShape(singleN, tilingData.opInfo.get_D(), singleM);
    mm5.SetBias(false);
    mm5.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);

    if (mm5.GetTiling(tilingData.mm5TilingData) != 0) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad mm5 GetTiling Failed.");
        return ge::GRAPH_FAILED;
    }
    tilingData.mm5TilingData.set_shareMode(0);
    tilingData.mm5TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.mm5TilingData.set_shareL0CSize(aicoreParams_.l0cSize);
    tilingData.mm5TilingData.set_iterateOrder(0);

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(NsaSelectedAttentionGrad, NsaSelectedAttentionGradTiling, 10);

} // namespace nsa
} // namespace optiling
