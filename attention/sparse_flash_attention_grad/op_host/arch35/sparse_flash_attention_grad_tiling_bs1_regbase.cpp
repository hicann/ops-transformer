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
 * \file sparse_flash_attention_grad_tiling_bs1_regbase.cpp
 * \brief
 */

#include <set>
#include "sparse_flash_attention_grad_tiling_bs1_regbase.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"

namespace optiling {
namespace sfag {
constexpr uint32_t WORKSPACE_BASE_CAL = 32 * 1024 * 1024; // 100MB系统预留
constexpr uint32_t BLOCK = 32;                            // 32B
constexpr uint32_t B32 = 4;                               // 4B
constexpr uint32_t B16 = 2;
constexpr uint32_t BASE_LEN_256 = 256;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t PING_PONG_BUFFER = 2;
constexpr uint32_t D_SIZE = 512;
constexpr uint32_t DROPE_SIZE = 64;

ge::graphStatus SparseFlashAttentionGradBs1Regbase::GetShapeAttrsInfo()
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

    OP_LOGI(context_, "SparseFlashAttentionGrad with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] d1[%ld]!",
            baseParams_->get_b(), baseParams_->get_n2(), baseParams_->get_g(), baseParams_->get_s1(),
            baseParams_->get_s2(), baseParams_->get_d(), baseParams_->get_d1());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBs1Regbase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    uint64_t l2CacheSize;
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const SparseFlashAttentionGradCompileInfo *>(context_->GetCompileInfo());
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

bool SparseFlashAttentionGradBs1Regbase::IsCapable()
{
    OP_LOGI(context_, "SparseFlashAttentionGrad basic template hit.");
    return true;
}

ge::graphStatus SparseFlashAttentionGradBs1Regbase::DoOpTiling()
{
    OP_LOGI(context_, "SparseFlashAttentionGrad DoTiling start");

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

ge::graphStatus SparseFlashAttentionGradBs1Regbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBs1Regbase::GetWorkspaceSize()
{
    int64_t coreNum = aicoreParams_.aicNum;
    int64_t launchBlockDims = aicoreParams_.numBlocks;

    // Tiling传递的内存大小、起始地址，统一为字节数，单位为B
    auto blockdim = CalcTschBlockDim(launchBlockDims, aicoreParams_.aicNum, aicoreParams_.numBlocks);
    OP_CHECK_IF(blockdim == 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "blockdim is 0, aicNum is %lu, aivNum is %lu.",
                                            aicoreParams_.aicNum, aicoreParams_.numBlocks),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);

    // kernel 使用 CrossCoreSetFlag/CrossCoreWaitFlag 做全核 rendezvous（deterministic scatter 的
    // flag11/flag12），必须 batch mode 让所有核同时启动，否则多流下会卡死
    context_->SetScheduleMode(1);

    // 系统预留
    int64_t sysLen = WORKSPACE_BASE_CAL;

    // Gather/Scatter
    int64_t selectedKWorkspaceLen = 128 * tmpData.d * B16;

    selectedKWorkspaceLen = AlignData(selectedKWorkspaceLen, GM_ALIGN) * 3;
    // selectedKWorkspaceLen = AlignData(selectedKWorkspaceLen, GM_ALIGN) * PING_PONG_BUFFER;

    int64_t mm4WorkspaceLen = tmpData.selected_block_count * tmpData.d * B32;
    int64_t mm5WorkspaceLen = tmpData.selected_block_count * tmpData.d1 * B32;
    mm4WorkspaceLen = AlignData(mm4WorkspaceLen, GM_ALIGN) * PING_PONG_BUFFER;
    mm5WorkspaceLen = AlignData(mm5WorkspaceLen, GM_ALIGN) * PING_PONG_BUFFER;

    int64_t dqWorkspaceLen = tmpData.dqWorkspaceLen;
    int64_t dkWorkspaceLen = tmpData.dkWorkspaceLen;
    int64_t dvWorkspaceLen = tmpData.dvWorkspaceLen;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = sysLen;
    workspaces[0] += selectedKWorkspaceLen * coreNum;
    workspaces[0] += (mm4WorkspaceLen + mm5WorkspaceLen) * coreNum;
    workspaces[0] += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    // dSink 跨核 reduce 中间 workspace：仅确定性 + HAS_SINKS 时分配（每 AIC 1 slot=MAX_N1×FP32=512B，无 DB）
    // 非 deter 走 AtomicAdd 直写 dSinksGm，无需中间 workspace（len=0）
    int64_t usedCoreNum = baseParams_->get_usedCoreNum();
    tmpData.dSinkWorkSpaceLen = 0;
    if (tmpData.hasSinks && tmpData.deterministic) {
        tmpData.dSinkWorkSpaceLen = AlignData(usedCoreNum * MAX_N1 * B32, GM_ALIGN);
    }
    workspaces[0] += tmpData.dSinkWorkSpaceLen;

    baseParams_->set_selectedKWorkSpaceOffset(0);
    int64_t workspaceOffsets = selectedKWorkspaceLen * coreNum;
    baseParams_->set_mm4ResWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets += mm4WorkspaceLen * coreNum;
    baseParams_->set_mm5ResWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets += mm5WorkspaceLen * coreNum;
    postTilingData_->set_dqWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + dqWorkspaceLen;
    postTilingData_->set_dkWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + dkWorkspaceLen;
    postTilingData_->set_dvWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + dvWorkspaceLen;
    // dSinkWorkSpace 紧跟 dv 区之后（对齐 dq/dk/dv 偏移模式）
    postTilingData_->set_dSinkWorkSpaceOffset(workspaceOffsets);

    return ge::GRAPH_SUCCESS;
}

uint64_t SparseFlashAttentionGradBs1Regbase::GetTilingKey() const
{
    int64_t inputDtypeSize = tmpData.queryType == ge::DT_FLOAT16 ? 0 : 1;
    int64_t isTnd = tmpData.layout == static_cast<uint32_t>(InputLayout::TND) ? 1 : 0;
    uint64_t tilingKey = 0;

    OP_LOGI(context_,
            "SparseFlashAttentionGrad get tilingkey, InputDType[%ld], IsTnd[%ld], GTemplateNum[%ld], "
            "S2TemplateNum[%ld], DTemplateNum[%ld], IsRope[%d], Deterministic[%d], KvMerge[%d], HasSinks[%d]",
            inputDtypeSize, isTnd, tmpData.singleM, tmpData.singleN, tmpData.d,
            static_cast<uint8_t>(tmpData.ropeEnable), static_cast<uint8_t>(tmpData.deterministic),
            static_cast<uint8_t>(tmpData.kvMerge), static_cast<uint8_t>(tmpData.hasSinks));
    // tmpData.singleM 为G方向上固定切分大小 tmpData.singleN为S2方向上固定切分大小
    tilingKey = GET_TPL_TILING_KEY(static_cast<uint8_t>(inputDtypeSize), static_cast<uint8_t>(isTnd),
                                   static_cast<uint16_t>(tmpData.singleM), static_cast<uint16_t>(tmpData.singleN),
                                   static_cast<uint16_t>(tmpData.d), static_cast<uint8_t>(tmpData.ropeEnable),
                                   static_cast<uint8_t>(tmpData.deterministic), static_cast<uint8_t>(tmpData.kvMerge),
                                   static_cast<uint8_t>(tmpData.hasSinks));

    OP_LOGI(context_,
            "SparseFlashAttentionGrad DoTiling success, tilingkey is"
            " %lu.",
            tilingKey);
    return tilingKey;
}

ge::graphStatus SparseFlashAttentionGradBs1Regbase::DoBlockTiling()
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
    baseParams_->set_totalSize(nNums);
    baseParams_->set_usedCoreNum(usedCoreNum);
    baseParams_->set_formerCoreNum(aicNum - remainCoreNum);
    baseParams_->set_formerCoreProcessNNum(formerCoreProcessNNums);
    baseParams_->set_remainCoreProcessNNum(formerCoreProcessNNums - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBs1Regbase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBs1Regbase::DoCastTiling()
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

    // key
    int64_t allNumKey = baseParams_->get_b() * baseParams_->get_n2() * baseParams_->get_s2() * dAlign;
    // TND时候要按照真实的key的num数计算
    if (baseParams_->get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumKey = tmpData.t2 * baseParams_->get_n2() * 1 * dAlign;
    }

    // Value
    int64_t allNumValue = baseParams_->get_b() * baseParams_->get_n2() * baseParams_->get_s2() * d1Align;
    // TND时候要按照真实的value的num数计算
    if (baseParams_->get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumValue = tmpData.t2 * baseParams_->get_n2() * 1 * baseParams_->get_d1();
    }

    uint32_t typeSize = B16;
    uint32_t usedCoreNum = aicoreParams_.numBlocks;
    constexpr uint32_t postNzCoexNode = 12;

    uint32_t postUbBaseSize = 0;
    uint32_t qPostBaseNum = 0;
    int64_t nzReservedSize = 0;
    int64_t curPostCoexNode = postNzCoexNode;
    postUbBaseSize = (aicoreParams_.ubSize - 2 * nzReservedSize) / curPostCoexNode / // 开DB预留2份nzReservedSize
                     BASE_LEN_256 * BASE_LEN_256;
    OP_LOGI(context_, "DoCastTiling postUbBaseSize: %ld.", postUbBaseSize);
    qPostBaseNum = postUbBaseSize / typeSize / dAlign * (baseParams_->get_d());
    OP_LOGI(context_, "DoCastTiling qPostBaseNum: %ld.", qPostBaseNum);

    OP_CHECK_IF(qPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "qPostBaseNum is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "castUsedCoreNum is 0."),
                return ge::GRAPH_FAILED);

    int64_t qPreBlockFactor = (allNumQuery + usedCoreNum - 1) / usedCoreNum;
    int64_t qPreBlockTotal = (allNumQuery + qPreBlockFactor - 1) / qPreBlockFactor;
    int64_t qPreBlockTailTmp = allNumQuery % qPreBlockFactor;
    int64_t qPreBlockTail = qPreBlockTailTmp == 0 ? qPreBlockFactor : qPreBlockTailTmp;

    int64_t kPreBlockFactor = (allNumKey + usedCoreNum - 1) / usedCoreNum;
    ;
    int64_t kPreBlockTotal = (allNumKey + kPreBlockFactor - 1) / kPreBlockFactor;
    int64_t kPreBlockTailTmp = allNumKey % kPreBlockFactor;
    int64_t kPreBlockTail = kPreBlockTailTmp == 0 ? kPreBlockFactor : kPreBlockTailTmp;

    int64_t vPreBlockFactor = (allNumValue + usedCoreNum - 1) / usedCoreNum;
    int64_t vPreBlockTotal = (allNumValue + vPreBlockFactor - 1) / vPreBlockFactor;
    int64_t vPreBlockTailTmp = allNumValue % vPreBlockFactor;
    int64_t vPreBlockTail = vPreBlockTailTmp == 0 ? vPreBlockFactor : vPreBlockTailTmp;

    preTilingData_->set_qPreBlockFactor(qPreBlockFactor);
    preTilingData_->set_qPreBlockTotal(qPreBlockTotal);
    preTilingData_->set_qPreBlockTail(qPreBlockTail);
    preTilingData_->set_kPreBlockFactor(kPreBlockFactor);
    preTilingData_->set_kPreBlockTotal(kPreBlockTotal);
    preTilingData_->set_kPreBlockTail(kPreBlockTail);
    preTilingData_->set_vPreBlockFactor(vPreBlockFactor);
    preTilingData_->set_vPreBlockTotal(vPreBlockTotal);
    preTilingData_->set_vPreBlockTail(vPreBlockTail);

    int64_t qPostBlockTotal = allNumQuery / dAlign * (baseParams_->get_d());
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t kPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(kPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "kPostBaseNum is 0."), return ge::GRAPH_FAILED);
    int64_t kPostBlockTotal = allNumKey / dAlign * (baseParams_->get_d());
    int64_t kPostTailNumTmp = kPostBlockTotal % kPostBaseNum;
    int64_t kPostTailNum = kPostTailNumTmp == 0 ? kPostBaseNum : kPostTailNumTmp;
    int64_t kPostBlockOuterTotal = (kPostBlockTotal + kPostBaseNum - 1) / kPostBaseNum;
    int64_t kPostBlockFactor = (kPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t vPostBaseNum = postUbBaseSize / typeSize / d1Align * baseParams_->get_d1();
    OP_CHECK_IF(vPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "vPostBaseNum is 0."), return ge::GRAPH_FAILED);
    int64_t vPostBlockTotal = allNumValue / d1Align * baseParams_->get_d1();
    int64_t vPostTailNumTmp = vPostBlockTotal % vPostBaseNum;
    int64_t vPostTailNum = vPostTailNumTmp == 0 ? vPostBaseNum : vPostTailNumTmp;
    int64_t vPostBlockOuterTotal = (vPostBlockTotal + vPostBaseNum - 1) / vPostBaseNum;
    int64_t vPostBlockFactor = (vPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    postTilingData_->set_postUbBaseSize(postUbBaseSize);

    postTilingData_->set_qPostBlockFactor(qPostBlockFactor);
    postTilingData_->set_qPostBlockTotal(qPostBlockTotal);
    postTilingData_->set_qPostBaseNum(qPostBaseNum);
    postTilingData_->set_qPostTailNum(qPostTailNum);

    postTilingData_->set_kPostBlockFactor(kPostBlockFactor);
    postTilingData_->set_kPostBlockTotal(kPostBlockTotal);
    postTilingData_->set_kPostBaseNum(kPostBaseNum);
    postTilingData_->set_kPostTailNum(kPostTailNum);

    postTilingData_->set_vPostBlockFactor(vPostBlockFactor);
    postTilingData_->set_vPostBlockTotal(vPostBlockTotal);
    postTilingData_->set_vPostBaseNum(vPostBaseNum);
    postTilingData_->set_vPostTailNum(vPostTailNum);

    tmpData.dqWorkspaceLen = (allNumQuery * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    tmpData.dkWorkspaceLen = (allNumKey * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    tmpData.dvWorkspaceLen = (allNumValue * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBs1Regbase::CheckOutShapeInfo(const gert::Shape &inputshape,
                                                                      const char *inputName,
                                                                      const gert::Shape &outputshape,
                                                                      const char *inputLayout)
{
    if (strcmp(inputLayout, TND_STR) == 0) {
        if (inputshape.GetDim(DIM_0) != outputshape.GetDim(DIM_0) ||
            inputshape.GetDim(DIM_1) != outputshape.GetDim(DIM_1) ||
            inputshape.GetDim(DIM_2) != outputshape.GetDim(DIM_2)) {
            OP_LOGE(context_,
                    "SparseFlashAttentionGrad Input %s [%ld, %ld, %ld] is not equal to Output d_%s [%ld, %ld, %ld]",
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
                    "SparseFlashAttentionGrad Input %s [%ld, %ld, %ld, %ld] is not equal to Output d_%s [%ld, %ld, "
                    "%ld, %ld]",
                    inputName, inputshape.GetDim(DIM_0), inputshape.GetDim(DIM_1), inputshape.GetDim(DIM_2),
                    inputshape.GetDim(DIM_3), inputName, outputshape.GetDim(DIM_0), outputshape.GetDim(DIM_1),
                    outputshape.GetDim(DIM_2), outputshape.GetDim(DIM_3));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBs1Regbase::GetBaseShapeInfo()
{
    OP_CHECK_IF(((context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY)) == nullptr) ||
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::KEY)) == nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "InputShape of query or key is nullptr."), return ge::GRAPH_FAILED);
    // input
    // TND: query [t1, n1, d]   k [t2, n2, d]  v [t2, n2, d1]   dy/attentionIn [t1, n1, d1]
    // BSND: query [b, s1, n1, d]   k [b, s2, n2, d]  v [b, s2, n2, d1]   dy/attentionIn [b, s1, n1, d1]
    const gert::Shape &queryShape = context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY))->GetStorageShape();
    const gert::Shape &keyShape = context_->GetInputShape(static_cast<size_t>(InputIndex::KEY))->GetStorageShape();
    const gert::StorageShape *valueStorageShape =
        context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::VALUE));
    tmpData.kvMerge = (valueStorageShape == nullptr);
    const gert::Shape &valueShape = tmpData.kvMerge ? keyShape : valueStorageShape->GetStorageShape();
    const gert::Shape &indicesShape =
        context_->GetInputShape(static_cast<size_t>(InputIndex::TOPK_INDICES))->GetStorageShape();
    auto qRopeTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::Q_ROPE));
    auto kRopeTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::K_ROPE));
    // sinks 输入（OPTIONAL, FP32, [N1]）：判空只在这一处，编入 tiling key bit35 HasSinks
    // 空 tensor（shape [0]）与 nullptr 等价，均视为无 sink（对应文档"为空时 sink 路径编译期消除"），
    // 否则空 sinks 会因 shape [0] != [N1] 在校验处被拒。仅用于支持无 sink 性能/执行路径。
    auto sinksTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::SINKS));
    bool sinksEmpty = false;
    if (sinksTensor != nullptr) {
        const gert::Shape &sinksShape = sinksTensor->GetStorageShape();
        sinksEmpty = (sinksShape.GetDimNum() == 1 && sinksShape.GetDim(0) == 0);
    }
    tmpData.hasSinks = (sinksTensor != nullptr) && !sinksEmpty;
    const gert::Shape &dqShape = context_->GetOutputShape(static_cast<size_t>(OutputIndex::DQ))->GetStorageShape();
    const gert::Shape &dkShape = context_->GetOutputShape(static_cast<size_t>(OutputIndex::DK))->GetStorageShape();
    auto dvOutputShape = context_->GetOutputShape(static_cast<size_t>(OutputIndex::DV));
    tmpData.deterministic = (context_->GetDeterministic() == 1);
    uint32_t dimSize = queryShape.GetDimNum();
    int64_t dimDq = queryShape.GetDim(dimSize - 1);
    int64_t dimDk = keyShape.GetDim(dimSize - 1);
    int64_t dimDv = valueShape.GetDim(dimSize - 1);

    // attrs
    const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    auto selected_block_count = indicesShape.GetDim(dimSize - 1);
    if (selected_block_count % 1024 != 0 || selected_block_count < 1024 || selected_block_count > 8192) {
        OP_LOGE(context_,
                "SparseFlashAttentionGrad only support selected_block_count [1024, 2048, 3072, 4096, 5120, 6144, 7168, "
                "8192] now, but got selected_block_count=%ld.",
                selected_block_count);
        return ge::GRAPH_FAILED;
    }
    auto selected_block_size =
        *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_SIZE));
    if (selected_block_size != 1) {
        OP_LOGE(context_,
                "SparseFlashAttentionGrad only support sparse_block_size [1] now, but got sparse_block_size=%ld.",
                selected_block_size);
        return ge::GRAPH_FAILED;
    }

    auto sparse_mode = *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SPARSE_MODE));
    if (sparse_mode == 3 || sparse_mode == 0) {
        OP_LOGI(context_, "SparseFlashAttentionGrad AttenMask enable.");
        tmpData.attenEnable = true;
    } else {
        OP_LOGE(context_, "SparseFlashAttentionGrad only support sparse_mode = 0 or 3, now sparse_mode=%d.",
                sparse_mode);
        return ge::GRAPH_FAILED;
    }
    if (dimDq != D_SIZE || dimDk != D_SIZE || dimDq != dimDk) {
        OP_LOGE(context_,
                "head_dim of Query[%ld] should be equal to head_dim of Key[%ld], and their value must be 512.", dimDq,
                dimDk);
        return ge::GRAPH_FAILED;
    }
    if (dimDq < dimDv) {
        OP_LOGE(context_, "head_dim of Query[%ld] can not less than head_dim of Value[%ld].", dimDq, dimDv);
        return ge::GRAPH_FAILED;
    }
    if (strcmp(inputLayout, TND_STR) != 0 && strcmp(inputLayout, BSND_STR) != 0) {
        OP_LOGE(context_, "SparseFlashAttentionGrad only support TND or BSND layout, now layout is %s.", inputLayout);
        return ge::GRAPH_FAILED;
    }
    if (static_cast<size_t>(dimSize) != strlen(inputLayout)) {
        OP_LOGE(context_,
                "SparseFlashAttentionGrad layout dims is not equal to the input's dim, now the query of dim is %u.",
                dimSize);
        return ge::GRAPH_FAILED;
    }
    // 对输入shape进行校验
    auto status = CheckOutShapeInfo(queryShape, "query", dqShape, inputLayout);
    if (status == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    status = CheckOutShapeInfo(keyShape, "key", dkShape, inputLayout);
    if (status == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (!tmpData.kvMerge) {
        OP_CHECK_IF(dvOutputShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "OutputShape of dv is nullptr."),
                    return ge::GRAPH_FAILED);
        const gert::Shape &dvShape = dvOutputShape->GetStorageShape();
        status = CheckOutShapeInfo(valueShape, "value", dvShape, inputLayout);
        if (status == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    } else if (dvOutputShape != nullptr) {
        const gert::Shape &dvShape = dvOutputShape->GetStorageShape();
        int64_t dvSize = dvShape.GetDimNum() == 0 ? 0 : 1;
        for (size_t i = 0; i < dvShape.GetDimNum(); ++i) {
            dvSize *= dvShape.GetDim(i);
        }
        if (dvShape.GetDimNum() > 0 && dvSize > 0) {
            OP_LOGE(
                context_,
                "When value is nullptr (kvMerge), dValueOut must also be nullptr/empty, but got a non-empty dv shape.");
            return ge::GRAPH_FAILED;
        }
    }
    if (qRopeTensor != nullptr && kRopeTensor != nullptr) {
        OP_LOGD(context_, "SparseFlashAttentionGrad qRope and kRope is not nullptr, rope is enabled.");
        tmpData.ropeEnable = true;
        const gert::Shape &qRopeShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::Q_ROPE))->GetStorageShape();
        const gert::Shape &kRopeShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::K_ROPE))->GetStorageShape();
        OP_CHECK_IF(context_->GetOutputShape(DIM_3) == nullptr || context_->GetOutputShape(DIM_4) == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "OutputShape of dq_rope or dk_rope is nullptr."),
                    return ge::GRAPH_FAILED);
        const gert::Shape &dqRopeShape = context_->GetOutputShape(DIM_3)->GetStorageShape();
        const gert::Shape &dkRopeShape = context_->GetOutputShape(DIM_4)->GetStorageShape();
        auto qRopeDim = qRopeShape.GetDim(dimSize - 1);
        auto kRopeDim = kRopeShape.GetDim(dimSize - 1);
        if (qRopeDim != DROPE_SIZE || kRopeDim != DROPE_SIZE) {
            OP_LOGE(context_,
                    "SparseFlashAttentionGrad headDim of qRope and kRope should be 64, but qRope[%ld], kRope[%ld].",
                    qRopeDim, kRopeDim);
            return ge::GRAPH_FAILED;
        }
        if (qRopeDim != kRopeDim) {
            OP_LOGE(context_, "SparseFlashAttentionGrad headDim of qRope and kRope should be equal.");
            return ge::GRAPH_FAILED;
        }
        tmpData.ropeDim = kRopeDim;
        status = CheckOutShapeInfo(qRopeShape, "query_rope", dqRopeShape, inputLayout);
        if (status == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
        status = CheckOutShapeInfo(kRopeShape, "key_rope", dkRopeShape, inputLayout);
        if (status == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    } else {
        tmpData.ropeEnable = false;
        tmpData.ropeDim = 0;
    }

    if (strcmp(inputLayout, TND_STR) == 0) {
        auto actSeqQLen = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_Q_LEN));
        auto actSeqKVLen = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
        OP_CHECK_IF(
            actSeqQLen == nullptr || actSeqKVLen == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "When layout is TND, actSeqQLen / actSeqKVLen can not be nullptr"),
            return ge::GRAPH_FAILED);
        const gert::Shape &actSeqQLenShape = actSeqQLen->GetStorageShape();
        const gert::Shape &actSeqKVLenShape = actSeqKVLen->GetStorageShape();
        tmpData.b = actSeqQLenShape.GetDim(DIM_0);
        OP_CHECK_IF(tmpData.b == 0 || actSeqKVLenShape.GetDim(DIM_0) != tmpData.b,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "actSeqQLen / actSeqKVLen shape is invalid for TND"),
                    return ge::GRAPH_FAILED);
        tmpData.t1 = queryShape.GetDim(DIM_0);
        tmpData.t2 = keyShape.GetDim(DIM_0);
        tmpData.s1 = tmpData.t1;
        tmpData.s2 = tmpData.t2;
        tmpData.n2 = keyShape.GetDim(DIM_1);
        OP_CHECK_IF(tmpData.n2 == 0, OP_LOGE(context_, "key headNum is 0"), return ge::GRAPH_FAILED);
        tmpData.g = queryShape.GetDim(DIM_1) / tmpData.n2;
        tmpData.layout = static_cast<uint32_t>(InputLayout::TND);
    } else { // BSND
        tmpData.b = queryShape.GetDim(DIM_0);
        tmpData.s1 = queryShape.GetDim(DIM_1);
        tmpData.s2 = keyShape.GetDim(DIM_1);
        tmpData.n2 = keyShape.GetDim(DIM_2);
        OP_CHECK_IF(tmpData.n2 == 0, OP_LOGE(context_, "key headNum is 0"), return ge::GRAPH_FAILED);
        tmpData.g = queryShape.GetDim(DIM_2) / tmpData.n2;
        tmpData.layout = static_cast<uint32_t>(InputLayout::BSND);
    }

    if (tmpData.g <= 0) {
        OP_LOGE(context_, "g (N1 / N2) should be larger than 0, but got g=%ld.", tmpData.g);
        return ge::GRAPH_FAILED;
    }

    int64_t n1 = tmpData.n2 * tmpData.g;
    std::set<uint32_t> gSizeSupportList = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128};
    if (tmpData.n2 != 1 || n1 > 128 || !gSizeSupportList.count(n1)) {
        OP_LOGE(context_,
                "SparseFlashAttentionGrad only support n2=1 and n1=1/2/3/4/6/8/12/16/24/32/48/64/128, but got n2=%ld "
                "n1=%ld.",
                tmpData.n2, n1);
        return ge::GRAPH_FAILED;
    }

    // sinks 校验：shape 必须为 [N1]（n1 = n2*g）、dtype 必须为 FP32
    if (tmpData.hasSinks) {
        const gert::Shape &sinksShape = sinksTensor->GetStorageShape();
        if (sinksShape.GetDimNum() != 1 || sinksShape.GetDim(0) != n1) {
            OP_LOGE(context_, "SparseFlashAttentionGrad sinks shape should be [N1]=[%ld], but got dimNum=%u dim0=%ld.",
                    n1, static_cast<uint32_t>(sinksShape.GetDimNum()), sinksShape.GetDim(0));
            return ge::GRAPH_FAILED;
        }
        auto sinksDtype = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SINKS))->GetDataType();
        if (sinksDtype != ge::DT_FLOAT) {
            OP_LOGE(context_, "SparseFlashAttentionGrad sinks dtype should be FP32, but got %d.",
                    static_cast<int>(sinksDtype));
            return ge::GRAPH_FAILED;
        }
    }

    baseParams_->set_b(tmpData.b);
    baseParams_->set_g(tmpData.g);
    OP_CHECK_IF(baseParams_->get_g() == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);
    baseParams_->set_isHeadNLe64(tmpData.g <= SFAG_HEAD_N_L1_RESIDENT ? 1 : 0);
    baseParams_->set_n2(tmpData.n2);
    baseParams_->set_s1(tmpData.s1);
    baseParams_->set_s2(tmpData.s2);
    // d=576 dq+drope /  d1=512 dq
    baseParams_->set_d(dimDq + tmpData.ropeDim);
    baseParams_->set_d1(dimDq);
    baseParams_->set_selectedBlockCount(selected_block_count);
    baseParams_->set_layout(tmpData.layout);
    baseParams_->set_sparseMode(sparse_mode);
    baseParams_->set_scaleValue(
        *context_->GetAttrs()->GetAttrPointer<float>(static_cast<size_t>(AttrIndex::SCALE_VALUE)));

    tmpData.d = baseParams_->get_d();
    tmpData.d1 = baseParams_->get_d1();
    tmpData.dataTypeSize = B32;
    tmpData.queryType =
        static_cast<uint32_t>(context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType());
    tmpData.selected_block_count = selected_block_count;
    tmpData.selected_block_size = selected_block_size;

    auto ret = CheckDtypeValid(context_);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SparseFlashAttentionGrad the dtype of input is invalid.");
        return ge::GRAPH_FAILED;
    }

    if (tmpData.layout == static_cast<uint32_t>(InputLayout::TND)) {
        ret = CheckTndShapeValid(context_, tmpData.t1, tmpData.n2 * tmpData.g, tmpData.d, tmpData.d1, tmpData.n2);
    }

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SparseFlashAttentionGrad the input shpae of TND Layout is invalid.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(SparseFlashAttentionGrad, SparseFlashAttentionGradBs1Regbase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace sfag
} // namespace optiling
