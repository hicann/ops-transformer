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
 * \file sparse_flash_attention_grad_tiling_bs1_basic.cpp
 * \brief
 */

#include "sparse_flash_attention_grad_tiling_bs1_basic.h"
#include "op_host/tiling_templates_registry.h"
#include <algorithm>

namespace optiling {
namespace sfag {
constexpr uint32_t WORKSPACE_BASE_CAL = 32 * 1024 * 1024; // 100MB系统预留
constexpr uint32_t BLOCK = 32;                            // 32B
constexpr uint32_t B32 = 4;                               // 4B
constexpr uint32_t B16 = 2;
constexpr uint32_t BASE_LEN_256 = 256;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t PING_PONG_BUFFER = 2;
constexpr uint32_t SCATTER_BUFFER_NUM = 3;
constexpr uint64_t KERNEL_UB_SIZE = 191 * 1024;

namespace {
struct OptimizedScatterGateDecision {
    bool enabled = false;
    const char *reason = "unknown";
    int64_t n1 = 0;
    int64_t gateS1 = 0;
    int64_t gateS2 = 0;
    int64_t effectiveSelectedTokens = 0;
};

OptimizedScatterGateDecision GetOptimizedScatterGateDecision(const TempParams &tmpData)
{
    OptimizedScatterGateDecision decision;
    int64_t s1 = tmpData.s1;
    int64_t s2 = tmpData.s2;
    int64_t n1 = tmpData.n2 * tmpData.g;
    int64_t selectedBlockCount = static_cast<int64_t>(tmpData.selected_block_count);
    int64_t selectedBlockSize = static_cast<int64_t>(tmpData.selected_block_size);
    int64_t selectedTokenLimit = selectedBlockCount * selectedBlockSize;
    decision.n1 = n1;
    if (tmpData.deterministic) {
        decision.reason = "disabled: deterministic path uses non-optimized scatter";
        return decision;
    }
    if (s1 <= 0 || s2 <= 0 || selectedBlockCount <= 0 || selectedBlockSize <= 0) {
        decision.reason = "disabled: invalid s1/s2/selected block count/selected block size";
        return decision;
    }
    if (n1 <= 0 || n1 >= 128) {
        decision.reason = "disabled: N1 must be in [1, 127] for optimized scatter";
        return decision;
    }
    int64_t gateS1 = s1;
    int64_t gateS2 = s2;
    if (tmpData.layout == static_cast<uint32_t>(InputLayout::TND) && tmpData.b > 1) {
        // TND s1/s2 are total tokens across batch; gate by per-sequence scale.
        gateS1 = s1 / tmpData.b;
        gateS2 = s2 / tmpData.b;
    }
    decision.gateS1 = gateS1;
    decision.gateS2 = gateS2;
    if (gateS1 <= 0 || gateS2 <= 0) {
        decision.reason = "disabled: invalid per-sequence S1/S2 after TND batch normalization";
        return decision;
    }
    int64_t effectiveSelectedTokens = std::min(selectedTokenLimit, gateS2);
    decision.effectiveSelectedTokens = effectiveSelectedTokens;

    if (selectedBlockSize >= 8) {
        decision.enabled =
            selectedBlockCount >= 1024 && effectiveSelectedTokens >= 8192 && gateS1 >= 2048 && gateS2 >= 8192;
        decision.reason = decision.enabled ?
                              "enabled: block_size>=8 and long-sequence thresholds satisfied" :
                              "disabled: block_size>=8 but selected_count/effective_tokens/S1/S2 below thresholds";
        return decision;
    }

    if (n1 <= 32) {
        decision.enabled = effectiveSelectedTokens >= 1024 && gateS1 >= 2048 && gateS2 >= 4096;
        decision.reason = decision.enabled ?
                              "enabled: block_size=1, N1<=32 and long-sequence thresholds satisfied" :
                              "disabled: block_size=1, N1<=32 but effective_tokens/S1/S2 below thresholds";
        return decision;
    }

    if (n1 == 64) {
        if (effectiveSelectedTokens >= 4096) {
            decision.enabled = gateS2 >= 8192 || gateS1 >= 2048;
            decision.reason = decision.enabled ?
                                  "enabled: block_size=1, N1=64, effective_tokens>=4096 and S1/S2 threshold satisfied" :
                                  "disabled: block_size=1, N1=64, effective_tokens>=4096 but S1/S2 below thresholds";
            return decision;
        }
        decision.enabled = effectiveSelectedTokens >= 2048 && (gateS2 >= 16384 || gateS1 >= 4096);
        decision.reason =
            decision.enabled ?
                "enabled: block_size=1, N1=64, effective_tokens>=2048 and longer S1/S2 threshold satisfied" :
                "disabled: block_size=1, N1=64 but effective_tokens/S1/S2 below thresholds";
        return decision;
    }

    decision.reason = "disabled: block_size=1 optimization only supports N1<=64";
    return decision;
}
} // namespace

bool EnableOptimizedScatterPath(const TempParams &tmpData)
{
    return GetOptimizedScatterGateDecision(tmpData).enabled;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::GetShapeAttrsInfo()
{
    /*
    Get all shape info and attr
    */
    opName = (context_ != nullptr) ? context_->GetNodeName() : nullptr;
    OP_CHECK_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "context is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetAttrs() == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "GetAttrs is nullptr."),
                return ge::GRAPH_FAILED);

    auto status = GetBaseShapeInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    OP_LOGI(context_, "SparseFlashAttentionGrad with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] d2[%ld]!",
            tilingData.opInfo.get_B(), tilingData.opInfo.get_N2(), tilingData.opInfo.get_G(),
            tilingData.opInfo.get_S1(), tilingData.opInfo.get_S2(), tilingData.opInfo.get_D(),
            tilingData.opInfo.get_D2());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::GetPlatformInfo()
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

bool SparseFlashAttentionGradBasicTiling::IsCapable()
{
    OP_LOGI(context_, "SparseFlashAttentionGrad basic template hit.");
    return true;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::DoOpTiling()
{
    OP_LOGI(context_, "SparseFlashAttentionGrad DoTiling start");

    // Init
    tmpData.singleM = tilingData.opInfo.get_G();
    tmpData.singleN = 128; // PER_LOOP_BLOCK_SIZE

    // setTilingData
    tilingData.splitCoreParams.set_singleM(tmpData.singleM);
    tilingData.splitCoreParams.set_singleN(tmpData.singleN);

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

ge::graphStatus SparseFlashAttentionGradBasicTiling::DoLibApiTiling()
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

ge::graphStatus SparseFlashAttentionGradBasicTiling::GetWorkspaceSize()
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
    int64_t dkWorkspaceLen = tilingData.opInfo.get_dkWorkspaceLen();
    int64_t dvWorkspaceLen = tilingData.opInfo.get_dvWorkspaceLen();

    // Gather/Scatter
    int64_t selectedKWorkspaceLen = selectedS2 * (tmpData.d + tmpData.ropeDim) * inputDtypeSize;
    selectedKWorkspaceLen = AlignData(selectedKWorkspaceLen, GM_ALIGN);

    selectedKWorkspaceLen *= 4;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = sysLen;
    workspaces[0] += selectedKWorkspaceLen * currentUseCoreNum;
    workspaces[0] += mm12WorkspaceLen * 4 * currentUseCoreNum;
    workspaces[0] += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    int64_t dAlign = (tilingData.opInfo.get_D() + tilingData.opInfo.get_ropeD() + 15) / 16 * 16;
    int64_t d2Align = (tilingData.opInfo.get_D2() + 15) / 16 * 16;
    uint32_t scatterBufferNum = tmpData.enableOptimizedScatter ? SCATTER_BUFFER_NUM : PING_PONG_BUFFER;
    workspaces[0] +=
        24 * scatterBufferNum * tmpData.selected_block_count * tmpData.selected_block_size * (dAlign + d2Align) * B32;

    tilingData.opInfo.set_mm12WorkspaceLen(mm12WorkspaceLen);
    tilingData.opInfo.set_selectedKWorkspaceLen(selectedKWorkspaceLen);
    tilingData.opInfo.set_selectedVWorkspaceLen(0);

    int64_t workspaceOffsets = selectedKWorkspaceLen * currentUseCoreNum;
    workspaceOffsets += mm12WorkspaceLen * 4 * currentUseCoreNum;
    tilingData.postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + tilingData.opInfo.get_dqWorkspaceLen();
    tilingData.postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + tilingData.opInfo.get_dkWorkspaceLen();
    tilingData.postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::PostTiling()
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

uint64_t SparseFlashAttentionGradBasicTiling::GetTilingKey() const
{
    uint64_t tilingKey = 10;
    if (tmpData.attenEnable) {
        tilingKey += 1;
    }
    tilingKey *= 10;
    if (tmpData.ropeDim != 0) {
        tilingKey += 1;
    }
    tilingKey *= 10;
    if (tmpData.layout == static_cast<uint32_t>(InputLayout::BSND)) {
        tilingKey += 1;
    }
    tilingKey *= 10;
    if (tmpData.deterministic) {
        tilingKey += 1;
    }
    tilingKey *= 10;
    if (tmpData.kvMerge) {
        tilingKey += 1;
    }

    OP_LOGI(context_,
            "SparseFlashAttentionGrad DoTiling success, tilingkey is"
            " %lu.",
            tilingKey);
    return tilingKey;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::DoBlockTiling()
{
    /*
     * 分核策略
     */
    int64_t nNums = tmpData.layout == static_cast<uint32_t>(InputLayout::TND) ? tmpData.t1 : tmpData.b * tmpData.s1;
    int64_t aicNum = static_cast<int64_t>(aicoreParams_.aicNum);
    int64_t usedCoreNum = nNums < aicNum ? nNums : aicNum;
    int64_t formerCoreProcessNNums = CeilCommon(nNums, aicNum);
    int64_t remainCoreNum = formerCoreProcessNNums * aicNum - nNums;

    tilingData.opInfo.set_usedCoreNum(static_cast<uint32_t>(usedCoreNum));
    tilingData.opInfo.set_formerCoreNum(static_cast<uint32_t>(aicNum - remainCoreNum));
    tilingData.opInfo.set_formerCoreProcessNNum(formerCoreProcessNNums);
    tilingData.opInfo.set_remainCoreProcessNNum(static_cast<uint32_t>(formerCoreProcessNNums - 1));
    tilingData.opInfo.set_castUsedCoreNum(aicoreParams_.numBlocks);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::DoSftTiling()
{
    /*
     * softmax tiling切分策略
     */
    constexpr uint32_t blockFp32 = BLOCK / B32;
    constexpr uint32_t optimizedUbRowSize = 4;
    constexpr uint32_t nonOptimizedUbRowSize = 16;
    constexpr uint32_t optimizedMaxGatherSize = 32;
    constexpr uint32_t nonOptimizedMaxGatherSize = 64;

    const uint32_t sftBaseN = tmpData.singleN;
    const uint64_t availableUbSize = std::min<uint64_t>(KERNEL_UB_SIZE, aicoreParams_.ubSize);
    const uint64_t inputTypeSize = tmpData.queryType == ge::DT_FLOAT ? B32 : B16;
    const uint64_t blockT1 = BLOCK / inputTypeSize;
    const uint64_t dimDAlign = CeilCommon(tmpData.d + tmpData.ropeDim, blockT1) * blockT1;
    const uint64_t dimD2Align = CeilCommon(tmpData.d2, blockT1) * blockT1;

    // Keep this calculation in the same order as VecOp::InitUB. The optimized
    // path persists row parameters; the non-optimized path recalculates one
    // tile and aliases Gather with the Process area.
    auto calcUbUsed = [&](uint32_t baseM) -> uint64_t {
        const uint64_t alignedBaseM = CeilCommon(baseM, blockFp32) * blockFp32;

        // attention/dAttention FP16(or FP32) + FP32, p/dp workspaces and
        // softmax/softmax-grad cast output.
        const uint64_t softmaxWorkSize = static_cast<uint64_t>(baseM) * (2ULL * tmpData.d2 * (inputTypeSize + B32) +
                                                                         sftBaseN * (2ULL * B32 + inputTypeSize));

        if (tmpData.enableOptimizedScatter) {
            // Optimized Process reuses rowSum/max/sum across K blocks, so the
            // complete aligned [G, 8] parameter area is persistent.
            const uint64_t alignedSingleM = CeilCommon(tmpData.singleM, blockFp32) * blockFp32;
            uint64_t persistentSize = 3ULL * alignedSingleM * BLOCK;
            persistentSize += 2ULL * alignedBaseM * B32; // maxTmp and sumTmp
            const uint64_t scatterVStride = tmpData.kvMerge ? dimDAlign : dimD2Align;
            const uint64_t scatterSize = 2ULL * optimizedUbRowSize * (dimDAlign + scatterVStride) * B32;
            const uint64_t scatterTmpSize = (2ULL * optimizedUbRowSize + 1) * (dimDAlign + dimD2Align) * B32;
            const uint64_t gatherSize = 2ULL * optimizedMaxGatherSize *
                                        (std::max<int64_t>(tmpData.d, tmpData.d2) + tmpData.ropeDim) * inputTypeSize;
            return persistentSize + softmaxWorkSize + scatterSize + scatterTmpSize + gatherSize;
        }

        // Non-optimized Process recalculates rowSum/max/sum for every K block,
        // so only one aligned tile is needed. Gather starts after topk indices
        // and aliases the whole Process area, matching the original layout.
        const uint64_t softmaxParamSize = 3ULL * alignedBaseM * BLOCK + 2ULL * alignedBaseM * B32;
        const uint64_t scatterSize = 2ULL * nonOptimizedUbRowSize * (dimDAlign + dimD2Align) * B32;
        const uint64_t gatherSize = 2ULL * nonOptimizedMaxGatherSize * (tmpData.d + tmpData.ropeDim) * inputTypeSize;
        return std::max(gatherSize, softmaxParamSize + std::max(softmaxWorkSize, scatterSize));
    };

    uint32_t sftBaseM = tmpData.singleM;
    // Brcb works on eight FP32 rows per repeat. For G >= 8, keeping baseM a
    // multiple of eight prevents one tile's broadcast from entering the next.
    if (sftBaseM >= blockFp32) {
        sftBaseM = sftBaseM / blockFp32 * blockFp32;
    }
    while (sftBaseM > 0 && calcUbUsed(sftBaseM) > availableUbSize) {
        sftBaseM -= sftBaseM > blockFp32 ? blockFp32 : 1;
    }
    OP_CHECK_IF(
        sftBaseM == 0,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName, "No valid sftBaseM fits UB: ubSize=%lu, G=%u, N=%u, D=%ld, D2=%ld, ropeD=%ld, optimized=%d.",
            availableUbSize, tmpData.singleM, sftBaseN, tmpData.d, tmpData.d2, tmpData.ropeDim,
            static_cast<int32_t>(tmpData.enableOptimizedScatter)),
        return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "SFAG sftBaseM=%u, estimated vector UB=%lu/%lu bytes, optimized=%d.", sftBaseM,
            calcUbUsed(sftBaseM), availableUbSize, static_cast<int32_t>(tmpData.enableOptimizedScatter));

    tilingData.splitCoreParams.set_sftBaseM(sftBaseM);
    tilingData.splitCoreParams.set_sftBaseN(sftBaseN);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::DoCastTiling()
{
    int64_t dAlign = (tilingData.opInfo.get_D() + tilingData.opInfo.get_ropeD() + 15) / 16 * 16;
    int64_t d2Align = (tilingData.opInfo.get_D2() + 15) / 16 * 16;
    // query
    int64_t allNumQuery = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_G() *
                          tilingData.opInfo.get_S1() * dAlign;
    // TND时候要按照真实的query的num数计算
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumQuery = tmpData.t1 * tilingData.opInfo.get_N2() * tilingData.opInfo.get_G() * dAlign;
    }

    // key
    int64_t allNumKey = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_S2() * dAlign;
    // TND时候要按照真实的key的num数计算
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumKey = tmpData.t2 * tilingData.opInfo.get_N2() * 1 * dAlign;
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
    nzReservedSize = dAlign / 16 * blockSize * postNzReservedN;                      // 16为一个单元长度
    postUbBaseSize = (aicoreParams_.ubSize - 2 * nzReservedSize) / curPostCoexNode / // 开DB预留2份nzReservedSize
                     BASE_LEN_256 * BASE_LEN_256;
    qPostBaseNum = postUbBaseSize / typeSize / dAlign * (tilingData.opInfo.get_D() + tilingData.opInfo.get_ropeD());

    OP_CHECK_IF(qPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "qPostBaseNum is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "castUsedCoreNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t qPostBlockTotal = allNumQuery / dAlign * (tilingData.opInfo.get_D() + tilingData.opInfo.get_ropeD());
    int64_t qSizeAlign = (qPostBlockTotal + BASE_LEN_256 - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t kPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(kPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "kPostBaseNum is 0."), return ge::GRAPH_FAILED);
    int64_t kPostBlockTotal = allNumKey / dAlign * (tilingData.opInfo.get_D() + tilingData.opInfo.get_ropeD());
    int64_t kSizeAlign = (kPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t kPostTailNumTmp = kPostBlockTotal % kPostBaseNum;
    int64_t kPostTailNum = kPostTailNumTmp == 0 ? kPostBaseNum : kPostTailNumTmp;
    int64_t kPostBlockOuterTotal = (kPostBlockTotal + kPostBaseNum - 1) / kPostBaseNum;
    int64_t kPostBlockFactor = (kPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t vPostBaseNum = postUbBaseSize / typeSize / d2Align * tilingData.opInfo.get_D2();
    OP_CHECK_IF(vPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "vPostBaseNum is 0."), return ge::GRAPH_FAILED);
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

ge::graphStatus SparseFlashAttentionGradBasicTiling::GetBaseShapeInfo()
{
    OP_CHECK_IF(((context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY)) == nullptr) ||
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::KEY)) == nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "InputShape of query, key or value is nullptr."),
                return ge::GRAPH_FAILED);
    // input
    // TND: query [t1, n1, d]   k [t2, n2, d]  v [t2, n2, d2]   dy/attentionIn [t1, n1, d2]
    // BSND: query [b, s1, n1, d]   k [b, s2, n2, d]  v [b, s2, n2, d2]   dy/attentionIn [b, s1, n1, d2]
    const gert::Shape &queryShape = context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY))->GetStorageShape();
    const gert::Shape &keyShape = context_->GetInputShape(static_cast<size_t>(InputIndex::KEY))->GetStorageShape();
    const gert::StorageShape *valueStorageShape =
        context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::VALUE));
    tmpData.kvMerge = (valueStorageShape == nullptr);
    const gert::Shape &valueShape = (valueStorageShape == nullptr) ? keyShape : valueStorageShape->GetStorageShape();
    const gert::Shape &indicesShape =
        context_->GetInputShape(static_cast<size_t>(InputIndex::TOPK_INDICES))->GetStorageShape();
    auto qRopeTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::Q_ROPE));
    auto kRopeTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::K_ROPE));
    uint32_t dimSize = queryShape.GetDimNum();
    int64_t dimDq = queryShape.GetDim(dimSize - 1);
    int64_t dimDk = keyShape.GetDim(dimSize - 1);
    int64_t dimDv = valueShape.GetDim(dimSize - 1);

    // attrs
    const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    auto selected_block_count = indicesShape.GetDim(dimSize - 1);

    auto selected_block_size =
        *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_SIZE));
    auto sparse_mode = *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SPARSE_MODE));
    if (sparse_mode == 0) {
        tmpData.attenEnable = false;
    } else if (sparse_mode == 3) {
        OP_LOGI(context_, "SparseFlashAttentionGrad AttenMask enable.");
        tmpData.attenEnable = true;
    } else {
        OP_LOGE(context_, "SparseFlashAttentionGrad only support sparse_mode=0 or 3, now sparse_mode=%d.", sparse_mode);
        return ge::GRAPH_FAILED;
    }

    if (dimDq != dimDk) {
        OP_LOGE(context_, "head_dim of Query[%ld] should be equal to head_dim of Key[%ld].", dimDq, dimDk);
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

    if (qRopeTensor != nullptr && kRopeTensor != nullptr) {
        OP_LOGD(context_, "SparseFlashAttentionGrad qRope and kRope is not nullptr, rope is enabled.");
        tmpData.ropeEnable = true;
        const gert::Shape &qRopeShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::Q_ROPE))->GetStorageShape();
        const gert::Shape &kRopeShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::K_ROPE))->GetStorageShape();
        auto qRopeDim = qRopeShape.GetDim(dimSize - 1);
        auto kRopeDim = kRopeShape.GetDim(dimSize - 1);
        if (qRopeDim != kRopeDim) {
            OP_LOGE(context_, "SparseFlashAttentionGrad headDim of qRope and kRope should be equal.");
            return ge::GRAPH_FAILED;
        }
        tmpData.ropeDim = kRopeDim;
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
        const gert::Shape &actSeqQLenShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_Q_LEN))->GetStorageShape();
        tmpData.b = actSeqQLenShape.GetDim(DIM_0);
        OP_CHECK_IF(tmpData.b == 0, OP_LOGE(context_, "batchNum is 0"), return ge::GRAPH_FAILED);
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
        OP_CHECK_IF(tmpData.b == 0, OP_LOGE(context_, "batchNum is 0"), return ge::GRAPH_FAILED);
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
    if (tmpData.n2 != 1 || n1 > 128 || (n1 & (n1 - 1)) != 0) {
        OP_LOGE(context_,
                "SparseFlashAttentionGrad only support n2=1 and n1=1/2/4/8/16/32/64/128, but got n2=%ld n1=%ld.",
                tmpData.n2, n1);
        return ge::GRAPH_FAILED;
    }

    tilingData.opInfo.set_B(tmpData.b);
    tilingData.opInfo.set_G(tmpData.g);
    OP_CHECK_IF(tilingData.opInfo.get_G() == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);
    tilingData.opInfo.set_N2(tmpData.n2);
    tilingData.opInfo.set_S1(tmpData.s1);
    tilingData.opInfo.set_S2(tmpData.s2);
    tilingData.opInfo.set_D(dimDq);
    tilingData.opInfo.set_D2(dimDv);
    tilingData.opInfo.set_ropeD(tmpData.ropeDim);
    tilingData.opInfo.set_layout(tmpData.layout);
    tilingData.opInfo.set_scaleValue(
        *context_->GetAttrs()->GetAttrPointer<float>(static_cast<size_t>(AttrIndex::SCALE_VALUE)));
    tilingData.opInfo.set_selectedBlockCount(selected_block_count);
    tilingData.opInfo.set_selectedBlockSize(selected_block_size);
    bool deterministic = (context_->GetDeterministic() == 1);
    tilingData.opInfo.set_deterministic(deterministic);
    tilingData.opInfo.set_kvMerge(tmpData.kvMerge ? 1U : 0U);

    tmpData.d = tilingData.opInfo.get_D();
    tmpData.d2 = tilingData.opInfo.get_D2();
    tmpData.dataTypeSize = B32;
    tmpData.queryType =
        static_cast<uint32_t>(context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType());
    tmpData.selected_block_count = selected_block_count;
    tmpData.selected_block_size = selected_block_size;
    tmpData.deterministic = deterministic;
    auto gateDecision = GetOptimizedScatterGateDecision(tmpData);
    tmpData.enableOptimizedScatter = gateDecision.enabled;
    tilingData.opInfo.set_enableOptimizedScatter(tmpData.enableOptimizedScatter);
    OP_LOGI(context_->GetNodeName(),
            "SFAG optimized scatter gate: enable=%d reason=%s layout=%u B=%ld N1=%ld N2=%ld G=%ld "
            "S1=%ld S2=%ld gateS1=%ld gateS2=%ld selectedBlockCount=%u selectedBlockSize=%u "
            "effectiveSelectedTokens=%ld D=%ld D2=%ld ropeD=%ld dtype=%u deterministic=%d attenEnable=%d",
            static_cast<int32_t>(tmpData.enableOptimizedScatter), gateDecision.reason, tmpData.layout, tmpData.b,
            gateDecision.n1, tmpData.n2, tmpData.g, tmpData.s1, tmpData.s2, gateDecision.gateS1, gateDecision.gateS2,
            tmpData.selected_block_count, tmpData.selected_block_size, gateDecision.effectiveSelectedTokens, tmpData.d,
            tmpData.d2, tmpData.ropeDim, tmpData.queryType, static_cast<int32_t>(tmpData.deterministic),
            static_cast<int32_t>(tmpData.attenEnable));

    auto ret = CheckDtypeValid(context_);

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SparseFlashAttentionGrad the dtype of input is invalid.");
        return ge::GRAPH_FAILED;
    }

    if (tmpData.layout == static_cast<uint32_t>(InputLayout::TND)) {
        ret = CheckTndShapeValid(context_, tmpData.t1, tmpData.n2 * tmpData.g, tmpData.d, tmpData.d2, tmpData.n2);
    }

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SparseFlashAttentionGrad the input shpae of TND Layout is invalid.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(SparseFlashAttentionGrad, SparseFlashAttentionGradBasicTiling,
                                   std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 1);

} // namespace sfag
} // namespace optiling
