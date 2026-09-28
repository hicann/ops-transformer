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
 * \file sparse_flash_attention_grad_tiling_bs1_basic.cpp
 * \brief
 */

#include "sparse_flash_attention_grad_tiling_bs1_basic.h"
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

constexpr uint32_t KSPLIT_COMPUTE_CORE_NUM = 22;
constexpr uint32_t KSPLIT_CHUNK_PER_CORE = 96;
constexpr uint32_t KSPLIT_LAST_CHUNK = 32;
constexpr uint32_t KSPLIT_USED_CORE_NUM = 24;
// Scatter retires two windows after gather; partials use five slots.
constexpr uint32_t KSPLIT_SCATTER_SLOT_NUM = 3;
constexpr uint32_t KSPLIT_PARTIAL_SLOT_NUM = 5;

// K-split is the only dispatch path. These are structural constraints, not
// performance gates: neither deterministic nor the S2/S1 ratio selects a fallback.
bool IsKsplitShapeSupported(const TempParams &params)
{
    return params.layout == static_cast<uint32_t>(InputLayout::TND) && params.n2 == 1 &&
           params.selected_block_size == 1 && params.selected_block_count == 2048 && params.d2 == 512 &&
           ((params.d == 512 && params.ropeDim == 64) || (params.d == 576 && params.ropeDim == 0)) && params.g > 0 &&
           params.g <= 128 && params.g % 16 == 0 &&
           (params.queryType == static_cast<uint32_t>(ge::DT_FLOAT16) ||
            params.queryType == static_cast<uint32_t>(ge::DT_BF16));
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::GetShapeAttrsInfo()
{
    /*
    Get all shape info and attr
    */
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("SparseFlashAttentionGrad", "context is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetAttrs() == nullptr, OP_LOGE(context_->GetNodeName(), "GetAttrs is nullptr."),
                return ge::GRAPH_FAILED);

    auto status = GetBaseShapeInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    OP_LOGI(context_->GetNodeName(),
            "SparseFlashAttentionGrad with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] d2[%ld]!",
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
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile_info is null."),
                    return ge::GRAPH_FAILED);
        aicoreParams_.blockDim = compileInfoPtr->aivNum;
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
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }

    OP_CHECK_IF((aicoreParams_.blockDim == 0) || (aicoreParams_.aicNum == 0),
                OP_LOGE(context_->GetNodeName(), "num of coreNum(aivNum) is %lu, num of aicNum is %lu.",
                        aicoreParams_.blockDim, aicoreParams_.aicNum),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(aicoreParams_.ubSize <= 0 || l2CacheSize <= 0,
                OP_LOGE(context_->GetNodeName(), "ubSize or l2CacheSize is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool SparseFlashAttentionGradBasicTiling::IsCapable()
{
    OP_LOGI(context_->GetNodeName(), "SparseFlashAttentionGrad basic template hit.");
    return true;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::DoOpTiling()
{
    OP_LOGI(context_->GetNodeName(), "SparseFlashAttentionGrad DoTiling start");

    // Init
    tmpData.singleM = tilingData.opInfo.get_G();
    tmpData.singleN = 128; // TODO

    // setTilingData
    tilingData.splitCoreParams.set_singleM(tmpData.singleM);
    tilingData.splitCoreParams.set_singleN(tmpData.singleN);

    tilingData.opInfo.set_kSplitChunkPerCore(KSPLIT_CHUNK_PER_CORE);
    tilingData.opInfo.set_kSplitLastChunk(KSPLIT_LAST_CHUNK);
    tilingData.opInfo.set_dqPartialWorkspaceOffset(0);
    tilingData.opInfo.set_dqPartialWorkspaceLen(0);

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
    int64_t inputDtypeSize = tmpData.queryType == ge::DT_FLOAT ? B32 : B16;
    // int64_t selectedS2 = tmpData.selected_block_count * tmpData.selected_block_size;
    // 每轮做singleN
    int64_t selectedS2 = tmpData.singleN;

    // MIX_AIC_1_2: 24 AIC blocks launch exactly 48 AIV participants.
    // DoBlockTiling checks that the device supplies at least this many cores.
    context_->SetBlockDim(KSPLIT_USED_CORE_NUM);

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
    // gather ws
    workspaces[0] += selectedKWorkspaceLen * currentUseCoreNum;
    workspaces[0] += mm12WorkspaceLen * 4 * currentUseCoreNum;
    workspaces[0] += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    int64_t dAlign = (tilingData.opInfo.get_D() + tilingData.opInfo.get_ropeD() + 15) / 16 * 16;
    int64_t d2Align = (tilingData.opInfo.get_D2() + 15) / 16 * 16;
    const uint32_t scatterBufferNum = KSPLIT_SCATTER_SLOT_NUM;
    const int64_t scatterTokenCapacity =
        static_cast<int64_t>(tmpData.selected_block_count) * tmpData.selected_block_size;
    workspaces[0] += 24 * scatterBufferNum * scatterTokenCapacity * (dAlign + d2Align) * B32;

    // Append dq partial storage after all three mm4/mm5 scatter slots.
    {
        int64_t dqPartialOffset = selectedKWorkspaceLen * currentUseCoreNum + mm12WorkspaceLen * 4 * currentUseCoreNum +
                                  dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen +
                                  24 * scatterBufferNum * scatterTokenCapacity * (dAlign + d2Align) * B32;
        dqPartialOffset = AlignData(dqPartialOffset, GM_ALIGN);
        int64_t dqPartialLen = static_cast<int64_t>(KSPLIT_COMPUTE_CORE_NUM) * KSPLIT_PARTIAL_SLOT_NUM *
                               tilingData.opInfo.get_G() * (tilingData.opInfo.get_D() + tilingData.opInfo.get_ropeD()) *
                               B32;
        dqPartialLen = AlignData(dqPartialLen, GM_ALIGN);
        workspaces[0] += dqPartialLen;
        tilingData.opInfo.set_dqPartialWorkspaceOffset(dqPartialOffset);
        tilingData.opInfo.set_dqPartialWorkspaceLen(dqPartialLen);
        OP_LOGI(context_->GetNodeName(),
                "SparseFlashAttentionGrad ksplit det workspace: dqPartialOffset=%ld dqPartialLen=%ld "
                "total=%zu.",
                dqPartialOffset, dqPartialLen, workspaces[0]);
    }

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
    OP_CHECK_IF(tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity(),
                OP_LOGE(context_->GetNodeName(),
                        "The size of TilingDataSize[%zu] is larger than the size of MaxDataCapacity[%zu].",
                        tilingData.GetDataSize(), context_->GetRawTilingData()->GetCapacity()),
                return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint64_t SparseFlashAttentionGradBasicTiling::GetTilingKey() const
{
    uint64_t tilingKey = 2000;
    if (tmpData.attenEnable) {
        tilingKey += 100;
    }
    if (tmpData.ropeDim != 0) {
        tilingKey += 10;
    }
    OP_LOGI(context_->GetNodeName(), "SparseFlashAttentionGrad ksplit tilingkey=%lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::DoBlockTiling()
{
    // The kernel has 22 compute pairs plus four reduce AIVs. Launch exactly
    // 24 pairs even on a larger device; fewer pairs cannot implement this schedule.
    OP_CHECK_IF(aicoreParams_.aicNum < KSPLIT_USED_CORE_NUM || aicoreParams_.blockDim < KSPLIT_USED_CORE_NUM * 2,
                OP_LOGE(context_->GetNodeName(), "K-split requires at least 24 AICs and 48 AIVs."),
                return ge::GRAPH_FAILED);
    tilingData.opInfo.set_usedCoreNum(KSPLIT_USED_CORE_NUM);
    tilingData.opInfo.set_formerCoreNum(KSPLIT_USED_CORE_NUM);
    tilingData.opInfo.set_formerCoreProcessNNum(1);
    tilingData.opInfo.set_remainCoreProcessNNum(0);
    tilingData.opInfo.set_castUsedCoreNum(KSPLIT_USED_CORE_NUM * 2);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseFlashAttentionGradBasicTiling::DoSftTiling()
{
    /*
     * softmax tiling切分策略：按 UB 预算反推 sftBaseM，与 VecOp::InitUB 的 buffer 布局保持一致
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
            const uint64_t scatterVStride = dimD2Align;
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
    OP_CHECK_IF(sftBaseM == 0,
                OP_LOGE(context_->GetNodeName(),
                        "No valid sftBaseM fits UB: ubSize=%lu, G=%u, N=%u, D=%ld, D2=%ld, "
                        "ropeD=%ld, optimized=%d.",
                        availableUbSize, tmpData.singleM, sftBaseN, tmpData.d, tmpData.d2, tmpData.ropeDim,
                        static_cast<int32_t>(tmpData.enableOptimizedScatter)),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(),
            "SparseFlashAttentionGrad sftBaseM=%u, estimated vector UB=%lu/%lu bytes, optimized=%d.", sftBaseM,
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

    OP_CHECK_IF(qPostBaseNum == 0, OP_LOGE(context_->GetNodeName(), "qPostBaseNum is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0, OP_LOGE(context_->GetNodeName(), "castUsedCoreNum is 0."), return ge::GRAPH_FAILED);
    int64_t qPostBlockTotal = allNumQuery / dAlign * (tilingData.opInfo.get_D() + tilingData.opInfo.get_ropeD());
    int64_t qSizeAlign = (qPostBlockTotal + BASE_LEN_256 - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t kPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(kPostBaseNum == 0, OP_LOGE(context_->GetNodeName(), "kPostBaseNum is 0."), return ge::GRAPH_FAILED);
    // int64_t kPostBlockTotal = allNumKey / dAlign * tilingData.opInfo.get_D();
    int64_t kPostBlockTotal = allNumKey / dAlign * (tilingData.opInfo.get_D() + tilingData.opInfo.get_ropeD());
    int64_t kSizeAlign = (kPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t kPostTailNumTmp = kPostBlockTotal % kPostBaseNum;
    int64_t kPostTailNum = kPostTailNumTmp == 0 ? kPostBaseNum : kPostTailNumTmp;
    int64_t kPostBlockOuterTotal = (kPostBlockTotal + kPostBaseNum - 1) / kPostBaseNum;
    int64_t kPostBlockFactor = (kPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t vPostBaseNum = postUbBaseSize / typeSize / d2Align * tilingData.opInfo.get_D2();
    OP_CHECK_IF(vPostBaseNum == 0, OP_LOGE(context_->GetNodeName(), "vPostBaseNum is 0."), return ge::GRAPH_FAILED);
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

    // The reduce AIVs write scaled/cast dq directly to the output.
    tilingData.opInfo.set_dqWorkspaceLen(0);
    tilingData.opInfo.set_dkWorkspaceLen((allNumKey * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    // v=k 语义：dv 梯度合入 dk，dv 输出恒 0；AtomicClean 直接清零输出 dvGm，不需要 dv workspace
    tilingData.opInfo.set_dvWorkspaceLen(0);

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
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::KEY)) == nullptr) ||
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::VALUE)) == nullptr) ||
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::TOPK_INDICES)) == nullptr)),
                OP_LOGE(context_->GetNodeName(), "InputShape of query, key, value or indices is nullptr."),
                return ge::GRAPH_FAILED);
    // input
    // TND: query [t1, n1, d]   k [t2, n2, d]  v [t2, n2, d2]   dy/attentionIn [t1, n1, d2]
    const gert::Shape &queryShape = context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY))->GetStorageShape();
    const gert::Shape &keyShape = context_->GetInputShape(static_cast<size_t>(InputIndex::KEY))->GetStorageShape();
    const gert::Shape &valueShape = context_->GetInputShape(static_cast<size_t>(InputIndex::VALUE))->GetStorageShape();
    const gert::Shape &indicesShape =
        context_->GetInputShape(static_cast<size_t>(InputIndex::TOPK_INDICES))->GetStorageShape();
    auto qRopeTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::Q_ROPE));
    auto kRopeTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::K_ROPE));
    uint32_t dimSize = queryShape.GetDimNum();
    OP_CHECK_IF(
        dimSize != 3 || keyShape.GetDimNum() != 3 || valueShape.GetDimNum() != 3 || indicesShape.GetDimNum() != 3,
        OP_LOGE(context_->GetNodeName(), "K-split requires rank-3 TND inputs."), return ge::GRAPH_FAILED);
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
        OP_LOGI(context_->GetNodeName(), "SparseFlashAttentionGrad AttenMask enable.");
        tmpData.attenEnable = true;
    } else {
        OP_LOGE(context_->GetNodeName(),
                "SparseFlashAttentionGrad only support sparse_mode=0 or 3, now sparse_mode=%d.", sparse_mode);
        return ge::GRAPH_FAILED;
    }

    if (dimDq != dimDk) {
        OP_LOGE(context_->GetNodeName(), "head_dim of Query[%ld] should be equal to head_dim of Key[%ld].", dimDq,
                dimDk);
        return ge::GRAPH_FAILED;
    }
    if (dimDq < dimDv) {
        OP_LOGE(context_->GetNodeName(), "head_dim of Query[%ld] can not less than head_dim of Value[%ld].", dimDq,
                dimDv);
        return ge::GRAPH_FAILED;
    }
    if (inputLayout == nullptr || strcmp(inputLayout, TND_STR) != 0) {
        OP_LOGE(context_->GetNodeName(), "K-split only supports TND layout.");
        return ge::GRAPH_FAILED;
    }

    if (qRopeTensor != nullptr && kRopeTensor != nullptr) {
        OP_LOGD(context_->GetNodeName(), "SparseFlashAttentionGrad qRope and kRope is not nullptr, rope is enabled.");
        tmpData.ropeEnable = true;
        const gert::Shape &qRopeShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::Q_ROPE))->GetStorageShape();
        const gert::Shape &kRopeShape =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::K_ROPE))->GetStorageShape();
        auto qRopeDim = qRopeShape.GetDim(dimSize - 1);
        auto kRopeDim = kRopeShape.GetDim(dimSize - 1);
        if (qRopeDim != kRopeDim) {
            OP_LOGE(context_->GetNodeName(), "SparseFlashAttentionGrad headDim of qRope and kRope should be equal.");
            return ge::GRAPH_FAILED;
        }
        tmpData.ropeDim = kRopeDim;
    } else {
        tmpData.ropeEnable = false;
        tmpData.ropeDim = 0;
    }

    auto qSeqShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::CUR_SEQ_Q_LEN));
    auto kSeqShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::CUR_SEQ_KV_LEN));
    OP_CHECK_IF(qSeqShape == nullptr || kSeqShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "K-split requires query and KV cumulative sequence lengths."),
                return ge::GRAPH_FAILED);
    const gert::Shape &qSeq = qSeqShape->GetStorageShape();
    const gert::Shape &kSeq = kSeqShape->GetStorageShape();
    OP_CHECK_IF(qSeq.GetDimNum() != 1 || kSeq.GetDimNum() != 1 || qSeq.GetDim(DIM_0) < 2 ||
                    qSeq.GetDim(DIM_0) != kSeq.GetDim(DIM_0),
                OP_LOGE(context_->GetNodeName(), "Cumulative sequence lengths must have matching B+1 shapes."),
                return ge::GRAPH_FAILED);
    tmpData.b = qSeq.GetDim(DIM_0) - 1;
    tmpData.t1 = queryShape.GetDim(DIM_0);
    tmpData.t2 = keyShape.GetDim(DIM_0);
    tmpData.s1 = tmpData.t1;
    tmpData.s2 = tmpData.t2;
    tmpData.n2 = keyShape.GetDim(DIM_1);
    OP_CHECK_IF(tmpData.n2 != 1 || valueShape.GetDim(DIM_1) != 1 || indicesShape.GetDim(DIM_1) != 1 ||
                    indicesShape.GetDim(DIM_0) != tmpData.t1,
                OP_LOGE(context_->GetNodeName(), "K-split requires N2=1 and one indices row per query."),
                return ge::GRAPH_FAILED);
    tmpData.g = queryShape.GetDim(DIM_1);
    tmpData.layout = static_cast<uint32_t>(InputLayout::TND);

    tilingData.opInfo.set_B(tmpData.b);
    tilingData.opInfo.set_G(tmpData.g);
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

    bool deterministic = *context_->GetAttrs()->GetAttrPointer<bool>(static_cast<size_t>(AttrIndex::DETERMINISTIC));
    deterministic = deterministic || (context_->GetDeterministic() == 1);
    tilingData.opInfo.set_deterministic(deterministic);

    tmpData.d = tilingData.opInfo.get_D();
    tmpData.d2 = tilingData.opInfo.get_D2();
    tmpData.dataTypeSize = B32;
    tmpData.queryType =
        static_cast<uint32_t>(context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType());
    tmpData.selected_block_count = selected_block_count;
    tmpData.selected_block_size = selected_block_size;
    tmpData.deterministic = deterministic;

    OP_CHECK_IF(
        !IsKsplitShapeSupported(tmpData),
        OP_LOGE(context_->GetNodeName(), "K-split requires TND, N2=1, topk=2048, blockSize=1, D2=512, "
                                         "(D,ropeD)=(512,64) or (576,0), G in [16,128] divisible by 16, fp16/bf16."),
        return ge::GRAPH_FAILED);
    // Every accepted input uses the K-split ND scatter/workspace layout,
    // including calls that do not request deterministic execution.
    tmpData.enableOptimizedScatter = false;
    tilingData.opInfo.set_enableOptimizedScatter(false);

    auto ret = CheckDtypeValid(context_);

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SparseFlashAttentionGrad the dtype of input is invalid.");
        return ge::GRAPH_FAILED;
    }

    if (tmpData.layout == static_cast<uint32_t>(InputLayout::TND)) {
        ret = CheckTndShapeValid(context_, tmpData.t1, tmpData.n2 * tmpData.g, tmpData.d, tmpData.d2, tmpData.n2);
    }

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SparseFlashAttentionGrad the input shpae of TND Layout is invalid.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// REGISTER_TILING_TEMPLATE("SparseFlashAttentionGrad", SparseFlashAttentionGradBasicTiling, 1);

} // namespace sfag
} // namespace optiling
