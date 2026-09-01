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
 * \file block_attn_res_prepare_mix_tiling.cpp
 * \brief Mixed Cube/Vector tiling for BlockAttnResPrepare.
 */

#include "block_attn_res_prepare_mix_tiling.h"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "log/log.h"

namespace {

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = 0;
} // namespace

namespace optiling {
namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t UB_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint64_t L1_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint32_t FP32_REG_ELEMS = 64U;
constexpr uint32_t CUBE_BLOCK = 16U;
constexpr uint32_t CUBE_MAX_BASE_S = 32U;
constexpr uint32_t CUBE_MIN_T = 32U;
constexpr uint32_t CUBE_MIN_S = 16U;
constexpr uint32_t CUBE_MIN_D = 256U;
constexpr uint32_t STAT_VECTOR_COUNT = 2U;
constexpr uint64_t DOUBLE_BUFFER_NUM = 2UL;
constexpr uint64_t MAX_RUNTIME_BASE_T = 16UL;
constexpr uint64_t L1_SLOT_NUM = 4UL;
constexpr uint64_t CUBE_BASE_S_LARGE = 128UL;
constexpr uint64_t CUBE_BASE_S_MEDIUM = 64UL;
constexpr uint64_t CUBE_BASE_T_LARGE = 8UL;
constexpr uint64_t CUBE_BASE_T_MEDIUM = 4UL;
constexpr uint64_t CUBE_BASE_T_SMALL = 2UL;
constexpr uint64_t CUBE_BASE_D_LARGE = 512UL;
constexpr uint64_t CUBE_BASE_D_MEDIUM = 256UL;
constexpr uint64_t CUBE_BASE_D_SMALL = 128UL;
constexpr uint64_t CUBE_MM1_MAX_BASE_K = 64UL;
constexpr size_t CUBE_BASE_S_CANDIDATE_NUM = 5;
constexpr size_t CUBE_BASE_T_CANDIDATE_NUM = 4;
constexpr size_t CUBE_BASE_D_CANDIDATE_NUM = 3;

} // namespace

struct BlockAttnResPrepareMixTiling::TileCandidate {
    uint64_t baseT;
    uint64_t baseS;
    uint64_t baseD;
    uint64_t baseDAlign;
    uint32_t sAlign;
    uint64_t dTileNum;
    uint64_t mm1NAlign;
    uint64_t qL1BufferNum;
    uint64_t vL1BufferNum;
    uint64_t vUbBufferNum;
    uint64_t qL1Elems;
    uint64_t vL1Elems;
    uint64_t eL1Elems;
    uint64_t vUbElems;
    uint64_t dotUbElems;
    uint64_t reduceUbElems;
    uint64_t softmaxUbElems;
};

struct BlockAttnResPrepareMixTiling::ResourceUsage {
    uint64_t l1Bytes;
    uint64_t ubBytes;
    uint64_t mm1L1SlotBytes;
    uint64_t mm2L1SlotBytes;
    uint64_t mm1L0ABytes;
    uint64_t mm2L0ABytes;
    uint64_t mm1L0BBytes;
    uint64_t mm2L0BBytes;
    uint64_t mm1L0CBytes;
    uint64_t mm2L0CBytes;
};

bool BlockAttnResPrepareMixTiling::IsCapable()
{
    if (aicCoreNum_ == 0U || aivCoreNum_ != aicCoreNum_ * AIV_CORE_NUM_PER_AIC) {
        return false;
    }
    if (totalT_ < CUBE_MIN_T || totalS_ < CUBE_MIN_S || totalD_ < CUBE_MIN_D) {
        return false;
    }
    // MM2 Fixpipe writes a strided [S, D] tile directly into O[S, T, D]. The destination row stride is T * D
    // elements and is encoded by the hardware interface as uint32_t.
    if (totalT_ > std::numeric_limits<uint32_t>::max() / totalD_) {
        return false;
    }
    return SelectMixTileShape();
}

bool BlockAttnResPrepareMixTiling::SelectMixTileShape()
{
    const uint64_t usableUbBytes = ubSize_ - UB_RESERVED_BYTES;
    const uint64_t usableL1Bytes = l1Size_ - L1_RESERVED_BYTES;
    const uint64_t mixedCoreNum = std::min<uint64_t>(aicCoreNum_, aivCoreNum_ / AIV_CORE_NUM_PER_AIC);
    if (mixedCoreNum == 0U) {
        return false;
    }
    tilingData_.nAlign = static_cast<uint32_t>(AlignUp<uint64_t>(totalN_, CUBE_BLOCK));
    tilingData_.dAlign = static_cast<uint32_t>(AlignUp<uint64_t>(totalD_, CUBE_BLOCK));

    // Pack adjacent token matrices into MM1's N dimension until one 16-column Cube tile is filled. The
    // load-balance guard keeps the maximum number of token/S tiles assigned to one core no larger than baseT=1,
    // because MM2 and its AIV epilogue still execute once per token.
    const std::array<uint64_t, CUBE_BASE_S_CANDIDATE_NUM> sCandidates = {
        totalS_, std::min<uint64_t>(totalS_, CUBE_BASE_S_LARGE), std::min<uint64_t>(totalS_, CUBE_BASE_S_MEDIUM),
        std::min<uint64_t>(totalS_, CUBE_MAX_BASE_S), std::min<uint64_t>(totalS_, CUBE_BLOCK)};
    uint64_t previousS = 0U;
    for (const uint64_t sCandidate : sCandidates) {
        if (sCandidate == 0U || sCandidate == previousS) {
            continue;
        }
        previousS = sCandidate;
        if (TrySelectTileForS(sCandidate, mixedCoreNum, usableL1Bytes, usableUbBytes)) {
            return true;
        }
    }
    return false;
}

bool BlockAttnResPrepareMixTiling::TrySelectTileForS(uint64_t sCandidate, uint64_t mixedCoreNum, uint64_t usableL1Bytes,
                                                     uint64_t usableUbBytes)
{
    const std::array<uint64_t, CUBE_BASE_T_CANDIDATE_NUM> tCandidates = {CUBE_BASE_T_LARGE, CUBE_BASE_T_MEDIUM,
                                                                         CUBE_BASE_T_SMALL, 1UL};
    const uint64_t candidateSTileNum = CeilDiv(totalS_, sCandidate);
    const uint64_t baselineWorkUnits = totalT_ * candidateSTileNum;
    const uint64_t baselineUsedCores = std::min<uint64_t>(mixedCoreNum, baselineWorkUnits);
    const uint64_t baselineTokenWork = CeilDiv(baselineWorkUnits, baselineUsedCores);
    for (const uint64_t candidateT : tCandidates) {
        if (candidateT > totalT_ || candidateT * totalN_ > CUBE_BLOCK) {
            continue;
        }
        const uint64_t candidateTGroups = CeilDiv(totalT_, candidateT);
        const uint64_t groupedWorkUnits = candidateTGroups * candidateSTileNum;
        const uint64_t groupedUsedCores = std::min<uint64_t>(mixedCoreNum, groupedWorkUnits);
        const uint64_t groupedTokenWork = CeilDiv(groupedWorkUnits, groupedUsedCores) * candidateT;
        if (groupedTokenWork <= baselineTokenWork &&
            TrySelectTileForT(sCandidate, candidateT, usableL1Bytes, usableUbBytes)) {
            return true;
        }
    }
    return false;
}

bool BlockAttnResPrepareMixTiling::TrySelectTileForT(uint64_t sCandidate, uint64_t candidateT, uint64_t usableL1Bytes,
                                                     uint64_t usableUbBytes)
{
    const std::array<uint64_t, CUBE_BASE_D_CANDIDATE_NUM> dCandidates = {
        std::min<uint64_t>(totalD_, CUBE_BASE_D_LARGE), std::min<uint64_t>(totalD_, CUBE_BASE_D_MEDIUM),
        std::min<uint64_t>(totalD_, CUBE_BASE_D_SMALL)};
    uint64_t previousD = 0U;
    for (const uint64_t candidateD : dCandidates) {
        if (candidateD == 0U || candidateD == previousD) {
            continue;
        }
        previousD = candidateD;
        const TileCandidate candidate = BuildTileCandidate(sCandidate, candidateT, candidateD);
        const ResourceUsage usage = CalculateResourceUsage(candidate);
        if (DoesCandidateFit(usage, usableL1Bytes, usableUbBytes)) {
            ApplyTileCandidate(candidate);
            return true;
        }
    }
    return false;
}

BlockAttnResPrepareMixTiling::TileCandidate BlockAttnResPrepareMixTiling::BuildTileCandidate(uint64_t sCandidate,
                                                                                             uint64_t candidateT,
                                                                                             uint64_t candidateD) const
{
    TileCandidate candidate{};
    candidate.baseT = candidateT;
    candidate.baseS = sCandidate;
    candidate.baseD = candidateD;
    candidate.baseDAlign = AlignUp<uint64_t>(candidateD, CUBE_BLOCK);
    candidate.sAlign = static_cast<uint32_t>(AlignUp<uint64_t>(sCandidate, CUBE_BLOCK));
    candidate.dTileNum = CeilDiv(totalD_, candidateD);
    candidate.mm1NAlign = AlignUp<uint64_t>(candidateT * totalN_, CUBE_BLOCK);
    candidate.qL1BufferNum = candidate.dTileNum > 1U ? DOUBLE_BUFFER_NUM : 1U;
    candidate.vL1BufferNum = candidate.dTileNum > 1U ? DOUBLE_BUFFER_NUM : 1U;
    candidate.vUbBufferNum = DOUBLE_BUFFER_NUM;
    candidate.qL1Elems = static_cast<uint64_t>(candidate.sAlign) * candidate.baseDAlign;
    candidate.vL1Elems = candidate.mm1NAlign * candidate.baseDAlign;
    candidate.eL1Elems = static_cast<uint64_t>(candidate.sAlign) * tilingData_.nAlign;
    candidate.vUbElems = static_cast<uint64_t>(tilingData_.nAlign) * candidate.baseDAlign;
    candidate.dotUbElems = static_cast<uint64_t>(candidate.sAlign) * tilingData_.nAlign + FP32_REG_ELEMS;
    candidate.reduceUbElems = FP32_REG_ELEMS;
    candidate.softmaxUbElems = static_cast<uint64_t>(STAT_VECTOR_COUNT) * candidate.sAlign;
    return candidate;
}

BlockAttnResPrepareMixTiling::ResourceUsage BlockAttnResPrepareMixTiling::CalculateResourceUsage(
    const TileCandidate &candidate) const
{
    ResourceUsage usage{};
    usage.l1Bytes = (candidate.qL1BufferNum * candidate.qL1Elems + candidate.vL1BufferNum * candidate.vL1Elems +
                     candidate.eL1Elems) *
                    FP32_BYTES;
    usage.ubBytes = (candidate.vUbBufferNum * candidate.vUbElems + candidate.dotUbElems + candidate.reduceUbElems +
                     candidate.softmaxUbElems) *
                    FP32_BYTES;
    const uint64_t mm1BaseK = std::min<uint64_t>(candidate.baseDAlign, CUBE_MM1_MAX_BASE_K);
    usage.mm1L0ABytes = static_cast<uint64_t>(candidate.sAlign) * mm1BaseK * FP32_BYTES;
    usage.mm2L0ABytes = static_cast<uint64_t>(candidate.sAlign) * tilingData_.nAlign * FP32_BYTES;
    usage.mm1L0BBytes = candidate.mm1NAlign * mm1BaseK * FP32_BYTES;
    usage.mm2L0BBytes = static_cast<uint64_t>(tilingData_.nAlign) * candidate.baseDAlign * FP32_BYTES;
    usage.mm1L0CBytes = static_cast<uint64_t>(candidate.sAlign) * candidate.mm1NAlign * FP32_BYTES;
    usage.mm2L0CBytes = static_cast<uint64_t>(candidate.sAlign) * candidate.baseDAlign * FP32_BYTES;
    usage.mm1L1SlotBytes =
        (static_cast<uint64_t>(candidate.sAlign) + candidate.mm1NAlign) * candidate.baseDAlign * FP32_BYTES;
    usage.mm2L1SlotBytes = (static_cast<uint64_t>(candidate.sAlign) * tilingData_.nAlign +
                            static_cast<uint64_t>(tilingData_.nAlign) * candidate.baseDAlign) *
                           FP32_BYTES;
    return usage;
}

bool BlockAttnResPrepareMixTiling::DoesCandidateFit(const ResourceUsage &usage, uint64_t usableL1Bytes,
                                                    uint64_t usableUbBytes) const
{
    return usage.l1Bytes <= usableL1Bytes && usage.ubBytes <= usableUbBytes &&
           usage.mm1L1SlotBytes <= l1Size_ / L1_SLOT_NUM && usage.mm2L1SlotBytes <= l1Size_ / L1_SLOT_NUM &&
           usage.mm1L0ABytes <= l0ASize_ && usage.mm2L0ABytes <= l0ASize_ && usage.mm1L0BBytes <= l0BSize_ &&
           usage.mm2L0BBytes <= l0BSize_ && usage.mm1L0CBytes <= l0CSize_ && usage.mm2L0CBytes <= l0CSize_;
}

void BlockAttnResPrepareMixTiling::ApplyTileCandidate(const TileCandidate &candidate)
{
    tilingData_.baseT = static_cast<uint32_t>(candidate.baseT);
    tilingData_.baseS = static_cast<uint32_t>(candidate.baseS);
    tilingData_.baseD = static_cast<uint32_t>(candidate.baseD);
    tilingData_.baseDAlign = static_cast<uint32_t>(candidate.baseDAlign);
    tilingData_.sAlign = candidate.sAlign;
    tilingData_.dTileNum = static_cast<uint32_t>(candidate.dTileNum);
    tilingData_.mm1NAlign = static_cast<uint32_t>(candidate.mm1NAlign);
    tilingData_.qL1BufferNum = static_cast<uint8_t>(candidate.qL1BufferNum);
    tilingData_.vL1BufferNum = static_cast<uint8_t>(candidate.vL1BufferNum);
    tilingData_.vUbBufferNum = static_cast<uint8_t>(candidate.vUbBufferNum);
    tilingData_.qL1Elems = candidate.qL1Elems;
    tilingData_.vL1Elems = candidate.vL1Elems;
    tilingData_.eL1Elems = candidate.eL1Elems;
    tilingData_.vUbElems = candidate.vUbElems;
    tilingData_.dotUbElems = candidate.dotUbElems;
    tilingData_.reduceUbElems = candidate.reduceUbElems;
    tilingData_.softmaxUbElems = candidate.softmaxUbElems;
}

ge::graphStatus BlockAttnResPrepareMixTiling::DoOpTiling()
{
    OP_CHECK_IF(tilingData_.baseT == 0U || tilingData_.baseD == 0U,
                OP_LOGE(context_->GetNodeName(),
                        "Mix tile was not selected: T=%lu, N=%lu, S=%lu, D=%lu, baseT=%u, baseS=%u, baseD=%u", totalT_,
                        totalN_, totalS_, totalD_, tilingData_.baseT, tilingData_.baseS, tilingData_.baseD),
                return ge::GRAPH_FAILED);

    uint64_t totalWorkUnits = 0U;
    if (CalculateMixWork(totalWorkUnits) != ge::GRAPH_SUCCESS || CalculateWorkspaceSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    FillTilingData(totalWorkUnits);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareMixTiling::CalculateMixWork(uint64_t &totalWorkUnits)
{
    tilingData_.sTileNum = static_cast<uint32_t>(CeilDiv<uint64_t>(totalS_, tilingData_.baseS));
    const uint64_t tTileNum = CeilDiv<uint64_t>(totalT_, tilingData_.baseT);
    OP_CHECK_IF(tTileNum > std::numeric_limits<uint64_t>::max() / tilingData_.sTileNum,
                OP_LOGE(context_->GetNodeName(), "Mix work count overflows uint64: tTileNum=%lu, sTileNum=%u", tTileNum,
                        tilingData_.sTileNum),
                return ge::GRAPH_FAILED);
    totalWorkUnits = tTileNum * tilingData_.sTileNum;
    const uint64_t mixedCoreNum = std::min<uint64_t>(aicCoreNum_, aivCoreNum_ / AIV_CORE_NUM_PER_AIC);
    WorkDistribution distribution;
    if (CalculateWorkDistribution(totalWorkUnits, mixedCoreNum, "Mix", distribution) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = distribution.usedCoreNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareMixTiling::CalculateWorkspaceSize()
{
    // MM1 writes rows with mm1NAlign pitch, which also reserves capacity for runtime token grouping. AIV and AIC use
    // one E slot per possible runtime token group, because the kernel may expand baseT after reading validN.
    // MM2 Fixpipe writes O directly and therefore needs no output workspace.
    const uint64_t dotWorkspaceElems = static_cast<uint64_t>(tilingData_.baseS) * tilingData_.mm1NAlign;
    const uint64_t eWorkspaceElems = static_cast<uint64_t>(tilingData_.sAlign) * tilingData_.nAlign;
    const uint64_t eWorkspaceBufferNum = std::min<uint64_t>(totalT_, MAX_RUNTIME_BASE_T);
    OP_CHECK_IF(eWorkspaceElems > std::numeric_limits<uint64_t>::max() / eWorkspaceBufferNum ||
                    dotWorkspaceElems > std::numeric_limits<uint64_t>::max() - eWorkspaceBufferNum * eWorkspaceElems,
                OP_LOGE(context_->GetNodeName(),
                        "Mix per-core workspace size overflows uint64: dotWorkspaceElems=%lu, "
                        "eWorkspaceElems=%lu, eWorkspaceBufferNum=%lu",
                        dotWorkspaceElems, eWorkspaceElems, eWorkspaceBufferNum),
                return ge::GRAPH_FAILED);
    const uint64_t workspacePerCoreElems = dotWorkspaceElems + eWorkspaceBufferNum * eWorkspaceElems;
    OP_CHECK_IF(workspacePerCoreElems > std::numeric_limits<uint64_t>::max() / usedCoreNum_ / FP32_BYTES,
                OP_LOGE(context_->GetNodeName(),
                        "Mix workspace size overflows uint64: workspacePerCoreElems=%lu, usedCoreNum=%u, "
                        "elementBytes=%lu",
                        workspacePerCoreElems, usedCoreNum_, FP32_BYTES),
                return ge::GRAPH_FAILED);
    const uint64_t mixWorkspaceSize = workspacePerCoreElems * usedCoreNum_ * FP32_BYTES;
    OP_CHECK_IF(systemWorkspaceSize_ > std::numeric_limits<uint64_t>::max() - mixWorkspaceSize,
                OP_LOGE(context_->GetNodeName(),
                        "Mix total workspace size overflows uint64: systemWorkspaceSize=%lu, mixWorkspaceSize=%lu",
                        systemWorkspaceSize_, mixWorkspaceSize),
                return ge::GRAPH_FAILED);
    workspaceSize_ = systemWorkspaceSize_ + mixWorkspaceSize;
    tilingData_.workspacePerCoreElems = workspacePerCoreElems;
    return ge::GRAPH_SUCCESS;
}

void BlockAttnResPrepareMixTiling::FillTilingData(uint64_t totalWorkUnits)
{
    tilingData_.totalT = static_cast<uint32_t>(totalT_);
    tilingData_.totalN = static_cast<uint8_t>(totalN_);
    tilingData_.totalS = static_cast<uint32_t>(totalS_);
    tilingData_.totalWorkUnits = static_cast<uint32_t>(totalWorkUnits);
    tilingData_.totalD = static_cast<uint32_t>(totalD_);
    tilingData_.usedCoreNum = static_cast<uint16_t>(usedCoreNum_);
    tilingData_.aicCoreNum = static_cast<uint16_t>(aicCoreNum_);
    tilingData_.aivCoreNum = static_cast<uint16_t>(aivCoreNum_);
    tilingData_.eps = eps_;
}

uint64_t BlockAttnResPrepareMixTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(BLOCK_ATTN_RES_PREPARE_TPL_MIX);
}

BlockAttnResPrepareBaseTiling::TilingDataView BlockAttnResPrepareMixTiling::GetTilingDataView() const
{
    return {&tilingData_, sizeof(tilingData_), "Mix"};
}

void BlockAttnResPrepareMixTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(),
            "BlockAttnResPrepare Mix tiling: T=%lu N=%lu S=%lu D=%lu cores=%u baseS=%u baseD=%u "
            "baseT=%u sLoops=%u dLoops=%u qL1/vL1/vUb buffers=%u/%u/%u nAlign/mm1NAlign=%u/%u "
            "workspacePerCore=%lu workspace=%lu",
            totalT_, totalN_, totalS_, totalD_, usedCoreNum_, tilingData_.baseS, tilingData_.baseD, tilingData_.baseT,
            tilingData_.sTileNum, tilingData_.dTileNum, tilingData_.qL1BufferNum, tilingData_.vL1BufferNum,
            tilingData_.vUbBufferNum, tilingData_.nAlign, tilingData_.mm1NAlign, tilingData_.workspacePerCoreElems,
            workspaceSize_);
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockAttnResPrepare, BlockAttnResPrepareMixTiling, supportedNpuArch,
                                   TILING_PRIORITY);

} // namespace optiling
