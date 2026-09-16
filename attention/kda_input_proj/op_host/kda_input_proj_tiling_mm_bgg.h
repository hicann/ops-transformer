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
 * \file kda_input_proj_tiling_mm_bgg.h
 * \brief Stage1 AIC Matmul(beta/gate/g)：DAV_3510 上沿用 MatMulV3 BASIC_ASWT 的
 *        ResetBase(256×256) + GetRebalanceBlock 选共用 (tTile,nTile)；
 *        负载按三路逻辑网格 tDim×nCntAll 计。Kernel 对任务池 round-robin。
 *        usedCore = min(aicNum, tileNumAll)，块数不够时允许不满核。
 *        L1 容量按 Blaze BlockMmad Basic slot 校验：l1Stages=2 时
 *        |APing,BPing,Bias| / |APong,BPong,Bias| 各占 L1/2。
 */

#ifndef KDA_INPUT_PROJ_TILING_MM_BGG_H
#define KDA_INPUT_PROJ_TILING_MM_BGG_H

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>

#include "kda_input_proj_tiling.h"
#include "platform/platform_infos_def.h"

namespace optiling {
class KdaInputProjMmBggTiling {
public:
    KdaInputProjMmBggTiling(const KdaInputProjTilingInfo &tilingInfo, gert::TilingContext *context)
        : tilingInfo_(tilingInfo),
          context_(context)
    {}

    ge::graphStatus CalcTiling(KdaInputProjMmBggParams &params) const
    {
        const KdaInputProjBaseParams &bp = tilingInfo_.baseParams;
        const uint32_t aicNum = GetAicNum();
        params = {};
        const char *opName = tilingInfo_.opName != nullptr ? tilingInfo_.opName : "KdaInputProj";
        if (bp.tSize == 0 || bp.hiddenSize == 0) {
            return ge::GRAPH_SUCCESS;
        }
        OP_CHECK_IF(aicNum == 0, OP_LOGE(opName, "MmBgg GetCoreNumAic returned 0."), return ge::GRAPH_FAILED);

        const uint32_t nCat = bp.betaSize + bp.gateSize + bp.gSize;
        OP_CHECK_IF(nCat == 0, OP_LOGE(opName, "MmBgg N_cat is 0."), return ge::GRAPH_FAILED);

        if (tilingInfo_.transWeightBeta != tilingInfo_.transWeightGate ||
            tilingInfo_.transWeightBeta != tilingInfo_.transWeightG) {
            OP_LOGW(opName, "MmBgg trans_weight mismatch: beta=%d gate=%d g=%d; Rebalance uses beta layout.",
                    static_cast<int32_t>(tilingInfo_.transWeightBeta),
                    static_cast<int32_t>(tilingInfo_.transWeightGate), static_cast<int32_t>(tilingInfo_.transWeightG));
        }

        uint32_t tTile = 0;
        uint32_t nTile = 0;
        OP_CHECK_IF(ElemAB() == 0U,
                    OP_LOGE(opName, "MmBgg x dtype size is 0 (dtype=%d).", static_cast<int32_t>(GetInputXType())),
                    return ge::GRAPH_FAILED);
        HwLimit hw;
        OP_CHECK_IF(!QueryHwLimit(hw), OP_LOGE(opName, "MmBgg QueryHwLimit failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(!GetRebalanceBlock(bp, aicNum, hw, tTile, nTile),
                    OP_LOGE(opName, "MmBgg GetRebalanceBlock failed."), return ge::GRAPH_FAILED);
        uint32_t l1K = GetMaxK1(hw, tTile, nTile);
        // kL1 是 L1 一次载入的 K 步长；不可大于实际 K，否则 Blaze 按 kL1 从 GM 拷贝会 MTE 越界。
        const uint32_t kAligned = MaxU(kAlign, CeilAlign(bp.hiddenSize, kAlign));
        l1K = MinU(l1K, kAligned);
        while (tTile > kAlign && !JudgeSpace(hw, tTile, nTile, l1K)) {
            tTile -= kAlign;
            l1K = MinU(GetMaxK1(hw, tTile, nTile), kAligned);
        }
        while (nTile > kAlign && !JudgeSpace(hw, tTile, nTile, l1K)) {
            nTile -= kAlign;
            l1K = MinU(GetMaxK1(hw, tTile, nTile), kAligned);
        }
        OP_CHECK_IF(!JudgeSpace(hw, tTile, nTile, l1K) || l1K == 0U,
                    OP_LOGE(opName, "MmBgg JudgeSpace failed tTile=%u nTile=%u l1K=%u.", tTile, nTile, l1K),
                    return ge::GRAPH_FAILED);

        const uint32_t nCntBeta = CeilDiv(bp.betaSize, nTile);
        const uint32_t nCntGate = CeilDiv(bp.gateSize, nTile);
        const uint32_t nCntG = CeilDiv(bp.gSize, nTile);
        const uint32_t nCntAll = nCntBeta + nCntGate + nCntG;
        OP_CHECK_IF(nCntAll == 0, OP_LOGE(opName, "MmBgg nCntAll is 0."), return ge::GRAPH_FAILED);

        const uint32_t tDim = CeilDiv(bp.tSize, tTile);
        const uint32_t tileNumAll = tDim * nCntAll;

        params.tTile = tTile;
        params.betaTile = nTile;
        params.gateTile = nTile;
        params.gTile = nTile;
        params.hiddenstatesTile = l1K;
        params.hiddenstatesL0Tile = ComputeL0K(hw, tTile, nTile, l1K);
        params.tDim = tDim;
        params.betaDim = nCntBeta;
        params.gateDim = nCntGate;
        params.gDim = nCntG;
        params.numBetaTile = tDim * nCntBeta;
        params.numGateTile = tDim * nCntGate;
        params.numGTile = tDim * nCntG;
        OP_CHECK_IF(
            params.hiddenstatesL0Tile == 0U,
            OP_LOGE(opName, "MmBgg hiddenstatesL0Tile is 0 tTile=%u nTile=%u hiddenstatesTile=%u.", tTile, nTile, l1K),
            return ge::GRAPH_FAILED);

        OP_LOGI(opName,
                "MmBgg v3-rebalance: T=%u K=%u N=(%u,%u,%u) Ncat=%u aic=%u transB=%d "
                "tTile=%u nTile=%u hiddenstatesTile=%u hiddenstatesL0Tile=%u tDim=%u "
                "nCnt=(%u,%u,%u) all=%u tileNumAll=%u "
                "usedCore=min(aic,tileNum) L1=%lu L0A=%lu L0B=%lu L0C=%lu L2=%lu.",
                bp.tSize, bp.hiddenSize, bp.betaSize, bp.gateSize, bp.gSize, nCat, aicNum,
                static_cast<int32_t>(tilingInfo_.transWeightBeta), tTile, nTile, l1K, params.hiddenstatesL0Tile, tDim,
                nCntBeta, nCntGate, nCntG, nCntAll, tileNumAll, static_cast<unsigned long>(hw.l1Size),
                static_cast<unsigned long>(hw.l0ASize), static_cast<unsigned long>(hw.l0BSize),
                static_cast<unsigned long>(hw.l0CSize), static_cast<unsigned long>(hw.l2Size));
        return ge::GRAPH_SUCCESS;
    }

private:
    static constexpr uint32_t kAlign = 16;
    static constexpr uint32_t kCubeMNMax = 256;
    static constexpr uint32_t kL0KHwMax = 128;
    static constexpr uint32_t kL0AStages = 2;
    static constexpr uint32_t kL0BStages = 2;
    static constexpr uint32_t kL1Stages = 2; // Blaze MatmulMultiBlockBasic l1Stages
    static constexpr uint32_t kElemAcc = 4;  // L0C accumulator（FP32）
    static constexpr uint32_t kElemBias = 4; // Blaze BiasType = float，即使无 bias 也占 slot
    static constexpr uint64_t kDb = 2;
    static constexpr uint64_t kK128B = 128;
    static constexpr uint64_t kK256B = 256;
    static constexpr uint64_t kK512B = 512;
    static constexpr uint64_t kMinTailBlock = 4096;
    static constexpr double kCubeBoundRatio = 0.85;
    static constexpr double kEps = 1e-9;
    static constexpr double kBalanceRateEdge = 0.9;

    struct HwLimit {
        uint64_t l1Size{0};
        uint64_t l0ASize{0};
        uint64_t l0BSize{0};
        uint64_t l0CSize{0};
        uint64_t l2Size{0};
        double hbmBw{0.0};
        double l2Bw{0.0};
        double coreFreq{0.0};
    };

    static uint32_t CeilDiv(uint32_t a, uint32_t b)
    {
        return (b == 0U) ? 0U : (a + b - 1U) / b;
    }

    static uint64_t CeilDivU64(uint64_t a, uint64_t b)
    {
        return (b == 0ULL) ? 0ULL : (a + b - 1ULL) / b;
    }

    static uint32_t CeilAlign(uint32_t x, uint32_t align)
    {
        return (align == 0U) ? x : ((x + align - 1U) / align) * align;
    }

    static uint64_t FloorAlignU64(uint64_t x, uint64_t align)
    {
        return (align == 0ULL) ? x : (x / align) * align;
    }

    static uint32_t MinU(uint32_t a, uint32_t b)
    {
        return (a < b) ? a : b;
    }

    static uint32_t MaxU(uint32_t a, uint32_t b)
    {
        return (a > b) ? a : b;
    }

    static uint32_t CountNCols(uint32_t nBeta, uint32_t nGate, uint32_t nG, uint32_t nTile)
    {
        return CeilDiv(nBeta, nTile) + CeilDiv(nGate, nTile) + CeilDiv(nG, nTile);
    }

    ge::DataType GetInputXType() const
    {
        if (context_ != nullptr) {
            const auto *desc = context_->GetInputDesc(X_INDEX);
            if (desc != nullptr) {
                return desc->GetDataType();
            }
        }
        return ge::DT_BF16;
    }

    uint32_t ElemAB() const
    {
        const int32_t sz = ge::GetSizeByDataType(GetInputXType());
        return (sz > 0) ? static_cast<uint32_t>(sz) : 0U;
    }

    uint32_t ComputeL0K(const HwLimit &hw, uint32_t l1M, uint32_t l1N, uint32_t l1K) const
    {
        const uint32_t sizeA = ElemAB();
        const uint32_t sizeB = sizeA;
        uint32_t byA = kAlign;
        uint32_t byB = kAlign;
        if (l1M != 0U && sizeA != 0U && hw.l0ASize != 0ULL) {
            byA = static_cast<uint32_t>(hw.l0ASize / (static_cast<uint64_t>(kL0AStages) * sizeA * l1M));
        }
        if (l1N != 0U && sizeB != 0U && hw.l0BSize != 0ULL) {
            byB = static_cast<uint32_t>(hw.l0BSize / (static_cast<uint64_t>(kL0BStages) * sizeB * l1N));
        }
        uint32_t l0K = (byA < byB) ? byA : byB;
        l0K = (l0K / kAlign) * kAlign;
        if (l0K > l1K) {
            l0K = l1K;
        }
        if (l0K > kL0KHwMax) {
            l0K = kL0KHwMax;
        }
        if (l0K < kAlign) {
            l0K = kAlign;
        }
        return l0K;
    }

    bool JudgeSpace(const HwLimit &hw, uint32_t l1M, uint32_t l1N, uint32_t l1K) const
    {
        const uint32_t sizeA = ElemAB();
        const uint32_t sizeB = sizeA;
        if (l1M == 0U || l1N == 0U || l1K == 0U || sizeA == 0U || sizeB == 0U) {
            return false;
        }
        if (hw.l1Size == 0ULL || hw.l0ASize == 0ULL || hw.l0BSize == 0ULL || hw.l0CSize == 0ULL) {
            return false;
        }
        if (l1M % kAlign != 0U || l1N % kAlign != 0U || l1K % kAlign != 0U) {
            return false;
        }
        if (l1M > kCubeMNMax || l1N > kCubeMNMax) {
            return false;
        }
        // Blaze BufferManager：slotSize = L1/4，stride = 4/l1Stages；
        // l1Stages=2 时 ping/pong 各占 L1/2：|A, B, Bias|。
        const uint64_t aL1One = static_cast<uint64_t>(l1M) * l1K * sizeA;
        const uint64_t bL1One = static_cast<uint64_t>(l1N) * l1K * sizeB;
        const uint64_t biasL1One = static_cast<uint64_t>(l1N) * kElemBias;
        const uint64_t perSlot = aL1One + bL1One + biasL1One;
        const uint64_t slotRegion = hw.l1Size / static_cast<uint64_t>(kL1Stages);
        const uint64_t l0cBytes = static_cast<uint64_t>(l1M) * l1N * kElemAcc;
        if (perSlot > slotRegion || l0cBytes > hw.l0CSize) {
            return false;
        }
        const uint32_t l0K = ComputeL0K(hw, l1M, l1N, l1K);
        const uint64_t l0aBytes = static_cast<uint64_t>(kL0AStages) * l1M * l0K * sizeA;
        const uint64_t l0bBytes = static_cast<uint64_t>(kL0BStages) * l1N * l0K * sizeB;
        return l0aBytes <= hw.l0ASize && l0bBytes <= hw.l0BSize;
    }

    uint32_t GetMaxK1(const HwLimit &hw, uint32_t m1, uint32_t n1) const
    {
        if (JudgeSpace(hw, m1, n1, 512)) {
            return 512;
        }
        if (JudgeSpace(hw, m1, n1, 256)) {
            return 256;
        }
        if (JudgeSpace(hw, m1, n1, 128)) {
            return 128;
        }
        return 64;
    }

    static bool ParseNonNegInt(const std::string &str, int64_t &out)
    {
        if (str.empty()) {
            return false;
        }
        char *endPtr = nullptr;
        errno = 0;
        const long long parsed = std::strtoll(str.c_str(), &endPtr, 10);
        if (errno != 0 || endPtr == str.c_str() || *endPtr != '\0' || parsed < 0) {
            return false;
        }
        out = static_cast<int64_t>(parsed);
        return true;
    }

    bool GetRequiredPlatformInt(const char *opName, const char *label, const char *key, int64_t &out) const
    {
        std::string val;
        if (tilingInfo_.platformInfo == nullptr) {
            OP_LOGE(opName, "MmBgg platformInfo is null, cannot get %s.%s.", label, key);
            return false;
        }
        if (!tilingInfo_.platformInfo->GetPlatformRes(label, key, val) || val.empty()) {
            OP_LOGE(opName, "MmBgg failed to get platform res %s.%s.", label, key);
            return false;
        }
        if (!ParseNonNegInt(val, out)) {
            OP_LOGE(opName, "MmBgg invalid platform res %s.%s='%s'.", label, key, val.c_str());
            return false;
        }
        return true;
    }

    bool QueryHwLimit(HwLimit &hw) const
    {
        hw = {};
        const char *opName = tilingInfo_.opName != nullptr ? tilingInfo_.opName : "KdaInputProj";
        if (tilingInfo_.platformInfo == nullptr) {
            OP_LOGE(opName, "MmBgg platformInfo is null.");
            return false;
        }
        platform_ascendc::PlatformAscendC plat(tilingInfo_.platformInfo);
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::L1, hw.l1Size);
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, hw.l0ASize);
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, hw.l0BSize);
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, hw.l0CSize);
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::L2, hw.l2Size);
        if (hw.l1Size == 0ULL || hw.l0ASize == 0ULL || hw.l0BSize == 0ULL || hw.l0CSize == 0ULL || hw.l2Size == 0ULL) {
            OP_LOGE(opName, "MmBgg GetCoreMemSize returned 0: L1=%lu L0A=%lu L0B=%lu L0C=%lu L2=%lu.",
                    static_cast<unsigned long>(hw.l1Size), static_cast<unsigned long>(hw.l0ASize),
                    static_cast<unsigned long>(hw.l0BSize), static_cast<unsigned long>(hw.l0CSize),
                    static_cast<unsigned long>(hw.l2Size));
            return false;
        }

        int64_t cubeFreq = 0;
        int64_t coreCnt = 0;
        int64_t ddrRate = 0;
        int64_t l2Rate = 0;
        if (!GetRequiredPlatformInt(opName, "AICoreSpec", "cube_freq", cubeFreq) ||
            !GetRequiredPlatformInt(opName, "SoCInfo", "ai_core_cnt", coreCnt) ||
            !GetRequiredPlatformInt(opName, "AICoreMemoryRates", "ddr_rate", ddrRate) ||
            !GetRequiredPlatformInt(opName, "AICoreMemoryRates", "l2_rate", l2Rate)) {
            return false;
        }
        if (cubeFreq == 0 || coreCnt == 0) {
            OP_LOGE(opName, "MmBgg invalid platform spec cube_freq=%ld ai_core_cnt=%ld.", cubeFreq, coreCnt);
            return false;
        }
        hw.coreFreq = static_cast<double>(cubeFreq) / 1000.0;
        hw.hbmBw = hw.coreFreq * static_cast<double>(coreCnt) * static_cast<double>(ddrRate) / 1024.0;
        hw.l2Bw = hw.coreFreq * static_cast<double>(coreCnt) * static_cast<double>(l2Rate) / 1024.0;
        if (hw.hbmBw <= 0.0 || hw.l2Bw <= 0.0) {
            OP_LOGE(opName, "MmBgg invalid bandwidth from platform hbmBw=%f l2Bw=%f (ddr_rate=%ld l2_rate=%ld).",
                    hw.hbmBw, hw.l2Bw, ddrRate, l2Rate);
            return false;
        }
        return true;
    }

    // 与 V3 GetMaxBaseWithLimit 同构；N 用三路最大宽，避免 N_cat 虚构出超单路的 baseN 上界。
    static uint64_t GetMaxBaseWithLimit(const HwLimit &hw, uint64_t shapeValue, uint64_t kValue, uint64_t dtypeSize,
                                        bool isATrans, bool isBTrans, uint64_t baseMNBufferLimit,
                                        uint64_t baseAlignUnit, bool isRightMatrix, bool isMemoryBound)
    {
        (void)isRightMatrix;
        const uint64_t kAlignValue = CeilDivU64(kValue, kAlign) * kAlign;
        const uint64_t kLimitValue = isMemoryBound ? kAlign : (kK128B / dtypeSize);
        const uint64_t minKL0 = std::min(kLimitValue, kAlignValue) * dtypeSize;
        uint64_t maxBaseMNWithBuffer = baseMNBufferLimit / kElemAcc / kAlign;
        uint64_t maxBaseBlock =
            std::min(hw.l0ASize / kDb / std::max(minKL0, static_cast<uint64_t>(1)), maxBaseMNWithBuffer);
        const uint64_t kAlignUnit =
            (!isATrans || isBTrans) ? ((isMemoryBound ? kK256B : kK512B) / dtypeSize) : static_cast<uint64_t>(kAlign);
        const uint64_t maxBaseMNWithKInner = hw.l1Size / (2ULL * kDb * dtypeSize * std::min(kAlignUnit, kAlignValue));
        maxBaseBlock = std::min(maxBaseBlock, maxBaseMNWithKInner);
        const uint64_t shapeAlign = CeilDivU64(shapeValue, baseAlignUnit) * baseAlignUnit;
        maxBaseBlock = std::min(shapeAlign, FloorAlignU64(maxBaseBlock, baseAlignUnit));
        if (shapeValue < baseAlignUnit) {
            maxBaseBlock = std::min(maxBaseBlock, CeilDivU64(shapeValue, kAlign) * kAlign);
        }
        if (maxBaseBlock == 0ULL) {
            maxBaseBlock = kAlign;
        }
        return std::min(maxBaseBlock, static_cast<uint64_t>(kCubeMNMax));
    }

    // 均衡率按三路逻辑块数 tDim×nCntAll，不是 CeilDiv(N_cat, nTile)。
    static double GetBalanceRate(uint32_t t, uint32_t nBeta, uint32_t nGate, uint32_t nG, uint32_t nCat,
                                 uint32_t usedCoreNum, uint32_t baseM, uint32_t baseN)
    {
        if (usedCoreNum == 0U || baseM == 0U || baseN == 0U) {
            return 0.0;
        }
        const uint64_t totalRound =
            static_cast<uint64_t>(CeilDiv(t, baseM)) * static_cast<uint64_t>(CountNCols(nBeta, nGate, nG, baseN));
        if (totalRound == 0ULL) {
            return 0.0;
        }
        const uint64_t mainRound = CeilDivU64(totalRound, usedCoreNum) - 1ULL;
        const uint64_t lastWave = totalRound - usedCoreNum * mainRound;
        const uint64_t totalTailSplit = (lastWave == 0ULL) ? 1ULL : (usedCoreNum / lastWave);
        const double workPerCore =
            static_cast<double>(t) * static_cast<double>(nCat) / static_cast<double>(usedCoreNum);
        if (mainRound == 0ULL || (static_cast<uint64_t>(baseM) * baseN /
                                  std::max(totalTailSplit, static_cast<uint64_t>(1))) < kMinTailBlock) {
            return workPerCore / (static_cast<double>(mainRound + 1ULL) * baseM * baseN);
        }
        const uint64_t tailSplitSqrt = static_cast<uint64_t>(std::sqrt(static_cast<double>(totalTailSplit)));
        const uint64_t offset =
            (totalTailSplit - tailSplitSqrt * tailSplitSqrt) / std::max(tailSplitSqrt, static_cast<uint64_t>(1)) + 1ULL;
        const double tailRound = 1.0 / static_cast<double>(tailSplitSqrt * (tailSplitSqrt + offset - 1ULL));
        return workPerCore / ((static_cast<double>(mainRound) + tailRound) * baseM * baseN);
    }

    bool GetRebalanceBlock(const KdaInputProjBaseParams &bp, uint32_t aicNum, const HwLimit &hw, uint32_t &tTile,
                           uint32_t &nTile) const
    {
        const uint32_t t = bp.tSize;
        const uint32_t k = bp.hiddenSize;
        const uint32_t nCat = bp.betaSize + bp.gateSize + bp.gSize;
        const uint32_t nMax = MaxU(bp.betaSize, MaxU(bp.gateSize, bp.gSize));
        const bool isATrans = false;
        const bool isBTrans = tilingInfo_.transWeightBeta;
        const uint64_t dtypeSize = ElemAB();

        // DAV_3510 ResetBase：256×256，再按 shape 裁尾。
        tTile = MinU(kCubeMNMax, MaxU(kAlign, CeilAlign(MinU(t, kCubeMNMax), kAlign)));
        nTile = MinU(kCubeMNMax, MaxU(kAlign, CeilAlign(MinU(nMax, kCubeMNMax), kAlign)));

        if (k == 0U) {
            const char *opName = tilingInfo_.opName != nullptr ? tilingInfo_.opName : "KdaInputProj";
            OP_LOGE(opName, "MmBgg hiddenSize K is 0.");
            return false;
        }

        const double computePower = hw.coreFreq * 8.0 * static_cast<double>(aicNum);
        const double cmr = (static_cast<double>(t) + nCat) / (static_cast<double>(t) * nCat);
        const double l2CacheUsage = std::max(
            static_cast<double>(static_cast<uint64_t>(t) + nCat) * k * dtypeSize / static_cast<double>(hw.l2Size), 1.0);
        double cubeBoundEdge = (hw.l2Bw / computePower) + l2CacheUsage * (1.0 - hw.l2Bw / hw.hbmBw) * cmr -
                               (1.0 + hw.l2Bw / hw.hbmBw) / static_cast<double>(k);

        const uint64_t baseMNBufferLimit = hw.l0CSize;
        const uint64_t baseMBest =
            std::min(static_cast<uint64_t>(CeilAlign(t, kAlign)), static_cast<uint64_t>(kCubeMNMax));
        const uint64_t baseNBest = std::max(
            static_cast<uint64_t>(kAlign),
            std::min(
                static_cast<uint64_t>(CeilAlign(nMax, kAlign)),
                FloorAlignU64(baseMNBufferLimit / kElemAcc / std::max(baseMBest, static_cast<uint64_t>(1)), kAlign)));
        const double cubeBoundParamBest =
            (1.0 / static_cast<double>(baseMBest)) + (1.0 / static_cast<double>(baseNBest));
        const bool isMemoryBound = cubeBoundParamBest > cubeBoundEdge;
        const uint64_t innerAlignUnit = isMemoryBound ? 128ULL : 64ULL;
        const double fixpBoundEdge =
            (static_cast<double>(t) * nCat * hw.hbmBw) / ((static_cast<double>(t) + nCat) * hw.l2Bw);
        const uint64_t baseMAlignUnit = isATrans ? (innerAlignUnit / dtypeSize) : static_cast<uint64_t>(kAlign);
        const uint64_t baseNAlignUnit = (static_cast<double>(k) < fixpBoundEdge) ?
                                            (kK256B / dtypeSize) :
                                            (isBTrans ? static_cast<uint64_t>(kAlign) : (innerAlignUnit / dtypeSize));

        uint64_t maxBaseM = GetMaxBaseWithLimit(hw, t, k, dtypeSize, isATrans, isBTrans, baseMNBufferLimit,
                                                baseMAlignUnit, false, isMemoryBound);
        uint64_t maxBaseN = GetMaxBaseWithLimit(hw, nMax, k, dtypeSize, isATrans, isBTrans, baseMNBufferLimit,
                                                baseNAlignUnit, true, isMemoryBound);
        if (maxBaseM < kAlign) {
            maxBaseM = kAlign;
        }
        if (maxBaseN < kAlign) {
            maxBaseN = kAlign;
        }

        uint32_t bestM = static_cast<uint32_t>(
            std::max(static_cast<uint64_t>(kAlign), std::min(maxBaseM, static_cast<uint64_t>(kCubeMNMax))));
        uint32_t bestN = static_cast<uint32_t>(
            std::max(static_cast<uint64_t>(kAlign),
                     std::min(maxBaseN, FloorAlignU64(baseMNBufferLimit / kElemAcc / bestM, baseNAlignUnit))));
        if (bestN == 0U) {
            bestN = kAlign;
        }
        double cubeBoundParam = (1.0 / bestM) + (1.0 / bestN);
        cubeBoundEdge *= kCubeBoundRatio;
        double balanceRate = GetBalanceRate(t, bp.betaSize, bp.gateSize, bp.gSize, nCat, aicNum, bestM, bestN);

        for (uint64_t curBaseM = maxBaseM; curBaseM >= 1ULL && curBaseM <= maxBaseM; curBaseM -= baseMAlignUnit) {
            uint64_t curMaxBaseN =
                std::min(maxBaseN, FloorAlignU64(baseMNBufferLimit / kElemAcc / curBaseM, baseNAlignUnit));
            if (curMaxBaseN < kAlign) {
                continue;
            }
            for (uint64_t curBaseN = curMaxBaseN; curBaseN >= 1ULL && curBaseN <= curMaxBaseN;
                 curBaseN -= baseNAlignUnit) {
                if (curBaseM > kCubeMNMax || curBaseN > kCubeMNMax) {
                    continue;
                }
                const uint32_t m32 = static_cast<uint32_t>(curBaseM);
                const uint32_t n32 = static_cast<uint32_t>(curBaseN);
                if (!JudgeSpace(hw, m32, n32, GetMaxK1(hw, m32, n32))) {
                    continue;
                }
                const double curCubeBoundParam =
                    (1.0 / static_cast<double>(curBaseM)) + (1.0 / static_cast<double>(curBaseN));
                const double curBalanceRate =
                    GetBalanceRate(t, bp.betaSize, bp.gateSize, bp.gSize, nCat, aicNum, m32, n32);
                bool skipCond = balanceRate >= kBalanceRateEdge && curCubeBoundParam > cubeBoundParam &&
                                curCubeBoundParam > cubeBoundEdge && cubeBoundEdge > 0.0;
                if (skipCond) {
                    continue;
                }
                const bool cubeBoundCond = curCubeBoundParam <= cubeBoundEdge && curBalanceRate > balanceRate;
                const bool balanceCond =
                    ((curCubeBoundParam / curBalanceRate) < (cubeBoundParam / balanceRate)) ||
                    ((std::abs(curCubeBoundParam / curBalanceRate - cubeBoundParam / balanceRate) < kEps) &&
                     curBalanceRate > balanceRate);
                if (cubeBoundCond || balanceCond) {
                    if (cubeBoundCond) {
                        cubeBoundEdge = curCubeBoundParam;
                    }
                    bestM = m32;
                    bestN = n32;
                    cubeBoundParam = curCubeBoundParam;
                    balanceRate = curBalanceRate;
                }
            }
        }

        tTile = MinU(CeilAlign(t, kAlign), bestM);
        nTile = MinU(CeilAlign(nMax, kAlign), bestN);
        if (tTile < kAlign) {
            tTile = kAlign;
        }
        if (nTile < kAlign) {
            nTile = kAlign;
        }
        return true;
    }

    uint32_t GetAicNum() const
    {
        if (tilingInfo_.platformInfo == nullptr) {
            return 0;
        }
        platform_ascendc::PlatformAscendC plat(tilingInfo_.platformInfo);
        return static_cast<uint32_t>(plat.GetCoreNumAic());
    }

    const KdaInputProjTilingInfo &tilingInfo_;
    gert::TilingContext *context_{nullptr};
};
} // namespace optiling

#endif // KDA_INPUT_PROJ_TILING_MM_BGG_H
