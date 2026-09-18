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
 * \file kda_input_proj_tiling_qmm_qkv.h
 * \brief Stage2 AIC: MX W8A8 QuantMatmul(qkv) tiling (Blaze without-batch).
 */

#ifndef KDA_INPUT_PROJ_TILING_QMM_QKV_H
#define KDA_INPUT_PROJ_TILING_QMM_QKV_H

#include <algorithm>
#include <cstdint>

#include "err/ops_err.h"
#include "op_common/op_host/util/math_util.h"
#include "kda_input_proj_tiling.h"

namespace optiling {

class KdaInputProjQmmQkvTiling {
public:
    explicit KdaInputProjQmmQkvTiling(const KdaInputProjTilingInfo &tilingInfo)
        : opName_(tilingInfo.opName != nullptr ? tilingInfo.opName : "KdaInputProj"),
          m_(tilingInfo.baseParams.tSize),
          n_(tilingInfo.baseParams.qkvSize),
          k_(tilingInfo.baseParams.hiddenSize),
          transWeight_(tilingInfo.transWeightQkv),
          aicNum_(tilingInfo.aicNum),
          l1Size_(tilingInfo.l1Size),
          l0cSize_(tilingInfo.l0cSize)
    {}

    ge::graphStatus CalcTiling(KdaInputProjQmmQkvParams &params) const
    {
        params = KdaInputProjQmmQkvParams{};

        uint64_t baseM = 0UL;
        uint64_t baseN = 0UL;
        uint64_t baseK = 0UL;
        if (!CalcBasicBlock(baseM, baseN, baseK)) {
            return ge::GRAPH_FAILED;
        }

        uint64_t stepKa = 0UL;
        uint64_t stepKb = 0UL;
        uint64_t scaleFactor = 0UL;
        if (!CalcL1Tiling(baseM, baseN, baseK, stepKa, stepKb, scaleFactor)) {
            return ge::GRAPH_FAILED;
        }

        const uint64_t initScaleKL1 = std::min(scaleFactor * stepKa * baseK, scaleFactor * stepKb * baseK);
        uint64_t kL1 = 0UL;
        uint64_t scaleKL1 = 0UL;
        uint8_t nBufferNum = L1_TWO_BUFFER;
        CalcNBufferNum(baseM, baseN, baseK, stepKa, stepKb, initScaleKL1, kL1, scaleKL1, nBufferNum);

        // 兜底：L1 超配在设备上只会表现为 LOAD2D 读越界（aicore error 507015），很难定位，
        // 这里在 host 侧把它拦成明确的 tiling 失败。
        const L1Estimate finalEst{kL1, scaleKL1, baseM, baseN};
        const uint64_t finalL1 = CalcUsedL1Size(finalEst, nBufferNum);
        if (finalL1 > l1Size_) {
            OP_LOGE(opName_, "QMM L1 overflow: used=%lu l1=%lu base(%lu,%lu,%lu) kL1=%lu scaleKL1=%lu nBuf=%u", finalL1,
                    l1Size_, baseM, baseN, baseK, kL1, scaleKL1, nBufferNum);
            return ge::GRAPH_FAILED;
        }

        params.kL1 = static_cast<uint32_t>(kL1);
        params.scaleKL1 = static_cast<uint32_t>(scaleKL1);
        params.baseM = static_cast<uint32_t>(baseM);
        params.baseN = static_cast<uint32_t>(baseN);
        params.baseK = static_cast<uint32_t>(baseK);
        params.nBufferNum = nBufferNum;
        params.dbL0C = CalcDbL0C(baseM, baseN);
        const uint64_t mCnt = Ops::Base::CeilDiv(m_, baseM);
        const uint64_t nCnt = Ops::Base::CeilDiv(n_, baseN);
        const uint64_t qmmTiles = mCnt * nCnt;
        const uint32_t qmmUsedAic = static_cast<uint32_t>(std::min(qmmTiles, static_cast<uint64_t>(aicNum_)));
        CalcTailTiles(baseM, baseN, qmmUsedAic, params);
        // 单 M 轮时 B 只扫一遍，kernel replay 还会清 L2；streaming 少占 cache。
        // mCnt>1 时同一核可能沿 M 复用 B，保持 L2 NORMAL。
        params.bMustHitL2 = (mCnt > 1UL) ? 1U : 0U;

        OP_LOGI(opName_,
                "KdaInputProj QMM tiling: m=%lu n=%lu k=%lu base(%u,%u,%u) tiles=%lu x %lu = %lu usedAic=%u/%u "
                "kL1=%u scaleKL1=%u nBuf=%u dbL0C=%u bMustHitL2=%u l1=%lu/%lu",
                m_, n_, k_, params.baseM, params.baseN, params.baseK, mCnt, nCnt, qmmTiles, qmmUsedAic, aicNum_,
                params.kL1, params.scaleKL1, params.nBufferNum, params.dbL0C, params.bMustHitL2, finalL1, l1Size_);
        return ge::GRAPH_SUCCESS;
    }

private:
    static constexpr uint64_t CUBE_BLOCK = 16UL;
    static constexpr uint64_t L1_ALIGN_SIZE = 32UL;
    static constexpr uint64_t L2_ALIGN_SIZE = 128UL;
    static constexpr uint64_t BASIC_BLOCK_256 = 256UL;
    static constexpr uint64_t BASIC_BLOCK_128 = 128UL;
    static constexpr uint64_t MX_GROUP_SIZE = 32UL;
    static constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t ESTIMATED_SCALE_K = 4096UL;
    static constexpr uint32_t DOUBLE_BUFFER = 2U;
    static constexpr uint8_t L1_TWO_BUFFER = 2U;
    static constexpr uint8_t L1_THREE_BUFFER = 3U;
    static constexpr uint8_t L1_FOUR_BUFFER = 4U;
    static constexpr uint32_t DATA_SIZE_L0C = 4U;

    struct L1Estimate {
        uint64_t kL1{0};
        uint64_t scaleKL1{0};
        uint64_t baseM{0};
        uint64_t baseN{0};
    };

    bool CalcBasicBlock(uint64_t &baseM, uint64_t &baseN, uint64_t &baseK) const
    {
        if (m_ == 0UL || n_ == 0UL || k_ == 0UL) {
            OP_LOGE(opName_, "Invalid QMM shape: m=%lu n=%lu k=%lu", m_, n_, k_);
            return false;
        }

        const uint64_t mAlign = CUBE_BLOCK; // A is ND [T, K]
        const uint64_t nAlign = transWeight_ ? CUBE_BLOCK : L1_ALIGN_SIZE;
        const uint64_t kAlign = MXFP_DIVISOR_SIZE;

        // 保持 256 级基本块。T=8、N=4608 时 nCnt=18
        baseM = Ops::Base::CeilAlign(std::min(m_, BASIC_BLOCK_256), mAlign);
        baseN = Ops::Base::CeilAlign(std::min(n_, BASIC_BLOCK_256), nAlign);
        baseK = Ops::Base::CeilAlign(std::min(k_, BASIC_BLOCK_128), kAlign);
        return baseM != 0UL && baseN != 0UL && baseK != 0UL;
    }

    uint64_t GetDepthA1B1(uint64_t baseM, uint64_t baseN, uint64_t baseK) const
    {
        // FP8 data + e8m0 scale: 1 byte/elem, so element count equals byte size.
        constexpr uint64_t INDEX = 2UL;
        uint64_t depth = 1UL;
        uint64_t scaleKL1 = std::min(k_, ESTIMATED_SCALE_K);
        const uint64_t baseABSize = baseM * baseK + baseN * baseK;
        const uint64_t baseScaleSize = baseM + baseN;
        uint64_t scaleL1Size =
            baseScaleSize * Ops::Base::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE * DOUBLE_BUFFER;
        while (depth * baseABSize + scaleL1Size <= l1Size_) {
            depth *= INDEX;
            const uint64_t kL1 = depth / DOUBLE_BUFFER * baseK;
            if (kL1 > scaleKL1) {
                scaleKL1 = kL1;
                scaleL1Size = baseScaleSize * Ops::Base::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE *
                              DOUBLE_BUFFER;
            }
        }
        return depth == 1UL ? depth : depth / INDEX;
    }

    bool CalStepKAndDepthK(uint64_t baseK, uint64_t &depthKa, uint64_t &depthKb, uint64_t &stepKa,
                           uint64_t &stepKb) const
    {
        stepKa = depthKa / DOUBLE_BUFFER;
        stepKb = depthKb / DOUBLE_BUFFER;
        if (stepKa * baseK > k_) {
            stepKa = Ops::Base::CeilDiv(k_, baseK);
        }
        if (stepKb * baseK > k_) {
            stepKb = Ops::Base::CeilDiv(k_, baseK);
        }
        if (stepKa == 0UL || stepKb == 0UL) {
            return false;
        }
        if (stepKa > stepKb) {
            stepKa = stepKa / stepKb * stepKb;
        }
        if (stepKb > stepKa) {
            stepKb = stepKb / stepKa * stepKa;
        }
        stepKa = std::min(stepKa, 4UL);
        stepKb = std::min(stepKb, 4UL);
        depthKa = stepKa * DOUBLE_BUFFER;
        depthKb = stepKb * DOUBLE_BUFFER;
        return true;
    }

    bool CalScaleFactors(uint64_t baseM, uint64_t baseN, uint64_t baseK, uint64_t depthKa, uint64_t depthKb,
                         uint64_t stepKa, uint64_t stepKb, uint64_t &scaleFactor) const
    {
        const uint64_t baseASize = baseM * baseK;
        const uint64_t baseBSize = baseN * baseK;
        const uint64_t usedABSize = depthKa * baseASize + depthKb * baseBSize;
        if (l1Size_ < usedABSize) {
            OP_LOGE(opName_, "L1 underflow for MX AB buffers: l1=%lu usedAB=%lu", l1Size_, usedABSize);
            return false;
        }
        const uint64_t leftL1 = l1Size_ - usedABSize;
        const uint64_t stepK = std::min(stepKa, stepKb);
        const uint64_t kL1 = stepK * baseK;
        const uint64_t scaleGroupL1Size = (baseM + baseN) * MXFP_MULTI_BASE_SIZE * DOUBLE_BUFFER;
        if (scaleGroupL1Size == 0UL || kL1 == 0UL) {
            return false;
        }

        uint64_t scaleKL1 = std::min(leftL1 / scaleGroupL1Size * MXFP_DIVISOR_SIZE, Ops::Base::CeilAlign(k_, kL1));
        scaleKL1 = Ops::Base::FloorAlign(scaleKL1, kL1);
        if (scaleKL1 == 0UL) {
            OP_LOGE(opName_, "Insufficient L1 for MX scale buffer");
            return false;
        }
        scaleFactor = scaleKL1 / kL1;
        return scaleFactor > 0UL;
    }

    bool CalcL1Tiling(uint64_t baseM, uint64_t baseN, uint64_t baseK, uint64_t &stepKa, uint64_t &stepKb,
                      uint64_t &scaleFactor) const
    {
        uint64_t depthKa = GetDepthA1B1(baseM, baseN, baseK);
        uint64_t depthKb = depthKa;
        if (!CalStepKAndDepthK(baseK, depthKa, depthKb, stepKa, stepKb)) {
            return false;
        }
        return CalScaleFactors(baseM, baseN, baseK, depthKa, depthKb, stepKa, stepKb, scaleFactor);
    }

    uint64_t CalcUsedL1Size(const L1Estimate &est, uint32_t l1BufferNum) const
    {
        uint64_t used = est.baseN * est.kL1 * l1BufferNum;
        used += est.baseN * Ops::Base::CeilDiv(est.scaleKL1, MX_GROUP_SIZE) * L1_TWO_BUFFER;
        used += est.baseM * est.kL1 * l1BufferNum;
        used += est.baseM * Ops::Base::CeilDiv(est.scaleKL1, MX_GROUP_SIZE) * L1_TWO_BUFFER;
        return used;
    }

    bool CanFitL1BufferNum(const L1Estimate &est, uint32_t l1BufferNum) const
    {
        return CalcUsedL1Size(est, l1BufferNum) <= l1Size_;
    }

    uint64_t GetHalfKFallbackScaleKL1(uint64_t kL1, uint64_t scaleKL1) const
    {
        if (scaleKL1 % ESTIMATED_SCALE_K == 0UL) {
            return scaleKL1;
        }
        const uint64_t halfK = Ops::Base::CeilDiv(k_, 2UL);
        if (scaleKL1 > halfK && scaleKL1 < k_) {
            const uint64_t adjusted = Ops::Base::CeilAlign(halfK, kL1);
            return adjusted < scaleKL1 ? adjusted : scaleKL1;
        }
        return scaleKL1;
    }

    uint64_t GetFullCoverScaleKL1IfPossible(L1Estimate params, uint32_t l1BufferNum) const
    {
        const uint64_t fullCover = Ops::Base::CeilAlign(k_, params.kL1);
        if (fullCover <= params.scaleKL1) {
            return params.scaleKL1;
        }

        const uint64_t fallback = params.scaleKL1;
        params.scaleKL1 = fullCover;
        return CanFitL1BufferNum(params, l1BufferNum) ? fullCover : fallback;
    }

    void ApplyMultiBuffer(const L1Estimate &params, uint32_t l1BufferNum, uint64_t &kL1, uint64_t &scaleKL1,
                          uint8_t &nBufferNum) const
    {
        kL1 = params.kL1;
        scaleKL1 = GetFullCoverScaleKL1IfPossible(params, l1BufferNum);
        nBufferNum = static_cast<uint8_t>(l1BufferNum);
    }

    bool TryL1Config(uint64_t baseM, uint64_t baseN, uint64_t kL1Cand, uint64_t initScaleKL1, uint32_t l1BufferNum,
                     L1Estimate &out) const
    {
        if (kL1Cand == 0UL || l1BufferNum < L1_TWO_BUFFER) {
            return false;
        }
        L1Estimate est{kL1Cand, GetHalfKFallbackScaleKL1(kL1Cand, initScaleKL1), baseM, baseN};
        est.scaleKL1 = GetFullCoverScaleKL1IfPossible(est, l1BufferNum);
        if (CanFitL1BufferNum(est, l1BufferNum)) {
            out = est;
            return true;
        }
        // 全 K scale 放不下时退回按 kL1 对齐的 scale 窗
        est.scaleKL1 = GetHalfKFallbackScaleKL1(kL1Cand, initScaleKL1);
        if (CanFitL1BufferNum(est, l1BufferNum)) {
            out = est;
            return true;
        }
        return false;
    }

    void CalcNBufferNum(uint64_t baseM, uint64_t baseN, uint64_t baseK, uint64_t stepKa, uint64_t stepKb,
                        uint64_t initScaleKL1, uint64_t &kL1, uint64_t &scaleKL1, uint8_t &nBufferNum) const
    {
        const uint64_t kAlign = std::max(baseK, L2_ALIGN_SIZE);
        const uint64_t kL1Max = Ops::Base::FloorAlign(k_, kAlign);
        L1Estimate best{};
        uint32_t bestBuf = L1_TWO_BUFFER;
        bool found = false;
        uint64_t bestScore = 0UL;

        for (uint64_t cand = kL1Max; cand >= kAlign; cand -= kAlign) {
            const uint32_t bufs[] = {L1_FOUR_BUFFER, L1_THREE_BUFFER, L1_TWO_BUFFER};
            for (uint32_t nBuf : bufs) {
                L1Estimate est{};
                if (!TryL1Config(baseM, baseN, cand, initScaleKL1, nBuf, est)) {
                    continue;
                }
                const uint64_t score =
                    est.kL1 * 1000UL + static_cast<uint64_t>(nBuf) * 10UL + (est.scaleKL1 >= k_ ? 1UL : 0UL);
                if (!found || score > bestScore) {
                    found = true;
                    best = est;
                    bestBuf = nBuf;
                    bestScore = score;
                }
            }
            if (found && best.kL1 >= cand) {
                break;
            }
        }

        if (!found) {
            const uint64_t stepK = std::min(stepKa, stepKb);
            const uint64_t fallbackKL1 = stepK * baseK;
            best = L1Estimate{fallbackKL1, GetHalfKFallbackScaleKL1(fallbackKL1, initScaleKL1), baseM, baseN};
            bestBuf = L1_TWO_BUFFER;
        }
        ApplyMultiBuffer(best, bestBuf, kL1, scaleKL1, nBufferNum);
    }

    uint8_t CalcDbL0C(uint64_t baseM, uint64_t baseN) const
    {
        const uint64_t need = baseM * baseN * DATA_SIZE_L0C * DOUBLE_BUFFER;
        return need <= l0cSize_ ? static_cast<uint8_t>(DOUBLE_BUFFER) : 1U;
    }

    void CalcTailTiles(uint64_t baseM, uint64_t baseN, uint32_t usedAic, KdaInputProjQmmQkvParams &params) const
    {
        params.mTailTile = 1;
        params.nTailTile = 1;
        params.mBaseTailSplitCnt = 1;
        params.nBaseTailSplitCnt = 1;
        params.mTailMain = 0;
        params.nTailMain = 0;

        const uint64_t mCnt = Ops::Base::CeilDiv(m_, baseM);
        const uint64_t nCnt = Ops::Base::CeilDiv(n_, baseN);
        const uint64_t total = mCnt * nCnt;
        if (total == 0UL || usedAic == 0U) {
            return;
        }
        const uint64_t tailBlocks = total % static_cast<uint64_t>(usedAic);
        if (tailBlocks == 0UL) {
            return;
        }

        uint64_t bestM = 1UL;
        uint64_t bestN = 1UL;
        for (uint64_t mt = 1UL; mt <= 4UL; ++mt) {
            for (uint64_t nt = 1UL; nt <= 4UL; ++nt) {
                if (tailBlocks * mt * nt <= static_cast<uint64_t>(usedAic) && mt * nt >= bestM * bestN) {
                    bestM = mt;
                    bestN = nt;
                }
            }
        }
        params.mTailTile = static_cast<uint32_t>(bestM);
        params.nTailTile = static_cast<uint32_t>(bestN);
    }

    const char *opName_{nullptr};
    const uint64_t m_{0};
    const uint64_t n_{0};
    const uint64_t k_{0};
    const bool transWeight_{true};
    const uint32_t aicNum_{0};
    const uint64_t l1Size_{0};
    const uint64_t l0cSize_{0};
};
} // namespace optiling

#endif // KDA_INPUT_PROJ_TILING_QMM_QKV_H
