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
 * \file causal_conv1d.h
 * \brief Base class — shared conv1d compute, ring buffer, and state write-back.
 */

#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_tiling_key.h"
#include "causal_conv1d_util.h"

namespace CausalConv1d {

using namespace AscendC;
using namespace AscendC::MicroAPI;
using namespace CausalConv1dUtil;

constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

constexpr CastTrait CastTraitB16ToB32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr CastTrait CastTraitB32ToB16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                         RoundMode::CAST_RINT};

enum SeqPartitionMode : int32_t {
    SEQ_PARTITION_MODE_VARLEN = 0,
    SEQ_PARTITION_MODE_BATCH = 1,
};

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
class CausalConv1d {
public:
    __aicore__ inline CausalConv1d() = default;

    __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR bias,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR y,
                                             GM_ADDR convStatesOut, const CausalConv1dTilingData *tilingData)
    {
        ResetRuntimeState(tilingData);
        this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        this->weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
        this->biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
        this->convStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
        this->convStatesOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStatesOut));
        this->queryStartLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryStartLoc));
        this->cacheIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cacheIndices));
        this->yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
    }

protected:
    static constexpr int32_t kTemplateWidth = static_cast<int32_t>(widthKey);
    static constexpr bool kHasBias = (hasBiasKey != 0);
    static constexpr bool kHasActivation = (activationKey != 0);
    static constexpr uint32_t kUbBlockSize = 32;

    __aicore__ inline void ResetRuntimeState(const CausalConv1dTilingData *tilingData)
    {
        this->tilingData_ = tilingData;
    }

    __aicore__ inline void InitBuffers()
    {
        this->dimBufferSize_ = static_cast<int32_t>(this->tilingData_->coBatch) * this->tilingData_->baseDim;
        this->weightBiasOffset_ = (static_cast<int32_t>(this->tilingData_->coBatch) - 1) * this->tilingData_->baseDim *
                                      static_cast<int32_t>(sizeof(float) / sizeof(T)) +
                                  this->tilingData_->baseDim;
        const int32_t width = static_cast<int32_t>(this->tilingData_->kernelWidth);
        const int32_t outBufNum = 2;

        const uint32_t inBytes = (width + 1) * this->dimBufferSize_ * sizeof(T);
        const uint32_t outBytes = outBufNum * this->dimBufferSize_ * sizeof(T);
        const uint32_t calcBytes = (kTemplateWidth + (kHasBias ? 1 : 0)) * this->dimBufferSize_ * sizeof(float);

        uint32_t outBufCount = outBufNum * this->dimBufferSize_;
        uint32_t calcBufCount = (kTemplateWidth + (kHasBias ? 1 : 0)) * this->dimBufferSize_;

        this->pipe_.InitBuffer(this->buffer_, inBytes + outBytes + calcBytes);

        this->inTensor_ = this->buffer_.template Get<T>();
        this->outTensor_ = this->buffer_.template GetWithOffset<T>(outBufCount, inBytes);
        this->calcTensor_ = this->buffer_.template GetWithOffset<float>(calcBufCount, inBytes + outBytes);
    }

    __aicore__ inline void InitCalcBuf()
    {
        this->pool_ = CalcBufPool::Init(this->calcTensor_, static_cast<int32_t>(this->tilingData_->kernelWidth),
                                        this->dimBufferSize_);
    }

    __aicore__ inline void WeightBiasCast(LocalTensor<float> weightF, LocalTensor<float> biasF, int32_t width,
                                          int32_t baseDim)
    {
        uint32_t origDataCount = static_cast<uint32_t>(baseDim);
        uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(baseDim, V_LENGTH));
        uint16_t rowCount = static_cast<uint16_t>(width);
        const uint16_t coBatchU16 = static_cast<uint16_t>(this->tilingData_->coBatch);
        const int32_t dimBufSize = this->dimBufferSize_;
        const int32_t baseDimLocal = baseDim;
        const int32_t rowStrideSrc = dimBufSize * static_cast<int32_t>(sizeof(float) / sizeof(T));
        __ubuf__ T *weightSrc = reinterpret_cast<__ubuf__ T *>(weightF.GetPhyAddr()) + this->weightBiasOffset_;
        __ubuf__ float *weightDst = (__ubuf__ float *)weightF.GetPhyAddr();
        __ubuf__ T *biasSrc = nullptr;
        __ubuf__ float *biasDst = nullptr;
        if constexpr (kHasBias) {
            biasSrc = reinterpret_cast<__ubuf__ T *>(biasF.GetPhyAddr()) + this->weightBiasOffset_;
            biasDst = (__ubuf__ float *)biasF.GetPhyAddr();
        }

        __VEC_SCOPE__
        {
            RegTensor<T> src;
            RegTensor<float> dst;
            MaskReg pregLoop;
            uint32_t curCount;
            // b outermost: when coBatchU16 == 1 compiler eliminates the loop entirely
            for (uint16_t b = 0; b < coBatchU16; ++b) {
                for (uint16_t j = 0; j < rowCount; ++j) {
                    curCount = origDataCount;
                    for (uint16_t k = 0; k < colLoopTimes; ++k) {
                        pregLoop = UpdateMask<float>(curCount);
                        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(src, weightSrc + j * rowStrideSrc + k * V_LENGTH);
                        Cast<float, T, CastTraitB16ToB32>(dst, src, pregLoop);
                        StoreAlign(weightDst + j * dimBufSize + k * V_LENGTH + b * baseDimLocal, dst, pregLoop);
                    }
                }
            }
            if constexpr (kHasBias) {
                for (uint16_t b = 0; b < coBatchU16; ++b) {
                    curCount = origDataCount;
                    for (uint16_t k = 0; k < colLoopTimes; ++k) {
                        pregLoop = UpdateMask<float>(curCount);
                        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(src, biasSrc + k * V_LENGTH);
                        Cast<float, T, CastTraitB16ToB32>(dst, src, pregLoop);
                        StoreAlign(biasDst + k * V_LENGTH + b * baseDimLocal, dst, pregLoop);
                    }
                }
            }
        }
    }

    __aicore__ inline void LoadWeightAndBias(int64_t dimStart, int32_t baseDim)
    {
        const int64_t dim = this->tilingData_->dim;
        const int32_t width = static_cast<int32_t>(this->tilingData_->kernelWidth);
        LocalTensor<float> &weightCalc = this->pool_.weight;
        LocalTensor<float> &biasCalc = this->pool_.bias;
        LocalTensor<T> weightDma;
        LocalTensor<T> biasDma;

        if constexpr (!std::is_same<T, float>::value) {
            weightDma = weightCalc.ReinterpretCast<T>();
            biasDma = biasCalc.ReinterpretCast<T>();
        }

        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t srcGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);
        const uint32_t dstGap = (this->dimBufferSize_ * sizeof(float) - baseDim * sizeof(T)) / kUbBlockSize;

        if constexpr (std::is_same<T, float>::value) {
            DataCopyPad(weightCalc[0], this->weightGm_[dimStart],
                        {static_cast<uint16_t>(width), blockBytes, srcGap, dstGap, 0}, {false, 0, 0, 0});
        } else {
            DataCopyPad(weightDma[this->weightBiasOffset_], this->weightGm_[dimStart],
                        {static_cast<uint16_t>(width), blockBytes, srcGap, dstGap, 0}, {false, 0, 0, 0});
        }

        if constexpr (kHasBias) {
            if constexpr (std::is_same<T, float>::value) {
                DataCopyPad(biasCalc, this->biasGm_[dimStart], {1, blockBytes, 0, 0, 0}, {false, 0, 0, 0});
            } else {
                DataCopyPad(biasDma[this->weightBiasOffset_], this->biasGm_[dimStart], {1, blockBytes, 0, 0, 0},
                            {false, 0, 0, 0});
            }
        }

        SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        if constexpr (!std::is_same<T, float>::value) {
            this->WeightBiasCast(weightCalc, biasCalc, width, baseDim);
        }
    }

#define CONV1D_ACC(ringN, wPtrN) \
    if constexpr (IsSameType<T, float>::value) { \
        LoadAlign(ringF, ringN + col * V_LENGTH); \
    } else { \
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(ringTmp, ringN + col * V_LENGTH); \
        Cast<float, T, CastTraitB16ToB32>(ringF, ringTmp, pregLoop); \
    } \
    LoadAlign(weight, wPtrN + col * V_LENGTH); \
    Mul(tmp, ringF, weight, pregLoop); \
    Add(bias, bias, tmp, pregLoop)

    __aicore__ inline void ComputeConv1dUnroll(LocalTensor<T> ring, LocalTensor<float> weight, LocalTensor<float> bias,
                                               LocalTensor<T> out, int32_t baseDim, int32_t t, uint16_t colLoopTimes,
                                               uint32_t dataCount)
    {
        const uint16_t ringSize = static_cast<uint16_t>(kTemplateWidth + 1);
        __ubuf__ T *ringAddr = reinterpret_cast<__ubuf__ T *>(ring.GetPhyAddr());
        __ubuf__ float *weightAddr = reinterpret_cast<__ubuf__ float *>(weight.GetPhyAddr());
        __ubuf__ float *biasAddr = reinterpret_cast<__ubuf__ float *>(bias.GetPhyAddr());
        __ubuf__ T *outAddr = reinterpret_cast<__ubuf__ T *>(out.GetPhyAddr());
        __ubuf__ T *ring0 = ringAddr + static_cast<int32_t>((static_cast<uint16_t>(t) + 0) % ringSize) * baseDim;
        __ubuf__ T *ring1 = ringAddr + static_cast<int32_t>((static_cast<uint16_t>(t) + 1) % ringSize) * baseDim;
        __ubuf__ T *ring2 = ringAddr + static_cast<int32_t>((static_cast<uint16_t>(t) + 2) % ringSize) * baseDim;
        __ubuf__ T *ring3 = ringAddr + static_cast<int32_t>((static_cast<uint16_t>(t) + 3) % ringSize) * baseDim;
        __ubuf__ float *wPtr0 = weightAddr + static_cast<int32_t>(0) * baseDim;
        __ubuf__ float *wPtr1 = weightAddr + static_cast<int32_t>(1) * baseDim;
        __ubuf__ float *wPtr2 = weightAddr + static_cast<int32_t>(2) * baseDim;
        __ubuf__ float *wPtr3 = weightAddr + static_cast<int32_t>(3) * baseDim;
        __VEC_SCOPE__
        {
            RegTensor<T> ringTmp;
            RegTensor<float> ringF;
            RegTensor<float> weight;
            RegTensor<float> bias;
            RegTensor<float> tmp;
            RegTensor<T> outVal;
            MaskReg pregLoop;
            for (uint16_t col = 0; col < colLoopTimes; col++) {
                pregLoop = UpdateMask<float>(dataCount);
                if constexpr (kHasBias) {
                    LoadAlign(bias, biasAddr + col * V_LENGTH);
                } else {
                    Duplicate(bias, 0.0f, pregLoop);
                }
                CONV1D_ACC(ring0, wPtr0);
                CONV1D_ACC(ring1, wPtr1);
                if constexpr (kTemplateWidth >= 3) {
                    CONV1D_ACC(ring2, wPtr2);
                }
                if constexpr (kTemplateWidth >= 4) {
                    CONV1D_ACC(ring3, wPtr3);
                }
                if constexpr (kHasActivation) {
                    Muls(tmp, bias, -1.0f, pregLoop);
                    Exp(tmp, tmp, pregLoop);
                    Adds(tmp, tmp, 1.0f, pregLoop);
                    Div(bias, bias, tmp, pregLoop);
                }
                if constexpr (IsSameType<T, float>::value) {
                    StoreAlign(outAddr + col * V_LENGTH, bias, pregLoop);
                } else {
                    Cast<T, float, CastTraitB32ToB16>(outVal, bias, pregLoop);
                    StoreAlign<T, StoreDist::DIST_PACK_B32>(outAddr + col * V_LENGTH, outVal, pregLoop);
                }
            }
        }
    }

#undef CONV1D_ACC

    __aicore__ inline void ComputeConv1d(LocalTensor<T> ring, LocalTensor<float> weight, LocalTensor<float> bias,
                                         LocalTensor<T> out, int32_t baseDim, int32_t t)
    {
        uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(baseDim, V_LENGTH));
        uint32_t dataCount = static_cast<uint32_t>(baseDim);
        if constexpr (kTemplateWidth >= 2 && kTemplateWidth <= 4) {
            ComputeConv1dUnroll(ring, weight, bias, out, baseDim, t, colLoopTimes, dataCount);
        } else {
            __ubuf__ T *ringAddr = reinterpret_cast<__ubuf__ T *>(ring.GetPhyAddr());
            __ubuf__ float *weightAddr = reinterpret_cast<__ubuf__ float *>(weight.GetPhyAddr());
            __ubuf__ float *biasAddr = reinterpret_cast<__ubuf__ float *>(bias.GetPhyAddr());
            __ubuf__ T *outAddr = reinterpret_cast<__ubuf__ T *>(out.GetPhyAddr());
            const int32_t kernelWidth = kTemplateWidth;
            const uint16_t ringSize = static_cast<uint16_t>(kernelWidth + 1);
            __VEC_SCOPE__
            {
                RegTensor<T> ringTmp;
                RegTensor<float> ringF;
                RegTensor<float> weight;
                RegTensor<float> bias;
                RegTensor<float> tmp;
                RegTensor<T> outVal;
                MaskReg pregLoop;
                for (uint16_t col = 0; col < colLoopTimes; col++) {
                    pregLoop = UpdateMask<float>(dataCount);
                    if constexpr (kHasBias) {
                        LoadAlign(bias, biasAddr + col * V_LENGTH);
                    } else {
                        Duplicate(bias, 0.0f, pregLoop);
                    }
                    for (uint16_t j = 0; j < static_cast<uint16_t>(kernelWidth); j++) {
                        const int32_t ringOffset =
                            static_cast<int32_t>((static_cast<uint16_t>(t) + j) % ringSize) * baseDim + col * V_LENGTH;
                        const int32_t weightOffset = static_cast<int32_t>(j) * baseDim + col * V_LENGTH;
                        if constexpr (IsSameType<T, float>::value) {
                            LoadAlign(ringF, ringAddr + ringOffset);
                        } else {
                            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(ringTmp, ringAddr + ringOffset);
                            Cast<float, T, CastTraitB16ToB32>(ringF, ringTmp, pregLoop);
                        }
                        LoadAlign(weight, weightAddr + weightOffset);
                        Mul(tmp, ringF, weight, pregLoop);
                        Add(bias, bias, tmp, pregLoop);
                    }
                    if constexpr (kHasActivation) {
                        Muls(tmp, bias, -1.0f, pregLoop);
                        Exp(tmp, tmp, pregLoop);
                        Adds(tmp, tmp, 1.0f, pregLoop);
                        Div(bias, bias, tmp, pregLoop);
                    }
                    if constexpr (IsSameType<T, float>::value) {
                        StoreAlign(outAddr + col * V_LENGTH, bias, pregLoop);
                    } else {
                        Cast<T, float, CastTraitB32ToB16>(outVal, bias, pregLoop);
                        StoreAlign<T, StoreDist::DIST_PACK_B32>(outAddr + col * V_LENGTH, outVal, pregLoop);
                    }
                }
            }
        }
    }

    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int64_t dimStart, int32_t baseDim, int64_t dim,
                                          int32_t destStart = 0)
    {
        const int64_t stateLen = this->tilingData_->stateLen;
        const int64_t width = this->tilingData_->kernelWidth;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        if (len <= 0) {
            return;
        }

        LocalTensor<T> ring = this->inTensor_;
        const int32_t ringSize = width + 1;

        // Write the last wCount tokens from the ring, going backwards from the newest.
        const int32_t aliveCount = (width - 1 < stateLen) ? (width - 1) : stateLen;
        uint16_t wCount = static_cast<uint16_t>(aliveCount);
        const int32_t endSlot = CausalConv1dUtil::CurrSlot(len - 1, width);
        const int32_t startSlot = CausalConv1dUtil::CurrSlot(len - wCount, width);

        const int64_t stateBaseOffset =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(destStart) * dim + dimStart;
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t dstGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);

        DataCopyExtParams copyParams;
        copyParams.blockLen = blockBytes;
        copyParams.srcStride = static_cast<uint32_t>(this->dimBufferSize_ - baseDim) * sizeof(T) / kUbBlockSize;
        copyParams.dstStride = dstGap;
        copyParams.rsv = 0;

        LoopModeParams loopParams;
        loopParams.loop1Size = static_cast<uint32_t>(coBatch);
        loopParams.loop2Size = 1;
        loopParams.loop1SrcStride = static_cast<uint64_t>(baseDim) * sizeof(T);
        loopParams.loop1DstStride = static_cast<uint64_t>(stateLen) * dim * sizeof(T);
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

        if (startSlot <= endSlot) {
            copyParams.blockCount = wCount;
            DataCopyPad(this->convStatesOutGm_[stateBaseOffset], ring[startSlot * this->dimBufferSize_], copyParams);
        } else {
            const uint16_t seg1 = static_cast<uint16_t>(ringSize - startSlot);
            const uint16_t seg2 = static_cast<uint16_t>(endSlot + 1);
            copyParams.blockCount = seg1;
            DataCopyPad(this->convStatesOutGm_[stateBaseOffset], ring[startSlot * this->dimBufferSize_], copyParams);
            copyParams.blockCount = seg2;
            DataCopyPad(this->convStatesOutGm_[stateBaseOffset + static_cast<int64_t>(seg1) * dim], ring[0],
                        copyParams);
        }

        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }

    template <int32_t kWindowMode>
    __aicore__ inline void GetSeqWindow(int32_t groupIdx, int64_t seqLen, int32_t coBatch, int64_t &curSeqStart,
                                        int64_t &curSeqEnd) const
    {
        if constexpr (kWindowMode == SEQ_PARTITION_MODE_VARLEN) {
            curSeqStart = this->queryStartLocGm_.GetValue(groupIdx);
            curSeqEnd = this->queryStartLocGm_.GetValue(groupIdx + 1);
        } else {
            curSeqStart = static_cast<int64_t>(groupIdx) * coBatch * seqLen;
            curSeqEnd = static_cast<int64_t>(groupIdx + 1) * seqLen;
        }
    }

    // 查询 batchId 对应的物理 cache slot，用于多请求复用 conv_states 的场景。
    // cacheIndices 是 1D tensor，shape (batch,)，每个元素告知对应 batch 读哪个 slot。
    // 当 cacheIndices[batchId] == nullBlockId 时返回 false，表示该 batch 为填充位应跳过。
    __aicore__ inline bool ResolveBatchCacheIndex(int32_t batchId, bool hasCacheIndices, int32_t &cacheIdx) const
    {
        cacheIdx = batchId;
        if (!hasCacheIndices) {
            return true;
        }

        const int32_t cacheIdxVal = this->cacheIndicesGm_.GetValue(batchId);
        if (cacheIdxVal == this->tilingData_->nullBlockId) {
            return false;
        }
        cacheIdx = cacheIdxVal;
        return true;
    }

    __aicore__ inline bool ResolveBatchHasInit(int32_t batchId, bool hasInitialStateMode) const
    {
        return hasInitialStateMode ? (this->initialStateModeGm_.GetValue(batchId) != 0) : false;
    }

    __aicore__ inline const CausalConv1dTilingData *GetTilingData() const { return tilingData_; }

protected:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> buffer_;
    LocalTensor<T> inTensor_;
    LocalTensor<T> outTensor_;
    LocalTensor<float> calcTensor_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> weightGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> convStatesGm_;
    GlobalTensor<T> convStatesOutGm_;
    GlobalTensor<int32_t> queryStartLocGm_;
    GlobalTensor<int32_t> cacheIndicesGm_;
    GlobalTensor<int32_t> initialStateModeGm_;
    GlobalTensor<int32_t> numAcceptedTokensGm_;
    GlobalTensor<T> yGm_;

    const CausalConv1dTilingData *tilingData_{nullptr};

    int32_t dimBufferSize_{0};
    int32_t weightBiasOffset_{0};

    CalcBufPool pool_;
};
} // namespace CausalConv1d
#endif // CAUSAL_CONV1D_H
