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
 * \file recurrent_gated_delta_rule_arch35.h
 * \brief
 */

#ifndef RECURRENT_GATED_DELTA_RULE_ARCH35_H
#define RECURRENT_GATED_DELTA_RULE_ARCH35_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "vf_vec_mul_mat.h"
#include "vf_outer_add.h"
#include "../recurrent_gated_delta_rule_tiling_data.h"

namespace RecurrentGatedDeltaRule {

using namespace matmul;
using namespace AscendC;
constexpr uint64_t SINGLE_BUFFER = 1;
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t MAX_MTP = 8;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint32_t BLOCK_SIZE_32B = 32;

// BUFFER的字节数
static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;

// GM->UB
/*
 * dealRowCount: 需要拷贝的行数
 * actDataLen: 一行需要拷贝的元素数
 * srcRowStride: gm上两行数据起始位置之间间隔元素数
 * dstRowStride: ub上两行数据起始位置之间间隔元素数
 * enableLargeStride默认为false, 当srcStrideOfDataCopy超过datacopypad范围时开启
 */
template <typename T, bool enableLargeStride = false>
__aicore__ inline void CopySingleMatrixNDToND(LocalTensor<T> ubTensor, const GlobalTensor<T> gmTensor,
                                              uint32_t dealRowCount, uint32_t actDataLen, uint64_t srcRowStride,
                                              uint64_t dstRowStride)
{
    constexpr uint64_t UINT16_MAX_VALUE = 65535u;
    constexpr uint64_t UINT32_MAX_VALUE = 4294967295u;
    uint32_t blockElemNum = BLOCK_SIZE_32B / sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        constexpr uint32_t HALF_SIZE_DIVISOR = 2;
        blockElemNum = blockElemNum * HALF_SIZE_DIVISOR;
        actDataLen = actDataLen / HALF_SIZE_DIVISOR;
        srcRowStride = srcRowStride / HALF_SIZE_DIVISOR;
    }
    if constexpr (enableLargeStride) {
        uint64_t srcStrideOfDataCopyPad = (srcRowStride - actDataLen) * sizeof(T);
        if (unlikely(srcStrideOfDataCopyPad > UINT32_MAX_VALUE)) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = actDataLen * sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            DataCopyPadExtParams<T> dataCopyPadParams;
            dataCopyPadParams.isPad = true;
            dataCopyPadParams.leftPadding = 0;
            dataCopyPadParams.rightPadding = (blockElemNum - (actDataLen % blockElemNum)) % blockElemNum;
            dataCopyPadParams.paddingValue = 0;

            for (uint32_t i = 0; i < dealRowCount; ++i) {
                DataCopyPad(ubTensor[i * dstRowStride], gmTensor[i * srcRowStride], dataCopyParams, dataCopyPadParams);
            }
            return;
        }
    }
    bool isPad = ((actDataLen % blockElemNum) != 0 || (srcRowStride % blockElemNum) != 0 ||
                  (dstRowStride % blockElemNum) != 0); // 判断是否32字节对齐，确定是否走datacopypad
    uint64_t srcStrideOfDataCopy = (srcRowStride - actDataLen) / blockElemNum;
    // 在有pad或srcStrideOfDataCopy不符合datacopy范围时，使用datacopypad拷贝完成
    if (unlikely(isPad || (srcStrideOfDataCopy > UINT16_MAX_VALUE))) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(dealRowCount); // 外部传入
        dataCopyParams.blockLen = actDataLen * sizeof(T);
        dataCopyParams.srcStride = (srcRowStride - actDataLen) * sizeof(T);
        dataCopyParams.dstStride = (dstRowStride - actDataLen) / blockElemNum; // 外部传入

        DataCopyPadExtParams<T> dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.leftPadding = 0;
        dataCopyPadParams.rightPadding = (blockElemNum - (actDataLen % blockElemNum)) % blockElemNum;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, dataCopyPadParams);
    } else { // 其他情况使用datacopy拷贝完成
        DataCopyParams repeatParams;
        repeatParams.blockCount = static_cast<uint16_t>(dealRowCount);
        repeatParams.blockLen = actDataLen / blockElemNum;
        repeatParams.srcStride = (srcRowStride - actDataLen) / blockElemNum;
        repeatParams.dstStride = (dstRowStride - actDataLen) / blockElemNum;
        DataCopy(ubTensor, gmTensor, repeatParams);
    }
}

struct RGDRInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gama;
    GM_ADDR gamaK;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR ssmStateIndices;
    GM_ADDR numAcceptedTokens;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename inType, typename outType, typename stateType = inType>
class RGDR {
public:
    __aicore__ inline RGDR(const RecurrentGatedDeltaRuleTilingData *tilingData)
    {
        B_ = tilingData->b;
        T_ = tilingData->t;
        NK_ = tilingData->nk;
        DK_ = tilingData->dk;
        NV_ = tilingData->nv;
        DV_ = tilingData->dv;
        scale_ = tilingData->scale;
        hasAcceptedTokens_ = (tilingData->hasAcceptedTokens == 1);
        hasGama_ = (tilingData->hasGama == 1);
        hasGamaK_ = (tilingData->hasGamaK == 1);
        alignDK_ = Ceil(tilingData->dk, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignDV_ = Ceil(tilingData->dv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        vStep_ = 8192 / alignDK_;
        if (vStep_ > 256) {
            vStep_ = 255;
        }
        vStep_ = vStep_ / BF16_NUM_PER_BLOCK * BF16_NUM_PER_BLOCK;
        load = 0;
        usedblk = 0;
        stateStride0_ = tilingData->stateStride0;
        stateStride1_ = tilingData->stateStride1;
    }

    __aicore__ inline void Init(const RGDRInitParams &initParams, TPipe *pipe)
    {
        uint64_t blockDim = GetBlockNum();
        blockIdx = GetBlockIdx();
        if (blockIdx >= blockDim) {
            return;
        }
        pipe_ = pipe;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void SetGlobalTensors(const RGDRInitParams &initParams)
    {
        queryGm_.SetGlobalBuffer((__gm__ inType *)initParams.query);
        keyGm_.SetGlobalBuffer((__gm__ inType *)initParams.key);
        valueGm_.SetGlobalBuffer((__gm__ inType *)initParams.value);
        gamaGm_.SetGlobalBuffer((__gm__ float *)initParams.gama);
        gamaKGm_.SetGlobalBuffer((__gm__ float *)initParams.gamaK);
        betaGm_.SetGlobalBuffer((__gm__ inType *)initParams.beta);
        initStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.initState);
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens);
        ssmStateIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
        numAcceptedTokensGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.numAcceptedTokens);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.finalState);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        constexpr uint32_t stateQueueSize =
            std::is_same_v<stateType, float> ? BUFFER_SIZE_BYTE_32K : BUFFER_SIZE_BYTE_16K;
        pipe_->InitBuffer(inputQueue1_, DOUBLE_BUFFER, stateQueueSize);
        pipe_->InitBuffer(stateOutQueue_, SINGLE_BUFFER, stateQueueSize);
        pipe_->InitBuffer(attnOutQueue_, SINGLE_BUFFER, BUFFER_SIZE_BYTE_16K);
        pipe_->InitBuffer(gamaBuff_, BUFFER_SIZE_BYTE_16K);
        pipe_->InitBuffer(stateBuff_, BUFFER_SIZE_BYTE_32K);
        pipe_->InitBuffer(queryBuff_, BUFFER_SIZE_BYTE_16K);
        pipe_->InitBuffer(keyBuff_, BUFFER_SIZE_BYTE_16K);
        pipe_->InitBuffer(valueBuff_, BUFFER_SIZE_BYTE_16K);
        pipe_->InitBuffer(tmpBuff2_, BUFFER_SIZE_BYTE_32K);
        pipe_->InitBuffer(tmpBuff1_, BUFFER_SIZE_BYTE_2K);

        gamaUb_ = gamaBuff_.Get<float>();
        stateUb_ = stateBuff_.Get<float>();
        queryUb_ = queryBuff_.Get<float>();
        keyUb_ = keyBuff_.Get<float>();
        valueUb_ = valueBuff_.Get<float>();
    }

    __aicore__ inline void ComputeAvgload()
    {
        uint64_t realT = 0;
        uint32_t maxSeqLen = 0;
        for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
            uint32_t bSeqLen = cuSeqlensGm_.GetValue(batch_i);
            realT += bSeqLen;
            if (bSeqLen > maxSeqLen) {
                maxSeqLen = bSeqLen;
            }
        }
        avgload = Ceil(realT * NV_, GetBlockNum());
        if (maxSeqLen > avgload) {
            avgload = maxSeqLen;
        }
    }

    __aicore__ inline void Process()
    {
        ComputeAvgload();
        uint32_t seq1 = 0;
        for (uint64_t batch_i = 0; batch_i < B_; batch_i++) {
            uint32_t seq0 = seq1;
            seq1 += (uint32_t)cuSeqlensGm_.GetValue(batch_i);
            uint32_t copyFlag = 0;
            uint32_t stateBlockIdx = 0;
            for (uint32_t head_i = 0; head_i < NV_; head_i++) {
                if (!IsCurrentBlock(seq1 - seq0)) {
                    continue;
                }
                copyFlag++;
                if (copyFlag == 1) {
                    // 每个batch首次计算时才计算stateOffset, 可以减少scalar计算量
                    uint32_t stateIdx = seq0;
                    if (hasAcceptedTokens_) {
                        stateIdx += numAcceptedTokensGm_.GetValue(batch_i) - 1;
                    }
                    stateBlockIdx = ssmStateIndicesGm_.GetValue(stateIdx);
                }
                ProcessHead(seq0, seq1, head_i, stateBlockIdx);
            }
        }
    }

private:
    __aicore__ inline void CopyInGamaK(uint32_t tIdxStart, uint32_t tIdxEnd, uint32_t head_i)
    {
        if (hasGamaK_) {
            uint64_t gkGmOffset = (uint64_t)tIdxStart * NV_ * DK_ + (uint64_t)head_i * DK_;
            uint32_t dealRowCount = tIdxEnd - tIdxStart;
            uint32_t actDataLen = DK_;
            uint64_t srcRowStride = NV_ * DK_;
            uint64_t dstRowStride = alignDK_;
            LocalTensor<float> gamaKInUb = inputQueue1_.AllocTensor<float>();
            CopySingleMatrixNDToND(gamaKInUb, gamaKGm_[gkGmOffset], dealRowCount, actDataLen, srcRowStride,
                                   dstRowStride);
            inputQueue1_.EnQue<float>(gamaKInUb);
            inputQueue1_.DeQue<float>();

            if (DK_ != alignDK_) {
                Duplicate(gamaUb_, (float)0, (tIdxEnd - tIdxStart) * alignDK_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            uint32_t blockElemNum = BLOCK_SIZE_32B / sizeof(float);
            DataCopyParams repeatParams;
            repeatParams.blockCount = static_cast<uint16_t>(dealRowCount);
            repeatParams.blockLen = Ceil(DK_, blockElemNum);
            repeatParams.srcStride = (dstRowStride - DK_) / blockElemNum;
            repeatParams.dstStride = (dstRowStride - DK_) / blockElemNum;
            DataCopy(gamaUb_, gamaKInUb, repeatParams);
            inputQueue1_.FreeTensor(gamaKInUb);
        } else {
            Duplicate(gamaUb_, (float)0, (tIdxEnd - tIdxStart) * alignDK_);
        }
    }

    __aicore__ inline void CopyInQK(uint32_t tIdxStart, uint32_t tIdxEnd, uint32_t head_i)
    {
        uint32_t kHeadIdx = head_i / (NV_ / NK_);
        uint64_t qkGmOffset = (uint64_t)tIdxStart * NK_ * DK_ + (uint64_t)kHeadIdx * DK_;
        uint64_t vGmOffset = (uint64_t)tIdxStart * NV_ * DV_ + (uint64_t)head_i * DV_;

        uint32_t dealRowCount = tIdxEnd - tIdxStart;
        uint32_t actDataLen = DK_;
        uint64_t srcRowStride = NK_ * DK_;
        uint64_t dstRowStride = alignDK_;
        LocalTensor<inType> queryInUb = inputQueue1_.AllocTensor<inType>();
        CopySingleMatrixNDToND(queryInUb, queryGm_[qkGmOffset], dealRowCount, actDataLen, srcRowStride, dstRowStride);
        inputQueue1_.EnQue<inType>(queryInUb);
        inputQueue1_.DeQue<inType>();
        Cast(queryUb_, queryInUb, RoundMode::CAST_NONE, dealRowCount * dstRowStride);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(queryUb_, queryUb_, scale_, dealRowCount * dstRowStride);
        inputQueue1_.FreeTensor(queryInUb);

        LocalTensor<inType> keyInUb = inputQueue1_.AllocTensor<inType>();
        CopySingleMatrixNDToND(keyInUb, keyGm_[qkGmOffset], dealRowCount, actDataLen, srcRowStride, dstRowStride);
        inputQueue1_.EnQue<inType>(keyInUb);
        inputQueue1_.DeQue<inType>();
        Cast(keyUb_, keyInUb, RoundMode::CAST_NONE, dealRowCount * dstRowStride);
        inputQueue1_.FreeTensor(keyInUb);
    }
    __aicore__ inline void CopyInV(uint32_t tIdxStart, uint32_t tIdxEnd, uint32_t head_i)
    {
        uint64_t vGmOffset = (uint64_t)tIdxStart * NV_ * DV_ + (uint64_t)head_i * DV_;
        uint32_t dealRowCount = tIdxEnd - tIdxStart;
        uint32_t actDataLen = DV_;
        uint64_t srcRowStride = NV_ * DV_;
        uint64_t dstRowStride = alignDV_;
        LocalTensor<inType> valueInUb = inputQueue1_.AllocTensor<inType>();
        CopySingleMatrixNDToND(valueInUb, valueGm_[vGmOffset], dealRowCount, actDataLen, srcRowStride, dstRowStride);
        inputQueue1_.EnQue<inType>(valueInUb);
        inputQueue1_.DeQue<inType>();
        Cast(valueUb_, valueInUb, RoundMode::CAST_NONE, dealRowCount * dstRowStride);
        inputQueue1_.FreeTensor(valueInUb);
    }

    __aicore__ inline void CopyInState(uint32_t stateBlockIdx, uint32_t head_i, uint32_t dvIdx, uint32_t curSingleV)
    {
        uint64_t stateGmOffset =
            (uint64_t)stateBlockIdx * stateStride0_ + (uint64_t)head_i * stateStride1_ + (uint64_t)dvIdx * DK_;
        uint32_t dealRowCount = curSingleV;
        uint32_t actDataLen = DK_;
        uint64_t srcRowStride = DK_;
        uint64_t dstRowStride = alignDK_;
        LocalTensor<stateType> stateInUb = inputQueue1_.AllocTensor<stateType>();
        CopySingleMatrixNDToND(stateInUb, initStateGm_[stateGmOffset], dealRowCount, actDataLen, srcRowStride,
                               dstRowStride);
        inputQueue1_.EnQue<stateType>(stateInUb);
        inputQueue1_.DeQue<stateType>();
        if constexpr (!std::is_same_v<stateType, float>) {
            Cast(stateUb_, stateInUb, RoundMode::CAST_NONE, dealRowCount * dstRowStride);
        } else {
            DataCopy(stateUb_, stateInUb, dealRowCount * dstRowStride);
        }
        inputQueue1_.FreeTensor(stateInUb);
    }

    __aicore__ inline void UpdateGamaBeta(uint32_t tIdx, uint32_t bSeqIdx, uint32_t head_i, uint32_t dvIdx)
    {
        uint64_t gamaBetaGmOffset = tIdx * NV_ + head_i;
        if (dvIdx == 0) {
            if (hasGama_) {
                gama_ = gamaGm_.GetValue(gamaBetaGmOffset);
                Adds(gamaUb_[bSeqIdx * alignDK_], gamaUb_[bSeqIdx * alignDK_], gama_, alignDK_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Exp(gamaUb_[bSeqIdx * alignDK_], gamaUb_[bSeqIdx * alignDK_], alignDK_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        beta_ = Cast(betaGm_.GetValue(gamaBetaGmOffset));
    }

    __aicore__ inline void CopyOutState(LocalTensor<float> &outStateFp32Ub, uint32_t tIdx, uint32_t head_i,
                                        uint32_t dvIdx, uint32_t curSingleV)
    {
        LocalTensor<stateType> stateOutLocal = stateOutQueue_.AllocTensor<stateType>();
        if constexpr (!std::is_same_v<stateType, float>) {
            Cast(stateOutLocal, outStateFp32Ub, AscendC::RoundMode::CAST_RINT, alignDK_ * curSingleV);
        } else {
            DataCopy(stateOutLocal, outStateFp32Ub, alignDK_ * curSingleV);
        }
        stateOutQueue_.EnQue<stateType>(stateOutLocal);
        stateOutQueue_.DeQue<stateType>();
        DataCopyExtParams stateOutParams;
        stateOutParams.blockCount = curSingleV;
        stateOutParams.blockLen = DK_ * sizeof(stateType);
        stateOutParams.srcStride = (alignDK_ - DK_) / (BLOCK_SIZE_32B / sizeof(stateType));
        stateOutParams.dstStride = 0;
        uint64_t outStateGmOffset = (uint64_t)ssmStateIndicesGm_.GetValue(tIdx) * stateStride0_ +
                                    (uint64_t)head_i * stateStride1_ + (uint64_t)dvIdx * DK_;
        DataCopyPad(finalStateGm_[outStateGmOffset], stateOutLocal, stateOutParams);
        stateOutQueue_.FreeTensor(stateOutLocal);
    }

    __aicore__ inline void CopyAttentionOut(LocalTensor<float> &attnOutFp32Ub, uint32_t tIdx, uint32_t head_i,
                                            uint32_t dvIdx, uint32_t curSingleV)
    {
        LocalTensor<outType> attnOutLocal = attnOutQueue_.AllocTensor<outType>();
        Cast(attnOutLocal, attnOutFp32Ub, AscendC::RoundMode::CAST_RINT, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        attnOutQueue_.EnQue<outType>(attnOutLocal);
        attnOutQueue_.DeQue<outType>();
        DataCopyExtParams attnOutParams;
        attnOutParams.blockCount = 1;
        attnOutParams.blockLen = curSingleV * sizeof(outType);
        attnOutParams.srcStride = 0;
        attnOutParams.dstStride = 0;
        uint64_t attnOutGmOffset = (uint64_t)tIdx * NV_ * DV_ + (uint64_t)head_i * DV_ + dvIdx;
        DataCopyPad(attnOutGm_[attnOutGmOffset], attnOutLocal, attnOutParams);
        attnOutQueue_.FreeTensor(attnOutLocal);
    }

    __aicore__ inline void Compute(uint32_t seq0, uint32_t seq1, uint32_t tIdx, uint32_t head_i, uint32_t dvIdx,
                                   uint32_t curSingleV)
    {
        uint32_t bSeqIdx = tIdx - seq0;
        uint32_t stateShape[2] = {curSingleV, alignDK_};
        VecMulMatVF(stateUb_, gamaUb_[bSeqIdx * alignDK_], stateUb_, curSingleV, alignDK_);
        if (dvIdx == 0 && bSeqIdx == 0) {
            CopyInQK(seq0, seq1, head_i);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // VecMulMat + RowSum
        LocalTensor<float> tmpUb = tmpBuff2_.Get<float>();
        VecMulMatVF(tmpUb, keyUb_[bSeqIdx * alignDK_], stateUb_, curSingleV, alignDK_);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> reduceSumResUb = tmpBuff1_.Get<float>();
        /* ReduceSum高阶API接口说明:
         * 1.isReuseSource设置成true时,接口不再申请额外UB内存
         * 2.接口当前reduce的列数是stateShape[1],不支持实际需参与计算的有效数据列数与UB上存储列数不相等的情况.
         *   此处stateShape[1]可以设置成alignDK_,而不设置成DK_是因为:
         *   拷贝queryGm&keyGm时无效数据pad为0了,计算后tmpUb每行的无效列也是0
         */
        ReduceSum<float, Pattern::Reduce::AR, true>(reduceSumResUb, tmpUb, stateShape, true);
        if (dvIdx == 0 && bSeqIdx == 0) {
            CopyInV(seq0, seq1, head_i);
        }
        AscendC::PipeBarrier<PIPE_V>();
        Sub(reduceSumResUb, valueUb_[bSeqIdx * alignDV_ + dvIdx], reduceSumResUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(reduceSumResUb, reduceSumResUb, beta_, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();

        // Outer + Add
        OuterAddVF(stateUb_, reduceSumResUb, keyUb_[bSeqIdx * alignDK_], stateUb_, curSingleV, alignDK_);
        AscendC::PipeBarrier<PIPE_V>();
        CopyOutState(stateUb_, tIdx, head_i, dvIdx, curSingleV);

        // State * query * scale
        VecMulMatVF(tmpUb, queryUb_[bSeqIdx * alignDK_], stateUb_, curSingleV, alignDK_);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum<float, Pattern::Reduce::AR, true>(reduceSumResUb, tmpUb, stateShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        CopyAttentionOut(reduceSumResUb, tIdx, head_i, dvIdx, curSingleV);
    }

    __aicore__ inline bool IsCurrentBlock(int32_t seqlen)
    {
        load += seqlen;
        bool ret = (blockIdx == usedblk && seqlen > 0);
        if (load >= avgload) {
            load = 0;
            usedblk++;
        }
        return ret;
    }

    __aicore__ inline void ProcessHead(uint32_t seq0, uint32_t seq1, uint32_t head_i, uint32_t stateBlockIdx)
    {
        CopyInGamaK(seq0, seq1, head_i);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint64_t dvIdx = 0; dvIdx < DV_; dvIdx += vStep_) {
            uint32_t curSingleV = (dvIdx + vStep_ > DV_) ? (DV_ - dvIdx) : vStep_;
            CopyInState(stateBlockIdx, head_i, dvIdx, curSingleV);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t tIdx = seq0; tIdx < seq1; tIdx++) {
                UpdateGamaBeta(tIdx, tIdx - seq0, head_i, dvIdx);
                Compute(seq0, seq1, tIdx, head_i, dvIdx, curSingleV);
            }
        }
    }

private:
    TPipe *pipe_;
    GlobalTensor<inType> queryGm_;
    GlobalTensor<inType> keyGm_;
    GlobalTensor<inType> valueGm_;
    GlobalTensor<outType> attnOutGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> ssmStateIndicesGm_;
    GlobalTensor<inType> betaGm_;
    GlobalTensor<stateType> initStateGm_;
    GlobalTensor<float> gamaGm_;
    GlobalTensor<float> gamaKGm_;
    GlobalTensor<int32_t> numAcceptedTokensGm_;
    GlobalTensor<stateType> finalStateGm_;

    TQue<QuePosition::VECIN, 1> inputQueue1_;
    TQue<QuePosition::VECOUT, 1> attnOutQueue_;
    TQue<QuePosition::VECOUT, 1> stateOutQueue_;
    TBuf<TPosition::VECCALC> gamaBuff_;
    TBuf<TPosition::VECCALC> stateBuff_;
    TBuf<TPosition::VECCALC> queryBuff_;
    TBuf<TPosition::VECCALC> keyBuff_;
    TBuf<TPosition::VECCALC> valueBuff_;
    TBuf<TPosition::VECCALC> tmpBuff1_;
    TBuf<TPosition::VECCALC> tmpBuff2_;

    LocalTensor<float> gamaUb_;
    LocalTensor<float> stateUb_;
    LocalTensor<float> queryUb_;
    LocalTensor<float> keyUb_;
    LocalTensor<float> valueUb_;

    uint32_t B_;
    uint32_t T_;
    uint32_t NK_;
    uint32_t alignDK_;
    uint32_t DK_;
    uint32_t NV_;
    uint32_t alignDV_;
    uint32_t DV_;
    uint32_t vStep_;
    uint32_t load;
    uint32_t usedblk;
    uint32_t avgload;
    bool hasAcceptedTokens_;
    bool hasGama_;
    bool hasGamaK_;
    float scale_;
    uint64_t blockIdx;
    uint64_t stateStride0_;
    uint64_t stateStride1_;

    float gama_;
    float beta_;
};
} // namespace RecurrentGatedDeltaRule
#endif
