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
 * \file matmul_all_reduce_quant_mul_cast.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_QUANT_MUL_CAST_H
#define MATMUL_ALL_REDUCE_QUANT_MUL_CAST_H

namespace MatmulAllReduceQuantMulCastImpl {
using namespace AscendC;
using namespace std;

constexpr int32_t TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 = 192 * 1024;
constexpr uint32_t BYTE512_MATMUL_ALLREDUCE_INT8 = 512;
constexpr uint32_t DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8 = 2;
constexpr int32_t SCALE_BUFF_CNT_MATMUL_ALLREDUCE_INT8 = 4;
constexpr int32_t BUF_CNT_SPLIT_M_MATMUL_ALLREDUCE_INT8 = 13;
constexpr int32_t BUF_CNT_SPLIT_MN_MATMUL_ALLREDUCE_INT8 = 15;
constexpr uint32_t SPLIT_M_MATMUL_ALLREDUCE_INT8 = 1;
constexpr uint32_t SPLIT_MN_MATMUL_ALLREDUCE_INT8 = 2;
constexpr uint32_t BYTE32_ALIGN = 32;

template <class T>
class MatmulAllReduceQuantMulCast {
public:
    __aicore__ inline MatmulAllReduceQuantMulCast() {}
    __aicore__ inline void Init(TPipe *tPipe, uint32_t quantUbSize)
    {
        pipe_ = tPipe;
        uint32_t scaleUbSize = quantUbSize;
        if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            scaleUbSize = this->alginN_;
        }
        if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            pipe_->InitBuffer(queueScale1_, 1, scaleUbSize * sizeof(T));
            pipe_->InitBuffer(queueScale2_, 1, scaleUbSize * sizeof(T));
        } else {
            pipe_->InitBuffer(queueScale1_, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, scaleUbSize * sizeof(T));
            pipe_->InitBuffer(queueScale2_, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, scaleUbSize * sizeof(T));
        }
        pipe_->InitBuffer(queueX_, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, quantUbSize * sizeof(float)); // 开doubleBuffer
        pipe_->InitBuffer(queueZ_, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, quantUbSize * sizeof(int8_t));
        pipe_->InitBuffer(tBufFP16_, quantUbSize * sizeof(half));
        pipe_->InitBuffer(tBufFP32_, quantUbSize * sizeof(float));
        pipe_->InitBuffer(tBufScaleDivFP32_, scaleUbSize * sizeof(float));
    }

    __aicore__ inline void MulCastSplitM(const uint32_t quantUbSize, int64_t &blockAddrOffset, uint32_t &tileCalCntM,
                                         uint32_t &tailCalCntM, uint32_t &aivLoopNum)
    {
        uint32_t vectorIndex = GetBlockIdx();
        uint32_t singleAivM = this->tileRankM_ / this->aivCoreNum_;
        uint32_t aivAddOneIndex = this->aivCoreNum_ + 1;
        if (this->tileRankM_ % this->aivCoreNum_ != 0) {
            aivAddOneIndex = this->aivCoreNum_ - (this->tileRankM_ % this->aivCoreNum_);
        }
        uint32_t usedAivCoreIndex = aivCoreNum_ - aivAddOneIndex;

        if (singleAivM == 0) { // M小于核数，singleAivM为0，核计算行数更新及偏移计算
            if (vectorIndex < usedAivCoreIndex) {
                singleAivM += 1;
                blockAddrOffset = static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleAivM) *
                                  static_cast<int64_t>(this->N_);
            } else {
                return;
            }
        } else { // M大于核数，singleAivM>0，核计算行数更新及偏移计算
            if ((aivAddOneIndex < this->aivCoreNum_ + 1) && (vectorIndex >= aivAddOneIndex)) {
                singleAivM += 1;
                blockAddrOffset = static_cast<int64_t>(aivAddOneIndex) * static_cast<int64_t>(singleAivM - 1) *
                                      static_cast<int64_t>(this->N_) +
                                  static_cast<int64_t>(vectorIndex - aivAddOneIndex) *
                                      static_cast<int64_t>(singleAivM) * static_cast<int64_t>(this->N_);
            } else {
                blockAddrOffset = static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleAivM) *
                                  static_cast<int64_t>(this->N_);
            }
        }

        tileCalCntM = quantUbSize / this->alginN_;
        aivLoopNum = singleAivM / tileCalCntM; // M很小的时候，这个值是0
        if (singleAivM % tileCalCntM != 0) {
            aivLoopNum += 1;
            tailCalCntM = singleAivM % tileCalCntM; // M很小的时候，直接算一次尾块大小N
        }
    }

    template <int commMode>
    __aicore__ inline void MulCastSplitMAndN(typename HcclTypeSelector<commMode>::type &hccl, int32_t &quantUbSize,
                                             uint32_t &needAivCoreNum, uint32_t &tileBlockCnt, uint32_t &tailBlockCnt,
                                             uint32_t &aivLoopNum, uint32_t &blockNumPerRow)
    {
        quantUbSize = TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 /
                      (BUF_CNT_SPLIT_MN_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T)));
        quantUbSize = quantUbSize / static_cast<int32_t>(BYTE512_MATMUL_ALLREDUCE_INT8) *
                      static_cast<int32_t>(BYTE512_MATMUL_ALLREDUCE_INT8);
        if (quantUbSize >= static_cast<int32_t>(this->N_)) {
            tileBlockCnt = this->N_;
            quantUbSize = static_cast<int32_t>(this->N_);
        } else {
            blockNumPerRow = Ceil(this->N_, static_cast<uint32_t>(quantUbSize));
            tileBlockCnt = static_cast<uint32_t>(quantUbSize);
            if ((this->N_ % static_cast<uint32_t>(quantUbSize)) != 0) {
                tailBlockCnt = this->N_ % static_cast<uint32_t>(quantUbSize);
            }
        }
        uint32_t rankM = this->M_ / hccl.GetRankDim();
        needAivCoreNum = blockNumPerRow * rankM;

        aivLoopNum = needAivCoreNum / this->aivCoreNum_;
        if ((needAivCoreNum % this->aivCoreNum_) != 0) {
            aivLoopNum += 1;
        }
    }

    template <int commMode>
    __aicore__ inline void InitInner(typename HcclTypeSelector<commMode>::type &hccl, uint32_t loopIdx,
                                     int64_t blockAddrOffsetSplitM, int64_t blockAddrOffsetSplitMN,
                                     int64_t scaleAddrOffsetSplitMN, uint32_t blockCntSpiltMN)
    {
        uint64_t curScaleCnt = 0;
        int64_t curBlockAddrOffset = 0;
        int64_t curScaleAddrOffset = 0;

        if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            curBlockAddrOffset = blockAddrOffsetSplitM;
            curScaleCnt = this->N_;
        } else {
            curBlockAddrOffset = blockAddrOffsetSplitMN;
            curScaleCnt = blockCntSpiltMN;
            curScaleAddrOffset = scaleAddrOffsetSplitMN;
        }
        tempBufferInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(this->inputGM_) + curBlockAddrOffset,
                                        this->tileRankM_ * this->N_ - curBlockAddrOffset); // 只计算reduceScatter结果
        tempBufferOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(this->outputGM_) + curBlockAddrOffset,
                                         this->tileRankM_ * this->N_ - curBlockAddrOffset); // 存放输出结果
        if (((loopIdx == 0) && (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8)) ||
            (this->splitMode_ == SPLIT_MN_MATMUL_ALLREDUCE_INT8)) {
            quantScale1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(this->quantScale1GM_) + curScaleAddrOffset,
                                           curScaleCnt);
            quantScale2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(this->quantScale2GM_) + curScaleAddrOffset,
                                           curScaleCnt);
        }
    }

    // 将 scale1 和 scale2 搬运到 ub 并计算 s2/s1
    __aicore__ inline void CopyAndCalScale(uint32_t copyBlockLen, DataCopyParams &copyParamsScaleQuant,
                                           DataCopyPadParams &padParams)
    {
        scale1Local_ = queueScale1_.AllocTensor<T>();
        scale2Local_ = queueScale2_.AllocTensor<T>();
        DataCopyPad(scale1Local_, quantScale1Gm_, copyParamsScaleQuant, padParams); // commaQuantScale1 Move to UB
        DataCopyPad(scale2Local_, quantScale2Gm_, copyParamsScaleQuant, padParams); // commaQuantScale2 Move to UB
        queueScale1_.EnQue(scale1Local_);
        queueScale2_.EnQue(scale2Local_);
        scaleFp32Local_ = tBufScaleDivFP32_.Get<float>(); // 无论是 bf16 还是 fp16 都直接 cast 成 fp32 进行计算
        LocalTensor<float> tempFp32Local = tBufFP32_.Get<float>();
        LocalTensor<T> qLocalScale1 = queueScale1_.DeQue<T>();
        LocalTensor<T> qLocalScale2 = queueScale2_.DeQue<T>();
        Cast(scaleFp32Local_, qLocalScale1, RoundMode::CAST_NONE, copyBlockLen); // bf16/fp16->fp32
        Cast(tempFp32Local, qLocalScale2, RoundMode::CAST_NONE, copyBlockLen);   // bf16/fp16->fp32
        PipeBarrier<PIPE_V>();
        Div(scaleFp32Local_, scaleFp32Local_, tempFp32Local, copyBlockLen);
        PipeBarrier<PIPE_V>();
        return;
    }

    __aicore__ inline void Process(uint32_t loopIdx, uint32_t aivLoopNum, uint32_t curBlockCntM,
                                   uint32_t blockCntSpiltMN)
    {
        uint32_t calcCnt = blockCntSpiltMN;
        uint16_t copyBlockCnt = 1;
        uint32_t copyBlockLen = calcCnt;
        uint16_t copyInUbStride = 0;
        uint16_t copyOutUbStride = 0;
        if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // 搬多行
            calcCnt = this->alginN_ * curBlockCntM;
            copyBlockCnt = curBlockCntM;
            copyBlockLen = this->N_;
            copyInUbStride =
                (this->alginN_ * sizeof(float) - Ceil(this->N_ * sizeof(float), BYTE32_ALIGN) * BYTE32_ALIGN) /
                BYTE32_ALIGN;
            copyOutUbStride =
                (this->alginN_ * sizeof(int8_t) - Ceil(this->N_ * sizeof(int8_t), BYTE32_ALIGN) * BYTE32_ALIGN) /
                BYTE32_ALIGN;
        }
        DataCopyParams copyParamsCurBlock = {copyBlockCnt, static_cast<uint16_t>(copyBlockLen * sizeof(float)), 0,
                                             copyInUbStride};
        DataCopyParams copyOutParamsCurBlock = {copyBlockCnt, static_cast<uint16_t>(copyBlockLen * sizeof(int8_t)),
                                                copyOutUbStride, 0};
        DataCopyParams copyParamsScaleQuant = {1, static_cast<uint16_t>(copyBlockLen * sizeof(T)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        // s2/s1 Div + Cast 成 fp32，保存下来
        if (((loopIdx == 0) && (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8)) ||
            (this->splitMode_ == SPLIT_MN_MATMUL_ALLREDUCE_INT8)) {
            CopyAndCalScale(copyBlockLen, copyParamsScaleQuant, padParams);
            queueScale1_.FreeTensor(scale1Local_);
            queueScale2_.FreeTensor(scale2Local_);
        }

        // reduceScatter res Move to UB
        LocalTensor<float> reduceScatterLocal = queueX_.AllocTensor<float>();
        DataCopyPad(reduceScatterLocal, tempBufferInGm_, copyParamsCurBlock, padParams);
        queueX_.EnQue(reduceScatterLocal);
        LocalTensor<float> reduceScatterOutLocal = queueX_.DeQue<float>();

        // 乘以 S1/S2，再由 fp32->fp16->int8
        LocalTensor<half> zFp16Local = tBufFP16_.Get<half>();
        LocalTensor<int8_t> zInt8Local = queueZ_.AllocTensor<int8_t>();
        if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // 切M
            for (uint32_t i = 0; i < curBlockCntM; i++) {
                Mul(reduceScatterOutLocal[i * this->alginN_], reduceScatterOutLocal[i * this->alginN_], scaleFp32Local_,
                    this->N_);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Mul(reduceScatterOutLocal, reduceScatterOutLocal, scaleFp32Local_, calcCnt);
            PipeBarrier<PIPE_V>();
        }
        Cast(zFp16Local, reduceScatterOutLocal, RoundMode::CAST_NONE, calcCnt); // fp32->fp16
        PipeBarrier<PIPE_V>();
        Cast(zInt8Local, zFp16Local, RoundMode::CAST_NONE, calcCnt); // fp16->int8
        PipeBarrier<PIPE_V>();
        queueX_.FreeTensor(reduceScatterOutLocal);

        // 搬出，按照卡数搬到对应位置
        queueZ_.EnQue<int8_t>(zInt8Local);
        LocalTensor<int8_t> outLocal = queueZ_.DeQue<int8_t>();
        DataCopyPad(tempBufferOutGm_, outLocal, copyOutParamsCurBlock);
        queueZ_.FreeTensor(zInt8Local);
    }

    template <int commMode>
    __aicore__ inline void SetParams(GM_ADDR inputGM, GM_ADDR quantScale1GM, GM_ADDR quantScale2GM, GM_ADDR outputGM,
                                     uint32_t M, uint32_t N, typename HcclTypeSelector<commMode>::type &hccl,
                                     uint32_t aivCoreNum, uint32_t &tileBlockCnt, uint32_t &tailBlockCnt,
                                     uint32_t &aivLoopNum, int64_t &blockAddrOffset, uint32_t &tailCalCntM,
                                     uint32_t &tileCalCntM, uint32_t &needAivCoreNum, uint32_t &blockNumPerRow,
                                     int32_t &nowQuantUbSize)
    {
        int32_t nowAlginN = Ceil(N * sizeof(float), BYTE512_MATMUL_ALLREDUCE_INT8) * BYTE512_MATMUL_ALLREDUCE_INT8;
        nowQuantUbSize =
            (TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 - static_cast<int32_t>(nowAlginN) * static_cast<int32_t>(sizeof(T)) *
                                                      static_cast<int32_t>(SCALE_BUFF_CNT_MATMUL_ALLREDUCE_INT8)) /
            (BUF_CNT_SPLIT_M_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T)));
        nowQuantUbSize = (nowQuantUbSize / nowAlginN) * nowAlginN;
        this->M_ = M;
        this->N_ = N;
        this->alginN_ = static_cast<uint32_t>(nowAlginN);
        this->aivCoreNum_ = aivCoreNum;
        this->inputGM_ = inputGM;
        this->quantScale1GM_ = quantScale1GM;
        this->quantScale2GM_ = quantScale2GM;
        this->outputGM_ = outputGM;
        this->rankCnt_ = hccl.GetRankDim();
        this->tileRankM_ = M / hccl.GetRankDim();
        if (nowQuantUbSize > this->tileRankM_ * nowAlginN) {
            nowQuantUbSize = this->tileRankM_ * nowAlginN;
        }

        if (nowQuantUbSize >= nowAlginN) { // 分核切M， 不切N
            this->splitMode_ = SPLIT_M_MATMUL_ALLREDUCE_INT8;
            this->MulCastSplitM(nowQuantUbSize, blockAddrOffset, tileCalCntM, tailCalCntM, aivLoopNum);
        } else { // N超大，分核策略采取切MN
            this->splitMode_ = SPLIT_MN_MATMUL_ALLREDUCE_INT8;
            this->template MulCastSplitMAndN<commMode>(hccl, nowQuantUbSize, needAivCoreNum, tileBlockCnt, tailBlockCnt,
                                                       aivLoopNum, blockNumPerRow);
        }
    }

    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> queueX_;
    TQue<QuePosition::VECIN, 1> queueScale1_;
    TQue<QuePosition::VECIN, 1> queueScale2_;
    TQue<QuePosition::VECOUT, 1> queueZ_;
    TBuf<TPosition::VECCALC> tBufFP16_;
    TBuf<TPosition::VECCALC> tBufFP32_;
    TBuf<TPosition::VECCALC> tBufScaleDivFP32_;
    GlobalTensor<float> tempBufferInGm_;
    GlobalTensor<int8_t> tempBufferOutGm_;
    GlobalTensor<T> quantScale1Gm_;
    GlobalTensor<T> quantScale2Gm_;
    LocalTensor<float> scaleFp32Local_;
    uint32_t tileRankM_;
    uint32_t rankCnt_;
    uint32_t aivCoreNum_;
    uint32_t N_;
    uint32_t M_;
    uint32_t alginN_;
    uint32_t splitMode_;
    uint32_t vectorIndex_; // 未使用
    GM_ADDR inputGM_;
    GM_ADDR outputGM_;
    GM_ADDR quantScale1GM_;
    GM_ADDR quantScale2GM_;
    LocalTensor<T> scale1Local_;
    LocalTensor<T> scale2Local_;
};

template <class T, int commMode>
__aicore__ inline void MatmulAllReduceQuantMulCastCommInt8(GM_ADDR inputGM, GM_ADDR quantScale1GM,
                                                           GM_ADDR quantScale2GM, GM_ADDR outputGM, uint32_t M,
                                                           uint32_t N, TPipe *tPipe,
                                                           typename HcclTypeSelector<commMode>::type &hccl)
{
    uint32_t aivCoreNum = GetBlockNum() * GetTaskRation();
    if ((g_coreType == AIC) || (GetBlockIdx() >= aivCoreNum)) {
        return;
    }
    uint32_t tileBlockCnt = 0;
    uint32_t tailBlockCnt = 0;
    uint32_t aivLoopNum = 0;
    int64_t blockAddrOffset = 0;
    uint32_t tailCalCntM = 0;
    uint32_t tileCalCntM = 0;
    uint32_t needAivCoreNum = 0;
    uint32_t blockNumPerRow = 1;
    int32_t nowQuantUbSize = 0;

    tPipe->Reset();
    MatmulAllReduceQuantMulCast<T> op;
    op.template SetParams<commMode>(inputGM, quantScale1GM, quantScale2GM, outputGM, M, N, hccl, aivCoreNum,
                                    tileBlockCnt, tailBlockCnt, aivLoopNum, blockAddrOffset, tailCalCntM, tileCalCntM,
                                    needAivCoreNum, blockNumPerRow, nowQuantUbSize);

    if (aivLoopNum == 0) {
        return;
    }
    op.Init(tPipe, nowQuantUbSize);                               // 按照最大开
    for (uint32_t loopIdx = 0; loopIdx < aivLoopNum; loopIdx++) { // 一轮外层循环对应着一次核间并行
        int64_t blockAddrOffsetSplitM = 0;
        uint32_t blockCntM = tileCalCntM;
        uint32_t blockCntSpiltMN = tileBlockCnt;
        int64_t blockAddrOffsetSplitMN = 0;
        int64_t scaleAddrOffsetSplitMN = 0;
        if (op.splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            if ((tailCalCntM != 0) && (loopIdx == aivLoopNum - 1)) {
                blockCntM = tailCalCntM;
            }
            blockAddrOffsetSplitM = static_cast<int64_t>(blockAddrOffset) +
                                    static_cast<int64_t>(loopIdx) * static_cast<int64_t>(tileCalCntM) *
                                        static_cast<int64_t>(N); // 取到当前核的偏移，再+当前块的地址偏移
        } else {
            uint32_t globalBlockIdx = loopIdx * aivCoreNum + GetBlockIdx();
            if (globalBlockIdx > (needAivCoreNum - 1)) {
                return;
            }
            if ((tailBlockCnt != 0) && (globalBlockIdx % blockNumPerRow == blockNumPerRow - 1)) {
                blockCntSpiltMN = tailBlockCnt;
            }
            blockAddrOffsetSplitMN =
                static_cast<int64_t>(globalBlockIdx / blockNumPerRow) * static_cast<int64_t>(N) +
                static_cast<int64_t>(globalBlockIdx % blockNumPerRow) * static_cast<int64_t>(tileBlockCnt);
            scaleAddrOffsetSplitMN =
                static_cast<int64_t>(globalBlockIdx % blockNumPerRow) * static_cast<int64_t>(tileBlockCnt);
        }
        op.template InitInner<commMode>(hccl, loopIdx, blockAddrOffsetSplitM, blockAddrOffsetSplitMN,
                                        scaleAddrOffsetSplitMN, blockCntSpiltMN);
        op.Process(loopIdx, aivLoopNum, blockCntM, blockCntSpiltMN);
    }
}
} // namespace MatmulAllReduceQuantMulCastImpl

#endif // MATMUL_ALL_REDUCE_QUANT_REDUCE_SUM_H
