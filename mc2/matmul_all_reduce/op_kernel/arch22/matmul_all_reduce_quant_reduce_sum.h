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
 * \file matmul_all_reduce_quant_reduce_sum.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_REDUCE_SUM_H
#define MATMUL_ALL_REDUCE_QUANT_REDUCE_SUM_H

namespace MatmulAllReduceReduceSumImpl {
using namespace AscendC;
using namespace std;

constexpr int32_t TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 = 191 * 1024;
constexpr uint32_t BYTE512_MATMUL_ALLREDUCE_INT8 = 512;
constexpr uint32_t DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8 = 2;
constexpr int32_t SCALE_BUFF_CNT_MATMUL_ALLREDUCE_INT8 = 4;
constexpr int32_t BUF_CNT_SPLIT_M_MATMUL_ALLREDUCE_INT8 = 7;
constexpr int32_t BUF_CNT_SPLIT_MN_MATMUL_ALLREDUCE_INT8 = 13;
constexpr uint32_t SPLIT_M_MATMUL_ALLREDUCE_INT8 = 1;
constexpr uint32_t SPLIT_MN_MATMUL_ALLREDUCE_INT8 = 2;
constexpr uint32_t BYTE32_ALIGN = 32;

template <class T> // commQuantScaleType = yType bf16/fp16
class MatmulAllReduceReduceSum {
public:
    __aicore__ inline MatmulAllReduceReduceSum() {}
    __aicore__ inline void Init(TPipe *tPipe, uint32_t quantUbSize)
    {
        pipe = tPipe;
        uint32_t scaleUbSize = quantUbSize;
        if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            scaleUbSize = this->alginN;
        }
        if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            pipe->InitBuffer(queueScale1, 1, scaleUbSize * sizeof(T));
            pipe->InitBuffer(queueScale2, 1, scaleUbSize * sizeof(T));
        } else {
            pipe->InitBuffer(queueScale1, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, scaleUbSize * sizeof(T));
            pipe->InitBuffer(queueScale2, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, scaleUbSize * sizeof(T));
        }
        pipe->InitBuffer(queueX, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, quantUbSize * sizeof(int8_t)); // 开doubleBuffer
        pipe->InitBuffer(queueZ, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, quantUbSize * sizeof(int8_t));
        pipe->InitBuffer(tBufFP16, quantUbSize * sizeof(half));
        pipe->InitBuffer(tBufFP32_1, quantUbSize * sizeof(float));
        pipe->InitBuffer(tBufFP32_2, quantUbSize * sizeof(float));
        pipe->InitBuffer(tBufScaleDivFP32, scaleUbSize * sizeof(float));
    }

    __aicore__ inline void MatmulAllReduceReduceSumSplitM(const uint32_t quantUbSize, int64_t &blockAddrOffset,
                                                          uint32_t &tileCalCntM, uint32_t &tailCalCntM,
                                                          uint32_t &aivLoopNum)
    {
        uint32_t vectorIndex = GetBlockIdx();
        uint32_t singleAivM = this->tileRankM / this->aivCoreNum;
        uint32_t aivAddOneIndex = this->aivCoreNum + 1;
        if (this->tileRankM % this->aivCoreNum != 0) {
            aivAddOneIndex = this->aivCoreNum - (this->tileRankM % this->aivCoreNum);
        }
        uint32_t usedAivCoreIndex = aivCoreNum - aivAddOneIndex;

        if (singleAivM == 0) { // M小于核数，singleAivM为0，核计算行数更新及偏移计算
            if (vectorIndex < usedAivCoreIndex) {
                singleAivM += 1;
                blockAddrOffset = static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleAivM) *
                                  static_cast<int64_t>(this->N);
            } else {
                return;
            }
        } else { // M大于核数，singleAivM>0，核计算行数更新及偏移计算
            if ((aivAddOneIndex < this->aivCoreNum + 1) && (vectorIndex >= aivAddOneIndex)) {
                singleAivM += 1;
                blockAddrOffset = static_cast<int64_t>(aivAddOneIndex) * static_cast<int64_t>(singleAivM - 1) *
                                      static_cast<int64_t>(this->N) +
                                  static_cast<int64_t>(vectorIndex - aivAddOneIndex) *
                                      static_cast<int64_t>(singleAivM) * static_cast<int64_t>(this->N);
            } else {
                blockAddrOffset = static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleAivM) *
                                  static_cast<int64_t>(this->N);
            }
        }

        tileCalCntM = quantUbSize / this->alginN;
        aivLoopNum = singleAivM / tileCalCntM; // M很小的时候，这个值是0
        if (singleAivM % tileCalCntM != 0) {
            aivLoopNum += 1;
            tailCalCntM = singleAivM % tileCalCntM; // M很小的时候，直接算一次尾块大小N
        }
    }

    __aicore__ inline void MatmulAllReduceReduceSumSplitMAndN(int32_t &quantUbSize, uint32_t &needAivCoreNum,
                                                              uint32_t &tileBlockCnt, uint32_t &tailBlockCnt,
                                                              uint32_t &aivLoopNum, uint32_t &blockNumPerRow)
    {
        quantUbSize = TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 /
                      (BUF_CNT_SPLIT_MN_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T)));
        quantUbSize = quantUbSize / static_cast<int32_t>(BYTE512_MATMUL_ALLREDUCE_INT8) *
                      static_cast<int32_t>(BYTE512_MATMUL_ALLREDUCE_INT8);
        if (quantUbSize >= static_cast<int32_t>(this->N)) {
            tileBlockCnt = this->N;
            quantUbSize = static_cast<int32_t>(this->N);
        } else {
            blockNumPerRow = Ceil(this->N, static_cast<uint32_t>(quantUbSize));
            tileBlockCnt = static_cast<uint32_t>(quantUbSize);
            if ((this->N % static_cast<uint32_t>(quantUbSize)) != 0) {
                tailBlockCnt = this->N % static_cast<uint32_t>(quantUbSize);
            }
        }
        uint32_t rankM = this->M / this->hccl_.GetRankDim();
        needAivCoreNum = blockNumPerRow * rankM;

        aivLoopNum = needAivCoreNum / this->aivCoreNum;
        if ((needAivCoreNum % this->aivCoreNum) != 0) {
            aivLoopNum += 1;
        }
    }

    __aicore__ inline void InitInner(uint32_t loopIdx, int64_t blockAddrOffsetSplitM, int64_t blockAddrOffsetSplitMN,
                                     int64_t scaleAddrOffsetSplitMN, uint32_t blockCntSpiltMN)
    {
        uint64_t curScaleCnt = 0;
        int64_t curBlockAddrOffset = 0;
        int64_t curScaleAddrOffset = 0;

        if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            curBlockAddrOffset = blockAddrOffsetSplitM;
            curScaleCnt = this->N;
        } else {
            curBlockAddrOffset = blockAddrOffsetSplitMN;
            curScaleCnt = blockCntSpiltMN;
            curScaleAddrOffset = scaleAddrOffsetSplitMN;
        }
        tempBufferGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(this->tempBufferGM) + curBlockAddrOffset,
                                      this->M * this->N - curBlockAddrOffset);
        if (((loopIdx == 0) && (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8)) ||
            (this->splitMode == SPLIT_MN_MATMUL_ALLREDUCE_INT8)) {
            quantScale1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(this->quantScale1GM) + curScaleAddrOffset,
                                           curScaleCnt);
            quantScale2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(this->quantScale2GM) + curScaleAddrOffset,
                                           curScaleCnt);
        }
    }

    // 将 scale1 和 scale2 搬运到 ub 并计算 s2/s1
    __aicore__ inline void CopyAndCalScale(uint32_t copyBlockLen, DataCopyParams &copyParamsScaleQuant,
                                           DataCopyPadParams &padParams)
    {
        scale1Local = queueScale1.AllocTensor<T>();
        scale2Local = queueScale2.AllocTensor<T>();
        DataCopyPad(scale1Local, quantScale1Gm_, copyParamsScaleQuant, padParams); // commaQuantScale1 Move to UB
        DataCopyPad(scale2Local, quantScale2Gm_, copyParamsScaleQuant, padParams); // commaQuantScale2 Move to UB
        queueScale1.EnQue(scale1Local);
        queueScale2.EnQue(scale2Local);
        scaleFp32Local = tBufScaleDivFP32.Get<float>(); // 无论是 bf16 还是 fp16 都直接 cast 成 fp32 进行计算
        LocalTensor<float> tempFp32Local = tBufFP32_1.Get<float>();
        LocalTensor<T> qLocalScale1 = queueScale1.DeQue<T>();
        LocalTensor<T> qLocalScale2 = queueScale2.DeQue<T>();
        Cast(scaleFp32Local, qLocalScale1, RoundMode::CAST_NONE, copyBlockLen); // bf16/fp16->fp32
        Cast(tempFp32Local, qLocalScale2, RoundMode::CAST_NONE, copyBlockLen);  // bf16/fp16->fp32
        PipeBarrier<PIPE_V>();
        Div(scaleFp32Local, scaleFp32Local, tempFp32Local, copyBlockLen);
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
        if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // 搬多行
            calcCnt = this->alginN * curBlockCntM;
            copyBlockCnt = curBlockCntM;
            copyBlockLen = this->N;
            copyInUbStride =
                (this->alginN * sizeof(int8_t) - Ceil(this->N * sizeof(int8_t), BYTE32_ALIGN) * BYTE32_ALIGN) /
                BYTE32_ALIGN;
            copyOutUbStride =
                (this->alginN * sizeof(int8_t) - Ceil(this->N * sizeof(int8_t), BYTE32_ALIGN) * BYTE32_ALIGN) /
                BYTE32_ALIGN;
        }
        DataCopyParams copyParamsCurBlock = {copyBlockCnt, static_cast<uint16_t>(copyBlockLen * sizeof(int8_t)), 0,
                                             copyInUbStride};
        DataCopyParams copyOutParamsCurBlock = {copyBlockCnt, static_cast<uint16_t>(copyBlockLen * sizeof(int8_t)),
                                                copyOutUbStride, 0};
        DataCopyParams copyParamsScaleQuant = {1, static_cast<uint16_t>(copyBlockLen * sizeof(T)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        // s2/s1 Div + Cast 成 fp32，保存下来
        if (((loopIdx == 0) && (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8)) ||
            (this->splitMode == SPLIT_MN_MATMUL_ALLREDUCE_INT8)) {
            CopyAndCalScale(copyBlockLen, copyParamsScaleQuant, padParams);
        }

        // all2all int8->fp16->fp32
        LocalTensor<int8_t> allToAllLocal = queueX.AllocTensor<int8_t>();
        DataCopyPad(allToAllLocal, tempBufferGm_, copyParamsCurBlock, padParams); // all2all res Move to UB
        queueX.EnQue(allToAllLocal);
        LocalTensor<half> xFp16Local = tBufFP16.Get<half>();
        LocalTensor<float> xFp32Local = tBufFP32_1.Get<float>();
        LocalTensor<int8_t> xInt8Local = queueX.DeQue<int8_t>();
        Cast(xFp16Local, xInt8Local, RoundMode::CAST_NONE, calcCnt);
        PipeBarrier<PIPE_V>();
        Cast(xFp32Local, xFp16Local, RoundMode::CAST_NONE, calcCnt);
        PipeBarrier<PIPE_V>();
        queueX.FreeTensor(allToAllLocal); // free 首块 xInt8Local

        // 偏移取其余核进行相加 Cast + Add
        for (uint32_t i = 1; i < this->rankCnt; i++) {
            LocalTensor<int8_t> allToAllAddLocal = queueX.AllocTensor<int8_t>();
            DataCopyPad(allToAllAddLocal, tempBufferGm_[i * this->tileRankM * this->N], copyParamsCurBlock,
                        padParams); // 其余卡数，对应位置 Move to UB
            queueX.EnQue(allToAllAddLocal);
            LocalTensor<int8_t> xAddInt8Local = queueX.DeQue<int8_t>();
            LocalTensor<half> xAddFp16Local = tBufFP16.Get<half>();
            Cast(xAddFp16Local, xAddInt8Local, RoundMode::CAST_NONE, calcCnt); // int8->fp16
            PipeBarrier<PIPE_V>();
            queueX.FreeTensor(allToAllAddLocal); // free 对应 rank 内容
            LocalTensor<float> xAddFp32Local = tBufFP32_2.Get<float>();
            Cast(xAddFp32Local, xAddFp16Local, RoundMode::CAST_NONE, calcCnt); // fp16->fp32
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, xAddFp32Local, calcCnt);
            PipeBarrier<PIPE_V>();
        }

        // 乘以 S1/S2，在由 fp32->fp16->int8
        LocalTensor<half> zFp16Local = tBufFP16.Get<half>();
        LocalTensor<int8_t> zInt8Local = queueZ.AllocTensor<int8_t>();
        if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // 切M
            for (uint32_t i = 0; i < curBlockCntM; i++) {
                Mul(xFp32Local[i * this->alginN], xFp32Local[i * this->alginN], scaleFp32Local, this->N);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Mul(xFp32Local, xFp32Local, scaleFp32Local, calcCnt);
            PipeBarrier<PIPE_V>();
        }
        Cast(zFp16Local, xFp32Local, RoundMode::CAST_NONE, calcCnt); // fp32->fp16
        PipeBarrier<PIPE_V>();
        Cast(zInt8Local, zFp16Local, RoundMode::CAST_NONE, calcCnt); // fp16->int8
        PipeBarrier<PIPE_V>();

        // 搬出，按照卡数搬到对应位置
        uint32_t rank_id = this->hccl_.GetRankId();
        queueZ.EnQue<int8_t>(zInt8Local);
        LocalTensor<int8_t> outLocal = queueZ.DeQue<int8_t>();
        DataCopyPad(tempBufferGm_[rank_id * this->tileRankM * this->N], outLocal, copyOutParamsCurBlock);
        queueZ.FreeTensor(zInt8Local);
        if (((loopIdx == (aivLoopNum - 1)) && (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8)) ||
            (this->splitMode == SPLIT_MN_MATMUL_ALLREDUCE_INT8)) {
            queueScale1.FreeTensor(scale1Local);
            queueScale2.FreeTensor(scale2Local);
        }
    }

    __aicore__ inline void SetParams(GM_ADDR tempBufferGM, GM_ADDR quantScale1GM, GM_ADDR quantScale2GM, uint32_t M,
                                     uint32_t N, Hccl<HCCL_SERVER_TYPE_AICPU> &hccl_, uint32_t aivCoreNum,
                                     uint32_t &tileBlockCnt, uint32_t &tailBlockCnt, uint32_t &aivLoopNum,
                                     int64_t &blockAddrOffset, uint32_t &tailCalCntM, uint32_t &tileCalCntM,
                                     uint32_t &needAivCoreNum, uint32_t &blockNumPerRow, int32_t &nowQuantUbSize)
    {
        int32_t nowAlginN = Ceil(N * sizeof(int8_t), BYTE512_MATMUL_ALLREDUCE_INT8) * BYTE512_MATMUL_ALLREDUCE_INT8;
        nowQuantUbSize =
            (TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 - static_cast<int32_t>(nowAlginN) * static_cast<int32_t>(sizeof(T)) *
                                                      static_cast<int32_t>(SCALE_BUFF_CNT_MATMUL_ALLREDUCE_INT8)) /
            (BUF_CNT_SPLIT_M_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T)));
        nowQuantUbSize = (nowQuantUbSize / nowAlginN) * nowAlginN;
        this->M = M;
        this->N = N;
        this->alginN = static_cast<uint32_t>(nowAlginN);
        this->aivCoreNum = aivCoreNum;
        this->tempBufferGM = tempBufferGM;
        this->quantScale1GM = quantScale1GM;
        this->quantScale2GM = quantScale2GM;
        this->hccl_ = hccl_;
        this->rankCnt = hccl_.GetRankDim();
        this->tileRankM = M / hccl_.GetRankDim();
        if (nowQuantUbSize > this->tileRankM * nowAlginN) {
            nowQuantUbSize = this->tileRankM * nowAlginN;
        }

        if (nowQuantUbSize >= nowAlginN) { // 分核切M， 不切N
            this->splitMode = SPLIT_M_MATMUL_ALLREDUCE_INT8;
            this->MatmulAllReduceReduceSumSplitM(nowQuantUbSize, blockAddrOffset, tileCalCntM, tailCalCntM, aivLoopNum);
        } else { // N超大，分核策略采取切MN
            this->splitMode = SPLIT_MN_MATMUL_ALLREDUCE_INT8;
            this->MatmulAllReduceReduceSumSplitMAndN(nowQuantUbSize, needAivCoreNum, tileBlockCnt, tailBlockCnt,
                                                     aivLoopNum, blockNumPerRow);
        }
    }

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> queueX;
    TQue<QuePosition::VECIN, 1> queueScale1;
    TQue<QuePosition::VECIN, 1> queueScale2;
    TQue<QuePosition::VECOUT, 1> queueZ;
    TBuf<TPosition::VECCALC> tBufFP16;
    TBuf<TPosition::VECCALC> tBufFP32_1;
    TBuf<TPosition::VECCALC> tBufFP32_2;
    TBuf<TPosition::VECCALC> tBufScaleDivFP32;
    GlobalTensor<int8_t> tempBufferGm_;
    GlobalTensor<T> quantScale1Gm_;
    GlobalTensor<T> quantScale2Gm_;
    LocalTensor<float> scaleFp32Local;
    uint32_t tileRankM;
    uint32_t rankCnt;
    uint32_t aivCoreNum;
    uint32_t N;
    uint32_t M;
    uint32_t alginN;
    uint32_t splitMode;
    uint32_t vectorIndex;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    GM_ADDR tempBufferGM;
    GM_ADDR quantScale1GM;
    GM_ADDR quantScale2GM;
    LocalTensor<T> scale1Local;
    LocalTensor<T> scale2Local;
};

template <class T>
__aicore__ inline void MatmulAllReduceReduceSumInt8(GM_ADDR tempBufferGM, GM_ADDR quantScale1GM, GM_ADDR quantScale2GM,
                                                    uint32_t M, uint32_t N, TPipe *tPipe,
                                                    Hccl<HCCL_SERVER_TYPE_AICPU> &hccl_)
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
    MatmulAllReduceReduceSum<T> op;
    op.SetParams(tempBufferGM, quantScale1GM, quantScale2GM, M, N, hccl_, aivCoreNum, tileBlockCnt, tailBlockCnt,
                 aivLoopNum, blockAddrOffset, tailCalCntM, tileCalCntM, needAivCoreNum, blockNumPerRow, nowQuantUbSize);

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
        uint32_t globalBlockIdx = 0;
        if (op.splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            if ((tailCalCntM != 0) && (loopIdx == aivLoopNum - 1)) {
                blockCntM = tailCalCntM;
            }
            blockAddrOffsetSplitM = static_cast<int64_t>(blockAddrOffset) +
                                    static_cast<int64_t>(loopIdx) * static_cast<int64_t>(tileCalCntM) *
                                        static_cast<int64_t>(N); // 取到当前核的偏移，再+当前块的地址偏移
        } else {
            globalBlockIdx = loopIdx * aivCoreNum + GetBlockIdx();
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
        op.InitInner(loopIdx, blockAddrOffsetSplitM, blockAddrOffsetSplitMN, scaleAddrOffsetSplitMN, blockCntSpiltMN);
        op.Process(loopIdx, aivLoopNum, blockCntM, blockCntSpiltMN);
    }
}
} // namespace MatmulAllReduceReduceSumImpl

#endif // MATMUL_ALL_REDUCE_QUANT_REDUCE_SUM_H
