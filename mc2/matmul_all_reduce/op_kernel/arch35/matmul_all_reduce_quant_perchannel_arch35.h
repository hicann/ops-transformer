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
 * \file matmul_all_reduce_quant_perchannel_arch35.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_QUANT_PERCHANNEL_ARCH35_H
#define MATMUL_ALL_REDUCE_QUANT_PERCHANNEL_ARCH35_H

namespace MatmulAllReduceQuantPerchannelImpl {
using namespace AscendC;
using namespace std;

constexpr int32_t TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 = 192 * 1024;
constexpr uint32_t DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8 = 2;
constexpr int32_t BF16_BUF_CNT_MATMUL_ALLREDUCE_INT8 = 8;
constexpr int32_t FP16_BUF_CNT_MATMUL_ALLREDUCE_INT8 = 4;
constexpr int32_t FP16_BUF_SPLIT_MN_CNT_MATMUL_ALLREDUCE_INT8 = 6;
constexpr int32_t BF16_BUF_SPLIT_MN_CNT_MATMUL_ALLREDUCE_INT8 = 10;
constexpr uint32_t SPLIT_M_MATMUL_ALLREDUCE_INT8 = 1;
constexpr uint32_t SPLIT_MN_MATMUL_ALLREDUCE_INT8 = 2;
constexpr uint32_t BYTE32_MATMUL_ALLREDUCE_INT8 = 32;
constexpr uint32_t BYTE512_MATMUL_ALLREDUCE_INT8 = 512;

template <class T>
class MatmulAllReduceQuantPerchannel {
public:
    __aicore__ inline MatmulAllReduceQuantPerchannel() {}
    __aicore__ inline void Init(TPipe *tPipe, uint32_t quantUbSize)
    {
        pipe_ = tPipe;
        uint32_t scaleUbSize = quantUbSize;
        if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            scaleUbSize = quantAlginN_;
        }
        if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            pipe_->InitBuffer(inQueueS_, 1, scaleUbSize * sizeof(T)); // quant_scale，X和S的大小相同，行向量一一对应
        } else {
            pipe_->InitBuffer(inQueueS_, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8,
                              scaleUbSize * sizeof(T)); // quant_scale，X和S的大小相同，行向量一一对应
        }
        pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, quantUbSize * sizeof(T)); // mmOut
        pipe_->InitBuffer(outQueueZ_, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8,
                          quantUbSize * sizeof(int8_t));                 // quant结果
        pipe_->InitBuffer(outHalfLocalTmp_, quantUbSize * sizeof(half)); // div的中间结果
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {       // bf16转换成half做div
            pipe_->InitBuffer(xFloatLocalTmp_, quantUbSize * sizeof(float));
            pipe_->InitBuffer(sFloatLocalTmp_, quantUbSize * sizeof(float));
        }
    }

    __aicore__ inline void InitInner(uint32_t loopIdx, int64_t blockAddrOffsetSplitM, uint32_t blockCntM,
                                     int64_t blockAddrOffsetSplitMN, int64_t scaleAddrOffsetSplitMN,
                                     uint32_t blockCntSpiltMN)
    {
        uint64_t curBlockCnt = 0;
        int64_t curBlockAddrOffset = 0;
        uint64_t curScaleCnt = 0;
        int64_t curScaleAddrOffset = 0;

        if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            curBlockCnt = blockCntM * quantN_;
            curBlockAddrOffset = blockAddrOffsetSplitM;
            curScaleCnt = quantN_;
        } else {
            curBlockCnt = blockCntSpiltMN;
            curBlockAddrOffset = blockAddrOffsetSplitMN;
            curScaleCnt = blockCntSpiltMN;
            curScaleAddrOffset = scaleAddrOffsetSplitMN;
        }

        mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(this->mmOut_) + curBlockAddrOffset,
                                 curBlockCnt); // 用于读入数据
        quantedOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(this->quantedOut_) + curBlockAddrOffset,
                                      curBlockCnt); // 用于输出数据
        if (((loopIdx == 0) && (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8)) ||
            (this->splitMode_ == SPLIT_MN_MATMUL_ALLREDUCE_INT8)) {
            quantScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(this->quantScale_) + curScaleAddrOffset,
                                          curScaleCnt);
        }
    }

    __aicore__ inline void MatmulAllReduceQuantPerchannelSplitM(const uint32_t quantUbSize, int64_t &blockAddrOffset,
                                                                uint32_t &tileCalCntM, uint32_t &tailCalCntM,
                                                                uint32_t &aivLoopNum)
    {
        uint32_t aivAddOneIndex = quantAivCoreNum_ + 1; // 要多算一轮的核的下标，如果不均分，使用后面 [aivAddOneIndex,
                                                        // quantAivCoreNum_ - 1] 核来完成多余一轮的计算
        if (quantM_ % quantAivCoreNum_ != 0) {
            aivAddOneIndex = quantAivCoreNum_ - (quantM_ % quantAivCoreNum_);
        }

        uint32_t vectorIndex = GetBlockIdx();             // [0, 23]
        uint32_t singleAivM = quantM_ / quantAivCoreNum_; // 单核要计算的总行数（多次循环累计）

        if (singleAivM == 0) { // M小于核数，singleAivM为0，核计算行数更新及偏移计算
            uint32_t usedAivCoreIndex = quantAivCoreNum_ - aivAddOneIndex;
            if (vectorIndex < usedAivCoreIndex) {
                singleAivM = singleAivM + 1;
                blockAddrOffset = static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleAivM) *
                                  static_cast<int64_t>(quantN_);
            } else {
                // 不用计算的核直接返回
                return;
            }
        } else { // M大于核数，singleAivM>0，核计算行数更新及偏移计算
            if ((aivAddOneIndex < quantAivCoreNum_ + 1) && (vectorIndex >= aivAddOneIndex)) {
                // 多算一行
                singleAivM = singleAivM + 1;
                blockAddrOffset = (static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleAivM) -
                                   static_cast<int64_t>(aivAddOneIndex)) *
                                  static_cast<int64_t>(quantN_);
            } else {
                blockAddrOffset = static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleAivM) *
                                  static_cast<int64_t>(quantN_);
            }
        }

        tileCalCntM = quantUbSize / quantAlginN_; // 单次循环计算行数
        aivLoopNum = singleAivM / tileCalCntM;    // 循环次数
        if (singleAivM % (quantUbSize / quantAlginN_) != 0) {
            tailCalCntM = singleAivM % tileCalCntM;
            aivLoopNum += 1;
        }
    }

    __aicore__ inline void MatmulAllReduceQuantPerchannelSplitMAndN(int32_t &quantUbSize, uint32_t &needAivCoreNum,
                                                                    uint32_t &tileBlockCnt, uint32_t &tailBlockCnt,
                                                                    uint32_t &aivLoopNum, uint32_t &blockNumPerRow)
    {
        quantUbSize = TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 /
                      (FP16_BUF_SPLIT_MN_CNT_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T)));
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            quantUbSize = TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 /
                          (BF16_BUF_SPLIT_MN_CNT_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T)));
        }
        quantUbSize = quantUbSize / static_cast<int32_t>(BYTE512_MATMUL_ALLREDUCE_INT8) *
                      static_cast<int32_t>(BYTE512_MATMUL_ALLREDUCE_INT8);
        if (quantUbSize >= static_cast<int32_t>(quantN_)) {
            quantUbSize = static_cast<int32_t>(quantN_);
            tileBlockCnt = quantN_;
        } else {
            blockNumPerRow = Ceil(quantN_, static_cast<uint32_t>(quantUbSize));
            tileBlockCnt = static_cast<uint32_t>(quantUbSize);
            if ((quantN_ % static_cast<uint32_t>(quantUbSize)) != 0) {
                tailBlockCnt = quantN_ % static_cast<uint32_t>(quantUbSize);
            }
        }
        needAivCoreNum = blockNumPerRow * quantM_;
        aivLoopNum = needAivCoreNum / quantAivCoreNum_;
        if ((needAivCoreNum % quantAivCoreNum_) != 0) {
            aivLoopNum = aivLoopNum + 1;
        }
    }

    __aicore__ inline void ProcessDiv(uint32_t loopIdx, uint32_t curBlockCntM, uint32_t calcCnt, uint32_t aivLoopNum,
                                      LocalTensor<T> &mmOutLocal, DataCopyParams &quantedOutCopyParams)
    {
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<int8_t> zLocal = outQueueZ_.AllocTensor<int8_t>();
        LocalTensor<half> outHalfLocalTemp = outHalfLocalTmp_.Get<half>();

        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> xFloatLocalTemp = xFloatLocalTmp_.Get<float>();
            Cast(xFloatLocalTemp, xLocal, RoundMode::CAST_NONE, calcCnt);
            PipeBarrier<PIPE_V>();
            if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // BF16 切M
                for (uint32_t i = 0; i < curBlockCntM; i++) {
                    Div(xFloatLocalTemp[i * quantAlginN_], xFloatLocalTemp[i * quantAlginN_], sFloatLocalTemp_,
                        quantN_);
                }
                PipeBarrier<PIPE_V>();
            } else { // BF16 切MN
                Div(xFloatLocalTemp, xFloatLocalTemp, sFloatLocalTemp_, calcCnt);
                PipeBarrier<PIPE_V>();
            }
            Cast(outHalfLocalTemp, xFloatLocalTemp, RoundMode::CAST_RINT, calcCnt);
            PipeBarrier<PIPE_V>();
        } else if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // FP16 切M
            for (uint32_t i = 0; i < curBlockCntM; i++) {
                Div(outHalfLocalTemp[i * quantAlginN_], xLocal[i * quantAlginN_], sLocal_, quantN_);
            }
            PipeBarrier<PIPE_V>();
        } else { // FP16 切MN
            Div(outHalfLocalTemp, xLocal, sLocal_, calcCnt);
            PipeBarrier<PIPE_V>();
        }

        inQueueX_.FreeTensor(mmOutLocal);
        Cast(zLocal, outHalfLocalTemp, RoundMode::CAST_RINT, calcCnt); // half->int8
        PipeBarrier<PIPE_V>();
        outQueueZ_.EnQue<int8_t>(zLocal);
        LocalTensor<int8_t> outLocal = outQueueZ_.DeQue<int8_t>();
        DataCopyPad(quantedOutGm_, outLocal, quantedOutCopyParams);
        outQueueZ_.FreeTensor(zLocal);
        if ((loopIdx == (aivLoopNum - 1)) && (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8)) {
            inQueueS_.FreeTensor(quantScaleLocal_);
        } else if (this->splitMode_ == SPLIT_MN_MATMUL_ALLREDUCE_INT8) {
            inQueueS_.FreeTensor(quantScaleLocal_);
        }
    }

    __aicore__ inline void Process(uint32_t loopIdx, uint32_t aivLoopNum, uint32_t curBlockCntM,
                                   uint32_t blockCntSpiltMN)
    {
        uint32_t calcCnt = blockCntSpiltMN;
        uint16_t copyInUbStride = 0;
        uint16_t copyOutUbStride = 0;
        uint16_t copyBlockCnt = 1;
        uint32_t copyBlockLen = calcCnt;
        if (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // 搬多行
            calcCnt = quantAlginN_ * curBlockCntM;
            copyInUbStride = (quantAlginN_ * sizeof(T) -
                              Ceil(quantN_ * sizeof(T), BYTE32_MATMUL_ALLREDUCE_INT8) * BYTE32_MATMUL_ALLREDUCE_INT8) /
                             BYTE32_MATMUL_ALLREDUCE_INT8;
            copyOutUbStride =
                (quantAlginN_ * sizeof(int8_t) -
                 Ceil(quantN_ * sizeof(int8_t), BYTE32_MATMUL_ALLREDUCE_INT8) * BYTE32_MATMUL_ALLREDUCE_INT8) /
                BYTE32_MATMUL_ALLREDUCE_INT8;
            copyBlockCnt = curBlockCntM;
            copyBlockLen = quantN_;
        }
        DataCopyParams mmOutCopyParams = {copyBlockCnt, static_cast<uint16_t>(copyBlockLen * sizeof(T)), 0,
                                          copyInUbStride};
        DataCopyParams quantedOutCopyParams = {copyBlockCnt, static_cast<uint16_t>(copyBlockLen * sizeof(int8_t)),
                                               copyOutUbStride, 0};
        DataCopyParams quantScaleCopyParams = {1, static_cast<uint16_t>(copyBlockLen * sizeof(T)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        LocalTensor<T> mmOutLocal = inQueueX_.AllocTensor<T>();
        DataCopyPad(mmOutLocal, mmOutGm_, mmOutCopyParams, padParams); // 从 GM 上读取数据 mmOut 的数据
        inQueueX_.EnQue(mmOutLocal);
        if (((loopIdx == 0) && (this->splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8)) ||
            (this->splitMode_ == SPLIT_MN_MATMUL_ALLREDUCE_INT8)) {
            quantScaleLocal_ = inQueueS_.AllocTensor<T>();
            DataCopyPad(quantScaleLocal_, quantScaleGm_, quantScaleCopyParams, padParams);
            inQueueS_.EnQue(quantScaleLocal_);
            sLocal_ = inQueueS_.DeQue<T>();
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                sFloatLocalTemp_ = sFloatLocalTmp_.Get<float>();
                Cast(sFloatLocalTemp_, sLocal_, RoundMode::CAST_NONE, copyBlockLen);
            }
        }
        ProcessDiv(loopIdx, curBlockCntM, calcCnt, aivLoopNum, mmOutLocal, quantedOutCopyParams);
    }

    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueS_;
    TQue<QuePosition::VECOUT, 1> outQueueZ_;
    TBuf<TPosition::VECCALC> outHalfLocalTmp_;
    TBuf<TPosition::VECCALC> xFloatLocalTmp_;
    TBuf<TPosition::VECCALC> sFloatLocalTmp_;
    TBuf<TPosition::VECCALC> outFloatLocalTmp_;
    GlobalTensor<T> mmOutGm_;
    GlobalTensor<T> quantScaleGm_;
    GlobalTensor<int8_t> quantedOutGm_; // 量化后的结果存放的位置
    LocalTensor<T> quantScaleLocal_;
    LocalTensor<T> sLocal_;
    LocalTensor<float> sFloatLocalTemp_;
    uint32_t quantAlginN_;     // 512B 对齐后的输入 N，512B 对齐可以命中 L2 cache 缓存
    uint32_t quantN_;          // 输入 N
    uint32_t quantM_;          // 输入 M
    uint32_t quantAivCoreNum_; // vector 核数
    uint32_t splitMode_;
    GM_ADDR quantScale_;
    GM_ADDR quantedOut_;
    GM_ADDR mmOut_;
};

/*
 * 接口说明：&outerLoop: 总的外循环个数，结合outerLoopId判断是否为最后一个循环
 */
template <class T>
__aicore__ inline void MatmulAllReduceQuantPerchannelCommInt8(GM_ADDR mmOut, GM_ADDR quantScale, GM_ADDR quantedOut,
                                                              TPipe *tPipe, uint32_t N, uint32_t M)
{
    uint32_t nowQuantAivCoreNum = GetBlockNum() * GetTaskRation();
    if ((g_coreType == AIC) || (GetBlockIdx() >= nowQuantAivCoreNum)) {
        return;
    }
    uint32_t tileBlockCnt = 0;
    uint32_t tailBlockCnt = 0;
    uint32_t quantAivLoopNum = 0;
    int64_t blockAddrOffset = 0;
    uint32_t tailCalCntM = 0;
    uint32_t tileCalCntM = 0;
    uint32_t needAivCoreNum = 0;
    uint32_t blockNumPerRow = 1;

    tPipe->Reset();
    int32_t nowQuantAlginN = Ceil(N * sizeof(int8_t), BYTE512_MATMUL_ALLREDUCE_INT8) * BYTE512_MATMUL_ALLREDUCE_INT8;
    int32_t nowQuantUbSize =
        (TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 - static_cast<int32_t>(nowQuantAlginN) * static_cast<int32_t>(sizeof(T))) /
        (FP16_BUF_CNT_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T))); // fp16
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {                  // bf16
        nowQuantUbSize = (TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 -
                          static_cast<int32_t>(nowQuantAlginN) * static_cast<int32_t>(sizeof(T))) /
                         (BF16_BUF_CNT_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T)));
    }
    nowQuantUbSize = (nowQuantUbSize / nowQuantAlginN) * nowQuantAlginN;
    if (nowQuantUbSize > M * nowQuantAlginN) {
        nowQuantUbSize = M * nowQuantAlginN;
    }
    MatmulAllReduceQuantPerchannel<T> op;
    op.quantM_ = M;
    op.quantN_ = N;
    op.quantAlginN_ = static_cast<uint32_t>(nowQuantAlginN);
    op.quantAivCoreNum_ = nowQuantAivCoreNum;
    op.mmOut_ = mmOut;
    op.quantScale_ = quantScale;
    op.quantedOut_ = quantedOut;
    if (nowQuantUbSize >= nowQuantAlginN) { // 分核切M， 不切N
        op.splitMode_ = SPLIT_M_MATMUL_ALLREDUCE_INT8;
        op.MatmulAllReduceQuantPerchannelSplitM(nowQuantUbSize, blockAddrOffset, tileCalCntM, tailCalCntM,
                                                quantAivLoopNum);
    } else { // N超大，分核策略采取切MN
        op.splitMode_ = SPLIT_MN_MATMUL_ALLREDUCE_INT8;
        op.MatmulAllReduceQuantPerchannelSplitMAndN(nowQuantUbSize, needAivCoreNum, tileBlockCnt, tailBlockCnt,
                                                    quantAivLoopNum, blockNumPerRow);
    }

    if (quantAivLoopNum == 0) {
        return;
    }
    op.Init(tPipe, nowQuantUbSize);                                    // 按照最大去开
    for (uint32_t loopIdx = 0; loopIdx < quantAivLoopNum; loopIdx++) { // 一轮外层循环对应着一次核间并行
        int64_t blockAddrOffsetSplitM = 0;
        uint32_t blockCntM = tileCalCntM;
        uint32_t blockCntSpiltMN = tileBlockCnt;
        int64_t blockAddrOffsetSplitMN = 0;
        int64_t scaleAddrOffsetSplitMN = 0;
        if (op.splitMode_ == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            if ((tailCalCntM != 0) && (loopIdx == quantAivLoopNum - 1)) {
                blockCntM = tailCalCntM;
            }
            blockAddrOffsetSplitM = static_cast<int64_t>(blockAddrOffset) +
                                    static_cast<int64_t>(loopIdx) * static_cast<int64_t>(tileCalCntM) *
                                        static_cast<int64_t>(N); // 取到当前核的偏移，再+当前块的地址偏移
        } else {
            uint32_t quantGlobalBlockIdx = loopIdx * nowQuantAivCoreNum + GetBlockIdx();
            if (quantGlobalBlockIdx > (needAivCoreNum - 1)) {
                return;
            }
            if ((tailBlockCnt != 0) && (quantGlobalBlockIdx % blockNumPerRow == blockNumPerRow - 1)) {
                blockCntSpiltMN = tailBlockCnt;
            }
            blockAddrOffsetSplitMN =
                static_cast<int64_t>(quantGlobalBlockIdx / blockNumPerRow) * static_cast<int64_t>(N) +
                static_cast<int64_t>(quantGlobalBlockIdx % blockNumPerRow) *
                    static_cast<int64_t>(tileBlockCnt); // 计算当前块的地址偏移
            scaleAddrOffsetSplitMN =
                static_cast<int64_t>(quantGlobalBlockIdx % blockNumPerRow) * static_cast<int64_t>(tileBlockCnt);
        }
        op.InitInner(loopIdx, blockAddrOffsetSplitM, blockCntM, blockAddrOffsetSplitMN, scaleAddrOffsetSplitMN,
                     blockCntSpiltMN);
        op.Process(loopIdx, quantAivLoopNum, blockCntM, blockCntSpiltMN);
    }
}
} // namespace MatmulAllReduceQuantPerchannelImpl

#endif // MATMUL_ALL_REDUCE_QUANT_PERCHANNEL_H
