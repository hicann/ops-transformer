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
 * \file lightning_indexer_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_VECTOR_H
#define LIGHTNING_INDEXER_VECTOR_H

#include "lightning_indexer_vector.h"
#include "kernel_operator.h"

namespace LIServiceVec {
using namespace AscendC;

constexpr int32_t NEG_INF = 0xFF800000;
constexpr int32_t INVALID_INDEX = -1;
// SECOND_MAX_LOGITS = MAX_LOGITS/2 = 0x7F7FFFFF 指数 -1。哨兵分数有两档：sink/local=MAX/2、
// 对角=MAX，均 ≥ 此值；有效分数被 service 侧 Mins 钳到 ≤ MAX/4 < 此值。用作 firstQS 滤哨兵的界。
constexpr uint32_t SECOND_MAX_LOGITS_BITS = 0x7EFFFFFF; // 0x7F7FFFFF 的 1/2（指数 -1）
constexpr uint8_t VEC_REPEAT_MAX = 255;
constexpr uint8_t B32_VEC_ELM_NUM = 64;
constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint8_t B32_VEC_REPEAT_STRIDE = 8;
constexpr uint64_t VEC_REPEAT_BYTES = 256;
constexpr int32_t CONST_TWO = 2;
constexpr int64_t VALUE_AND_INDEX_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t MRG_QUE_0 = 0;
constexpr int64_t MRG_QUE_1 = 1;
constexpr int64_t MRG_QUE_2 = 2;
constexpr int64_t MRG_QUE_3 = 3;
constexpr int64_t MRG_BLOCK_2 = 2;
constexpr int64_t MRG_BLOCK_3 = 3;
constexpr int64_t MRG_BLOCK_4 = 4;

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<float> &mmOutUb, LocalTensor<T> &weightsUb, GlobalTensor<float> &mMoutGm,
                              GlobalTensor<T> &weightScaleGm, int64_t MMout_gmoffset, int64_t weights_gmoffset,
                              int64_t groupInner, int64_t s2Inner, int64_t mmUbStride)
{
    // 将MMout_gmoffset copy到UB上
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;
    dataCopymMoutParams.blockLen = s2Inner * sizeof(float);
    dataCopymMoutParams.srcStride = 0;
    dataCopymMoutParams.dstStride = mmUbStride;
    dataCopymMoutParams.rsv = 0;
    AscendC::DataCopyPad(mmOutUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);

    // 将weights_gmoffset copy到UB
    AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopyweightParams;
    dataCopyweightParams.blockCount = 1;
    dataCopyweightParams.blockLen = groupInner * sizeof(T);
    dataCopyweightParams.srcStride = 0;
    dataCopyweightParams.dstStride = 0;
    dataCopyweightParams.rsv = 0;
    AscendC::DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t copyCount)
{
    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(T);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    AscendC::DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}

__aicore__ inline void DoScale(const LocalTensor<float> &reduceCacheBuf, LocalTensor<float> &mmOutUb,
                               LocalTensor<float> &weightsUb, LocalTensor<float> &tmpBuff, int64_t groupInner,
                               int64_t s2Inner, int32_t outerGidx)
{
    // weight broadcast: [groupInner, 1] -> [groupInner, 8]
    AscendC::Brcb(tmpBuff, weightsUb, LICommon::CeilDiv(groupInner, static_cast<int64_t>(B32_BLOCK_ALIGN_NUM)),
                  {1, B32_VEC_REPEAT_STRIDE});
    AscendC::PipeBarrier<PIPE_V>();

    // do scale: [groupInner, 8] * [groupInner, s2Inner]
    uint64_t countPerRepeat = VEC_REPEAT_BYTES / sizeof(float);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            AscendC::Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        } else {
            AscendC::Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM], countPerRepeat,
                         repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        }
    }

    if (outerGidx != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline uint64_t FindNearestPower2(uint64_t value)
{
    if (value <= CONST_TWO) {
        return value;
    } else {
        const uint64_t pow = 63 - clz(value); // clz返回前导0的个数，对于64位整数，最大有效位位置 = 63 - 前导0个数
        return (1 << pow);
    }
}

// dstTensor 需要初始化0
__aicore__ inline void DoReduce(const LocalTensor<float> &srcTensor, LocalTensor<float> &dstTensor, int32_t rNum,
                                int32_t aNum)
{
    if (rNum == 1) {
        AscendC::Adds<float>(dstTensor, srcTensor, 0, aNum);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }

    uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        AscendC::Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > CONST_TWO) {
        nowRows = nowRows / CONST_TWO;
        AscendC::Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add(dstTensor, srcTensor, srcTensor[aNum], aNum);
    AscendC::PipeBarrier<PIPE_V>();
}

/**
  src: 传入的初始化空间
  eleNum: 需要初始化的元素个数需为64整数倍，元素将被初始化为交错排布的-inf，-1
 */
__aicore__ inline void InitSortOutBuf(const LocalTensor<float> &src, int64_t eleNum)
{
    uint64_t mask1[2] = {0x5555555555555555, 0};
    uint64_t mask0[2] = {0xaaaaaaaaaaaaaaaa, 0};
    int64_t repeatNum = eleNum / B32_VEC_ELM_NUM;
    int64_t forLoop = repeatNum / VEC_REPEAT_MAX;
    int64_t forRemain = repeatNum % VEC_REPEAT_MAX;
    for (int i = 0; i < forLoop; i++) {
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[i * VEC_REPEAT_MAX * B32_VEC_ELM_NUM], NEG_INF,
                           mask1, VEC_REPEAT_MAX, 1, B32_VEC_REPEAT_STRIDE);
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[i * VEC_REPEAT_MAX * B32_VEC_ELM_NUM], INVALID_INDEX,
                           mask0, VEC_REPEAT_MAX, 1, B32_VEC_REPEAT_STRIDE);
    }
    if (forRemain > 0) {
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM], NEG_INF,
                           mask1, forRemain, 1, B32_VEC_REPEAT_STRIDE);
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM],
                           INVALID_INDEX, mask0, forRemain, 1, B32_VEC_REPEAT_STRIDE);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

/**
  初始化 globalTopkUb_ 为分离格式: [scores(-inf) × slotSize | indices(0) × slotSize] per token
  slotSize = virTopK + QS_OVF, tokenStride = slotSize * 2
 */
__aicore__ inline void InitSortOutBufSeparated(const LocalTensor<float> &src, int32_t numTokens, int32_t tokenStride,
                                               int32_t slotSize)
{
    for (int t = 0; t < numTokens; t++) {
        int32_t base = t * tokenStride;
        // scores 区: [base : base+slotSize] = -inf
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[base], NEG_INF, slotSize);
        // indices 区: [base+slotSize : base+slotSize*2] = -1 (INVALID_INDEX)
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[base + slotSize], INVALID_INDEX, slotSize);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

/**
  src: logits和索引，前logitsNum为logits，后logitsNum为索引
  tmp: 计算使用到的临时空间，大小与src一致
  logitsNum: 排序的元素个数, 暂只支持[128,256,384,512,1024,2048]
 */
__aicore__ inline void SortAll(LocalTensor<float> &src, LocalTensor<float> &tmp, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / BLOCK_BYTES;
    AscendC::Sort32(tmp, src, src[logitsNum].ReinterpretCast<uint32_t>(), sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();

    int64_t mrgGroups = sort32Repeats;
    int64_t mrgElements = BLOCK_BYTES;
    int64_t i = 0;
    AscendC::LocalTensor<float> srcTensor;
    AscendC::LocalTensor<float> dstTensor;
    while (true) {
        if (i % CONST_TWO == 0) {
            srcTensor = tmp;
            dstTensor = src;
        } else {
            srcTensor = src;
            dstTensor = tmp;
        }
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = mrgElements;
        params.elementLengths[MRG_QUE_1] = mrgElements;
        params.elementLengths[MRG_QUE_2] = mrgElements;
        params.elementLengths[MRG_QUE_3] = mrgElements;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = srcTensor[0];
        srcList.src2 = srcTensor[MRG_QUE_1 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src3 = srcTensor[MRG_QUE_2 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src4 = srcTensor[MRG_QUE_3 * VALUE_AND_INDEX_NUM * mrgElements];
        if (mrgGroups <= MRG_BLOCK_4) {
            params.repeatTimes = 1;
            if (mrgGroups == 1) {
                break;
            } else if (mrgGroups == MRG_BLOCK_2) {
                params.validBit = 0b0011;
            } else if (mrgGroups == MRG_BLOCK_3) {
                params.validBit = 0b0111;
            } else if (mrgGroups == MRG_BLOCK_4) {
                params.validBit = 0b1111;
            }
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            break;
        } else {
            params.repeatTimes = mrgGroups / MRG_BLOCK_4;
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            mrgElements = mrgElements * MRG_BLOCK_4;
            mrgGroups = mrgGroups / MRG_BLOCK_4;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (i % CONST_TWO == 0) {
        AscendC::DataCopy(src, tmp, logitsNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
  dst: 输出全排序的结果，排布方式为value，index
  srcValue：输入的待排序浮点数
  srcIndex：浮点数的索引
  tmp: 计算使用到的临时空间，大小为srcValue+srcIndex
  logitsNum: 排序的元素个数
 */
__aicore__ inline void SortAll(LocalTensor<float> &dst, LocalTensor<float> &srcValue, LocalTensor<uint32_t> &srcIndex,
                               LocalTensor<float> &tmpTensor, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / BLOCK_BYTES;
    AscendC::Sort<float, true>(dst, srcValue, srcIndex, tmpTensor, sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();
}

/**
  mrgDst: 合并进的Tensor
  mrgSrc: 待合并的Tensor
  tmpTensor：空间为mrgDst+mrgSrc
 */
__aicore__ inline void MergeSort(const LocalTensor<float> &mrgDst, int32_t mrgDstNum, LocalTensor<float> &mrgSrc,
                                 int32_t mrgSrcNum, LocalTensor<float> &tmpTensor)
{
    if (mrgDstNum <= 3072) {
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = mrgDstNum;
        params.elementLengths[1] = mrgSrcNum;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b0011;
        params.repeatTimes = 1;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = mrgDst;
        srcList.src2 = mrgSrc;

        AscendC::MrgSort<float>(tmpTensor, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(mrgDst, tmpTensor, mrgDstNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    } else { // elementLengths最大值为4095
        // int64_t unitElements = 1024;
        // int64_t segNum = mrgDstNum / unitElements;
        // int64_t mrgQuelen_1 = (segNum + 2) / 3;
        // int64_t mrgQuelen_2 = ((segNum - mrgQuelen_1) + 1) / 2;
        // int64_t mrgQuelen_3 = segNum - mrgQuelen_1 - mrgQuelen_2;
        int64_t unitElements = 128;
        int64_t segNum = mrgDstNum / unitElements;
        int64_t mrgQuelen_1 = (segNum + 2) / 3;
        int64_t mrgQuelen_2 = ((segNum - mrgQuelen_1) + 1) / 2;
        int64_t mrgQuelen_3 = segNum - mrgQuelen_1 - mrgQuelen_2;

        AscendC::MrgSort4Info params;
        params.elementLengths[0] = mrgQuelen_1 * unitElements;
        params.elementLengths[1] = mrgQuelen_2 * unitElements;
        params.elementLengths[2] = mrgQuelen_3 * unitElements;
        params.elementLengths[3] = mrgSrcNum;

        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;
        params.repeatTimes = 1;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = mrgDst[0];
        srcList.src2 = mrgDst[mrgQuelen_1 * VALUE_AND_INDEX_NUM * unitElements];
        srcList.src3 = mrgDst[(mrgQuelen_1 + mrgQuelen_2) * VALUE_AND_INDEX_NUM * unitElements];
        srcList.src4 = mrgSrc;

        AscendC::MrgSort<float>(tmpTensor, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(mrgDst, tmpTensor, mrgDstNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
 * @brief 合并基础块函数
 * @param dst 归并后的输出, 大小为blockNum * basicBlockSize * 2 * sizeof(float)
 * @param src 基本块输入
 * @param blockNum 基本块的数量
 * @param basicBlockSize 基础块的大小
 * @return 无
 */
__aicore__ inline void MrgBasicBlock(const LocalTensor<float> &dst, const LocalTensor<float> &src, int64_t blockNum,
                                     int64_t basicBlockSize)
{
    // 初始化合并排序参数
    AscendC::MrgSort4Info params;
    params.elementLengths[MRG_QUE_0] = basicBlockSize;
    params.elementLengths[MRG_QUE_1] = basicBlockSize;
    params.elementLengths[MRG_QUE_2] = basicBlockSize;
    params.elementLengths[MRG_QUE_3] = basicBlockSize;
    params.ifExhaustedSuspension = false;
    // 根据块的数量设置有效位
    if (blockNum == MRG_BLOCK_2) {
        params.validBit = 0b0011;
    } else if (blockNum == MRG_BLOCK_3) {
        params.validBit = 0b0111;
    } else if (blockNum == MRG_BLOCK_4) {
        params.validBit = 0b1111;
    } else {
        AscendC::DataCopy(dst, src, basicBlockSize * VALUE_AND_INDEX_NUM);
        return;
    }
    // 初始化源列表
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = src[0];
    srcList.src2 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_1];
    srcList.src3 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_2];
    srcList.src4 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_3];
    // 执行合并排序
    AscendC::MrgSort<float>(dst, srcList, params);
}

/**
 * @brief 从两个队列中选择topk
 * @param dst 已经归并好的topk数据
 * @param needsMerging 需要合并的有序数据
 * @param tmp 临时空间
 * @param topk topk的元素个数
 * @param mergSize 待合并的元素个数
 * @return 无
 */
template <bool needMrg = true>
__aicore__ inline void SparseTopK(const LocalTensor<float> &dst, const LocalTensor<float> &needsMerging,
                                  const LocalTensor<float> &tmp, int64_t topk, int64_t mergSize)
{
    // 如果不需要合并，则直接复制数据
    if (!needMrg) {
        AscendC::DataCopy(dst, needsMerging, mergSize * VALUE_AND_INDEX_NUM);
        return;
    }
    // 初始化合并排序参数
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = topk;
    params.elementLengths[1] = mergSize;
    params.ifExhaustedSuspension = (topk == mergSize);
    params.validBit = 0b0011;
    // 初始化源列表
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = dst;
    srcList.src2 = needsMerging;
    // 执行合并排序
    AscendC::MrgSort<float>(tmp, srcList, params);
    // 将结果复制到目标张量
    AscendC::DataCopy(dst, tmp, topk * VALUE_AND_INDEX_NUM);
}

__aicore__ inline void ExtractIndex(const LocalTensor<uint32_t> &idxULocal, const LocalTensor<uint32_t> &sortLocal,
                                    int64_t extractNum)
{
    AscendC::GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(extractNum * sizeof(float) * VALUE_AND_INDEX_NUM, VEC_REPEAT_BYTES);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
    gatherMaskParams.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
    uint8_t src1Pattern = 2; // 固定模式2,表示筛选出奇数索引的数
    AscendC::GatherMask(idxULocal, sortLocal, src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

struct QSParams {
    int32_t target;      // 目标保留数（virTopK）
    int32_t overflowCap; // 容差（QS_OVF）
    int32_t newDataLen;  // 新数据长度（blockS2LenVecAlign，64 对齐）
    int32_t maxIter;     // 最大迭代次数
    float lastThreshold; // 上轮收敛 pivot（作为非首次 QS 的二分下界，单调递增）
    float suggestedMax; // 非首次 QS 的 curMax 上界：carry（上轮实测紧上界）或兜底 128；firstQS 内部自算
    int32_t firstQS; // 本 token 首次 QS：内部用 ReduceMin(glob) 算下界 + 滤哨兵 ReduceMax(glob) 算上界
    int32_t carrySafeCnt; // 安全计数线 = target+ovf-maxNewData；aboveCnt≤此值的 pivot 下轮仍是合法上界

    __aicore__ inline int32_t cap() const
    {
        return target + overflowCap + newDataLen;
    }
    __aicore__ inline int32_t workingNAlign() const
    {
        return (cap() + 63) / 64 * 64;
    }
    __aicore__ inline int32_t newDataOffset() const
    {
        return target + overflowCap;
    }
    // tmpBuf 中各段起始（float offset）
    __aicore__ inline int32_t srcIndicesOff() const
    {
        return cap();
    }
    __aicore__ inline int32_t dstScoresOff() const
    {
        return cap() * 2;
    }
};

/**
 * @brief QuickSelect: 从 tmpBuf 工作区中选出 top-target 个元素
 * @param tmpBuf   主工作区: [srcScores(cap) | srcIndices(cap) | dstScores(实测仅写 ≤4096)]
 * @param auxBuf   辅助工作区: [dstIndices(cap) | cmpMask] (放在 tmpSortBuf 中)
 * @return above 总数 (>=target 表示收敛), 0 = 未收敛
 */
__aicore__ inline int32_t QuickSelectPartition(LocalTensor<float> &tmpBuf, LocalTensor<float> &auxBuf,
                                               const QSParams &params, float &outThreshold, float &outCarryMax)
{
    int32_t cap = params.cap();
    int32_t workingN = params.workingNAlign();
    int32_t target = params.target;
    int32_t overflowCap = params.overflowCap;
    int32_t newDataLen = params.newDataLen;

    // 主工作区切分: tmpBuf = [srcScores(cap) | srcIndices(cap) | dstScores(cap)]
    LocalTensor<float> srcScores = tmpBuf;
    LocalTensor<int32_t> srcIndices = tmpBuf[cap].template ReinterpretCast<int32_t>();
    LocalTensor<float> dstScores = tmpBuf[cap * 2];
    // 辅助工作区切分: auxBuf = [dstIndices(cap) | cmpMask(cap/8 bytes)]
    LocalTensor<int32_t> dstIndices = auxBuf.template ReinterpretCast<int32_t>();
    int32_t cmpMaskFloatOff = cap; // cmpMask 紧接 dstIndices 之后
    LocalTensor<uint8_t> cmpMask = auxBuf[cmpMaskFloatOff].template ReinterpretCast<uint8_t>();

    float curMin;
    float curMax;
    if (params.firstQS) {
        ReduceMin(dstScores, srcScores, dstScores[target], target, false);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        curMin = dstScores.GetValue(0);
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

        const float SENTINEL_BOUND = *((const float *)&SECOND_MAX_LOGITS_BITS);
        LocalTensor<float> selBuf = dstIndices.template ReinterpretCast<float>();
        CompareScalar(cmpMask, srcScores, SENTINEL_BOUND, CMPMODE::LT, target);
        PipeBarrier<PIPE_V>();
        Select(selBuf, cmpMask, srcScores, -3.4e38f, SELMODE::VSEL_TENSOR_SCALAR_MODE, static_cast<uint32_t>(target));
        PipeBarrier<PIPE_V>();
        ReduceMax(dstScores, selBuf, dstScores[target], target, false);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        curMax = dstScores.GetValue(0);
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        curMax = max(curMax, curMin); // 防 bracket 反转（极端：glob 全哨兵→有效 max=-inf）
    } else {
        curMin = params.lastThreshold;
        curMax = max(params.suggestedMax, curMin);
    }

    AscendC::GatherMaskParams gp = {1, 1, 8, 0};
    float pivot = curMax;
    bool probing = true;
    uint32_t wNA = static_cast<uint32_t>(workingN);
    // 割线插值状态：最近一个真实测量点 (pivot, aboveCnt)，用 aboveCnt 大小信息外推真阈值。
    float prevPivot = 0.0f;
    int32_t prevAbove = -1; // -1 = 尚无有效上一轮点
    // carry: 跟踪 aboveCnt≤carrySafeCnt 中 pivot 最低（=aboveCnt 最大）的点，作下轮 curMax。
    float bestCarry = 3.4e38f;
    bool haveCarry = false;

    for (int32_t iter = 0; iter < params.maxIter; iter++) {
        CompareScalar(cmpMask, srcScores, pivot, CMPMODE::GE, workingN);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint32_t> pattern = cmpMask.template ReinterpretCast<uint32_t>();
        uint64_t rsvdCnt = 0;
        GatherMask(dstScores.template ReinterpretCast<uint16_t>(), srcScores.template ReinterpretCast<uint16_t>(),
                   cmpMask.template ReinterpretCast<uint16_t>(), true, wNA, gp, rsvdCnt);
        PipeBarrier<PIPE_V>();

        int32_t aboveCnt = static_cast<int32_t>(rsvdCnt);
        bool converged = (aboveCnt >= target && aboveCnt <= target + overflowCap);
        // carry: 此 pivot 的 aboveCnt≤safeCnt → 下轮加 maxNewData 后 count≤target+ovf，仍不触发扩界。
        //        取其中最低 pivot（aboveCnt 最大）= 最紧合法上界。
        if (aboveCnt <= params.carrySafeCnt && pivot < bestCarry) {
            bestCarry = pivot;
            haveCarry = true;
        }

        if (!converged) {
            if (probing) {
                if (aboveCnt >= target) {
                    // count(≥curMax)≥target：上界不足。旧 curMax 成为合法下界，几何扩界后继续探测。
                    // 几何扩界：正数翻倍上抬；负数折半向 0（即抬高）；跨零（接近 0）种子一个正值。
                    curMin = curMax;
                    constexpr float EXP_EPS = 0.0625f;
                    if (curMax > EXP_EPS) {
                        curMax *= 2.0f;
                    } else if (curMax < -EXP_EPS) {
                        curMax *= 0.5f;
                    } else {
                        curMax = 1.0f;
                    }
                    pivot = curMax;
                } else {
                    // count(≥curMax)<target：curMax 是合法上界，切入二分相。
                    probing = false;
                    prevPivot = pivot; // 记录 (curMax, aboveCnt) 作为割线第一个真实点
                    prevAbove = aboveCnt;
                    pivot = (curMax + curMin) * 0.5f;
                }
            } else {
                // 二分相：收窄 bracket。
                if (aboveCnt < target) {
                    curMax = pivot; // 选少 → 压上界
                } else {
                    curMin = pivot; // 选多 → 抬下界
                }
                // 割线：用最近两真实点 (prevPivot,prevAbove)-(pivot,aboveCnt) 外推到 count=target。
                // count 随 pivot 单调下降 → 分母 (aboveCnt-prevAbove) 与 (pivot-prevPivot) 反号、非 0。
                int32_t denom = aboveCnt - prevAbove;
                float nextPivot = (denom != 0) ?
                                      pivot + (pivot - prevPivot) * (float)(target - aboveCnt) / (float)denom :
                                      (curMax + curMin) * 0.5f;
                // 有向 clamp：secant 越界说明真阈值紧贴该侧端点，直接钉到端点附近（偏置 1/8），
                // 而非退回中点。BIAS 越小越贴边、越激进。
                constexpr float GUARD_BIAS = 0.125f;
                if (nextPivot <= curMin) {
                    nextPivot = curMin + (curMax - curMin) * GUARD_BIAS;
                } else if (nextPivot >= curMax) {
                    nextPivot = curMax - (curMax - curMin) * GUARD_BIAS;
                }
                prevPivot = pivot;
                prevAbove = aboveCnt;
                pivot = nextPivot;
            }
            if (!probing && (pivot == curMin || pivot == curMax)) {
                outThreshold = pivot; // 区间夹死：返回 0（未收敛），由 SortAll 兜底
                outCarryMax = haveCarry ? bestCarry : params.suggestedMax;
                return 0;
            }
            continue;
        }

        // 收敛：GatherMask 输出 scores + indices
        Duplicate(dstScores[target], -3.4e38f, overflowCap);
        Duplicate(dstIndices[target], INVALID_INDEX, overflowCap);
        PipeBarrier<PIPE_V>();
        GatherMask(dstScores, srcScores, pattern, true, wNA, gp, rsvdCnt);
        PipeBarrier<PIPE_V>();
        uint64_t tmp = 0;
        GatherMask(dstIndices.template ReinterpretCast<uint32_t>(), srcIndices.template ReinterpretCast<uint32_t>(),
                   pattern, true, wNA, gp, tmp);
        PipeBarrier<PIPE_V>();
        outThreshold = pivot;
        outCarryMax = haveCarry ? bestCarry : params.suggestedMax;
        return aboveCnt;
    }
    outCarryMax = haveCarry ? bestCarry : params.suggestedMax;
    return 0;
}

} // namespace LIServiceVec
#endif // LIGHTNING_INDEXER_VECTOR_H
