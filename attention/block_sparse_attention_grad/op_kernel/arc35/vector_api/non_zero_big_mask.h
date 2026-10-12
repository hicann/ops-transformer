/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file non_zero_big_mask.h
 * \brief
 */

#ifndef CANN_NON_ZERO_BIG_MASK_H
#define CANN_NON_ZERO_BIG_MASK_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"

namespace NonZero {
using namespace Ops::Base;
using namespace AscendC;
using AscendC::Reg::AddrReg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UnPack;

const int32_t BUFFER_NUM = 2;
const int32_t MULTIP_UB_SIZE = 32 * 3;
const int32_t REDUCE_UB_SIZE = 72 * 128;
const int32_t WORKSPACE_SIZE = 72 * 128;

constexpr uint64_t NON_ZERO_OUTSHAPE_DIM = 0x80000002;
constexpr int32_t IDX_NUM_0 = 0;
constexpr int32_t IDX_NUM_1 = 1;
constexpr int32_t IDX_NUM_2 = 2;
constexpr int32_t IDX_NUM_3 = 3;
constexpr int32_t IDX_NUM_4 = 4;
constexpr int32_t IDX_NUM_5 = 5;
constexpr int32_t IDX_NUM_6 = 6;
constexpr int32_t IDX_NUM_7 = 7;
constexpr int32_t IDX_NUM_8 = 8;
constexpr int32_t IDX_NUM_9 = 9;
constexpr int32_t IDX_NUM_10 = 10;
constexpr int32_t IDX_NUM_11 = 11;
constexpr int32_t IDX_NUM_12 = 12;
constexpr int32_t IDX_NUM_13 = 13;
constexpr int32_t IDX_NUM_14 = 14;
constexpr int32_t IDX_NUM_15 = 15;
constexpr int32_t IDX_NUM_16 = 16;
constexpr int32_t IDX_NUM_17 = 17;
constexpr int32_t IDX_NUM_18 = 18;
constexpr int32_t IDX_NUM_19 = 19;
constexpr int32_t IDX_NUM_20 = 20;
constexpr int32_t IDX_NUM_21 = 21;
constexpr int32_t IDX_NUM_22 = 22;
constexpr int32_t IDX_NUM_23 = 23;
constexpr int32_t OFFSET_NUM_16 = 16;
constexpr int32_t OFFSET_NUM_32 = 32;
constexpr int32_t OFFSET_NUM_64 = 64;
constexpr int32_t OFFSET_NUM_128 = 128;
constexpr int32_t OFFSET_NUM_192 = 192;

template <typename T1, typename T2>
class NonZeroBigMask {
public:
    __aicore__ inline NonZeroBigMask(){};
    __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace,
                                    const mNonZeroTiling* tilingData, TPipe* tPipe);
    template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
    __aicore__ inline void ProcessBase(CLS_NAME* objPtr);

protected:
    __aicore__ inline void ComputeOutputBaseFunc(__local_mem__ uint32_t* srcPtr, __local_mem__ uint32_t* dstPtr,
                                                 __local_mem__ uint32_t* dstLastPtr, MaskReg& preg, AddrReg& vagReg,
                                                 RegTensor<uint32_t>& srcReg, RegTensor<uint32_t>& subReg,
                                                 RegTensor<uint32_t>& shapeReg, RegTensor<uint32_t>& mulReg,
                                                 RegTensor<uint32_t>& mReg, RegTensor<uint32_t>& divReg0,
                                                 RegTensor<uint32_t>& divReg1, uint32_t sValue, uint32_t mValue,
                                                 int16_t kValue);
    __aicore__ inline void CastOutput(LocalTensor<int32_t>& yUb, int32_t srcOffset, int32_t num, uint16_t dims);

private:
    __aicore__ inline void CalcNonZeroNumPerCore();
    __aicore__ inline void CalcNonZeroNumPerCoreB64();
    __aicore__ inline void SetMultipDim();
    __aicore__ inline void GetAllNumAndOffset();
    __aicore__ inline void GetAllNumAndOffsetB64();
    __aicore__ inline void ProcessOutputShape();
    __aicore__ inline void CopyInAndCalcNum(LocalTensor<int32_t>& addUbSize, int32_t copyNum, int32_t idxOutter,
                                            int32_t idxInner);
    __aicore__ inline void CopyInAndCalcNumB64(LocalTensor<int64_t>& addUbSize, int32_t copyNum, int32_t idxOutter,
                                               int32_t idxInner);
    __aicore__ inline void GetNonZeroNumB8(int32_t processNum, __local_mem__ T1* xUbPtr,
                                           __local_mem__ int32_t* dstUbPtr);
    __aicore__ inline void GetNonZeroNumB16(int32_t processNum, __local_mem__ T1* xUbPtr,
                                            __local_mem__ int32_t* dstUbPtr);
    __aicore__ inline void GetNonZeroNumB32(int32_t processNum, __local_mem__ T1* xUbPtr,
                                            __local_mem__ int32_t* dstUbPtr);
    __aicore__ inline void GetNonZeroNumB64(int32_t processNum, __local_mem__ T1* xUbPtr,
                                            __local_mem__ int64_t* dstUbPtr);
    __aicore__ inline void VfReduceSum(LocalTensor<int32_t>& addUbSize, int32_t num);
    __aicore__ inline void VfReduceSumB64(LocalTensor<int64_t>& addUbSize, int32_t num);
    __aicore__ inline void ClacAllNum(LocalTensor<int32_t>& addUbSize);
    __aicore__ inline void ClacAllNumB64(LocalTensor<int64_t>& addUbSize);
    __aicore__ inline void CopyIn(int32_t num, int32_t idx);
    __aicore__ inline void SqzNonZeroNumB8(int32_t processNum, __local_mem__ T1* xUbPtr, __local_mem__ int32_t* yUbPtr,
                                           int32_t idx);
    __aicore__ inline void SqzNonZeroNumB16(int32_t processNum, __local_mem__ T1* xUbPtr, __local_mem__ int32_t* yUbPtr,
                                            int32_t idx);
    __aicore__ inline void SqzNonZeroNumB32(int32_t processNum, __local_mem__ T1* xUbPtr, __local_mem__ int32_t* yUbPtr,
                                            int32_t idx);
    __aicore__ inline void SqzNonZeroNumB64(int32_t processNum, __local_mem__ T1* xUbPtr, __local_mem__ int32_t* yUbPtr,
                                            int32_t idx);
    __aicore__ inline void ComputeOutputAndTrans(LocalTensor<int32_t>& yUb);
    __aicore__ inline void TransOutputDim2(__local_mem__ uint32_t* srcPtr, __local_mem__ uint32_t* dstPtr);
    __aicore__ inline void TransOutputDim4(__local_mem__ uint32_t* srcPtr, __local_mem__ uint32_t* dstPtr);
    __aicore__ inline void TransOutput(__local_mem__ uint32_t* srcPtr, __local_mem__ uint32_t* dstPtr);
    __aicore__ inline void CopyOutWithTrans();
    __aicore__ inline void ComputeOutput(LocalTensor<int32_t>& yUb);
    __aicore__ inline void CopyOut();
    __aicore__ inline void CopyOutDim1(LocalTensor<int32_t>& yUb);

    template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
    __aicore__ inline void ProcessPerFactor(CLS_NAME* objPtr);
    template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
    __aicore__ inline void ComputeAndCopyOut(int32_t num, int32_t idx, CLS_NAME* objPtr);

protected:
    const mNonZeroTiling* tiling_;
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY_;
    GlobalTensor<T1> xGm_;
    GlobalTensor<uint64_t> shapeGm_;
    GlobalTensor<int32_t> yGm_, workspaceGm_;
    GlobalTensor<int64_t> workspaceGmB64_;
    TBuf<QuePosition::VECCALC> addUb, multipDimUb;
    DataCopyExtParams copyParams_;
    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    uint64_t gmOffset_ = 0;
    uint64_t allNum_ = 0;

    int64_t blkProcessNum_ = 0;
    int64_t blockOffset_ = 0;
    int32_t processSize_ = 1;
    uint32_t repeatElmB32_ = GetVRegSize() / sizeof(int32_t);
    uint32_t repeatElmB64_ = GetVRegSize() / sizeof(int64_t);
    uint32_t repeatElmB16_ = GetVRegSize() / sizeof(uint16_t);
    uint32_t repeatElmB8_ = GetVRegSize() / sizeof(uint8_t);
    uint32_t repeatElmT1_ = GetVRegSize() / sizeof(T1);
    uint64_t arNum_ = 0;
    uint64_t arOffset_ = 0;
    uint32_t blockIdx_ = 0;
    uint16_t shapeDim_ = 3;
    int32_t elementPerBlock_ = 32 / sizeof(T2);
    int64_t nonZeroNum_ = 0;
};

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::InitBase(GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace,
                                                        const mNonZeroTiling* tilingData, TPipe* tPipe)
{
    pipe = tPipe;
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    yGm_.SetGlobalBuffer((__gm__ int32_t*)y);
    shapeGm_.SetGlobalBuffer((__gm__ uint64_t*)outShape);
    if constexpr (sizeof(T1) == sizeof(int64_t)) {
        workspaceGmB64_.SetGlobalBuffer((__gm__ int64_t*)workspace, WORKSPACE_SIZE);
    } else {
        workspaceGm_.SetGlobalBuffer((__gm__ int32_t*)workspace, WORKSPACE_SIZE);
    }
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    processSize_ = tiling_->ubFactorNum;
    shapeDim_ = tiling_->inputDims;

    int32_t xInputSize = processSize_ * sizeof(T1);                                  // inputx 需要的ub大小
    int32_t yOutputSize = processSize_ * (sizeof(int32_t) + shapeDim_ * sizeof(T2)); // 输出所需的ub大小

    pipe->InitBuffer(inQueX_, BUFFER_NUM, xInputSize);
    pipe->InitBuffer(outQueY_, BUFFER_NUM, yOutputSize);
    pipe->InitBuffer(addUb, REDUCE_UB_SIZE);
    pipe->InitBuffer(multipDimUb, MULTIP_UB_SIZE);
}

template <typename T1, typename T2>
template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
__aicore__ inline void NonZeroBigMask<T1, T2>::ProcessBase(CLS_NAME* objPtr)
{
    // 注意：不能在 SyncAll 之前 return。realCoreNum 可能小于实际启动的 AIV 核数
    // （例如 mask 元素个数很少时），未参与计算的核也必须走到同一个 SyncAll，否则同步会挂死。
    bool isActiveCore = blockIdx_ < tiling_->realCoreNum;

    if (isActiveCore) {
        blkProcessNum_ = tiling_->numPerCore;
        blockOffset_ = blockIdx_ * tiling_->numPerCore;
        if (blockIdx_ < tiling_->numTailCore) {
            blkProcessNum_ += 1;
            blockOffset_ += blockIdx_;
        } else {
            blockOffset_ += tiling_->numTailCore;
        }

        if constexpr (sizeof(T1) == sizeof(int64_t)) {
            CalcNonZeroNumPerCoreB64();
        } else {
            CalcNonZeroNumPerCore();
        }
    }
    SyncAll(); // 多核同步
    if (!isActiveCore) {
        return;
    }
    SetMultipDim();
    if constexpr (sizeof(T1) == sizeof(int64_t)) {
        GetAllNumAndOffsetB64();
    } else {
        GetAllNumAndOffset();
    }
    ProcessPerFactor<CLS_NAME, ComputeOutputPtr>(objPtr);
    ProcessOutputShape();
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::CalcNonZeroNumPerCore()
{
    int32_t loopInner = CeilDivision(blkProcessNum_, processSize_);
    int32_t tailSize = blkProcessNum_ - (loopInner - 1) * processSize_;

    int32_t loopOutter = CeilDivision(loopInner, repeatElmB32_);
    int32_t tailLoop = loopInner - (loopOutter - 1) * repeatElmB32_;

    LocalTensor<int32_t> addUbSize = addUb.Get<int32_t>();
    // 外层循环，非尾循环
    for (int32_t idxOutter = 0; idxOutter < loopOutter - 1; idxOutter++) {
        for (int32_t idxInner = 0; idxInner < repeatElmB32_; idxInner++) { // 每次处理processSize_
            CopyInAndCalcNum(addUbSize, processSize_, idxOutter, idxInner);
        }
        // reducesum 64个数
        VfReduceSum(addUbSize, repeatElmB32_);
    }

    // 外层循环，尾循环
    for (int32_t idxInner = 0; idxInner < tailLoop; idxInner++) {
        if (idxInner != tailLoop - 1) { // 每次处理processSize_
            CopyInAndCalcNum(addUbSize, processSize_, loopOutter - 1, idxInner);
        } else { // 处理tailSize
            CopyInAndCalcNum(addUbSize, tailSize, loopOutter - 1, idxInner);
        }
    }
    // vcadd tailOutter个数
    VfReduceSum(addUbSize, tailLoop);
    // 搬运到workspace
    addUbSize.SetValue(0, nonZeroNum_);
    event_t eventIdS2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdS2Mte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdS2Mte3);
    DataCopyPad(workspaceGm_[blockIdx_ * OFFSET_NUM_32], addUbSize, {1, 4, 0, 0});
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::CalcNonZeroNumPerCoreB64()
{
    int32_t loopInner = CeilDivision(blkProcessNum_, processSize_);
    int32_t tailSize = blkProcessNum_ - (loopInner - 1) * processSize_;

    int32_t loopOutter = CeilDivision(loopInner, repeatElmB64_);
    int32_t tailLoop = loopInner - (loopOutter - 1) * repeatElmB64_;

    LocalTensor<int64_t> addUbSize = addUb.Get<int64_t>();
    // 外层循环，非尾循环
    for (int32_t idxOutter = 0; idxOutter < loopOutter - 1; idxOutter++) {
        for (int32_t idxInner = 0; idxInner < repeatElmB64_; idxInner++) { // 每次处理processSize_
            CopyInAndCalcNumB64(addUbSize, processSize_, idxOutter, idxInner);
        }
        // reducesum 64个数
        VfReduceSumB64(addUbSize, repeatElmB64_);
    }

    // 外层循环，尾循环
    for (int32_t idxInner = 0; idxInner < tailLoop; idxInner++) {
        if (idxInner != tailLoop - 1) { // 每次处理processSize_
            CopyInAndCalcNumB64(addUbSize, processSize_, loopOutter - 1, idxInner);
        } else { // 处理tailSize
            CopyInAndCalcNumB64(addUbSize, tailSize, loopOutter - 1, idxInner);
        }
    }
    // vcadd tailOutter个数
    VfReduceSumB64(addUbSize, tailLoop);
    // 搬运到workspace
    addUbSize.SetValue(0, nonZeroNum_);
    event_t eventIdS2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdS2Mte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdS2Mte3);
    DataCopyPad(workspaceGmB64_[blockIdx_ * OFFSET_NUM_16], addUbSize, {1, sizeof(int64_t), 0, 0});
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::SetMultipDim()
{
    // set multipDimUb
    LocalTensor<uint32_t> multipDim = multipDimUb.Get<uint32_t>();
    for (uint32_t i = 0; i < shapeDim_; i++) {
        multipDim.SetValue(i, tiling_->mulInDimRList[i]);
        multipDim.SetValue(i + IDX_NUM_8, tiling_->quickDivRKList[i]);
        multipDim.SetValue(i + IDX_NUM_16, tiling_->quickDivRMList[i]);
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::GetAllNumAndOffset()
{
    LocalTensor<int32_t> addUbSize = addUb.Get<int32_t>();
    copyParams_.blockCount = 1;
    copyParams_.blockLen = tiling_->realCoreNum * OFFSET_NUM_128;
    copyParams_.srcStride = 0;
    copyParams_.dstStride = 0;

    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(addUbSize, workspaceGm_, copyParams_, padParams);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    ClacAllNum(addUbSize);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    gmOffset_ = tiling_->needTranspose ?
                    static_cast<uint64_t>(addUbSize.GetValue(8)) * static_cast<uint64_t>(shapeDim_) :
                    static_cast<uint64_t>(addUbSize.GetValue(8));
    allNum_ = addUbSize.GetValue(0);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::GetAllNumAndOffsetB64()
{
    LocalTensor<int64_t> addUbSize = addUb.Get<int64_t>();
    copyParams_.blockCount = 1;
    copyParams_.blockLen = tiling_->realCoreNum * OFFSET_NUM_128;
    copyParams_.srcStride = 0;
    copyParams_.dstStride = 0;

    DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
    DataCopyPad(addUbSize, workspaceGmB64_, copyParams_, padParams);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    ClacAllNumB64(addUbSize);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    gmOffset_ = tiling_->needTranspose ? addUbSize.GetValue(8) * shapeDim_ : addUbSize.GetValue(8);
    allNum_ = addUbSize.GetValue(0);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::ProcessOutputShape()
{
    LocalTensor<int32_t> outShapeUb = addUb.Get<int32_t>();
    LocalTensor<uint64_t> castOutShape = outShapeUb.ReinterpretCast<uint64_t>();
    if (tiling_->needTranspose) {
        castOutShape.SetValue(0, NON_ZERO_OUTSHAPE_DIM);
        castOutShape.SetValue(1, allNum_);
        castOutShape.SetValue(2, shapeDim_);
    } else {
        castOutShape.SetValue(0, NON_ZERO_OUTSHAPE_DIM);
        castOutShape.SetValue(1, shapeDim_);
        castOutShape.SetValue(2, allNum_);
    }
    if (blockIdx_ == 0) {
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        DataCopyPad(shapeGm_, castOutShape, {1, 24, 0, 0});
    }
}

template <typename T1, typename T2>
template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
__aicore__ inline void NonZeroBigMask<T1, T2>::ProcessPerFactor(CLS_NAME* objPtr)
{
    int32_t loopInner = blkProcessNum_ / processSize_;
    int32_t tailSize = blkProcessNum_ - loopInner * processSize_;
    for (int32_t idx = 0; idx < loopInner; idx++) {
        CopyIn(processSize_, idx);
        ComputeAndCopyOut<CLS_NAME, ComputeOutputPtr>(processSize_, idx, objPtr);
    }
    // process tail
    if (tailSize > 0) {
        CopyIn(tailSize, loopInner);
        ComputeAndCopyOut<CLS_NAME, ComputeOutputPtr>(tailSize, loopInner, objPtr);
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::CopyInAndCalcNum(LocalTensor<int32_t>& addUbSize, int32_t copyNum,
                                                                int32_t idxOutter, int32_t idxInner)
{
    LocalTensor<T1> xUb = inQueX_.AllocTensor<T1>();
    copyParams_.blockCount = 1;
    copyParams_.blockLen = copyNum * sizeof(T1);
    copyParams_.srcStride = 0;
    copyParams_.dstStride = 0;
    DataCopyPad(xUb, xGm_[blockOffset_ + (idxOutter * repeatElmB32_ + idxInner) * processSize_], copyParams_,
                padParams_);
    inQueX_.EnQue(xUb);
    LocalTensor<T1> xUbCalc = inQueX_.DeQue<T1>();
    if constexpr (sizeof(T1) == sizeof(int8_t)) {
        GetNonZeroNumB8(copyNum, (__local_mem__ T1*)xUbCalc.GetPhyAddr(),
                        (__local_mem__ int32_t*)addUbSize[idxInner].GetPhyAddr());
    } else if constexpr (sizeof(T1) == sizeof(int16_t)) {
        GetNonZeroNumB16(copyNum, (__local_mem__ T1*)xUbCalc.GetPhyAddr(),
                         (__local_mem__ int32_t*)addUbSize[idxInner].GetPhyAddr());
    } else if constexpr (sizeof(T1) == sizeof(int32_t)) {
        GetNonZeroNumB32(copyNum, (__local_mem__ T1*)xUbCalc.GetPhyAddr(),
                         (__local_mem__ int32_t*)addUbSize[idxInner].GetPhyAddr());
    }
    inQueX_.FreeTensor(xUbCalc);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::CopyInAndCalcNumB64(LocalTensor<int64_t>& addUbSize, int32_t copyNum,
                                                                   int32_t idxOutter, int32_t idxInner)
{
    LocalTensor<T1> xUb = inQueX_.AllocTensor<T1>();
    copyParams_.blockCount = 1;
    copyParams_.blockLen = copyNum * sizeof(T1);
    copyParams_.srcStride = 0;
    copyParams_.dstStride = 0;
    DataCopyPad(xUb, xGm_[blockOffset_ + (idxOutter * repeatElmB64_ + idxInner) * processSize_], copyParams_,
                padParams_);
    inQueX_.EnQue(xUb);
    LocalTensor<T1> xUbCalc = inQueX_.DeQue<T1>();
    GetNonZeroNumB64(copyNum, (__local_mem__ T1*)xUbCalc.GetPhyAddr(),
                     (__local_mem__ int64_t*)addUbSize[idxInner].GetPhyAddr());
    inQueX_.FreeTensor(xUbCalc);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::VfReduceSum(LocalTensor<int32_t>& addUbSize, int32_t num)
{
    auto addUbPtr = (__local_mem__ int32_t*)addUbSize.GetPhyAddr();
    uint32_t allMask = num;
    uint32_t oneMask = 1;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> addReg;
        AscendC::Reg::RegTensor<int32_t> dstReg;
        AscendC::Reg::MaskReg addMask = AscendC::Reg::UpdateMask<int32_t>(allMask);
        AscendC::Reg::MaskReg oneMaskReg = AscendC::Reg::UpdateMask<int32_t>(oneMask);
        DataCopy(addReg, addUbPtr);
        ReduceSum(dstReg, addReg, addMask);
        DataCopy(addUbPtr, dstReg, oneMaskReg);
    }
    event_t eventIdS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdS2V);
    WaitFlag<HardEvent::V_S>(eventIdS2V);
    // vcadd的值做累加, 得到非0的总数
    nonZeroNum_ += addUbSize.GetValue(0);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::VfReduceSumB64(LocalTensor<int64_t>& addUbSize, int32_t num)
{
    auto addUbPtr = (__local_mem__ int64_t*)addUbSize.GetPhyAddr();
    uint32_t allMask = num;
    uint32_t oneMask = 1;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int64_t> addReg;
        AscendC::Reg::RegTensor<int64_t> dstReg;
        AscendC::Reg::MaskReg addMask = AscendC::Reg::UpdateMask<int64_t>(allMask);
        AscendC::Reg::MaskReg oneMaskReg = AscendC::Reg::UpdateMask<int64_t>(oneMask);
        DataCopy(addReg, addUbPtr);
        ReduceSum(dstReg, addReg, addMask);
        DataCopy(addUbPtr, dstReg, oneMaskReg);
    }
    event_t eventIdS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdS2V);
    WaitFlag<HardEvent::V_S>(eventIdS2V);
    // vcadd的值做累加, 得到非0的总数
    nonZeroNum_ += addUbSize.GetValue(0);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::ClacAllNum(LocalTensor<int32_t>& addUbSize)
{
    auto addUbPtr = (__local_mem__ int32_t*)addUbSize.GetPhyAddr();
    uint32_t realCoreNum = tiling_->realCoreNum;
    uint32_t oneMask = 1;
    uint32_t blockNum = blockIdx_;
    uint16_t loopNum = CeilDivision(realCoreNum, repeatElmB32_);
    uint32_t repeatElmB32 = repeatElmB32_;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> idxReg;
        AscendC::Reg::RegTensor<int32_t> dstReg;
        AscendC::Reg::RegTensor<int32_t> offsetReg;
        Duplicate(dstReg, 0);
        Duplicate(offsetReg, 0);
        AscendC::Reg::MaskReg coreMask;
        AscendC::Reg::MaskReg oneMaskReg;
        AscendC::Reg::MaskReg blockMask;
        oneMaskReg = AscendC::Reg::UpdateMask<int32_t>(oneMask);
        AscendC::Reg::Arange(idxReg, 0);
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::Reg::RegTensor<int32_t> addReg, idxRegTmp;
            AscendC::Reg::RegTensor<int32_t> dstRegTmp;
            AscendC::Reg::RegTensor<int32_t> offsetRegTmp;
            coreMask = AscendC::Reg::UpdateMask<int32_t>(realCoreNum);
            blockMask = AscendC::Reg::UpdateMask<int32_t>(blockNum);
            Muls(idxRegTmp, idxReg, OFFSET_NUM_32, coreMask);
            DataCopyGather(addReg, addUbPtr, (AscendC::Reg::RegTensor<uint32_t>&)idxRegTmp, coreMask);
            ReduceSum(dstRegTmp, addReg, coreMask);
            ReduceSum(offsetRegTmp, addReg, blockMask);
            Add(dstReg, dstReg, dstRegTmp, oneMaskReg);
            Add(offsetReg, offsetReg, offsetRegTmp, oneMaskReg);
            AscendC::Reg::Adds(idxReg, idxReg, repeatElmB32, coreMask);
        }
        DataCopy(addUbPtr, dstReg, oneMaskReg);
        DataCopy(addUbPtr + 8, offsetReg, oneMaskReg);
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::ClacAllNumB64(LocalTensor<int64_t>& addUbSize)
{
    auto addUbPtr = (__local_mem__ int64_t*)addUbSize.GetPhyAddr();
    uint32_t realCoreNum = tiling_->realCoreNum;
    uint32_t oneMask = 1;
    uint32_t blockNum = blockIdx_;
    uint16_t loopNum = CeilDivision(realCoreNum, repeatElmT1_);
    uint32_t repeatElmT1 = repeatElmT1_;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int64_t> idxReg;
        AscendC::Reg::RegTensor<int64_t> dstReg;
        AscendC::Reg::RegTensor<int64_t> offsetReg;
        Duplicate(dstReg, 0);
        Duplicate(offsetReg, 0);
        AscendC::Reg::MaskReg coreMask;
        AscendC::Reg::MaskReg oneMaskReg;
        AscendC::Reg::MaskReg blockMask;
        oneMaskReg = AscendC::Reg::UpdateMask<int64_t>(oneMask);
        AscendC::Reg::Arange(idxReg, 0);
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::Reg::RegTensor<int64_t> addReg, idxRegTmp;
            AscendC::Reg::RegTensor<int64_t> dstRegTmp;
            AscendC::Reg::RegTensor<int64_t> offsetRegTmp;
            coreMask = AscendC::Reg::UpdateMask<int64_t>(realCoreNum);
            blockMask = AscendC::Reg::UpdateMask<int64_t>(blockNum);
            Muls(idxRegTmp, idxReg, OFFSET_NUM_16, coreMask);
            DataCopyGather(addReg, addUbPtr, (AscendC::Reg::RegTensor<uint64_t>&)idxRegTmp, coreMask);
            ReduceSum(dstRegTmp, addReg, coreMask);
            ReduceSum(offsetRegTmp, addReg, blockMask);
            Add(dstReg, dstReg, dstRegTmp, oneMaskReg);
            Add(offsetReg, offsetReg, offsetRegTmp, oneMaskReg);
            AscendC::Reg::Adds(idxReg, idxReg, repeatElmT1, coreMask);
        }
        DataCopy(addUbPtr, dstReg, oneMaskReg);
        DataCopy(addUbPtr + 8, offsetReg, oneMaskReg);
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::CopyIn(int32_t num, int32_t idx)
{
    LocalTensor<T1> xUb = inQueX_.AllocTensor<T1>();
    copyParams_.blockCount = 1;
    copyParams_.blockLen = num * sizeof(T1);
    copyParams_.srcStride = 0;
    copyParams_.dstStride = 0;
    DataCopyPad(xUb, xGm_[blockOffset_ + idx * processSize_], copyParams_, padParams_);
    inQueX_.EnQue(xUb);
}

template <typename T1, typename T2>
template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
__aicore__ inline void NonZeroBigMask<T1, T2>::ComputeAndCopyOut(int32_t num, int32_t idx, CLS_NAME* objPtr)
{
    LocalTensor<int32_t> yUb = outQueY_.AllocTensor<int32_t>();
    LocalTensor<T1> xUbCalc = inQueX_.DeQue<T1>();
    if constexpr (sizeof(T1) == sizeof(int8_t)) {
        SqzNonZeroNumB8(num, (__local_mem__ T1*)xUbCalc.GetPhyAddr(), (__local_mem__ int32_t*)yUb.GetPhyAddr(), idx);
    } else if constexpr (sizeof(T1) == sizeof(int16_t)) {
        SqzNonZeroNumB16(num, (__local_mem__ T1*)xUbCalc.GetPhyAddr(), (__local_mem__ int32_t*)yUb.GetPhyAddr(), idx);
    } else if constexpr (sizeof(T1) == sizeof(int32_t)) {
        SqzNonZeroNumB32(num, (__local_mem__ T1*)xUbCalc.GetPhyAddr(), (__local_mem__ int32_t*)yUb.GetPhyAddr(), idx);
    } else if constexpr (sizeof(T1) == sizeof(int64_t)) {
        SqzNonZeroNumB64(num, (__local_mem__ T1*)xUbCalc.GetPhyAddr(), (__local_mem__ int32_t*)yUb.GetPhyAddr(), idx);
    }
    inQueX_.FreeTensor(xUbCalc);

    if (shapeDim_ == 1) {
        CopyOutDim1(yUb);
        arOffset_ += arNum_;
    } else if (tiling_->needTranspose) {
        ComputeOutputAndTrans(yUb);
        CopyOutWithTrans();
        arOffset_ += arNum_ * shapeDim_;
    } else {
        (objPtr->*ComputeOutputPtr)(yUb);
        CopyOut();
        arOffset_ += arNum_;
    }

    outQueY_.FreeTensor(yUb);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::ComputeOutputAndTrans(LocalTensor<int32_t>& yUb)
{
    auto srcPtr = (__local_mem__ uint32_t*)yUb.GetPhyAddr();
    __local_mem__ uint32_t* dstPtr = nullptr;
    if constexpr (IsSameType<T2, int64_t>::value) {
        dstPtr = (__local_mem__ uint32_t*)yUb[processSize_ * (shapeDim_ + 1)].GetPhyAddr();
    } else {
        dstPtr = (__local_mem__ uint32_t*)yUb[processSize_].GetPhyAddr();
    }

    if (shapeDim_ == 2) {
        TransOutputDim2(srcPtr, dstPtr);
    } else if (shapeDim_ == 4) {
        TransOutputDim4(srcPtr, dstPtr);
    } else {
        TransOutput(srcPtr, dstPtr);
    }

    CastOutput(yUb, processSize_ * (shapeDim_ + 1), arNum_ * shapeDim_, 1);
    outQueY_.EnQue(yUb);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::TransOutputDim2(__local_mem__ uint32_t* srcPtr,
                                                               __local_mem__ uint32_t* dstPtr)
{
    uint16_t repeatTimes = arNum_ / repeatElmB32_;
    uint16_t tailNum = arNum_ - repeatTimes * repeatElmB32_;
    uint16_t tailLoop = tailNum > 0 ? 1 : 0;
    uint32_t sreg = (uint32_t)tailNum * 2;
    LocalTensor<uint32_t> multipDim = multipDimUb.Get<uint32_t>();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg preg, preg1, preg2;
        AscendC::Reg::RegTensor<uint32_t> srcReg, divReg, shapeReg, subReg, mulReg, mReg, divReg0;
        AscendC::Reg::RegTensor<uint32_t> trans1Reg, trans2Reg;
        preg = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        Duplicate(shapeReg, multipDim.GetValue(IDX_NUM_0));
        Duplicate(mReg, multipDim.GetValue(IDX_NUM_16));
        int16_t kValue = multipDim.GetValue(IDX_NUM_8);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            DataCopy(srcReg, srcPtr + i * repeatElmB32_);
            Mull(divReg0, divReg, srcReg, mReg, preg);
            Add(divReg0, srcReg, divReg, preg);
            ShiftRights(divReg, divReg0, kValue, preg);
            Mul(mulReg, divReg, shapeReg, preg);
            Sub(subReg, srcReg, mulReg, preg);
            // 2维转置
            Interleave(trans1Reg, trans2Reg, divReg, subReg);
            DataCopy(dstPtr + i * repeatElmB32_ * 2, trans1Reg, preg);
            DataCopy(dstPtr + i * repeatElmB32_ * 2 + repeatElmB32_, trans2Reg, preg);
        }
        for (uint16_t i = 0; i < tailLoop; i++) {
            preg1 = AscendC::Reg::UpdateMask<int32_t>(sreg);
            preg2 = AscendC::Reg::UpdateMask<int32_t>(sreg);
            DataCopy(srcReg, srcPtr + repeatTimes * repeatElmB32_);
            Mull(divReg0, divReg, srcReg, mReg, preg);
            Add(divReg0, srcReg, divReg, preg);
            ShiftRights(divReg, divReg0, kValue, preg);
            Mul(mulReg, divReg, shapeReg, preg);
            Sub(subReg, srcReg, mulReg, preg);
            // 2维转置
            Interleave(trans1Reg, trans2Reg, divReg, subReg);
            DataCopy(dstPtr + repeatTimes * repeatElmB32_ * 2, trans1Reg, preg1);
            DataCopy(dstPtr + repeatTimes * repeatElmB32_ * 2 + repeatElmB32_, trans2Reg, preg2);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::TransOutputDim4(__local_mem__ uint32_t* srcPtr,
                                                               __local_mem__ uint32_t* dstPtr)
{
    uint16_t repeatTimes = arNum_ / repeatElmB32_;
    uint16_t tailNum = arNum_ - repeatTimes * repeatElmB32_;
    uint16_t tailLoop = tailNum > 0 ? 1 : 0;
    uint32_t sreg = (uint32_t)tailNum * 4;
    LocalTensor<uint32_t> multipDim = multipDimUb.Get<uint32_t>();
    __VEC_SCOPE__
    {
        uint32_t repeatElm = repeatElmB32_;
        AscendC::Reg::MaskReg preg, preg1, preg2, preg3, preg4;
        AscendC::Reg::RegTensor<uint32_t> srcReg, divReg, shapeReg0, shapeReg1, shapeReg2, subReg, mulReg;
        AscendC::Reg::RegTensor<uint32_t> divReg1, divReg2, trans1Reg, trans2Reg, trans3Reg, trans4Reg;
        AscendC::Reg::RegTensor<uint32_t> mReg0, mReg1, mReg2, divReg0;
        Duplicate(shapeReg0, multipDim.GetValue(IDX_NUM_0));
        Duplicate(shapeReg1, multipDim.GetValue(IDX_NUM_1));
        Duplicate(shapeReg2, multipDim.GetValue(IDX_NUM_2));
        Duplicate(mReg0, multipDim.GetValue(IDX_NUM_16));
        Duplicate(mReg1, multipDim.GetValue(IDX_NUM_16 + 1));
        Duplicate(mReg2, multipDim.GetValue(IDX_NUM_16 + 2));
        preg = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        int16_t kValue0 = multipDim.GetValue(IDX_NUM_8);
        int16_t kValue1 = multipDim.GetValue(IDX_NUM_8 + 1);
        int16_t kValue2 = multipDim.GetValue(IDX_NUM_8 + 2);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            DataCopy(srcReg, srcPtr + i * repeatElm);
            Mull(divReg0, divReg, srcReg, mReg0, preg);
            Add(divReg0, srcReg, divReg, preg);
            ShiftRights(divReg, divReg0, kValue0, preg);

            Mul(mulReg, divReg, shapeReg0, preg);
            Sub(subReg, srcReg, mulReg, preg);

            Mull(divReg0, divReg1, subReg, mReg1, preg);
            Add(divReg0, subReg, divReg1, preg);
            ShiftRights(divReg1, divReg0, kValue1, preg);
            Mul(mulReg, divReg1, shapeReg1, preg);
            Sub(subReg, subReg, mulReg, preg);

            Mull(divReg0, divReg2, subReg, mReg2, preg);
            Add(divReg0, subReg, divReg2, preg);
            ShiftRights(divReg2, divReg0, kValue2, preg);

            Mul(mulReg, divReg2, shapeReg2, preg);
            Sub(subReg, subReg, mulReg, preg);
            // 4维度转置
            Interleave(trans1Reg, trans2Reg, divReg, divReg2);
            Interleave(trans3Reg, trans4Reg, divReg1, subReg);
            Interleave(divReg, divReg1, trans1Reg, trans3Reg);
            Interleave(divReg2, subReg, trans2Reg, trans4Reg);

            DataCopy(dstPtr + i * repeatElm * 4, divReg, preg);
            DataCopy(dstPtr + i * repeatElm * 4 + repeatElm, divReg1, preg);
            DataCopy(dstPtr + i * repeatElm * 4 + repeatElm * 2, divReg2, preg);
            DataCopy(dstPtr + i * repeatElm * 4 + repeatElm * 3, subReg, preg);
        }
        // process tail
        for (uint16_t i = 0; i < tailLoop; i++) {
            DataCopy(srcReg, srcPtr + repeatTimes * repeatElm);
            Mull(divReg0, divReg, srcReg, mReg0, preg);
            Add(divReg0, srcReg, divReg, preg);
            ShiftRights(divReg, divReg0, kValue0, preg);

            Mul(mulReg, divReg, shapeReg0, preg);
            Sub(subReg, srcReg, mulReg, preg);

            Mull(divReg0, divReg1, subReg, mReg1, preg);
            Add(divReg0, subReg, divReg1, preg);
            ShiftRights(divReg1, divReg0, kValue1, preg);
            Mul(mulReg, divReg1, shapeReg1, preg);
            Sub(subReg, subReg, mulReg, preg);

            Mull(divReg0, divReg2, subReg, mReg2, preg);
            Add(divReg0, subReg, divReg2, preg);
            ShiftRights(divReg2, divReg0, kValue2, preg);

            Mul(mulReg, divReg2, shapeReg2, preg);
            Sub(subReg, subReg, mulReg, preg);

            Interleave(trans1Reg, trans2Reg, divReg, divReg2);
            Interleave(trans3Reg, trans4Reg, divReg1, subReg);
            Interleave(divReg, divReg1, trans1Reg, trans3Reg);
            Interleave(divReg2, subReg, trans2Reg, trans4Reg);
            preg1 = AscendC::Reg::UpdateMask<int32_t>(sreg);
            preg2 = AscendC::Reg::UpdateMask<int32_t>(sreg);
            preg3 = AscendC::Reg::UpdateMask<int32_t>(sreg);
            preg4 = AscendC::Reg::UpdateMask<int32_t>(sreg);
            DataCopy(dstPtr + repeatTimes * repeatElm * 4, divReg, preg1);
            DataCopy(dstPtr + repeatTimes * repeatElm * 4 + repeatElm, divReg1, preg2);
            DataCopy(dstPtr + repeatTimes * repeatElm * 4 + repeatElm * 2, divReg2, preg3);
            DataCopy(dstPtr + repeatTimes * repeatElm * 4 + repeatElm * 3, subReg, preg4);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::TransOutput(__local_mem__ uint32_t* srcPtr,
                                                           __local_mem__ uint32_t* dstPtr)
{
    uint16_t repeatTimes = CeilDivision(arNum_, repeatElmB32_);
    uint16_t loopShape = shapeDim_ - 2;
    uint32_t sreg = (uint32_t)arNum_;
    uint32_t repeatElmB32 = repeatElmB32_;
    LocalTensor<uint32_t> multipDim = multipDimUb.Get<uint32_t>();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg preg;
        AscendC::Reg::RegTensor<uint32_t> srcReg, divReg1, divReg, shape12Reg, shape2Reg, mReg, divReg0;
        AscendC::Reg::RegTensor<uint32_t> mulReg, mul2Reg, subReg, idxTransReg, addidxReg, mReg1;
        AscendC::Reg::RegTensor<int32_t> idxReg;
        Duplicate(shape12Reg, multipDim.GetValue(IDX_NUM_0));
        Duplicate(mReg, multipDim.GetValue(IDX_NUM_16));
        int16_t kValue = (int16_t)(multipDim.GetValue(IDX_NUM_8));
        AscendC::Reg::Arange(idxReg, 0);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<int32_t>(sreg);
            DataCopy(srcReg, srcPtr + i * repeatElmB32);
            // first output
            Mull(divReg0, divReg, srcReg, mReg, preg);
            Add(divReg0, srcReg, divReg, preg);
            ShiftRights(divReg, divReg0, kValue, preg);

            Mul(mulReg, divReg, shape12Reg, preg);
            Sub(subReg, srcReg, mulReg, preg);
            Muls(idxTransReg, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, shapeDim_, preg);
            DataCopyScatter(dstPtr, divReg, idxTransReg, preg);
            AscendC::Reg::Adds(idxReg, idxReg, repeatElmB32, preg);
            // middle outputs
            for (uint16_t j = 0; j < loopShape; j++) {
                Duplicate(shape2Reg, multipDim.GetValue(j + 1));
                Duplicate(mReg1, multipDim.GetValue(j + IDX_NUM_16 + 1));
                Mull(divReg0, divReg1, subReg, mReg1, preg);
                Add(divReg0, subReg, divReg1, preg);
                ShiftRights(divReg1, divReg0, (int16_t)(multipDim.GetValue(j + IDX_NUM_8 + 1)), preg);
                Mul(mul2Reg, divReg1, shape2Reg, preg);
                Sub(subReg, subReg, mul2Reg, preg);
                Adds(addidxReg, idxTransReg, j + 1, preg);
                DataCopyScatter(dstPtr, divReg1, addidxReg, preg);
            }
            // last output
            Adds(addidxReg, idxTransReg, loopShape + 1, preg);
            DataCopyScatter(dstPtr, subReg, addidxReg, preg);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::CopyOutWithTrans()
{
    LocalTensor<int32_t> yUb = outQueY_.DeQue<int32_t>();

    DataCopyExtParams copyOutIndiceParams{1, 0, 0, 0, 0};
    copyOutIndiceParams.blockCount = 1;
    copyOutIndiceParams.blockLen = arNum_ * shapeDim_ * sizeof(T2);
    copyOutIndiceParams.srcStride = 0;
    copyOutIndiceParams.dstStride = 0;

    uint64_t offset = gmOffset_ + arOffset_;
    if constexpr (IsSameType<T2, int64_t>::value) {
        offset *= 2;
    }
    DataCopyPad(yGm_[offset], yUb[processSize_], copyOutIndiceParams);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::ComputeOutputBaseFunc(
    __local_mem__ uint32_t* srcPtr, __local_mem__ uint32_t* dstPtr, __local_mem__ uint32_t* dstLastPtr, MaskReg& preg,
    AddrReg& vagReg, RegTensor<uint32_t>& srcReg, RegTensor<uint32_t>& subReg, RegTensor<uint32_t>& shapeReg,
    RegTensor<uint32_t>& mulReg, RegTensor<uint32_t>& mReg, RegTensor<uint32_t>& divReg0, RegTensor<uint32_t>& divReg1,
    uint32_t sValue, uint32_t mValue, int16_t kValue)
{
    DataCopy(srcReg, srcPtr, vagReg);
    Duplicate(shapeReg, sValue);
    Duplicate(mReg, mValue);
    Mull(divReg0, divReg1, srcReg, mReg, preg);
    Add(divReg0, srcReg, divReg1, preg);
    ShiftRights(divReg1, divReg0, kValue, preg);
    DataCopy(dstPtr, divReg1, vagReg, preg);
    Mul(mulReg, divReg1, shapeReg, preg);
    Sub(subReg, srcReg, mulReg, preg);
    DataCopy(dstLastPtr, subReg, vagReg, preg);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::ComputeOutput(LocalTensor<int32_t>& yUb)
{
    LocalTensor<uint32_t> multipDim = multipDimUb.Get<uint32_t>();
    uint16_t repeatTimes = CeilDivision(arNum_, repeatElmB32_);
    uint16_t loopShape = shapeDim_ - 2;
    uint32_t sreg = (uint32_t)arNum_;
    auto srcPtr = (__local_mem__ uint32_t*)yUb.GetPhyAddr();
    __local_mem__ uint32_t* dstPtr = nullptr;
    if constexpr (IsSameType<T2, int64_t>::value) {
        dstPtr = (__local_mem__ uint32_t*)yUb[processSize_ * (shapeDim_ + 1)].GetPhyAddr();
    } else {
        dstPtr = (__local_mem__ uint32_t*)yUb[processSize_].GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg preg;
        AscendC::Reg::RegTensor<uint32_t> srcReg, subReg, divReg, divReg1, shapeReg0, shape2Reg;
        AscendC::Reg::RegTensor<uint32_t> mulReg, mReg0, mReg1, divReg0;
        AscendC::Reg::RegTensor<uint32_t> mul2Reg;
        Duplicate(shapeReg0, multipDim.GetValue(IDX_NUM_0));
        Duplicate(mReg0, multipDim.GetValue(IDX_NUM_16));
        int16_t kValue = (int16_t)(multipDim.GetValue(IDX_NUM_8));
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<int32_t>(sreg);
            DataCopy(srcReg, srcPtr + i * repeatElmB32_);
            // first output
            Mull(divReg0, divReg, srcReg, mReg0, preg);
            Add(divReg0, srcReg, divReg, preg);
            ShiftRights(divReg, divReg0, kValue, preg);
            DataCopy(dstPtr + i * repeatElmB32_, divReg, preg);
            Mul(mulReg, divReg, shapeReg0, preg);
            Sub(subReg, srcReg, mulReg, preg);
            // middle outputs
            for (uint16_t j = 0; j < loopShape; j++) {
                Duplicate(shape2Reg, multipDim.GetValue(j + 1));
                Duplicate(mReg1, multipDim.GetValue(j + IDX_NUM_16 + 1));
                Mull(divReg0, divReg1, subReg, mReg1, preg);
                Add(divReg0, subReg, divReg1, preg);
                ShiftRights(divReg1, divReg0, (int16_t)(multipDim.GetValue(j + IDX_NUM_8 + 1)), preg);
                DataCopy(dstPtr + processSize_ * (j + 1) + i * repeatElmB32_, divReg1, preg);
                Mul(mul2Reg, divReg1, shape2Reg, preg);
                Sub(subReg, subReg, mul2Reg, preg);
            }
            // last output
            DataCopy(dstPtr + processSize_ * (shapeDim_ - 1) + i * repeatElmB32_, subReg, preg);
        }
    }
    CastOutput(yUb, processSize_ * (shapeDim_ + 1), arNum_, shapeDim_);
    outQueY_.EnQue(yUb);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::CopyOut()
{
    LocalTensor<int32_t> yUb = outQueY_.DeQue<int32_t>();

    DataCopyExtParams copyOutIndiceParams{1, 0, 0, 0, 0};
    copyOutIndiceParams.blockCount = shapeDim_;
    copyOutIndiceParams.blockLen = arNum_ * sizeof(T2);
    int32_t alignNum = CeilDivision(arNum_, elementPerBlock_) * elementPerBlock_;
    copyOutIndiceParams.srcStride = (processSize_ - alignNum) / elementPerBlock_;
    copyOutIndiceParams.dstStride = (allNum_ - arNum_) * sizeof(T2);
    uint64_t offset = gmOffset_ + arOffset_;
    if constexpr (IsSameType<T2, int64_t>::value) {
        offset *= 2;
    }
    DataCopyPad(yGm_[offset], yUb[processSize_], copyOutIndiceParams);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::CopyOutDim1(LocalTensor<int32_t>& yUb)
{
    CastOutput(yUb, 0, arNum_, 1);
    outQueY_.EnQue(yUb);
    LocalTensor<int32_t> yUbOut = outQueY_.DeQue<int32_t>();

    DataCopyExtParams copyOutIndiceParams{1, 0, 0, 0, 0};
    copyOutIndiceParams.blockCount = 1;
    copyOutIndiceParams.blockLen = arNum_ * sizeof(T2);
    copyOutIndiceParams.srcStride = 0;
    copyOutIndiceParams.dstStride = (allNum_ - arNum_) * sizeof(T2);

    if constexpr (IsSameType<T2, int64_t>::value) {
        DataCopyPad(yGm_[(gmOffset_ + arOffset_) * 2], yUbOut[processSize_], copyOutIndiceParams);
    } else {
        DataCopyPad(yGm_[gmOffset_ + arOffset_], yUbOut, copyOutIndiceParams);
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::CastOutput(LocalTensor<int32_t>& yUb, int32_t srcOffset, int32_t num,
                                                          uint16_t dims)
{
    if constexpr (IsSameType<T2, int64_t>::value) {
        // need cast to int64
        uint16_t repeatTimes = CeilDivision(num, repeatElmB64_);
        uint16_t tail = num % repeatElmB64_ * 2;
        tail = tail == 0 ? repeatElmB64_ * 2 : tail;
        auto srcPtr = (__local_mem__ int32_t*)yUb[srcOffset].GetPhyAddr();
        auto dstPtr = (__local_mem__ int32_t*)yUb[processSize_].GetPhyAddr();
        uint16_t srcOffsetJ = processSize_;
        uint16_t dstOffsetJ = processSize_ * 2;
        uint16_t srcOffsetI = repeatElmB64_;
        uint16_t dstOffsetI = repeatElmB32_;
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg pregTail, pregAll, preg;
            AscendC::Reg::RegTensor<int32_t> srcReg;
            AscendC::Reg::RegTensor<int64_t> dstReg;
            for (uint16_t j = 0; j < dims; j++) {
                uint32_t sreg = num * 2;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = AscendC::Reg::UpdateMask<int32_t>(sreg);
                    AscendC::Reg::AddrReg srcAreg = AscendC::Reg::CreateAddrReg<uint32_t>(j, srcOffsetJ, i, srcOffsetI);
                    AscendC::Reg::AddrReg dstAreg = AscendC::Reg::CreateAddrReg<uint32_t>(j, dstOffsetJ, i, dstOffsetI);
                    DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_UNPACK_B32>(srcReg, srcPtr, srcAreg);
                    DataCopy(dstPtr, srcReg, dstAreg, preg);
                }
            }
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::SqzNonZeroNumB8(int32_t processNum, __local_mem__ T1* xUbPtr,
                                                               __local_mem__ int32_t* yUbPtr, int32_t idx)
{
    uint16_t repeatTimes = CeilDivision(processNum, repeatElmB8_);
    int32_t startIdx = blockOffset_ + idx * processSize_;
    int32_t offsetNum64 = OFFSET_NUM_64;
    int32_t offsetNumSub192 = -OFFSET_NUM_192;
    int32_t repeatElmB8 = repeatElmB8_;
    int32_t offsetNum = offsetNumSub192 + repeatElmB8;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1> xSrcReg;
        uint32_t sreg = processNum;
        AscendC::Reg::ClearSpr<SpecialPurposeReg::AR>();
        AscendC::Reg::MaskReg preg, preg1, cmpReg, pregLower, pregHigher, maskReg1, maskReg2, maskReg3, maskReg4;
        AscendC::Reg::UnalignReg ureg0;
        RegTensor<int32_t> vsqzReg;
        RegTensor<int32_t> idsReg;
        preg1 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange(idsReg, startIdx);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<T1>(sreg);
            DataCopy(xSrcReg, xUbPtr + i * repeatElmB8);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);

            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(pregLower, cmpReg);
            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(maskReg1, pregLower);
            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskReg2, pregLower);

            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(pregHigher, cmpReg);
            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(maskReg3, pregHigher);
            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskReg4, pregHigher);

            AscendC::Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, maskReg1);
            AscendC::Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(yUbPtr, vsqzReg, ureg0);

            AscendC::Reg::Adds(idsReg, idsReg, offsetNum64, preg1);
            AscendC::Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, maskReg2);
            AscendC::Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(yUbPtr, vsqzReg, ureg0);

            AscendC::Reg::Adds(idsReg, idsReg, offsetNum64, preg1);
            AscendC::Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, maskReg3);
            AscendC::Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(yUbPtr, vsqzReg, ureg0);

            AscendC::Reg::Adds(idsReg, idsReg, offsetNum64, preg1);
            AscendC::Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, maskReg4);
            AscendC::Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(yUbPtr, vsqzReg, ureg0);

            AscendC::Reg::Adds(idsReg, idsReg, offsetNum, preg1);
        }
        AscendC::Reg::DataCopyUnAlignPost(yUbPtr, ureg0);
    }
    arNum_ = (AscendC::Reg::GetSpr<SpecialPurposeReg::AR>()) / 4;
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::SqzNonZeroNumB16(int32_t processNum, __local_mem__ T1* xUbPtr,
                                                                __local_mem__ int32_t* yUbPtr, int32_t idx)
{
    uint16_t repeatTimes = CeilDivision(processNum, repeatElmB16_);
    int32_t startIdx = blockOffset_ + idx * processSize_;
    int32_t repeatElmB16 = repeatElmB16_;
    int32_t offsetNum64 = OFFSET_NUM_64;
    int32_t offsetNum = repeatElmB16_ - OFFSET_NUM_64;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1> xSrcReg;
        uint32_t sreg = processNum;
        AscendC::Reg::ClearSpr<SpecialPurposeReg::AR>();
        AscendC::Reg::MaskReg preg, preg1, cmpReg, pregLower, pregHigher;
        Reg::UnalignReg ureg0;
        RegTensor<int32_t> vsqzReg;
        RegTensor<int32_t> idsReg;
        preg1 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange(idsReg, startIdx);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<T1>(sreg);
            DataCopy(xSrcReg, xUbPtr + i * repeatElmB16_);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);

            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(pregLower, cmpReg);
            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(pregHigher, cmpReg);

            AscendC::Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, pregLower);
            AscendC::Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(yUbPtr, vsqzReg, ureg0);

            AscendC::Reg::Adds(idsReg, idsReg, offsetNum64, preg1);
            AscendC::Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, pregHigher);
            AscendC::Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(yUbPtr, vsqzReg, ureg0);

            AscendC::Reg::Adds(idsReg, idsReg, offsetNum, preg1);
        }
        AscendC::Reg::DataCopyUnAlignPost(yUbPtr, ureg0);
    }
    arNum_ = (AscendC::Reg::GetSpr<SpecialPurposeReg::AR>()) / 4;
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::SqzNonZeroNumB32(int32_t processNum, __local_mem__ T1* xUbPtr,
                                                                __local_mem__ int32_t* yUbPtr, int32_t idx)
{
    uint16_t repeatTimes = CeilDivision(processNum, repeatElmB32_);
    int32_t startIdx = blockOffset_ + idx * processSize_;
    int32_t repeatElmB32 = repeatElmB32_;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1> xSrcReg;
        uint32_t sreg = processNum;
        AscendC::Reg::ClearSpr<SpecialPurposeReg::AR>();
        AscendC::Reg::MaskReg preg, preg1, cmpReg;
        Reg::UnalignReg ureg0;
        RegTensor<int32_t> vsqzReg;
        RegTensor<int32_t> idsReg;
        preg1 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange(idsReg, startIdx);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<T1>(sreg);
            DataCopy(xSrcReg, xUbPtr + i * repeatElmB32);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            AscendC::Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, cmpReg);
            AscendC::Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(yUbPtr, vsqzReg, ureg0);
            AscendC::Reg::Adds(idsReg, idsReg, repeatElmB32, preg1);
        }
        AscendC::Reg::DataCopyUnAlignPost(yUbPtr, ureg0);
    }
    arNum_ = (AscendC::Reg::GetSpr<SpecialPurposeReg::AR>()) / 4;
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::SqzNonZeroNumB64(int32_t processNum, __local_mem__ T1* xUbPtr,
                                                                __local_mem__ int32_t* yUbPtr, int32_t idx)
{
    uint16_t repeatTimes = CeilDivision(processNum, repeatElmB64_);
    uint64_t startIdx = blockOffset_ + idx * processSize_;
    int32_t repeatElmB64 = repeatElmB64_;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T1> xSrcReg;
        uint32_t sreg = processNum;
        AscendC::Reg::ClearSpr<SpecialPurposeReg::AR>();
        AscendC::Reg::MaskReg preg, preg1, cmpReg;
        Reg::UnalignReg ureg0;
        RegTensor<int32_t> vsqzReg;
        RegTensor<int32_t> idsReg;
        preg1 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange(idsReg, startIdx);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<T1>(sreg);
            DataCopy(xSrcReg, xUbPtr + i * repeatElmB64);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            AscendC::Reg::MaskReg maskHalf;
            AscendC::Reg::MaskPack<Reg::HighLowPart::LOWEST>(maskHalf, cmpReg);
            AscendC::Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, maskHalf);
            AscendC::Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(yUbPtr, vsqzReg, ureg0);

            AscendC::Reg::Adds(idsReg, idsReg, repeatElmB64, preg1);
        }
        AscendC::Reg::DataCopyUnAlignPost(yUbPtr, ureg0);
    }
    arNum_ = (AscendC::Reg::GetSpr<SpecialPurposeReg::AR>()) / 4;
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::GetNonZeroNumB8(int32_t processNum, __local_mem__ T1* xUbPtr,
                                                               __local_mem__ int32_t* dstUbPtr)
{
    uint16_t repeatTimes = CeilDivision(processNum, repeatElmB8_);
    uint32_t addMask = repeatElmB32_;
    int32_t oneMask = 1;
    __VEC_SCOPE__
    {
        uint32_t sreg = processNum;
        AscendC::Reg::MaskReg preg, cmpReg, addComReg;
        AscendC::Reg::RegTensor<T1> xSrcReg;
        AscendC::Reg::RegTensor<int8_t> src1Reg, src0Reg, selectReg;
        AscendC::Reg::RegTensor<int16_t> uint16Reg0, uint16Reg1;
        AscendC::Reg::RegTensor<int32_t> dstReg, addReg, uint32Reg0, uint32Reg1, uint32Reg2, uint32Reg3;
        AscendC::Reg::UnalignReg u0;
        Duplicate(src1Reg, (uint8_t)1);
        Duplicate(src0Reg, (uint8_t)0);
        Duplicate(addReg, (int32_t)0);
        addComReg = AscendC::Reg::UpdateMask<int32_t>(addMask);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<T1>(sreg);
            DataCopy(xSrcReg, xUbPtr + i * repeatElmB8_);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            Select(selectReg, src1Reg, src0Reg, cmpReg);

            UnPack<int16_t, int8_t, AscendC::Reg::HighLowPart::LOWEST>(uint16Reg0, selectReg);
            UnPack<int32_t, int16_t, AscendC::Reg::HighLowPart::LOWEST>(uint32Reg0, uint16Reg0);
            UnPack<int32_t, int16_t, AscendC::Reg::HighLowPart::HIGHEST>(uint32Reg1, uint16Reg0);
            Add(uint32Reg0, uint32Reg0, uint32Reg1, addComReg);

            UnPack<int16_t, int8_t, AscendC::Reg::HighLowPart::HIGHEST>(uint16Reg1, selectReg);
            UnPack<int32_t, int16_t, AscendC::Reg::HighLowPart::LOWEST>(uint32Reg2, uint16Reg1);
            UnPack<int32_t, int16_t, AscendC::Reg::HighLowPart::HIGHEST>(uint32Reg3, uint16Reg1);
            Add(uint32Reg2, uint32Reg2, uint32Reg3, addComReg);

            Add(addReg, addReg, uint32Reg0, addComReg);
            Add(addReg, addReg, uint32Reg2, addComReg);
        }
        ReduceSum(dstReg, addReg, addComReg);
        DataCopyUnAlign(dstUbPtr, dstReg, u0, 1);
        AscendC::Reg::DataCopyUnAlignPost(dstUbPtr, u0, 0);
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::GetNonZeroNumB16(int32_t processNum, __local_mem__ T1* xUbPtr,
                                                                __local_mem__ int32_t* dstUbPtr)
{
    uint16_t repeatTimes = CeilDivision(processNum, repeatElmB16_);
    uint32_t addMask = repeatElmB32_;
    int32_t oneMask = 1;
    __VEC_SCOPE__
    {
        uint32_t sreg = processNum;
        AscendC::Reg::MaskReg preg, cmpReg, addComReg;
        AscendC::Reg::RegTensor<T1> xSrcReg;
        AscendC::Reg::RegTensor<int16_t> src1Reg, src0Reg, selectReg;
        AscendC::Reg::RegTensor<int32_t> dstReg, addReg, uint32Reg0, uint32Reg1;
        AscendC::Reg::UnalignReg u0;
        Duplicate(src1Reg, (uint16_t)1);
        Duplicate(src0Reg, (uint16_t)0);
        Duplicate(addReg, (int32_t)0);
        addComReg = AscendC::Reg::UpdateMask<int32_t>(addMask);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<T1>(sreg);
            DataCopy(xSrcReg, xUbPtr + i * repeatElmB16_);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            Select(selectReg, src1Reg, src0Reg, cmpReg);

            UnPack<int32_t, int16_t, AscendC::Reg::HighLowPart::LOWEST>(uint32Reg0, selectReg);
            Add(addReg, addReg, uint32Reg0, addComReg);
            UnPack<int32_t, int16_t, AscendC::Reg::HighLowPart::HIGHEST>(uint32Reg1, selectReg);
            Add(addReg, addReg, uint32Reg1, addComReg);
        }
        ReduceSum(dstReg, addReg, addComReg);
        DataCopyUnAlign(dstUbPtr, dstReg, u0, 1);
        AscendC::Reg::DataCopyUnAlignPost(dstUbPtr, u0, 0);
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::GetNonZeroNumB32(int32_t processNum, __local_mem__ T1* xUbPtr,
                                                                __local_mem__ int32_t* dstUbPtr)
{
    uint32_t addMask = repeatElmB32_;
    uint16_t repeatTimes = CeilDivision(processNum, repeatElmB32_);
    __VEC_SCOPE__
    {
        uint32_t sreg = processNum;
        AscendC::Reg::MaskReg preg, cmpReg, addComReg;
        AscendC::Reg::RegTensor<T1> xSrcReg;
        AscendC::Reg::RegTensor<int32_t> src0Reg, src1Reg, selectReg, dstReg, addReg;
        AscendC::Reg::UnalignReg u0;
        Duplicate(src1Reg, (int32_t)1);
        Duplicate(src0Reg, (int32_t)0);
        Duplicate(addReg, (int32_t)0);
        addComReg = AscendC::Reg::UpdateMask<int32_t>(addMask);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<T1>(sreg);
            DataCopy(xSrcReg, xUbPtr + i * repeatElmB32_);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            Select(selectReg, src1Reg, src0Reg, cmpReg);
            Add(addReg, addReg, selectReg, addComReg);
        }
        ReduceSum(dstReg, addReg, addComReg);
        DataCopyUnAlign(dstUbPtr, dstReg, u0, 1);
        AscendC::Reg::DataCopyUnAlignPost(dstUbPtr, u0, 0);
    }
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMask<T1, T2>::GetNonZeroNumB64(int32_t processNum, __local_mem__ T1* xUbPtr,
                                                                __local_mem__ int64_t* dstUbPtr)
{
    uint32_t addMask = repeatElmB64_;
    uint16_t repeatTimes = CeilDivision(processNum, repeatElmB64_);
    __VEC_SCOPE__
    {
        uint32_t sreg = processNum;
        AscendC::Reg::MaskReg preg, cmpReg, addComReg;
        AscendC::Reg::RegTensor<T1> xSrcReg;
        AscendC::Reg::RegTensor<int64_t> src0Reg, src1Reg, selectReg, dstReg, addReg;
        AscendC::Reg::UnalignReg u0;
        Duplicate(src1Reg, (int64_t)1);
        Duplicate(src0Reg, (int64_t)0);
        Duplicate(addReg, (int64_t)0);
        addComReg = AscendC::Reg::UpdateMask<int64_t>(addMask);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::Reg::UpdateMask<T1>(sreg);
            DataCopy(xSrcReg, xUbPtr + i * repeatElmB64_);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            Select(selectReg, src1Reg, src0Reg, cmpReg);
            Add(addReg, addReg, selectReg, addComReg);
        }
        ReduceSum(dstReg, addReg, addComReg);
        DataCopyUnAlign(dstUbPtr, dstReg, u0, 1);
        AscendC::Reg::DataCopyUnAlignPost(dstUbPtr, u0, 0);
    }
}

} // namespace NonZero

#endif // CANN_NON_ZERO_BIG_MASK_H
