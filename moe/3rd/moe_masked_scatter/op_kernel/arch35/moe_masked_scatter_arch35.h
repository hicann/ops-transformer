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
 * \file moe_masked_scatter_arch35.h
 * \brief
 */

#ifndef MASKED_SCATTER_ARCH35_H
#define MASKED_SCATTER_ARCH35_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

#include "simt_api/asc_simt.h"
namespace MoeMaskedScatter {
using namespace AscendC;

constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint32_t USED_THREAD_NUMS = 1024;
constexpr int64_t CUSTOM_OFFSET = 32;

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUMS) inline void MoeMaskedScatterSimt(
    int64_t curCoreHandleData, U preCoreMaskSum, __gm__ T *xGm, __gm__ bool *maskGm, __gm__ T *updatesGm, __gm__ T *yGm,
    __gm__ U *prefixSumGm)
{
    for (int64_t i = threadIdx.x; i < curCoreHandleData; i += blockDim.x) {
        if (maskGm[i] == 0) {
            yGm[i] = xGm[i];
        } else {
            yGm[i] = updatesGm[prefixSumGm[i] + preCoreMaskSum];
        }
    }
}

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUMS) void MoeMaskedScatterToUBSimt(
    int64_t curLoopHandleData, U preCoreMaskSum, __local_mem__ T *x, __local_mem__ bool *mask, __gm__ T *updatesGm,
    __local_mem__ T *y, __gm__ U *prefixSumGm)
{
    for (int64_t i = threadIdx.x; i < curLoopHandleData; i += blockDim.x) {
        if (mask[i] == 0) {
            y[i] = x[i];
        } else {
            y[i] = updatesGm[prefixSumGm[i] + preCoreMaskSum];
        }
    }
}

template <typename T, typename U>
class MoeMaskedScatterImpl {
public:
    __aicore__ inline MoeMaskedScatterImpl(const MoeMaskedScatterTilingData &tilingData, TPipe &pipe)
        : tilingData_(tilingData),
          pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyInMask(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyInX(int64_t offset, int64_t dataLen);
    __aicore__ inline void ComputePrefixSum(int64_t dataLen);
    __aicore__ inline void ComputeSingleLoopPrefixSum(uint32_t dataLen, __ubuf__ int32_t *prefixSumAddr,
                                                      __ubuf__ int32_t *tmpAddr);
    template <bool isCopyOutMaskSum = false, bool notNeedCopyPrefixSum = false>
    __aicore__ inline void CopyOutMask(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOutY(int64_t offset, int64_t dataLen);
    __aicore__ inline void ComputePreCoreMaskSum();
    __aicore__ inline void CustomReduceSum(const LocalTensor<U> &dst, const LocalTensor<U> &src, uint16_t dataLen);
    __aicore__ inline void MoeMaskedScatterToUB(int64_t offset, int64_t dataLen);
    __aicore__ inline void ProcessLowBit(int64_t curCoreHandleData);
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<bool> maskGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<U> prefixSumGm_;
    AscendC::GlobalTensor<U> prefixSumGmTmp_;
    AscendC::GlobalTensor<U> maskSumGm_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> maskQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TQue<QuePosition::VECOUT, 1> prefixSumQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TPipe &pipe_;
    const MoeMaskedScatterTilingData &tilingData_;
    U curMaskSum_{0};
    U preCoreMaskSum_{0};
};

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::Init(GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y,
                                                        GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ T *)(x) + GetBlockIdx() * tilingData_.normalCoreData);
    updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
    yGm_.SetGlobalBuffer((__gm__ T *)(y) + GetBlockIdx() * tilingData_.normalCoreData);
    maskGm_.SetGlobalBuffer((__gm__ bool *)(mask) + GetBlockIdx() * tilingData_.normalCoreData);
    maskSumGm_.SetGlobalBuffer((__gm__ U *)(workspace), GetBlockNum());
    prefixSumGm_.SetGlobalBuffer((__gm__ U *)(workspace) + GetBlockNum() + GetBlockIdx() * tilingData_.normalCoreData);
    prefixSumGmTmp_ = prefixSumGm_[1];
    pipe_.InitBuffer(maskQueue_, DOUBLE_BUFFER, tilingData_.ubFactor);
    pipe_.InitBuffer(tmpBuf_, tilingData_.ubFactor * sizeof(U));
    pipe_.InitBuffer(prefixSumQueue_, 1, tilingData_.ubFactor * sizeof(U));
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::CopyInMask(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = {1, static_cast<uint32_t>(dataLen), 0, 0, 0};
    DataCopyPadExtParams<bool> padParams = {false, 0, 0, false};
    LocalTensor<bool> maskLocal = maskQueue_.AllocTensor<bool>();
    DataCopyPad(maskLocal, maskGm_[offset], inParams, padParams);
    maskQueue_.EnQue(maskLocal);
}

template <typename T, typename U>
template <bool isCopyOutMaskSum, bool notNeedCopyPrefixSum>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::CopyOutMask(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams = {1, static_cast<uint32_t>(dataLen * sizeof(U)), 0, 0, 0};
    LocalTensor<U> prefixSumLocal = prefixSumQueue_.DeQue<U>();
    if constexpr (!notNeedCopyPrefixSum) {
        DataCopyPad(prefixSumGmTmp_[offset], prefixSumLocal, outParams);
    }
    if constexpr (isCopyOutMaskSum) {
        outParams = {1, static_cast<uint32_t>(1 * sizeof(U)), 0, 0, 0};
        LocalTensor<U> maskSumLocal = tmpBuf_.Get<U>();
        maskSumLocal.SetValue(0, curMaskSum_);
        maskSumLocal.SetValue(CUSTOM_OFFSET, 0);
        auto mte3WiatSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(mte3WiatSEventID);
        WaitFlag<HardEvent::S_MTE3>(mte3WiatSEventID);
        DataCopyPad(maskSumGm_[GetBlockIdx()], maskSumLocal, outParams);
        DataCopyPad(prefixSumGm_, maskSumLocal[CUSTOM_OFFSET], outParams);
    }
    prefixSumQueue_.FreeTensor(prefixSumLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::ComputePrefixSum(int64_t dataLen)
{
    LocalTensor<U> prefixSumLocal = prefixSumQueue_.AllocTensor<U>();
    LocalTensor<int32_t> tmpLocal = tmpBuf_.Get<int32_t>();
    auto prefixSumAddr = (__ubuf__ int32_t *)prefixSumLocal.GetPhyAddr();
    auto tmpAddr = (__ubuf__ int32_t *)tmpLocal.GetPhyAddr();
    ComputeSingleLoopPrefixSum(dataLen, tmpAddr, prefixSumAddr);
    if constexpr (sizeof(U) == sizeof(int32_t)) {
        Adds(prefixSumLocal, tmpLocal, curMaskSum_, dataLen);
    } else {
        Cast(prefixSumLocal, tmpLocal, AscendC::RoundMode::CAST_NONE, dataLen);
        Adds(prefixSumLocal, prefixSumLocal, curMaskSum_, dataLen);
    }
    auto sWiatVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(sWiatVEventID);
    WaitFlag<HardEvent::V_S>(sWiatVEventID);
    curMaskSum_ = prefixSumLocal.GetValue(dataLen - 1);
    prefixSumQueue_.EnQue(prefixSumLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::ComputeSingleLoopPrefixSum(uint32_t dataLen,
                                                                              __ubuf__ int32_t *prefixSumAddr,
                                                                              __ubuf__ int32_t *tmpAddr)
{
    LocalTensor<bool> maskLocal = maskQueue_.DeQue<bool>();
    auto maskAddr = (__ubuf__ uint8_t *)maskLocal.GetPhyAddr();
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint32_t rows = vfLen;
    uint32_t cols = (dataLen + vfLen - 1) / vfLen;

    uint16_t size0 = cols;
    uint16_t size1 = cols / vfLen;
    uint16_t tailSize1 = cols - size1 * vfLen;
    if (tailSize1 == 0) {
        size1--;
        tailSize1 = vfLen;
    }
    __VEC_SCOPE__
    {
        uint32_t main = rows;
        AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<int32_t>(main); // vfLen的mask
        AscendC::MicroAPI::RegTensor<uint8_t> mask;
        auto prefixSumTmpAddr = prefixSumAddr;
        for (uint16_t i = 0; i < size0; ++i) {
            AscendC::MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        MicroAPI::LoadDist::DIST_UNPACK4_B8>(mask, maskAddr, vfLen);
            AscendC::MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                prefixSumTmpAddr, (AscendC::MicroAPI::RegTensor<int32_t> &)mask, vfLen, p0);
        }
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        uint32_t stride = rows;
        AscendC::MicroAPI::RegTensor<int32_t> tmp;
        AscendC::MicroAPI::RegTensor<uint32_t> sequence;
        AscendC::MicroAPI::RegTensor<uint32_t> index;
        AscendC::MicroAPI::Arange(tmp, 0);
        sequence = (AscendC::MicroAPI::RegTensor<uint32_t> &)tmp; // 以0开始的序列
        AscendC::MicroAPI::RegTensor<int32_t> v0;
        AscendC::MicroAPI::RegTensor<int32_t> v1;
        AscendC::MicroAPI::RegTensor<int32_t> v2;
        AscendC::MicroAPI::UnalignReg u1;

        AscendC::MicroAPI::Duplicate(v1, 0, p0);               // 构造全为0的寄存器v1
        AscendC::MicroAPI::Muls(sequence, sequence, cols, p0); // 构造以0开始，以cols为步长的序列
        prefixSumTmpAddr = prefixSumAddr;
        auto tempAddr = tmpAddr;
        for (uint16_t i = 0; i < size0; ++i) {
            AscendC::MicroAPI::Adds(index, sequence, (uint32_t)i, p0); // 构造以i开始，以cols为步长的序列
            AscendC::MicroAPI::DataCopyGather(v0, prefixSumTmpAddr, index, p0);
            AscendC::MicroAPI::Add(v2, v0, v1, p0);
            AscendC::MicroAPI::Copy(v1, v2, p0);
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<int32_t>(i, stride);
            AscendC::MicroAPI::DataCopy(tempAddr, v2, offset, p0);
        }
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

        AscendC::MicroAPI::Arange(tmp, 0);
        sequence = (AscendC::MicroAPI::RegTensor<uint32_t> &)tmp;
        AscendC::MicroAPI::MaskReg pregFull =
            AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Muls(sequence, sequence, rows, pregFull);
        uint32_t spReg1 = tailSize1 - 1;
        AscendC::MicroAPI::MaskReg sp1 = AscendC::MicroAPI::UpdateMask<int32_t>(spReg1);
        AscendC::MicroAPI::RegTensor<int32_t> v3;
        AscendC::MicroAPI::RegTensor<int32_t> v4;
        AscendC::MicroAPI::RegTensor<uint32_t> vdSque;
        AscendC::MicroAPI::Duplicate(v1, 0, pregFull);
        for (uint16_t i = 0; i < static_cast<uint16_t>(rows); i++) {
            uint32_t mainCols = cols;
            AscendC::MicroAPI::Adds(vdSque, sequence, (uint32_t)i, pregFull);
            for (uint16_t j = 0; j < size1; j++) {
                AscendC::MicroAPI::MaskReg p1 = AscendC::MicroAPI::UpdateMask<int32_t>(mainCols);
                AscendC::MicroAPI::Adds(index, vdSque, (uint32_t)(j * vfLen * vfLen), p1);
                AscendC::MicroAPI::DataCopyGather(v0, tmpAddr, index, p1);
                AscendC::MicroAPI::Add(v2, v0, v1, p1);
                AscendC::MicroAPI::DataCopyUnAlign(prefixSumAddr, v2, u1, rows);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(prefixSumAddr, u1, 0);
            AscendC::MicroAPI::MaskReg p1 = AscendC::MicroAPI::UpdateMask<int32_t>(mainCols);
            AscendC::MicroAPI::Adds(index, vdSque, (uint32_t)(size1 * vfLen * vfLen), p1);
            AscendC::MicroAPI::DataCopyGather(v0, tmpAddr, index, p1);
            AscendC::MicroAPI::Add(v2, v0, v1, p1);
            AscendC::MicroAPI::DataCopyUnAlign(prefixSumAddr, v2, u1, tailSize1);
            AscendC::MicroAPI::Duplicate(v4, 0, sp1);
            AscendC::MicroAPI::Copy<int32_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(v2, v4, sp1);
            AscendC::MicroAPI::ReduceSum(v3, v2, p1);
            AscendC::MicroAPI::Duplicate(v1, v3, pregFull);
            AscendC::MicroAPI::DataCopyUnAlignPost(prefixSumAddr, u1, 0);
        }
    }
    maskQueue_.FreeTensor(maskLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::ComputePreCoreMaskSum()
{
    LocalTensor<U> tmpLocal = tmpBuf_.Get<U>();
    DataCopyExtParams inParams = {1, static_cast<uint32_t>(GetBlockNum() * sizeof(U)), 0, 0, 0};
    DataCopyPadExtParams<U> padParams = {false, 0, 0, 0};
    DataCopyPad(tmpLocal, maskSumGm_, inParams, padParams);
    if (GetBlockIdx() == 1) {
        auto sWiatVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(sWiatVEventID);
        WaitFlag<HardEvent::MTE2_S>(sWiatVEventID);
        preCoreMaskSum_ = tmpLocal.GetValue(0);
    } else if (GetBlockIdx() != 0) {
        auto vWiatMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(vWiatMTEEventID);
        WaitFlag<HardEvent::MTE2_V>(vWiatMTEEventID);
        CustomReduceSum(tmpLocal, tmpLocal, GetBlockIdx());
        auto sWiatVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(sWiatVEventID);
        WaitFlag<HardEvent::V_S>(sWiatVEventID);
        preCoreMaskSum_ = tmpLocal.GetValue(0);
    }
    assert(preCoreMaskSum_ + curMaskSum_ <= tilingData_.updateEleNums,
           "The num of true in mask is larger than the num of elements in update.");
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::CustomReduceSum(const LocalTensor<U> &dst, const LocalTensor<U> &src,
                                                                   uint16_t dataLen)
{
    uint16_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t loopSize = (dataLen + vfLen - 1) / vfLen;
    auto srcAddr = (__ubuf__ U *)src.GetPhyAddr();
    auto dstAddr = (__ubuf__ U *)dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> src;
        AscendC::MicroAPI::RegTensor<U> dst;
        AscendC::MicroAPI::RegTensor<U> tmpSum;
        uint32_t pnum = static_cast<uint32_t>(dataLen);
        uint32_t sumMask = 1;
        AscendC::MicroAPI::MaskReg oneMask = AscendC::MicroAPI::UpdateMask<U>(sumMask);
        AscendC::MicroAPI::Duplicate(dst, 0, oneMask);
        for (uint16_t i = 0; i < loopSize; i++) {
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<U>(pnum);
            AscendC::MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(src, srcAddr, vfLen);
            AscendC::MicroAPI::ReduceSum(tmpSum, src, pMask);
            AscendC::MicroAPI::Add(dst, dst, tmpSum, oneMask);
        }
        AscendC::MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, dst, 0, oneMask);
    }
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::CopyInX(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = {1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    DataCopyPad(xLocal, xGm_[offset], inParams, padParams);
    xQueue_.EnQue(xLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::CopyOutY(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams = {1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yQueue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yLocal, outParams);
    yQueue_.FreeTensor(yLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::MoeMaskedScatterToUB(int64_t offset, int64_t dataLen)
{
    LocalTensor<T> xLocal = xQueue_.DeQue<T>();
    LocalTensor<bool> maskLocal = maskQueue_.DeQue<bool>();
    LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
    auto curLoopPrefixSumGm = prefixSumGm_[offset];
    asc_vf_call<MoeMaskedScatterToUBSimt<T, U>>(
        dim3(USED_THREAD_NUMS), dataLen, preCoreMaskSum_, (__local_mem__ T *)(xLocal.GetPhyAddr()),
        (__local_mem__ bool *)(maskLocal.GetPhyAddr()), (__gm__ T *)(updatesGm_.GetPhyAddr()),
        (__local_mem__ T *)(yLocal.GetPhyAddr()), (__gm__ U *)(curLoopPrefixSumGm.GetPhyAddr()));
    yQueue_.EnQue(yLocal);
    xQueue_.FreeTensor(xLocal);
    maskQueue_.FreeTensor(maskLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::ProcessLowBit(int64_t curCoreHandleData)
{
    pipe_.Reset();
    pipe_.InitBuffer(maskQueue_, DOUBLE_BUFFER, tilingData_.bufferSize);
    pipe_.InitBuffer(xQueue_, DOUBLE_BUFFER, tilingData_.bufferSize * sizeof(T));
    pipe_.InitBuffer(tmpBuf_, tilingData_.prefixSumUbSize);
    pipe_.InitBuffer(yQueue_, 1, tilingData_.bufferSize * sizeof(T));
    ComputePreCoreMaskSum();
    int64_t normalLen = tilingData_.bufferSize;
    int64_t loopNum = Ops::Base::CeilDiv(curCoreHandleData, normalLen);
    int64_t tailLen = curCoreHandleData % normalLen;
    if (tailLen == 0) {
        tailLen = normalLen;
    }
    int64_t offset = 0;
    for (int64_t i = 0; i < loopNum - 1; ++i) {
        offset = i * normalLen;
        CopyInMask(offset, normalLen);
        CopyInX(offset, normalLen);
        MoeMaskedScatterToUB(offset, normalLen);
        CopyOutY(offset, normalLen);
    }
    offset = (loopNum - 1) * normalLen;
    CopyInMask(offset, tailLen);
    CopyInX(offset, tailLen);
    MoeMaskedScatterToUB(offset, tailLen);
    CopyOutY(offset, tailLen);
}

template <typename T, typename U>
__aicore__ inline void MoeMaskedScatterImpl<T, U>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    int64_t loopSize = tilingData_.blockFactor;
    int64_t tailUbFactor = tilingData_.tailUbFactor;
    int64_t curCoreHandleData = tilingData_.normalCoreData;
    if (GetBlockIdx() == GetBlockNum() - 1) {
        loopSize = tilingData_.tailBlockFactor;
        tailUbFactor = tilingData_.tailCoreTailUbFactor;
        curCoreHandleData = tilingData_.tailCoreData;
    }
    int64_t offset = 0;
    int64_t dataLen = tilingData_.ubFactor;
    for (int64_t i = 0; i < loopSize - 1; ++i) {
        offset = i * tilingData_.ubFactor;
        CopyInMask(offset, dataLen);
        ComputePrefixSum(dataLen);
        CopyOutMask(offset, dataLen);
    }
    offset = (loopSize - 1) * tilingData_.ubFactor;
    dataLen = tailUbFactor;
    CopyInMask(offset, dataLen);
    ComputePrefixSum(dataLen);
    if (dataLen == 1) {
        CopyOutMask<true, true>(offset, dataLen - 1);
    } else {
        CopyOutMask<true, false>(offset, dataLen - 1);
    }
    SyncAll();
    if constexpr (sizeof(T) == sizeof(bool) || sizeof(T) == sizeof(int16_t)) {
        ProcessLowBit(curCoreHandleData);
    } else {
        ComputePreCoreMaskSum();
        asc_vf_call<MoeMaskedScatterSimt<T, U>>(dim3(USED_THREAD_NUMS), curCoreHandleData, preCoreMaskSum_,
                                                (__gm__ T *)(xGm_.GetPhyAddr()), (__gm__ bool *)(maskGm_.GetPhyAddr()),
                                                (__gm__ T *)(updatesGm_.GetPhyAddr()), (__gm__ T *)(yGm_.GetPhyAddr()),
                                                (__gm__ U *)(prefixSumGm_.GetPhyAddr()));
    }
}
} // namespace MoeMaskedScatter

#endif // MASKED_SCATTER_ARCH35_H
