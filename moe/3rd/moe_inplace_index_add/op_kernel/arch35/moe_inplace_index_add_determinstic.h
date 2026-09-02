/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_inplace_index_add_determinstic.h
 * \brief moe_inplace_index_add_determinstic
 */
#ifndef ASCENDC_MOE_INPLACE_INDEX_ADD_DETERMINSTIC_H_
#define ASCENDC_MOE_INPLACE_INDEX_ADD_DETERMINSTIC_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "moe_inplace_index_add_common.h"

namespace MoeInplaceIndexAdd {
using namespace AscendC;

static constexpr MicroAPI::CastTrait castTraitFp32ToInt32 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr MicroAPI::CastTrait castTraitInt32ToFp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr MicroAPI::CastTrait castTraitFp32ToVarT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename VAR_T, typename IDX_T>
class MoeInplaceIndexAddDeterminstic {
public:
    __aicore__ inline MoeInplaceIndexAddDeterminstic(const MoeInplaceIndexAddDeterminsticTilingData &tilingData,
                                                     TPipe &pipe)
        : tilingData_(tilingData),
          pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha, GM_ADDR workspace);
    __aicore__ inline void CopyIndicesAndUpdatesIn(int64_t preAxisIdx, int64_t rowIdx, int64_t colIdx, int64_t rowLen,
                                                   int64_t colLen);
    __aicore__ inline void ComputeUniqueIdNum(int64_t dataLen);
    __aicore__ inline void ComputeUinqueIdTimes(uint32_t uniqueIdNum);
    __aicore__ inline void ComputeSum(uint32_t uniqueIdNum, int64_t colLen);
    __aicore__ inline void ComputeSumForSameIndex(int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopySumOutToWs(int64_t preAxisIdx, int64_t rowIdx, int64_t colIdx, int64_t rowLen,
                                          int64_t colLen);
    __aicore__ inline void CopySumAndRValueIn(int64_t preAxisIdx, int64_t rowIdx, int64_t colIdx, int64_t rowLen,
                                              int64_t colLen);
    __aicore__ inline void QuantizeForSum(int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyQuantizedSumOutToIntWs(int64_t preAxisIdx, int64_t rowIdx, int64_t colIdx,
                                                      int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyRValueAndIntWsIn(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyRValueAndIntWsAndIndexInOpti(int64_t preAxisIdx, int64_t rowIdx, int64_t colIdx,
                                                            int64_t rowLen, int64_t colLen);
    __aicore__ inline void InverseQuantize(int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyInverseQuantizedValueOutOpti(int64_t preAxisIdx, int64_t colLen);
    __aicore__ inline void CopyInverseQuantizedValueOut(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessFirstStep(int64_t preAxisIdx);
    __aicore__ inline void ProcessSecondStep(int64_t preAxisIdx);
    __aicore__ inline void ProcessThirdStep();
    __aicore__ inline void ProcessThirdStepOpti(int64_t preAxisIdx);
    __aicore__ inline void Process();

private:
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;
    GlobalTensor<VAR_T> alpha_;

    GlobalTensor<float> workspaceInitGm_;
    GlobalTensor<float> updateSumWsGm_;
    GlobalTensor<IDX_T> updateSumIdxWsGm_;
    GlobalTensor<float> updateRValueWsGm_;
    GlobalTensor<uint32_t> sameIdxCountWsGm_;
    GlobalTensor<int32_t> sumQuanToIntWsGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesQue_;
    TBuf<QuePosition::VECCALC> sortIndicesQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesOriginIdexQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> updatesCastQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> updateSumQue_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> uniqueIdCountQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> updateSumIdxQue_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sumQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sumIdxQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> rValueQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> quantaSumQue_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sumQuanToIntQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> invQuanDataQue_;

    TPipe &pipe_;
    const MoeInplaceIndexAddDeterminsticTilingData &tilingData_;

    VAR_T alphaValue_{0};
    int64_t varNumel_{0};
    int64_t sumWsSize_{0};
    int64_t shiftOffset_{0};
    int64_t rValueWsSize_{0};

    int64_t curPreAxisCount_{0};
    int64_t curCoreIndexCount_{0};
    int64_t curCoreVarCount_{0};
    uint32_t uniqueIdNum_{0};
    uint64_t afterAxisAlignSize_{0};
    uint64_t afterAxisAlignFp32_{0};
    uint64_t uniqueSumIdNum_{0};
};

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                                          GM_ADDR alpha, GM_ADDR workspace)
{
    var_.SetGlobalBuffer((__gm__ VAR_T *)(var));
    indices_.SetGlobalBuffer((__gm__ IDX_T *)(indices));
    updates_.SetGlobalBuffer((__gm__ VAR_T *)(updates));
    alpha_.SetGlobalBuffer((__gm__ VAR_T *)(alpha));
    if (tilingData_.isWithAlpha) {
        alphaValue_ = alpha_(0);
    }

    afterAxisAlignSize_ = Ops::Base::CeilAlign(tilingData_.afterAxis * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    afterAxisAlignFp32_ = Ops::Base::CeilAlign(tilingData_.afterAxis * sizeof(float), UB_AGLIN_VALUE) / sizeof(float);

    varNumel_ = tilingData_.preAxis * tilingData_.varInAxis * tilingData_.afterAxis;
    curCoreIndexCount_ = (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCoreIndexCount :
                                                                                 tilingData_.tailCoreIndexCount);
    curCoreVarCount_ = (GetBlockIdx() != (tilingData_.usedCoreNumAfter - 1) ? tilingData_.eachCoreVarCount :
                                                                              tilingData_.tailCoreVarCount);
    sumWsSize_ = tilingData_.eachCoreIndexCount * tilingData_.afterAxis;
    /* one more col to store counter */
    int64_t varRowCount = tilingData_.preAxis * tilingData_.varInAxis;
    rValueWsSize_ = varNumel_ + varRowCount;

    /* init workspace for multi cores */
    int64_t wsInitLen = rValueWsSize_ + varNumel_;
    int64_t mainCoreLen = wsInitLen / GetBlockNum();
    int64_t tailCoreLen = wsInitLen - (GetBlockNum() - 1) * mainCoreLen;
    int64_t curCoreLen = (GetBlockIdx() == (GetBlockNum() - 1)) ? tailCoreLen : mainCoreLen;
    int64_t wsInitOfset = GetBlockIdx() * mainCoreLen;
    workspaceInitGm_.SetGlobalBuffer((__gm__ float *)workspace + wsInitOfset, curCoreLen);
    InitGlobalMemory(workspaceInitGm_, curCoreLen, (float)0);
    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);

    sameIdxCountWsGm_.SetGlobalBuffer((__gm__ uint32_t *)workspace, varRowCount);
    updateRValueWsGm_.SetGlobalBuffer((__gm__ float *)workspace + varRowCount, varNumel_);
    sumQuanToIntWsGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + rValueWsSize_, varNumel_);
    updateSumWsGm_.SetGlobalBuffer((__gm__ float *)workspace + rValueWsSize_ + varNumel_ + GetBlockIdx() * sumWsSize_,
                                   sumWsSize_);
    auto updateSumStartAddr = (__gm__ float *)workspace + rValueWsSize_ + varNumel_ + GetBlockNum() * sumWsSize_;
    updateSumIdxWsGm_.SetGlobalBuffer(
        (__gm__ IDX_T *)updateSumStartAddr + GetBlockIdx() * tilingData_.eachCoreIndexCount,
        tilingData_.eachCoreIndexCount);

    InitGlobalMemory(updateSumWsGm_, sumWsSize_, (float)(0));
    auto vWaitMte3EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(vWaitMte3EventID);
    WaitFlag<HardEvent::MTE3_V>(vWaitMte3EventID);
    InitGlobalMemory(updateSumIdxWsGm_, tilingData_.eachCoreIndexCount, (IDX_T)(-1));
    AscendC::SyncAll();
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::CopyIndicesAndUpdatesIn(
    int64_t preAxisIdx, int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<float> updatesCastLocal = updatesCastQue_.AllocTensor<float>();
    int64_t preOfst = preAxisIdx * tilingData_.updatesInAxis;
    int64_t indicesOfst = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.ubIndexFactor;
    int64_t offset = (preOfst + indicesOfst) * tilingData_.afterAxis;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};

    if constexpr (!IsSameType<VAR_T, float>::value) {
        LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
        DataCopyPad(updatesLocal, updates_[offset], copyParams, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(updatesCastLocal, updatesLocal, RoundMode::CAST_NONE, rowLen * afterAxisAlignSize_);
        updatesQue_.FreeTensor(updatesLocal);
    } else {
        DataCopyPad(updatesCastLocal, updates_[offset], copyParams, padParams);
    }
    updatesCastQue_.EnQue(updatesCastLocal);

    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    offset = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.ubIndexFactor;
    CopyIn<IDX_T>(indicesLocal, indices_[offset], rowLen);
    indicesQue_.EnQue(indicesLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::ComputeUniqueIdNum(int64_t dataLen)
{
    LocalTensor<IDX_T> indicesLocal = sortIndicesQue_.Get<IDX_T>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();

    __local_mem__ IDX_T *indicesAddr = (__local_mem__ IDX_T *)indicesLocal[shiftOffset_].GetPhyAddr();
    __local_mem__ int32_t *uniqueIdCountsAddr = (__local_mem__ int32_t *)uniqueIdCountLocal.GetPhyAddr();

    int64_t vfLen = Ops::Base::GetVRegSize() / sizeof(IDX_T);
    uint16_t loopCnt = Ops::Base::CeilDiv(dataLen + 1, vfLen);
    uint32_t counter = dataLen + 1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> orderReg;
        AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg;
        AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxShiftOneReg;
        AscendC::MicroAPI::RegTensor<int32_t> selReg;
        AscendC::MicroAPI::MaskReg cmpMask;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * vfLen);
            maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
            auto startAddr = indicesAddr + i * vfLen;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            if constexpr (std::is_same<int64_t, IDX_T>::value) {
                AscendC::MicroAPI::MaskReg maskHalf;
                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg,
                                                                                                     maskHalf);
            } else {
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg,
                                                                                                     cmpMask);
            }
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }
    uniqueIdNum_ = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;

    LocalTensor<IDX_T> updateSumIdxLocal = updateSumIdxQue_.AllocTensor<IDX_T>();
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    for (uint32_t idx = 0; idx < uniqueIdNum_; idx++) {
        auto offset = shiftOffset_ + uniqueIdCountLocal(idx);
        updateSumIdxLocal(idx) = indicesLocal(offset);
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    updateSumIdxQue_.EnQue(updateSumIdxLocal);
    uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::ComputeUinqueIdTimes(uint32_t uniqueIdNum)
{
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.DeQue<int32_t>();
    __local_mem__ int32_t *uniqueIdCountsAddr = (__local_mem__ int32_t *)uniqueIdCountLocal.GetPhyAddr();

    // compute repeated num of each id
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint16_t loopSize = Ops::Base::CeilDiv(uniqueIdNum, vfLen);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> preReg;
        AscendC::MicroAPI::RegTensor<int32_t> postReg;
        AscendC::MicroAPI::RegTensor<int32_t> subReg;
        AscendC::MicroAPI::UnalignReg uIn;
        AscendC::MicroAPI::MaskReg maskReg;
        for (uint16_t i = 0; i < loopSize; ++i) {
            maskReg = AscendC::MicroAPI::UpdateMask<int32_t>(uniqueIdNum);
            auto startAddr = uniqueIdCountsAddr + i * vfLen;
            auto startAddrOfstOne = startAddr + 1;
            DataCopy(preReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(uIn, startAddrOfstOne);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t>(postReg, uIn, startAddrOfstOne, vfLen);
            AscendC::MicroAPI::Sub(subReg, postReg, preReg, maskReg);
            DataCopy(startAddr, subReg, maskReg);
        }
    }
    uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::ComputeSum(uint32_t uniqueIdNum, int64_t colLen)
{
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.DeQue<int32_t>();
    LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.DeQue<uint32_t>();
    LocalTensor<float> updatesCastLocal = updatesCastQue_.DeQue<float>();
    LocalTensor<float> updateSumLocal = updateSumQue_.AllocTensor<float>();

    __local_mem__ float *updatesAddr = (__local_mem__ float *)updatesCastLocal.GetPhyAddr();
    __local_mem__ float *updateSumAddr = (__local_mem__ float *)updateSumLocal.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(float);
    int32_t loopSize = (colLen + vfLen - 1) / vfLen;
    int32_t idLocation = 0;
    __VEC_SCOPE__
    {
        for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); i++) {
            AscendC::MicroAPI::RegTensor<float> sumReg;
            AscendC::MicroAPI::RegTensor<float> updateReg;
            AscendC::MicroAPI::UnalignReg uIn;
            AscendC::MicroAPI::MaskReg maskReg;
            AscendC::MicroAPI::MaskReg zeroMask = AscendC::MicroAPI::CreateMask<int32_t>();
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountLocal(i));
            for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
                maskReg = AscendC::MicroAPI::UpdateMask<int32_t>(maskLen);
                AscendC::MicroAPI::Duplicate(sumReg, (float)0, zeroMask);
                for (uint16_t k = 0; k < idRepeatTimes; k++) {
                    auto updatesOffet = updatesOriginIdexLocal(idLocation + k) * afterAxisAlignSize_ + j * vfLen;
                    auto startAddr = updatesAddr + updatesOffet;
                    AscendC::MicroAPI::DataCopy(updateReg, startAddr);
                    AscendC::MicroAPI::Add(sumReg, sumReg, updateReg, maskReg);
                }
                auto updateSumAddrOfst = updateSumAddr + i * afterAxisAlignFp32_ + j * vfLen;
                DataCopy(updateSumAddrOfst, sumReg, maskReg);
            }
            idLocation += idRepeatTimes;
        }
    }
    updateSumQue_.EnQue(updateSumLocal);
    updatesCastQue_.FreeTensor(updatesCastLocal);
    uniqueIdCountQue_.FreeTensor(uniqueIdCountLocal);
    updatesOriginIdexQue_.FreeTensor(updatesOriginIdexLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::ComputeSumForSameIndex(int64_t rowLen,
                                                                                            int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
    LocalTensor<IDX_T> sortIndicesLocal = sortIndicesQue_.Get<IDX_T>();

    /* recore updates origin location index */
    LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
    LocalTensor<IDX_T> shiftSortLocal = sortIndicesLocal[shiftOffset_];
    AscendC::Sort<IDX_T, true, sortConfig>(shiftSortLocal, updatesOriginIdexLocal, indicesLocal,
                                           static_cast<uint32_t>(rowLen));
    Duplicate(sortIndicesLocal, (IDX_T)-1, shiftOffset_);
    shiftSortLocal(rowLen) = -1;
    PipeBarrier<PIPE_V>();
    indicesQue_.FreeTensor(indicesLocal);
    updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);

    ComputeUniqueIdNum(rowLen);
    ComputeUinqueIdTimes(uniqueIdNum_);
    ComputeSum(uniqueIdNum_, colLen);
}

template <typename VAR_T, typename IDX_T>
__aicore__ void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::CopySumOutToWs(int64_t preAxisIdx, int64_t rowIdx,
                                                                             int64_t colIdx, int64_t rowLen,
                                                                             int64_t colLen)
{
    int64_t rowOfset = rowIdx * tilingData_.ubIndexFactor;
    LocalTensor<IDX_T> updateSumIdxLocal = updateSumIdxQue_.DeQue<IDX_T>();
    CopyOut<IDX_T>(updateSumIdxWsGm_[rowOfset], updateSumIdxLocal, uniqueIdNum_);

    int64_t outOfset = rowOfset * tilingData_.afterAxis;
    LocalTensor<float> updateSumLocal = updateSumQue_.DeQue<float>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(uniqueIdNum_), static_cast<uint32_t>(colLen * sizeof(float)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(updateSumWsGm_[outOfset], updateSumLocal, copyParams);
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

    Abs(updateSumLocal, updateSumLocal, uniqueIdNum_ * afterAxisAlignFp32_);
    for (int32_t i = 0; i < uniqueIdNum_; i++) {
        int64_t sumIdx = updateSumIdxLocal(i);
        sumIdx += preAxisIdx * tilingData_.varInAxis;
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);

        outOfset = static_cast<int64_t>(sumIdx * tilingData_.afterAxis);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicMax<float>();
        CopyOut<float>(updateRValueWsGm_[outOfset], updateSumLocal[i * afterAxisAlignFp32_], colLen);
        SetAtomicNone();
        AscendC::AtomicAdd(const_cast<__gm__ uint32_t *>(sameIdxCountWsGm_[sumIdx].GetPhyAddr()), uint32_t(1));
    }

    updateSumQue_.FreeTensor(updateSumLocal);
    updateSumIdxQue_.FreeTensor(updateSumIdxLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::CopySumAndRValueIn(int64_t preAxisIdx,
                                                                                        int64_t rowIdx, int64_t colIdx,
                                                                                        int64_t rowLen, int64_t colLen)
{
    LocalTensor<float> sumLocal = sumQue_.AllocTensor<float>();
    LocalTensor<IDX_T> sumIdxLocal = sumIdxQue_.AllocTensor<IDX_T>();
    LocalTensor<float> rValueLocal = rValueQue_.AllocTensor<float>(); /* reuse updatelocal for Rvalue */

    int32_t rowOfset = rowIdx * tilingData_.ubQuantaIndxFactor;
    CopyIn<IDX_T>(sumIdxLocal, updateSumIdxWsGm_[rowOfset], rowLen);
    int32_t inOfset = rowOfset;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(float)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<float>(0)};
    DataCopyPad(sumLocal, updateSumWsGm_[inOfset], copyParams, padParams);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    for (int64_t i = 0; i < rowLen; i++) {
        int64_t sumIdx = preAxisIdx * tilingData_.varInAxis + sumIdxLocal(i);
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(sameIdxCountWsGm_[sumIdx]);
        uint32_t count = sameIdxCountWsGm_(sumIdx);
        int64_t rowOfset = sumIdx * tilingData_.afterAxis;
        int64_t localOfset = i * afterAxisAlignFp32_;

        event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        CopyIn<float>(rValueLocal[localOfset], updateRValueWsGm_[rowOfset], colLen);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::Muls(rValueLocal[localOfset], rValueLocal[localOfset], static_cast<float>(count), colLen);
    }

    sumQue_.EnQue(sumLocal);
    sumIdxQue_.EnQue(sumIdxLocal);
    rValueQue_.EnQue(rValueLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::QuantizeForSum(int64_t rowLen, int64_t colLen)
{
    LocalTensor<float> sumLocal = sumQue_.DeQue<float>();
    LocalTensor<float> rValueLocal = rValueQue_.DeQue<float>();
    LocalTensor<int32_t> quantaSumLocal = quantaSumQue_.AllocTensor<int32_t>();

    __local_mem__ float *sumAddr = (__local_mem__ float *)sumLocal.GetPhyAddr();
    __local_mem__ float *rValueAddr = (__local_mem__ float *)rValueLocal.GetPhyAddr();
    __local_mem__ int32_t *quantaSumAddr = (__local_mem__ int32_t *)quantaSumLocal.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    float scaling = static_cast<float>(1 << 30);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> dataReg;
        AscendC::MicroAPI::RegTensor<float> rValueReg;
        AscendC::MicroAPI::RegTensor<float> resReg;
        AscendC::MicroAPI::RegTensor<float> oneReg;
        AscendC::MicroAPI::RegTensor<int32_t> scaleReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg cmpReg;
        for (uint16_t rowIdx = 0; rowIdx < static_cast<uint16_t>(rowLen); rowIdx++) {
            uint32_t maskLen = static_cast<uint32_t>(tilingData_.afterAxis);
            AscendC::MicroAPI::Duplicate(oneReg, (float)1);
            for (uint16_t i = 0; i < loopCnt; i++) {
                maskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                auto rowAlignOfst = rowIdx * afterAxisAlignFp32_ + i * vfLen;
                auto sumAddrOfst = sumAddr + rowAlignOfst;
                auto rValueAddrOfst = rValueAddr + rowAlignOfst;
                auto quantaSumAddrOfst = quantaSumAddr + rowAlignOfst;
                DataCopy(dataReg, sumAddrOfst);
                DataCopy(rValueReg, rValueAddrOfst);
                /* 防止除0操作 */
                CompareScalar<float, CMPMODE::EQ>(cmpReg, rValueReg, (float)0, maskReg);
                Select(rValueReg, oneReg, rValueReg, cmpReg);
                Div(resReg, dataReg, rValueReg, maskReg);
                Muls(resReg, resReg, scaling, maskReg);
                AscendC::MicroAPI::Cast<int32_t, float, castTraitFp32ToInt32>(scaleReg, resReg, maskReg);
                DataCopy(quantaSumAddrOfst, scaleReg, maskReg);
            }
        }
    }
    sumQue_.FreeTensor(sumLocal);
    rValueQue_.FreeTensor(rValueLocal);
    quantaSumQue_.EnQue(quantaSumLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::CopyQuantizedSumOutToIntWs(int64_t preAxisIdx,
                                                                                         int64_t rowIdx, int64_t colIdx,
                                                                                         int64_t rowLen, int64_t colLen)
{
    LocalTensor<int32_t> quantaSumLocal = quantaSumQue_.DeQue<int32_t>();
    LocalTensor<IDX_T> sumIdxLocal = sumIdxQue_.DeQue<IDX_T>();

    int64_t sumIdx = preAxisIdx * tilingData_.varInAxis;
    SetAtomicAdd<int32_t>();
    for (int32_t i = 0; i < rowLen; i++) {
        int64_t rowOfset = (sumIdx + sumIdxLocal(i)) * tilingData_.afterAxis;
        int64_t outOfset = rowOfset;
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        CopyOut<int32_t>(sumQuanToIntWsGm_[outOfset], quantaSumLocal[i * afterAxisAlignFp32_], colLen);
    }
    SetAtomicNone();

    quantaSumQue_.FreeTensor(quantaSumLocal);
    sumIdxQue_.FreeTensor(sumIdxLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::CopyRValueAndIntWsIn(int64_t rowIdx,
                                                                                          int64_t colIdx,
                                                                                          int64_t rowLen,
                                                                                          int64_t colLen)
{
    LocalTensor<int32_t> sumQuanToIntLocal = sumQuanToIntQue_.AllocTensor<int32_t>();
    LocalTensor<float> rValueLocal = rValueQue_.AllocTensor<float>();

    int64_t rowOfset = GetBlockIdx() * tilingData_.eachCoreVarCount + rowIdx * tilingData_.ubVarFactor;
    int64_t inOfset = rowOfset * tilingData_.afterAxis;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(int32_t)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<int32_t> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                               static_cast<int32_t>(0)};
    DataCopyPad(sumQuanToIntLocal, sumQuanToIntWsGm_[inOfset], copyParams, padParams);

    copyParams.blockLen = static_cast<uint32_t>(colLen * sizeof(float));
    DataCopyPadExtParams<float> rValuepadParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                   static_cast<float>(0)};
    DataCopyPad(rValueLocal, updateRValueWsGm_[inOfset], copyParams, rValuepadParams);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    for (int64_t i = 0; i < rowLen; i++) {
        int64_t localOfset = i * afterAxisAlignFp32_;
        uint64_t gmOfset = rowOfset + i;
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(sameIdxCountWsGm_[gmOfset]);
        uint32_t count = sameIdxCountWsGm_(gmOfset);
        AscendC::Muls(rValueLocal[localOfset], rValueLocal[localOfset], static_cast<float>(count), colLen);
    }

    sumQuanToIntQue_.EnQue(sumQuanToIntLocal);
    rValueQue_.EnQue(rValueLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::CopyRValueAndIntWsAndIndexInOpti(
    int64_t preAxisIdx, int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<IDX_T> sumIdxLocal = sumIdxQue_.AllocTensor<IDX_T>();
    LocalTensor<float> rValueLocal = rValueQue_.AllocTensor<float>();
    LocalTensor<int32_t> sumQuanToIntLocal = sumQuanToIntQue_.AllocTensor<int32_t>();

    int64_t outOfset = rowIdx * tilingData_.ubVarOptiFactor;
    CopyIn<IDX_T>(sumIdxLocal, updateSumIdxWsGm_[outOfset], rowLen);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    uint64_t leftFlag = 0;
    for (uint64_t i = 0; i < rowLen; i++) {
        int64_t sumIdx = preAxisIdx * tilingData_.varInAxis + sumIdxLocal(i);
        if (sumIdx < 0 || sumIdx >= tilingData_.preAxis * tilingData_.varInAxis) {
            continue;
        }
        uint32_t count =
            AscendC::AtomicExch(const_cast<__gm__ uint32_t *>(sameIdxCountWsGm_[sumIdx].GetPhyAddr()), uint32_t(0));
        if (count == 0) {
            continue;
        }
        sumIdxLocal(leftFlag) = sumIdx;
        int64_t rowOfset = sumIdx * tilingData_.afterAxis;
        int64_t localOfset = (leftFlag++) * afterAxisAlignFp32_;
        CopyIn<float>(rValueLocal[localOfset], updateRValueWsGm_[rowOfset], colLen);
        CopyIn<int32_t>(sumQuanToIntLocal[localOfset], sumQuanToIntWsGm_[rowOfset], colLen);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::Muls(rValueLocal[localOfset], rValueLocal[localOfset], static_cast<float>(count), colLen);
    }
    uniqueSumIdNum_ = leftFlag;
    sumIdxQue_.EnQue(sumIdxLocal);
    rValueQue_.EnQue(rValueLocal);
    sumQuanToIntQue_.EnQue(sumQuanToIntLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::InverseQuantize(int64_t rowLen, int64_t colLen)
{
    LocalTensor<int32_t> sumQuanToIntLocal = sumQuanToIntQue_.DeQue<int32_t>();
    LocalTensor<float> rValueLocal = rValueQue_.DeQue<float>();
    LocalTensor<VAR_T> inverseQuantData = invQuanDataQue_.AllocTensor<VAR_T>();

    __local_mem__ int32_t *sumQuanToIntAddr = (__ubuf__ int32_t *)sumQuanToIntLocal.GetPhyAddr();
    __local_mem__ float *rValueAddr = (__ubuf__ float *)rValueLocal.GetPhyAddr();
    __local_mem__ VAR_T *invQuantDataAddr = (__ubuf__ VAR_T *)inverseQuantData.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    float scaleValue = static_cast<float>(1 << 30);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> dataReg;
        AscendC::MicroAPI::RegTensor<float> rReg;
        AscendC::MicroAPI::RegTensor<float> resReg;
        AscendC::MicroAPI::RegTensor<float> scaleReg;
        AscendC::MicroAPI::RegTensor<VAR_T> varTReg;
        AscendC::MicroAPI::MaskReg pregLoop;
        AscendC::MicroAPI::MaskReg maskReg;
        maskReg = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Duplicate(scaleReg, scaleValue, maskReg);
        for (uint16_t rowIdx = 0; rowIdx < static_cast<uint16_t>(rowLen); rowIdx++) {
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            for (uint16_t i = 0; i < loopCnt; ++i) {
                pregLoop = AscendC::MicroAPI::UpdateMask<uint32_t>(maskLen);
                auto rowAlignOfst = rowIdx * afterAxisAlignFp32_ + i * vfLen;
                DataCopy(dataReg, sumQuanToIntAddr + rowAlignOfst);
                DataCopy(rReg, rValueAddr + rowAlignOfst);
                Cast<float, int32_t, castTraitInt32ToFp32>(resReg, dataReg, pregLoop);
                Mul(resReg, resReg, rReg, pregLoop);
                Div(resReg, resReg, scaleReg, pregLoop);

                auto outOfset = invQuantDataAddr + rowIdx * afterAxisAlignSize_ + i * vfLen;
                if constexpr (!std::is_same<float, VAR_T>::value) {
                    Cast<VAR_T, float, castTraitFp32ToVarT>(varTReg, resReg, pregLoop);
                    DataCopy<VAR_T, MicroAPI::StoreDist::DIST_PACK_B32>(outOfset, varTReg, pregLoop);
                } else {
                    DataCopy(outOfset, resReg, pregLoop);
                }
            }
        }
    }
    invQuanDataQue_.EnQue(inverseQuantData);
    sumQuanToIntQue_.FreeTensor(sumQuanToIntLocal);
    rValueQue_.FreeTensor(rValueLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::CopyInverseQuantizedValueOutOpti(int64_t preAxisIdx,
                                                                                               int64_t colLen)
{
    LocalTensor<VAR_T> inverseQuantData = invQuanDataQue_.DeQue<VAR_T>();
    LocalTensor<IDX_T> sumIdxLocal = sumIdxQue_.DeQue<IDX_T>();
    SetAtomicAdd<VAR_T>();
    for (uint64_t i = 0; i < uniqueSumIdNum_; i++) {
        int64_t sumIdx = sumIdxLocal(i);
        int64_t rowOfset = sumIdx * tilingData_.afterAxis;
        Muls(inverseQuantData[i * afterAxisAlignSize_], inverseQuantData[i * afterAxisAlignSize_], alphaValue_, colLen);

        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        CopyOut<VAR_T>(var_[rowOfset], inverseQuantData[i * afterAxisAlignSize_], colLen);
    }
    SetAtomicNone();
    invQuanDataQue_.FreeTensor(inverseQuantData);
    sumIdxQue_.FreeTensor(sumIdxLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::CopyInverseQuantizedValueOut(int64_t rowIdx,
                                                                                           int64_t colIdx,
                                                                                           int64_t rowLen,
                                                                                           int64_t colLen)
{
    LocalTensor<VAR_T> inverseQuantData = invQuanDataQue_.DeQue<VAR_T>();

    int64_t rowOfset = GetBlockIdx() * tilingData_.eachCoreVarCount + rowIdx * tilingData_.ubVarFactor;
    int64_t outOfset = rowOfset * tilingData_.afterAxis;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    Muls(inverseQuantData, inverseQuantData, alphaValue_, rowLen * afterAxisAlignSize_);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    SetAtomicAdd<VAR_T>();
    DataCopyPad(var_[outOfset], inverseQuantData, copyParams);
    SetAtomicNone();
    invQuanDataQue_.FreeTensor(inverseQuantData);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::ProcessFirstStep(int64_t preAxisIdx)
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    /* step 1: compute R value */
    pipe_.Reset();
    shiftOffset_ = UB_AGLIN_VALUE / sizeof(IDX_T);
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER, tilingData_.ubIndexFactor * sizeof(IDX_T));
    pipe_.InitBuffer(
        sortIndicesQue_,
        Ops::Base::CeilAlign(tilingData_.ubIndexFactor * sizeof(IDX_T) + 2 * UB_AGLIN_VALUE, UB_AGLIN_VALUE));
    pipe_.InitBuffer(updatesOriginIdexQue_, DOUBLE_BUFFER, tilingData_.ubIndexFactor * sizeof(uint32_t));
    pipe_.InitBuffer(uniqueIdCountQue_, DOUBLE_BUFFER,
                     Ops::Base::CeilAlign((tilingData_.ubIndexFactor + 1) * sizeof(int32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(updateSumIdxQue_, DOUBLE_BUFFER,
                     Ops::Base::CeilAlign((tilingData_.ubIndexFactor + 1) * sizeof(IDX_T), UB_AGLIN_VALUE));

    pipe_.InitBuffer(updatesQue_, DOUBLE_BUFFER, tilingData_.ubIndexFactor * afterAxisAlignSize_ * sizeof(VAR_T));
    pipe_.InitBuffer(updatesCastQue_, DOUBLE_BUFFER, tilingData_.ubIndexFactor * afterAxisAlignSize_ * sizeof(float));
    pipe_.InitBuffer(updateSumQue_, DOUBLE_BUFFER, tilingData_.ubIndexFactor * afterAxisAlignFp32_ * sizeof(float));

    int64_t rowLoopNum = Ops::Base::CeilDiv(curCoreIndexCount_, tilingData_.ubIndexFactor);
    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowTailDataLen = curCoreIndexCount_ - tilingData_.ubIndexFactor * (rowLoopNum - 1);

    for (int32_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
        CopyIndicesAndUpdatesIn(preAxisIdx, rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
        ComputeSumForSameIndex(rowMainDataLen, tilingData_.afterAxis);
        CopySumOutToWs(preAxisIdx, rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
    }
    CopyIndicesAndUpdatesIn(preAxisIdx, rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
    ComputeSumForSameIndex(rowTailDataLen, tilingData_.afterAxis);
    CopySumOutToWs(preAxisIdx, rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::ProcessSecondStep(int64_t preAxisIdx)
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    /* step 2: quantize value in sumworkspace */
    pipe_.Reset();
    pipe_.InitBuffer(sumIdxQue_, DOUBLE_BUFFER, tilingData_.ubQuantaIndxFactor * sizeof(IDX_T));
    pipe_.InitBuffer(sumQue_, DOUBLE_BUFFER, tilingData_.ubQuantaIndxFactor * afterAxisAlignFp32_ * sizeof(float));
    pipe_.InitBuffer(quantaSumQue_, DOUBLE_BUFFER,
                     tilingData_.ubQuantaIndxFactor * afterAxisAlignFp32_ * sizeof(int32_t));
    /* actual useful len is less equal ubQuantaIndxFactor */
    pipe_.InitBuffer(rValueQue_, DOUBLE_BUFFER, tilingData_.ubQuantaIndxFactor * afterAxisAlignFp32_ * sizeof(float));

    int64_t rowLoopNum = Ops::Base::CeilDiv(curCoreIndexCount_, tilingData_.ubQuantaIndxFactor);
    int64_t rowMainDataLen = tilingData_.ubQuantaIndxFactor;
    int64_t rowTailDataLen = curCoreIndexCount_ - tilingData_.ubQuantaIndxFactor * (rowLoopNum - 1);

    for (int32_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
        CopySumAndRValueIn(preAxisIdx, rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
        QuantizeForSum(rowMainDataLen, tilingData_.afterAxis);
        CopyQuantizedSumOutToIntWs(preAxisIdx, rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
    }
    CopySumAndRValueIn(preAxisIdx, rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
    QuantizeForSum(rowTailDataLen, tilingData_.afterAxis);
    CopyQuantizedSumOutToIntWs(preAxisIdx, rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::ProcessThirdStep()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumAfter) {
        return;
    }
    /* step 3: invers quantization and add to var */
    pipe_.Reset();
    pipe_.InitBuffer(sumQuanToIntQue_, DOUBLE_BUFFER, tilingData_.ubVarFactor * afterAxisAlignFp32_ * sizeof(int32_t));
    pipe_.InitBuffer(rValueQue_, DOUBLE_BUFFER, tilingData_.ubVarFactor * afterAxisAlignFp32_ * sizeof(float));
    pipe_.InitBuffer(invQuanDataQue_, DOUBLE_BUFFER, tilingData_.ubVarFactor * afterAxisAlignSize_ * sizeof(VAR_T));

    int64_t rowLoopNum = Ops::Base::CeilDiv(curCoreVarCount_, tilingData_.ubVarFactor);
    int64_t rowMainDataLen = tilingData_.ubVarFactor; // align by 32 bytes
    int64_t rowTailDataLen = curCoreVarCount_ - tilingData_.ubVarFactor * (rowLoopNum - 1);

    for (int32_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
        CopyRValueAndIntWsIn(rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
        InverseQuantize(rowMainDataLen, tilingData_.afterAxis);
        CopyInverseQuantizedValueOut(rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
    }
    CopyRValueAndIntWsIn(rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
    InverseQuantize(rowTailDataLen, tilingData_.afterAxis);
    CopyInverseQuantizedValueOut(rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::ProcessThirdStepOpti(int64_t preAxisIdx)
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    int64_t rowMainDataLen = tilingData_.ubVarOptiFactor;
    int64_t curCoreProcessCount = curCoreIndexCount_;
    int64_t rowLoopNum = Ops::Base::CeilDiv(curCoreProcessCount, rowMainDataLen);
    int64_t rowTailDataLen = curCoreProcessCount - rowMainDataLen * (rowLoopNum - 1);
    pipe_.Reset();
    pipe_.InitBuffer(sumQuanToIntQue_, DOUBLE_BUFFER, rowMainDataLen * afterAxisAlignFp32_ * sizeof(int32_t));
    pipe_.InitBuffer(rValueQue_, DOUBLE_BUFFER, rowMainDataLen * afterAxisAlignFp32_ * sizeof(float));
    pipe_.InitBuffer(invQuanDataQue_, DOUBLE_BUFFER, rowMainDataLen * afterAxisAlignSize_ * sizeof(VAR_T));
    pipe_.InitBuffer(sumIdxQue_, DOUBLE_BUFFER, rowMainDataLen * sizeof(IDX_T));
    for (uint64_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
        CopyRValueAndIntWsAndIndexInOpti(preAxisIdx, rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
        InverseQuantize(uniqueSumIdNum_, tilingData_.afterAxis);
        CopyInverseQuantizedValueOutOpti(preAxisIdx, tilingData_.afterAxis);
    }
    CopyRValueAndIntWsAndIndexInOpti(preAxisIdx, rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
    InverseQuantize(uniqueSumIdNum_, tilingData_.afterAxis);
    CopyInverseQuantizedValueOutOpti(preAxisIdx, tilingData_.afterAxis);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminstic<VAR_T, IDX_T>::Process()
{
    for (int64_t preAxisIdx = 0; preAxisIdx < tilingData_.preAxis; preAxisIdx++) {
        ProcessFirstStep(preAxisIdx);
        AscendC::SyncAll();

        ProcessSecondStep(preAxisIdx);
        AscendC::SyncAll();
    }
    if (tilingData_.isOpti == 1) {
        for (int64_t preAxisIdx = 0; preAxisIdx < tilingData_.preAxis; preAxisIdx++) {
            ProcessThirdStepOpti(preAxisIdx);
        }
        return;
    }
    ProcessThirdStep();
}
} // namespace MoeInplaceIndexAdd

#endif
