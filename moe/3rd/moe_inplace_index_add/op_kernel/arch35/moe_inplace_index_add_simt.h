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
 * \file moe_inplace_index_add_simt.h
 * \brief moe_inplace_index_add
 */
#ifndef ASCENDC_MOE_INPLACE_INDEX_ADD_SIMT_H_
#define ASCENDC_MOE_INPLACE_INDEX_ADD_SIMT_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "moe_inplace_index_add_common.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace MoeInplaceIndexAdd {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t USED_THREAD = 256;
#else
constexpr uint32_t USED_THREAD = 2048;
#endif
constexpr int64_t DB_BUFFER = 1;
constexpr int64_t LEAST_DEAL_SIZE = 512;

template <typename VAR_T, typename IDX_T, typename COMP_T, typename CAST_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
class MoeInplaceIndexAddSimt {
public:
    __aicore__ inline MoeInplaceIndexAddSimt(const MoeInplaceIndexAddSimtTilingData &tilingData, TPipe &pipe)
        : td_(tilingData),
          pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha, GM_ADDR workspace);
    __aicore__ inline void CopyVarToWsPerLoop(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyWsToVarPerLoop(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyVarToWs();
    __aicore__ inline void CopyWsToVar();
    __aicore__ inline void Process();

private:
    static __simt_vf__ __aicore__ inline void SimtCompute(COMP_T varInAxis, COMP_T afterAxis, COMP_T updatesStride0,
                                                          COMP_T updatesAxis, COMP_T m0, COMP_T shift0, COMP_T m1,
                                                          COMP_T shift1, __gm__ VAR_T *var, __gm__ IDX_T *indices,
                                                          __gm__ VAR_T *updates, __gm__ VAR_T *alpha,
                                                          __gm__ CAST_T *varWorkspaceGm, COMP_T blockIdx,
                                                          COMP_T blockNum, COMP_T indicesStride);

private:
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;
    GlobalTensor<VAR_T> alpha_;
    GlobalTensor<CAST_T> varWorkspaceGm_;
    TQue<QuePosition::VECIN, DB_BUFFER> varQue_;
    TQue<QuePosition::VECIN, DB_BUFFER> varCastQue_;
    TPipe &pipe_;

    const MoeInplaceIndexAddSimtTilingData &td_;
    COMP_T blockIdx_;
    COMP_T blockNum_;
    int64_t normBlockData_{0};
    int64_t usedCoreNum_{0};
    int64_t loopNum_{0};
    int64_t tailLoopLength_{0};
};

template <typename VAR_T, typename IDX_T, typename COMP_T, typename CAST_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimt<VAR_T, IDX_T, COMP_T, CAST_T, WITH_ALPHA, IS_CONTIGUOUS>::Init(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    var_.SetGlobalBuffer((__gm__ VAR_T *)(var));
    indices_.SetGlobalBuffer((__gm__ IDX_T *)(indices));
    updates_.SetGlobalBuffer((__gm__ VAR_T *)(updates));
    alpha_.SetGlobalBuffer((__gm__ VAR_T *)(alpha));
    if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value ||
                  IsSameType<VAR_T, int16_t>::value) {
        varWorkspaceGm_.SetGlobalBuffer((__gm__ CAST_T *)workspace);

        int64_t varAxis = td_.preAxis * td_.varInAxis * td_.afterAxis;
        normBlockData_ = Ops::Base::CeilDiv(varAxis, static_cast<int64_t>(blockNum_));
        int64_t minDealNum = LEAST_DEAL_SIZE / sizeof(VAR_T);
        normBlockData_ = normBlockData_ > minDealNum ? normBlockData_ : minDealNum;
        usedCoreNum_ = Ops::Base::CeilDiv(varAxis, normBlockData_);
        int64_t tailBlockData = varAxis - (usedCoreNum_ - 1) * normBlockData_;
        int64_t curCoreData = blockIdx_ != (usedCoreNum_ - 1) ? normBlockData_ : tailBlockData;
        loopNum_ = Ops::Base::FloorDiv(curCoreData, td_.ubFactor);
        tailLoopLength_ = curCoreData - loopNum_ * td_.ubFactor;

        pipe_.InitBuffer(varQue_, DB_BUFFER, td_.ubFactor * sizeof(VAR_T));
        pipe_.InitBuffer(varCastQue_, DB_BUFFER, td_.ubFactor * sizeof(CAST_T));
    }
}

template <typename VAR_T, typename IDX_T, typename COMP_T, typename CAST_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void
MoeInplaceIndexAddSimt<VAR_T, IDX_T, COMP_T, CAST_T, WITH_ALPHA, IS_CONTIGUOUS>::CopyVarToWsPerLoop(int64_t offset,
                                                                                                    int64_t dataLen)
{
    LocalTensor<VAR_T> xLocal = varQue_.AllocTensor<VAR_T>();
    CopyIn<VAR_T>(xLocal, var_[offset], dataLen);
    varQue_.EnQue(xLocal);

    event_t eventMte3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3toV);
    WaitFlag<HardEvent::MTE3_V>(eventMte3toV);

    LocalTensor<VAR_T> yLocal = varQue_.DeQue<VAR_T>();
    LocalTensor<CAST_T> castDst = varCastQue_.AllocTensor<CAST_T>();
    if constexpr (IsSameType<CAST_T, half>::value) {
        Cast(castDst, yLocal, RoundMode::CAST_NONE, dataLen);
    } else {
        CastToInt32<VAR_T, CAST_T>(castDst, yLocal, static_cast<uint32_t>(dataLen));
    }
    varCastQue_.EnQue(castDst);

    event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);

    LocalTensor<CAST_T> dstLocal = varCastQue_.DeQue<CAST_T>();
    CopyOut<CAST_T>(varWorkspaceGm_[offset], dstLocal, dataLen);
    varQue_.FreeTensor(yLocal);
    varCastQue_.FreeTensor(dstLocal);
}

template <typename VAR_T, typename IDX_T, typename COMP_T, typename CAST_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void
MoeInplaceIndexAddSimt<VAR_T, IDX_T, COMP_T, CAST_T, WITH_ALPHA, IS_CONTIGUOUS>::CopyWsToVarPerLoop(int64_t offset,
                                                                                                    int64_t dataLen)
{
    LocalTensor<CAST_T> xLocal = varCastQue_.AllocTensor<CAST_T>();
    CopyIn<CAST_T>(xLocal, varWorkspaceGm_[offset], dataLen);
    varCastQue_.EnQue(xLocal);

    event_t eventMte3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3toV);
    WaitFlag<HardEvent::MTE3_V>(eventMte3toV);

    LocalTensor<CAST_T> yLocal = varCastQue_.DeQue<CAST_T>();
    LocalTensor<VAR_T> castDst = varQue_.AllocTensor<VAR_T>();
    if constexpr (IsSameType<CAST_T, half>::value) {
        Cast(castDst, yLocal, RoundMode::CAST_RINT, dataLen);
    } else {
        CastToOrigin<VAR_T, CAST_T>(castDst, yLocal, static_cast<uint32_t>(dataLen));
    }
    varQue_.EnQue(castDst);

    event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);

    LocalTensor<VAR_T> dstLocal = varQue_.DeQue<VAR_T>();
    CopyOut<VAR_T>(var_[offset], dstLocal, dataLen);
    varCastQue_.FreeTensor(yLocal);
    varQue_.FreeTensor(dstLocal);
}

template <typename VAR_T, typename IDX_T, typename COMP_T, typename CAST_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimt<VAR_T, IDX_T, COMP_T, CAST_T, WITH_ALPHA, IS_CONTIGUOUS>::CopyVarToWs()
{
    int64_t offset = 0;
    int64_t blockOffset = static_cast<int64_t>(blockIdx_) * normBlockData_;
    for (int64_t idx = 0; idx < loopNum_; idx++) {
        offset = blockOffset + idx * td_.ubFactor;
        CopyVarToWsPerLoop(offset, td_.ubFactor);
    }

    if (tailLoopLength_ > 0) {
        offset = blockOffset + loopNum_ * td_.ubFactor;
        CopyVarToWsPerLoop(offset, tailLoopLength_);
    }
}

template <typename VAR_T, typename IDX_T, typename COMP_T, typename CAST_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimt<VAR_T, IDX_T, COMP_T, CAST_T, WITH_ALPHA, IS_CONTIGUOUS>::CopyWsToVar()
{
    int64_t offset = 0;
    int64_t blockOffset = static_cast<int64_t>(blockIdx_) * normBlockData_;
    for (int64_t idx = 0; idx < loopNum_; idx++) {
        offset = blockOffset + idx * td_.ubFactor;
        CopyWsToVarPerLoop(offset, td_.ubFactor);
    }

    if (tailLoopLength_ > 0) {
        offset = blockOffset + loopNum_ * td_.ubFactor;
        CopyWsToVarPerLoop(offset, tailLoopLength_);
    }
}

template <typename VAR_T, typename IDX_T, typename COMP_T, typename CAST_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void MoeInplaceIndexAddSimt<
    VAR_T, IDX_T, COMP_T, CAST_T, WITH_ALPHA, IS_CONTIGUOUS>::SimtCompute(COMP_T varInAxis, COMP_T afterAxis,
                                                                          COMP_T updatesStride0, COMP_T updatesAxis,
                                                                          COMP_T m0, COMP_T shift0, COMP_T m1,
                                                                          COMP_T shift1, __gm__ VAR_T *var,
                                                                          __gm__ IDX_T *indices, __gm__ VAR_T *updates,
                                                                          __gm__ VAR_T *alpha,
                                                                          __gm__ CAST_T *varWorkspaceGm,
                                                                          COMP_T blockIdx, COMP_T blockNum,
                                                                          COMP_T indicesStride)
{
    COMP_T varStride0 = varInAxis * afterAxis;
    VAR_T alphaValue = 1;
    if constexpr (WITH_ALPHA) {
        alphaValue = alpha[0];
    }

    for (COMP_T i = blockIdx * blockDim.x + threadIdx.x; i < updatesAxis; i += blockNum * blockDim.x) {
        COMP_T dim0Idx = Simt::UintDiv(i, m0, shift0);
        COMP_T dim0Rem = i - dim0Idx * updatesStride0;

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim2Idx = dim0Rem - dim1Idx * afterAxis;
        COMP_T indicesValue = 0;

        if constexpr (IS_CONTIGUOUS) {
            indicesValue = static_cast<COMP_T>(indices[dim1Idx]);
        } else {
            indicesValue = static_cast<COMP_T>(indices[dim1Idx * indicesStride]);
        }

        COMP_T varOffset = dim0Idx * varStride0 + indicesValue * afterAxis + dim2Idx;
        if constexpr (WITH_ALPHA) {
            if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value ||
                          IsSameType<VAR_T, int16_t>::value) {
                asc_atomic_add(varWorkspaceGm + varOffset,
                               static_cast<CAST_T>(static_cast<float>(updates[i]) * static_cast<float>(alphaValue)));
            } else if constexpr (IsSameType<VAR_T, bfloat16_t>::value || IsSameType<VAR_T, half>::value) {
                asc_atomic_add(var + varOffset,
                               static_cast<VAR_T>(static_cast<float>(updates[i]) * static_cast<float>(alphaValue)));
            } else {
                asc_atomic_add(var + varOffset, updates[i] * alphaValue);
            }
        } else {
            if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value ||
                          IsSameType<VAR_T, int16_t>::value) {
                asc_atomic_add(varWorkspaceGm + varOffset, static_cast<CAST_T>(updates[i]));
            } else {
                asc_atomic_add(var + varOffset, updates[i]);
            }
        }
    }
}

template <typename VAR_T, typename IDX_T, typename COMP_T, typename CAST_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimt<VAR_T, IDX_T, COMP_T, CAST_T, WITH_ALPHA, IS_CONTIGUOUS>::Process()
{
    if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value ||
                  IsSameType<VAR_T, int16_t>::value) {
        if (blockIdx_ < usedCoreNum_) {
            CopyVarToWs();
        }
        SyncAll();
    }

    COMP_T afterAxis = td_.afterAxis;
    COMP_T updatesStride0 = td_.updatesInAxis * afterAxis;
    COMP_T updatesAxis = td_.preAxis * updatesStride0;
    COMP_T varInAxis = td_.varInAxis;
    COMP_T m0 = 1;
    COMP_T shift0 = 1;
    COMP_T m1 = 1;
    COMP_T shift1 = 1;
    COMP_T indicesStride = static_cast<COMP_T>(td_.indicesStride);
    GetUintDivMagicAndShift(m0, shift0, updatesStride0);
    GetUintDivMagicAndShift(m1, shift1, afterAxis);

    asc_vf_call<MoeInplaceIndexAddSimt<VAR_T, IDX_T, COMP_T, CAST_T, WITH_ALPHA, IS_CONTIGUOUS>::SimtCompute>(
        dim3(USED_THREAD), varInAxis, afterAxis, updatesStride0, updatesAxis, m0, shift0, m1, shift1,
        (__gm__ VAR_T *)(var_.GetPhyAddr()), (__gm__ IDX_T *)(indices_.GetPhyAddr()),
        (__gm__ VAR_T *)(updates_.GetPhyAddr()), (__gm__ VAR_T *)(alpha_.GetPhyAddr()),
        (__gm__ CAST_T *)(varWorkspaceGm_.GetPhyAddr()), blockIdx_, blockNum_, indicesStride);
    if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value ||
                  IsSameType<VAR_T, int16_t>::value) {
        SyncAll();
        if (blockIdx_ < usedCoreNum_) {
            CopyWsToVar();
        }
    }
}
} // namespace MoeInplaceIndexAdd

#endif
