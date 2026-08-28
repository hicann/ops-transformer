/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTARY_POSITION_EMBEDDING3D_H
#define ROTARY_POSITION_EMBEDDING3D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "rotary_position_embedding3d_tiling_data.h"
#include "rotary_position_embedding3d_tiling_key.h"

using namespace AscendC;

namespace NsRotaryPositionEmbedding3d {

template <typename T>
class RotaryPositionEmbedding3d {
public:
    __aicore__ inline RotaryPositionEmbedding3d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR z, GM_ADDR,
                                const RotaryPositionEmbedding3dTilingData *td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<T> xGm_, cosGm_, zGm_;
    int64_t blockLength_;
    int64_t headDim_;
    int32_t halfD_;
    int32_t tileNum_;
    uint32_t tileLength_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
};

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR z, GM_ADDR,
                                                          const RotaryPositionEmbedding3dTilingData *td)
{
    headDim_ = td->headDim;
    halfD_ = static_cast<int32_t>(headDim_ / 2);

    blockLength_ = td->blockLength;
    tileNum_ = static_cast<int32_t>(td->tileNum);
    tileLength_ = static_cast<uint32_t>(td->tileLength);

    int64_t blockOffset = blockLength_ * GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T *)x + blockOffset, blockLength_);
    cosGm_.SetGlobalBuffer((__gm__ T *)cos + blockOffset, blockLength_);
    zGm_.SetGlobalBuffer((__gm__ T *)z + blockOffset, blockLength_);

    pipe_.InitBuffer(inQueue_, 1, tileLength_ * sizeof(T) * 2);
    pipe_.InitBuffer(outQueue_, 1, tileLength_ * sizeof(T));
    pipe_.InitBuffer(tmpBuf_, static_cast<uint32_t>(halfD_) * sizeof(T) * 2);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> buf = inQueue_.AllocTensor<T>();
    DataCopy(buf, xGm_[progress * tileLength_], tileLength_);
    DataCopy(buf[tileLength_], cosGm_[progress * tileLength_], tileLength_);
    inQueue_.EnQue(buf);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::Compute(int32_t progress)
{
    LocalTensor<T> buf = inQueue_.DeQue<T>();
    LocalTensor<T> xLocal = buf;
    LocalTensor<T> cosLocal = buf[tileLength_];
    LocalTensor<T> zLocal = outQueue_.AllocTensor<T>();

    int32_t halfD = halfD_;
    int32_t headDim = static_cast<int32_t>(headDim_);
    int32_t numPositions = static_cast<int32_t>(tileLength_ / headDim_);

    auto tmp = tmpBuf_.Get<T>();
    for (int32_t p = 0; p < numPositions; p++) {
        int32_t base = p * headDim;

        // z[k]       = x[k] * cos[k] - x[k+halfD] * sin[k]
        // z[k+halfD] = x[k] * sin[k] + x[k+halfD] * cos[k]
        Mul(zLocal[base], xLocal[base], cosLocal[base], halfD);                 // zL = xL * cos
        Mul(zLocal[base + halfD], xLocal[base], cosLocal[base + halfD], halfD); // zR = xL * sin
        Mul(tmp, xLocal[base + halfD], cosLocal[base + halfD], halfD);          // tmp = xR * sin
        PipeBarrier<PIPE_V>();
        Sub(zLocal[base], zLocal[base], tmp, halfD);           // zL = xL*cos - xR*sin
        Mul(tmp, xLocal[base + halfD], cosLocal[base], halfD); // tmp = xR * cos
        PipeBarrier<PIPE_V>();
        Add(zLocal[base + halfD], zLocal[base + halfD], tmp, halfD); // zR = xL*sin + xR*cos
    }

    outQueue_.EnQue(zLocal);
    inQueue_.FreeTensor(buf);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> zLocal = outQueue_.DeQue<T>();
    DataCopy(zGm_[progress * tileLength_], zLocal, tileLength_);
    outQueue_.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbedding3d<T>::Process()
{
    if (tileNum_ == 0)
        return;
    for (int32_t i = 0; i < tileNum_; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsRotaryPositionEmbedding3d
#endif
