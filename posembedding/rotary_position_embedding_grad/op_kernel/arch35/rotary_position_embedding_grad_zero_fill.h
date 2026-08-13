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
 * \file rotary_position_embedding_grad_zero_fill.h
 * \brief dy/x 为空 tensor 时的 dcos/dsin 清零 kernel（空集求和语义，dcos/dsin 应为全 0）
 */

#ifndef __ROTARY_POSITION_EMBEDDING_GRAD_ZERO_FILL_H__
#define __ROTARY_POSITION_EMBEDDING_GRAD_ZERO_FILL_H__

namespace RotaryPositionEmbeddingGrad {
using namespace AscendC;

template <typename T>
class RopeGradZeroFill {
public:
    __aicore__ inline RopeGradZeroFill(TPipe *pipe)
        : pipe_(pipe){};

    __aicore__ inline void Init(GM_ADDR cosGrad, GM_ADDR sinGrad, int64_t cosShapeSize)
    {
        cosShapeSize_ = cosShapeSize;
        cosGradGm_.SetGlobalBuffer((__gm__ T *)cosGrad, cosShapeSize_);
        sinGradGm_.SetGlobalBuffer((__gm__ T *)sinGrad, cosShapeSize_);
        pipe_->InitBuffer(ubBuf_, UB_ELEMS * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int64_t blockNum = GetBlockNum();
        int64_t blockIdx = GetBlockIdx();
        int64_t factor = (cosShapeSize_ + blockNum - 1) / blockNum;
        int64_t start = blockIdx * factor;
        if (start >= cosShapeSize_) {
            return;
        }
        int64_t count = cosShapeSize_ - start;
        count = (count < factor) ? count : factor;

        LocalTensor<T> zeroBuf = ubBuf_.Get<T>();
        DataCopyExtParams copyParams{1, 0, 0, 0, 0};
        for (int64_t offset = 0; offset < count; offset += UB_ELEMS) {
            int64_t curr = count - offset;
            curr = (curr < UB_ELEMS) ? curr : UB_ELEMS;
            Duplicate<T>(zeroBuf, static_cast<T>(0), static_cast<int32_t>(curr));
            // Duplicate(V) -> DataCopyPad(MTE3) 存在跨 pipe 依赖，用 PIPE_ALL 保证写完成后再搬出；
            // 循环末尾的 PIPE_ALL 同时保证 MTE3 读完后下一轮 Duplicate 再写入
            PipeBarrier<PIPE_ALL>();
            copyParams.blockLen = static_cast<uint32_t>(curr * sizeof(T));
            DataCopyPad(cosGradGm_[start + offset], zeroBuf, copyParams);
            DataCopyPad(sinGradGm_[start + offset], zeroBuf, copyParams);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    constexpr static int64_t UB_ELEMS = 8192; // 清零块大小（fp32 32KB / fp16/bf16 16KB）
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> ubBuf_;
    GlobalTensor<T> cosGradGm_;
    GlobalTensor<T> sinGradGm_;
    int64_t cosShapeSize_ = 0;
};

} // namespace RotaryPositionEmbeddingGrad
#endif // __ROTARY_POSITION_EMBEDDING_GRAD_ZERO_FILL_H__
