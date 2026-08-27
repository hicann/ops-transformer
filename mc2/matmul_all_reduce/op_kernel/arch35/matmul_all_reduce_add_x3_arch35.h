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
 * \file matmul_all_reduce_add_x3_arch35.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_ADD_X3_ARCH35_H
#define MATMUL_ALL_REDUCE_ADD_X3_ARCH35_H

namespace AscendC {
constexpr uint32_t DOUBLE_BUFFER = 2;

template <class T>
class MatmulAllReduceAddX3 {
public:
    __aicore__ inline MatmulAllReduceAddX3() {}
    __aicore__ inline void Init(GM_ADDR mmOutput, GM_ADDR add, uint64_t totalCnt, uint64_t tileCnt, TPipe *tPipe,
                                uint32_t coreNum)
    {
        pipe_ = tPipe;
        this->blockCnt_ = totalCnt / coreNum;
        uint64_t blockAddr = this->blockCnt_ * GetBlockIdx();
        if ((coreNum - 1) == GetBlockIdx()) {
            this->blockCnt_ = totalCnt - this->blockCnt_ * GetBlockIdx();
        }
        this->tileNum_ = Ceil(this->blockCnt_, tileCnt);
        this->tileCnt_ = tileCnt;

        mmOutGm_.SetGlobalBuffer((__gm__ T *)mmOutput + blockAddr, this->blockCnt_);
        addGm_.SetGlobalBuffer((__gm__ T *)add + blockAddr, this->blockCnt_);
        pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER, tileCnt * sizeof(T));
        pipe_->InitBuffer(inQueueY_, DOUBLE_BUFFER, tileCnt * sizeof(T));
        pipe_->InitBuffer(outQueueZ_, DOUBLE_BUFFER, tileCnt * sizeof(T));
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#else
        if (std::is_same<T, bfloat16_t>::value) {
            pipe_->InitBuffer(tempQueOutFp32_, tileCnt * sizeof(float));
            pipe_->InitBuffer(tempQueAddFp32_, tileCnt * sizeof(float));
        }
#endif
    }
    __aicore__ inline void Process(uint64_t progress)
    {
        if (this->blockCnt_ == 0) {
            return;
        }
        uint64_t calcCnt =
            (progress == (this->tileNum_ - 1)) ? (this->blockCnt_ - progress * this->tileCnt_) : this->tileCnt_;
        DataCopyParams copyParams = {1, static_cast<uint16_t>(calcCnt * sizeof(T)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        LocalTensor<T> mmOutLocal = inQueueX_.AllocTensor<T>();
        DataCopyPad(mmOutLocal, mmOutGm_[progress * this->tileCnt_], copyParams, padParams);
        inQueueX_.EnQue(mmOutLocal);

        LocalTensor<T> addLocal = inQueueY_.AllocTensor<T>();
        DataCopyPad(addLocal, addGm_[progress * this->tileCnt_], copyParams, padParams);
        inQueueY_.EnQue(addLocal);

        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY_.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ_.AllocTensor<T>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        Add(zLocal, xLocal, yLocal, calcCnt);
        PipeBarrier<PIPE_V>();
#else
        if (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<float> outFp32LocalTmp = tempQueOutFp32_.Get<float>();
            LocalTensor<float> addFp32LocalTmp = tempQueAddFp32_.Get<float>();
            Cast(outFp32LocalTmp, xLocal, RoundMode::CAST_NONE, calcCnt);
            Cast(addFp32LocalTmp, yLocal, RoundMode::CAST_NONE, calcCnt);
            PipeBarrier<PIPE_V>();
            Add(outFp32LocalTmp, outFp32LocalTmp, addFp32LocalTmp, calcCnt);
            PipeBarrier<PIPE_V>();
            Cast(zLocal, outFp32LocalTmp, RoundMode::CAST_RINT, calcCnt);
            PipeBarrier<PIPE_V>();
        } else if (std::is_same<T, half>::value) {
            Add(zLocal, xLocal, yLocal, calcCnt);
            PipeBarrier<PIPE_V>();
        }
#endif
        outQueueZ_.EnQue<T>(zLocal);
        inQueueX_.FreeTensor(mmOutLocal);
        inQueueY_.FreeTensor(addLocal);

        LocalTensor<T> outLocal = outQueueZ_.DeQue<T>();
        DataCopyPad(mmOutGm_[progress * this->tileCnt_], outLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

    TPipe *pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueX_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueY_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueZ_;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#else
    TBuf<TPosition::VECCALC> tempQueOutFp32_;
    TBuf<TPosition::VECCALC> tempQueAddFp32_;
#endif
    GlobalTensor<T> mmOutGm_;
    GlobalTensor<T> addGm_;
    uint64_t blockCnt_;
    uint64_t tileNum_;
    uint64_t tileCnt_;
};

template <class T>
__aicore__ inline void MatmulAllReduceAddX3Kernel(GM_ADDR mmOutput, GM_ADDR add, uint64_t totalCnt, uint64_t tileCnt,
                                                  TPipe *tPipe)
{
    uint32_t coreNum = GetBlockNum() * GetTaskRation();
    if (g_coreType == AIC || (GetBlockIdx() >= coreNum)) {
        return;
    }
    tPipe->Reset();
    MatmulAllReduceAddX3<T> op;
    op.Init(mmOutput, add, totalCnt, tileCnt, tPipe, coreNum);
    for (uint64_t i = 0; i < op.tileNum_; i++) {
        op.Process(i);
    }
}

// ============================================================================
// MatmulAllReduceEpilogue: 二维感知 bias epilogue（bias broadcast add + 可选 x3 add）
// bias 移至 vec 累加：matmul 完成后、AllReduce 之前执行 mmOut + bias (+ x3)
// 循环顺序：外层 N-chunk（bias 每个 chunk 只搬入一次），内层 M 行
// ============================================================================
template <class T>
class MatmulAllReduceEpilogue {
public:
    __aicore__ inline MatmulAllReduceEpilogue() {}

    __aicore__ inline void Init(GM_ADDR mmOutput, GM_ADDR bias, GM_ADDR add, uint64_t mTileValue, uint64_t nValue,
                                uint64_t biasUbCnt, bool hasAdd, TPipe *tPipe, uint32_t coreNum)
    {
        pipe_ = tPipe;
        mTileValue_ = mTileValue;
        nValue_ = nValue;
        biasUbCnt_ = biasUbCnt;
        hasAdd_ = hasAdd;
        coreNum_ = coreNum;

        // M 行按核切分：各核分到 myRowStart_ ~ myRowEnd_，尾核取余数
        uint64_t rowsPerCore = mTileValue_ / coreNum_;
        myRowStart_ = rowsPerCore * GetBlockIdx();
        if ((coreNum_ - 1) == GetBlockIdx()) {
            myRowEnd_ = mTileValue_;
        } else {
            myRowEnd_ = myRowStart_ + rowsPerCore;
        }

        // biasUbCnt 为 0 或无 M 行时不处理（安全保护）
        if (biasUbCnt_ == 0 || mTileValue_ == 0) {
            myRowStart_ = 0;
            myRowEnd_ = 0;
        }

        mmOutGm_.SetGlobalBuffer((__gm__ T *)mmOutput, mTileValue_ * nValue_);
        biasGm_.SetGlobalBuffer((__gm__ T *)bias, nValue_);
        if (hasAdd_) {
            addGm_.SetGlobalBuffer((__gm__ T *)add, mTileValue_ * nValue_);
        }

        // UB buffer 初始化，每个 buffer 大小 = biasUbCnt * sizeof(T)
        // bias 走 TBuf：每个 N-chunk 只搬入一次，对所有 M 行复用，避免重复 GM 读取
        pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER, biasUbCnt_ * sizeof(T));
        pipe_->InitBuffer(biasBuf_, biasUbCnt_ * sizeof(T));
        pipe_->InitBuffer(outQueueZ_, DOUBLE_BUFFER, biasUbCnt_ * sizeof(T));
        if (hasAdd_) {
            pipe_->InitBuffer(inQueueY_, DOUBLE_BUFFER, biasUbCnt_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (myRowStart_ >= myRowEnd_) {
            return;
        }
        // 外层 N-chunk：bias 每个 chunk 只搬入一次，对所有 M 行复用
        for (uint64_t nOff = 0; nOff < nValue_; nOff += biasUbCnt_) {
            // 确保上一 N-chunk 的所有 MTE3 (UB->GM 写回) 完成后再开始新 chunk。
            // 根因：hasAdd=true 路径的 V 流水线比 hasAdd=false 更慢（多一次 inQueueY_ MTE2 +
            // 第二次 in-place Add + 两次 PipeBarrier<PIPE_V>），导致每行 MTE3 issued 更晚。
            // 当 N 被分为 2+ chunk 时，上一 chunk 末行的 MTE3 在下一 chunk 开始时仍在 in-flight，
            // TQue 的 event 追踪不足以保证 AllocTensor(Z) 正确等待 MTE3 完成，
            // 导致下一 chunk 的 V 计算读到/写入未完成的 Z buffer，结果错误。
            // 在 chunk 边界加 PipeBarrier<PIPE_ALL> 显式排空全部流水线。
            // 第一个 chunk（nOff=0）此 barrier 为 no-op（无前置操作），无额外开销。
            PipeBarrier<PIPE_ALL>();
            uint64_t calcCnt = (nValue_ - nOff < biasUbCnt_) ? (nValue_ - nOff) : biasUbCnt_;
            DataCopyParams copyParams = {1, static_cast<uint16_t>(calcCnt * sizeof(T)), 0, 0};
            DataCopyPadParams padParams = {false, 0, 0, 0};

            // 1. 搬入 bias[nOff:nOff+calcCnt] → UB_bias（每 chunk 一次，所有行复用）
            LocalTensor<T> biasLocal = biasBuf_.Get<T>();
            DataCopyPad(biasLocal, biasGm_[nOff], copyParams, padParams);
            PipeBarrier<PIPE_V>(); // TBuf 不走 EnQue/DeQue，需手动同步

            // 内层 M 行：对每行的同一 N-chunk 执行 Add
            for (uint64_t row = myRowStart_; row < myRowEnd_; ++row) {
                uint64_t gmOff = row * nValue_ + nOff;

                // 2. 搬入 mmOut[row, nOff:nOff+calcCnt] → UB_x
                LocalTensor<T> mmOutLocal = inQueueX_.AllocTensor<T>();
                DataCopyPad(mmOutLocal, mmOutGm_[gmOff], copyParams, padParams);
                inQueueX_.EnQue(mmOutLocal);

                // 3. 若 hasAdd: 搬入 x3[row, nOff:nOff+calcCnt] → UB_y
                LocalTensor<T> addLocal;
                if (hasAdd_) {
                    addLocal = inQueueY_.AllocTensor<T>();
                    DataCopyPad(addLocal, addGm_[gmOff], copyParams, padParams);
                    inQueueY_.EnQue(addLocal);
                }

                LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
                LocalTensor<T> zLocal = outQueueZ_.AllocTensor<T>();
                if (hasAdd_) {
                    LocalTensor<T> yLocal = inQueueY_.DeQue<T>();
                    // result = mmOut + bias + x3（直接 Add，不走 fp32 cast）
                    Add(zLocal, xLocal, biasLocal, calcCnt);
                    PipeBarrier<PIPE_V>(); // 确保 zLocal 写入完成后再读（RAW 依赖）
                    Add(zLocal, zLocal, yLocal, calcCnt);
                } else {
                    // result = mmOut + bias
                    Add(zLocal, xLocal, biasLocal, calcCnt);
                }
                PipeBarrier<PIPE_V>();
                inQueueX_.FreeTensor(mmOutLocal);
                if (hasAdd_) {
                    inQueueY_.FreeTensor(addLocal);
                }

                // 4. 搬出 result → cGM[row, nOff:nOff+calcCnt]
                outQueueZ_.EnQue<T>(zLocal);
                LocalTensor<T> outLocal = outQueueZ_.DeQue<T>();
                DataCopyPad(mmOutGm_[gmOff], outLocal, copyParams);
                outQueueZ_.FreeTensor(zLocal);
            }
        }
    }

    TPipe *pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueX_;   // mmOut
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueY_;   // x3 (仅 hasAdd 时使用)
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueZ_; // result
    TBuf<TPosition::VECCALC> biasBuf_;                   // bias（TBuf，每个 N-chunk 搬入一次，多行复用）

    GlobalTensor<T> mmOutGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> addGm_;
    bool hasAdd_;
    uint64_t mTileValue_;
    uint64_t nValue_;
    uint64_t biasUbCnt_;
    uint64_t myRowStart_;
    uint64_t myRowEnd_;
    uint32_t coreNum_;
};

template <class T>
__aicore__ inline void MatmulAllReduceEpilogueKernel(GM_ADDR mmOutput, GM_ADDR bias, GM_ADDR add, uint64_t mTileValue,
                                                     uint64_t nValue, uint64_t biasUbCnt, bool hasAdd, TPipe *tPipe)
{
    uint32_t coreNum = GetBlockNum() * GetTaskRation();
    if (g_coreType == AIC || (GetBlockIdx() >= coreNum)) {
        return;
    }
    tPipe->Reset();
    MatmulAllReduceEpilogue<T> op;
    op.Init(mmOutput, bias, add, mTileValue, nValue, biasUbCnt, hasAdd, tPipe, coreNum);
    op.Process();
}
} // namespace AscendC
#endif // MATMUL_ALL_REDUCE_ADD_X3_H
