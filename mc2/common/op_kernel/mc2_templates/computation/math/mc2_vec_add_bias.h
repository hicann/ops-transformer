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
 * \file mc2_vec_add_bias.h
 * \brief Vector Add Bias: 将 1D bias [N] 逐行广播加到 matmul 结果 [M, N] 上 (单卡, in-place)。
 *
 * 流水线: MTE2(GM→UB) → V(Cast↑→Add→Cast↓) → MTE3(UB→GM), half 双缓冲 pingpong。
 * 同步: 手动 SetFlag/WaitFlag (HardEvent), 无 EnQue/DeQue。
 *
 * 精度策略:
 *   bias 为 FP32 时 (BiasDataType=float): Cast(matmul→float) → Add<float> → Cast(float→matmul), 避免 FP16
 * 加法精度损失。 bias 与 matmul 同类型时 (FP16/BF16): 直接 Add<MatmulDataType>, 无需 Cast。
 *
 * 分核策略 (SplitModeVal):
 *   M_SPLIT  (1) — 按行分区: 每核负责 [mStart, mEnd) × 全 N
 *   N_SPLIT  (2) — 按列分区: 每核负责 全 M × [nStart, nEnd)
 *   MN_SPLIT (3) — 按 tile 索引分区: 每核分到一组 (mTile, nTile) 组合
 */

#ifndef MC2_VEC_ADD_BIAS_H
#define MC2_VEC_ADD_BIAS_H

#include <type_traits>

namespace MC2KernelTemplate {
using namespace AscendC;

struct MC2AddBiasContext {
    GM_ADDR matmulAddr = 0;    // matmul 结果 [M, N], row-major, in-place
    GM_ADDR biasAddr = 0;      // bias [N]
    uint64_t matmulOffset = 0; // 每个 task 之间的 matmul 偏移 (字节)
    uint64_t biasOffset = 0;   // 每个 task 之间的 bias 偏移 (字节, bias 为一维数组时恒为 0)
    uint64_t M = 0;            // matmul 结果行数
    uint64_t N = 0;            // matmul 结果列数 (= bias 长度)
};

template <typename MatmulDataType, typename BiasDataType = MatmulDataType, uint32_t SplitModeVal = 1>
class MC2VecAddBias {
public:
    enum SplitMode : uint32_t {
        M_SPLIT = 1,
        N_SPLIT = 2,
        MN_SPLIT = 3
    };
    static constexpr SplitMode MODE = static_cast<SplitMode>(SplitModeVal);

    struct TileRange {
        uint64_t start, step, count;
    };

    __aicore__ inline MC2VecAddBias(TPipe *tPipe)
        : tPipe_(tPipe){};
    __aicore__ inline MC2AddBiasContext *GetContextPtr();
    __aicore__ inline void Init() {};
    __aicore__ inline void Process(uint32_t taskIndex);

protected:
    __aicore__ inline void Init(GM_ADDR matmulAddr, GM_ADDR biasAddr);
    __aicore__ inline void Process();
    __aicore__ inline void Destroy();

    static constexpr uint32_t ELEMS_PER_BLOCK = 32 / sizeof(MatmulDataType);
    static constexpr uint32_t ELEMS_PER_REPEAT = 256 / sizeof(MatmulDataType);
    static constexpr uint32_t TILE_M = 64;
    static constexpr uint32_t TILE_N = 256;
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t AIV_PER_AIC = 2;

    // FP32 计算常量
    static constexpr uint32_t FP32_ELEMS_PER_BLOCK = 32 / sizeof(float);
    static constexpr uint32_t FP32_ELEMS_PER_REPEAT = 256 / sizeof(float);

    // matmul 为 half/bf16 时需要 FP32 中间计算; bias 非 float 时需要 Cast→float
    static constexpr bool NEED_FP32_COMPUTE = !std::is_same<MatmulDataType, float>::value;
    static constexpr bool NEED_BIAS_CAST = !std::is_same<BiasDataType, float>::value;

    MC2AddBiasContext context_;
    GlobalTensor<MatmulDataType> matmulGm_;
    GlobalTensor<BiasDataType> biasGm_;
    TPipe *tPipe_;
    TBuf<TPosition::VECCALC> matmulTileBuf_[BUFFER_NUM];
    TBuf<TPosition::VECCALC> biasTileBuf_[BUFFER_NUM];
    TBuf<TPosition::VECCALC> matmulFp32Buf_; // FP32 计算缓冲 (单缓冲, V pipe in-order 保证不冲突)
    TBuf<TPosition::VECCALC> matmulOutBuf_[BUFFER_NUM]; // CastBack 输出缓冲 (双缓冲, 仅 NEED_FP32_COMPUTE 时分配)
    TBuf<TPosition::VECCALC> biasFp32Buf_;              // bias Cast→float (NEED_BIAS_CAST 时)
    LocalTensor<MatmulDataType> matmulTile_[BUFFER_NUM];
    LocalTensor<BiasDataType> biasTile_[BUFFER_NUM];
    LocalTensor<float> matmulFp32_;
    LocalTensor<MatmulDataType> matmulOut_[BUFFER_NUM];
    AscendC::TEventID copyInEvent_;  // MTE2_S: CopyIn 完成 → Compute Wait (同 tile, 共享)
    AscendC::TEventID computeEvent_; // V_MTE3: Compute 完成 → CopyOut Wait (同 tile, 共享)
    AscendC::TEventID
        copyOutDoneEvent_[BUFFER_NUM]; // MTE3_S: CopyOut 完成 → CastBack(next) Wait (释放 matmulOut_, 双缓冲)
    AscendC::TEventID castDoneEvent_[BUFFER_NUM]; // V_MTE2: Cast 完成 → CopyIn(next) Wait (释放 matmulTile_, 双缓冲)
    AscendC::TEventID biasInEvent_;               // MTE2_S: LoadBias 完成 → PrepareBias Wait (共享)
    AscendC::TEventID biasCastDoneEvent_; // V_MTE2: PrepareBias Cast 完成 → LoadBias(next) Wait (释放 biasTile_, 共享)

    // M_SPLIT / N_SPLIT: 数据维度分区范围 [start, end)
    uint32_t myMStart_ = 0, myMEnd_ = 0;
    uint32_t myNStart_ = 0, myNEnd_ = 0;
    // MN_SPLIT: tile 索引分区
    uint32_t tileMCnt_ = 0, tileNCnt_ = 0;
    uint32_t tailM_ = 0, tailN_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t nTileStart_ = 0, nTileEnd_ = 0;
    uint32_t myTileStart_ = 0, myTileEnd_ = 0;

    static __aicore__ inline uint32_t TileSize(uint32_t idx, uint32_t cnt, uint32_t tail, uint32_t full)
    {
        return (idx == cnt - 1) ? tail : full;
    }

    template <typename T>
    static __aicore__ inline DataCopyPadExtParams<T> PadParams()
    {
        return {true, 0, 0, static_cast<T>(0)};
    }

    __aicore__ inline void CalcSplit()
    {
        usedCoreNum_ = static_cast<uint32_t>(GetBlockNum()) * AIV_PER_AIC;

        if constexpr (MODE == SplitMode::MN_SPLIT) {
            tileMCnt_ = Ceil(context_.M, TILE_M);
            tileNCnt_ = Ceil(context_.N, TILE_N);
            if (tileMCnt_ == 0 || tileNCnt_ == 0) {
                usedCoreNum_ = 0;
                return;
            }
            tailM_ = context_.M - static_cast<uint64_t>(tileMCnt_ - 1) * TILE_M;
            tailN_ = context_.N - static_cast<uint64_t>(tileNCnt_ - 1) * TILE_N;
            uint64_t total64 = static_cast<uint64_t>(tileMCnt_) * static_cast<uint64_t>(tileNCnt_);
            uint32_t total = static_cast<uint32_t>(total64);
            if (total < usedCoreNum_) {
                usedCoreNum_ = total;
            }
            uint32_t per = Ceil(total, usedCoreNum_);
            myTileStart_ = static_cast<uint32_t>(static_cast<uint64_t>(GetBlockIdx()) * per);
            myTileEnd_ = (myTileStart_ + per > total) ? total : myTileStart_ + per;
            nTileStart_ = myTileStart_ / tileMCnt_;
            nTileEnd_ = (myTileEnd_ == 0) ? 0 : (myTileEnd_ - 1) / tileMCnt_ + 1;
        } else {
            if (context_.M == 0 || context_.N == 0) {
                usedCoreNum_ = 0;
                return;
            }
            uint32_t coreId = static_cast<uint32_t>(GetBlockIdx());

            if constexpr (MODE == SplitMode::M_SPLIT) {
                uint32_t mDim = static_cast<uint32_t>(context_.M);
                uint32_t nDim = static_cast<uint32_t>(context_.N);
                uint32_t mPerCore = Ceil(mDim, usedCoreNum_);
                myMStart_ = coreId * mPerCore;
                myMEnd_ = (myMStart_ + mPerCore > mDim) ? mDim : (myMStart_ + mPerCore);
                myNStart_ = 0;
                myNEnd_ = nDim;
            } else {
                uint32_t mDim = static_cast<uint32_t>(context_.M);
                uint32_t nDim = static_cast<uint32_t>(context_.N);
                uint32_t nPerCore = Ceil(nDim, usedCoreNum_);
                myNStart_ = coreId * nPerCore;
                myNEnd_ = (myNStart_ + nPerCore > nDim) ? nDim : (myNStart_ + nPerCore);
                myMStart_ = 0;
                myMEnd_ = mDim;
            }
        }
    }

    // --- MN_SPLIT 辅助 ---

    __aicore__ inline TileRange GetMRange(uint32_t nTile) const
    {
        uint32_t base = nTile * tileMCnt_;
        uint32_t lo = (base < myTileStart_) ? (myTileStart_ - base) : 0;
        uint32_t hi = (base + tileMCnt_ > myTileEnd_) ? (myTileEnd_ - base) : tileMCnt_;
        return {lo, 1, (hi > lo) ? (hi - lo) : 0};
    }

    // --- Bias 加载 ---

    __aicore__ inline void LoadBias(uint32_t nStart, uint32_t len, uint32_t bufIdx)
    {
        // 等上一轮 PrepareBias 的 Cast 读完 biasTile_ 后, 才允许覆写
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(biasCastDoneEvent_);
        DataCopyExtParams params{1, static_cast<uint32_t>(len * sizeof(BiasDataType)), 0, 0, 0};
        DataCopyPad(biasTile_[bufIdx], biasGm_[nStart], params, PadParams<BiasDataType>());
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(biasInEvent_);
    }

    __aicore__ inline void LoadBiasTile(uint32_t nTile, uint32_t bufIdx)
    {
        uint32_t nStart = nTile * TILE_N;
        uint32_t len = TileSize(nTile, tileNCnt_, tailN_, TILE_N);
        LoadBias(nStart, len, bufIdx);
    }

    // Cast bias 到 float (若类型不同), 整个 M 循环内复用同一份 bias
    // Cast 与后续 Add 同在 V-pipe, 硬件保序保证 Cast 先完成, 无需额外同步
    __aicore__ inline LocalTensor<float> PrepareBias(uint32_t biasBufIdx, uint32_t curTileN)
    {
        if constexpr (NEED_BIAS_CAST) {
            LocalTensor<float> biasFp32 = biasFp32Buf_.template Get<float>();
            Cast(biasFp32, biasTile_[biasBufIdx], RoundMode::CAST_NONE, (int32_t)curTileN);
            // Cast 完成 → biasTile_ 释放, 下一轮 LoadBias 可覆写
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasCastDoneEvent_);
            return biasFp32;
        } else {
            return biasTile_[biasBufIdx].template ReinterpretCast<float>();
        }
    }

    // --- MTE2 / V / MTE3 三级流水 ---

    // CopyIn: FP32 路径等 castDoneEvent (Cast 完成释放 matmulTile_); 直连路径等 copyOutDoneEvent
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t curTileM, uint32_t curTileN, uint32_t bufIdx)
    {
        if constexpr (NEED_FP32_COMPUTE) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(castDoneEvent_[bufIdx]);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copyOutDoneEvent_[bufIdx]);
        }
        uint32_t blockLen = curTileN * sizeof(MatmulDataType);
        DataCopyExtParams params{static_cast<uint16_t>(curTileM), blockLen,
                                 static_cast<int64_t>((context_.N - curTileN) * sizeof(MatmulDataType)), 0, 0};
        DataCopyPad(matmulTile_[bufIdx], matmulGm_[offset], params, PadParams<MatmulDataType>());
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copyInEvent_);
    }

    __aicore__ inline void Compute(const LocalTensor<float> &bias, uint32_t curTileM, uint32_t curTileN,
                                   uint32_t ubRowStride, uint32_t bufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(copyInEvent_);
        if constexpr (NEED_FP32_COMPUTE) {
            ComputeFp32(bias, curTileM, curTileN, ubRowStride, bufIdx);
        } else {
            ComputeDirect(bias, curTileM, curTileN, ubRowStride, bufIdx);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(computeEvent_);
    }

    // FP32 路径: Cast↑(matmul→float) → Set castDone → Add<float> → Wait copyOutDone → Cast↓(float→matmulOut_)
    __aicore__ inline void ComputeFp32(const LocalTensor<float> &bias, uint32_t curTileM, uint32_t curTileN,
                                       uint32_t ubRowStride, uint32_t bufIdx)
    {
        uint32_t ubRowStrideFp32 = Ceil(curTileN, FP32_ELEMS_PER_BLOCK) * FP32_ELEMS_PER_BLOCK;
        bool rowContiguous = (ubRowStride == curTileN);

        if (rowContiguous) {
            Cast<float, MatmulDataType>(matmulFp32_, matmulTile_[bufIdx], RoundMode::CAST_NONE,
                                        (int32_t)(curTileM * curTileN));
        } else {
            for (uint32_t r = 0; r < curTileM; r++) {
                Cast<float, MatmulDataType>(matmulFp32_[r * ubRowStrideFp32], matmulTile_[bufIdx][r * ubRowStride],
                                            RoundMode::CAST_NONE, (int32_t)curTileN);
            }
        }
        // Cast 完成 → matmulTile_ 释放, 下一轮 CopyIn 可开始
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(castDoneEvent_[bufIdx]);

        // Add<float> 广播 (V pipe in-order, 与 Cast 和 CastBack 保序)
        uint32_t segCnt = Ceil(curTileN, FP32_ELEMS_PER_REPEAT);
        uint8_t repStrideFp32 = static_cast<uint8_t>(ubRowStrideFp32 / FP32_ELEMS_PER_BLOCK);
        for (uint32_t s = 0; s < segCnt; s++) {
            uint32_t off = s * FP32_ELEMS_PER_REPEAT;
            uint32_t len = (curTileN - off > FP32_ELEMS_PER_REPEAT) ? FP32_ELEMS_PER_REPEAT : (curTileN - off);
            if (len == FP32_ELEMS_PER_REPEAT) {
                Add<float>(matmulFp32_[off], matmulFp32_[off], bias[off], (uint64_t)FP32_ELEMS_PER_REPEAT,
                           (uint8_t)curTileM, BinaryRepeatParams(1, 1, 1, repStrideFp32, repStrideFp32, 0));
            } else {
                for (uint32_t r = 0; r < curTileM; r++) {
                    uint64_t rowOff = static_cast<uint64_t>(r) * ubRowStrideFp32 + off;
                    Add<float>(matmulFp32_[rowOff], matmulFp32_[rowOff], bias[off], (int32_t)len);
                }
            }
        }

        // Cast float→half, 写入 matmulOut_[bufIdx] (等上上轮 CopyOut 释放)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copyOutDoneEvent_[bufIdx]);
        if (rowContiguous) {
            Cast<MatmulDataType, float>(matmulOut_[bufIdx], matmulFp32_, RoundMode::CAST_ROUND,
                                        (int32_t)(curTileM * curTileN));
        } else {
            for (uint32_t r = 0; r < curTileM; r++) {
                Cast<MatmulDataType, float>(matmulOut_[bufIdx][r * ubRowStride], matmulFp32_[r * ubRowStrideFp32],
                                            RoundMode::CAST_ROUND, (int32_t)curTileN);
            }
        }
    }

    __aicore__ inline void ComputeDirect(const LocalTensor<float> &bias, uint32_t curTileM, uint32_t curTileN,
                                         uint32_t ubRowStride, uint32_t bufIdx)
    {
        LocalTensor<float> matmulFp32 = matmulTile_[bufIdx].template ReinterpretCast<float>();
        uint32_t segCnt = Ceil(curTileN, FP32_ELEMS_PER_REPEAT);
        uint8_t repStride = static_cast<uint8_t>(ubRowStride / FP32_ELEMS_PER_BLOCK);
        for (uint32_t s = 0; s < segCnt; s++) {
            uint32_t off = s * FP32_ELEMS_PER_REPEAT;
            uint32_t len = (curTileN - off > FP32_ELEMS_PER_REPEAT) ? FP32_ELEMS_PER_REPEAT : (curTileN - off);
            if (len == FP32_ELEMS_PER_REPEAT) {
                Add<float>(matmulFp32[off], matmulFp32[off], bias[off], (uint64_t)FP32_ELEMS_PER_REPEAT,
                           (uint8_t)curTileM, BinaryRepeatParams(1, 1, 1, repStride, repStride, 0));
            } else {
                for (uint32_t r = 0; r < curTileM; r++) {
                    uint64_t rowOff = static_cast<uint64_t>(r) * ubRowStride + off;
                    Add<float>(matmulFp32[rowOff], matmulFp32[rowOff], bias[off], (int32_t)len);
                }
            }
        }
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t curTileM, uint32_t curTileN, uint32_t bufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(computeEvent_);
        uint32_t blockLen = curTileN * sizeof(MatmulDataType);
        DataCopyExtParams params{static_cast<uint16_t>(curTileM), blockLen, 0,
                                 static_cast<int64_t>((context_.N - curTileN) * sizeof(MatmulDataType)), 0};
        if constexpr (NEED_FP32_COMPUTE) {
            DataCopyPad(matmulGm_[offset], matmulOut_[bufIdx], params);
        } else {
            DataCopyPad(matmulGm_[offset], matmulTile_[bufIdx], params);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copyOutDoneEvent_[bufIdx]);
    }

    __aicore__ inline void ProcessMTile(uint64_t offset, uint32_t curTileM, uint32_t curTileN, uint32_t ubRowStride,
                                        const LocalTensor<float> &bias, uint32_t bufIdx)
    {
        CopyIn(offset, curTileM, curTileN, bufIdx);
        Compute(bias, curTileM, curTileN, ubRowStride, bufIdx);
        CopyOut(offset, curTileM, curTileN, bufIdx);
    }

    // --- MN_SPLIT: tile 索引迭代 ---

    __aicore__ inline void ProcessNTile(uint32_t nTile, uint32_t biasBufIdx, uint32_t nextBiasBufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(biasInEvent_);
        uint32_t curTileN = TileSize(nTile, tileNCnt_, tailN_, TILE_N);
        LocalTensor<float> bias = PrepareBias(biasBufIdx, curTileN);
        if constexpr (NEED_BIAS_CAST) {
            if (nTile + 1 < nTileEnd_) {
                LoadBiasTile(nTile + 1, nextBiasBufIdx);
            }
        }
        uint32_t ubRowStride = Ceil(curTileN, ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;
        TileRange mr = GetMRange(nTile);
        uint32_t matmulBufIdx = 0;
        for (uint64_t i = 0; i < mr.count; i++) {
            uint32_t mTile = static_cast<uint32_t>(i * mr.step + mr.start);
            if (mTile >= tileMCnt_) {
                break;
            }
            uint32_t curTileM = TileSize(mTile, tileMCnt_, tailM_, TILE_M);
            uint64_t offset =
                static_cast<uint64_t>(mTile) * TILE_M * context_.N + static_cast<uint64_t>(nTile) * TILE_N;
            ProcessMTile(offset, curTileM, curTileN, ubRowStride, bias, matmulBufIdx);
            matmulBufIdx ^= 1;
        }
        if constexpr (!NEED_BIAS_CAST) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasCastDoneEvent_);
            if (nTile + 1 < nTileEnd_) {
                LoadBiasTile(nTile + 1, nextBiasBufIdx);
            }
        }
    }

    // --- M_SPLIT / N_SPLIT: 数据维度迭代 ---

    __aicore__ inline void ProcessRange()
    {
        uint32_t biasBufIdx = 0;
        uint32_t firstN = (myNEnd_ - myNStart_ < TILE_N) ? (myNEnd_ - myNStart_) : TILE_N;
        LoadBias(myNStart_, firstN, biasBufIdx);

        for (uint32_t nPos = myNStart_; nPos < myNEnd_; nPos += TILE_N) {
            uint32_t curTileN = (nPos + TILE_N > myNEnd_) ? (myNEnd_ - nPos) : TILE_N;
            uint32_t nextBiasBufIdx = biasBufIdx ^ 1;

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(biasInEvent_);

            LocalTensor<float> bias = PrepareBias(biasBufIdx, curTileN);
            uint32_t nextNPos = nPos + TILE_N;
            if constexpr (NEED_BIAS_CAST) {
                if (nextNPos < myNEnd_) {
                    uint32_t nextLen = (nextNPos + TILE_N > myNEnd_) ? (myNEnd_ - nextNPos) : TILE_N;
                    LoadBias(nextNPos, nextLen, nextBiasBufIdx);
                }
            }
            uint32_t ubRowStride = Ceil(curTileN, ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;

            uint32_t matmulBufIdx = 0;
            for (uint32_t mPos = myMStart_; mPos < myMEnd_; mPos += TILE_M) {
                uint32_t curTileM = (mPos + TILE_M > myMEnd_) ? (myMEnd_ - mPos) : TILE_M;
                uint64_t offset = static_cast<uint64_t>(mPos) * context_.N + nPos;
                ProcessMTile(offset, curTileM, curTileN, ubRowStride, bias, matmulBufIdx);
                matmulBufIdx ^= 1;
            }
            if constexpr (!NEED_BIAS_CAST) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasCastDoneEvent_);
                if (nextNPos < myNEnd_) {
                    uint32_t nextLen = (nextNPos + TILE_N > myNEnd_) ? (myNEnd_ - nextNPos) : TILE_N;
                    LoadBias(nextNPos, nextLen, nextBiasBufIdx);
                }
            }

            biasBufIdx = nextBiasBufIdx;
        }
    }
};

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline MC2AddBiasContext *MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::GetContextPtr()
{
    return &context_;
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Process(uint32_t taskIndex)
{
    Init(context_.matmulAddr + static_cast<uint64_t>(taskIndex) * context_.matmulOffset,
         context_.biasAddr + static_cast<uint64_t>(taskIndex) * context_.biasOffset);
    Process();
    Destroy();
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Init(GM_ADDR matmulAddr,
                                                                                       GM_ADDR biasAddr)
{
    tPipe_->Reset();
    CalcSplit();
    matmulGm_.SetGlobalBuffer((__gm__ MatmulDataType *)matmulAddr);
    biasGm_.SetGlobalBuffer((__gm__ BiasDataType *)biasAddr);

    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        tPipe_->InitBuffer(matmulTileBuf_[i],
                           static_cast<uint32_t>(static_cast<uint64_t>(TILE_M) * TILE_N * sizeof(MatmulDataType)));
        tPipe_->InitBuffer(biasTileBuf_[i],
                           static_cast<uint32_t>(static_cast<uint64_t>(TILE_N) * sizeof(BiasDataType)));
        matmulTile_[i] = matmulTileBuf_[i].template Get<MatmulDataType>();
        biasTile_[i] = biasTileBuf_[i].template Get<BiasDataType>();
    }
    if constexpr (NEED_FP32_COMPUTE) {
        tPipe_->InitBuffer(matmulFp32Buf_,
                           static_cast<uint32_t>(static_cast<uint64_t>(TILE_M) * TILE_N * sizeof(float)));
        matmulFp32_ = matmulFp32Buf_.template Get<float>();
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            tPipe_->InitBuffer(matmulOutBuf_[i],
                               static_cast<uint32_t>(static_cast<uint64_t>(TILE_M) * TILE_N * sizeof(MatmulDataType)));
            matmulOut_[i] = matmulOutBuf_[i].template Get<MatmulDataType>();
        }
    }
    if constexpr (NEED_BIAS_CAST) {
        tPipe_->InitBuffer(biasFp32Buf_, static_cast<uint32_t>(static_cast<uint64_t>(TILE_N) * sizeof(float)));
    }
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Process()
{
    if (static_cast<uint32_t>(GetBlockIdx()) >= usedCoreNum_) {
        return;
    }

    copyInEvent_ = tPipe_->template AllocEventID<AscendC::HardEvent::MTE2_S>();
    computeEvent_ = tPipe_->template AllocEventID<AscendC::HardEvent::V_MTE3>();
    biasInEvent_ = tPipe_->template AllocEventID<AscendC::HardEvent::MTE2_S>();
    biasCastDoneEvent_ = tPipe_->template AllocEventID<AscendC::HardEvent::V_MTE2>();
    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        copyOutDoneEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::MTE3_S>();
        castDoneEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::V_MTE2>();
        if constexpr (NEED_FP32_COMPUTE) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(castDoneEvent_[i]);    // CopyIn 首轮通过
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copyOutDoneEvent_[i]); // CastBack 首轮通过
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copyOutDoneEvent_[i]); // CopyIn 首轮通过
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasCastDoneEvent_); // LoadBias 首轮通过

    if constexpr (MODE == SplitMode::MN_SPLIT) {
        if (nTileStart_ < nTileEnd_) {
            uint32_t biasBufIdx = 0;
            LoadBiasTile(nTileStart_, biasBufIdx);
            for (uint32_t n = nTileStart_; n < nTileEnd_; n++) {
                uint32_t nextBiasBufIdx = biasBufIdx ^ 1;
                ProcessNTile(n, biasBufIdx, nextBiasBufIdx);
                biasBufIdx = nextBiasBufIdx;
            }
        }
    } else {
        if (myMStart_ < myMEnd_ && myNStart_ < myNEnd_) {
            ProcessRange();
        }
    }

    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        if constexpr (NEED_FP32_COMPUTE) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(castDoneEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copyOutDoneEvent_[i]);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copyOutDoneEvent_[i]);
        }
        tPipe_->template ReleaseEventID<AscendC::HardEvent::MTE3_S>(copyOutDoneEvent_[i]);
        tPipe_->template ReleaseEventID<AscendC::HardEvent::V_MTE2>(castDoneEvent_[i]);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(biasCastDoneEvent_);
    tPipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_S>(copyInEvent_);
    tPipe_->template ReleaseEventID<AscendC::HardEvent::V_MTE3>(computeEvent_);
    tPipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_S>(biasInEvent_);
    tPipe_->template ReleaseEventID<AscendC::HardEvent::V_MTE2>(biasCastDoneEvent_);
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Destroy()
{}

#ifndef DEFINE_MC2_ADD_BIAS_FOR_MATH_COMPUTATION
#define DEFINE_MC2_ADD_BIAS_FOR_MATH_COMPUTATION(MatmulDataType, BiasDataType, SplitModeVal, AddBiasType) \
    using AddBiasType = MC2KernelTemplate::MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>
#endif

} // namespace MC2KernelTemplate

#endif
