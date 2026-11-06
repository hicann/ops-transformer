/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "../hc_gm_copy.hpp"

namespace Catlass::Epilogue {

struct EpilogueHcV0Rms {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = 1;
};

template <class Element, class ArchTag = Arch::AtlasA2>
class BlockEpilogueHcV0 {
public:
    using ElementCompute = float;
    static constexpr uint32_t TMP_ELEMS = 512;

    CATLASS_DEVICE
    BlockEpilogueHcV0() {}

    // GlobalTensor 只在 kernel 入口绑一次。后续按元素偏移取视图，L2 hint 随偏移保留。
    CATLASS_DEVICE
    void Init(GM_ADDR gmX, GM_ADDR gmXn, GM_ADDR gmGamma, GM_ADDR gmInvRms, bool l2HintX, bool l2HintXn)
    {
        using namespace AscendC;
        gX_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmX));
        if (l2HintX) {
            gX_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        gXn_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmXn));
        // xn 写出后只被本核 c0、v2 各读一次。大 shape 与 v2 一样不占 L2。
        if (l2HintXn) {
            gXn_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        hasGamma_ = gmGamma != nullptr;
        if (hasGamma_) {
            gGamma_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmGamma));
        }
        hasInv_ = gmInvRms != nullptr;
        if (hasInv_) {
            gInvRms_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmInvRms));
        }
    }

    template <class Resource>
    CATLASS_DEVICE void operator()(Resource& resource, int64_t offX, int64_t offXn, uint32_t actualM, uint32_t d,
                                   uint32_t channels, uint32_t hidden, float eps, bool useCombine,
                                   bool disableSubSplit = false, int64_t offInv = 0)
    {
        using namespace AscendC;
        if (actualM == 0) {
            return;
        }
        const uint32_t subNum = GetSubBlockNum() == 0 ? 1 : GetSubBlockNum();
        const uint32_t subIdx = GetSubBlockIdx();
        const uint32_t row0 = (actualM + subNum - 1) / subNum;
        uint32_t rowStart = subIdx * row0;
        uint32_t rowEnd = rowStart + row0 < actualM ? rowStart + row0 : actualM;
        if (disableSubSplit) {
            rowStart = 0;
            rowEnd = actualM;
        }
        (void)useCombine;
        GlobalTensor<Element> gX = gX_[offX];
        GlobalTensor<Element> gXn = gXn_[offXn];
        GlobalTensor<float> gGamma;
        if (hasGamma_) {
            gGamma = gGamma_;
        }
        GlobalTensor<float> gInvRms;
        if (hasInv_) {
            gInvRms = gInvRms_[offInv];
        }

        // A2 与 nn 仓 KernelRmsNorm NORMAL 一致：UB 按一个通道（一行 H）划分，不按整行 D。
        // inQueueX(H) + gamma(H) + outQueueY(H) + rstd(64) + x_fp32(H) + sqx(H) + reduce(64)。
        uint32_t off = 0;
        LocalTensor<uint8_t> ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        auto align32 = [](uint32_t n) { return (n + 31u) & ~31u; };
        uint32_t colBytesT = align32(hidden * static_cast<uint32_t>(sizeof(Element)));
        uint32_t colBytesF = align32(hidden * static_cast<uint32_t>(sizeof(float)));
        LocalTensor<Element> ubRowT = ub[off].template ReinterpretCast<Element>();
        LocalTensor<float> ubRowF;
        LocalTensor<float> ubXFp32;
        if constexpr (std::is_same_v<Element, float>) {
            ubRowF = ub[off + colBytesT].template ReinterpretCast<float>();
            ubXFp32 = ubRowT;
            off += colBytesT;
        } else {
            off += colBytesT;
            ubXFp32 = ub[off].template ReinterpretCast<float>();
            off += colBytesF;
            ubRowF = ub[off].template ReinterpretCast<float>();
        }
        off += colBytesF;
        LocalTensor<Element> ubY;
        ubY = ub[off].template ReinterpretCast<Element>();
        off += colBytesT;
        // 第二套输入/输出，让下一通道的 MTE2 叠在当前通道的 V 上，和 nn 仓队列腾出 x 之后的行为一致。
        LocalTensor<Element> ubIn1 = ub[off].template ReinterpretCast<Element>();
        off += colBytesT;
        LocalTensor<Element> ubOut1 = ub[off].template ReinterpretCast<Element>();
        off += colBytesT;
        LocalTensor<float> ubMean = ub[off].template ReinterpretCast<float>();
        off += 64 * sizeof(float);
        LocalTensor<float> ubTmp = ub[off].template ReinterpretCast<float>();
        off += 64 * sizeof(float);
        LocalTensor<float> ubGamma;
        bool gammaHoisted = false;
        if (hasGamma_) {
            // gamma 很小（C×H×4B），一次搬进 UB 常驻复用，省掉逐行逐通道的 GM 重搬与 +1。
            // 超大 C×H 放不下 UB 时退回逐通道搬运的旧路径。
            uint32_t gammaAllBytes = align32(channels * hidden * static_cast<uint32_t>(sizeof(float)));
            gammaHoisted = gammaAllBytes <= 65536u;
            ubGamma = ub[off].template ReinterpretCast<float>();
            off += gammaHoisted ? gammaAllBytes : align32(hidden * static_cast<uint32_t>(sizeof(float)));
        }
        // inv_rms 批量落盘：WholeReduceSum 直写非对齐紧凑槽攒「原始 sum」（替代 SetValue），
        // 攒满一批（最多 256 个 float）后先整批向量算 inv 再一次 DataCopy 落 GM，
        // 替代原来每行一次的 DataCopyPad + flag 往返。
        // gmInvRms 为空时不写不落盘。
        constexpr uint32_t ACC_MAX_ELEMS = 256;
        uint32_t accRows = hasInv_ && channels > 0 ? (ACC_MAX_ELEMS / channels > 0 ? ACC_MAX_ELEMS / channels : 1) : 1;
        uint32_t accElems = accRows * channels;
        LocalTensor<float> ubAcc = ub[off].template ReinterpretCast<float>();
        off += align32((hasInv_ && channels > 0 ? accElems : channels) * static_cast<uint32_t>(sizeof(float)));
        // 向量 inv 链的常数 1.0（Div 分子，整批大小时用）与 Brcb 广播块（1 个 32B block）。
        uint32_t onesElems = (hasInv_ && channels > 0) ? accElems : 8u;
        if (onesElems < 8u) {
            onesElems = 8u;
        }
        LocalTensor<float> ubOnes = ub[off].template ReinterpretCast<float>();
        off += align32(onesElems * static_cast<uint32_t>(sizeof(float)));
        LocalTensor<float> ubBrc = ub[off].template ReinterpretCast<float>();
        // Brcb repeat=1 读 1 个源 block（8 元素）、写 8 个目的 block（每元素独占一个 block）。
        off += 8u * 32u;
        Duplicate(ubOnes, 1.0f, onesElems);
        PipeBarrier<PIPE_V>();

        if (hasGamma_ && gammaHoisted) {
            for (uint32_t c = 0; c < channels; ++c) {
                DataCopyExtParams gammaParams{1, hidden * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> gammaPad{false, 0, 0, 0.f};
                DataCopyPad(ubGamma[c * hidden], gGamma[static_cast<int64_t>(c) * hidden], gammaParams, gammaPad);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
            Adds(ubGamma, ubGamma, 1.0f, channels * hidden);
            PipeBarrier<PIPE_V>();
        }

        // 对齐 nn 仓 KernelRmsNorm::Compute：Cast 一次、count 模式 Mul/Add 做半区间归约、
        // Muls 乘 rstd、再 Cast 回原 dtype。一行对齐时一次 DataCopy，不再按 32 字节拆开搬。
        // 行/通道循环里不变的量统一在循环外算一次：归约分解（body/tail 只由 hidden 决定）、
        // 搬运参数结构体、1/hidden；行内所有偏移都是 rowBytes 的整数倍，
        // 逐通道的对齐判定结果恒等于 rowBytesAligned，也一并外提。
        constexpr int32_t kRep = 64;
        auto powerTwo = [](int32_t n1) {
            n1 |= n1 >> 1;
            n1 |= n1 >> 2;
            n1 |= n1 >> 4;
            n1 |= n1 >> 8;
            n1 |= n1 >> 16;
            return (n1 + 1) >> 1;
        };
        const uint32_t rowBytes = hidden * static_cast<uint32_t>(sizeof(Element));
        const bool rowBytesAligned = (rowBytes & 31u) == 0u;
        const int32_t count = static_cast<int32_t>(hidden);
        const int32_t body0 = count > kRep ? powerTwo(count) : count;
        const int32_t tail0 = count > kRep ? count - body0 : 0;
        const uint64_t maskCount = count > kRep ? static_cast<uint64_t>(kRep) : static_cast<uint64_t>(count);
        const float invHidden = count > 0 ? 1.0f / static_cast<float>(count) : 0.0f;
        DataCopyExtParams xParams;
        xParams.blockCount = 1;
        xParams.blockLen = rowBytes;
        xParams.srcStride = 0;
        xParams.dstStride = 0;
        DataCopyPadExtParams<Element> xPad{false, 0, 0, static_cast<Element>(0)};
        DataCopyExtParams gammaParams{1, hidden * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> gammaPad{false, 0, 0, 0.f};
        LocalTensor<Element> inSlot[2] = {ubRowT, ubIn1};
        LocalTensor<Element> outSlot[2] = {ubY, ubOut1};

        uint32_t pendingRows = 0;      // 自上次落盘以来攒下的行数
        uint32_t flushBase = rowStart; // 攒批首行的全局行号（GM 落盘基址）
        // 输入、输出各两块，按 m*C+c 单调 ping-pong。进循环前各 Set 两次，表示两块都空闲；
        // 下标不在通道边界清零，行末也不把队列排空。下一格可以落到下一行的 c=0。
        static constexpr uint32_t kEv[2] = {EVENT_ID0, EVENT_ID1};
        SetFlag<HardEvent::V_MTE2>(kEv[0]);
        SetFlag<HardEvent::V_MTE2>(kEv[1]);
        SetFlag<HardEvent::MTE3_V>(kEv[0]);
        SetFlag<HardEvent::MTE3_V>(kEv[1]);
        if (channels > 0 && rowStart < rowEnd) {
            uint32_t headSlot = (rowStart * channels) & 1u;
            event_t headEv = static_cast<event_t>(kEv[headSlot]);
            WaitFlag<HardEvent::V_MTE2>(headEv);
            if (rowBytesAligned) {
                DataCopy(inSlot[headSlot], gX[static_cast<int64_t>(rowStart) * d], hidden);
            } else {
                DataCopyPad(inSlot[headSlot], gX[static_cast<int64_t>(rowStart) * d], xParams, xPad);
            }
            SetFlag<HardEvent::MTE2_V>(headEv);
        }
        const BinaryRepeatParams mulBrcParams{1, 1, 0, 8, 8, 0};
        uint32_t c = 0;
        uint32_t m = rowStart;
        const uint32_t mcBegin = rowStart * channels;
        const uint32_t mcEnd = rowEnd * channels;
        for (uint32_t mc = mcBegin; mc < mcEnd; ++mc) {
            uint32_t slot = mc & 1u;
            event_t ev = static_cast<event_t>(kEv[slot]);
            WaitFlag<HardEvent::MTE2_V>(ev);
            bool hasNext = mc + 1u < mcEnd;
            if (hasNext) {
                uint32_t nextC = c + 1u;
                uint32_t nextM = m;
                if (nextC == channels) {
                    nextC = 0;
                    nextM = m + 1u;
                }
                uint32_t nxt = slot ^ 1u;
                event_t evN = static_cast<event_t>(kEv[nxt]);
                WaitFlag<HardEvent::V_MTE2>(evN);
                int64_t nxtOff = static_cast<int64_t>(nextM) * d + static_cast<int64_t>(nextC) * hidden;
                if (rowBytesAligned) {
                    DataCopy(inSlot[nxt], gX[nxtOff], hidden);
                } else {
                    DataCopyPad(inSlot[nxt], gX[nxtOff], xParams, xPad);
                }
                SetFlag<HardEvent::MTE2_V>(evN);
            }
            LocalTensor<Element> curIn = inSlot[slot];
            if constexpr (std::is_same_v<Element, float>) {
                Mul(ubRowF, curIn, curIn, hidden);
            } else {
                // bf16 下这是 inSlot 的最后一次向量读，之后只碰 ubXFp32 / ubRowF。
                Cast(ubXFp32, curIn, RoundMode::CAST_NONE, hidden);
                SetFlag<HardEvent::V_MTE2>(ev);
                PipeBarrier<PIPE_V>();
                Mul(ubRowF, ubXFp32, ubXFp32, hidden);
            }
            PipeBarrier<PIPE_V>();
            if (count > kRep) {
                if (tail0 > 0) {
                    Add(ubRowF, ubRowF, ubRowF[body0], tail0);
                    PipeBarrier<PIPE_V>();
                }
                int32_t body = body0;
                while (body > kRep) {
                    body /= 2;
                    Add(ubRowF, ubRowF, ubRowF[body], body);
                    PipeBarrier<PIPE_V>();
                }
            }
            AscendCUtils::SetMask<float>(maskCount);
            if (g_coreType == AIV) {
                // 原始 sum 只归约一次：WholeReduceSum 直写非对齐紧凑槽 ubAcc（dst 可为任意
                // 元素偏移，同 block_attention_residuals 的 vecMeta_[metaIdx] 用法），替代 SetValue。
                // 无 backward（不落盘）时直写对齐 ubTmp，省一次回拷。
                if (hasInv_) {
                    WholeReduceSum<float, false>(ubAcc[pendingRows * channels + c], ubRowF, static_cast<uint64_t>(kRep),
                                                 1, 0, 1, 0);
                } else {
                    WholeReduceSum<float, false>(ubTmp, ubRowF, static_cast<uint64_t>(kRep), 1, 0, 1, 0);
                }
            }
            SetMaskNorm();
            ResetMask();
            PipeBarrier<PIPE_V>();
            // 广播用 inv（第一遍，广播后算）：Brcb 先行——把包含本元素的对齐 block 展开，
            // 展开后一个元素独占一个 block（8 lane 全是该值的重复），本元素落在对齐的
            // ubBrc[lane*8]，之后 Muls/Adds/Sqrt/Div 全部在 block 对齐地址上做（紧凑非对齐
            // 槽做不了这些链式操作）。其余 lane/block 是相邻通道或陈旧数据，链式结果无人读；
            // 块内 8 lane 均为本值副本，链式输出天然就是行缩放 Mul 的广播块。
            uint32_t brcLane = 0;
            if (hasInv_) {
                uint32_t accOff = pendingRows * channels + c;
                brcLane = accOff & 7u;
                Brcb(ubBrc, ubAcc[accOff - brcLane], 1, {1, 8});
            } else {
                Brcb(ubBrc, ubTmp, 1, {1, 8});
            }
            PipeBarrier<PIPE_V>();
            LocalTensor<float> brcBlk = ubBrc[brcLane * 8];
            // inv = 1/sqrt(sum*invHidden+eps)（同 reduce_common_arch22 InvRmsInPlace），
            // Sqrt+Div 精确路径，不判断 sum>0，全程向量无 GetValue/SetValue。
            Muls(brcBlk, brcBlk, invHidden, 8);
            PipeBarrier<PIPE_V>();
            Adds(brcBlk, brcBlk, eps, 8);
            PipeBarrier<PIPE_V>();
            Sqrt(brcBlk, brcBlk, 8);
            PipeBarrier<PIPE_V>();
            Div(brcBlk, ubOnes, brcBlk, 8);
            PipeBarrier<PIPE_V>();
            {
                // 行缩放：src1 以 blkStride/repStride=0 固定复用广播块 brcBlk
                // （同 MulRowByBrcBlock）。COUNTER mask 单次上限 4096，超长行分段。
                constexpr uint32_t kCntrMax = 4096;
                uint32_t rowOff = 0;
                do {
                    uint32_t len = hidden - rowOff < kCntrMax ? hidden - rowOff : kCntrMax;
                    SetMaskCount();
                    SetVectorMask<float, MaskMode::COUNTER>(len);
                    if constexpr (std::is_same_v<Element, float>) {
                        Mul<float, false>(ubRowF[rowOff], curIn[rowOff], brcBlk, MASK_PLACEHOLDER, 1, mulBrcParams);
                    } else {
                        Mul<float, false>(ubRowF[rowOff], ubXFp32[rowOff], brcBlk, MASK_PLACEHOLDER, 1, mulBrcParams);
                    }
                    PipeBarrier<PIPE_V>();
                    rowOff += len;
                } while (rowOff < hidden);
                SetMaskNorm();
                ResetMask();
                if constexpr (std::is_same_v<Element, float>) {
                    // float 的行缩放还在读 curIn，这是 inSlot 的最后一次向量读。
                    SetFlag<HardEvent::V_MTE2>(ev);
                }
            }
            if (hasGamma_) {
                if (gammaHoisted) {
                    // gamma 已在启动时整体搬入 UB 并加过 1.0，直接按通道偏移复用。
                    Mul(ubRowF, ubRowF, ubGamma[c * hidden], hidden);
                    PipeBarrier<PIPE_V>();
                } else {
                    // Reuse one H-element gamma tile; wait before overwriting it.
                    PipeBarrier<PIPE_ALL>();
                    DataCopyPad(ubGamma, gGamma[static_cast<int64_t>(c) * hidden], gammaParams, gammaPad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
                    Adds(ubGamma, ubGamma, 1.0f, hidden);
                    PipeBarrier<PIPE_V>();
                    Mul(ubRowF, ubRowF, ubGamma, hidden);
                    PipeBarrier<PIPE_V>();
                }
            }
            // 同一输出槽上一次 GM 搬运结束后才能改写。两块缓冲随平坦下标轮转，不按行重置。
            WaitFlag<HardEvent::MTE3_V>(ev);
            if constexpr (!std::is_same_v<Element, float>) {
                Cast(outSlot[slot], ubRowF, RoundMode::CAST_RINT, hidden);
                PipeBarrier<PIPE_V>();
            } else {
                DataCopy(outSlot[slot], ubRowF, hidden);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(ev);
            WaitFlag<HardEvent::V_MTE3>(ev);
            int64_t outOff = static_cast<int64_t>(m) * d + static_cast<int64_t>(c) * hidden;
            if (rowBytesAligned) {
                DataCopy(gXn[outOff], outSlot[slot], hidden);
            } else {
                HcStoreGmElems(gXn, outOff, outSlot[slot], hidden, ubMean.template ReinterpretCast<Element>());
            }
            SetFlag<HardEvent::MTE3_V>(ev);
            if (c + 1u == channels) {
                if (hasInv_) {
                    // 攒满 accRows 行（或到达本核末行）再批量落盘：一批最多 256 个 float。
                    ++pendingRows;
                    if (pendingRows == accRows || m + 1u == rowEnd) {
                        uint32_t accNum = pendingRows * channels;
                        // inv 搬出前算第二遍：ubAcc 攒的是原始 sum，这里整批向量算
                        // inv = 1/sqrt(sum*invHidden+eps)，保证落盘缓冲连续、一次搬出。
                        Muls(ubAcc, ubAcc, invHidden, accNum);
                        PipeBarrier<PIPE_V>();
                        Adds(ubAcc, ubAcc, eps, accNum);
                        PipeBarrier<PIPE_V>();
                        Sqrt(ubAcc, ubAcc, accNum);
                        PipeBarrier<PIPE_V>();
                        Div(ubAcc, ubOnes, ubAcc, accNum);
                        PipeBarrier<PIPE_V>();
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                        DataCopyExtParams flushParams;
                        flushParams.blockCount = 1;
                        flushParams.blockLen = accNum * static_cast<uint32_t>(sizeof(float));
                        flushParams.srcStride = 0;
                        flushParams.dstStride = 0;
                        DataCopyPad(gInvRms[static_cast<int64_t>(flushBase) * channels], ubAcc, flushParams);
                        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
                        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                        pendingRows = 0;
                        flushBase = m + 1u;
                    }
                }
                c = 0;
                ++m;
            } else {
                ++c;
            }
        }
        WaitFlag<HardEvent::V_MTE2>(kEv[0]);
        WaitFlag<HardEvent::V_MTE2>(kEv[1]);
        WaitFlag<HardEvent::MTE3_V>(kEv[0]);
        WaitFlag<HardEvent::MTE3_V>(kEv[1]);
    }

private:
    AscendC::GlobalTensor<Element> gX_;
    AscendC::GlobalTensor<Element> gXn_;
    AscendC::GlobalTensor<float> gGamma_;
    AscendC::GlobalTensor<float> gInvRms_;
    bool hasGamma_ = false;
    bool hasInv_ = false;
};

} // namespace Catlass::Epilogue
