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
#include "../../hc_vector_fallback.hpp"

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../hc_gm_copy.hpp"

namespace Catlass::Epilogue {

struct EpilogueHcV2GateMean {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = 1;
};

template <class Element, class ArchTag = Arch::AtlasA2>
class BlockEpilogueHcV2 {
public:
    CATLASS_DEVICE
    BlockEpilogueHcV2() {}

    // up / xn / mixed 基址只绑一次。大 shape 的 L2 hint 在这里设置，小 shape 不关 L2。
    CATLASS_DEVICE
    void Init(GM_ADDR gmUp, GM_ADDR gmXn, GM_ADDR gmMixed, bool l2HintUp, bool l2HintXn)
    {
        using namespace AscendC;
        gUp_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmUp));
        if (l2HintUp) {
            gUp_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        gXn_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmXn));
        if (l2HintXn) {
            gXn_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        gMixed_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmMixed));
    }

    template <class Resource>
    CATLASS_DEVICE void operator()(Resource& resource, int64_t offUp, int64_t offXn, int64_t offMixed, uint32_t actualM,
                                   uint32_t channels, uint32_t hidden, bool disableSubSplit = false)
    {
        using namespace AscendC;
        if (actualM == 0 || hidden == 0 || channels == 0) {
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
        const float invC = 1.0f / static_cast<float>(static_cast<int32_t>(channels));
        const uint32_t d = channels * hidden;

        GlobalTensor<Element> gUp = gUp_[offUp];
        GlobalTensor<Element> gXn = gXn_[offXn];
        GlobalTensor<Element> gMixed = gMixed_[offMixed];

        if (static_cast<uint64_t>(d) * 14u + static_cast<uint64_t>(hidden) * 8u + 256u > ArchTag::UB_SIZE) {
            HcV2Striped<Element>(resource, gUp, gXn, gMixed, rowStart, rowEnd, channels, hidden);
            return;
        }
        uint32_t off = 0;
        LocalTensor<uint8_t> ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        LocalTensor<float> ubUp = ub[off].template ReinterpretCast<float>();
        off += d * sizeof(float) + 32u;
        off = (off + 31u) & ~31u;
        LocalTensor<Element> ubXnT = ub[off].template ReinterpretCast<Element>();
        off += d * sizeof(Element) + 32u;
        off = (off + 31u) & ~31u;
        LocalTensor<float> ubXn;
        if constexpr (std::is_same_v<Element, float>) {
            ubXn = ubXnT.template ReinterpretCast<float>();
        } else {
            ubXn = ub[off].template ReinterpretCast<float>();
            off += d * sizeof(float);
            off = (off + 31u) & ~31u;
        }
        LocalTensor<float> ubTmp = ub[off].template ReinterpretCast<float>();
        off += d * sizeof(float);
        off = (off + 31u) & ~31u;
        LocalTensor<float> ubOnes = ub[off].template ReinterpretCast<float>();
        off += 8u * sizeof(float);
        off = (off + 31u) & ~31u;
        // 输出双缓冲：MTE3 读一份时，下一行写入另一份。计算用的 ubMixed 在 BF16 上仍是单缓冲，
        // Cast 读完就可以被下一行复用。
        constexpr uint32_t kOutStages = 2;
        uint32_t mixedBytes = (hidden * static_cast<uint32_t>(sizeof(float)) + 31u) & ~31u;
        uint32_t mixedStride = mixedBytes / static_cast<uint32_t>(sizeof(float));
        LocalTensor<float> ubMixed = ub[off].template ReinterpretCast<float>();
        off += mixedBytes * (std::is_same_v<Element, float> ? kOutStages : 1u);
        off = (off + 31u) & ~31u;
        uint32_t outBytes = (hidden * static_cast<uint32_t>(sizeof(Element)) + 31u) & ~31u;
        uint32_t outStride = outBytes / static_cast<uint32_t>(sizeof(Element));
        LocalTensor<Element> ubMixedT = ub[off].template ReinterpretCast<Element>();
        if constexpr (!std::is_same_v<Element, float>) {
            off += outBytes * kOutStages;
        }
        (void)off;
        (void)mixedStride;
        (void)outStride;

        if (rowEnd <= rowStart) {
            return;
        }
        Duplicate(ubOnes, 1.0f, 8);
        PipeBarrier<PIPE_V>();
        // 输出槽用事件数组做 ping-pong。进入循环前先 Set 两次，循环里按 i&1 取槽，不再写 if。
        static constexpr uint32_t kOutEv[2] = {EVENT_ID0, EVENT_ID1};
        SetFlag<HardEvent::MTE3_V>(kOutEv[0]);
        SetFlag<HardEvent::MTE3_V>(kOutEv[1]);

        DataCopyExtParams upParams;
        upParams.blockCount = 1;
        upParams.blockLen = d * static_cast<uint32_t>(sizeof(Element));
        upParams.srcStride = 0;
        upParams.dstStride = 0;
        DataCopyPadExtParams<Element> upPad{false, 0, 0, static_cast<Element>(0)};
        DataCopyExtParams xnParams;
        xnParams.blockCount = 1;
        xnParams.blockLen = d * static_cast<uint32_t>(sizeof(Element));
        xnParams.srcStride = 0;
        xnParams.dstStride = 0;
        DataCopyPadExtParams<Element> xnPad{false, 0, 0, static_cast<Element>(0)};
        DataCopyExtParams mixedParams;
        mixedParams.blockCount = 1;
        mixedParams.blockLen = hidden * static_cast<uint32_t>(sizeof(Element));
        mixedParams.srcStride = 0;
        mixedParams.dstStride = 0;

        PipeBarrier<PIPE_ALL>();
        for (uint32_t m = rowStart; m < rowEnd; ++m) {
            uint32_t stage = (m - rowStart) & 1u;
            event_t slotEv = static_cast<event_t>(kOutEv[stage]);
            uint32_t mNext = m + 1u;
            bool hasNext = mNext < rowEnd;
            if (m == rowStart) {
                DataCopyPad(ubUp.template ReinterpretCast<Element>(), gUp[static_cast<int64_t>(m) * d], upParams,
                            upPad);
                DataCopyPad(ubXnT, gXn[static_cast<int64_t>(m) * d], xnParams, xnPad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            } else {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
            }
            if constexpr (std::is_same_v<Element, float>) {
                WaitFlag<HardEvent::MTE3_V>(slotEv);
            } else {
                Cast(ubXn, ubXnT, RoundMode::CAST_NONE, d);
                if (hasNext) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    DataCopyPad(ubXnT, gXn[static_cast<int64_t>(mNext) * d], xnParams, xnPad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
                }
            }
            // 取负之后 up 不再被读。下一条 up 从这里开始搬，和后面的 Exp/Div 重叠。
            // Expand the Cube output in the existing float scratch.
            if constexpr (!std::is_same_v<Element, float>) {
                Cast(ubTmp, ubUp.template ReinterpretCast<Element>(), RoundMode::CAST_NONE, d);
                PipeBarrier<PIPE_V>();
                DataCopy(ubUp, ubTmp, d);
                PipeBarrier<PIPE_V>();
            }
            Muls(ubTmp, ubUp, -1.0f, d);
            if (hasNext) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                DataCopyPad(ubUp.template ReinterpretCast<Element>(), gUp[static_cast<int64_t>(mNext) * d], upParams,
                            upPad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
            }
            LocalTensor<float> mixedRow = ubMixed;
            if constexpr (std::is_same_v<Element, float>) {
                mixedRow = ubMixed[stage * mixedStride];
            }
            SigmoidMulAcc(mixedRow, ubOnes, ubXn, ubTmp, channels, hidden, invC);
            if constexpr (std::is_same_v<Element, float>) {
                if (hasNext) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    DataCopyPad(ubXnT, gXn[static_cast<int64_t>(mNext) * d], xnParams, xnPad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
                }
                SetFlag<HardEvent::V_MTE3>(slotEv);
                WaitFlag<HardEvent::V_MTE3>(slotEv);
                DataCopyPad(gMixed[static_cast<int64_t>(m) * hidden], mixedRow, mixedParams);
            } else {
                WaitFlag<HardEvent::MTE3_V>(slotEv);
                LocalTensor<Element> mixedT = ubMixedT[stage * outStride];
                Cast(mixedT, mixedRow, RoundMode::CAST_RINT, hidden);
                SetFlag<HardEvent::V_MTE3>(slotEv);
                WaitFlag<HardEvent::V_MTE3>(slotEv);
                DataCopyPad(gMixed[static_cast<int64_t>(m) * hidden], mixedT, mixedParams);
            }
            SetFlag<HardEvent::MTE3_V>(slotEv);
        }
        WaitFlag<HardEvent::MTE3_V>(kOutEv[0]);
        WaitFlag<HardEvent::MTE3_V>(kOutEv[1]);
        PipeBarrier<PIPE_ALL>();
    }

private:
    // ubTmp 进入时已经是 -up。ones 是 1 个 block 的 1，Counter 模式下 src0 步长为 0。
    CATLASS_DEVICE
    static void SigmoidMulAcc(AscendC::LocalTensor<float> const& mixedRow, AscendC::LocalTensor<float> const& ubOnes,
                              AscendC::LocalTensor<float> const& ubXn, AscendC::LocalTensor<float> const& ubTmp,
                              uint32_t channels, uint32_t hidden, float invC)
    {
        using namespace AscendC;
        const uint32_t d = channels * hidden;
        Exp(ubTmp, ubTmp, d);
        PipeBarrier<PIPE_V>();
        Adds(ubTmp, ubTmp, 1.0f, d);
        PipeBarrier<PIPE_V>();
        // Counter 一次 Div：mask=d，repeat=1。src0 是 1 个 block 的 1，blk/rep 步长为 0。
        const BinaryRepeatParams divRep{1, 0, 1, 8, 0, 8};
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(d);
        Div<float, false>(ubTmp, ubOnes, ubTmp, MASK_PLACEHOLDER, 1, divRep);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        Mul(ubTmp, ubTmp, ubXn, d);
        PipeBarrier<PIPE_V>();
        Muls(ubTmp, ubTmp, invC, d);
        PipeBarrier<PIPE_V>();
        Duplicate(mixedRow, 0.0f, hidden);
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < channels; ++c) {
            uint32_t off = c * hidden;
            Add(mixedRow, mixedRow, ubTmp[off], hidden);
            PipeBarrier<PIPE_V>();
        }
    }

    AscendC::GlobalTensor<Element> gUp_;
    AscendC::GlobalTensor<Element> gXn_;
    AscendC::GlobalTensor<Element> gMixed_;
};

} // namespace Catlass::Epilogue
