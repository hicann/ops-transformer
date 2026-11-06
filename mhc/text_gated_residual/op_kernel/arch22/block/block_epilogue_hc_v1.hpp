/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "../hc_gm_copy.hpp"

namespace Catlass::Epilogue {

// src、dst、work 等长。dst = 2 * sigmoid(src / C)，不读写 GM。
CATLASS_DEVICE void HcInjectCompute(AscendC::LocalTensor<float> const& dst, AscendC::LocalTensor<float> const& src,
                                    AscendC::LocalTensor<float> const& work, uint32_t n, float invC)
{
    using namespace AscendC;
    constexpr uint32_t kStep = 8192;
    for (uint32_t cursor = 0; cursor < n; cursor += kStep) {
        uint32_t len = n - cursor < kStep ? n - cursor : kStep;
        auto d = src[cursor];
        auto o = dst[cursor];
        auto e = work[cursor];
        Muls(d, d, invC, len);
        PipeBarrier<PIPE_V>();
        Muls(e, d, -1.0f, len);
        PipeBarrier<PIPE_V>();
        Exp(e, e, len);
        PipeBarrier<PIPE_V>();
        Adds(e, e, 1.0f, len);
        PipeBarrier<PIPE_V>();
        Duplicate(o, 1.0f, len);
        PipeBarrier<PIPE_V>();
        Div(o, o, e, len);
        PipeBarrier<PIPE_V>();
        Muls(o, o, 2.0f, len);
        PipeBarrier<PIPE_V>();
    }
}

struct EpilogueHcV1DivSilu {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = 2;
};

template <class Element, class ArchTag = Arch::AtlasA2>
class BlockEpilogueHcV1 {
public:
    using ElementCompute = float;
    static constexpr uint32_t COMPUTE_LENGTH = 8192;
    static constexpr uint32_t kUbBudget = 160u * 1024u;

    CATLASS_DEVICE
    BlockEpilogueHcV1() {}

    CATLASS_DEVICE
    void Init(GM_ADDR gmC0Base, GM_ADDR gmDownBase, GM_ADDR gmInjectBase, GM_ADDR gmDownPreActBase,
              bool l2HintDownPreAct, uint32_t channels, uint32_t rDim)
    {
        using namespace AscendC;
        gC0_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmC0Base));
        gGate_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmC0Base));
        gDown_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmDownBase));
        gInj_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmInjectBase));
        hasPreAct_ = gmDownPreActBase != nullptr;
        if (hasPreAct_) {
            gDownPreAct_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmDownPreActBase));
            gC0FromDpa_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmDownPreActBase));
            // 只给反向保存一次，不参与前向复用，不占用 L2。小 shape 该缓冲很小，关闭 L2
            // 属于大 shape 的专属优化，小路径不设置。
            if (l2HintDownPreAct) {
                gDownPreAct_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
        }
        invC_ = channels == 0 ? 0.0f : 1.0f / static_cast<float>(static_cast<int32_t>(channels));
        uint32_t yRowBytes = std::is_same_v<Element, float> ? 0u : rDim * static_cast<uint32_t>(sizeof(Element));
        uint32_t perRow = rDim * static_cast<uint32_t>(sizeof(float)) * 5u + yRowBytes * 2u;
        rowsOnceMax_ = 1;
        if (perRow > 0 && perRow <= kUbBudget) {
            uint32_t fit = kUbBudget / perRow;
            if (fit > 0) {
                rowsOnceMax_ = fit;
            }
        }
    }

    // c0FromPreAct：c0dp 路径下 C0 读入来自 down_pre_act（保持原实现不带 L2 hint），
    // 否则来自 c0 槽工作区。各 offset 均为相对 Init 基址的元素偏移。
    template <class Resource>
    CATLASS_DEVICE void operator()(Resource& resource, int64_t offC0, bool c0FromPreAct, int64_t offDown,
                                   int64_t offInject, uint32_t actualM, uint32_t nWs, uint32_t nConcat, uint32_t rDim,
                                   uint32_t channels, bool useCombine, bool disableSubSplit = false,
                                   bool savePreAct = false, int64_t offDownPreAct = 0, int64_t offGate = 0)
    {
        using namespace AscendC;
        if (actualM == 0 || rDim == 0) {
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
        const uint32_t rows = rowEnd > rowStart ? rowEnd - rowStart : 0;
        if (rows == 0) {
            return;
        }
        (void)nConcat;
        (void)nWs;
        const float invC = invC_;
        const bool doInject = useCombine && channels > 0;
        uint32_t rowsOnce = rowsOnceMax_;
        if (rowsOnce > rows) {
            rowsOnce = rows;
        }

        // 每任务仅做基址 offset 平移（operator[] 返回带偏移的 GlobalTensor 拷贝，
        // L2 hint 编码在地址高位，随偏移保留），不再执行 SetGlobalBuffer。
        GlobalTensor<Element> gC0 = (c0FromPreAct && hasPreAct_) ? gC0FromDpa_[offC0] : gC0_[offC0];
        GlobalTensor<Element> gGate = gGate_[offGate];
        GlobalTensor<Element> gDown = gDown_[offDown];
        GlobalTensor<Element> gDownPreAct;
        if (savePreAct && hasPreAct_) {
            gDownPreAct = gDownPreAct_[offDownPreAct];
        }

        uint32_t slotFloats = rowsOnce * rDim;
        uint32_t slotBytes = (slotFloats * static_cast<uint32_t>(sizeof(float)) + 31u) & ~31u;
        uint32_t slotStride = slotBytes / static_cast<uint32_t>(sizeof(float));
        uint32_t yBytes = 0;
        uint32_t yStride = 0;
        if constexpr (!std::is_same_v<Element, float>) {
            yBytes = (slotFloats * static_cast<uint32_t>(sizeof(Element)) + 31u) & ~31u;
            yStride = yBytes / static_cast<uint32_t>(sizeof(Element));
        }

        LocalTensor<uint8_t> ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        uint32_t offb = 0;
        LocalTensor<float> ubIn = ub[offb].template ReinterpretCast<float>();
        offb += slotBytes * 2u;
        LocalTensor<float> ubOut = ub[offb].template ReinterpretCast<float>();
        offb += slotBytes * 2u;
        LocalTensor<float> work = ub[offb].template ReinterpretCast<float>();
        offb += slotBytes;
        LocalTensor<Element> y = ub[offb].template ReinterpretCast<Element>();
        if constexpr (!std::is_same_v<Element, float>) {
            offb += yBytes * 2u;
        }
        (void)offb;

        DataCopyExtParams inParams;
        inParams.blockCount = 1;
        inParams.srcStride = 0;
        inParams.dstStride = 0;
        DataCopyPadExtParams<Element> inPad{false, 0, 0, static_cast<Element>(0)};
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        SetMaskNorm();
        ResetMask();
        static constexpr uint32_t kEv[2] = {EVENT_ID0, EVENT_ID1};
        SetFlag<HardEvent::V_MTE2>(kEv[0]);
        SetFlag<HardEvent::V_MTE2>(kEv[1]);
        SetFlag<HardEvent::MTE3_V>(kEv[0]);
        SetFlag<HardEvent::MTE3_V>(kEv[1]);
        uint32_t round = 0;
        for (uint32_t base = 0; base < rows; base += rowsOnce) {
            uint32_t stage = round & 1u;
            event_t ev = static_cast<event_t>(kEv[stage]);
            uint32_t nThis = rows - base < rowsOnce ? rows - base : rowsOnce;
            uint32_t nElem = nThis * rDim;
            uint32_t m0 = rowStart + base;
            LocalTensor<float> src = ubIn[stage * slotStride];
            LocalTensor<float> dst = ubOut[stage * slotStride];
            LocalTensor<Element> ySlot = y[stage * yStride];
            WaitFlag<HardEvent::V_MTE2>(ev);
            inParams.blockLen = nElem * static_cast<uint32_t>(sizeof(Element));
            DataCopyPad(src.template ReinterpretCast<Element>(), gC0[static_cast<int64_t>(m0) * rDim], inParams, inPad);
            SetFlag<HardEvent::MTE2_V>(ev);
            WaitFlag<HardEvent::MTE3_V>(ev);
            WaitFlag<HardEvent::MTE2_V>(ev);
            if constexpr (!std::is_same_v<Element, float>) {
                Cast(work, src.template ReinterpretCast<Element>(), RoundMode::CAST_NONE, nElem);
                PipeBarrier<PIPE_V>();
                if ((nElem & 7u) == 0u) {
                    DataCopy(src, work, nElem);
                } else {
                    Adds(src, work, 0.0f, nElem);
                }
                PipeBarrier<PIPE_V>();
            }
            // c0dp 语义：down_pre_act 统一保存 C0 原始输出（未除 C），与大分支 C0 直写一致，
            // 反向自行重算 /C。保存必须在下面 invC 的 Muls 之前做，否则存的是已除 C 的值。
            if (savePreAct) {
                if constexpr (!std::is_same_v<Element, float>) {
                    Cast(ySlot, src, RoundMode::CAST_RINT, nElem);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                outParams.blockLen = nElem * static_cast<uint32_t>(sizeof(Element));
                if constexpr (std::is_same_v<Element, float>) {
                    auto pre = src.template ReinterpretCast<Element>();
                    DataCopyPad(gDownPreAct[static_cast<int64_t>(m0) * rDim], pre, outParams);
                } else {
                    DataCopyPad(gDownPreAct[static_cast<int64_t>(m0) * rDim], ySlot, outParams);
                }
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
            for (uint32_t cursor = 0; cursor < nElem; cursor += COMPUTE_LENGTH) {
                uint32_t n = nElem - cursor < COMPUTE_LENGTH ? nElem - cursor : COMPUTE_LENGTH;
                Muls(src[cursor], src[cursor], invC, n);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t cursor = 0; cursor < nElem; cursor += COMPUTE_LENGTH) {
                uint32_t n = nElem - cursor < COMPUTE_LENGTH ? nElem - cursor : COMPUTE_LENGTH;
                auto d = src[cursor];
                auto o = dst[cursor];
                auto e = work[cursor];
                Muls(e, d, -1.0f, n);
                PipeBarrier<PIPE_V>();
                Exp(e, e, n);
                PipeBarrier<PIPE_V>();
                Adds(e, e, 1.0f, n);
                PipeBarrier<PIPE_V>();
                Div(o, d, e, n);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE2>(ev);
            if constexpr (!std::is_same_v<Element, float>) {
                Cast(ySlot, dst, RoundMode::CAST_RINT, nElem);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(ev);
            WaitFlag<HardEvent::V_MTE3>(ev);
            outParams.blockLen = nElem * static_cast<uint32_t>(sizeof(Element));
            if constexpr (std::is_same_v<Element, float>) {
                auto out = dst.template ReinterpretCast<Element>();
                DataCopyPad(gDown[static_cast<int64_t>(m0) * rDim], out, outParams);
            } else {
                DataCopyPad(gDown[static_cast<int64_t>(m0) * rDim], ySlot, outParams);
            }
            SetFlag<HardEvent::MTE3_V>(ev);
            ++round;
        }
        if (doInject) {
            uint32_t cap = slotFloats / channels;
            if (cap == 0) {
                cap = 1;
            }
            if (cap > 65535u) {
                cap = 65535u;
            }
            GlobalTensor<Element> gInj = gInj_[offInject];
            DataCopyExtParams gateParams;
            gateParams.blockCount = 1;
            gateParams.srcStride = 0;
            gateParams.dstStride = 0;
            for (uint32_t base = 0; base < rows; base += cap) {
                uint32_t stage = round & 1u;
                event_t ev = static_cast<event_t>(kEv[stage]);
                uint32_t nInj = rows - base < cap ? rows - base : cap;
                uint32_t nElem = nInj * channels;
                uint32_t m0 = rowStart + base;
                LocalTensor<float> src = ubIn[stage * slotStride];
                LocalTensor<float> dst = ubOut[stage * slotStride];
                LocalTensor<Element> ySlot = y[stage * yStride];
                WaitFlag<HardEvent::V_MTE2>(ev);
                gateParams.blockLen = nElem * static_cast<uint32_t>(sizeof(Element));
                DataCopyPad(src.template ReinterpretCast<Element>(), gGate[static_cast<int64_t>(m0) * channels],
                            gateParams, inPad);
                SetFlag<HardEvent::MTE2_V>(ev);
                WaitFlag<HardEvent::MTE3_V>(ev);
                WaitFlag<HardEvent::MTE2_V>(ev);
                if constexpr (!std::is_same_v<Element, float>) {
                    Cast(work, src.template ReinterpretCast<Element>(), RoundMode::CAST_NONE, nElem);
                    PipeBarrier<PIPE_V>();
                    if ((nElem & 7u) == 0u) {
                        DataCopy(src, work, nElem);
                    } else {
                        Adds(src, work, 0.0f, nElem);
                    }
                    PipeBarrier<PIPE_V>();
                }
                HcInjectCompute(dst, src, work, nElem, invC);
                SetFlag<HardEvent::V_MTE2>(ev);
                if constexpr (!std::is_same_v<Element, float>) {
                    Cast(ySlot, dst, RoundMode::CAST_RINT, nElem);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE3>(ev);
                WaitFlag<HardEvent::V_MTE3>(ev);
                outParams.blockLen = nElem * static_cast<uint32_t>(sizeof(Element));
                if constexpr (std::is_same_v<Element, float>) {
                    auto out = dst.template ReinterpretCast<Element>();
                    DataCopyPad(gInj[static_cast<int64_t>(m0) * channels], out, outParams);
                } else {
                    DataCopyPad(gInj[static_cast<int64_t>(m0) * channels], ySlot, outParams);
                }
                SetFlag<HardEvent::MTE3_V>(ev);
                ++round;
            }
        }
        WaitFlag<HardEvent::V_MTE2>(kEv[0]);
        WaitFlag<HardEvent::V_MTE2>(kEv[1]);
        WaitFlag<HardEvent::MTE3_V>(kEv[0]);
        WaitFlag<HardEvent::MTE3_V>(kEv[1]);
    }

private:
    // kernel 入口 Init 一次的常驻基址；c0 槽工作区同时充当 gate 槽基址。
    AscendC::GlobalTensor<Element> gC0_;
    AscendC::GlobalTensor<Element> gGate_;
    AscendC::GlobalTensor<Element> gDown_;
    AscendC::GlobalTensor<Element> gInj_;
    // down_pre_act 双视图：保存路径带 CACHE_MODE_DISABLE，c0dp 读路径保持无 hint。
    AscendC::GlobalTensor<Element> gDownPreAct_;
    AscendC::GlobalTensor<Element> gC0FromDpa_;
    bool hasPreAct_ = false;
    float invC_ = 0.0f;
    uint32_t rowsOnceMax_ = 1;
};

} // namespace Catlass::Epilogue
