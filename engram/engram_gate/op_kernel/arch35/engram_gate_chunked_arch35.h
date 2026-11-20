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
 * \file engram_gate_chunked_arch35.h
 * \brief chunked 数据组织：规约轴 hidden_size 列分块
 *
 * 依赖 engram_gate_arch35.h 和 engram_gate_pipeline_arch35.h；运行时由 TilingData 的
 * chunkedEnabled 分派，TilingKey 固定为 0。
 */
#ifndef ENGRAM_GATE_CHUNKED_ARCH35_H
#define ENGRAM_GATE_CHUNKED_ARCH35_H

#include "engram_gate_pipeline_arch35.h"

namespace EngramGateOps {
/*!
 * \brief chunked 数据组织：规约轴 hidden_size 列分块。
 *
 * tiling 契约: pass1Chunk < dim (整行驻留 UB 超预算, host 收缩 pass1Chunk 后仍不足整行)。
 * 结构: pass1 分块累加 (EngramGatePass1ChunkVF + accUb 跨块) -> EngramGateFinalizeVF
 *       -> 逐 chunk emit (EngramGateEmitVecVF + MTE3 分块搬出); x/v 逐 chunk 重载。
 */
template <typename T>
class EngramGateArch35Chunked : public EngramGatePipelineBase {
public:
    __aicore__ inline void Process()
    {
        if (!InitRuntime()) {
            return;
        }
        if (!InitLayout()) {
            return;
        }
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> root;
        pipe.InitBuffer(root, ubSafeBytes_);
        auto buffer = root.Get<uint8_t>();
        __ubuf__ uint8_t* ub = reinterpret_cast<__ubuf__ uint8_t*>(buffer.GetPhyAddr());
        BindUb(ub);
        LoadResidentWeight(ub);
        InitPipeline();
        ProcessRows();
        FlushStats();
        FinalizePipeline();
    }

private:
    __aicore__ inline bool InitRuntime()
    {
        const EngramGateTilingData& tilingDataIn = *tilingData_;
        if ((tilingDataIn.rows / tilingDataIn.heads) == 0) {
            return false;
        }
        xGm_ = (__gm__ T*)hiddenStates_;
        kvGm_ = (__gm__ T*)kv_;
        wGm_ = (__gm__ float*)weight_;
        outGm_ = (__gm__ T*)output_;
        dotGm_ = (__gm__ float*)dot_;
        gateGm_ = (__gm__ float*)gate_;
        rstdXGm_ = (__gm__ float*)rstdX_;
        rstdKGm_ = (__gm__ float*)rstdK_;
        hcMult_ = tilingDataIn.heads;
        dim_ = tilingDataIn.dim;
        pass1Chunk_ = tilingDataIn.chunk;
        totalRows_ = tilingDataIn.rows;
        usedCoreNums_ = tilingDataIn.usedCores;
        weightResident_ = tilingDataIn.weightResident;
        needBackward_ = (tilingDataIn.saveStats != 0) && (dot_ != nullptr) && (gate_ != nullptr) &&
                        (rstdX_ != nullptr) && (rstdK_ != nullptr);
        eps_ = tilingDataIn.eps;
        clampValue_ = tilingDataIn.clampValue;
        invDim_ = tilingDataIn.invDim;
        invSqrtDim_ = tilingDataIn.invSqrtDim;
        ubSafeBytes_ = tilingDataIn.ubLayoutBytes;
        return ubSafeBytes_ != 0;
    }

    __aicore__ inline bool InitLayout()
    {
        const uint32_t accOffRel = EngramGateOps::UB_ALIGN;
        xOff_ = accOffRel + EngramGateOps::ACCUMULATOR_COUNT * EngramGateOps::VEC_LANES * sizeof(float);
        dataSlots_ = EngramGateOps::DBUF_SLOTS;
        chunkBytes_ = EngramGateOps::AlignUp32(static_cast<uint32_t>(pass1Chunk_) * sizeof(T));
        const uint32_t wChunkBytes = EngramGateOps::AlignUp32(static_cast<uint32_t>(pass1Chunk_) * sizeof(float));
        const uint32_t outChunkBytes = EngramGateOps::AlignUp32(static_cast<uint32_t>(pass1Chunk_) * sizeof(T));
        const uint32_t wRegionBytes = weightResident_ ?
                                          EngramGateOps::AlignUp32(static_cast<uint32_t>(dim_) * sizeof(float)) :
                                          EngramGateOps::DBUF_SLOTS * wChunkBytes;
        outSlots_ = EngramGateOps::SINGLE_BUFFER_SLOT;
        if (xOff_ + dataSlots_ * EngramGateOps::ACCUMULATOR_COUNT * chunkBytes_ + wRegionBytes +
                outSlots_ * outChunkBytes + EngramGateOps::STATS_TASK_BYTES >
            ubSafeBytes_) {
            return false;
        }
        xSlotElems_ = chunkBytes_ / sizeof(T);
        vSlotElems_ = chunkBytes_ / sizeof(T);
        wSlotFloats_ = wChunkBytes / sizeof(float);
        outSlotElems_ = outChunkBytes / sizeof(T);
        const uint32_t kOffRel = xOff_ + dataSlots_ * xSlotElems_ * sizeof(T);
        const uint32_t vOffRel = kOffRel + dataSlots_ * xSlotElems_ * sizeof(T);
        const uint32_t wOffRel = vOffRel + dataSlots_ * vSlotElems_ * sizeof(T);
        const uint32_t outOffRel = wOffRel + wRegionBytes;
        return InitStatsLayout(accOffRel, kOffRel, vOffRel, wOffRel, outOffRel);
    }

    __aicore__ inline bool InitStatsLayout(uint32_t accOffRel, uint32_t kOffRel, uint32_t vOffRel, uint32_t wOffRel,
                                           uint32_t outOffRel)
    {
        const uint32_t dataTail = outOffRel + outSlots_ * outSlotElems_ * sizeof(T);
        const uint32_t stagePerTask = EngramGateOps::STATS_TASK_BYTES;
        const uint32_t stageCapacity = ubSafeBytes_ > dataTail ? (ubSafeBytes_ - dataTail) / stagePerTask : 0;
        if (stageCapacity == 0) {
            return false;
        }
        flushWindow_ = EngramGateOps::SINGLE_BUFFER_SLOT;
        while (flushWindow_ * EngramGateOps::STATS_WINDOW_GROWTH_FACTOR <= stageCapacity &&
               flushWindow_ * EngramGateOps::STATS_WINDOW_GROWTH_FACTOR <=
                   static_cast<uint32_t>(EngramGateOps::STATS_MAX_WINDOW_TASKS)) {
            flushWindow_ *= EngramGateOps::STATS_WINDOW_GROWTH_FACTOR;
        }
        stageBytes_ = flushWindow_ * stagePerTask;
        accOff_ = stageBytes_ + accOffRel;
        kOff_ = kOffRel + stageBytes_;
        vOff_ = vOffRel + stageBytes_;
        wOff_ = wOffRel + stageBytes_;
        outOff_ = outOffRel + stageBytes_;
        return true;
    }

    __aicore__ inline void BindUb(__ubuf__ uint8_t* ub)
    {
        stageUb_ = (__ubuf__ float*)ub;
        accUb_ = (__ubuf__ float*)(ub + accOff_);
        xUb_ = (__ubuf__ T*)(ub + xOff_ + stageBytes_);
        kUb_ = (__ubuf__ T*)(ub + kOff_);
        vUb_ = (__ubuf__ T*)(ub + vOff_);
        wUb_ = (__ubuf__ float*)(ub + wOff_);
        outUb_ = (__ubuf__ T*)(ub + outOff_);
        winTasks_ = 0;
        winTaskBase_ = 0;
    }

    __aicore__ inline void LoadResidentWeight(__ubuf__ uint8_t* ub)
    {
        if (weightResident_) {
            const int64_t row0 = GetBlockIdx();
            const int64_t hc0 = row0 % hcMult_;
            const uint32_t wBytes = static_cast<uint32_t>(dim_) * sizeof(float);
            AscendC::CopyGmToUbufAlignV2<float>((__ubuf__ float*)(ub + wOff_), (__gm__ float*)(wGm_ + hc0 * dim_), 1,
                                                wBytes, 0, 0, 0, 0, true, 0, false);
        }
    }

    __aicore__ inline void InitPipeline()
    {
        for (uint32_t s = 0; s < dataSlots_; ++s) {
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EngramGateOps::REL_EVT + s));
        }
        for (uint32_t s = 0; s < outSlots_; ++s) {
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EngramGateOps::OUT_DONE_EVT + s));
        }
        issued_ = 0;
        consumed_ = 0;
        pfRow_ = GetBlockIdx();
        pfC_ = 0;
        pfEmit_ = false;
        pfValid_ = pfRow_ < totalRows_;
    }

    __aicore__ inline void PrefetchPass1()
    {
        while (pfValid_ && issued_ < consumed_ + dataSlots_ && !pfEmit_) {
            const uint32_t ps = static_cast<uint32_t>(issued_ % dataSlots_);
            const int64_t pToken = pfRow_ / hcMult_;
            const int64_t pHc = pfRow_ - pToken * hcMult_;
            const int64_t pLen = (pass1Chunk_ < dim_ - pfC_) ? pass1Chunk_ : (dim_ - pfC_);
            const uint32_t xb = static_cast<uint32_t>(pLen) * sizeof(T);
            const uint32_t wb = static_cast<uint32_t>(pLen) * sizeof(float);
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EngramGateOps::REL_EVT + ps));
            AscendC::CopyGmToUbufAlignV2<T>(xUb_ + ps * xSlotElems_, (__gm__ T*)(xGm_ + pfRow_ * dim_ + pfC_), 1, xb, 0,
                                            0, 0, 0, true, 0, false);
            AscendC::CopyGmToUbufAlignV2<T>(kUb_ + ps * xSlotElems_,
                                            (__gm__ T*)(kvGm_ + pToken * (hcMult_ + 1) * dim_ + pHc * dim_ + pfC_), 1,
                                            xb, 0, 0, 0, 0, true, 0, false);
            if (!weightResident_) {
                AscendC::CopyGmToUbufAlignV2<float>(wUb_ + ps * wSlotFloats_, (__gm__ float*)(wGm_ + pHc * dim_ + pfC_),
                                                    1, wb, 0, 0, 0, 0, true, 0, false);
            }
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(EngramGateOps::BUF_EVT + ps));
            ++issued_;
            AdvancePass1Prefetch();
        }
    }

    __aicore__ inline void AdvancePass1Prefetch()
    {
        if (pfC_ + pass1Chunk_ < dim_) {
            pfC_ += pass1Chunk_;
        } else {
            pfC_ = 0;
            pfEmit_ = true;
        }
    }

    __aicore__ inline void ProcessPass1()
    {
        for (int64_t c = 0; c < dim_; c += pass1Chunk_) {
            const uint32_t s = static_cast<uint32_t>(consumed_ % dataSlots_);
            const int64_t len = (pass1Chunk_ < dim_ - c) ? pass1Chunk_ : (dim_ - c);
            PrefetchPass1();
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(EngramGateOps::BUF_EVT + s));
            __ubuf__ float* wPtr = weightResident_ ? (wUb_ + c) : (wUb_ + s * wSlotFloats_);
            if (c == 0) {
                EngramGateOps::EngramGatePass1ChunkVF<T, true>(xUb_ + s * xSlotElems_, kUb_ + s * xSlotElems_, wPtr,
                                                               accUb_, static_cast<int32_t>(len));
            } else {
                EngramGateOps::EngramGatePass1ChunkVF<T, false>(xUb_ + s * xSlotElems_, kUb_ + s * xSlotElems_, wPtr,
                                                                accUb_, static_cast<int32_t>(len));
            }
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EngramGateOps::REL_EVT + s));
            ++consumed_;
        }
    }

    __aicore__ inline void PrefetchEmit()
    {
        while (pfValid_ && issued_ < consumed_ + dataSlots_ && pfEmit_) {
            const uint32_t ps = static_cast<uint32_t>(issued_ % dataSlots_);
            const int64_t pToken = pfRow_ / hcMult_;
            const int64_t pHc = pfRow_ - pToken * hcMult_;
            const int64_t pLen = (pass1Chunk_ < dim_ - pfC_) ? pass1Chunk_ : (dim_ - pfC_);
            const uint32_t xb = static_cast<uint32_t>(pLen) * sizeof(T);
            (void)pHc;
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(EngramGateOps::REL_EVT + ps));
            AscendC::CopyGmToUbufAlignV2<T>(xUb_ + ps * xSlotElems_, (__gm__ T*)(xGm_ + pfRow_ * dim_ + pfC_), 1, xb, 0,
                                            0, 0, 0, true, 0, false);
            AscendC::CopyGmToUbufAlignV2<T>(vUb_ + ps * vSlotElems_,
                                            (__gm__ T*)(kvGm_ + pToken * (hcMult_ + 1) * dim_ + hcMult_ * dim_ + pfC_),
                                            1, xb, 0, 0, 0, 0, true, 0, false);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(EngramGateOps::BUF_EVT + ps));
            ++issued_;
            AdvanceEmitPrefetch();
        }
    }

    __aicore__ inline void AdvanceEmitPrefetch()
    {
        if (pfC_ + pass1Chunk_ < dim_) {
            pfC_ += pass1Chunk_;
        } else {
            pfEmit_ = false;
            pfRow_ += usedCoreNums_;
            pfC_ = 0;
            if (pfRow_ >= totalRows_) {
                pfValid_ = false;
            }
        }
    }

    __aicore__ inline void ProcessEmit(__gm__ T* outRow)
    {
        for (int64_t c = 0; c < dim_; c += pass1Chunk_) {
            const int64_t len = (pass1Chunk_ < dim_ - c) ? pass1Chunk_ : (dim_ - c);
            PrefetchEmit();
            const uint32_t s = static_cast<uint32_t>(consumed_ % dataSlots_);
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(EngramGateOps::BUF_EVT + s));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(EngramGateOps::OUT_DONE_EVT));
            EngramGateOps::EmitVF<T>(outUb_, xUb_ + s * xSlotElems_, vUb_ + s * vSlotElems_,
                                     stageUb_ + winTasks_ * EngramGateOps::STATS_TASK_FLOATS,
                                     static_cast<int32_t>(len));
            SetFlag<HardEvent::V_MTE3>(EngramGateOps::OUT_EVT);
            WaitFlag<HardEvent::V_MTE3>(EngramGateOps::OUT_EVT);
            const uint32_t outBytes = static_cast<uint32_t>(len) * sizeof(T);
            AscendC::CopyUbufToGmAlignV2<T>(outRow + c, outUb_, 1, outBytes, 0, 0, true, 0);
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(EngramGateOps::OUT_DONE_EVT));
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(EngramGateOps::REL_EVT + s));
            ++consumed_;
        }
    }

    __aicore__ inline void StageStats(int64_t row)
    {
        if (winTasks_ == 0) {
            winTaskBase_ = row;
        }
        ++winTasks_;
        if (winTasks_ >= flushWindow_) {
            FlushStats();
        }
    }

    __aicore__ inline void FlushStats()
    {
        if (winTasks_ > 0) {
            if (needBackward_) {
                const int64_t flushRow0 = winTaskBase_;
                const int64_t flushStride = usedCoreNums_;
                EngramGateOps::EngramGateCopyStats<false, false>(stageUb_, dotGm_, gateGm_, rstdXGm_, rstdKGm_,
                                                                 flushRow0, flushStride, winTasks_);
            }
            winTasks_ = 0;
        }
    }

    __aicore__ inline void ProcessRows()
    {
        for (int64_t row = GetBlockIdx(); row < totalRows_; row += usedCoreNums_) {
            const int64_t token = row / hcMult_;
            const int64_t hc = row - token * hcMult_;
            __gm__ T* outRow = (__gm__ T*)(outGm_ + row * dim_);
            ProcessPass1();
            EngramGateOps::EngramGateFinalizeVF(accUb_, stageUb_ + winTasks_ * EngramGateOps::STATS_TASK_FLOATS, eps_,
                                                clampValue_, invDim_, invSqrtDim_);
            ProcessEmit(outRow);
            StageStats(row);
        }
    }

    __aicore__ inline void FinalizePipeline()
    {
        for (uint32_t slot = 0; slot < dataSlots_; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(REL_EVT + slot));
        }
        for (uint32_t slot = 0; slot < outSlots_; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(OUT_DONE_EVT + slot));
        }
    }

    __gm__ T* xGm_ = nullptr;
    __gm__ T* kvGm_ = nullptr;
    __gm__ float* wGm_ = nullptr;
    __gm__ T* outGm_ = nullptr;
    __gm__ float* dotGm_ = nullptr;
    __gm__ float* gateGm_ = nullptr;
    __gm__ float* rstdXGm_ = nullptr;
    __gm__ float* rstdKGm_ = nullptr;
    __ubuf__ float* stageUb_ = nullptr;
    __ubuf__ float* accUb_ = nullptr;
    __ubuf__ T* xUb_ = nullptr;
    __ubuf__ T* kUb_ = nullptr;
    __ubuf__ T* vUb_ = nullptr;
    __ubuf__ float* wUb_ = nullptr;
    __ubuf__ T* outUb_ = nullptr;
    int64_t hcMult_ = 0;
    int64_t dim_ = 0;
    int64_t pass1Chunk_ = 0;
    int64_t totalRows_ = 0;
    int64_t usedCoreNums_ = 0;
    int64_t weightResident_ = 0;
    int64_t issued_ = 0;
    int64_t consumed_ = 0;
    int64_t pfRow_ = 0;
    int64_t pfC_ = 0;
    int64_t winTaskBase_ = 0;
    uint32_t ubSafeBytes_ = 0;
    uint32_t dataSlots_ = 0;
    uint32_t outSlots_ = 0;
    uint32_t chunkBytes_ = 0;
    uint32_t xSlotElems_ = 0;
    uint32_t vSlotElems_ = 0;
    uint32_t wSlotFloats_ = 0;
    uint32_t outSlotElems_ = 0;
    uint32_t flushWindow_ = 0;
    uint32_t stageBytes_ = 0;
    uint32_t xOff_ = 0;
    uint32_t accOff_ = 0;
    uint32_t kOff_ = 0;
    uint32_t vOff_ = 0;
    uint32_t wOff_ = 0;
    uint32_t outOff_ = 0;
    uint32_t winTasks_ = 0;
    bool needBackward_ = false;
    bool pfEmit_ = false;
    bool pfValid_ = false;
    float eps_ = 0.0F;
    float clampValue_ = 0.0F;
    float invDim_ = 0.0F;
    float invSqrtDim_ = 0.0F;
};
} // namespace EngramGateOps

#endif // ENGRAM_GATE_CHUNKED_ARCH35_H
