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
#include "qsag_common.h"

namespace QSAG {

template <typename T, typename Tiling, bool DETERMINISTIC = false>
class CubeOp {
    static constexpr uint32_t L0_OPERAND_BYTES = 64 * 1024;
    static constexpr uint32_t L0_RESULT_BYTES = 128 * 1024;
    const ConstInfo<Tiling>* info_;
    SharedBuffers<T>* shared_;
    TBuf<TPosition::A1> l1_;
    TBuf<TPosition::A2> a_;
    TBuf<TPosition::B2> b_;
    TBuf<TPosition::CO1> c_;
    LocalTensor<T> queryL1_[SLOT_COUNT], doutL1_[SLOT_COUNT], ql_, dol_, la_, lb_;
    LocalTensor<float> lc_;
    GlobalTensor<T> q_, dout_;
    GlobalTensor<float> wq_;
    TEventID operandReuseEvent_[SLOT_COUNT], resultReuseEvent_[SLOT_COUNT];
    uint32_t l0Slots_ = 1, operandSlot_ = 0, resultSlot_ = 0;

public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR dout, GM_ADDR workspace, const ConstInfo<Tiling>& info,
                                SharedBuffers<T>& shared, TPipe* pipe)
    {
        info_ = &info;
        shared_ = &shared;
        q_.SetGlobalBuffer((__gm__ T*)query);
        dout_.SetGlobalBuffer((__gm__ T*)dout);
        wq_.SetGlobalBuffer((__gm__ float*)(workspace + info_->tiling->dqOffset));
        // L1：Q/dO 每槽 [Gcap,Dp]，K 每槽 [Ktile,Dp]，V 每槽 [Ktile,Dcache]。
        // P/dS 各双槽 [Gcap,Ktile]。
        const uint32_t queryElements = info_->groupCapacity * info_->dp;
        const uint32_t keyElements = info_->n * info_->dp;
        const uint32_t valueElements = info_->n * info_->dCache;
        const uint32_t probabilityElements = info_->groupCapacity * info_->n;
        const uint32_t l1Elements = 2 * info_->querySlots * queryElements +
                                    SLOT_COUNT * (keyElements + valueElements + 2 * probabilityElements);
        pipe->InitBuffer(l1_, l1Elements * sizeof(T));
        uint32_t l1Offset = 0;
        for (uint32_t slot = 0; slot < info_->querySlots; ++slot) {
            queryL1_[slot] = TakeL1(queryElements, l1Offset);
            doutL1_[slot] = TakeL1(queryElements, l1Offset);
        }
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            shared_->kl[slot] = TakeL1(keyElements, l1Offset);
            shared_->vl[slot] = TakeL1(valueElements, l1Offset);
            shared_->pl[slot] = TakeL1(probabilityElements, l1Offset);
            shared_->dsl[slot] = TakeL1(probabilityElements, l1Offset);
        }
        if ASCEND_IS_AIC {
            InitConstValueParams<T> zeroParams{1, static_cast<uint16_t>(queryElements * sizeof(T) / BLOCK_BYTES), 0,
                                               static_cast<T>(0)};
            for (uint32_t slot = 0; slot < info_->querySlots; ++slot) {
                InitConstValue(queryL1_[slot], zeroParams);
                InitConstValue(doutL1_[slot], zeroParams);
            }
            if (info_->halfRows < info_->halfCapacity) {
                InitConstValueParams<T> probabilityZero{
                    1, static_cast<uint16_t>(probabilityElements * sizeof(T) / BLOCK_BYTES), 0, static_cast<T>(0)};
                for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
                    InitConstValue(shared_->pl[slot], probabilityZero);
                    InitConstValue(shared_->dsl[slot], probabilityZero);
                }
            }
            PipeBarrier<PIPE_MTE2>();
            const auto plan =
                MakeCubeCapacityPlan<T>(info_->groupCapacity, info_->n, info_->dp, info_->dTile, info_->querySlots);
            l0Slots_ = plan.l0ABytes <= L0_OPERAND_BYTES / SLOT_COUNT &&
                               plan.l0BBytes <= L0_OPERAND_BYTES / SLOT_COUNT &&
                               plan.l0CBytes <= L0_RESULT_BYTES / SLOT_COUNT ?
                           SLOT_COUNT :
                           1;
            for (uint32_t slot = 0; slot < l0Slots_; ++slot) {
                operandReuseEvent_[slot] = pipe->AllocEventID<HardEvent::M_MTE1>();
                resultReuseEvent_[slot] = pipe->AllocEventID<HardEvent::FIX_M>();
                SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(operandReuseEvent_[slot]));
                SetFlag<HardEvent::FIX_M>(static_cast<event_t>(resultReuseEvent_[slot]));
            }
            pipe->InitBuffer(a_, L0_OPERAND_BYTES);
            pipe->InitBuffer(b_, L0_OPERAND_BYTES);
            pipe->InitBuffer(c_, L0_RESULT_BYTES);
            la_ = a_.Get<T>();
            lb_ = b_.Get<T>();
            lc_ = c_.Get<float>();
        }
    }

    __aicore__ inline void Mm12(const TileTask& task, bool firstTile)
    {
        uint32_t slot = task.id % SLOT_COUNT;
        ql_ = queryL1_[task.querySlot];
        dol_ = doutL1_[task.querySlot];

        if (firstTile) {
            LoadQL1Cube(task.token, task.head, info_->groupRows);
        }
        if (info_->directResult && !firstTile) {
            Fence<HardEvent::MTE2_MTE1>();
        }
        // K 完整驻留至 MM3；V 仅缓存一个 D 分片。
        if (!info_->directResult) {
            CopySelectedToL1(shared_->kl[slot], shared_->selK[slot], info_->dp);
        }
        for (uint32_t dStart = 0; dStart < info_->dp; dStart += info_->dTile) {
            uint32_t width = Min(info_->dTile, info_->dp - dStart);
            // 上一个 V 分片已搬入 L0B，才允许覆盖当前 L1 槽。
            if (!info_->directResult) {
                Fence<HardEvent::MTE1_MTE2>();
                CopySelectedToL1(shared_->vl[slot], shared_->selV[slot][dStart], width);
            }
            Product(dol_[dStart * info_->groupCapacity], shared_->vl[slot], info_->groupCapacity, width, info_->n,
                    width, false, false, info_->groupCapacity, info_->n, width, dStart == 0);
        }
        ToUB(shared_->grad[slot]);
        for (uint32_t dStart = 0; dStart < info_->dp; dStart += info_->dTile) {
            uint32_t width = Min(info_->dTile, info_->dp - dStart);
            Product(ql_[dStart * info_->groupCapacity], shared_->kl[slot][dStart * info_->n], info_->groupCapacity,
                    width, info_->n, width, false, false, info_->groupCapacity, info_->n, width, dStart == 0);
        }
        ToUB(shared_->score[slot]);
    }

    __aicore__ inline void Mm345(const TileTask& task)
    {
        uint32_t slot = task.id % SLOT_COUNT;
        ql_ = queryL1_[task.querySlot];
        dol_ = doutL1_[task.querySlot];

        // dQ 的行是 query head，直接累加到 [T1,N1,Dp]；不借用 [Ktile,Dp] 的 MM4 区。
        uint64_t dqOffset = info_->Offset(task.token, task.head, true, info_->dp);
        uint64_t dqStride = info_->dp;
        if (info_->tiling->inputLayout == 1) {
            dqStride = info_->tiling->qSeqLen * info_->dp;
        }
        for (uint32_t dStart = 0; dStart < info_->dp; dStart += info_->dTile) {
            uint32_t width = Min(info_->dTile, info_->dp - dStart);
            Product(shared_->dsl[slot], shared_->kl[slot][dStart * info_->n], info_->groupCapacity, info_->n, info_->n,
                    width, false, true, info_->groupCapacity, width, info_->n);
            ToGM(wq_[dqOffset + dStart], info_->groupRows, width, info_->groupCapacity, dqStride, true);
        }
        // MM3 是 K/V 的最后读者；所有 D 分片已由 MTE1 搬入 L0。
        // 只释放 K/V 槽，MM4/MM5 仍读取独立的 Q/dO/P/dS。
        CrossCoreSetFlag<2, PIPE_MTE1>(KV_FREE + slot);
        // MM4/MM5 的行是被选 KV token，等 Scatter 释放旧结果后再写入。
        CrossCoreWaitFlag<4, PIPE_FIX>(RESULT_FREE + info_->ResultSlot(slot));
        CrossCoreWaitFlag<4, PIPE_FIX>(RESULT_FREE + info_->ResultSlot(slot) + AIV_FLAG_OFFSET);
        for (uint32_t dStart = 0; dStart < info_->dp; dStart += info_->dTile) {
            uint32_t width = Min(info_->dTile, info_->dp - dStart);
            Product(shared_->dsl[slot], ql_[dStart * info_->groupCapacity], info_->groupCapacity, info_->n,
                    info_->groupCapacity, width, true, true, info_->n, width, info_->groupCapacity);
            if (info_->directResult) {
                ToResultUB(shared_->resultK[info_->ResultSlot(slot)][dStart], width, task.selected);
            } else {
                ToGM(shared_->mm4[slot][dStart], info_->n, width, info_->n, info_->dp, false);
            }
            Product(shared_->pl[slot], dol_[dStart * info_->groupCapacity], info_->groupCapacity, info_->n,
                    info_->groupCapacity, width, true, true, info_->n, width, info_->groupCapacity);
            if (info_->directResult) {
                ToResultUB(shared_->resultV[info_->ResultSlot(slot)][dStart], width, task.selected);
            } else {
                ToGM(shared_->mm5[slot][dStart], info_->n, width, info_->n, info_->dp, false);
            }
        }
    }

    __aicore__ inline void Destroy(TPipe* pipe)
    {
        PipeBarrier<PIPE_ALL>();
        for (uint32_t slot = 0; slot < l0Slots_; ++slot) {
            WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(operandReuseEvent_[slot]));
            WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(resultReuseEvent_[slot]));
            pipe->ReleaseEventID<HardEvent::M_MTE1>(operandReuseEvent_[slot]);
            pipe->ReleaseEventID<HardEvent::FIX_M>(resultReuseEvent_[slot]);
        }
    }

private:
    template <HardEvent E>
    __aicore__ inline void Fence()
    {
        SetFlag<E>(EVENT_ID0);
        WaitFlag<E>(EVENT_ID0);
    }

    __aicore__ inline LocalTensor<T> TakeL1(uint32_t elements, uint32_t& offset)
    {
        auto tensor = l1_.GetWithOffset<T>(elements, offset);
        offset += elements * sizeof(T);
        return tensor;
    }

    __aicore__ inline void LoadQL1Cube(uint64_t token, uint64_t head, uint32_t rows)
    {
        Fence<HardEvent::MTE1_MTE2>();
        uint64_t off = info_->Offset(token, head, true, info_->d);
        uint64_t stride = info_->tiling->inputLayout == 1 ? info_->tiling->qSeqLen * info_->d : info_->d;
        Nd2NzParams cp;
        cp.ndNum = 1;
        cp.nValue = rows;
        cp.dValue = info_->d;
        cp.srcDValue = stride;
        cp.srcNdMatrixStride = 0;
        cp.dstNzC0Stride = info_->groupCapacity;
        cp.dstNzNStride = 1;
        cp.dstNzMatrixStride = 0;
        if (stride <= MAX_ND2NZ_SRC_STRIDE) {
            DataCopy(ql_, q_[off], cp);
            DataCopy(dol_, dout_[off], cp);
        } else {
            cp.nValue = 1;
            cp.srcDValue = info_->d;
            for (uint32_t g = 0; g < rows; ++g) {
                DataCopy(ql_[g * NZ_WIDTH], q_[off + g * stride], cp);
                DataCopy(dol_[g * NZ_WIDTH], dout_[off + g * stride], cp);
            }
        }
        Fence<HardEvent::MTE2_MTE1>();
    }

    __aicore__ inline void CopySelectedToL1(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t columns)
    {
        Nd2NzParams cp;
        cp.ndNum = 1;
        cp.nValue = info_->n;
        cp.dValue = columns;
        cp.srcDValue = info_->dp;
        cp.srcNdMatrixStride = 0;
        cp.dstNzC0Stride = info_->n;
        cp.dstNzNStride = 1;
        cp.dstNzMatrixStride = 0;
        DataCopy(dst, src, cp);
        Fence<HardEvent::MTE2_MTE1>();
    }

    __aicore__ inline void LoadNZ(LocalTensor<T> dst, LocalTensor<T> src, uint32_t rows, uint32_t cols, bool transpose)
    {
        LoadData2DParamsV2 cp;
        cp.kStartPosition = 0;
        cp.kStep = cols / NZ_WIDTH;
        cp.srcStride = rows / NZ_WIDTH;
        cp.ifTranspose = transpose;
        if (!transpose) {
            cp.mStartPosition = 0;
            cp.mStep = rows / NZ_WIDTH;
            cp.dstStride = rows / NZ_WIDTH;
            LoadData(dst, src, cp);
        } else {
            cp.mStartPosition = 0;
            cp.mStep = rows / NZ_WIDTH;
            cp.dstStride = cols / NZ_WIDTH;
            LoadData(dst, src, cp);
        }
    }

    __aicore__ inline void Product(LocalTensor<T> left, LocalTensor<T> right, uint32_t lr, uint32_t lc, uint32_t rr,
                                   uint32_t rc, bool transA, bool transB, uint32_t m, uint32_t n, uint32_t k,
                                   bool firstReduction = true)
    {
        operandSlot_ = (operandSlot_ + 1) % l0Slots_;
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(operandReuseEvent_[operandSlot_]));
        la_ = a_.Get<T>()[operandSlot_ * L0_OPERAND_BYTES / l0Slots_ / sizeof(T)];
        lb_ = b_.Get<T>()[operandSlot_ * L0_OPERAND_BYTES / l0Slots_ / sizeof(T)];
        LoadNZ(la_, left, lr, lc, transA);
        LoadNZ(lb_, right, rr, rc, transB);
        Fence<HardEvent::MTE1_M>();
        if (firstReduction) {
            resultSlot_ = (resultSlot_ + 1) % l0Slots_;
            WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(resultReuseEvent_[resultSlot_]));
            lc_ = c_.Get<float>()[resultSlot_ * L0_RESULT_BYTES / l0Slots_ / sizeof(float)];
        }
        MmadParams cp;
        cp.m = m;
        cp.n = n;
        cp.k = k;
        cp.cmatrixInitVal = firstReduction;
        cp.unitFlag = 0;
        Mmad(lc_, la_, lb_, cp);
        if (l0Slots_ == 1) {
            PipeBarrier<PIPE_M>();
        }
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(operandReuseEvent_[operandSlot_]));
    }

    __aicore__ inline void ToUB(LocalTensor<float> dst)
    {
        Fence<HardEvent::M_FIX>();
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> cp;
        cp.mSize = AIV_PER_CUBE * info_->halfRows;
        cp.nSize = info_->n;
        cp.srcStride = info_->groupCapacity;
        cp.dstStride = info_->n;
        cp.dualDstCtl = 1;
        cp.unitFlag = 0;
        static constexpr FixpipeConfig config = {CO2Layout::ROW_MAJOR, true};
        Fixpipe<float, float, config>(dst, lc_, cp);
        SetFlag<HardEvent::FIX_M>(static_cast<event_t>(resultReuseEvent_[resultSlot_]));
    }

    __aicore__ inline void ToResultUB(LocalTensor<float> dst, uint32_t width, uint32_t selected)
    {
        Fence<HardEvent::M_FIX>();
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> cp;
        cp.mSize = info_->n;
        if constexpr (DETERMINISTIC) {
            // Dual destination divides these rows equally; each half is 16-row aligned.
            cp.mSize = (selected + 2 * NZ_WIDTH - 1) / (2 * NZ_WIDTH) * (2 * NZ_WIDTH);
        }
        cp.nSize = width;
        cp.srcStride = info_->n;
        cp.dstStride = 2 * info_->dp; // direct UB 行布局固定为 [dK, dV]。
        cp.dualDstCtl = 1;
        cp.unitFlag = 0;
        static constexpr FixpipeConfig config = {CO2Layout::ROW_MAJOR, true};
        Fixpipe<float, float, config>(dst, lc_, cp);
        SetFlag<HardEvent::FIX_M>(static_cast<event_t>(resultReuseEvent_[resultSlot_]));
    }

    __aicore__ inline void ToGM(GlobalTensor<float> dst, uint32_t rows, uint32_t cols, uint32_t sourceRows,
                                uint64_t stride, bool atomic)
    {
        Fence<HardEvent::M_FIX>();
        FixpipeParamsV220 cp;
        cp.mSize = rows;
        cp.nSize = cols;
        cp.srcStride = sourceRows;
        cp.unitFlag = 0;
        if (atomic) {
            SetAtomicType<float>();
        }
        if (stride <= MAX_FIXPIPE_STRIDE) {
            cp.dstStride = static_cast<uint32_t>(stride);
            Fixpipe<float, float, CFG_ROW_MAJOR>(dst, lc_, cp);
        } else {
            cp.mSize = 1;
            cp.dstStride = cols;
            for (uint32_t row = 0; row < rows; ++row) {
                Fixpipe<float, float, CFG_ROW_MAJOR>(dst[row * stride], lc_[row * NZ_WIDTH], cp);
            }
        }
        if (atomic) {
            SetAtomicNone();
        }
        SetFlag<HardEvent::FIX_M>(static_cast<event_t>(resultReuseEvent_[resultSlot_]));
    }
};
} // namespace QSAG
