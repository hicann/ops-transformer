/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_INDICES_CACHE_9102_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_INDICES_CACHE_9102_H_

#include "kernel_operator.h"

// [PLAN_A_INDICES_UB_OPT] Build-time A/B switch. No host/tiling ABI change.
#ifndef QBSA_9102_MX_INDICES_UB_CACHE
#define QBSA_9102_MX_INDICES_UB_CACHE 1
#endif
static_assert(QBSA_9102_MX_INDICES_UB_CACHE == 0 || QBSA_9102_MX_INDICES_UB_CACHE == 1,
              "QBSA_9102_MX_INDICES_UB_CACHE must be 0 or 1");

namespace BaseApi {
// One AIV-only, immutable sparse row at a time. RunInfo owns copies, not UB pointers.
// AIC retains its original GM scalar reads; C:V=1:1 does not imply scalar UB sharing.
class MxIndicesUbCache {
public:
    static constexpr uint32_t CAPACITY = 2048U;
    static constexpr uint32_t MIN_COUNT = 64U;
    static constexpr uint32_t BUFFER_BYTES = CAPACITY * sizeof(int32_t);

    __aicore__ inline void Init(AscendC::TPipe* pipe, uint32_t maxKb)
    {
        cachedCount_ = 0U;
        cachedBase_ = 0U;
        allocated_ = false;
        if constexpr (QBSA_9102_MX_INDICES_UB_CACHE != 0) {
            if ASCEND_IS_AIV {
                if (maxKb >= MIN_COUNT) {
                    pipe->InitBuffer(storage_, BUFFER_BYTES);
                    indices_ = storage_.template Get<int32_t>();
                    scalarToMte2_ = pipe->AllocEventID<AscendC::HardEvent::S_MTE2>();
                    mte2ToScalar_ = pipe->AllocEventID<AscendC::HardEvent::MTE2_S>();
                    allocated_ = true;
                }
            }
        }
    }

    __aicore__ inline void Prepare(AscendC::GlobalTensor<int32_t>& src, uint64_t base, uint32_t count, uint32_t maxKb)
    {
        // Invalidate even on fallback rows and on repeated bases in a new batch/launch.
        cachedCount_ = 0U;
        if constexpr (QBSA_9102_MX_INDICES_UB_CACHE != 0) {
            if ASCEND_IS_AIV {
                if (!allocated_ || count < MIN_COUNT || count > CAPACITY || count > maxKb) {
                    return;
                }
                // Protect previous scalar reads before reusing this single UB buffer.
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
                AscendC::DataCopyExtParams copy{};
                copy.blockCount = 1U;
                copy.blockLen = count * sizeof(int32_t);
                AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                // Copy only live entries: rounding the GM read up could cross the last row.
                AscendC::DataCopyPad(indices_, src[base], copy, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalar_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalar_);
                cachedBase_ = base;
                cachedCount_ = count;
            }
        }
    }

    __aicore__ inline int32_t Read(AscendC::GlobalTensor<int32_t>& src, uint64_t base, uint32_t index)
    {
        if constexpr (QBSA_9102_MX_INDICES_UB_CACHE != 0) {
            if ASCEND_IS_AIV {
                if (cachedBase_ == base && index < cachedCount_) {
                    return indices_.GetValue(index);
                }
            }
        }
        return src.GetValue(base + index);
    }

    __aicore__ inline void ReadGroup(AscendC::GlobalTensor<int32_t>& src, uint64_t base, uint32_t start, uint32_t count,
                                     int64_t* dst)
    {
        if constexpr (QBSA_9102_MX_INDICES_UB_CACHE != 0) {
            if ASCEND_IS_AIV {
                // Select the source once per group, not once for each of its 4/8 entries.
                if (cachedBase_ == base && start < cachedCount_ && count <= cachedCount_ - start) {
                    for (uint32_t i = 0U; i < count; ++i) {
                        dst[i] = indices_.GetValue(start + i);
                    }
                    return;
                }
            }
        }
        for (uint32_t i = 0U; i < count; ++i) {
            dst[i] = src.GetValue(base + start + i);
        }
    }

    __aicore__ inline void FreeEventID(AscendC::TPipe* pipe)
    {
        if constexpr (QBSA_9102_MX_INDICES_UB_CACHE != 0) {
            if ASCEND_IS_AIV {
                if (allocated_) {
                    // Every Prepare consumes both flags immediately; none is left set at teardown.
                    pipe->ReleaseEventID<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
                    pipe->ReleaseEventID<AscendC::HardEvent::MTE2_S>(mte2ToScalar_);
                }
            }
        }
        cachedCount_ = 0U;
        allocated_ = false;
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECIN> storage_;
    AscendC::LocalTensor<int32_t> indices_;
    AscendC::TEventID scalarToMte2_{};
    AscendC::TEventID mte2ToScalar_{};
    uint64_t cachedBase_ = 0U;
    uint32_t cachedCount_ = 0U;
    bool allocated_ = false;
};
} // namespace BaseApi

#endif
