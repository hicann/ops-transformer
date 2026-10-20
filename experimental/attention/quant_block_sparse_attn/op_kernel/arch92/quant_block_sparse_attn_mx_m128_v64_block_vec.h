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
 * \file quant_block_sparse_attn_mx_m128_v64_block_vec.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化 vector 路径。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_M128_V64_BLOCK_VEC_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_M128_V64_BLOCK_VEC_H_

#include "kernel_operator.h"
#include "quant_block_sparse_attn_mx_m128_v64_layout.h"
#include "../quant_block_sparse_attn_mx_tiling_data.h"
#include "../common/quant_block_sparse_attn_common_arch35.h"
#include "../common/quant_block_sparse_attn_attenmask.h"
#include "../common/buffer_manager.h"
#include "../common/buffers_policy.h"
#include "../common/util_regbase_mx.h"
#include "../common/vf/vf_div_cast.h"
#include "../common/vf/vf_flashupdate_new.h"
#include "vf_mx_m128_v1_fulln.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbasemx;

namespace BaseApi {
// V1 执行 softmax/P 量化，V2 更新并写回 attention output/LSE。
template <typename inputType, typename mmType, typename outputType, QBSALayout layout, QBSALayout kvLayout,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          DTemplateType dVTemplateType, bool hasAtten, bool hasLse, bool isPa, bool useDn = true>
class QuantBlockSparseAttnMxM128V64BlockVec {
public:
    using INPUT_T = inputType;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = mmType;
    using OUTPUT_T = outputType;
    using MxBmm2Buf = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U>;

    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_ATTEN = hasAtten;
    static constexpr bool HAS_LSE = hasLse;
    static constexpr bool IS_PA = isPa;
    static constexpr QBSALayout LAYOUT = layout;
    static constexpr QBSALayout KV_LAYOUT = kvLayout;
    static constexpr uint32_t LOGICAL_M_BASE = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t M_BASE = 64U; // VF/mask geometry is deliberately unchanged
    static constexpr uint32_t CUBE_M_BASE = MxM128V64Layout::CUBE_M;
    static constexpr uint32_t S2_BASE = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t VEC_M_BASE = M_BASE;
    static constexpr uint32_t D_BASE = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t DV_BASE = static_cast<uint32_t>(dVTemplateType);
    static_assert(LAYOUT == QBSALayout::TND && KV_LAYOUT == QBSALayout::PA_BNBD && IS_PA,
                  "MX vector currently only supports TND query and PA_BNBD KV");
    static_assert(LOGICAL_M_BASE == 128U && S2_BASE == 512U && D_BASE == 128U && DV_BASE == 128U,
                  "MX vector currently only supports logical S1=128, S2=512, D=128 and DV=128");
    static constexpr uint32_t DB = 2U;
    static constexpr uint32_t PRELOAD_N = 2U;
    static constexpr uint32_t MX_SCALE_GROUP = 32U;
    static constexpr uint32_t dTemplateAlign64 = 128U;
    static constexpr uint32_t DN_VSELR_INDEX_SIZE = 256U;
    static constexpr uint32_t DN_VSELR_GROUP_NUM = 4U;
    static constexpr uint32_t DN_VSELR_GROUP_SIZE = DN_VSELR_INDEX_SIZE / DN_VSELR_GROUP_NUM;
    static constexpr uint32_t ATTEN_MASK_UB_SIZE = S2_SPLIT * VEC_M_BASE;
    static constexpr uint32_t MASK_COPY_BLOCK_BYTES = 32U;
    // VF 输出的 P 使用 DN packed layout：每个 AIV 处理 VEC_M_BASE 行，每行 S2_SPLIT 个 P 元素，
    // 并额外保留 S2_SPLIT / MX_SCALE_GROUP 个元素作为行内 padding/重排空间。
    static constexpr uint32_t STAGE1_CAST_ROW_STRIDE = S2_SPLIT + S2_SPLIT / MX_SCALE_GROUP;
    static constexpr uint32_t STAGE1_OUT_UB_SIZE = VEC_M_BASE * STAGE1_CAST_ROW_STRIDE * sizeof(INPUT_T);
    static constexpr uint32_t PSCALE_SUB_LOOP_UB_SIZE = VEC_M_BASE * S2_SPLIT / MX_SCALE_GROUP * sizeof(SCALE_T);
    // [M128_P_SCALE_PAIR] Each packed-P owner also owns its 512B e8m0 tail.
    // One V->MTE3 ready/free protocol protects both outputs of the same VF.
    static constexpr uint32_t STAGE1_QUEUE_UB_SIZE = STAGE1_OUT_UB_SIZE + PSCALE_SUB_LOOP_UB_SIZE;
    static_assert(STAGE1_OUT_UB_SIZE % 32U == 0U && sizeof(INPUT_T) == 1U && sizeof(SCALE_T) == 1U,
                  "M128 P/scale regions must be disjoint, DMA-aligned byte tensors");
    static constexpr uint32_t SOFTMAX_STATE_UB_SIZE = CUBE_M_BASE * sizeof(float);
    // [MX_EMPTY_LSE] Output-only -Inf, matching arch35/golden. Keep the V1
    // internal max sentinel finite (-FLT_MAX) for the existing safe Log/Add.
    static constexpr float MX_EMPTY_LSE_VALUE = -3e+99;

    __aicore__ inline void Init(TPipe* pipe, __gm__ uint8_t* pScale, __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* attentionOut, __gm__ uint8_t* attenMask, uint32_t attenMaskS2Size,
                                uint8_t pScaleDtype)
    {
        if ASCEND_IS_AIV {
            // Optional inputs may differ between consecutive cases. Restore their neutral launch defaults first.
            attenMaskS2Size_ = 0U;
            pScaleValue_ = 1.0F;
            tPipe_ = pipe;
            attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
            if constexpr (HAS_LSE) {
                softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
            }
            if (pScale != nullptr) {
                if (pScaleDtype == optiling::MX_PSCALE_DTYPE_FP32) {
                    GlobalTensor<float> pScaleGmFp32;
                    pScaleGmFp32.SetGlobalBuffer((__gm__ float*)pScale);
                    pScaleValue_ = pScaleGmFp32.GetValue(0);
                } else {
                    GlobalTensor<uint8_t> pScaleGm;
                    pScaleGm.SetGlobalBuffer((__gm__ uint8_t*)pScale);
                    pScaleValue_ = DecodeE8M0Scale(pScaleGm.GetValue(0));
                }
            }
            if constexpr (HAS_ATTEN) {
                attenMaskGm_.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
                attenMaskS2Size_ = attenMaskS2Size;
            }
            negativeFloatScalar_ = *((const MM_T*)&NEGATIVE_MIN_VALUE_FP32);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // [M128_V64] Two independent [64,128] FP32 accumulators in one 64KiB allocation.
            tPipe_->InitBuffer(stage2OutBuf_, CUBE_M_BASE * dTemplateAlign64 * sizeof(MM_T));
            // [M128_P_SCALE_PAIR] Two 17KiB P+scale stages, with no extra queue/event IDs.
            tPipe_->InitBuffer(stage1OutQue_[0], 1, STAGE1_QUEUE_UB_SIZE);
            tPipe_->InitBuffer(stage1OutQue_[1], 1, STAGE1_QUEUE_UB_SIZE);
            // Each of the three ring states contains row0[64] and row1[64]; no VF grows.
            tPipe_->InitBuffer(softmaxSumBuf_[0], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxSumBuf_[1], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxSumBuf_[2], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxMaxBuf_[0], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxMaxBuf_[1], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxMaxBuf_[2], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxExpBuf_[0], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxExpBuf_[1], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxExpBuf_[2], SOFTMAX_STATE_UB_SIZE);
            // 进入当前 512-column block 前的 Max；Sum 直接在 softmaxSumBuf_ 上递推。
            tPipe_->InitBuffer(preLoopMaxBuf_, SOFTMAX_STATE_UB_SIZE);
            if constexpr (HAS_LSE) {
                tPipe_->InitBuffer(softmaxLseQueue_, 1, VEC_M_BASE * sizeof(float) * 8U);
            }
            if constexpr (HAS_ATTEN) {
                // DN mask UB：[256,64]，按 sparse segment 拼接。
                if constexpr (QBSA_9102_MX_M128_MASK_PREFETCH != 0) {
                    InitMaskPrefetchBuffers();
                } else {
                    tPipe_->InitBuffer(attenMaskInQue_, 1, ATTEN_MASK_UB_SIZE);
                }
            }
            // DN VF lane 重排索引；保留 USE_DN 分支，便于后续补充非 DN 索引表。
            if constexpr (USE_DN) {
                tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)],
                                   DN_VSELR_INDEX_SIZE);
                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
                for (uint32_t i = 0U; i < DN_VSELR_GROUP_NUM; ++i) {
                    for (uint32_t j = 0U; j < DN_VSELR_GROUP_SIZE; ++j) {
                        vselrIndexesTensor.SetValue(i * DN_VSELR_GROUP_SIZE + j, i + (j << 2));
                    }
                }
            }

            InitV2OutputSync();
        }
    }

    __aicore__ inline void FreeEventID()
    {
        if ASCEND_IS_AIV {
            FreeV2OutputSync();
            FreeMaskPrefetchEvents();
        }
    }

    __aicore__ inline void ClearEmptyQBlock(uint32_t n1Idx, uint32_t queryTokenBase, uint32_t actualS1Size,
                                            uint32_t s1OuterIdx, const MxConstInfo& constInfo)
    {
        if ASCEND_IS_AIV {
            const uint32_t blockS1Idx = s1OuterIdx * constInfo.qSparseBlockSize;
            if (unlikely(blockS1Idx >= actualS1Size)) {
                return;
            }
            const uint32_t blockMSize = blockS1Idx + constInfo.qSparseBlockSize > actualS1Size ?
                                            actualS1Size - blockS1Idx :
                                            constInfo.qSparseBlockSize;
            // 9102r1 只有一个 V 核；按 64 行切片清完整 logical Q block。这样清零范围与 metadata
            // 分配范围一致，不需要不存在的跨 AIV 全局屏障，也不会覆盖其他核已经写回的有效 head。
            for (uint32_t mBase = 0U; mBase < blockMSize; mBase += VEC_M_BASE) {
                const uint32_t actMSize = blockMSize - mBase > VEC_M_BASE ? VEC_M_BASE : blockMSize - mBase;
                const uint64_t tokenOffset = static_cast<uint64_t>(queryTokenBase) + blockS1Idx + mBase;

                DataCopyExtParams outCopyParams{};
                outCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
                outCopyParams.blockCount = actMSize;
                outCopyParams.srcStride = 0U;
                outCopyParams.dstStride = constInfo.attentionOutStride;
                const uint64_t outOffset = (tokenOffset * constInfo.realN2Size + n1Idx) * constInfo.dSizeV;
                LocalTensor<OUTPUT_T> emptyOut = stage1OutQue_[0].template AllocTensor<OUTPUT_T>();
                Duplicate<OUTPUT_T>(emptyOut, 0.0F, actMSize * constInfo.dSizeV);
                stage1OutQue_[0].template EnQue(emptyOut);
                emptyOut = stage1OutQue_[0].template DeQue<OUTPUT_T>();
                DataCopyPad(attentionOutGm_[outOffset], emptyOut, outCopyParams);
                stage1OutQue_[0].template FreeTensor(emptyOut);

                if constexpr (HAS_LSE) {
                    DataCopyExtParams lseCopyParams{};
                    lseCopyParams.blockLen = sizeof(float);
                    lseCopyParams.blockCount = actMSize;
                    lseCopyParams.srcStride = 0U;
                    lseCopyParams.dstStride = constInfo.softmaxLseStride;
                    const uint64_t lseOffset = tokenOffset * constInfo.realN2Size + n1Idx;
                    LocalTensor<float> emptyLse = softmaxLseQueue_.template AllocTensor<float>();
                    // LSE 的 UB->GM 搬运按 E2B 布局读取，每一行占一个 32B 槽。
                    Duplicate<float>(emptyLse, MX_EMPTY_LSE_VALUE, actMSize * 8U);
                    softmaxLseQueue_.template EnQue(emptyLse);
                    emptyLse = softmaxLseQueue_.template DeQue<float>();
                    DataCopyPad(softmaxLseGm_[lseOffset], emptyLse, lseCopyParams);
                    softmaxLseQueue_.template FreeTensor(emptyLse);
                }
            }
        }
    }

    // [M128_MASK_PREFETCH] DMA-only producer. No Duplicate and no wait on PIPE_V here.
    // Slot == Q row slice; each prepared slot is consumed exactly once, even for S2 tails.
    template <uint32_t SUB_LOOP>
    __aicore__ inline void PrefetchAttenMask(const MxRunInfo& runInfo)
    {
        static_assert(SUB_LOOP < 2U, "mask subLoop must be 0/1");
        if constexpr (HAS_ATTEN && QBSA_9102_MX_M128_MASK_PREFETCH != 0) {
            if ((runInfo.attenMaskSubLoopBits & (1U << SUB_LOOP)) == 0U) {
                return;
            }
            const uint32_t slot = runInfo.vecMbaseIdx;
            WaitFlag<HardEvent::V_MTE2>(maskFreeId_[slot]);
            LocalTensor<uint8_t> mask = maskPrefetchBuf_[slot].template Get<uint8_t>();
            PrepareMaskSegments<SUB_LOOP, true>(mask, runInfo, GetSubLoopS2Size<SUB_LOOP>(runInfo));
            // Also orders the slot-release wait if there are no DMA segments.
            SetFlag<HardEvent::MTE2_V>(maskReadyId_[slot]);
        }
    }

    // V1：C1[64,256] -> P/PScale；两个 subLoop 写成 P[64,512]。
    template <uint32_t SUB_LOOP>
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD, 1U>& outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U>& bmm1ResBuf,
                                       MxRunInfo& runInfo, const MxConstInfo& constInfo)
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        // [M128_V64] Even an empty second row slice must produce defined P/PScale for the
        // padded 128-row C2. MakeVecRunInfo disables its mask; V2/LSE never write invalid rows.
        // P-ready is published ONCE by the kernel after BOTH row slices and all S2 halves.

        const uint32_t softmaxBufIdx = runInfo.mLoop % (PRELOAD_N + 1U);
        const uint32_t expBufIdx = runInfo.loop % (PRELOAD_N + 1U);
#if QBSA_9102_MX_M128_P_PINGPONG
        // [M128_P_PINGPONG] ExecuteTask visits s0r0,s0r1,s1r0,s1r1. Alternate
        // packed-P stages 0,1,0,1, rather than SUB_LOOP's 0,0,1,1, so the next
        // VF need not reuse the P tensor that MTE3 is still reading. Keep all
        // TQue Alloc/EnQue/DeQue/Free calls: FreeTensor is NOT a synchronous free.
        // Both row slices execute even for Q tails; no persistent slot counter
        // is needed across single-half tasks, empty groups or section drains.
        static_assert(MxM128V64Layout::ROW_SLICES == 2U, "packed-P has two stages");
        const uint32_t stage1Offset = runInfo.vecMbaseIdx;
#else
        constexpr uint32_t stage1Offset = SUB_LOOP;
#endif
        LocalTensor<float> sumUb =
            softmaxSumBuf_[softmaxBufIdx].template Get<float>()[runInfo.vecMbaseIdx * VEC_M_BASE];
        LocalTensor<float> maxUb =
            softmaxMaxBuf_[softmaxBufIdx].template Get<float>()[runInfo.vecMbaseIdx * VEC_M_BASE];
        LocalTensor<float> preLoopMaxUb = preLoopMaxBuf_.template Get<float>()[runInfo.vecMbaseIdx * VEC_M_BASE];
        LocalTensor<float> expUb = softmaxExpBuf_[expBufIdx].template Get<float>()[runInfo.vecMbaseIdx * VEC_M_BASE];
        LocalTensor<MM_T> mmRes = bmm1ResBuf.template GetTensor<MM_T>();
        LocalTensor<INPUT_T> stage1CastTensor = stage1OutQue_[stage1Offset].template AllocTensor<INPUT_T>();
        LocalTensor<SCALE_T> pScaleSubLoopTensor =
            stage1CastTensor.template ReinterpretCast<SCALE_T>()[STAGE1_OUT_UB_SIZE / sizeof(SCALE_T)];

        const uint32_t s2CalcSize = GetSubLoopS2Size<SUB_LOOP>(runInfo);
        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (USE_DN) {
            if constexpr (HAS_ATTEN) {
                constexpr uint8_t attenMaskBit = static_cast<uint8_t>(1U << SUB_LOOP);
                if (unlikely((runInfo.attenMaskSubLoopBits & attenMaskBit) != 0U)) {
                    // [PLAN_A_PERF] Only masked subLoops allocate/use the mask.
                    // Keep the 9102 allocator + explicit MTE2_V protocol.
                    if constexpr (QBSA_9102_MX_M128_MASK_PREFETCH != 0) {
                        attenMaskUb = ConsumePrefetchedAttenMask<SUB_LOOP>(runInfo, s2CalcSize);
                    } else {
                        attenMaskUb = CopyAttenMaskToUb<SUB_LOOP>(runInfo, s2CalcSize);
                    }
                    ExecuteVec1Vf<true, SUB_LOOP>(bmm1ResBuf, stage1CastTensor, sumUb, maxUb, mmRes, expUb, attenMaskUb,
                                                  pScaleSubLoopTensor, preLoopMaxUb, runInfo, constInfo, s2CalcSize);
                    if constexpr (QBSA_9102_MX_M128_MASK_PREFETCH != 0) {
                        ReleasePrefetchedAttenMask(runInfo);
                    } else {
                        attenMaskInQue_.template FreeTensor(attenMaskUb);
                    }
                } else {
                    ExecuteVec1Vf<false, SUB_LOOP>(bmm1ResBuf, stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                                                   attenMaskUb, pScaleSubLoopTensor, preLoopMaxUb, runInfo, constInfo,
                                                   s2CalcSize);
                }
            } else {
                ExecuteVec1Vf<false, SUB_LOOP>(bmm1ResBuf, stage1CastTensor, sumUb, maxUb, mmRes, expUb, attenMaskUb,
                                               pScaleSubLoopTensor, preLoopMaxUb, runInfo, constInfo, s2CalcSize);
            }
        } else {
            static_assert(USE_DN, "MXFullQuantMode vector currently implements DN VF only; "
                                  "use_dn=false is reserved for future extension");
        }

        // [M128_P_SCALE_PAIR] Publish both disjoint regions after the same VF.
        // Rebind the tail to the dequeued owner; free only after BOTH DMA reads.
        // The next row slice uses the other stage, not this in-flight scale tail.
        stage1OutQue_[stage1Offset].template EnQue(stage1CastTensor);
        stage1CastTensor = stage1OutQue_[stage1Offset].template DeQue<INPUT_T>();
        pScaleSubLoopTensor =
            stage1CastTensor.template ReinterpretCast<SCALE_T>()[STAGE1_OUT_UB_SIZE / sizeof(SCALE_T)];
        CopyPScaleToL1<SUB_LOOP>(outputBuf, pScaleSubLoopTensor, runInfo);
        CopyPToL1<SUB_LOOP>(outputBuf, stage1CastTensor, runInfo);
        stage1OutQue_[stage1Offset].template FreeTensor(stage1CastTensor);

        if (unlikely(runInfo.isLastS2Loop && IsLastSubLoop<SUB_LOOP>(runInfo))) {
            // 最后一个 S2/subLoop 后写 LSE。
            if constexpr (HAS_LSE) {
                SoftmaxLseCopyOut(runInfo, constInfo, sumUb, maxUb);
            }
        }
    }

    // V2：更新 C2[64,DV]，最后归一化并写回 bf16。
    __aicore__ inline void ProcessVec2(MxBmm2Buf& bmm2ResBuf, MxRunInfo& runInfo, const MxConstInfo& constInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0U)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        const uint32_t vecMSize = runInfo.actVecMSize;
        const uint32_t loopIdx = runInfo.loop % (PRELOAD_N + 1U);
        const uint32_t mLoopIdx = runInfo.mLoop % (PRELOAD_N + 1U);
        const bool isFirstS2Loop = runInfo.isFirstS2Loop;
        const bool isLastS2Loop = runInfo.isLastS2Loop;
        const int64_t vec2CalcSize = static_cast<int64_t>(vecMSize) * dTemplateAlign64;
        LocalTensor<MM_T> vec2ResUb =
            stage2OutBuf_.template Get<MM_T>()[runInfo.vecMbaseIdx * VEC_M_BASE * dTemplateAlign64];
        LocalTensor<MM_T> mmRes = bmm2ResBuf.template GetTensor<MM_T>();
        WaitV2OutputReuse(runInfo.vecMbaseIdx);
        if (unlikely(isFirstS2Loop)) {
            // 首个 S2 tile 直接作为累计值。
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<MM_T> expUb = softmaxExpBuf_[loopIdx].template Get<MM_T>()[runInfo.vecMbaseIdx * VEC_M_BASE];
            if (!isLastS2Loop) {
                // 中间 tile 保留 fp32 累计值。
                FaVectorApi::FlashUpdateNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            } else {
                // 最后一个 tile 完成 update 与归一化。
                LocalTensor<float> sumUb =
                    softmaxSumBuf_[mLoopIdx].template Get<float>()[runInfo.vecMbaseIdx * VEC_M_BASE];
                FaVectorApi::FlashUpdateLastNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, true>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (isLastS2Loop) {
            if (unlikely(isFirstS2Loop)) {
                // 单 tile 场景单独归一化。
                if constexpr (QBSA_9102_MX_M128_MTE3_SCOPE != 0) {
                    PipeBarrier<PIPE_V>(); // UB copy stores must precede LastDiv's reads.
                }
                LocalTensor<float> sumUb =
                    softmaxSumBuf_[mLoopIdx].template Get<float>()[runInfo.vecMbaseIdx * VEC_M_BASE];
                FaVectorApi::LastDivNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                    vec2ResUb, vec2ResUb, sumUb, vecMSize, static_cast<uint16_t>(dTemplateAlign64), 1.0f);
            }
            CopyOutAttentionOut(runInfo, constInfo, vec2ResUb);
        }
        RecordV2OutputDma(runInfo.vecMbaseIdx, isLastS2Loop);
    }

private:
    // [M128_MTE3_SCOPE] Two disjoint [64,128] FP32 regions share stage2OutBuf_.
    // Only the final in-place cast/output DMA reads these regions from MTE3.
    // P/PScale/LSE keep their own TQue lifetime synchronization.
    __aicore__ inline void InitV2OutputSync()
    {
        for (uint32_t row = 0U; row < V2_OUTPUT_SYNC_SLOTS; ++row) {
            mte3ToVId_[row] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
            vToMte3Id_[row] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
            outputDmaPending_[row] = false;
        }
        if constexpr (QBSA_9102_MX_M128_MTE3_SCOPE == 0) {
            SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        }
    }

    __aicore__ inline void WaitV2OutputReuse(uint32_t row)
    {
        if constexpr (QBSA_9102_MX_M128_MTE3_SCOPE != 0) {
            if (outputDmaPending_[row]) {
                WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[row]);
                outputDmaPending_[row] = false;
            }
            // Retain true V->V RAW/WAR ordering (accumulator, exp/sum, prior cast).
            // This fences PIPE_V only; it must not fence unrelated P/LSE MTE3 copies.
            PipeBarrier<PIPE_V>();
        } else {
            WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        }
    }

    __aicore__ inline void RecordV2OutputDma(uint32_t row, bool hasOutputDma)
    {
        if constexpr (QBSA_9102_MX_M128_MTE3_SCOPE != 0) {
            if (hasOutputDma) {
                // Called AFTER CopyOutAttentionOut submits DataCopyPad, not after every V2.
                SetFlag<HardEvent::MTE3_V>(mte3ToVId_[row]);
                outputDmaPending_[row] = true;
            }
        } else {
            SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        }
    }

    __aicore__ inline void FreeV2OutputSync()
    {
        for (uint32_t row = 0U; row < V2_OUTPUT_SYNC_SLOTS; ++row) {
            if constexpr (QBSA_9102_MX_M128_MTE3_SCOPE != 0) {
                if (outputDmaPending_[row]) {
                    WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[row]);
                    outputDmaPending_[row] = false;
                }
            } else {
                WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[row]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId_[row]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id_[row]);
        }
    }

    __aicore__ inline void InitMaskPrefetchBuffers()
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            tPipe_->InitBuffer(maskPrefetchBuf_[slot], ATTEN_MASK_UB_SIZE);
            maskReadyId_[slot] = tPipe_->AllocEventID<HardEvent::MTE2_V>();
            maskFreeId_[slot] = tPipe_->AllocEventID<HardEvent::V_MTE2>();
            SetFlag<HardEvent::V_MTE2>(maskFreeId_[slot]);
        }
    }

    __aicore__ inline void FreeMaskPrefetchEvents()
    {
        if constexpr (HAS_ATTEN && QBSA_9102_MX_M128_MASK_PREFETCH != 0) {
            for (uint32_t slot = 0U; slot < 2U; ++slot) {
                // Includes never-used slots; every ready is consumed inside ProcessVec1.
                WaitFlag<HardEvent::V_MTE2>(maskFreeId_[slot]);
                tPipe_->ReleaseEventID<HardEvent::MTE2_V>(maskReadyId_[slot]);
                tPipe_->ReleaseEventID<HardEvent::V_MTE2>(maskFreeId_[slot]);
            }
        }
    }

    template <uint32_t SUB_LOOP, bool COPY_MASKED>
    __aicore__ inline bool PrepareMaskSegments(LocalTensor<uint8_t>& mask, const MxRunInfo& runInfo,
                                               uint32_t s2CalcSize)
    {
        constexpr uint32_t begin = SUB_LOOP * S2_SPLIT;
        const uint32_t end = begin + s2CalcSize;
        bool touched = false;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockBegin = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockBegin + runInfo.sparseBlockRealSize[i];
            if (blockEnd <= begin || blockBegin >= end || runInfo.sparseBlockPartialMask[i] != COPY_MASKED) {
                continue;
            }
            const uint32_t first = blockBegin > begin ? blockBegin : begin;
            const uint32_t last = blockEnd < end ? blockEnd : end;
            if (last <= first) {
                continue;
            }
            const uint64_t token = runInfo.sparseBlockTokenOffset[i] + first - blockBegin;
            CopyAttenMaskSegment(mask, runInfo, first - begin, last - first, token, COPY_MASKED);
            touched = true;
        }
        return touched;
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline LocalTensor<uint8_t> ConsumePrefetchedAttenMask(const MxRunInfo& runInfo, uint32_t s2CalcSize)
    {
        const uint32_t slot = runInfo.vecMbaseIdx;
        WaitFlag<HardEvent::MTE2_V>(maskReadyId_[slot]);
        LocalTensor<uint8_t> mask = maskPrefetchBuf_[slot].template Get<uint8_t>();
        // Visible segments are V writes, NOT DMA. Keep them at consumption, after slot readiness.
        if (PrepareMaskSegments<SUB_LOOP, false>(mask, runInfo, s2CalcSize)) {
            PipeBarrier<PIPE_V>(); // complete Duplicate stores before the VF reads this mask
        }
        return mask;
    }

    __aicore__ inline void ReleasePrefetchedAttenMask(const MxRunInfo& runInfo)
    {
        // The MTE2 producer may overwrite the slot only after this VF has finished all reads.
        SetFlag<HardEvent::V_MTE2>(maskFreeId_[runInfo.vecMbaseIdx]);
    }

    template <bool VF_HAS_ATTEN, uint32_t SUB_LOOP>
    __aicore__ inline void ExecuteVec1Vf(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U>& bmm1ResBuf,
                                         const LocalTensor<INPUT_T>& stage1CastTensor, const LocalTensor<float>& sumUb,
                                         const LocalTensor<float>& maxUb, const LocalTensor<MM_T>& mmRes,
                                         const LocalTensor<float>& expUb, const LocalTensor<uint8_t>& attenMaskUb,
                                         const LocalTensor<SCALE_T>& pScaleSubLoopTensor,
                                         const LocalTensor<float>& preLoopMaxUb, const MxRunInfo& runInfo,
                                         const MxConstInfo& constInfo, uint32_t s2CalcSize)
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        static_assert(!VF_HAS_ATTEN || HAS_ATTEN, "masked VF requires an attention-mask kernel instance");
        // Mask preparation (if any) precedes this wait. Every path still
        // consumes C1 once and returns exactly one reverse release flag.
        bmm1ResBuf.WaitCrossCore();
        // Keep the validated recursive Sum/PScale path; only mask is specialized.
        // isUpdate selects previous logical S2 history, not subLoop 0/1.
        if (unlikely(runInfo.isFirstS2Loop)) {
            FaVectorApi::ProcessVec1VfDnMxfp8RecursiveSumFullN<MM_T, INPUT_T, false, VF_HAS_ATTEN, S2_SPLIT, SUB_LOOP>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBuf_, attenMaskUb, pScaleSubLoopTensor,
                s2CalcSize, static_cast<MM_T>(constInfo.scaleValue), pScaleValue_, negativeFloatScalar_, preLoopMaxUb);
        } else {
            FaVectorApi::ProcessVec1VfDnMxfp8RecursiveSumFullN<MM_T, INPUT_T, true, VF_HAS_ATTEN, S2_SPLIT, SUB_LOOP>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBuf_, attenMaskUb, pScaleSubLoopTensor,
                s2CalcSize, static_cast<MM_T>(constInfo.scaleValue), pScaleValue_, negativeFloatScalar_, preLoopMaxUb);
        }
        bmm1ResBuf.SetCrossCore();
    }

    __aicore__ inline float DecodeE8M0Scale(uint8_t scale) const
    {
        // E8M0 code 0 is 2^-127 (an FP32 subnormal), not FP32 zero.
        // Code 255 is the format's NaN encoding; all other codes map directly
        // to the FP32 exponent field.
        uint32_t bits = scale == 0U ? 0x00400000U : (scale == 0xFFU ? 0x7FC00000U : static_cast<uint32_t>(scale) << 23);
        return *((float*)&bits);
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline uint32_t GetSubLoopS2Size(const MxRunInfo& runInfo) const
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        if (runInfo.actSingleLoopS2Size <= S2_SPLIT) {
            return SUB_LOOP == 0U ? runInfo.actSingleLoopS2Size : 0U;
        }
        return SUB_LOOP == 0U ? S2_SPLIT : runInfo.actSingleLoopS2Size - S2_SPLIT;
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline bool IsLastSubLoop(const MxRunInfo& runInfo) const
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        return (runInfo.actSingleLoopS2Size <= S2_SPLIT) || (SUB_LOOP != 0U);
    }

    __aicore__ inline void BoolCopyInMxMask(const LocalTensor<uint8_t>& dstTensor, uint64_t srcOffset, uint32_t s2Len,
                                            uint32_t qLen)
    {
        // DN mask 按 GM row stride 搬到 [S2,64] UB。
        DataCopyParams dataCopyParams{};
        dataCopyParams.blockCount = s2Len;
        dataCopyParams.blockLen = CeilDiv(qLen, MASK_COPY_BLOCK_BYTES);
        dataCopyParams.dstStride = CeilDiv(VEC_M_BASE, MASK_COPY_BLOCK_BYTES) - dataCopyParams.blockLen;
        // Host 固定 mask 为 [2048,2048]，当前 AIV 固定搬 64 列，源/目标行宽均满足 32B 对齐。
        dataCopyParams.srcStride = (attenMaskS2Size_ - qLen) / MASK_COPY_BLOCK_BYTES;
        DataCopy(dstTensor, attenMaskGm_[srcOffset], dataCopyParams);
    }

    __aicore__ inline uint64_t ComputeCausalMaskOffset(const MxRunInfo& runInfo, uint64_t segmentS2TokenOffset) const
    {
        const int64_t deltaN = static_cast<int64_t>(runInfo.actS1Size) - static_cast<int64_t>(runInfo.actS2Size);
        const int64_t deltaCausalOrNext =
            static_cast<int64_t>(runInfo.s1Idx) - static_cast<int64_t>(segmentS2TokenOffset) - deltaN;
        const uint64_t maskOffset = static_cast<uint64_t>(
            regbaseutil::ComputeOffsetForCausal(deltaCausalOrNext, M_BASE, S2_SPLIT, attenMaskS2Size_, 0, USE_DN));
        return maskOffset;
    }

    __aicore__ inline bool CopyAttenMaskSegment(LocalTensor<uint8_t>& attenMaskUb, const MxRunInfo& runInfo,
                                                uint32_t dstS2Offset, uint32_t segmentS2Len,
                                                uint64_t segmentS2TokenOffset, bool needAtten)
    {
        // 返回值仅表示本段是否发起了 GM->UB 的 MTE2 搬运；Duplicate 写 UB 不需要 MTE2_V 同步。
        const uint64_t dstOffset = static_cast<uint64_t>(dstS2Offset) * VEC_M_BASE;
        if (unlikely(segmentS2Len == 0U)) {
            return false;
        }
        // mask_mode=3 的 MX kernel 固定使用 right-down causal mask；完全可见段直接填 1。
        if (!needAtten) {
            Duplicate<uint8_t>(attenMaskUb[dstOffset], 1, segmentS2Len * VEC_M_BASE);
            return false;
        }

        const uint64_t maskOffset = ComputeCausalMaskOffset(runInfo, segmentS2TokenOffset);
        BoolCopyInMxMask(attenMaskUb[dstOffset], maskOffset, segmentS2Len, VEC_M_BASE);
        return true;
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline LocalTensor<uint8_t> CopyAttenMaskToUb(const MxRunInfo& runInfo, uint32_t s2CalcSize)
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue_.template AllocTensor<uint8_t>();

        constexpr uint32_t subLoopStart = SUB_LOOP * S2_SPLIT;
        const uint32_t subLoopEnd = subLoopStart + s2CalcSize;
        bool needMte2ToVSync = false;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockS2Len = runInfo.sparseBlockRealSize[i];
            const uint32_t blockTileStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockTileEnd = blockTileStart + blockS2Len;
            if (blockTileEnd <= subLoopStart || blockTileStart >= subLoopEnd) {
                continue;
            }
            const uint32_t overlapStart = blockTileStart > subLoopStart ? blockTileStart : subLoopStart;
            const uint32_t overlapEnd = blockTileEnd < subLoopEnd ? blockTileEnd : subLoopEnd;
            const uint32_t segmentS2Len = overlapEnd - overlapStart;
            const uint32_t segmentOffsetInBlock = overlapStart - blockTileStart;
            const uint64_t segmentS2TokenOffset = runInfo.sparseBlockTokenOffset[i] + segmentOffsetInBlock;
            const bool hasMte2Copy =
                CopyAttenMaskSegment(attenMaskUb, runInfo, overlapStart - subLoopStart, segmentS2Len,
                                     segmentS2TokenOffset, runInfo.sparseBlockPartialMask[i]);
            needMte2ToVSync = needMte2ToVSync || hasMte2Copy;
        }
        if (needMte2ToVSync) {
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
        }
        // This queue is used only as an UB allocator.  MTE2 completion has
        // already been consumed explicitly above.  EnQue/DeQue would create a
        // second MTE2_V dependency for the same transfer; on DAV_9202 that
        // leaves the permission queue in an inconsistent state across kernel
        // launches (for example MXSTC002 followed by MXSTC003).
        return attenMaskUb;
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline void CopyPScaleToL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD, 1U>& outputBuf,
                                          LocalTensor<SCALE_T>& pScaleSubLoopTensor, const MxRunInfo& runInfo)
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        // [M128_V64] Full P[128,512], dynamic scale[1024], resident ones[1024].
        constexpr uint64_t pScaleL1Offset = static_cast<uint64_t>(CUBE_M_BASE) * S2_BASE;
        LocalTensor<SCALE_T> pScaleL1Tensor =
            outputBuf.template GetTensor<SCALE_T>(pScaleL1Offset + runInfo.vecMbaseIdx * PSCALE_SUB_LOOP_UB_SIZE);
        // Caller dequeues the joint P+scale owner before issuing either DMA.
        // 唯一 AIV 处理完整 M_BASE 行；每 MX_SCALE_GROUP 个 P 元素生成一个 scale。
        // DataCopy 以 32B 为基本块，fp8_e8m0_t 每块 32 个元素。每行有
        // S2_SPLIT / MX_SCALE_GROUP 个 scale；DN packed 后按 4 个搬运列组写入。
        constexpr uint16_t pScaleDstStride = ((S2_SPLIT / MX_SCALE_GROUP) >> 1U) - 1U;
        if constexpr (SUB_LOOP == 1U) {
            // subLoop1 生成第一半 P 的修正系数，写 PScale 前 512B；4 个 column group 起点相隔 32B。
            // Each row slice writes its own 512B in the dynamic half; the last 1024B stay constant.
            for (uint16_t i = 0U; i < 4U; ++i) {
                DataCopy(pScaleL1Tensor[i << 5U], pScaleSubLoopTensor, {4, 1, 0, pScaleDstStride});
            }
        } else if (runInfo.actSingleLoopS2Size <= S2_SPLIT) {
            // 单 subLoop 时，两次 256B 拷贝合计覆盖 PScale 前 512B。
            // The resident half is not consumed by a single-S2-half C2.
            for (uint16_t i = 0U; i < 2U; ++i) {
                DataCopy(pScaleL1Tensor[i << 8U], pScaleSubLoopTensor, {1, 8, 0, 0});
            }
        }
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline void CopyPToL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD, 1U>& outputBuf,
                                     LocalTensor<INPUT_T>& stage1CastTensor, const MxRunInfo& runInfo)
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        // 唯一 AIV 的两个 subLoop 分别写入 P 的前/后 256 列。
        LocalTensor<INPUT_T> pL1Tensor = outputBuf.template GetTensor<INPUT_T>();
        const uint64_t subLoopOffset = MxM128V64Layout::PDataOffset(SUB_LOOP, runInfo.vecMbaseIdx);
        constexpr uint16_t elementSize = 32U;
        constexpr uint32_t singleProcessSOuterSize = VEC_M_BASE;
        constexpr uint32_t actCopyCount = (singleProcessSOuterSize + elementSize - 1U) / elementSize;
        for (uint32_t i = 0U; i < actCopyCount; ++i) {
            uint64_t dstOffset = static_cast<uint64_t>(32U) * S2_SPLIT * i + subLoopOffset;
            uint64_t srcOffset = static_cast<uint64_t>(i) * (65U << 5);
            DataCopy(pL1Tensor[dstOffset], stage1CastTensor[srcOffset], {4, S2_SPLIT >> 2, (S2_SPLIT >> 2) + 2U, 0});
        }
    }

    __aicore__ inline void CopyOutAttentionOut(MxRunInfo& runInfo, const MxConstInfo& constInfo,
                                               LocalTensor<MM_T>& vec2ResUb)
    {
        // [STC060_V2_ORDER] --cce-auto-sync=off: the preceding LastDiv/FlashUpdate
        // writes this FP32 buffer. A later V_MTE3 flag cannot order these V->V
        // dependencies retroactively. Complete normalization before reading it.
        PipeBarrier<PIPE_V>();
        if constexpr (HAS_ATTEN) {
            // 使用所有 S2 loop 累计后的 max 判断整行是否始终无有效 key。
            LocalTensor<float> maxUb = softmaxMaxBuf_[runInfo.mLoop % (PRELOAD_N + 1U)]
                                           .template Get<float>()[runInfo.vecMbaseIdx * VEC_M_BASE];
            FaVectorApi::RowInvalidUpdateVF<MM_T>(vec2ResUb, maxUb, runInfo.actVecMSize, constInfo.dSizeV,
                                                  static_cast<int64_t>(dTemplateAlign64));
            // RowInvalidUpdateVF writes FP32 in-place; Cast below shrinks the
            // same allocation to BF16. Finish all FP32 stores before Cast can
            // read/write the overlapping address range.
            PipeBarrier<PIPE_V>();
        }
        // 按 TND token/head stride 写回。
        LocalTensor<OUTPUT_T> attenOut;
        attenOut.SetAddr(vec2ResUb.address_);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, static_cast<int64_t>(runInfo.actVecMSize) * dTemplateAlign64);
        // [M128_MTE3_SCOPE] Also separate forward events: row1 may publish ready
        // while MTE3 has not yet consumed row0's ready. Reuse of the SAME row's
        // ready is protected by WaitV2OutputReuse -> previous output DMA done.
        const uint32_t outputSlot = QBSA_9102_MX_M128_MTE3_SCOPE ? runInfo.vecMbaseIdx : 0U;
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[outputSlot]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[outputSlot]);

        DataCopyExtParams dataCopyParams{};
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
        dataCopyParams.blockCount = runInfo.actVecMSize;
        dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) >> 4;
        dataCopyParams.dstStride = constInfo.attentionOutStride;
        const uint64_t dstOffset =
            ((runInfo.queryTokenBase + runInfo.s1Idx) * constInfo.realN2Size + runInfo.realN2Idx) * constInfo.dSizeV;
        DataCopyPad(attentionOutGm_[dstOffset], attenOut, dataCopyParams);
    }

    __aicore__ inline void SoftmaxLseCopyOut(MxRunInfo& runInfo, const MxConstInfo& constInfo,
                                             LocalTensor<float>& sumUb, LocalTensor<float>& maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0U)) {
            return;
        }
        LocalTensor<float> lseUb = softmaxLseQueue_.template AllocTensor<float>();
        // Keep sum==0 arithmetic finite; only the final Select emits -Inf.
        FaVectorApi::ComputeLseOutputVF<float, true>(lseUb, sumUb, maxUb, runInfo.actVecMSize);
        softmaxLseQueue_.template EnQue(lseUb);
        lseUb = softmaxLseQueue_.template DeQue<float>();
        DataCopyExtParams dataCopyParams{};
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.blockCount = runInfo.actVecMSize;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = constInfo.softmaxLseStride;
        const uint64_t dstOffset = (runInfo.queryTokenBase + runInfo.s1Idx) * constInfo.realN2Size + runInfo.realN2Idx;
        DataCopyPad(softmaxLseGm_[dstOffset], lseUb, dataCopyParams);
        softmaxLseQueue_.template FreeTensor(lseUb);
    }

    TPipe* tPipe_ = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<uint8_t> attenMaskGm_;
    uint32_t attenMaskS2Size_ = 0U;
    TQue<QuePosition::VECIN, 1> attenMaskInQue_;
    TBuf<TPosition::VECIN> maskPrefetchBuf_[2];
    TEventID maskReadyId_[2]{};
    TEventID maskFreeId_[2]{};
    TQue<QuePosition::VECOUT, 1> stage1OutQue_[2];
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue_;
    TBuf<> stage2OutBuf_;
    TBuf<> softmaxMaxBuf_[PRELOAD_N + 1U];
    TBuf<> softmaxSumBuf_[PRELOAD_N + 1U];
    TBuf<> softmaxExpBuf_[PRELOAD_N + 1U];
    TBuf<> preLoopMaxBuf_;
    TBuf<> vselrIndexesBuf_[4];
    static constexpr uint32_t V2_OUTPUT_SYNC_SLOTS = QBSA_9102_MX_M128_MTE3_SCOPE ? 2U : 1U;
    TEventID mte3ToVId_[V2_OUTPUT_SYNC_SLOTS];
    bool outputDmaPending_[V2_OUTPUT_SYNC_SLOTS]{};
    TEventID vToMte3Id_[V2_OUTPUT_SYNC_SLOTS];
    MM_T negativeFloatScalar_ = 0.0f;
    float pScaleValue_ = 1.0f;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_M128_V64_BLOCK_VEC_H_
