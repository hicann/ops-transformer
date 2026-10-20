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
 * \file quant_block_sparse_attn_mx_block_cube.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化 cube 路径。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_

#include "kernel_operator.h"
#include "../common/quant_block_sparse_attn_common_arch35.h"
#include "../common/CopyInL1.h"
#include "../common/FixpipeOut.h"
#include "../common/buffer_manager.h"
#include "../common/buffers_policy.h"
#include "../common/matmul.h"
#include "../common/util_regbase_mx.h"

// [PLAN_A_RESIDENT_PSCALE_OPT] Set to 0 for the original per-C2 L1 fill.
#ifndef QBSA_9102_MX_RESIDENT_P_SCALE
#define QBSA_9102_MX_RESIDENT_P_SCALE 1
#endif
static_assert(QBSA_9102_MX_RESIDENT_P_SCALE == 0 || QBSA_9102_MX_RESIDENT_P_SCALE == 1,
              "QBSA_9102_MX_RESIDENT_P_SCALE must be 0 or 1");

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace fa_base_matmul;
using namespace regbasemx;

namespace BaseApi {
template <typename inputType, typename mmType, typename outputType, QBSALayout layout, QBSALayout kvLayout,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          DTemplateType dVTemplateType, bool isPa, bool useDn = true>
class QuantBlockSparseAttnMxBlockCube {
public:
    using INPUT_T = inputType;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = mmType;
    using MxBmm2Buf = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U>;
    using MxPBuf = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD, 1U>;

    // [PLAN_A_PREFETCH_OPT] Non-owning handles, scoped to one ExecuteTask call.
    // No new L1/UB allocation, event ID, persistent cursor or RunInfo slot.
    struct Bmm1Prefetch {
        Buffer<BufferType::L1> query;
        Buffer<BufferType::L1> key[2];
    };
    struct Bmm2Prefetch {
        Buffer<BufferType::L1> value[2];
    };

    static constexpr bool USE_DN = useDn;
    static constexpr bool IS_PA = isPa;
    static constexpr QBSALayout LAYOUT = layout;
    static constexpr QBSALayout KV_LAYOUT = kvLayout;
    static constexpr uint32_t LOGICAL_M_BASE = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t M_BASE = 64U;
    static constexpr uint32_t S2_BASE = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t D_BASE = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t DV_BASE = static_cast<uint32_t>(dVTemplateType);
    static_assert(LAYOUT == QBSALayout::TND && KV_LAYOUT == QBSALayout::PA_BNBD && IS_PA,
                  "MX cube currently only supports TND query and PA_BNBD KV");
    static_assert(LOGICAL_M_BASE == 128U && S2_BASE == 512U && D_BASE == 128U && DV_BASE == 128U,
                  "MX cube currently only supports logical S1=128, S2=512, D=128 and DV=128");
    // 每 32 个 fp8 数据元素对应一个 e8m0 scale。
    static constexpr uint32_t MX_SCALE_GROUP = 32U;
    // V scale 为 [blockNum,N,blockSize/64,DV,2]。
    static constexpr uint32_t MX_TOKEN_GROUP = 64U;
    static constexpr uint32_t MX_SCALE_LAST_DIM = 2U;
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};

    __aicore__ inline void Init(TPipe* pipe, BufferManager<BufferType::L1>* l1BuffMgr, __gm__ uint8_t* query,
                                __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* qScale, __gm__ uint8_t* kScale, __gm__ uint8_t* vScale)
    {
        if ASCEND_IS_AIC {
            tPipe_ = pipe;
            l1BufferManager_ = l1BuffMgr;
            queryGm_.SetGlobalBuffer((__gm__ INPUT_T*)query);
            keyGm_.SetGlobalBuffer((__gm__ INPUT_T*)key);
            valueGm_.SetGlobalBuffer((__gm__ INPUT_T*)value);
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
            qScaleGm_.SetGlobalBuffer((__gm__ SCALE_T*)qScale);
            kScaleGm_.SetGlobalBuffer((__gm__ SCALE_T*)kScale);
            vScaleGm_.SetGlobalBuffer((__gm__ SCALE_T*)vScale);
        }
    }

    // Initialize the existing last 512B of all three P slots, once per kernel launch.
    // Must run BEFORE InitBuffers(): its temporary MTE2_MTE1 event is returned before Q/K/V allocate theirs.
    __aicore__ inline void InitResidentPScale(
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD, 1U>& pBuffers)
    {
        if constexpr (QBSA_9102_MX_RESIDENT_P_SCALE != 0) {
            if ASCEND_IS_AIC {
                const TEventID readyId = tPipe_->AllocEventID<HardEvent::MTE2_MTE1>();
                for (uint32_t slot = 0U; slot < 3U; ++slot) {
                    // Get() has an independent cursor; three calls wrap it without touching GetCube/GetVec.
                    FillPScaleOnes(pBuffers.Get());
                }
                SetFlag<HardEvent::MTE2_MTE1>(readyId);
                WaitFlag<HardEvent::MTE2_MTE1>(readyId);
                tPipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(readyId);
            }
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIC {
            static_assert(USE_DN, "MXFullQuantMode cube currently implements DN copy/load path; "
                                  "use_dn is kept for future non-DN extension");
            // data 与 scale 连续存放，LoadData 在 L1->L0 时完成随路反量化。
            constexpr uint32_t qSize = M_BASE * D_BASE * sizeof(INPUT_T);
            constexpr uint32_t qScaleSize = M_BASE * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
            constexpr uint32_t kvSize = S2_SPLIT * D_BASE * sizeof(INPUT_T);
            constexpr uint32_t kvScaleSize = S2_SPLIT * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
            constexpr uint32_t valueScaleSize =
                (S2_SPLIT / MX_TOKEN_GROUP) * DV_BASE * MX_SCALE_LAST_DIM * sizeof(SCALE_T);

            // Q 双 buffer；K 四 buffer；V 双 buffer。
            l1QBuffers_.Init(*l1BufferManager_, qSize + qScaleSize);
            l1KVBuffers_.Init(*l1BufferManager_, kvSize + kvScaleSize);
            l1VBuffers_.Init(*l1BufferManager_, kvSize + valueScaleSize);

            // L0A/L0B 双 buffer；L0C 双 buffer。单槽 C1 为 [256,M_BASE] fp32。
            l0aBufferManager_.Init(tPipe_, 64U * 1024U);
            l0bBufferManager_.Init(tPipe_, 64U * 1024U);
            l0cBufferManager_.Init(tPipe_, 256U * 1024U);
            mmL0ABuffers_.Init(l0aBufferManager_, 32U * 1024U);
            mmL0BBuffers_.Init(l0bBufferManager_, 32U * 1024U);
            mmL0CBuffers_.Init(l0cBufferManager_, S2_SPLIT * M_BASE * sizeof(MM_T));
        }
    }

    __aicore__ inline void FreeEventID()
    {
        if ASCEND_IS_AIC {
            // Consume the final release flag of every inner-core buffer before task_reset.
            l1QBuffers_.Uninit(*l1BufferManager_);
            l1KVBuffers_.Uninit(*l1BufferManager_);
            l1VBuffers_.Uninit(*l1BufferManager_);
            mmL0ABuffers_.Uninit(l0aBufferManager_);
            mmL0BBuffers_.Uninit(l0bBufferManager_);
            mmL0CBuffers_.Uninit(l0cBufferManager_);
        }
    }

    // [PLAN_A_K1_LATE_OPT] Prepare only Q/K0 before C1(0). The paired wrapper
    // prepares K1 after C1(0), retaining the same Q and the existing ready/free protocol.
    __aicore__ inline void PrefetchBmm1Inputs(Bmm1Prefetch& inputs, const MxRunInfo& runInfo,
                                              const MxConstInfo& constInfo)
    {
        inputs.query = PrefetchQuery(runInfo, constInfo);
        inputs.key[0] = PrefetchKey<0U>(runInfo, constInfo);
    }

    __aicore__ inline void PrefetchBmm2Inputs(Bmm2Prefetch& inputs, const MxRunInfo& runInfo,
                                              const MxConstInfo& constInfo)
    {
        const uint32_t kLoops = (runInfo.actSingleLoopS2Size + S2_SPLIT - 1U) / S2_SPLIT;
        for (uint32_t kIdx = 0U; kIdx < kLoops; ++kIdx) {
            const uint32_t realK = (kIdx == kLoops - 1U) ? runInfo.actSingleLoopS2Size - kIdx * S2_SPLIT : S2_SPLIT;
            inputs.value[kIdx] = l1VBuffers_.Get();
            Buffer<BufferType::L1>& vBuf = inputs.value[kIdx];
            vBuf.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<INPUT_T> vTensor = vBuf.GetTensor<INPUT_T>();
            InitValuePadding(vTensor, realK, constInfo.dSizeV);
            CopyValueToL1(vTensor, runInfo, constInfo, realK, kIdx);
            const uint64_t vScaleOffset = static_cast<uint64_t>(AlignUp64(realK)) * constInfo.dSizeV;
            LocalTensor<SCALE_T> vScaleTensor = vBuf.GetTensor<SCALE_T>(vScaleOffset);
            CopyValueScaleToL1(vScaleTensor, runInfo, constInfo, realK, kIdx);
            if constexpr (QBSA_9102_MX_RESIDENT_P_SCALE != 0) {
                vBuf.Set<HardEvent::MTE2_MTE1>();
            }
        }
    }

    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U>& outputBuf,
                                       MxRunInfo& runInfo, const MxConstInfo& constInfo, Bmm1Prefetch& inputs)
    {
        IterateBmm1Impl<0U, true, true>(outputBuf, runInfo, constInfo, inputs.query, inputs.key[0]);
    }

    __aicore__ inline void IterateBmm1ReuseQ(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U>& outputBuf0,
                                             Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U>& outputBuf1,
                                             MxRunInfo& runInfo, const MxConstInfo& constInfo, Bmm1Prefetch& inputs)
    {
        // A 257..512-token C1 task shares Q/QScale between its two subLoops. Keep the exact Q tile in L0B:
        // subLoop0 loads without retiring it; subLoop1 reuses and retires it after the second MMAD.
        IterateBmm1Impl<0U, true, false>(outputBuf0, runInfo, constInfo, inputs.query, inputs.key[0]);
        // [PLAN_A_K1_LATE_OPT] Do not place K1 PA preparation on C1(0)'s issue path.
        inputs.key[1] = PrefetchKey<1U>(runInfo, constInfo);
        IterateBmm1Impl<1U, false, true>(outputBuf1, runInfo, constInfo, inputs.query, inputs.key[1]);
    }

private:
    __aicore__ inline Buffer<BufferType::L1> PrefetchQuery(const MxRunInfo& runInfo, const MxConstInfo& constInfo)
    {
        Buffer<BufferType::L1> qBuf;
        // V1 and C2 consume all 64 query lanes, including a short M-part. Keep the Q NZ stride,
        // scale offset and C1 N dimension at 64 so every lane has defined data and scale.
        const uint64_t qScaleOffset = static_cast<uint64_t>(M_BASE) * constInfo.dSize;
        if (unlikely(runInfo.isFirstS2Loop)) {
            // Q 用 ND2NZ，QScale 用 DN2NZ；后续 S2 loop 复用。
            qBuf = l1QBuffers_.Get();
            qBuf.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<INPUT_T> qTensor = qBuf.GetTensor<INPUT_T>();
            LocalTensor<SCALE_T> qScaleTensor = qBuf.GetTensor<SCALE_T>(qScaleOffset);
            InitMxRowPadding(qTensor, qScaleTensor, runInfo.actMSize, M_BASE, constInfo.dSize);
            // Use an explicit destination row stride. CopyToL1Nd2Nz's legacy architecture branch
            // can otherwise choose Align16(realM), while MX LoadData expects an Align32 stride.
            GlobalTensor<INPUT_T> qSrcGm = queryGm_[runInfo.queryOffset];
            DataCopyGmNDToL1(qTensor, qSrcGm, runInfo.actMSize, M_BASE, constInfo.dSize, constInfo.n2GD);
            MxCopyScaleToL1Dn2Nz<SCALE_T>(qScaleTensor, qScaleGm_[runInfo.queryScaleOffset],
                                          constInfo.queryScaleDSize * constInfo.scaleLastDim, runInfo.actMSize,
                                          constInfo.qScaleN1D);
            qBuf.Set<HardEvent::MTE2_MTE1>();
        } else {
            // A reused Q has no new MTE2 transfer and must not publish another ready flag.
            qBuf = l1QBuffers_.GetPre();
        }
        return qBuf;
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline Buffer<BufferType::L1> PrefetchKey(const MxRunInfo& runInfo, const MxConstInfo& constInfo)
    {
        const uint32_t s2CalcSize = GetSubLoopS2Size<SUB_LOOP>(runInfo);
        Buffer<BufferType::L1> kBuf = l1KVBuffers_.Get();
        kBuf.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<INPUT_T> kTensor = kBuf.GetTensor<INPUT_T>();
        const uint64_t kScaleOffset = static_cast<uint64_t>(AlignUp32(s2CalcSize)) * constInfo.dSize;
        LocalTensor<SCALE_T> kScaleTensor = kBuf.GetTensor<SCALE_T>(kScaleOffset);
        // LoadData rounds the tail to a fractal; MMAD also promotes M=1 to M=16.
        // Padding must be initialized before those lanes are read, even if V1 later masks them.
        InitMxRowPadding(kTensor, kScaleTensor, s2CalcSize, AlignUp32(s2CalcSize), constInfo.dSize);
        // [PLAN_A_PA_REUSE_OPT] K/KScale 共用 sparse overlap 和 PA 页号，分别搬运 data/scale。
        CopyKeyAndScaleToL1<SUB_LOOP>(kTensor, kScaleTensor, runInfo, constInfo, s2CalcSize);
        kBuf.Set<HardEvent::MTE2_MTE1>();
        return kBuf;
    }

    template <uint32_t SUB_LOOP, bool LOAD_Q_TO_L0B, bool RELEASE_Q_L0B>
    __aicore__ inline void IterateBmm1Impl(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U>& outputBuf,
                                           MxRunInfo& runInfo, const MxConstInfo& constInfo,
                                           Buffer<BufferType::L1>& qBuf, Buffer<BufferType::L1>& kBuf)
    {
        static_assert(SUB_LOOP < 2U, "MX C1 only supports subLoop 0/1");
        const uint32_t s2CalcSize = GetSubLoopS2Size<SUB_LOOP>(runInfo);
        const uint64_t qScaleOffset = static_cast<uint64_t>(M_BASE) * constInfo.dSize;
        const uint64_t kScaleOffset = static_cast<uint64_t>(AlignUp32(s2CalcSize)) * constInfo.dSize;
        LocalTensor<INPUT_T> kTensor = kBuf.GetTensor<INPUT_T>();
        LocalTensor<SCALE_T> kScaleTensor = kBuf.GetTensor<SCALE_T>(kScaleOffset);
        kBuf.Wait<HardEvent::MTE2_MTE1>();
        if constexpr (LOAD_Q_TO_L0B) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                qBuf.Wait<HardEvent::MTE2_MTE1>();
            }
        }

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers_.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();
        // MxMatmulFull 同时消费 fp8 data 与 e8m0 scale。
        MMParam param = MxMakeMMParam(s2CalcSize, M_BASE, constInfo.dSize, false, true);
        if constexpr (LOAD_Q_TO_L0B && RELEASE_Q_L0B) {
            MxMatmulFull<INPUT_T, INPUT_T, MM_T, S2_SPLIT, M_BASE, D_BASE, ABLayout::MK, ABLayout::KN,
                         BuffersPolicyDB<BufferType::L0A>, BuffersPolicyDB<BufferType::L0B>, SCALE_T, SCALE_T,
                         mx_fp8_e4m3_t, mx_fp8_e4m3_t>(kTensor, qBuf.template GetTensor<INPUT_T>(), mmL0ABuffers_,
                                                       mmL0BBuffers_, mm1ResL0C.template GetTensor<MM_T>(), param,
                                                       kScaleTensor, qBuf.template GetTensor<SCALE_T>(qScaleOffset));
        } else {
            MxMatmulFullReuseB<LOAD_Q_TO_L0B, RELEASE_Q_L0B, INPUT_T, INPUT_T, MM_T, S2_SPLIT, M_BASE, D_BASE,
                               ABLayout::MK, ABLayout::KN, BuffersPolicyDB<BufferType::L0A>,
                               BuffersPolicyDB<BufferType::L0B>, SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                kTensor, qBuf.template GetTensor<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                mm1ResL0C.template GetTensor<MM_T>(), param, kScaleTensor,
                qBuf.template GetTensor<SCALE_T>(qScaleOffset));
        }

        if constexpr (RELEASE_Q_L0B) {
            if (unlikely(runInfo.isLastS2Loop)) {
                qBuf.Set<HardEvent::MTE1_MTE2>();
            }
        }
        kBuf.Set<HardEvent::MTE1_MTE2>();
        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();
        FixpipeMm1(outputBuf.template GetTensor<MM_T>(), mm1ResL0C.GetTensor<MM_T>(), runInfo, s2CalcSize);
        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

public:
    __aicore__ inline void IterateBmm2(
        MxBmm2Buf& outputBuf, BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD, 1U>& inputBuf,
        MxRunInfo& runInfo, const MxConstInfo& constInfo, Bmm2Prefetch& inputs)
    {
        // C2: P[64,512] x V[512,DV]，P/V 均携带 e8m0 scale。
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD, 1U> pBuf = inputBuf.GetCube();
        pBuf.WaitCrossCore();
        constexpr uint64_t pScaleOffset = static_cast<uint64_t>(M_BASE) * S2_BASE;
        if constexpr (QBSA_9102_MX_RESIDENT_P_SCALE == 0) {
            if (runInfo.actSingleLoopS2Size > S2_SPLIT) {
                FillPScaleOnes(pBuf);
            }
        }
        LocalTensor<SCALE_T> pScaleTensor = pBuf.template GetTensor<SCALE_T>(pScaleOffset);

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers_.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();
        constexpr uint32_t baseK = S2_SPLIT;
        constexpr uint64_t l1BaseKOffset = static_cast<uint64_t>(baseK) * M_BASE;
        constexpr uint64_t l1ScaleOffset = l1BaseKOffset / MX_SCALE_GROUP;
        const uint32_t kLoops = (runInfo.actSingleLoopS2Size + baseK - 1U) / baseK;
        for (uint32_t kIdx = 0U; kIdx < kLoops; ++kIdx) {
            const uint32_t realK = (kIdx == kLoops - 1U) ? runInfo.actSingleLoopS2Size - kIdx * baseK : baseK;
            Buffer<BufferType::L1>& vBuf = inputs.value[kIdx];
            LocalTensor<INPUT_T> vTensor = vBuf.GetTensor<INPUT_T>();
            const uint64_t vScaleOffset = static_cast<uint64_t>(AlignUp64(realK)) * constInfo.dSizeV;
            LocalTensor<SCALE_T> vScaleTensor = vBuf.GetTensor<SCALE_T>(vScaleOffset);
            if constexpr (QBSA_9102_MX_RESIDENT_P_SCALE == 0) {
                // [PLAN_A_COMBINED_OPT] V was prefetched before the per-task PScale fill.
                // Publish its existing ready event here so it also orders that later L1 write.
                // The producer suppresses this Set in this mode; each V slot still has one Set/Wait.
                vBuf.Set<HardEvent::MTE2_MTE1>();
            }
            vBuf.Wait<HardEvent::MTE2_MTE1>();

            MMParam param =
                MxMakeMMParam(M_BASE, constInfo.dSizeV, AlignUp64(realK), true, false, kIdx == 0U, kIdx == 0U);
            // P 转置，V 保持 KN。
            MxMatmulFull<INPUT_T, INPUT_T, MM_T, M_BASE, DV_BASE, baseK, ABLayout::MK, ABLayout::KN,
                         BuffersPolicyDB<BufferType::L0A>, BuffersPolicyDB<BufferType::L0B>, SCALE_T, SCALE_T,
                         mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                pBuf.template GetTensor<INPUT_T>()[kIdx * l1BaseKOffset], vTensor, mmL0ABuffers_, mmL0BBuffers_,
                mm2ResL0C.GetTensor<MM_T>(), param, pScaleTensor[kIdx * l1ScaleOffset], vScaleTensor);
            vBuf.Set<HardEvent::MTE1_MTE2>();
        }

        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();
        outputBuf.WaitCrossCore();
        FixpipeMm2(outputBuf.template GetTensor<MM_T>(), mm2ResL0C.GetTensor<MM_T>(), runInfo, constInfo);
        mm2ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

private:
    __aicore__ inline void FillPScaleOnes(MxPBuf& pBuf)
    {
        // P: [0,32768), dynamic PScale: [32768,33280), constant PScale: [33280,33792).
        // Both V1 subLoop paths only write the dynamic half. E8M0 1.0 is byte 0x7f.
        constexpr uint64_t pScaleOffset = static_cast<uint64_t>(M_BASE) * S2_BASE;
        constexpr uint32_t int16Bytes = 2U;
        LocalTensor<int16_t> pScalePadTensor = pBuf.template GetTensor<int16_t>(
            pScaleOffset / int16Bytes + M_BASE * S2_SPLIT / MX_SCALE_GROUP / int16Bytes);
        InitConstValue(pScalePadTensor, {1, M_BASE * S2_SPLIT / MX_SCALE_GROUP * sizeof(SCALE_T) / 32U, 0, 0x7f7f});
    }

    __aicore__ inline uint32_t AlignUp32(uint32_t value) const
    {
        return (value + 31U) >> 5 << 5;
    }

    __aicore__ inline uint32_t AlignUp64(uint32_t value) const
    {
        return (value + 63U) >> 6 << 6;
    }

    __aicore__ inline void InitMxRowPadding(const LocalTensor<INPUT_T>& dataL1, const LocalTensor<SCALE_T>& scaleL1,
                                            uint32_t realRows, uint32_t storageRows, uint32_t headDim)
    {
        if (likely(realRows == storageRows)) {
            return;
        }
        // Initialize the existing, row-aligned data+scale views before copying valid GM rows.
        // A1 InitConstValue uses 32-byte blocks. 0x7f is E8M0 1.0, not a zero scale.
        LocalTensor<int16_t> dataBits = dataL1.template ReinterpretCast<int16_t>();
        LocalTensor<int16_t> scaleBits = scaleL1.template ReinterpretCast<int16_t>();
        InitConstValueParams<int16_t> dataInit{};
        dataInit.repeatTimes = 1U;
        dataInit.blockNum = storageRows * headDim * sizeof(INPUT_T) / 32U;
        dataInit.initValue = 0;
        InitConstValue(dataBits, dataInit);
        InitConstValueParams<int16_t> scaleInit{};
        scaleInit.repeatTimes = 1U;
        scaleInit.blockNum = storageRows * headDim / MX_SCALE_GROUP * sizeof(SCALE_T) / 32U;
        scaleInit.initValue = 0x7f7f;
        InitConstValue(scaleBits, scaleInit);
        // Both initialization and the following GM copy write L1 through MTE2.
        // Finish initialization before the copy overwrites the valid rows; the existing buffer
        // MTE2_MTE1 flag then orders both writes before MX LoadData.
        PipeBarrier<PIPE_MTE2>();
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline uint32_t GetSubLoopS2Size(const MxRunInfo& runInfo) const
    {
        static_assert(SUB_LOOP < 2U, "MX C1 only supports subLoop 0/1");
        if (runInfo.actSingleLoopS2Size <= S2_SPLIT) {
            return SUB_LOOP == 0U ? runInfo.actSingleLoopS2Size : 0U;
        }
        return SUB_LOOP == 0U ? S2_SPLIT : runInfo.actSingleLoopS2Size - S2_SPLIT;
    }

    __aicore__ inline uint32_t GetBlockOverlapStart(uint32_t tileStart, uint32_t blockStart) const
    {
        return tileStart > blockStart ? tileStart : blockStart;
    }

    __aicore__ inline uint32_t GetBlockOverlapEnd(uint32_t tileEnd, uint32_t blockEnd) const
    {
        return tileEnd < blockEnd ? tileEnd : blockEnd;
    }

    // [PLAN_A_PA_REUSE_OPT] K and KScale cover the same token/page range.
    // Resolve each physical PA page once, then use the existing ND2NZ/DN2NZ copies.
    // This private helper is only called for the BNBD K/KScale shapes constructed below.
    __aicore__ inline void CopyKeyAndScalePageRangeToL1(LocalTensor<INPUT_T>& keyDst, LocalTensor<SCALE_T>& scaleDst,
                                                        const PAShape& keyShape, const PAShape& scaleShape,
                                                        const Position& startPos)
    {
        const uint64_t tableBase = static_cast<uint64_t>(startPos.bIdx) * keyShape.maxblockNumPerBatch;
        constexpr uint32_t blockElements = 32U / sizeof(INPUT_T);
        uint32_t finished = 0U;
        uint64_t token = startPos.s2Offset;
        while (finished < keyShape.copyRowNum) {
            const uint64_t logicalPage = token / keyShape.blockSize;
            const uint32_t inPage = static_cast<uint32_t>(token % keyShape.blockSize);
            const uint64_t physicalPage = blockTableGm_.GetValue(tableBase + logicalPage);
            uint32_t rows = keyShape.blockSize - inPage;
            if (finished + rows > keyShape.copyRowNum) {
                rows = keyShape.copyRowNum - finished;
            }
            const uint64_t keyOffset = physicalPage * keyShape.pageStride +
                                       static_cast<uint64_t>(startPos.n2Idx) * keyShape.blockSize * keyShape.rowStride +
                                       static_cast<uint64_t>(inPage) * keyShape.rowStride + startPos.dIdx;
            LocalTensor<INPUT_T> keyPart = keyDst[finished * blockElements];
            GlobalTensor<INPUT_T> keySource = keyGm_[keyOffset];
            DataCopyGmNDToL1(keyPart, keySource, rows, keyShape.copyRowNumAlign, keyShape.actHeadDim,
                             keyShape.rowStride);

            const uint64_t scaleOffset =
                physicalPage * scaleShape.pageStride +
                static_cast<uint64_t>(startPos.n2Idx) * scaleShape.blockSize * scaleShape.rowStride +
                static_cast<uint64_t>(inPage) * scaleShape.rowStride + startPos.dIdx;
            MxCopyScaleToL1Dn2Nz(scaleDst[finished * scaleShape.actHeadDim], kScaleGm_[scaleOffset],
                                 scaleShape.actHeadDim, rows, scaleShape.rowStride);
            finished += rows;
            token += rows;
        }
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline void CopyKeyAndScaleToL1(LocalTensor<INPUT_T>& keyDstTensor,
                                               LocalTensor<SCALE_T>& keyScaleDstTensor, const MxRunInfo& runInfo,
                                               const MxConstInfo& constInfo, uint32_t s2CalcSize)
    {
        static_assert(SUB_LOOP < 2U, "MX C1 only supports subLoop 0/1");
        // PA BNBD K 与 KScale 使用相同的 sparse 顺序和 token overlap。
        constexpr uint32_t tileStart = SUB_LOOP * S2_SPLIT;
        const uint32_t tileEnd = tileStart + s2CalcSize;
        const uint32_t copyRowNumAlign = AlignUp32(s2CalcSize);
        constexpr uint32_t blockElementCnt = 32U / sizeof(INPUT_T);

        Position startPos{};
        startPos.bIdx = runInfo.bIdx;
        startPos.n2Idx = runInfo.n2Idx;
        startPos.dIdx = 0U;

        PAShape keyShape{};
        keyShape.blockSize = constInfo.paBlockSize;
        keyShape.headNum = constInfo.n2Size;
        keyShape.headDim = constInfo.dSize;
        keyShape.actHeadDim = constInfo.dSize;
        keyShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        keyShape.copyRowNumAlign = copyRowNumAlign;
        keyShape.pageStride = constInfo.paBlockStride;
        keyShape.rowStride = constInfo.dSize;

        // KScale GM 为 [physicalBlock,kvHeads,blockSize,D/64,2]，单行使用 DN2NZ。
        PAShape keyScaleShape{};
        keyScaleShape.blockSize = constInfo.paBlockSize;
        keyScaleShape.headNum = constInfo.n2Size;
        keyScaleShape.headDim = constInfo.keyScaleDSize * constInfo.scaleLastDim;
        keyScaleShape.actHeadDim = keyScaleShape.headDim;
        keyScaleShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        keyScaleShape.copyRowNumAlign = copyRowNumAlign;
        keyScaleShape.pageStride = constInfo.n2Size * constInfo.kScaleN2D;
        keyScaleShape.rowStride = keyScaleShape.headDim;

        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockStart + blockRealSize;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;

            startPos.s2Offset = runInfo.sparseBlockTokenOffset[i] + copyStartInBlock;
            keyShape.copyRowNum = copyRowNum;
            LocalTensor<INPUT_T> keyBlockDstTensor = keyDstTensor[(overlapStart - tileStart) * blockElementCnt];

            keyScaleShape.copyRowNum = copyRowNum;
            LocalTensor<SCALE_T> keyScaleBlockDstTensor =
                keyScaleDstTensor[(overlapStart - tileStart) * keyScaleShape.actHeadDim];
            CopyKeyAndScalePageRangeToL1(keyBlockDstTensor, keyScaleBlockDstTensor, keyShape, keyScaleShape, startPos);
        }
    }

    __aicore__ inline void CopyValueToL1(LocalTensor<INPUT_T>& dstTensor, const MxRunInfo& runInfo,
                                         const MxConstInfo& constInfo, uint32_t realK, uint32_t kIdx)
    {
        // V 使用与 K 相同的 PA 拼接顺序。
        const uint32_t tileStart = kIdx * S2_SPLIT;
        const uint32_t tileEnd = tileStart + realK;
        constexpr uint32_t blockElementCnt = 32U / sizeof(INPUT_T);

        Position startPos{};
        startPos.bIdx = runInfo.bIdx;
        startPos.n2Idx = runInfo.n2Idx;
        startPos.dIdx = 0U;

        PAShape valueShape{};
        valueShape.blockSize = constInfo.paBlockSize;
        valueShape.headNum = constInfo.n2Size;
        valueShape.headDim = constInfo.dSizeV;
        valueShape.actHeadDim = constInfo.dSizeV;
        valueShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        valueShape.copyRowNumAlign = AlignUp64(realK);
        valueShape.pageStride = constInfo.paBlockStride;
        valueShape.rowStride = constInfo.dSizeV;

        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockStart + blockRealSize;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;

            startPos.s2Offset = runInfo.sparseBlockTokenOffset[i] + copyStartInBlock;
            valueShape.copyRowNum = copyRowNum;
            LocalTensor<INPUT_T> valueDstTensor = dstTensor[(overlapStart - tileStart) * blockElementCnt];
            GmCopyInToL1PA<INPUT_T>(valueDstTensor, valueGm_, blockTableGm_, KVLAYOUT::BNBD, valueShape, startPos);
        }
    }

    __aicore__ inline void CopyValueScaleToL1(LocalTensor<SCALE_T>& dstTensor, const MxRunInfo& runInfo,
                                              const MxConstInfo& constInfo, uint32_t realK, uint32_t kIdx)
    {
        // VScale GM 为 [physicalBlock, kvHeads, blockSize/64, DV, 2]，offset 单位为 64-token group。
        const uint32_t tileStart = kIdx * S2_SPLIT;
        const uint32_t tileEnd = tileStart + realK;
        const uint32_t totalGroupCount = AlignUp64(realK) / MX_TOKEN_GROUP;
        const uint32_t scaleBlockSize = constInfo.paBlockSize / MX_TOKEN_GROUP;
        constexpr uint32_t blockElementCnt = 32U / sizeof(SCALE_T);
        uint32_t copiedGroupCount = 0U;

        Position startPos{};
        startPos.bIdx = runInfo.bIdx;
        startPos.n2Idx = runInfo.n2Idx;
        startPos.dIdx = 0U;

        PAShape valueScaleShape{};
        valueScaleShape.blockSize = scaleBlockSize;
        valueScaleShape.headNum = constInfo.n2Size;
        valueScaleShape.headDim = constInfo.valueScaleDSize;
        valueScaleShape.actHeadDim = constInfo.valueScaleDSize;
        valueScaleShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        valueScaleShape.copyRowNumAlign = totalGroupCount;
        valueScaleShape.pageStride = constInfo.n2Size * constInfo.valueScaleN2D;
        valueScaleShape.rowStride = valueScaleShape.headDim;

        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockStart + blockRealSize;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;
            const uint32_t groupStartInBlock = copyStartInBlock / MX_TOKEN_GROUP;
            const uint32_t groupEndInBlock = (copyStartInBlock + copyRowNum + MX_TOKEN_GROUP - 1U) / MX_TOKEN_GROUP;
            const uint32_t copyGroupNum = groupEndInBlock - groupStartInBlock;

            startPos.s2Offset = runInfo.sparseBlockTokenOffset[i] / MX_TOKEN_GROUP + groupStartInBlock;
            valueScaleShape.copyRowNum = copyGroupNum;
            LocalTensor<SCALE_T> valueScaleDstTensor = dstTensor[copiedGroupCount * blockElementCnt];
            MxGmFlatScaleCopyInToL1PAForND<SCALE_T>(valueScaleDstTensor, vScaleGm_, blockTableGm_, valueScaleShape,
                                                    startPos);
            copiedGroupCount += copyGroupNum;
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<MM_T>& dstTensor, const LocalTensor<MM_T>& l0C,
                                      const MxRunInfo&, uint32_t s2RealSize)
    {
        // C1 写为 VF 使用的 row-major UB layout。
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams{};
        fixpipeParams.nSize = M_BASE;
        fixpipeParams.mSize = (s2RealSize + 7U) >> 3 << 3;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15U) >> 4) << 4U;
        fixpipeParams.dstStride = 64U;
        // arch92 的唯一 AIV 消费完整 N=64 结果，不做双目标拆分。
        fixpipeParams.dualDstCtl = 0U;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void FixpipeMm2(const LocalTensor<MM_T>& dstTensor, const LocalTensor<MM_T>& l0C,
                                      const MxRunInfo&, const MxConstInfo& constInfo)
    {
        // C2 以 fp32 写入 UB，供 V2 update。
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams{};
        fixpipeParams.nSize = (constInfo.dSizeV + 7U) >> 3 << 3;
        fixpipeParams.mSize = M_BASE;
        fixpipeParams.srcStride = ((M_BASE + 15U) >> 4) << 4U;
        fixpipeParams.dstStride = (DV_BASE + 15U) >> 4 << 4;
        // arch92 的唯一 AIV 消费完整 M=64 结果，不做双目标拆分。
        fixpipeParams.dualDstCtl = 0U;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void InitValuePadding(const LocalTensor<INPUT_T>& valueL1, uint32_t realK, uint32_t curN)
    {
        // V tail 补零到 64-token 对齐。
        const uint32_t curPadK = AlignUp64(realK);
        if (likely(curPadK == realK)) {
            return;
        }
        InitConstValueParams<half> initConstValueParams{};
        initConstValueParams.repeatTimes = AlignUp64(curN) >> 5;
        initConstValueParams.blockNum = curPadK - realK;
        initConstValueParams.dstGap = realK;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>()[realK << 4U], initConstValueParams);
    }

    TPipe* tPipe_ = nullptr;
    BufferManager<BufferType::L1>* l1BufferManager_ = nullptr;
    GlobalTensor<INPUT_T> queryGm_;
    GlobalTensor<INPUT_T> keyGm_;
    GlobalTensor<INPUT_T> valueGm_;
    GlobalTensor<SCALE_T> qScaleGm_;
    GlobalTensor<SCALE_T> kScaleGm_;
    GlobalTensor<SCALE_T> vScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;

    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    BufferManager<BufferType::L0C> l0cBufferManager_;
    BuffersPolicyDB<BufferType::L1> l1QBuffers_;
    BuffersPolicy4buff<BufferType::L1> l1KVBuffers_;
    BuffersPolicyDB<BufferType::L1> l1VBuffers_;
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers_;
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers_;
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers_;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_
