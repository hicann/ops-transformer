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
 * \file quant_block_sparse_attn_mx_block_vec.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化 vector 路径。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_VEC_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_VEC_H_

#include "kernel_operator.h"
#include "../quant_block_sparse_attn_mx_tiling_data.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "quant_block_sparse_attn_attenmask.h"
#include "common/buffer_manager.h"
#include "common/buffers_policy.h"
#include "common/util_regbase_mx.h"
#include "vf/vf_div_cast.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbasemx;

namespace BaseApi {
// V1 执行 softmax/P 量化，V2 更新并写回 attention output/LSE。
template <typename inputType, typename mmType, typename outputType, QBSALayout layout, QBSALayout kvLayout,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          DTemplateType dVTemplateType, bool hasAtten, bool hasLse, bool isPa, bool useDn = true>
class QuantBlockSparseAttnMxBlockVec {
public:
    using INPUT_T = inputType;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = mmType;
    using OUTPUT_T = outputType;
    using MxBmm2Buf = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_ATTEN = hasAtten;
    static constexpr bool HAS_LSE = hasLse;
    static constexpr bool IS_PA = isPa;
    static constexpr QBSALayout LAYOUT = layout;
    static constexpr QBSALayout KV_LAYOUT = kvLayout;
    static constexpr uint32_t M_BASE = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t S2_BASE = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t VEC_M_BASE = M_BASE >> 1U;
    static constexpr uint32_t D_BASE = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t DV_BASE = static_cast<uint32_t>(dVTemplateType);
    static_assert(LAYOUT == QBSALayout::TND && KV_LAYOUT == QBSALayout::PA_BNBD && IS_PA,
                  "MX vector currently only supports TND query and PA_BNBD KV");
    static_assert(M_BASE == 128U && S2_BASE == 512U && D_BASE == 128U && DV_BASE == 128U,
                  "MX vector currently only supports S1=128, S2=512, D=128 and DV=128");
    // V2 比 V1 延迟三个 task，四槽状态避免 V1 覆盖尚未被 V2 消费的 softmax 状态。
    static constexpr uint32_t SOFTMAX_BUFFER_NUM = 4U;
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
    static constexpr uint32_t SOFTMAX_STATE_UB_SIZE = VEC_M_BASE * sizeof(float);

    __aicore__ inline void Init(TPipe *pipe, __gm__ uint8_t *pScale, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *attenMask, uint32_t attenMaskS2Size,
                                uint8_t pScaleDtype)
    {
        if ASCEND_IS_AIV {
            tPipe_ = pipe;
            attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
            if constexpr (HAS_LSE) {
                softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
            }
            if (pScale != nullptr) {
                if (pScaleDtype == optiling::MX_PSCALE_DTYPE_FP32) {
                    GlobalTensor<float> pScaleGmFp32;
                    pScaleGmFp32.SetGlobalBuffer((__gm__ float *)pScale);
                    pScaleValue_ = pScaleGmFp32.GetValue(0);
                } else {
                    GlobalTensor<uint8_t> pScaleGm;
                    pScaleGm.SetGlobalBuffer((__gm__ uint8_t *)pScale);
                    pScaleValue_ = DecodeE8M0Scale(pScaleGm.GetValue(0));
                }
            }
            if constexpr (HAS_ATTEN) {
                attenMaskGm_.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
                attenMaskS2Size_ = attenMaskS2Size;
            }
            negativeFloatScalar_ = *((const MM_T *)&NEGATIVE_MIN_VALUE_FP32);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // V2 累计输出：[VEC_M_BASE,dTemplateAlign64] fp32。
            tPipe_->InitBuffer(stage2OutBuf_, VEC_M_BASE * dTemplateAlign64 * sizeof(MM_T));
            // V1 DN packed P 使用双队列。
            tPipe_->InitBuffer(stage1OutQue_[0], 1, STAGE1_OUT_UB_SIZE);
            tPipe_->InitBuffer(stage1OutQue_[1], 1, STAGE1_OUT_UB_SIZE);
            // 单个 256-column subLoop 的 e8m0 PScale。
            tPipe_->InitBuffer(pScaleSubLoop0Que_, 1, PSCALE_SUB_LOOP_UB_SIZE);
            // 四槽 softmax 状态覆盖 current V1 与延迟三个 task 的 V2。
            tPipe_->InitBuffer(softmaxSumBuf_[0], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxSumBuf_[1], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxSumBuf_[2], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxSumBuf_[3], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxMaxBuf_[0], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxMaxBuf_[1], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxMaxBuf_[2], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxMaxBuf_[3], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxExpBuf_[0], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxExpBuf_[1], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxExpBuf_[2], SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(softmaxExpBuf_[3], SOFTMAX_STATE_UB_SIZE);
            // online softmax update 临时状态。
            tPipe_->InitBuffer(preLoopMaxBuf_, SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(preLoopSumBuf_, SOFTMAX_STATE_UB_SIZE);
            tPipe_->InitBuffer(firstLoopSumBuf_, SOFTMAX_STATE_UB_SIZE);
            if constexpr (HAS_LSE) {
                tPipe_->InitBuffer(softmaxLseQueue_, 1, (M_BASE >> 1U) * sizeof(float) * 8U);
            }
            if constexpr (HAS_ATTEN) {
                // DN mask UB：[256,64]，按 sparse segment 拼接。
                tPipe_->InitBuffer(attenMaskInQue_, 1, ATTEN_MASK_UB_SIZE);
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

            // 同步 V2 cast 与 GM copy。
            mte3ToVId_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
            vToMte3Id_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
            SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        }
    }

    __aicore__ inline void ClearEmptyQBlock(uint32_t n1Idx, uint32_t queryTokenBase, uint32_t actualS1Size,
                                            uint32_t s1OuterIdx, const MxConstInfo &constInfo)
    {
        if ASCEND_IS_AIV {
            const uint32_t s1Idx = s1OuterIdx * constInfo.qSparseBlockSize;
            uint32_t actMSize = constInfo.qSparseBlockSize;
            if (unlikely(s1Idx + actMSize > actualS1Size)) {
                actMSize = actualS1Size - s1Idx;
            }
            const uint32_t actMSizeAlign32 = (actMSize + 31U) >> 5U << 5U;
            uint32_t actVecMSize = actMSize <= 16U ? actMSize : (actMSizeAlign32 >> 1U);
            uint32_t vecMbaseIdx = 0U;
            if (constInfo.subBlockIdx == 1U) {
                vecMbaseIdx = actVecMSize;
                actVecMSize = actMSize - actVecMSize;
            }
            if (unlikely(actVecMSize == 0U)) {
                return;
            }

            DataCopyExtParams outCopyParams;
            outCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
            outCopyParams.blockCount = actVecMSize;
            outCopyParams.srcStride = 0U;
            outCopyParams.dstStride = constInfo.attentionOutStride;
            const uint64_t tokenOffset = static_cast<uint64_t>(queryTokenBase) + s1Idx + vecMbaseIdx;
            const uint64_t outOffset = (tokenOffset * constInfo.realN2Size + n1Idx) * constInfo.dSizeV;
            // 空行是冷路径，复用 V1 queue 作为临时零块，避免常驻占用 16 KiB UB。
            LocalTensor<OUTPUT_T> emptyOut = stage1OutQue_[0].template AllocTensor<OUTPUT_T>();
            Duplicate<OUTPUT_T>(emptyOut, 0.0F, actVecMSize * constInfo.dSizeV);
            stage1OutQue_[0].template EnQue(emptyOut);
            emptyOut = stage1OutQue_[0].template DeQue<OUTPUT_T>();
            DataCopyPad(attentionOutGm_[outOffset], emptyOut, outCopyParams);
            stage1OutQue_[0].template FreeTensor(emptyOut);

            if constexpr (HAS_LSE) {
                DataCopyExtParams lseCopyParams;
                lseCopyParams.blockLen = sizeof(float);
                lseCopyParams.blockCount = actVecMSize;
                lseCopyParams.srcStride = 0U;
                lseCopyParams.dstStride = constInfo.softmaxLseStride;
                const uint64_t lseOffset = tokenOffset * constInfo.realN2Size + n1Idx;
                LocalTensor<float> emptyLse = softmaxLseQueue_.template AllocTensor<float>();
                // UB->GM 以 4B block 搬运时，每行源数据占一个 32B data block。
                // 与 ComputeLseOutputVF 的 E2B 布局保持一致，避免第 2 行起读取未初始化的 32B 槽位。
                Duplicate<float>(emptyLse, QBSA_EMPTY_LSE_VALUE, actVecMSize * 8U);
                softmaxLseQueue_.template EnQue(emptyLse);
                emptyLse = softmaxLseQueue_.template DeQue<float>();
                DataCopyPad(softmaxLseGm_[lseOffset], emptyLse, lseCopyParams);
                softmaxLseQueue_.template FreeTensor(emptyLse);
            }
        }
    }

    // V1：C1[128,256] -> P/PScale；两个 subLoop 写成 P[128,512]。
    template <uint32_t SUB_LOOP>
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       MxRunInfo &runInfo, const MxConstInfo &constInfo)
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        const bool isLastSubLoop = IsLastSubLoop<SUB_LOOP>(runInfo);
        if (unlikely(runInfo.actVecMSize == 0U)) {
            bmm1ResBuf.WaitCrossCore();
            bmm1ResBuf.SetCrossCore();
            if (isLastSubLoop) {
                outputBuf.SetCrossCore();
            }
            return;
        }

        const uint32_t softmaxBufIdx = runInfo.mLoop % SOFTMAX_BUFFER_NUM;
        const uint32_t expBufIdx = runInfo.loop % SOFTMAX_BUFFER_NUM;
        // 每个 task 固定按 subLoop 0/1 使用两个 queue slot；loop * 2 对模 2 恒为 0。
        constexpr uint32_t stage1Offset = SUB_LOOP;
        LocalTensor<float> sumUb = softmaxSumBuf_[softmaxBufIdx].template Get<float>()[0];
        LocalTensor<float> maxUb = softmaxMaxBuf_[softmaxBufIdx].template Get<float>()[0];
        LocalTensor<float> preLoopMaxUb = preLoopMaxBuf_.template Get<float>()[0];
        LocalTensor<float> preLoopSumUb = preLoopSumBuf_.template Get<float>()[0];
        LocalTensor<float> firstLoopSumUb = firstLoopSumBuf_.template Get<float>()[0];
        LocalTensor<float> expUb = softmaxExpBuf_[expBufIdx].template Get<float>()[0];
        LocalTensor<MM_T> mmRes = bmm1ResBuf.template GetTensor<MM_T>();
        LocalTensor<INPUT_T> stage1CastTensor = stage1OutQue_[stage1Offset].template AllocTensor<INPUT_T>();
        LocalTensor<SCALE_T> pScaleSubLoopTensor = pScaleSubLoop0Que_.template AllocTensor<SCALE_T>();

        const uint32_t s2CalcSize = GetSubLoopS2Size<SUB_LOOP>(runInfo);
        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (USE_DN) {
            if constexpr (HAS_ATTEN) {
                constexpr uint8_t attenMaskBit = static_cast<uint8_t>(1U << SUB_LOOP);
                if (unlikely((runInfo.attenMaskSubLoopBits & attenMaskBit) != 0U)) {
                    // 真实mask先搬入UB，与C1生产阶段重叠；完全可见subLoop不分配mask队列。
                    attenMaskUb = CopyAttenMaskToUb<SUB_LOOP>(runInfo, s2CalcSize);
                    ExecuteVec1Vf<true, SUB_LOOP>(bmm1ResBuf, stage1CastTensor, sumUb, maxUb, mmRes, expUb, attenMaskUb,
                                                  pScaleSubLoopTensor, preLoopMaxUb, preLoopSumUb, firstLoopSumUb,
                                                  runInfo, constInfo, s2CalcSize);
                    attenMaskInQue_.template FreeTensor(attenMaskUb);
                } else {
                    ExecuteVec1Vf<false, SUB_LOOP>(bmm1ResBuf, stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                                                   attenMaskUb, pScaleSubLoopTensor, preLoopMaxUb, preLoopSumUb,
                                                   firstLoopSumUb, runInfo, constInfo, s2CalcSize);
                }
            } else {
                ExecuteVec1Vf<false, SUB_LOOP>(bmm1ResBuf, stage1CastTensor, sumUb, maxUb, mmRes, expUb, attenMaskUb,
                                               pScaleSubLoopTensor, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, runInfo,
                                               constInfo, s2CalcSize);
            }
        } else {
            static_assert(USE_DN, "MXFullQuantMode vector currently implements DN VF only; "
                                  "use_dn=false is reserved for future extension");
        }

        // PScale 与 P 写入同一 L1 buffer，最后一个 subLoop 后通知 C2。
        CopyPScaleToL1<SUB_LOOP>(outputBuf, pScaleSubLoopTensor, runInfo, constInfo);
        pScaleSubLoop0Que_.template FreeTensor(pScaleSubLoopTensor);

        stage1OutQue_[stage1Offset].template EnQue(stage1CastTensor);
        stage1OutQue_[stage1Offset].template DeQue<INPUT_T>();
        CopyPToL1<SUB_LOOP>(outputBuf, stage1CastTensor, constInfo);
        stage1OutQue_[stage1Offset].template FreeTensor(stage1CastTensor);

        if (isLastSubLoop) {
            outputBuf.SetCrossCore();
        }
        if (unlikely(runInfo.isLastS2Loop && isLastSubLoop)) {
            // 最后一个 S2/subLoop 后写 LSE。
            if constexpr (HAS_LSE) {
                SoftmaxLseCopyOut(runInfo, constInfo, sumUb, maxUb);
            }
        }
    }

    // V2：更新 C2[128,DV]，最后归一化并写回 bf16。
    __aicore__ inline void ProcessVec2(MxBmm2Buf &bmm2ResBuf, MxRunInfo &runInfo, const MxConstInfo &constInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0U)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        const uint32_t vecMSize = runInfo.actVecMSize;
        const uint32_t loopIdx = runInfo.loop % SOFTMAX_BUFFER_NUM;
        const uint32_t mLoopIdx = runInfo.mLoop % SOFTMAX_BUFFER_NUM;
        const bool isFirstS2Loop = runInfo.isFirstS2Loop;
        const bool isLastS2Loop = runInfo.isLastS2Loop;
        const int64_t vec2CalcSize = static_cast<int64_t>(vecMSize) * dTemplateAlign64;
        LocalTensor<MM_T> vec2ResUb = stage2OutBuf_.template Get<MM_T>();
        LocalTensor<MM_T> mmRes = bmm2ResBuf.template GetTensor<MM_T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        if (unlikely(isFirstS2Loop)) {
            // 首个 S2 tile 直接作为累计值。
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<MM_T> expUb = softmaxExpBuf_[loopIdx].template Get<MM_T>();
            if (!isLastS2Loop) {
                // 中间 tile 保留 fp32 累计值。
                FlashUpdateNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            } else {
                // 最后一个 tile 完成 update 与归一化。
                LocalTensor<float> sumUb = softmaxSumBuf_[mLoopIdx].template Get<float>();
                // MXFP8 keeps OUT at zero when the accumulated softmax denominator is zero.
                FlashUpdateLastNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, true>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (isLastS2Loop) {
            if (unlikely(isFirstS2Loop)) {
                // 单 tile 场景单独归一化。
                LocalTensor<float> sumUb = softmaxSumBuf_[mLoopIdx].template Get<float>();
                LastDivNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, vecMSize,
                                                                      static_cast<uint16_t>(dTemplateAlign64), 1.0f);
            }
            CopyOutAttentionOut(runInfo, constInfo, vec2ResUb);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
    }

private:
    __aicore__ inline float DecodeE8M0Scale(uint8_t scale) const
    {
        // E8M0 code 0 is 2^-127 (an FP32 subnormal), not FP32 zero.
        // Code 255 is the format's NaN encoding; all other codes map directly
        // to the FP32 exponent field.
        uint32_t bits = scale == 0U ? 0x00400000U : (scale == 0xFFU ? 0x7FC00000U : static_cast<uint32_t>(scale) << 23);
        return *((float *)&bits);
    }

    template <bool IS_UPDATE, bool VF_HAS_ATTEN, uint32_t SUB_LOOP>
    __aicore__ inline void RunVec1Vf(const LocalTensor<INPUT_T> &stage1CastTensor, const LocalTensor<float> &sumUb,
                                     const LocalTensor<float> &maxUb, const LocalTensor<MM_T> &mmRes,
                                     const LocalTensor<float> &expUb, const LocalTensor<uint8_t> &attenMaskUb,
                                     const LocalTensor<SCALE_T> &pScaleSubLoopTensor,
                                     const LocalTensor<float> &preLoopMaxUb, const LocalTensor<float> &preLoopSumUb,
                                     const LocalTensor<float> &firstLoopSumUb, const MxRunInfo &runInfo,
                                     const MxConstInfo &constInfo, uint32_t s2CalcSize)
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        FaVectorApi::ProcessVec1VfDnMxfp8<MM_T, INPUT_T, IS_UPDATE, VF_HAS_ATTEN, S2_SPLIT, SUB_LOOP>(
            stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBuf_, attenMaskUb, pScaleSubLoopTensor,
            s2CalcSize, static_cast<MM_T>(constInfo.scaleValue), pScaleValue_, negativeFloatScalar_, preLoopMaxUb,
            preLoopSumUb, firstLoopSumUb);
    }

    template <bool VF_HAS_ATTEN, uint32_t SUB_LOOP>
    __aicore__ inline void ExecuteVec1Vf(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         const LocalTensor<INPUT_T> &stage1CastTensor, const LocalTensor<float> &sumUb,
                                         const LocalTensor<float> &maxUb, const LocalTensor<MM_T> &mmRes,
                                         const LocalTensor<float> &expUb, const LocalTensor<uint8_t> &attenMaskUb,
                                         const LocalTensor<SCALE_T> &pScaleSubLoopTensor,
                                         const LocalTensor<float> &preLoopMaxUb, const LocalTensor<float> &preLoopSumUb,
                                         const LocalTensor<float> &firstLoopSumUb, const MxRunInfo &runInfo,
                                         const MxConstInfo &constInfo, uint32_t s2CalcSize)
    {
        static_assert(!VF_HAS_ATTEN || HAS_ATTEN, "masked VF requires an attention-mask kernel instance");
        // mask搬运发生在调用前；随后等待C1并执行纯编译期mask/subLoop VF实例。
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.isFirstS2Loop)) {
            RunVec1Vf<false, VF_HAS_ATTEN, SUB_LOOP>(stage1CastTensor, sumUb, maxUb, mmRes, expUb, attenMaskUb,
                                                     pScaleSubLoopTensor, preLoopMaxUb, preLoopSumUb, firstLoopSumUb,
                                                     runInfo, constInfo, s2CalcSize);
        } else {
            RunVec1Vf<true, VF_HAS_ATTEN, SUB_LOOP>(stage1CastTensor, sumUb, maxUb, mmRes, expUb, attenMaskUb,
                                                    pScaleSubLoopTensor, preLoopMaxUb, preLoopSumUb, firstLoopSumUb,
                                                    runInfo, constInfo, s2CalcSize);
        }
        bmm1ResBuf.SetCrossCore();
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline uint32_t GetSubLoopS2Size(const MxRunInfo &runInfo) const
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        if constexpr (SUB_LOOP == 0U) {
            return runInfo.actSingleLoopS2Size <= S2_SPLIT ? runInfo.actSingleLoopS2Size : S2_SPLIT;
        }
        return runInfo.actSingleLoopS2Size > S2_SPLIT ? runInfo.actSingleLoopS2Size - S2_SPLIT : 0U;
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline bool IsLastSubLoop(const MxRunInfo &runInfo) const
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        if constexpr (SUB_LOOP == 0U) {
            return runInfo.actSingleLoopS2Size <= S2_SPLIT;
        }
        return true;
    }

    __aicore__ inline void BoolCopyInMxMask(const LocalTensor<uint8_t> &dstTensor, uint64_t srcOffset, uint32_t s2Len,
                                            uint32_t qLen)
    {
        // DN mask 按 GM row stride 搬到 [S2,64] UB。
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s2Len;
        dataCopyParams.blockLen = CeilDiv(qLen, MASK_COPY_BLOCK_BYTES);
        dataCopyParams.dstStride = CeilDiv(VEC_M_BASE, MASK_COPY_BLOCK_BYTES) - dataCopyParams.blockLen;
        // Host 固定 mask 为 [2048,2048]，当前 AIV 固定搬 64 列，源/目标行宽均满足 32B 对齐。
        dataCopyParams.srcStride = (attenMaskS2Size_ - qLen) / MASK_COPY_BLOCK_BYTES;
        DataCopy(dstTensor, attenMaskGm_[srcOffset], dataCopyParams);
    }

    __aicore__ inline uint64_t ComputeCausalMaskOffset(const MxRunInfo &runInfo, uint64_t segmentS2TokenOffset) const
    {
        const int64_t deltaN = static_cast<int64_t>(runInfo.actS1Size) - static_cast<int64_t>(runInfo.actS2Size);
        const int64_t deltaCausalOrNext =
            static_cast<int64_t>(runInfo.s1Idx) - static_cast<int64_t>(segmentS2TokenOffset) - deltaN;
        const uint64_t maskOffset = static_cast<uint64_t>(regbaseutil::ComputeOffsetForCausal(
            deltaCausalOrNext, M_BASE, S2_SPLIT, attenMaskS2Size_, static_cast<int64_t>(runInfo.vecMbaseIdx), USE_DN));
        return maskOffset;
    }

    __aicore__ inline bool CopyAttenMaskSegment(LocalTensor<uint8_t> &attenMaskUb, const MxRunInfo &runInfo,
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
    __aicore__ inline LocalTensor<uint8_t> CopyAttenMaskToUb(const MxRunInfo &runInfo, uint32_t s2CalcSize)
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
        attenMaskInQue_.template EnQue(attenMaskUb);
        return attenMaskInQue_.template DeQue<uint8_t>();
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline void CopyPScaleToL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                          LocalTensor<SCALE_T> &pScaleSubLoopTensor, const MxRunInfo &runInfo,
                                          const MxConstInfo &constInfo)
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        // 将 subLoop PScale 整理到完整 P[128,512] 后方。
        constexpr uint64_t pScaleL1Offset = static_cast<uint64_t>(M_BASE) * S2_BASE;
        LocalTensor<SCALE_T> pScaleL1Tensor = outputBuf.GetTensor<SCALE_T>(pScaleL1Offset);
        pScaleSubLoop0Que_.template EnQue(pScaleSubLoopTensor);
        pScaleSubLoop0Que_.template DeQue<SCALE_T>();
        // 两个 AIV 各处理 VEC_M_BASE 行；PScale 每 MX_SCALE_GROUP 个 P 元素生成一个 scale，
        // 因此一个 subBlock 的 PScale 长度为 VEC_M_BASE * S2_SPLIT / MX_SCALE_GROUP。
        constexpr uint64_t pScaleDataLen = VEC_M_BASE * S2_SPLIT / MX_SCALE_GROUP;
        // DataCopy 以 32B 为基本块，fp8_e8m0_t 每块 32 个元素。每行有
        // S2_SPLIT / MX_SCALE_GROUP 个 scale；DN packed 后按 4 个搬运列组写入。
        constexpr uint16_t pScaleDstStride = ((S2_SPLIT / MX_SCALE_GROUP) >> 1U) - 1U;
        const uint64_t vecOffset = constInfo.subBlockIdx * pScaleDataLen;
        if constexpr (SUB_LOOP == 1U) {
            // subLoop1 写 P 的后 256 列：4 个 column group，每组起点相隔 32 个 E8M0 元素。
            DataCopy(pScaleL1Tensor[vecOffset], pScaleSubLoopTensor, {4, 1, 0, pScaleDstStride});
            DataCopy(pScaleL1Tensor[vecOffset + 32U], pScaleSubLoopTensor, {4, 1, 0, pScaleDstStride});
            DataCopy(pScaleL1Tensor[vecOffset + 64U], pScaleSubLoopTensor, {4, 1, 0, pScaleDstStride});
            DataCopy(pScaleL1Tensor[vecOffset + 96U], pScaleSubLoopTensor, {4, 1, 0, pScaleDstStride});
        } else if (runInfo.actSingleLoopS2Size <= S2_SPLIT) {
            // 只有一个 256-token subLoop 时，需要把 PScale 同步填到完整 512-token tile 的两个半区。
            // 每半区跨度为 256 个 E8M0 元素，即 8 个 32B 块。
            DataCopy(pScaleL1Tensor[vecOffset], pScaleSubLoopTensor, {1, 8, 0, 0});
            DataCopy(pScaleL1Tensor[vecOffset + 256U], pScaleSubLoopTensor, {1, 8, 0, 0});
        }
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline void CopyPToL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                     LocalTensor<INPUT_T> &stage1CastTensor, const MxConstInfo &constInfo)
    {
        static_assert(SUB_LOOP < 2U, "MX V1 only supports subLoop 0/1");
        // 两个 AIV、两个 subLoop 分别写入 P 的对应象限。
        LocalTensor<INPUT_T> pL1Tensor = outputBuf.GetTensor<INPUT_T>();
        constexpr uint64_t subLoopOffset = static_cast<uint64_t>(S2_SPLIT) * M_BASE * SUB_LOOP;
        constexpr uint64_t vecBlockOffset = static_cast<uint64_t>(S2_SPLIT) * VEC_M_BASE;
        constexpr uint64_t secondCopyDstOffset = static_cast<uint64_t>(32U) * S2_SPLIT;
        constexpr uint64_t secondCopySrcOffset = 65U << 5U;
        const uint64_t dstBase = subLoopOffset + constInfo.subBlockIdx * vecBlockOffset;
        DataCopy(pL1Tensor[dstBase], stage1CastTensor, {4, S2_SPLIT >> 2, (S2_SPLIT >> 2) + 2U, 0});
        DataCopy(pL1Tensor[dstBase + secondCopyDstOffset], stage1CastTensor[secondCopySrcOffset],
                 {4, S2_SPLIT >> 2, (S2_SPLIT >> 2) + 2U, 0});
    }

    __aicore__ inline void CopyOutAttentionOut(MxRunInfo &runInfo, const MxConstInfo &constInfo,
                                               LocalTensor<MM_T> &vec2ResUb)
    {
        if constexpr (HAS_ATTEN) {
            // 使用所有 S2 loop 累计后的 max 判断整行是否始终无有效 key。
            LocalTensor<float> maxUb = softmaxMaxBuf_[runInfo.mLoop % SOFTMAX_BUFFER_NUM].template Get<float>();
            FaVectorApi::RowInvalidUpdateVF<MM_T>(vec2ResUb, maxUb, runInfo.actVecMSize, constInfo.dSizeV,
                                                  static_cast<int64_t>(dTemplateAlign64));
        }
        // 按 TND token/head stride 写回。
        LocalTensor<OUTPUT_T> attenOut;
        attenOut.SetAddr(vec2ResUb.address_);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, static_cast<int64_t>(runInfo.actVecMSize) * dTemplateAlign64);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
        dataCopyParams.blockCount = runInfo.actVecMSize;
        dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) >> 4;
        dataCopyParams.dstStride = constInfo.attentionOutStride;
        const uint64_t tokenOffset =
            static_cast<uint64_t>(runInfo.queryTokenBase) + runInfo.s1Idx + runInfo.vecMbaseIdx;
        const uint64_t dstOffset = (tokenOffset * constInfo.realN2Size + runInfo.realN2Idx) * constInfo.dSizeV;
        DataCopyPad(attentionOutGm_[dstOffset], attenOut, dataCopyParams);
    }

    __aicore__ inline void SoftmaxLseCopyOut(MxRunInfo &runInfo, const MxConstInfo &constInfo,
                                             LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        LocalTensor<float> lseUb = softmaxLseQueue_.template AllocTensor<float>();
        ComputeLseOutputVF(lseUb, sumUb, maxUb, runInfo.actVecMSize);
        softmaxLseQueue_.template EnQue(lseUb);
        softmaxLseQueue_.template DeQue<float>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.blockCount = runInfo.actVecMSize;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = constInfo.softmaxLseStride;
        const uint64_t tokenOffset =
            static_cast<uint64_t>(runInfo.queryTokenBase) + runInfo.s1Idx + runInfo.vecMbaseIdx;
        const uint64_t dstOffset = tokenOffset * constInfo.realN2Size + runInfo.realN2Idx;
        DataCopyPad(softmaxLseGm_[dstOffset], lseUb, dataCopyParams);
        softmaxLseQueue_.template FreeTensor(lseUb);
    }

    TPipe *tPipe_ = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<uint8_t> attenMaskGm_;
    uint32_t attenMaskS2Size_ = 0U;
    TQue<QuePosition::VECIN, 1> attenMaskInQue_;
    TQue<QuePosition::VECOUT, 1> stage1OutQue_[2];
    TQue<QuePosition::VECOUT, 1> pScaleSubLoop0Que_;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue_;
    TBuf<> stage2OutBuf_;
    TBuf<> softmaxMaxBuf_[SOFTMAX_BUFFER_NUM];
    TBuf<> softmaxSumBuf_[SOFTMAX_BUFFER_NUM];
    TBuf<> softmaxExpBuf_[SOFTMAX_BUFFER_NUM];
    TBuf<> preLoopMaxBuf_;
    TBuf<> preLoopSumBuf_;
    TBuf<> firstLoopSumBuf_;
    TBuf<> vselrIndexesBuf_[4];
    TEventID mte3ToVId_[1];
    TEventID vToMte3Id_[1];
    MM_T negativeFloatScalar_ = 0.0f;
    float pScaleValue_ = 1.0f;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_VEC_H_
