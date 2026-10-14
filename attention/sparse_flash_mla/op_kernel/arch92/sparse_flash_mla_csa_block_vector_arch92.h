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
 * \file sparse_flash_mla_csa_block_vector_arch92.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH92_H
#define SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH92_H

#include "util_regbase_arch92.h"
#include "sparse_flash_mla_common_arch92.h"
#include "../../../common/op_kernel/buffers_policy.h"
#include "../../../common/op_kernel/attn_buffer_manager.h"
#include "../../../common/op_kernel/attn_buffer.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace optiling;
using namespace optiling::detail;
using namespace regbaseutil;
using namespace matmul;

namespace SMLAKernel {
TEMPLATES_DEF
class CSABlockVec {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = s1BaseSize + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr float R0 = 1.0f;

    // ==================== Functions ======================
    __aicore__ inline CSABlockVec(){};
    __aicore__ inline void InitVecBlock(TPipe* pipe, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                        __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedOriKv,
                                        __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            // CleanOutput is called before InitLocalBuffer and uses this event when needInit is set.
            initOutputEventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensOriKv);
            }
            if (cuSeqlensCmpKv != nullptr) {
                cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensCmpKv);
            }
            if (sequsedOriKv != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t*)sequsedOriKv);
            }
            if (sequsedCmpKv != nullptr) {
                actualSeqLengthsCmpKVGm.SetGlobalBuffer((__gm__ int32_t*)sequsedCmpKv);
            }
            if (cmpResidualKv != nullptr) {
                cmpResidualKVGm.SetGlobalBuffer((__gm__ int32_t*)cmpResidualKv);
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe* pipe, ConstInfo& constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, ConstInfo& constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                            __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                            __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks,
                                            __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv,
                                            __gm__ uint8_t* cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo& constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1,
                                       const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf,
                                       RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void ProcessVec2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm2ResBuf,
                                       RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void FreeEvent(ConstInfo& constInfo);

private:
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1,
                                           const RunInfo& runInfo, ConstInfo& constInfo);

    __aicore__ inline void CalSparseCalSize(const RunInfo& runInfo, ConstInfo& constInfo);

    __aicore__ inline void CastNd2Nz(LocalTensor<Q_T> dstTensor, LocalTensor<KV_T> srcTensor, int64_t dealRow);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1,
                                          LocalTensor<Q_T> kvOutUb, int64_t v0Loop, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo& runInfo, ConstInfo& constInfo);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo& runInfo, ConstInfo& constInfo, LocalTensor<VEC2_RES_T>& vec2ResUb,
                                           int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo& runInfo, ConstInfo& constInfo,
                                               LocalTensor<VEC2_RES_T>& vec2ResUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void GetExtremeValue(T& negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo& constInfo);

    TPipe* tPipe;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<uint8_t> keyGmBytes;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> oriSparseIndicesGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> actualSeqLengthsCmpKVGm;
    GlobalTensor<int32_t> cmpResidualKVGm;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue; // 2份表示可能存在pingpong
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId;
    TEventID vToMte3Id;
    TEventID mte2ToVV0Id[2];
    TEventID vToMte2V0Id[2];
    TEventID vToMte3V0Id[2];
    TEventID mte3ToVV0Id[2];
    TEventID Mte3ToMte2V0Id;
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> outLseBuf[2];
    TBuf<> stage0InBuf[2];
    TBuf<> stage0OutBuf[2];
    TBuf<> vselrIndexesBuf[2];

    TBuf<> sparseIndiceBuf;
    TBuf<> paBlockIdxOffsetBuf;
    TBuf<> tailBlockOffsetBuf;
    TBuf<> paBlockIdBuf;
    TBuf<> totalOffsetBuf;

    LocalTensor<int32_t> sparseIndiceTensor;
    LocalTensor<int32_t> paBlockIdxOffsetTensor;
    LocalTensor<uint32_t> tailBlockOffsetTensor;
    LocalTensor<int32_t> paBlockIdTensor;
    LocalTensor<uint32_t> totalOffsetTensor;
    uint32_t pingPongV0 = 0;
    uint32_t v0BaseSize = 8;
    uint32_t NUM2 = 2;
    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;

    TEventID initOutputEventId; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
    TEventID mte3ToVLseOutId;
    TEventID vToMte3LseOutId;
    bool isSoftmaxLseGmValid = false;
};

// fp8->fp32
static constexpr Reg::CastTrait castTraitFp8_1 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr Reg::CastTrait castTraitFp8_2 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr Reg::CastTrait castTraitFp8_3 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr Reg::CastTrait castTraitFp8_4 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

/**
    i / b -> (i * m) >> p
 */
__simd_vf__ void DivPABlockIdxVF(__ubuf__ int32_t* paBlockIdxOffsetUb, __ubuf__ uint32_t* tailBlockOffsetUb,
                                 __ubuf__ int32_t* sparseIndiceUb, const int32_t sparseBlockSize,
                                 const uint32_t blockSize, const uint32_t mHigh, const uint32_t mLow, const uint8_t p,
                                 const uint32_t s2Stride)
{
    AscendC::Reg::MaskReg preg32_all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg preg64_all = Reg::CreateMask<uint64_t, Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<uint32_t> vreg_sparse_indices_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_sparse_indices_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_m_l_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_m_l_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_mul_res_h_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_mul_res_h_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_m_l_0_dst0;
    AscendC::Reg::RegTensor<uint32_t> vreg_m_l_0_dst1;
    AscendC::Reg::RegTensor<uint32_t> vreg_m_l_1_dst0;
    AscendC::Reg::RegTensor<uint32_t> vreg_m_l_1_dst1;
    AscendC::Reg::RegTensor<uint32_t> vreg_res_h_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_res_h_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_mul_res_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_mul_res_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_mul_res_2;
    AscendC::Reg::RegTensor<uint32_t> vreg_mul_res_3;
    AscendC::Reg::RegTensor<uint32_t> vreg_pa_block_idx_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_pa_block_idx_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_zero;
    AscendC::Reg::RegTensor<uint32_t> vreg_whole_blocks2_offset_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_tail_blocks2_offset_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_tail_block_offset_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_whole_blocks2_offset_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_tail_blocks2_offset_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_tail_block_offset_1;
    AscendC::Reg::Duplicate(vreg_m_l_0, mLow, preg32_all);
    AscendC::Reg::Duplicate(vreg_m_l_1, mLow, preg32_all);
    AscendC::Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t>&)vreg_sparse_indices_0,
                                                               sparseIndiceUb);
    AscendC::Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t>&)vreg_sparse_indices_1,
                                                               sparseIndiceUb + 64);
    AscendC::Reg::Muls(vreg_sparse_indices_0, vreg_sparse_indices_0, sparseBlockSize, preg32_all);
    AscendC::Reg::Muls(vreg_sparse_indices_1, vreg_sparse_indices_1, sparseBlockSize, preg32_all);
    AscendC::Reg::Mull(vreg_m_l_0_dst0, vreg_m_l_0_dst1, vreg_sparse_indices_0, vreg_m_l_0, preg32_all);
    AscendC::Reg::Mull(vreg_m_l_1_dst0, vreg_m_l_1_dst1, vreg_sparse_indices_1, vreg_m_l_1, preg32_all);
    AscendC::Reg::Muls(vreg_mul_res_h_0, vreg_sparse_indices_0, mHigh, preg32_all);
    AscendC::Reg::Muls(vreg_mul_res_h_1, vreg_sparse_indices_1, mHigh, preg32_all);
    AscendC::Reg::Add(vreg_res_h_0, vreg_mul_res_h_0, vreg_m_l_0_dst1, preg32_all);
    AscendC::Reg::Add(vreg_res_h_1, vreg_mul_res_h_1, vreg_m_l_1_dst1, preg32_all);
    AscendC::Reg::Interleave(vreg_mul_res_0, vreg_mul_res_1, vreg_m_l_0_dst0, vreg_res_h_0);
    AscendC::Reg::Interleave(vreg_mul_res_2, vreg_mul_res_3, vreg_m_l_1_dst0, vreg_res_h_1);
    AscendC::Reg::ShiftRights((Reg::RegTensor<uint64_t>&)vreg_mul_res_0, (Reg::RegTensor<uint64_t>&)vreg_mul_res_0,
                              (int16_t)p, preg64_all);
    AscendC::Reg::ShiftRights((Reg::RegTensor<uint64_t>&)vreg_mul_res_1, (Reg::RegTensor<uint64_t>&)vreg_mul_res_1,
                              (int16_t)p, preg64_all);
    AscendC::Reg::ShiftRights((Reg::RegTensor<uint64_t>&)vreg_mul_res_2, (Reg::RegTensor<uint64_t>&)vreg_mul_res_2,
                              (int16_t)p, preg64_all);
    AscendC::Reg::ShiftRights((Reg::RegTensor<uint64_t>&)vreg_mul_res_3, (Reg::RegTensor<uint64_t>&)vreg_mul_res_3,
                              (int16_t)p, preg64_all);
    AscendC::Reg::DeInterleave(vreg_pa_block_idx_0, vreg_zero, vreg_mul_res_0, vreg_mul_res_1);
    AscendC::Reg::DeInterleave(vreg_pa_block_idx_1, vreg_zero, vreg_mul_res_2, vreg_mul_res_3);
    AscendC::Reg::Muls(vreg_whole_blocks2_offset_0, vreg_pa_block_idx_0, blockSize, preg32_all);
    AscendC::Reg::Muls(vreg_whole_blocks2_offset_1, vreg_pa_block_idx_1, blockSize, preg32_all);
    AscendC::Reg::Sub(vreg_tail_blocks2_offset_0, vreg_sparse_indices_0, vreg_whole_blocks2_offset_0, preg32_all);
    AscendC::Reg::Sub(vreg_tail_blocks2_offset_1, vreg_sparse_indices_1, vreg_whole_blocks2_offset_1, preg32_all);
    AscendC::Reg::Muls(vreg_tail_block_offset_0, vreg_tail_blocks2_offset_0, s2Stride, preg32_all);
    AscendC::Reg::Muls(vreg_tail_block_offset_1, vreg_tail_blocks2_offset_1, s2Stride, preg32_all);
    AscendC::Reg::Muls(vreg_pa_block_idx_0, vreg_pa_block_idx_0, sizeof(uint32_t), preg32_all);
    AscendC::Reg::Muls(vreg_pa_block_idx_1, vreg_pa_block_idx_1, sizeof(uint32_t), preg32_all);
    AscendC::Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_NORM>(
        paBlockIdxOffsetUb, (Reg::RegTensor<int32_t>&)vreg_pa_block_idx_0, preg32_all);
    AscendC::Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_NORM>(
        paBlockIdxOffsetUb + 64, (Reg::RegTensor<int32_t>&)vreg_pa_block_idx_1, preg32_all);
    AscendC::Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(tailBlockOffsetUb, vreg_tail_block_offset_0,
                                                                  preg32_all);
    AscendC::Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(tailBlockOffsetUb + 64, vreg_tail_block_offset_1,
                                                                  preg32_all);
}

__simd_vf__ void GetTotalOffsetVF(__ubuf__ uint32_t* totalOffsetUb, __ubuf__ int32_t* paBlockIdUb,
                                  __ubuf__ uint32_t* tailBlockOffsetUb, const uint32_t blockNumStrideL,
                                  const uint32_t blockNumStrideH)
{
    AscendC::Reg::MaskReg preg32_all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg preg64_all = Reg::CreateMask<uint64_t, Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg carry_l_0;
    AscendC::Reg::MaskReg carry_l_1;
    AscendC::Reg::MaskReg carry_h_0;
    AscendC::Reg::MaskReg carry_h_1;

    AscendC::Reg::RegTensor<uint32_t> vreg_pa_block_id_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_pa_block_id_1;

    AscendC::Reg::RegTensor<uint32_t> vreg_block_num_stride_l;

    AscendC::Reg::RegTensor<uint32_t> vreg_whole_block_offset_l_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_whole_block_offset_h_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_tmp_whole_block_offset_h_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_whole_block_offset_l_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_whole_block_offset_h_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_tmp_whole_block_offset_h_1;

    AscendC::Reg::RegTensor<uint32_t> vreg_l_0_dst1;
    AscendC::Reg::RegTensor<uint32_t> vreg_l_1_dst1;

    AscendC::Reg::RegTensor<uint32_t> vreg_tail_block_offset_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_tail_block_offset_1;

    AscendC::Reg::RegTensor<uint32_t> vreg_total_offset_l_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_total_offset_h_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_total_offset_l_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_total_offset_h_1;

    AscendC::Reg::RegTensor<uint32_t> vreg_zero;

    AscendC::Reg::RegTensor<uint32_t> vreg_total_offset_0;
    AscendC::Reg::RegTensor<uint32_t> vreg_total_offset_1;
    AscendC::Reg::RegTensor<uint32_t> vreg_total_offset_2;
    AscendC::Reg::RegTensor<uint32_t> vreg_total_offset_3;

    AscendC::Reg::Duplicate(vreg_block_num_stride_l, blockNumStrideL, preg32_all);
    AscendC::Reg::Duplicate(vreg_zero, 0, preg32_all);

    AscendC::Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t>&)vreg_pa_block_id_0,
                                                               paBlockIdUb);
    AscendC::Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t>&)vreg_pa_block_id_1,
                                                               paBlockIdUb + 64);
    AscendC::Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vreg_tail_block_offset_0, tailBlockOffsetUb);
    AscendC::Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vreg_tail_block_offset_1, tailBlockOffsetUb + 64);
    AscendC::Reg::Mull(vreg_whole_block_offset_l_0, vreg_l_0_dst1, vreg_pa_block_id_0, vreg_block_num_stride_l,
                       preg32_all);
    AscendC::Reg::Mull(vreg_whole_block_offset_l_1, vreg_l_1_dst1, vreg_pa_block_id_1, vreg_block_num_stride_l,
                       preg32_all);
    AscendC::Reg::Muls(vreg_tmp_whole_block_offset_h_0, vreg_pa_block_id_0, blockNumStrideH, preg32_all);
    AscendC::Reg::Muls(vreg_tmp_whole_block_offset_h_1, vreg_pa_block_id_1, blockNumStrideH, preg32_all);

    AscendC::Reg::Add(vreg_whole_block_offset_h_0, vreg_tmp_whole_block_offset_h_0, vreg_l_0_dst1, preg32_all);
    AscendC::Reg::Add(vreg_whole_block_offset_h_1, vreg_tmp_whole_block_offset_h_1, vreg_l_1_dst1, preg32_all);
    AscendC::Reg::Add(carry_l_0, vreg_total_offset_l_0, vreg_whole_block_offset_l_0, vreg_tail_block_offset_0,
                      preg32_all);
    AscendC::Reg::Add(carry_l_1, vreg_total_offset_l_1, vreg_whole_block_offset_l_1, vreg_tail_block_offset_1,
                      preg32_all);

    AscendC::Reg::AddC(carry_h_0, vreg_total_offset_h_0, vreg_whole_block_offset_h_0, vreg_zero, carry_l_0, preg32_all);
    AscendC::Reg::AddC(carry_h_1, vreg_total_offset_h_1, vreg_whole_block_offset_h_1, vreg_zero, carry_l_1, preg32_all);

    AscendC::Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(totalOffsetUb, vreg_total_offset_l_0,
                                                                       vreg_total_offset_h_0, preg32_all);
    AscendC::Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(totalOffsetUb + 128, vreg_total_offset_l_1,
                                                                       vreg_total_offset_h_1, preg32_all);
}

__simd_vf__ void GetTndOffsetVf(__ubuf__ uint32_t* totalOffsetUb, __ubuf__ int32_t* sparseIndciesUb,
                                const uint32_t tPrefix, const uint32_t dim)
{
    AscendC::Reg::MaskReg preg32_all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    constexpr int32_t SIZE_64 = 64;
    constexpr int32_t SIZE_128 = 128;

    AscendC::Reg::RegTensor<uint32_t> indices;
    AscendC::Reg::RegTensor<uint32_t> indices_unroll;

    AscendC::Reg::RegTensor<uint32_t> prefix_reg;
    AscendC::Reg::RegTensor<uint32_t> dim_reg;

    AscendC::Reg::RegTensor<uint32_t> high_reg;
    AscendC::Reg::RegTensor<uint32_t> high_reg_unroll;

    AscendC::Reg::RegTensor<uint32_t> low_reg;
    AscendC::Reg::RegTensor<uint32_t> low_reg_unroll;

    AscendC::Reg::Duplicate(prefix_reg, tPrefix, preg32_all);
    AscendC::Reg::Duplicate(dim_reg, dim, preg32_all);

    AscendC::Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t>&)indices, sparseIndciesUb);
    AscendC::Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t>&)indices_unroll,
                                                               sparseIndciesUb + SIZE_64);

    AscendC::Reg::Add(indices, indices, prefix_reg, preg32_all);
    AscendC::Reg::Add(indices_unroll, indices_unroll, prefix_reg, preg32_all);

    AscendC::Reg::Mull(low_reg, high_reg, indices, dim_reg, preg32_all);
    AscendC::Reg::Mull(low_reg_unroll, high_reg_unroll, indices_unroll, dim_reg, preg32_all);

    AscendC::Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(totalOffsetUb, low_reg, high_reg, preg32_all);
    AscendC::Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(totalOffsetUb + SIZE_128, low_reg_unroll,
                                                                       high_reg_unroll, preg32_all);
}

__simd_vf__ void GetBsndOffsetVf(__ubuf__ uint32_t* totalOffsetUb, __ubuf__ int32_t* sparseIndciesUb,
                                 const uint32_t prefixLow, const uint32_t prefixHigh, const uint32_t d)
{
    AscendC::Reg::MaskReg preg32_all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg carry_l_0;
    AscendC::Reg::MaskReg carry_l_1;
    AscendC::Reg::MaskReg carry_h_0;
    AscendC::Reg::MaskReg carry_h_1;

    constexpr int32_t SIZE_64 = 64;
    constexpr int32_t SIZE_128 = 128;

    AscendC::Reg::RegTensor<uint32_t> indices;
    AscendC::Reg::RegTensor<uint32_t> indices_unroll;

    AscendC::Reg::RegTensor<uint32_t> prefix_low_reg;
    AscendC::Reg::RegTensor<uint32_t> prefix_high_reg;
    AscendC::Reg::RegTensor<uint32_t> dim_reg;

    AscendC::Reg::RegTensor<uint32_t> r_low_reg;
    AscendC::Reg::RegTensor<uint32_t> r_low_reg_unroll;

    AscendC::Reg::RegTensor<uint32_t> r_high_reg;
    AscendC::Reg::RegTensor<uint32_t> r_high_reg_unroll;

    AscendC::Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t>&)indices, sparseIndciesUb);
    AscendC::Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t>&)indices_unroll,
                                                               sparseIndciesUb + SIZE_64);

    Reg::Duplicate(dim_reg, d, preg32_all);
    Reg::Duplicate(prefix_low_reg, prefixLow, preg32_all);
    Reg::Duplicate(prefix_high_reg, prefixHigh, preg32_all);

    AscendC::Reg::Mull(r_low_reg, r_high_reg, indices, dim_reg, preg32_all);
    AscendC::Reg::Mull(r_low_reg_unroll, r_high_reg_unroll, indices_unroll, dim_reg, preg32_all);

    AscendC::Reg::Add(carry_l_0, r_low_reg, r_low_reg, prefix_low_reg, preg32_all);
    AscendC::Reg::Add(carry_l_1, r_low_reg_unroll, r_low_reg_unroll, prefix_low_reg, preg32_all);

    AscendC::Reg::AddC(carry_h_0, r_high_reg, r_high_reg, prefix_high_reg, carry_l_0, preg32_all);
    AscendC::Reg::AddC(carry_h_1, r_high_reg_unroll, r_high_reg_unroll, prefix_high_reg, carry_l_1, preg32_all);

    AscendC::Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(totalOffsetUb, r_low_reg, r_high_reg,
                                                                       preg32_all);
    AscendC::Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(totalOffsetUb + SIZE_128, r_low_reg_unroll,
                                                                       r_high_reg_unroll, preg32_all);
}

__aicore__ inline void DivPABlockIdx(const LocalTensor<int32_t>& paBlockIdxOffsetTensor,
                                     const LocalTensor<uint32_t>& tailBlockOffsetTensor,
                                     const LocalTensor<int32_t>& sparseIndiceTensor, const int32_t sparseBlockSize,
                                     const uint32_t blockSize, const uint32_t mHigh, const uint32_t mLow,
                                     const uint8_t p, const uint32_t s2Stride)
{
    __ubuf__ int32_t* paBlockIdxOffsetUb = (__ubuf__ int32_t*)paBlockIdxOffsetTensor.GetPhyAddr();
    __ubuf__ uint32_t* tailBlockOffsetUb = (__ubuf__ uint32_t*)tailBlockOffsetTensor.GetPhyAddr();
    __ubuf__ int32_t* sparseIndiceUb = (__ubuf__ int32_t*)sparseIndiceTensor.GetPhyAddr();

    DivPABlockIdxVF(paBlockIdxOffsetUb, tailBlockOffsetUb, sparseIndiceUb, sparseBlockSize, blockSize, mHigh, mLow, p,
                    s2Stride);
}

__aicore__ inline void GetPaTotalOffset(const LocalTensor<uint32_t>& totalOffsetTensor,
                                        const LocalTensor<int32_t>& paBlockIdTensor,
                                        const LocalTensor<uint32_t>& tailBlockOffsetTensor,
                                        const uint32_t blockNumStrideL, const uint32_t blockNumStrideH)
{
    __ubuf__ uint32_t* totalOffsetUb = (__ubuf__ uint32_t*)totalOffsetTensor.GetPhyAddr();
    __ubuf__ int32_t* paBlockIdUb = (__ubuf__ int32_t*)paBlockIdTensor.GetPhyAddr();
    __ubuf__ uint32_t* tailBlockOffsetUb = (__ubuf__ uint32_t*)tailBlockOffsetTensor.GetPhyAddr();

    GetTotalOffsetVF(totalOffsetUb, paBlockIdUb, tailBlockOffsetUb, blockNumStrideL, blockNumStrideH);
}

__aicore__ inline void GetTndTotalOffset(const LocalTensor<uint32_t>& totalOffsetTensor,
                                         const LocalTensor<int32_t>& sparseIndices, const uint32_t tPrefix,
                                         const uint32_t n2Size, const uint32_t dSizeVInput)
{
    __ubuf__ uint32_t* totalOffsetUb = (__ubuf__ uint32_t*)totalOffsetTensor.GetPhyAddr();
    __ubuf__ int32_t* sparseIndicesUb = (__ubuf__ int32_t*)sparseIndices.GetPhyAddr();
    const uint32_t dim = n2Size * dSizeVInput;
    GetTndOffsetVf(totalOffsetUb, sparseIndicesUb, tPrefix, dim);
}

__aicore__ inline void GetBsndTotalOffset(const LocalTensor<uint32_t>& totalOffsetTensor,
                                          const LocalTensor<int32_t>& sparseIndices, const uint32_t s2n2dLow,
                                          const uint32_t s2n2dHigh, const uint32_t d)
{
    __ubuf__ uint32_t* totalOffsetUb = (__ubuf__ uint32_t*)totalOffsetTensor.GetPhyAddr();
    __ubuf__ int32_t* sparseIndicesUb = (__ubuf__ int32_t*)sparseIndices.GetPhyAddr();
    GetBsndOffsetVf(totalOffsetUb, sparseIndicesUb, s2n2dLow, s2n2dHigh, d);
}

template <typename T>
__simd_vf__ void CastNd2NzVFImpl(__ubuf__ T* ubDstAddr, __ubuf__ T* ubSrcAddr, const uint32_t dealRowCount,
                                 const uint32_t blockStride)
{
    Reg::RegTensor<T> kvData0;
    Reg::RegTensor<T> kvData1;
    Reg::RegTensor<T> kvData2;
    Reg::RegTensor<T> kvData3;

    Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    const uint32_t repeatStride = 1;
    const uint32_t combineDim = 512;
    const uint32_t elementNumPerVL = 128; // 2字节类型: 256B / 2B = 128

    __ubuf__ T* ubSrcAddrTmp0 = ubSrcAddr;
    __ubuf__ T* ubSrcAddrTmp1 = ubSrcAddr + elementNumPerVL;
    __ubuf__ T* ubSrcAddrTmp2 = ubSrcAddr + elementNumPerVL * 2;
    __ubuf__ T* ubSrcAddrTmp3 = ubSrcAddr + elementNumPerVL * 3;

    __ubuf__ T* ubDstAddrTmp0 = ubDstAddr;
    __ubuf__ T* ubDstAddrTmp1 = ubDstAddr + elementNumPerVL * blockStride;
    __ubuf__ T* ubDstAddrTmp2 = ubDstAddr + elementNumPerVL * 2 * blockStride;
    __ubuf__ T* ubDstAddrTmp3 = ubDstAddr + elementNumPerVL * 3 * blockStride;

    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(kvData0, ubSrcAddrTmp0,
                                                                                        combineDim);
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(kvData1, ubSrcAddrTmp1,
                                                                                        combineDim);
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(kvData2, ubSrcAddrTmp2,
                                                                                        combineDim);
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(kvData3, ubSrcAddrTmp3,
                                                                                        combineDim);

        Reg::StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp0, kvData0, blockStride, repeatStride, maskAll);
        Reg::StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp1, kvData1, blockStride, repeatStride, maskAll);
        Reg::StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp2, kvData2, blockStride, repeatStride, maskAll);
        Reg::StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp3, kvData3, blockStride, repeatStride, maskAll);
    }
}

/**
 * @brief 将16x512的Nd排布转换为Nz排布，按 BF16 的 16 位 raw bits 处理
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CastNd2Nz(LocalTensor<Q_T> dstTensor, LocalTensor<KV_T> srcTensor,
                                                             int64_t dealRow)
{
    __ubuf__ uint16_t* ubDstAddr = (__ubuf__ uint16_t*)(dstTensor.GetPhyAddr());
    __ubuf__ uint16_t* ubSrcAddr = (__ubuf__ uint16_t*)(srcTensor.GetPhyAddr());
    const uint32_t blockStride = dealRow | 0x1; // odd row num to avoid bank confict
    CastNd2NzVFImpl(ubDstAddr, ubSrcAddr, dealRow, blockStride);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1, LocalTensor<Q_T> kvOutUb, int64_t v0Loop,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo& runInfo, ConstInfo& constInfo)
{
    // DataCopy 的一个 block 是 32B，BF16 每个 block 含 16 个元素。
    constexpr uint64_t blockElementNum = 32U / sizeof(Q_T);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = (dealRow | 0x1) - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * blockElementNum], kvOutUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo& runInfo, ConstInfo& constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = GetBlockIdx();
        uint32_t v0S2SizeFirstCore = runInfo.s2RealSize;
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize;
        sparseS2Start = 0;
        sparseCalSize = runInfo.s2RealSize;
        sparseS2End = sparseS2Start + sparseCalSize;
    } else {
        int64_t s2PerVecLoop = 2LL;
        int64_t vecNum = 2LL;
        int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize);
        sparseS2End = GetSubBlockIdx() == 0 ? Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize) : runInfo.s2RealSize;
        sparseCalSize = sparseS2End - sparseS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1, const RunInfo& runInfo, ConstInfo& constInfo)
{
    if (runInfo.s2LoopCount < runInfo.oriKvLoopEndIdx) {
        return;
    }
    keyGm = cmpKVGm;
    keyGmBytes.SetGlobalBuffer((__gm__ uint8_t*)keyGm.GetPhyAddr());
    cuSeqlensKvGm = cuSeqlensCmpKvGm;
    sparseIndicesGm = cmpSparseIndicesGm;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        blockTableGm = cmpBlockTableGm;
        blockSize = constInfo.cmpBlockSize;
        maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
    }
    CalSparseCalSize(runInfo, constInfo);
    ProcessSparseKv(outputL1, runInfo, constInfo);
    outputL1.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1, const RunInfo& runInfo, ConstInfo& constInfo)
{
    int64_t s2ProcessBaseSize = v0BaseSize;
    int64_t s2RealSize = runInfo.s2RealSize; // 使用sparsecalsize还是s2realSize
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.cmpSparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.cmpSparseBlockCount +
                      runInfo.s1oIdx * constInfo.cmpSparseBlockCount; // B, S1, N2(1), K
    }
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx;
    int64_t topkKIdx = cmpS2LoopCnt * constInfo.s2BaseSize;
    WaitFlag<HardEvent::MTE3_MTE2>(Mte3ToMte2V0Id);
    AscendC::DataCopyParams copyInParams;
    copyInParams.blockCount = 1U;
    copyInParams.blockLen = (s2RealSize * sizeof(int32_t) + 31U) / 32U; // 32B 为单位，向上取整
    copyInParams.srcGap = 0U;
    copyInParams.dstGap = 0U;
    DataCopy(sparseIndiceTensor,
             cmpSparseIndicesGm[topkBS1Idx + runInfo.s2StartIdx + cmpS2LoopCnt * constInfo.s2BaseSize], copyInParams);
    SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
    WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);

    if (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        uint32_t s2Stride = constInfo.dSizeVInput * sizeof(KV_T);
        int32_t mLow;
        int32_t mHigh;
        int32_t p;
        if (runInfo.isCmp) {
            mLow = constInfo.cmpMLow;
            mHigh = constInfo.cmpMHigh;
            p = constInfo.cmpP;
        } else {
            mLow = constInfo.oriMLow;
            mHigh = constInfo.oriMHigh;
            p = constInfo.oriP;
        }
        DivPABlockIdx(paBlockIdxOffsetTensor, tailBlockOffsetTensor, sparseIndiceTensor, constInfo.sparseBlockSize,
                      blockSize, mHigh, mLow, p, s2Stride);
        SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        GatherParams gatherParams1{
            static_cast<uint16_t>(s2RealSize),
            static_cast<uint32_t>(4),
            static_cast<uint32_t>(4),
        };
        GatherPadParams<int32_t> gatherPadparams1;
        GatherDataCopy(paBlockIdTensor, blockTableGm[runInfo.boIdx * maxBlockNumPerBatch], paBlockIdxOffsetTensor,
                       gatherParams1, gatherPadparams1);
        SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        uint32_t blockNumStrideL = (runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride) * sizeof(KV_T);
        uint32_t blockNumStrideH = 0;
        GetPaTotalOffset(totalOffsetTensor, paBlockIdTensor, tailBlockOffsetTensor, blockNumStrideL, blockNumStrideH);
    } else if (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        uint32_t tPrefix =
            runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) : cuSeqlensOriKvGm.GetValue(runInfo.boIdx); // n2=1
        GetTndTotalOffset(totalOffsetTensor, sparseIndiceTensor, static_cast<uint32_t>(tPrefix),
                          static_cast<uint32_t>(constInfo.n2Size),
                          static_cast<uint32_t>(constInfo.dSizeVInput * sizeof(KV_T)));
    } else { // BSND
        int64_t bs2n2d =
            (runInfo.isCmp ? runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSizeVInput :
                             runInfo.boIdx * constInfo.n2S2Dv) *
            sizeof(KV_T);
        uint64_t u_bs2n2d = (uint64_t)bs2n2d;
        uint32_t s2n2Low = (uint32_t)(u_bs2n2d & 0xFFFFFFFFULL);
        uint32_t s2n2High = (uint32_t)(u_bs2n2d >> 32U);
        GetBsndTotalOffset(totalOffsetTensor, sparseIndiceTensor, s2n2Low, s2n2High,
                           constInfo.dSizeVInput * sizeof(KV_T));
    }
    SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
    LocalTensor<int64_t> totalOffset64Tensor = totalOffsetTensor.ReinterpretCast<int64_t>();
    LocalTensor<KV_T> kvInUb = stage0InBuf[pingPongV0].Get<KV_T>();
    LocalTensor<uint8_t> kvInUbBytes = stage0InBuf[pingPongV0].Get<uint8_t>();
    LocalTensor<Q_T> kvOutUb = stage0OutBuf[pingPongV0].Get<Q_T>();
    int64_t s2ProcessBaseLoop = CeilDiv(s2RealSize, s2ProcessBaseSize);
    int64_t s2Start = 0;
    for (uint16_t i = 0; i < s2ProcessBaseLoop; ++i) {
        int64_t remain = s2RealSize - i * s2ProcessBaseSize;
        int64_t dealRows;
        if (s2RealSize < s2ProcessBaseSize) {
            dealRows = s2RealSize;
        } else {
            dealRows = (s2ProcessBaseSize < remain) ? s2ProcessBaseSize : remain;
        }
        WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        GatherParams gatherParams2{
            static_cast<uint16_t>(dealRows),
            static_cast<uint32_t>(constInfo.dSizeVInput * sizeof(KV_T)),
            static_cast<uint32_t>(constInfo.dSizeVInput * sizeof(KV_T)),
        };
        GatherPadParams<uint8_t> gatherPadparams2{GatherPaddingMode::CONSTANT_PADDING_MODE, static_cast<uint8_t>(2)};
        GatherDataCopy(kvInUbBytes, keyGmBytes, totalOffset64Tensor[s2ProcessBaseSize * i], gatherParams2,
                       gatherPadparams2);

        SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        CastNd2Nz(kvOutUb, kvInUb, dealRows);
        if (i != s2ProcessBaseLoop - 1) {
            SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        CopyOutKvUb2L1(outputL1, kvOutUb, 0, dealRows, s2Start, runInfo, constInfo);
        SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        s2Start += dealRows;
    }
    SetFlag<HardEvent::MTE3_MTE2>(Mte3ToMte2V0Id);
    pingPongV0 ^= 1;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf, RunInfo& runInfo, ConstInfo& constInfo)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    auto stage1CastTensor = this->stage1OutQue.template AllocTensor<Q_T>();
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    if (runInfo.s2LoopCount == 0 && !isSinks) {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (runInfo.s2LoopCount == 0 && isSinks) {
            int64_t sinksOffset =
                (GetBlockIdx() & 1) ? (IS_SPLIT_G ? s1BaseSize : constInfo.gSize - runInfo.halfMRealSize) : 0;
            LocalTensor<float> sinksUb = this->sinksBuf.template Get<float>();
            InitSoftmaxFromSinks<T>(sumUb, maxUb, sinksUb, sinksOffset, R0, runInfo.halfMRealSize);
        }
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();
    this->stage1OutQue.template EnQue(stage1CastTensor);
    this->stage1OutQue.template DeQue<Q_T>();
    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (runInfo.mRealSize - runInfo.halfMRealSize)],
                 stage1CastTensor,
                 {s2BaseSize / 16, (uint16_t)(runInfo.halfMRealSize), (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
                  (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }
    this->stage1OutQue.template FreeTensor(stage1CastTensor);
    outputBuf.SetCrossCore();
    if (runInfo.s2LoopCount != 0 || (runInfo.s2LoopCount == 0 && isSinks)) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    if (constInfo.returnSoftmaxLse && isSoftmaxLseGmValid && runInfo.halfMRealSize > 0 &&
        runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        LocalTensor<float> outLse = outLseBuf[runInfo.multiCoreIdxMod2].template Get<float>();
        DataCopyExtParams lseParams;
        lseParams.blockCount = 1;
        lseParams.blockLen = static_cast<uint32_t>(runInfo.halfMRealSize * sizeof(float));
        lseParams.srcStride = 0;
        lseParams.dstStride = 0;
        WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        ComputeLse<float>(outLse, sumUb, maxUb, static_cast<uint32_t>(runInfo.halfMRealSize));
        SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
        WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
        DataCopyPad(softmaxLseGm[runInfo.softmaxLseOffset], outLse, lseParams);
        SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm2ResBuf, RunInfo& runInfo, ConstInfo& constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
    LocalTensor<T> test = commonTBuf.Get<T>();
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();

            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize,
                                                                  dTemplateAlign64, 1.0);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo& runInfo, ConstInfo& constInfo,
                                                                   LocalTensor<VEC2_RES_T>& vec2ResUb,
                                                                   int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3Id);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo& runInfo, ConstInfo& constInfo,
                                                                       LocalTensor<VEC2_RES_T>& vec2ResUb,
                                                                       int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo& constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if (LAYOUT_T == SMLA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if (LAYOUT_T == SMLA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }
    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                         0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }
    if (constInfo.returnSoftmaxLse && isSoftmaxLseGmValid) {
        uint64_t totalLseSize = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
            totalLseSize = constInfo.bSize * constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
        } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            totalLseSize = constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
        }
        uint64_t coreOffset = 0;
        if (coreNum != 0 && totalLseSize > 0) {
            uint64_t singleCoreLseSize = (totalLseSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
            coreOffset = constInfo.aivIdx * singleCoreLseSize;
            if (coreOffset < totalLseSize) {
                uint64_t singleInitLseSize = Min(singleCoreLseSize, totalLseSize - coreOffset);
                WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
                matmul::InitOutput<float>(softmaxLseGm[coreOffset], singleInitLseSize, 0);
                SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse,
                                                               ConstInfo& constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        if (constInfo.returnSoftmaxLse && softmaxLse != nullptr) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
            isSoftmaxLseGmValid = true;
        }
        if (constInfo.needInit == 1) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            InitOutputSingleCore(constInfo);
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
    __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks,
    __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
{
    if (oriKV != nullptr) {
        oriKVGm.SetGlobalBuffer((__gm__ KV_T*)(oriKV));
    }

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)oriBlockTable);
    }

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T*)cmpKV);
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)cmpBlockTable);
        }
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t*)cmpSparseIndices);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T*)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo& constInfo)
{
    LocalTensor<float> sinksUb = this->sinksBuf.template Get<float>();
    const uint32_t maxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(float);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<float> padParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, padParams);
    TEventID mte2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe* pipe, ConstInfo& constInfo)
{
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512);   // sinksBuf内存申请512B

    tPipe->InitBuffer(stage0InBuf[0],
                      dVTemplateType * v0BaseSize * sizeof(KV_T)); // V0阶段每次处理8个seq, 开2 buffer  共10Kb
    tPipe->InitBuffer(stage0InBuf[1], dVTemplateType * v0BaseSize * sizeof(KV_T));
    tPipe->InitBuffer(
        stage0OutBuf[0],
        dVTemplateType * (v0BaseSize + 1) * sizeof(Q_T)); // kv输入D轴608, V0阶段每次处理8个seq, 开2 buffer
    tPipe->InitBuffer(stage0OutBuf[1], dVTemplateType * (v0BaseSize + 1) * sizeof(Q_T));

    tPipe->InitBuffer(stage1OutQue, 1, vec1Srcstride * s2BaseSize * sizeof(Q_T)); // 16.25KB

    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    if (constInfo.returnSoftmaxLse) {
        tPipe->InitBuffer(outLseBuf[0], 256U);
        tPipe->InitBuffer(outLseBuf[1], 256U);
    }

    tPipe->InitBuffer(sparseIndiceBuf, s2BaseSize * sizeof(int32_t));
    tPipe->InitBuffer(paBlockIdxOffsetBuf, s2BaseSize * sizeof(int32_t));
    tPipe->InitBuffer(tailBlockOffsetBuf, s2BaseSize * sizeof(uint32_t));
    tPipe->InitBuffer(paBlockIdBuf, s2BaseSize * sizeof(int32_t));
    tPipe->InitBuffer(totalOffsetBuf, s2BaseSize * NUM2 * sizeof(uint32_t));

    sparseIndiceTensor = sparseIndiceBuf.Get<int32_t>();
    paBlockIdxOffsetTensor = paBlockIdxOffsetBuf.Get<int32_t>();
    tailBlockOffsetTensor = tailBlockOffsetBuf.Get<uint32_t>();
    paBlockIdTensor = paBlockIdBuf.Get<int32_t>();
    totalOffsetTensor = totalOffsetBuf.Get<uint32_t>();

    mte3ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    vToMte3Id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
    mte3ToVLseOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    vToMte3LseOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    mte2ToVV0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    mte2ToVV0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    vToMte2V0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    vToMte2V0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    vToMte3V0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3V0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    mte3ToVV0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVV0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    Mte3ToMte2V0Id = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[1]);

    SetFlag<HardEvent::MTE3_MTE2>(Mte3ToMte2V0Id);
    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::FreeEvent(ConstInfo& constInfo) {}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetExtremeValue(T& negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float*)&tmp1);
}

TEMPLATES_DEF
class CSABlockVecDummy {
public:
    __aicore__ inline CSABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, ConstInfo& constInfo)
    {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                            __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                            __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks,
                                            __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv,
                                            __gm__ uint8_t* cmpResidualKv)
    {}
    __aicore__ inline void InitVecBlock(TPipe* pipe, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                        __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedOriKv,
                                        __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
    {}
    __aicore__ inline void InitLocalBuffer(TPipe* pipe, ConstInfo& constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf,
                                       RunInfo& runInfo, ConstInfo& constInfo)
    {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos& bmm2ResBuf, RunInfo& runInfo, ConstInfo& constInfo) {}
};
} // namespace SMLAKernel
#endif // SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH92_H
