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
 * \file quant_sparse_flash_mla_csa_block_vector_arch92.h
 * \brief
 */

#ifndef QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH92_H
#define QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH92_H

#include "qsmla_util_regbase_arch92.h"
#include "quant_sparse_flash_mla_common_arch92.h"
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

namespace BaseApi {
TEMPLATES_DEF
class SCFABlockVec {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = s1BaseSize + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr uint32_t dVTemplateTypeInput = 512;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步

    // ==================== Functions ======================
    __aicore__ inline SCFABlockVec(){};
    __aicore__ inline void InitVecBlock(TPipe* pipe, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                        __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedOriKv,
                                        __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensOriKv);
            }
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE) {
                if (cuSeqlensCmpKv != nullptr) {
                    cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensCmpKv);
                }
            }
            if (sequsedOriKv != nullptr) {
                actualSeqOriKvGm.SetGlobalBuffer((__gm__ int32_t*)sequsedOriKv);
            }
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE) {
                if (sequsedCmpKv != nullptr) {
                    actualSeqCmpKvGm.SetGlobalBuffer((__gm__ int32_t*)sequsedCmpKv);
                }
                cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t*)cmpResidualKv);
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    using mmResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using outputl1Pos = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using gmPos = Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>;

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe* pipe, ConstInfo& constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t* attentionOut, ConstInfo& constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* qDescale,
                                            __gm__ uint8_t* oriKVDescale, __gm__ uint8_t* cmpKVDescale,
                                            __gm__ uint8_t* cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                                            __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t* sequsedQ,
                                            __gm__ uint8_t* sinks, __gm__ uint8_t* sequsedOriKv,
                                            __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo& constInfo);
    __aicore__ inline void ProcessVec0(outputl1Pos& outputL1, const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void ProcessVec1(outputl1Pos& outputBuf, mmResPos& bmm1ResBuf, RunInfo& runInfo,
                                       ConstInfo& constInfo);
    __aicore__ inline void ProcessVec2(mmResPos& bmm2ResBuf, RunInfo& runInfo, ConstInfo& constInfo);

private:
    __aicore__ inline void ProcessSparseKv(outputl1Pos& outputL1, const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void ProcessNotSparseKv(outputl1Pos& outputL1, const RunInfo& runInfo, ConstInfo& constInfo);

    __aicore__ inline void CalSparseCalSize(const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int32_t* tokenData, int64_t s2IdxInBase, const RunInfo& runInfo,
                                           ConstInfo& constInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
                                             int64_t s2StartIdx, const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int32_t* tokenData,
                                              const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void CastNd2Nz(LocalTensor<Q_T> dstTensor, LocalTensor<KV_T> srcTensor, int64_t dealRow);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1,
                                          LocalTensor<Q_T> kvOutUb, int64_t v0Loop, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void CopyOutMrgeResult(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1,
                                             int64_t mte2Size, int64_t mte3Size, int64_t s2keyOffset,
                                             int64_t mergeMte3Idx, const RunInfo& runInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset);
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
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<float> qDescaleGm;
    GlobalTensor<float> oriKVDescaleGm;
    GlobalTensor<float> cmpKVDescaleGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqOriKvGm;
    GlobalTensor<int32_t> actualSeqCmpKvGm;
    GlobalTensor<int32_t> cmpResidualKvGm;

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
};

// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

/**
    i / blockSize -> i >> shiftRightNum（blockSize 为 2 的幂）
 */
__simd_vf__ void DivPABlockIdxVF(__ubuf__ int32_t* paBlockIdxOffsetUb, __ubuf__ uint32_t* tailBlockOffsetUb,
                                 __ubuf__ int32_t* sparseIndiceUb, const int32_t sparseBlockSize,
                                 const uint32_t blockSize, const int16_t shiftRightNum, const uint32_t s2Stride)
{
    AscendC::MicroAPI::MaskReg preg32_all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_sparse_indices_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_sparse_indices_1;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_pa_block_idx_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_pa_block_idx_1;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_whole_blocks2_offset_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_tail_blocks2_offset_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_tail_block_offset_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_whole_blocks2_offset_1;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_tail_blocks2_offset_1;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_tail_block_offset_1;
    AscendC::MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
        (MicroAPI::RegTensor<int32_t>&)vreg_sparse_indices_0, sparseIndiceUb);
    AscendC::MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
        (MicroAPI::RegTensor<int32_t>&)vreg_sparse_indices_1, sparseIndiceUb + 64);
    AscendC::MicroAPI::Muls(vreg_sparse_indices_0, vreg_sparse_indices_0, sparseBlockSize, preg32_all);
    AscendC::MicroAPI::Muls(vreg_sparse_indices_1, vreg_sparse_indices_1, sparseBlockSize, preg32_all);
    AscendC::MicroAPI::ShiftRights(vreg_pa_block_idx_0, vreg_sparse_indices_0, shiftRightNum, preg32_all);
    AscendC::MicroAPI::ShiftRights(vreg_pa_block_idx_1, vreg_sparse_indices_1, shiftRightNum, preg32_all);
    AscendC::MicroAPI::Muls(vreg_whole_blocks2_offset_0, vreg_pa_block_idx_0, blockSize, preg32_all);
    AscendC::MicroAPI::Muls(vreg_whole_blocks2_offset_1, vreg_pa_block_idx_1, blockSize, preg32_all);
    AscendC::MicroAPI::Sub(vreg_tail_blocks2_offset_0, vreg_sparse_indices_0, vreg_whole_blocks2_offset_0, preg32_all);
    AscendC::MicroAPI::Sub(vreg_tail_blocks2_offset_1, vreg_sparse_indices_1, vreg_whole_blocks2_offset_1, preg32_all);
    AscendC::MicroAPI::Muls(vreg_tail_block_offset_0, vreg_tail_blocks2_offset_0, s2Stride, preg32_all);
    AscendC::MicroAPI::Muls(vreg_tail_block_offset_1, vreg_tail_blocks2_offset_1, s2Stride, preg32_all);
    AscendC::MicroAPI::Muls(vreg_pa_block_idx_0, vreg_pa_block_idx_0, sizeof(uint32_t), preg32_all);
    AscendC::MicroAPI::Muls(vreg_pa_block_idx_1, vreg_pa_block_idx_1, sizeof(uint32_t), preg32_all);
    AscendC::MicroAPI::StoreAlign<int32_t, MicroAPI::StoreDist::DIST_NORM>(
        paBlockIdxOffsetUb, (MicroAPI::RegTensor<int32_t>&)vreg_pa_block_idx_0, preg32_all);
    AscendC::MicroAPI::StoreAlign<int32_t, MicroAPI::StoreDist::DIST_NORM>(
        paBlockIdxOffsetUb + 64, (MicroAPI::RegTensor<int32_t>&)vreg_pa_block_idx_1, preg32_all);
    AscendC::MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(tailBlockOffsetUb, vreg_tail_block_offset_0,
                                                                            preg32_all);
    AscendC::MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(tailBlockOffsetUb + 64,
                                                                            vreg_tail_block_offset_1, preg32_all);
}

__simd_vf__ void GetTotalOffsetVF(__ubuf__ uint32_t* totalOffsetUb, __ubuf__ int32_t* paBlockIdUb,
                                  __ubuf__ uint32_t* tailBlockOffsetUb, const uint32_t blockNumStrideL,
                                  const uint32_t blockNumStrideH)
{
    AscendC::MicroAPI::MaskReg preg32_all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg preg64_all = MicroAPI::CreateMask<uint64_t, MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg carry_l_0;
    AscendC::MicroAPI::MaskReg carry_l_1;
    AscendC::MicroAPI::MaskReg carry_h_0;
    AscendC::MicroAPI::MaskReg carry_h_1;

    AscendC::MicroAPI::RegTensor<uint32_t> vreg_pa_block_id_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_pa_block_id_1;

    AscendC::MicroAPI::RegTensor<uint32_t> vreg_block_num_stride_l;

    AscendC::MicroAPI::RegTensor<uint32_t> vreg_whole_block_offset_l_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_whole_block_offset_h_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_tmp_whole_block_offset_h_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_whole_block_offset_l_1;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_whole_block_offset_h_1;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_tmp_whole_block_offset_h_1;

    AscendC::MicroAPI::RegTensor<uint32_t> vreg_l_0_dst1;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_l_1_dst1;

    AscendC::MicroAPI::RegTensor<uint32_t> vreg_tail_block_offset_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_tail_block_offset_1;

    AscendC::MicroAPI::RegTensor<uint32_t> vreg_total_offset_l_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_total_offset_h_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_total_offset_l_1;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_total_offset_h_1;

    AscendC::MicroAPI::RegTensor<uint32_t> vreg_zero;

    AscendC::MicroAPI::RegTensor<uint32_t> vreg_total_offset_0;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_total_offset_1;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_total_offset_2;
    AscendC::MicroAPI::RegTensor<uint32_t> vreg_total_offset_3;

    AscendC::MicroAPI::Duplicate(vreg_block_num_stride_l, blockNumStrideL, preg32_all);
    AscendC::MicroAPI::Duplicate(vreg_zero, 0, preg32_all);

    AscendC::MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
        (MicroAPI::RegTensor<int32_t>&)vreg_pa_block_id_0, paBlockIdUb);
    AscendC::MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
        (MicroAPI::RegTensor<int32_t>&)vreg_pa_block_id_1, paBlockIdUb + 64);
    AscendC::MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vreg_tail_block_offset_0, tailBlockOffsetUb);
    AscendC::MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vreg_tail_block_offset_1,
                                                                          tailBlockOffsetUb + 64);
    // 低位实际地址偏移
    AscendC::MicroAPI::Mull(vreg_whole_block_offset_l_0, vreg_l_0_dst1, vreg_pa_block_id_0, vreg_block_num_stride_l,
                            preg32_all);
    AscendC::MicroAPI::Mull(vreg_whole_block_offset_l_1, vreg_l_1_dst1, vreg_pa_block_id_1, vreg_block_num_stride_l,
                            preg32_all);
    // 高位实际地址偏移
    AscendC::MicroAPI::Muls(vreg_tmp_whole_block_offset_h_0, vreg_pa_block_id_0, blockNumStrideH, preg32_all);
    AscendC::MicroAPI::Muls(vreg_tmp_whole_block_offset_h_1, vreg_pa_block_id_1, blockNumStrideH, preg32_all);

    AscendC::MicroAPI::Add(vreg_whole_block_offset_h_0, vreg_tmp_whole_block_offset_h_0, vreg_l_0_dst1, preg32_all);
    AscendC::MicroAPI::Add(vreg_whole_block_offset_h_1, vreg_tmp_whole_block_offset_h_1, vreg_l_1_dst1, preg32_all);
    // 支持进位
    AscendC::MicroAPI::Add(carry_l_0, vreg_total_offset_l_0, vreg_whole_block_offset_l_0, vreg_tail_block_offset_0,
                           preg32_all);
    AscendC::MicroAPI::Add(carry_l_1, vreg_total_offset_l_1, vreg_whole_block_offset_l_1, vreg_tail_block_offset_1,
                           preg32_all);

    AscendC::MicroAPI::AddC(carry_h_0, vreg_total_offset_h_0, vreg_whole_block_offset_h_0, vreg_zero, carry_l_0,
                            preg32_all);
    AscendC::MicroAPI::AddC(carry_h_1, vreg_total_offset_h_1, vreg_whole_block_offset_h_1, vreg_zero, carry_l_1,
                            preg32_all);

    AscendC::MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(totalOffsetUb, vreg_total_offset_l_0,
                                                                                 vreg_total_offset_h_0, preg32_all);
    AscendC::MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
        totalOffsetUb + 128, vreg_total_offset_l_1, vreg_total_offset_h_1, preg32_all);
}

__simd_vf__ void GetTndOffsetVf(__ubuf__ uint32_t* totalOffsetUb, __ubuf__ int32_t* sparseIndciesUb,
                                const uint32_t tPrefix, const uint32_t dim)
{
    AscendC::MicroAPI::MaskReg preg32_all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    constexpr int32_t SIZE_64 = 64;
    constexpr int32_t SIZE_128 = 128;

    AscendC::MicroAPI::RegTensor<uint32_t> indices;
    AscendC::MicroAPI::RegTensor<uint32_t> indices_unroll;

    AscendC::MicroAPI::RegTensor<uint32_t> prefix_reg;
    AscendC::MicroAPI::RegTensor<uint32_t> dim_reg;

    AscendC::MicroAPI::RegTensor<uint32_t> high_reg;
    AscendC::MicroAPI::RegTensor<uint32_t> high_reg_unroll;

    AscendC::MicroAPI::RegTensor<uint32_t> low_reg;
    AscendC::MicroAPI::RegTensor<uint32_t> low_reg_unroll;

    AscendC::MicroAPI::Duplicate(prefix_reg, tPrefix, preg32_all);
    AscendC::MicroAPI::Duplicate(dim_reg, dim, preg32_all);

    AscendC::MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)indices,
                                                                         sparseIndciesUb);
    AscendC::MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)indices_unroll,
                                                                         sparseIndciesUb + SIZE_64);

    AscendC::MicroAPI::Add(indices, indices, prefix_reg, preg32_all);
    AscendC::MicroAPI::Add(indices_unroll, indices_unroll, prefix_reg, preg32_all);

    // 低位实际地址偏移
    AscendC::MicroAPI::Mull(low_reg, high_reg, indices, dim_reg, preg32_all);
    AscendC::MicroAPI::Mull(low_reg_unroll, high_reg_unroll, indices_unroll, dim_reg, preg32_all);

    AscendC::MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(totalOffsetUb, low_reg, high_reg,
                                                                                 preg32_all);
    AscendC::MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
        totalOffsetUb + SIZE_128, low_reg_unroll, high_reg_unroll, preg32_all);
}

__simd_vf__ void GetBsndOffsetVf(__ubuf__ uint32_t* totalOffsetUb, __ubuf__ int32_t* sparseIndciesUb,
                                 const uint32_t prefixLow, const uint32_t prefixHigh, const uint32_t d)
{
    AscendC::MicroAPI::MaskReg preg32_all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg carry_l_0;
    AscendC::MicroAPI::MaskReg carry_l_1;
    AscendC::MicroAPI::MaskReg carry_h_0;
    AscendC::MicroAPI::MaskReg carry_h_1;

    constexpr int32_t SIZE_64 = 64;
    constexpr int32_t SIZE_128 = 128;

    AscendC::MicroAPI::RegTensor<uint32_t> indices;
    AscendC::MicroAPI::RegTensor<uint32_t> indices_unroll;

    AscendC::MicroAPI::RegTensor<uint32_t> prefix_low_reg;
    AscendC::MicroAPI::RegTensor<uint32_t> prefix_high_reg;
    AscendC::MicroAPI::RegTensor<uint32_t> dim_reg;

    AscendC::MicroAPI::RegTensor<uint32_t> r_low_reg;
    AscendC::MicroAPI::RegTensor<uint32_t> r_low_reg_unroll;

    AscendC::MicroAPI::RegTensor<uint32_t> r_high_reg;
    AscendC::MicroAPI::RegTensor<uint32_t> r_high_reg_unroll;

    AscendC::MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)indices,
                                                                         sparseIndciesUb);
    AscendC::MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)indices_unroll,
                                                                         sparseIndciesUb + SIZE_64);

    MicroAPI::Duplicate(dim_reg, d, preg32_all);
    MicroAPI::Duplicate(prefix_low_reg, prefixLow, preg32_all);
    MicroAPI::Duplicate(prefix_high_reg, prefixHigh, preg32_all);

    // calc s2 * n2 * d
    AscendC::MicroAPI::Mull(r_low_reg, r_high_reg, indices, dim_reg, preg32_all);
    AscendC::MicroAPI::Mull(r_low_reg_unroll, r_high_reg_unroll, indices_unroll, dim_reg, preg32_all);

    // prefix + s2 * n2 * d
    AscendC::MicroAPI::Add(carry_l_0, r_low_reg, r_low_reg, prefix_low_reg, preg32_all);
    AscendC::MicroAPI::Add(carry_l_1, r_low_reg_unroll, r_low_reg_unroll, prefix_low_reg, preg32_all);

    // handle carry
    AscendC::MicroAPI::AddC(carry_h_0, r_high_reg, r_high_reg, prefix_high_reg, carry_l_0, preg32_all);
    AscendC::MicroAPI::AddC(carry_h_1, r_high_reg_unroll, r_high_reg_unroll, prefix_high_reg, carry_l_1, preg32_all);

    AscendC::MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(totalOffsetUb, r_low_reg, r_high_reg,
                                                                                 preg32_all);
    AscendC::MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
        totalOffsetUb + SIZE_128, r_low_reg_unroll, r_high_reg_unroll, preg32_all);
}

__aicore__ inline bool IsPowerOfTwoBlockSize(uint32_t blockSize)
{
    return blockSize != 0U && (blockSize & (blockSize - 1U)) == 0U;
}

/**
    blockSize 为 2 的幂：i / blockSize -> i >> shiftRightNum
 */
__aicore__ inline void DivPABlockIdx(const LocalTensor<int32_t>& paBlockIdxOffsetTensor,
                                     const LocalTensor<uint32_t>& tailBlockOffsetTensor,
                                     const LocalTensor<int32_t>& sparseIndiceTensor, const int32_t sparseBlockSize,
                                     const uint32_t blockSize, const uint32_t s2Stride)
{
    __ubuf__ int32_t* paBlockIdxOffsetUb = (__ubuf__ int32_t*)paBlockIdxOffsetTensor.GetPhyAddr();
    __ubuf__ uint32_t* tailBlockOffsetUb = (__ubuf__ uint32_t*)tailBlockOffsetTensor.GetPhyAddr();
    __ubuf__ int32_t* sparseIndiceUb = (__ubuf__ int32_t*)sparseIndiceTensor.GetPhyAddr();

    int32_t blkSize = static_cast<int32_t>(blockSize);
    int16_t shiftRightNum = 0;
    while (blkSize > 1) {
        blkSize >>= 1;
        shiftRightNum++;
    }
    DivPABlockIdxVF(paBlockIdxOffsetUb, tailBlockOffsetUb, sparseIndiceUb, sparseBlockSize, blockSize, shiftRightNum,
                    s2Stride);
}

/**
    blockSize 非 2 的幂：与 A5 保持一致，用标量整数除法 i / blockSize、i % blockSize
 */
__aicore__ inline void SetPaBlockIdxOffsetScalar(const LocalTensor<int32_t>& paBlockIdxOffsetTensor,
                                                 const LocalTensor<uint32_t>& tailBlockOffsetTensor, uint32_t rowIdx,
                                                 uint32_t sparseIdx, uint32_t sparseBlockSize, uint32_t blockSize,
                                                 uint32_t s2Stride)
{
    const uint32_t scaledIdx = sparseIdx * sparseBlockSize;
    paBlockIdxOffsetTensor.SetValue(rowIdx, static_cast<int32_t>(scaledIdx / blockSize * sizeof(uint32_t)));
    tailBlockOffsetTensor.SetValue(rowIdx, scaledIdx % blockSize * s2Stride);
}

/**
    标量写 UB 后，分别同步给后续读取的 vector（tailBlockOffset）和 MTE2（paBlockIdxOffset）
 */
__aicore__ inline void SyncPaBlockIdxOffsetScalarToPipe()
{
    event_t eventIdScalarToVec = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdScalarToVec);
    WaitFlag<HardEvent::S_V>(eventIdScalarToVec);
    event_t eventIdScalarToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdScalarToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdScalarToMte2);
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

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetRealCmpS2Idx(int32_t* tokenData, int64_t s2IdxInBase,
                                                                    const RunInfo& runInfo, ConstInfo& constInfo)
{
    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.sparseBlockCount +
                      runInfo.s1oIdx * constInfo.sparseBlockCount; // B, S1, N2(1), K
    }
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx;
    uint64_t topkKIdx = s2IdxInBase + cmpS2LoopCnt * constInfo.s2BaseSize;
    for (uint64_t i = 0; i < v0BaseSize; ++i) {
        uint64_t idx = topkBS1Idx + runInfo.s2StartIdx + topkKIdx + i;
        if (likely((topkKIdx + i < constInfo.sparseBlockCount) && (s2IdxInBase + i < sparseS2End))) {
            tokenData[i] = cmpSparseIndicesGm.GetValue(idx);
        } else {
            break;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SCFABlockVec<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx, const RunInfo& runInfo,
                                                                    ConstInfo& constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) * paBlockStride +
                        blkTableOffset * constInfo.dSizeVInput; // BlockNum, BlockSize, N(1), D
    } else {
        int64_t kvS2Idx = s2Idx;
        if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
            kvS2Idx +=
                runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) : cuSeqlensOriKvGm.GetValue(runInfo.boIdx);
            realkeyOffset = (kvS2Idx * constInfo.n2Size + runInfo.n2oIdx) * constInfo.dSizeVInput;
        } else {
            realkeyOffset = ((runInfo.boIdx * constInfo.s2Size + kvS2Idx) * constInfo.n2Size + runInfo.n2oIdx) *
                            constInfo.dSizeVInput; // B, S, N, D
        }
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                   int64_t keyOffset)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;

    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    uint32_t combineBytes = dVTemplateTypeInput;
    intriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t SCFABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                       int32_t* tokenData, const RunInfo& runInfo,
                                                                       ConstInfo& constInfo)
{
    int64_t s2IdLimit = runInfo.s2RealSize;
    s2IdLimit = (runInfo.s2RealSize - runInfo.actualS1Size + runInfo.s1oIdx + 1) / constInfo.cmpRatio;
    uint32_t dealRow = 0;
    for (uint32_t i = 0; i < v0BaseSize; i += NUM2) {
        int64_t keyOffset0 = GetkeyOffset(tokenData[i], runInfo, constInfo);
        int64_t keyOffset1 = GetkeyOffset(tokenData[i + 1], runInfo, constInfo);
        if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
            return dealRow;
        }
        uint32_t combineBytes = constInfo.dSizeVInput;
        int64_t keySrcStride =
            (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) - combineBytes;
        if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || constInfo.sparseBlockSize > 1) {
            // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
            CopyInSingleKv(kvInUb, startRow, keyOffset0);
            CopyInSingleKv(kvInUb, startRow + 1, keyOffset1);
        } else {
            DataCopyExtParams intriParams;
            intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
            intriParams.blockLen = combineBytes;
            intriParams.dstStride = 0;
            intriParams.srcStride = keySrcStride;
            DataCopyPadExtParams<KV_T> padParams;

            int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
            if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
                keyOffset = keyOffset1;
            }

            // 当前仅支持COMBINE模式
            uint32_t combineDim = combineBytes / sizeof(KV_T);
            uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
        }
        dealRow += (keyOffset0 >= 0) + (keyOffset1 >= 0);
        startRow += NUM2;
    }
    return dealRow;
}

__simd_vf__ void CastB8Nd2NzVFImpl(__ubuf__ int8_t* ubDstAddr, __ubuf__ int8_t* ubSrcAddr, const uint32_t dealRowCount,
                                   const uint32_t blockStride)
{
    MicroAPI::RegTensor<int8_t> kvData0;
    MicroAPI::RegTensor<int8_t> kvData1;

    MicroAPI::MaskReg b8MaskAll = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();
    const uint32_t repeatStride = 1;
    const uint32_t combineDim = 512;
    const uint32_t elementNumPerVL = 256;

    __ubuf__ int8_t* ubSrcAddrTmp = ubSrcAddr + elementNumPerVL;
    __ubuf__ int8_t* ubDstAddrTmp = ubDstAddr + elementNumPerVL * blockStride;

    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
            kvData0, ubSrcAddr, combineDim);
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
            kvData1, ubSrcAddrTmp, combineDim);
        MicroAPI::StoreAlign<int8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr, kvData0, blockStride, repeatStride, b8MaskAll);
        MicroAPI::StoreAlign<int8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, kvData1, blockStride, repeatStride, b8MaskAll);
    }
}

/**
 * @brief 将16x512的Nd排布转换为Nz排布，数据类型为8位数据类型
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CastNd2Nz(LocalTensor<Q_T> dstTensor, LocalTensor<KV_T> srcTensor,
                                                              int64_t dealRow)
{
    __ubuf__ int8_t* ubDstAddr = (__ubuf__ int8_t*)(dstTensor.GetPhyAddr());
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(srcTensor.GetPhyAddr());
    const uint32_t blockStride = dealRow | 0x1; // odd row num to avoid bank confict
    CastB8Nd2NzVFImpl(ubDstAddr, ubSrcAddr, dealRow, blockStride);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1, LocalTensor<Q_T> kvOutUb, int64_t v0Loop,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo& runInfo, ConstInfo& constInfo)
{
    uint64_t blockElementNum = 32;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = (dealRow | 0x1) - dealRow;
    dataCopyParams.dstGap = Align32Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * blockElementNum], kvOutUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessNotSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1, const RunInfo& runInfo, ConstInfo& constInfo)
{
    int64_t s2ProcessBaseSize = v0BaseSize;
    int64_t s2RealSize = runInfo.s2RealSize; // 使用sparsecalsize还是s2realSize
    int64_t s2LoopCount = (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) ?
                              (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) :
                              runInfo.s2LoopCount;
    int64_t s2Idx = s2LoopCount * constInfo.s2BaseSize + runInfo.s2StartIdx;

    WaitFlag<HardEvent::MTE3_MTE2>(Mte3ToMte2V0Id);
    AscendC::CreateVecIndex(sparseIndiceTensor, (int32_t)s2Idx, (uint32_t)s2RealSize);
    PipeBarrier<PIPE_V>();
    if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::PA_BBND) {
        int32_t sparseBlockSize = 1;
        uint32_t s2Stride = constInfo.dSizeVInput;

        if (IsPowerOfTwoBlockSize(blockSize)) {
            DivPABlockIdx(paBlockIdxOffsetTensor, tailBlockOffsetTensor, sparseIndiceTensor, sparseBlockSize, blockSize,
                          s2Stride);
            // s2 / block_size
            SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
            WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        } else {
            uint32_t validCnt = Min(static_cast<uint32_t>(s2RealSize), s2BaseSize);
            for (uint32_t i = 0; i < validCnt; ++i) {
                SetPaBlockIdxOffsetScalar(paBlockIdxOffsetTensor, tailBlockOffsetTensor, i,
                                          static_cast<uint32_t>(s2Idx + i), 1U, blockSize, s2Stride);
            }
            SyncPaBlockIdxOffsetScalarToPipe();
        }
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
        uint32_t blockNumStrideL = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        uint32_t blockNumStrideH = 0;
        GetPaTotalOffset(totalOffsetTensor, paBlockIdTensor, tailBlockOffsetTensor, blockNumStrideL, blockNumStrideH);
    } else if (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        uint32_t tPrefix =
            runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) : cuSeqlensOriKvGm.GetValue(runInfo.boIdx);
        GetTndTotalOffset(totalOffsetTensor, sparseIndiceTensor, static_cast<uint32_t>(tPrefix),
                          static_cast<uint32_t>(constInfo.n2Size), static_cast<uint32_t>(constInfo.dSizeVInput));
    } else {
        int64_t bs2n2d = runInfo.isCmp ?
                             runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSizeVInput :
                             runInfo.boIdx * constInfo.n2S2Dv;
        uint64_t u_bs2n2d = (uint64_t)bs2n2d;
        uint32_t s2n2Low = (uint32_t)(u_bs2n2d & 0xFFFFFFFFULL);
        uint32_t s2n2High = (uint32_t)(u_bs2n2d >> 32U);
        GetBsndTotalOffset(totalOffsetTensor, sparseIndiceTensor, s2n2Low, s2n2High, constInfo.dSizeVInput);
    }

    SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
    LocalTensor<int64_t> totalOffset64Tensor = totalOffsetTensor.ReinterpretCast<int64_t>();
    LocalTensor<KV_T> kvInUb = stage0InBuf[pingPongV0].Get<KV_T>();
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
            static_cast<uint32_t>(constInfo.dSizeVInput),
            static_cast<uint32_t>(constInfo.dSizeVInput),
        };
        GatherPadParams<int32_t> gatherPadparams2;
        constexpr uint8_t offsetType = get_imm(1);
        const bool constPadingCtrl = true;
        const uint8_t cacheMode = ExtractCacheMode(keyGm);
        gather_gm_to_ubuf((__ubuf__ uint8_t*)kvInUb.GetPhyAddr(), (__gm__ uint8_t*)(keyGm.GetPhyAddr()), 0, offsetType,
                          constPadingCtrl, (__ubuf__ int64_t*)totalOffset64Tensor[s2ProcessBaseSize * i].GetPhyAddr(),
                          cacheMode, gatherParams2.burstLen, gatherParams2.dstStride, gatherParams2.offsetNum);
        SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        CastNd2Nz(kvOutUb, kvInUb, dealRows);
        PipeBarrier<PIPE_V>();
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
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop,
                                                                      int64_t dealRow, int64_t s2StartOffset,
                                                                      const RunInfo& runInfo, ConstInfo& constInfo)
{
    int64_t s2LoopCount = (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) ?
                              (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) :
                              runInfo.s2LoopCount;
    int64_t s2Idx = s2StartOffset + s2LoopCount * constInfo.s2BaseSize + runInfo.s2StartIdx;
    uint32_t combineBytes = constInfo.dSizeVInput;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    DataCopyExtParams intriParams;
    intriParams.blockCount = dealRow;
    intriParams.blockLen = combineBytes;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    if constexpr (isPa) {
        uint64_t blockTableBaseOffset = runInfo.boIdx * maxBlockNumPerBatch;
        uint64_t dstOffset = 0;
        uint32_t copyFinishElmenCnt = 0;
        uint32_t curSequence = s2Idx;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        while (copyFinishElmenCnt < dealRow) {
            uint64_t blockIdOffset = curSequence / blockSize;
            uint64_t remainElmenCnt = curSequence % blockSize;
            uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
            uint32_t copyElmenCnt = blockSize - remainElmenCnt;
            if (copyElmenCnt + copyFinishElmenCnt > dealRow) {
                copyElmenCnt = dealRow - copyFinishElmenCnt;
            }
            uint64_t srcOffset = idInBlockTable * paBlockStride + remainElmenCnt * constInfo.n2Size * combineBytes +
                                 (uint64_t)(runInfo.n2oIdx * combineBytes); // BlockNum, BlockSize, N, D
            intriParams.blockCount = copyElmenCnt;                          // base s2 size
            DataCopyPad(kvMergUb[dstOffset * combineDimAlign], keyGm[srcOffset], intriParams, padParams);
            dstOffset += copyElmenCnt;
            copyFinishElmenCnt += copyElmenCnt;
            curSequence += copyElmenCnt;
        }
    } else {
        uint64_t srcOffset = 0;
        if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
            uint64_t kvS2Idx = s2Idx + (runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) :
                                                        cuSeqlensOriKvGm.GetValue(runInfo.boIdx));
            srcOffset = (kvS2Idx * constInfo.n2Size + runInfo.n2oIdx) * combineDim;
        } else {
            srcOffset = ((runInfo.boIdx * constInfo.s2Size + s2Idx) * constInfo.n2Size + runInfo.n2oIdx) * combineDim;
        }
        DataCopyPad(kvMergUb, keyGm[srcOffset], intriParams, padParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo& runInfo, ConstInfo& constInfo)
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
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1, const RunInfo& runInfo, ConstInfo& constInfo)
{
    bool isCmp = runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx;
    if (isCmp) {
        keyGm = cmpKVGm;
        blockTableGm = cmpBlockTableGm;
        blockSize = constInfo.cmpBlockSize;
        maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
    } else {
        keyGm = oriKVGm;
        blockTableGm = oriBlockTableGm;
        blockSize = constInfo.oriBlockSize;
        maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (isCmp) {
            CalSparseCalSize(runInfo, constInfo);
            ProcessSparseKv(outputL1, runInfo, constInfo);
        } else {
            ProcessNotSparseKv(outputL1, runInfo, constInfo);
        }
    } else {
        ProcessNotSparseKv(outputL1, runInfo, constInfo);
    }
    outputL1.SetCrossCore(); // 核间同步
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputL1, const RunInfo& runInfo, ConstInfo& constInfo)
{
    int64_t s2ProcessBaseSize = v0BaseSize;
    int64_t s2RealSize = runInfo.s2RealSize; // 使用sparsecalsize还是s2realSize
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.sparseBlockCount +
                      runInfo.s1oIdx * constInfo.sparseBlockCount; // B, S1, N2(1), K
    }
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx;
    int64_t topkKIdx = cmpS2LoopCnt * constInfo.s2BaseSize;
    WaitFlag<HardEvent::MTE3_MTE2>(Mte3ToMte2V0Id);
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = s2RealSize * sizeof(int32_t);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(sparseIndiceTensor,
                cmpSparseIndicesGm[topkBS1Idx + runInfo.s2StartIdx + cmpS2LoopCnt * constInfo.s2BaseSize], copyInParams,
                padParams);
    SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
    WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);

    if (KV_LAYOUT_T == QSMLA_LAYOUT::PA_BBND) {
        uint32_t s2Stride = constInfo.dSizeVInput;
        if (IsPowerOfTwoBlockSize(blockSize)) {
            DivPABlockIdx(paBlockIdxOffsetTensor, tailBlockOffsetTensor, sparseIndiceTensor, constInfo.sparseBlockSize,
                          blockSize, s2Stride);
            SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
            WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        } else {
            uint32_t validCnt = Min(static_cast<uint32_t>(s2RealSize), s2BaseSize);
            int64_t sparseIdxBase =
                static_cast<int64_t>(topkBS1Idx) + runInfo.s2StartIdx + cmpS2LoopCnt * constInfo.s2BaseSize;
            for (uint32_t i = 0; i < validCnt; ++i) {
                SetPaBlockIdxOffsetScalar(paBlockIdxOffsetTensor, tailBlockOffsetTensor, i,
                                          static_cast<uint32_t>(cmpSparseIndicesGm.GetValue(sparseIdxBase + i)),
                                          static_cast<uint32_t>(constInfo.sparseBlockSize), blockSize, s2Stride);
            }
            SyncPaBlockIdxOffsetScalarToPipe();
        }
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
        uint32_t blockNumStrideL = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        uint32_t blockNumStrideH = 0;
        GetPaTotalOffset(totalOffsetTensor, paBlockIdTensor, tailBlockOffsetTensor, blockNumStrideL, blockNumStrideH);
    } else if (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        uint32_t tPrefix =
            runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) : cuSeqlensOriKvGm.GetValue(runInfo.boIdx); // n2=1
        GetTndTotalOffset(totalOffsetTensor, sparseIndiceTensor, static_cast<uint32_t>(tPrefix),
                          static_cast<uint32_t>(constInfo.n2Size), static_cast<uint32_t>(constInfo.dSizeVInput));
    } else { // BSND
        int64_t bs2n2d = runInfo.isCmp ?
                             runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSizeVInput :
                             runInfo.boIdx * constInfo.n2S2Dv;
        uint64_t u_bs2n2d = (uint64_t)bs2n2d;
        uint32_t s2n2Low = (uint32_t)(u_bs2n2d & 0xFFFFFFFFULL);
        uint32_t s2n2High = (uint32_t)(u_bs2n2d >> 32U);
        GetBsndTotalOffset(totalOffsetTensor, sparseIndiceTensor, s2n2Low, s2n2High, constInfo.dSizeVInput);
    }
    SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
    LocalTensor<int64_t> totalOffset64Tensor = totalOffsetTensor.ReinterpretCast<int64_t>();
    LocalTensor<KV_T> kvInUb = stage0InBuf[pingPongV0].Get<KV_T>();
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
            static_cast<uint32_t>(constInfo.dSizeVInput),
            static_cast<uint32_t>(constInfo.dSizeVInput),
        };
        GatherPadParams<int32_t> gatherPadparams2{GatherPaddingMode::CONSTANT_PADDING_MODE, 2};
        constexpr uint8_t offsetType = get_imm(1);
        const bool constPadingCtrl = true;
        const uint8_t cacheMode = ExtractCacheMode(keyGm);
        gather_gm_to_ubuf((__ubuf__ uint8_t*)kvInUb.GetPhyAddr(), (__gm__ uint8_t*)(keyGm.GetPhyAddr()), 0, offsetType,
                          constPadingCtrl, (__ubuf__ int64_t*)totalOffset64Tensor[s2ProcessBaseSize * i].GetPhyAddr(),
                          cacheMode, gatherParams2.burstLen, gatherParams2.dstStride, gatherParams2.offsetNum);

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
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec1(
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
    float softmaxScale =
        runInfo.isCmp ? static_cast<T>(constInfo.softmaxScale * qDescaleGm.GetValue(0) * cmpKVDescaleGm.GetValue(0)) :
                        static_cast<T>(constInfo.softmaxScale * qDescaleGm.GetValue(0) * oriKVDescaleGm.GetValue(0));
    if (runInfo.s2LoopCount == 0 && !isSinks) {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
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
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
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
                 {s2BaseSize / 32, (uint16_t)(runInfo.halfMRealSize), (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
                  (uint16_t)(Align32Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }
    this->stage1OutQue.template FreeTensor(stage1CastTensor);
    outputBuf.SetCrossCore();
    if (runInfo.s2LoopCount != 0 || (runInfo.s2LoopCount == 0 && isSinks)) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm2ResBuf, RunInfo& runInfo, ConstInfo& constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

    bool lastLoopIsOri = runInfo.s2LoopCount <= runInfo.oriKvLoopEndIdx;
    float vDescale = runInfo.isCmp ? cmpKVDescaleGm.GetValue(0) : oriKVDescaleGm.GetValue(0);
    float vDescalePre = lastLoopIsOri ? oriKVDescaleGm.GetValue(0) : cmpKVDescaleGm.GetValue(0);

    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == 1)) {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, true, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, vDescale,
                    vDescalePre);
            } else {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, vDescale, 1.0);
            }
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            if (unlikely(runInfo.s2LoopCount == 1)) {
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, true, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, vDescale,
                    vDescalePre);
            } else {
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, vDescale,
                    1.0);
            }
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize,
                                                                  dTemplateAlign64, vDescale);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo& runInfo, ConstInfo& constInfo,
                                                                    LocalTensor<VEC2_RES_T>& vec2ResUb,
                                                                    int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    constexpr float hifp8ScaleValueRec = 1 / 16.0;
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;
    // 除以缩放系数
    if constexpr (IsSameType<Q_T, hifloat8_t>::value) {
        Muls(vec2ResUb, vec2ResUb, hifp8ScaleValueRec, vec2CalcSize);
    }
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
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo& runInfo, ConstInfo& constInfo,
                                                                        LocalTensor<VEC2_RES_T>& vec2ResUb,
                                                                        int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo& constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if (LAYOUT_T == QSMLA_LAYOUT::TND) {
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
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t* attentionOut, ConstInfo& constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        if (constInfo.needInit == 1) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            InitOutputSingleCore(constInfo);
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* qDescale, __gm__ uint8_t* oriKVDescale,
    __gm__ uint8_t* cmpKVDescale, __gm__ uint8_t* cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
    __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks, __gm__ uint8_t* sequsedOriKv,
    __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
{
    if (qDescale != nullptr) {
        qDescaleGm.SetGlobalBuffer((__gm__ float*)qDescale);
    }

    if (oriKV != nullptr) {
        oriKVGm.SetGlobalBuffer((__gm__ KV_T*)(oriKV));
        oriKVDescaleGm.SetGlobalBuffer((__gm__ float*)oriKVDescale);
    }

    if constexpr (isPa) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)oriBlockTable);
    }

    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T*)cmpKV);
        if constexpr (isPa) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)cmpBlockTable);
        }
        cmpKVDescaleGm.SetGlobalBuffer((__gm__ float*)cmpKVDescale);
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t*)cmpSparseIndices);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T*)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
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
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo& constInfo)
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
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe* pipe, ConstInfo& constInfo)
{
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512);   // sinksBuf内存申请512B

    tPipe->InitBuffer(stage0InBuf[0],
                      dVTemplateTypeInput * 8 * sizeof(KV_T)); // V0阶段每次处理8个seq, 开2 buffer  共10Kb
    tPipe->InitBuffer(stage0InBuf[1], dVTemplateTypeInput * 8 * sizeof(KV_T));
    tPipe->InitBuffer(stage0OutBuf[0],
                      dVTemplateTypeInput * (8 + 1) * sizeof(Q_T)); // kv输入D轴608, V0阶段每次处理8个seq, 开2 buffer
    tPipe->InitBuffer(stage0OutBuf[1], dVTemplateTypeInput * (8 + 1) * sizeof(Q_T));

    tPipe->InitBuffer(stage1OutQue, 1, vec1Srcstride * s2BaseSize * sizeof(Q_T)); // 16.25KB

    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

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

    constexpr int32_t SIZE_128 = 128;
    constexpr int32_t SIZE_64 = 64;
    constexpr int32_t SIZE_2 = 2;
    constexpr int32_t SIZE_4 = 4;

    // vselrIndexesBuf
    tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)], SIZE_128);
    tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)], SIZE_64);

    LocalTensor<uint8_t> vselrIndexesTensor =
        vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
    for (int i = 0; i < SIZE_128; i++) {
        vselrIndexesTensor.SetValue(i, i * SIZE_2);
    }
    vselrIndexesTensor =
        vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
    for (int i = 0; i < SIZE_64; i++) {
        vselrIndexesTensor.SetValue(i, i * SIZE_4);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetExtremeValue(T& negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float*)&tmp1);
}

TEMPLATES_DEF
class SCFABlockVecDummy {
public:
    __aicore__ inline SCFABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t* attentionOut, ConstInfo& constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* qDescale,
                                            __gm__ uint8_t* oriKVDescale, __gm__ uint8_t* cmpKVDescale,
                                            __gm__ uint8_t* cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                                            __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t* sequsedQ,
                                            __gm__ uint8_t* sinks, __gm__ uint8_t* sequsedOriKv,
                                            __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
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
} // namespace BaseApi
#endif // QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH92_H
