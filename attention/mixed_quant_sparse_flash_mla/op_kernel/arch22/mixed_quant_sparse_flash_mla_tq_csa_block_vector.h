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
 * \file mixed_quant_sparse_flash_mla_tq_csa_block_vector.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_TQ_CSA_BLOCK_VECTOR_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_TQ_CSA_BLOCK_VECTOR_H

#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "mixed_quant_sparse_flash_mla_tq_common.h"
#include "mixed_quant_sparse_flash_mla_int8_csa_input.h"

namespace SASKernel {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SAST>
class KvQuantSparseFlashMlaCsaBlockVector {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename SAST::kvType;
    using OUT_T = typename SAST::outputType;
    using UPDATE_T = T;
    using SINKS_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline KvQuantSparseFlashMlaCsaBlockVector(){};
    template <bool INT8_FAST = true>
    __aicore__ inline void ProcessVec0L(const RunInfo& mqVecRunInfo);
    __aicore__ inline void InitInt8Input(const Int8CsaInput<SAST>& input)
    {
        int8Input = input;
    }
    __aicore__ inline void InitInt8Outputs();
    __aicore__ inline void PrepareInt8SelectionGeneric(Int8CsaInput<SAST>& selection, uint64_t planOffset,
                                                       uint32_t& oriRows, uint32_t& cmpRows);
    template <bool INT8_FAST = true>
    __aicore__ inline void ProcessInt8Kv(const RunInfo& info, uint32_t stageRowStart, bool wrapOutput);
    __aicore__ inline void ConvertInt8Batch(uint32_t slot, uint32_t staged, uint64_t outOffset, bool wrapOutput);
    __aicore__ inline void SetInt8SelectionPrepared(bool prepared)
    {
        preparedInt8 = prepared;
    }
    __aicore__ inline void PrepareInt8Selection(Int8CsaInput<SAST>& selection, uint64_t planOffset, uint32_t& oriRows,
                                                uint32_t& cmpRows);
    __aicore__ inline void ProcessVec1L(const RunInfo& mqVecInfo);
    __aicore__ inline void ProcessVec2L(const RunInfo& mqVecInfo);
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitParams(const struct ConstInfo& mqVecConstInfo,
                                      const optiling::MixedQuantSparseFlashMlaTqTilingData* __restrict tilingData);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<KV_T>& kvMergeGm, const GlobalTensor<KV_T>& oriKvGm,
                                                const GlobalTensor<KV_T>& cmpKvGm, const GlobalTensor<uint8_t>& cmpTqGm,
                                                const GlobalTensor<int32_t>& oriBlockTableGm,
                                                const GlobalTensor<int32_t>& cmpBlockTableGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm,
                                                GlobalTensor<int32_t> topKGm, GlobalTensor<T> sinksGm,
                                                GlobalTensor<T> softmaxLseGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CopySinksIn();
    __aicore__ inline void SliceAndContactSinksValue(uint32_t nIdx, uint32_t mqDealRows);
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t mqDealRows, uint32_t mqColumns, uint32_t mqActualColumns);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t mqDealRows, uint32_t mqColumns, uint32_t mqActualColumns);
    // ================================Vector0==========================================
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const RunInfo& mqVecRunInfo, int64_t s2IdLimit,
                                             const LocalTensor<int32_t>& blockTableCache, uint32_t blockTableCacheSize);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t& realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo& mqVecRunInfo, const LocalTensor<int32_t>& topkCache,
                                        int64_t topkCacheStart, uint32_t topkCacheSize);
    __aicore__ inline void CopyInKv(int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo& mqVecRunInfo,
                                    const LocalTensor<int32_t>& blockTableCache, uint32_t blockTableCacheSize);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo& mqVecRunInfo);
    __aicore__ inline void DequantTq4Rows(LocalTensor<uint8_t> slots, LocalTensor<KV_T> output, int64_t rowCount);
    __aicore__ inline void CopyInSingleKv(int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo& mqVecRunInfo);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo,
                                                uint32_t startRow, uint32_t mqDealRows, uint32_t mqColumns,
                                                uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo,
                                                 LocalTensor<T>& mmResUb, LocalTensor<uint8_t>& softmaxTmpUb,
                                                 uint32_t startRow, uint32_t mqDealRows, uint32_t mqColumns,
                                                 uint32_t mqActualColumns);

    __aicore__ inline void ElewiseCompute(const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo, uint32_t startRow,
                                          const LocalTensor<T>& mmResUb, uint32_t mqDealRows, uint32_t mqColumns);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T>& mmResUb, uint32_t mqDealRows, uint32_t mqColumns,
                                       int64_t startId, int64_t endId);
    __aicore__ inline void ProcessLse(const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo,
                                                uint32_t startRow, uint32_t mqDealRows, uint32_t mqColumns,
                                                uint32_t mqActualColumns);
    __aicore__ inline void ProcessVec2Inner(const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo,
                                            uint32_t mStartRow, uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo& mqVecInfo, LocalTensor<OUT_T>& attenOutUb,
                                                uint32_t mqWorkspaceRow, uint32_t mqDealRows, uint32_t mqColumns,
                                                uint32_t mqActualColumns);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo& mqVecInfo, LocalTensor<T>& mqBmm2ResultUb,
                                          uint32_t mqWorkspaceRow, uint32_t mqDealRows, uint32_t mqColumns,
                                          uint32_t mqActualColumns);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo& mqVecInfo, LocalTensor<T>& mqBmm2ResultUb,
                                              uint32_t mqWorkspaceRow, uint32_t mqDealRows, uint32_t mqColumns,
                                              uint32_t mqActualColumns);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo& mqVecInfo, LocalTensor<T>& mqBmm2ResultUb,
                                             uint32_t mqWorkspaceRow, uint32_t mqDealRows, uint32_t mqColumns,
                                             uint32_t mqActualColumns);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    Int8CsaInput<SAST> int8Input;
    bool preparedInt8 = false;
    uint32_t preparedOriRows = 0;
    static constexpr bool PAGE_ATTENTION = SAST::pageAttention;
    static constexpr int TEMPLATE_MODE = SAST::templateMode;
    static constexpr bool FLASH_DECODE = SAST::flashDecode;
    static constexpr SAS_LAYOUT LAYOUT_T = SAST::layout;
    static constexpr SAS_LAYOUT KV_LAYOUT_T = SAST::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 3;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint64_t SYNC_INPUT_V0BUF_FLAG = 7;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::MQ_BUFFER_BYTES_32K;
    static constexpr uint32_t INPUT2_BUFFER_OFFSET = ConstInfo::MQ_BUFFER_BYTES_16K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::MQ_BUFFER_BYTES_1K;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::MQ_BUFFER_BYTES_32K / sizeof(T);
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t MAX_N1_SIZE = 128U;
    // tmpBuff1: dtype-specific dequant scratch followed by two aligned raw-slot regions.
    static constexpr uint32_t TQ4_MAX_STAGED_ROWS = INPUT2_BUFFER_OFFSET / (TQ4_HEAD_DIM * sizeof(KV_T));
    static constexpr uint32_t TQ4_RAW_REGION_BYTES = TQ4_MAX_STAGED_ROWS * TQ4_SLOT_ROW_BYTES;
    static constexpr uint32_t TQ4_RAW_BUFFER_BYTES = 2U * TQ4_RAW_REGION_BYTES;
    static constexpr bool TQ4_FAST_BF16 = IsSameType<KV_T, bfloat16_t>::value;
    static constexpr uint32_t TQ4_SLOW_DEQUANT_BYTES_PER_ROW =
        TQ4_HEAD_DIM * (sizeof(float) + sizeof(half) + sizeof(int32_t));
    // BF16 lookup uses outputBuff1 for indices, then reuses it as FP32 scale work after Gather.
    static constexpr uint32_t TQ4_FAST_DEQUANT_BATCH_ROWS = 16U;
    static constexpr uint32_t TQ4_DEQUANT_BATCH_ROWS =
        TQ4_FAST_BF16 ? TQ4_FAST_DEQUANT_BATCH_ROWS :
                        (ConstInfo::MQ_BUFFER_BYTES_32K - TQ4_RAW_BUFFER_BYTES) / TQ4_SLOW_DEQUANT_BYTES_PER_ROW;
    static constexpr uint32_t TQ4_PACKED_BATCH_BYTES = TQ4_FAST_BF16 ? TQ4_DEQUANT_BATCH_ROWS * TQ4_PACKED_BYTES : 0U;
    static constexpr uint32_t TQ4_WORK_BYTES =
        TQ4_FAST_BF16 ? 0U : TQ4_DEQUANT_BATCH_ROWS * TQ4_HEAD_DIM * sizeof(float);
    static constexpr uint32_t TQ4_LOOKUP_ELEMENT_COUNT =
        TQ4_FAST_BF16 ? TQ4_DEQUANT_BATCH_ROWS * TQ4_PACKED_BYTES : TQ4_DEQUANT_BATCH_ROWS * TQ4_HEAD_DIM;
    static constexpr uint32_t TQ4_SIGNED_BYTES = TQ4_LOOKUP_ELEMENT_COUNT * sizeof(half);
    static constexpr uint32_t TQ4_INDEX_BYTES = TQ4_FAST_BF16 ? 0U : TQ4_LOOKUP_ELEMENT_COUNT * sizeof(int32_t);
    static constexpr uint32_t TQ4_SIGNED_OFFSET = TQ4_PACKED_BATCH_BYTES + TQ4_WORK_BYTES;
    static constexpr uint32_t TQ4_INDEX_OFFSET = TQ4_SIGNED_OFFSET + TQ4_SIGNED_BYTES;
    static constexpr uint32_t TQ4_RAW_OFFSET = TQ4_INDEX_OFFSET + TQ4_INDEX_BYTES;
    static constexpr uint32_t TQ4_CENTROID_COUNT = 16U;
    static constexpr uint32_t TQ4_SCALE_GATHER_INDEX_OFFSET = TQ4_CENTROID_COUNT * sizeof(float);
    static constexpr uint32_t TQ4_SCALE_GATHER_INDEX_BYTES = TQ4_DEQUANT_BATCH_ROWS * sizeof(uint32_t);
    static constexpr uint32_t TQ4_SCALE_CARRIER_BYTES = TQ4_DEQUANT_BATCH_ROWS * BYTE_BLOCK;
    static constexpr uint32_t TQ4_SCALE_BROADCAST_OFFSET =
        (TQ4_DEQUANT_BATCH_ROWS * sizeof(float) + BYTE_BLOCK - 1U) / BYTE_BLOCK * BYTE_BLOCK;
    static constexpr uint32_t TQ4_SCALE_BROADCAST_BYTES = (TQ4_DEQUANT_BATCH_ROWS + FP32_BLOCK_ELEMENT_NUM - 1U) /
                                                          FP32_BLOCK_ELEMENT_NUM * FP32_BLOCK_ELEMENT_NUM *
                                                          FP32_BLOCK_ELEMENT_NUM * sizeof(float);
    // Vec0 and Vec2 execute in different pipeline stages, but keep Vec2's existing 512B scratch region reserved.
    static constexpr uint32_t V0_VEC2_TEMP_OFFSET = 384U * sizeof(T);
    static constexpr uint32_t V0_VEC2_TEMP_BYTES = ConstInfo::MQ_BUFFER_BYTES_512;
    static constexpr uint32_t TQ4_TOPK_CACHE_BYTES = V0_VEC2_TEMP_OFFSET;
    static constexpr uint32_t TQ4_BLOCK_TABLE_CACHE_OFFSET = V0_VEC2_TEMP_OFFSET + V0_VEC2_TEMP_BYTES;
    static constexpr uint32_t TQ4_BLOCK_TABLE_CACHE_BYTES =
        ConstInfo::MQ_BUFFER_BYTES_8K - TQ4_BLOCK_TABLE_CACHE_OFFSET;
    static constexpr uint32_t TQ4_TOPK_CACHE_CAPACITY = TQ4_TOPK_CACHE_BYTES / sizeof(int32_t);
    static constexpr uint32_t TQ4_BLOCK_TABLE_CACHE_CAPACITY = TQ4_BLOCK_TABLE_CACHE_BYTES / sizeof(int32_t);
    static_assert(TQ4_DEQUANT_BATCH_ROWS > 0U, "TQ4 dequant scratch cannot hold one row");
    static_assert(TQ4_DEQUANT_BATCH_ROWS <= TQ4_MAX_STAGED_ROWS, "TQ4 dequant batch exceeds staged rows");
    static_assert(TQ4_RAW_OFFSET + TQ4_RAW_BUFFER_BYTES <= ConstInfo::MQ_BUFFER_BYTES_32K,
                  "TQ4 batched dequant scratch exceeds tmpBuff1");
    static_assert(!TQ4_FAST_BF16 ||
                      TQ4_DEQUANT_BATCH_ROWS * TQ4_HEAD_DIM * sizeof(float) <= ConstInfo::MQ_BUFFER_BYTES_32K,
                  "TQ4 BF16 scale work exceeds outputBuff1");
    static_assert(!TQ4_FAST_BF16 || TQ4_LOOKUP_ELEMENT_COUNT * sizeof(int32_t) <= ConstInfo::MQ_BUFFER_BYTES_32K,
                  "TQ4 BF16 indices exceed outputBuff1");
    static_assert(TQ4_SCALE_GATHER_INDEX_OFFSET + TQ4_SCALE_GATHER_INDEX_BYTES <= ConstInfo::MQ_BUFFER_BYTES_256,
                  "TQ4 scale gather indices overlap tqCentBuff");
    static_assert(TQ4_SCALE_CARRIER_BYTES + TQ4_DEQUANT_BATCH_ROWS * sizeof(half) <= TQ4_SIGNED_BYTES,
                  "TQ4 scale extraction exceeds signed-code scratch");
    static_assert((TQ4_FAST_BF16 && TQ4_SCALE_BROADCAST_OFFSET + TQ4_SCALE_BROADCAST_BYTES <= TQ4_PACKED_BATCH_BYTES) ||
                      (!TQ4_FAST_BF16 && TQ4_SCALE_BROADCAST_OFFSET + TQ4_SCALE_BROADCAST_BYTES <= TQ4_INDEX_BYTES),
                  "TQ4 scale broadcast exceeds index scratch");
    static_assert(TQ4_TOPK_CACHE_BYTES <= TQ4_BLOCK_TABLE_CACHE_OFFSET, "TQ4 index caches overlap in v0ValidSizeBuff");
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr SINKS_T R0 = 1.0f;

    const optiling::MixedQuantSparseFlashMlaTqTilingData* __restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo mqVecConstInfo = {};

    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<T> sinksGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<T> softmaxLseGm;

    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<KV_T> oriKvGm_;
    GlobalTensor<KV_T> cmpKvGm_;
    GlobalTensor<uint8_t> cmpTqGm_;
    GlobalTensor<int32_t> oriBlockTableGm_;
    GlobalTensor<int32_t> cmpBlockTableGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;  // 32K
    TBuf<> inputBuff2;  // 16K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 32K

    TBuf<> tmpBuff1;        // 32K
    TBuf<> v0ValidSizeBuff; // 8K
    TBuf<> tqCentBuff;      // 256B
    TBuf<> tqByteLutBuff;   // 1K

    TBuf<> sinksBuff;     // 1K
    TBuf<> sinksBrcbBuff; // 12K

    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
    LocalTensor<SINKS_T> sinksUb;
    LocalTensor<SINKS_T> sinksBrcbUb;

    uint32_t mergeMte3Idx = 0;
};

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::MQ_BUFFER_BYTES_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::MQ_BUFFER_BYTES_16K * 2); // 2:pingpong
    pipe->InitBuffer(outputBuff1, ConstInfo::MQ_BUFFER_BYTES_32K);
    if (mqVecConstInfo.returnSoftmaxLse) {
        pipe->InitBuffer(outputBuff2, ConstInfo::MQ_BUFFER_BYTES_1K);
    }

    pipe->InitBuffer(tmpBuff1, ConstInfo::MQ_BUFFER_BYTES_32K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::MQ_BUFFER_BYTES_8K);
    if (mqVecConstInfo.kvQuantMode == 3) {
        pipe->InitBuffer(tqCentBuff, ConstInfo::MQ_BUFFER_BYTES_256);
        if constexpr (TQ4_FAST_BF16) {
            pipe->InitBuffer(tqByteLutBuff, ConstInfo::MQ_BUFFER_BYTES_1K);
            LocalTensor<uint32_t> byteLut = tqByteLutBuff.Get<uint32_t>();
            // RNE BF16 bits in physical nibble order; identical to LoadTq4Centroids()[n ^ 8].
            constexpr uint32_t centroidBits[TQ4_CENTROID_COUNT] = {0xBDF8U, 0xBDBBU, 0xBD92U, 0xBD62U, 0xBD29U, 0xBCECU,
                                                                   0xBC8BU, 0xBBBAU, 0x3BB3U, 0x3C8AU, 0x3CEAU, 0x3D28U,
                                                                   0x3D61U, 0x3D91U, 0x3DBBU, 0x3DF7U};
            LocalTensor<int32_t> lowBits = tqCentBuff.Get<int32_t>();
#pragma unroll
            for (uint32_t low = 0; low < TQ4_CENTROID_COUNT; ++low) {
                lowBits.SetValue(low, static_cast<int32_t>(centroidBits[low]));
            }
            PipeBarrier<PIPE_ALL>();
            LocalTensor<int32_t> pairs = byteLut.ReinterpretCast<int32_t>();
#pragma unroll
            for (uint32_t high = 0; high < TQ4_CENTROID_COUNT; ++high) {
                Adds(pairs[high * TQ4_CENTROID_COUNT], lowBits, static_cast<int32_t>(centroidBits[high] << 16U),
                     TQ4_CENTROID_COUNT);
            }
        } else {
            LoadTq4Centroids(tqCentBuff.Get<float>());
        }
        LocalTensor<uint32_t> scaleGatherIndices =
            tqCentBuff.Get<uint32_t>()[TQ4_SCALE_GATHER_INDEX_OFFSET / sizeof(uint32_t)];
        for (uint32_t row = 0; row < TQ4_DEQUANT_BATCH_ROWS; ++row) {
            scaleGatherIndices.SetValue(row, row * BYTE_BLOCK);
        }
        PipeBarrier<PIPE_ALL>();
    }

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::MQ_BUFFER_BYTES_1K * mqVecConstInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::MQ_BUFFER_BYTES_1K * mqVecConstInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::MQ_BUFFER_BYTES_1K * mqVecConstInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::MQ_BUFFER_BYTES_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::MQ_BUFFER_BYTES_1K);

    pipe->InitBuffer(sinksBuff, MAX_N1_SIZE * sizeof(SINKS_T));
    // 分配256+N1大小内存，其中256是m轴VEC最大切块
    pipe->InitBuffer(sinksBrcbBuff, MAX_N1_SIZE * sizeof(SINKS_T) * BLOCK_ELEMENT_NUM * 3U);

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff2.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();

    sinksUb = sinksBuff.Get<SINKS_T>();
    sinksBrcbUb = sinksBrcbBuff.Get<SINKS_T>();
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::InitParams(
    const struct ConstInfo& mqVecConstInfo, const optiling::MixedQuantSparseFlashMlaTqTilingData* __restrict tilingData)
{
    this->mqVecConstInfo = mqVecConstInfo;
    this->tilingData = tilingData;
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::InitVec0GlobalTensor(
    const GlobalTensor<KV_T>& kvMergeGm, const GlobalTensor<KV_T>& oriKvGm, const GlobalTensor<KV_T>& cmpKvGm,
    const GlobalTensor<uint8_t>& cmpTqGm, const GlobalTensor<int32_t>& oriBlockTableGm,
    const GlobalTensor<int32_t>& cmpBlockTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->oriKvGm_ = oriKvGm;
    this->cmpKvGm_ = cmpKvGm;
    this->cmpTqGm_ = cmpTqGm;
    this->oriBlockTableGm_ = oriBlockTableGm;
    this->cmpBlockTableGm_ = cmpBlockTableGm;
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm, GlobalTensor<int32_t> actualSeqLengthsQGm,
    GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<int32_t> topKGm, GlobalTensor<SINKS_T> sinksGm,
    GlobalTensor<T> softmaxLseGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->topkGm_ = topKGm;
    this->sinksGm = sinksGm;
    this->softmaxLseGm = softmaxLseGm;
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::InitVec2GlobalTensor(
    GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm,
    GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::CopySinksIn()
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = mqVecConstInfo.qHeadNum * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, sinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    uint32_t repeatTimes = (mqVecConstInfo.qHeadNum + BLOCK_ELEMENT_NUM - 1U) / BLOCK_ELEMENT_NUM;
    Brcb(sinksBrcbUb, sinksUb, repeatTimes, {1, BLOCK_ELEMENT_NUM});
    PipeBarrier<PIPE_V>();

    DataCopyParams repeatParams;
    repeatParams.blockCount = 1; // 搬到有一个块超过单个vec核减分核M轴大小即可，核间切分每个vec256
    repeatParams.blockLen = mqVecConstInfo.qHeadNum;
    repeatParams.srcStride = 0U;
    repeatParams.dstStride = 0U;
    for (uint32_t i = 1U; i <= 256U / mqVecConstInfo.qHeadNum; i++) {
        DataCopy(sinksBrcbUb[mqVecConstInfo.qHeadNum * BLOCK_ELEMENT_NUM * i], sinksBrcbUb, repeatParams);
    }
    PipeBarrier<PIPE_V>();
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::SliceAndContactSinksValue(uint32_t nIdx,
                                                                                            uint32_t mqDealRows)
{
    // WholeReduceMax接口中repeatTimes支持范围（0,255），因此需要分多次调用WholeReduceMax，每次repeatTime=128
    uint32_t repeatTimesOnce = 128;
    uint32_t loopTimes = (mqDealRows + repeatTimesOnce - 1) / repeatTimesOnce;
    uint32_t repeatTimes = repeatTimesOnce;

    for (uint32_t loop = 0; loop < loopTimes; ++loop) {
        if (loop == loopTimes - 1) {
            repeatTimes = mqDealRows - loop * repeatTimesOnce;
        }
        WholeReduceMax(softmaxMaxDefaultUb[loop * repeatTimesOnce],
                       sinksBrcbUb[(nIdx + loop * repeatTimesOnce) * BLOCK_ELEMENT_NUM],
                       BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM, repeatTimes, 1, 0, 1, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::InitSoftmaxDefaultBuffer()
{
    CopySinksIn();
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, R0, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ElewiseCompute(
    const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo, uint32_t startRow, const LocalTensor<T>& mmResUb,
    uint32_t mqDealRows, uint32_t mqColumns)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->tqBaseParams.softmaxScale), mqDealRows * mqColumns);
    PipeBarrier<PIPE_V>();

    // INT8 input tiles contain only selected, valid tokens. The softmax
    // implementation already masks the aligned tail using the actual N size.
    if constexpr (SAST::int8Kv) {
        return;
    }

    // cmp_sparse_indices is capacity-sized and may be padded with -1.  The Cube
    // path still computes the capacity-sized tile, so mask the padded columns
    // before softmax instead of allowing stale merge-cache rows to contribute.
    if (mqVecInfo.isOriOnly) {
        uint32_t rowBase = mqVecSplitInfo.nBufferStartM + mqVecSplitInfo.vecStartM + startRow;
        int64_t tileStart = static_cast<int64_t>(mqVecInfo.s2StartPoint) +
                            static_cast<int64_t>(mqVecInfo.s2Idx) * mqVecConstInfo.s2BaseSize;
        for (uint32_t row = 0; row < mqDealRows; ++row) {
            uint32_t qRel = (rowBase + row) / mqVecConstInfo.gSize;
            int64_t queryPos = static_cast<int64_t>(mqVecInfo.s1Idx) + qRel;
            int64_t maskRight = static_cast<int64_t>(mqVecInfo.actOriS2Size) -
                                static_cast<int64_t>(mqVecInfo.actS1Size) + queryPos + mqVecConstInfo.oriWinRight;
            int64_t maskLeft = maskRight - mqVecConstInfo.oriWinRight - mqVecConstInfo.oriWinLeft;
            if (maskLeft < 0) {
                maskLeft = 0;
            }
            int64_t localLeft = maskLeft - tileStart;
            int64_t localRight = maskRight - tileStart;
            if (localLeft > 0) {
                int64_t leftMaskEnd = Min(localLeft - 1, static_cast<int64_t>(mqColumns) - 1);
                SetInfInBlk(mmResUb[row * mqColumns], 1, mqColumns, 0, leftMaskEnd);
            }
            if (localRight + 1 < static_cast<int64_t>(mqColumns)) {
                int64_t rightMaskStart = Max(localRight + 1, static_cast<int64_t>(0));
                SetInfInBlk(mmResUb[row * mqColumns], 1, mqColumns, rightMaskStart,
                            static_cast<int64_t>(mqColumns) - 1);
            }
        }
    } else if (mqVecConstInfo.kvQuantMode == 3 && mqVecConstInfo.sparseBlockCount > 0 && mqVecInfo.cmpS2IdLimit > 0) {
        // TurboQuant tiles one query's gSize heads together, so every row in this block shares one top-k row.
        uint64_t topkRow = mqVecInfo.topKBaseOffset;
        int32_t probeCount = static_cast<int32_t>(mqVecConstInfo.sparseBlockCount);
        if (mqVecInfo.cmpS2IdLimit < probeCount) {
            probeCount = mqVecInfo.cmpS2IdLimit;
        }
        int32_t validCount = probeCount;
        int32_t probeTopk = probeCount > 0 ? topkGm_.GetValue(topkRow + probeCount - 1) : -1;
        if (probeTopk < 0) {
            validCount = 0;
            while (validCount < probeCount && topkGm_.GetValue(topkRow + static_cast<uint32_t>(validCount)) >= 0) {
                ++validCount;
            }
        }
        int32_t tileStart = static_cast<int32_t>(mqVecInfo.relativeS2Idx * mqVecConstInfo.s2BaseSize);
        int32_t maskStart = validCount - tileStart;
        if (maskStart < 0) {
            maskStart = 0;
        }
        if (maskStart < static_cast<int32_t>(mqVecInfo.actualSingleProcessSInnerSize)) {
            SetInfInBlk(mmResUb, mqDealRows, mqColumns, maskStart, static_cast<int64_t>(mqColumns) - 1);
        }
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::SetInfInBlk(const LocalTensor<T>& mmResUb,
                                                                              uint32_t mqDealRows, uint32_t mqColumns,
                                                                              int64_t startId, int64_t endId)
{
    if (startId > endId) {
        return;
    }
    int64_t firstBlock = startId / BLOCK_ELEMENT_NUM;
    int64_t lastBlock = endId / BLOCK_ELEMENT_NUM;
    for (int64_t block = firstBlock; block <= lastBlock; ++block) {
        int64_t blockStart = block * BLOCK_ELEMENT_NUM;
        int64_t begin = startId > blockStart ? startId - blockStart : 0;
        int64_t finish = endId < blockStart + BLOCK_ELEMENT_NUM - 1 ? endId - blockStart : BLOCK_ELEMENT_NUM - 1;
        uint64_t maskValue = 0;
        for (int64_t bit = begin; bit <= finish; ++bit) {
            maskValue |= (1ULL << static_cast<uint64_t>(bit));
        }
        uint64_t mask[2] = {maskValue, 0};
        Duplicate(mmResUb[blockStart], SOFTMAX_MIN_NUM, mask, mqDealRows, 1, mqColumns / BLOCK_ELEMENT_NUM);
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ProcessLse(const RunInfo& mqVecInfo,
                                                                             const MSplitInfo& mqVecSplitInfo)
{
    if (mqVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t lseOffset;
    if (mqVecConstInfo.outputLayout == SAS_LAYOUT::TND) {
        uint32_t tBase = actualSeqLengthsQGm.GetValue(mqVecInfo.bIdx);
        lseOffset = (tBase + mqVecInfo.s1Idx) * mqVecConstInfo.gSize +                    // T轴、s1轴偏移
                    mqVecInfo.n2IdxReal * mqVecConstInfo.qSeqSize * mqVecConstInfo.gSize; // N2轴偏移
    } else if (mqVecConstInfo.outputLayout == SAS_LAYOUT::BSND) {
        lseOffset =
            mqVecInfo.bIdx * mqVecConstInfo.qSeqSize * mqVecConstInfo.kvHeadNum * mqVecConstInfo.gSize + // B轴偏移
            mqVecInfo.n2IdxReal * mqVecConstInfo.qSeqSize * mqVecConstInfo.gSize +                       // N2轴偏移
            mqVecInfo.s1Idx * mqVecConstInfo.gSize;                                                      // S1轴偏移
    }
    lseOffset = lseOffset + mqVecSplitInfo.nBufferStartM + mqVecSplitInfo.vecStartM;
    uint32_t baseOffset = mqVecSplitInfo.nBufferStartM / 2;
    uint32_t outIdx = mqVecInfo.loop % (mqVecConstInfo.preLoadNum);
    uint32_t softmaxOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    auto sumTensor = softmaxSumUb[softmaxOffset];
    auto maxTensor = softmaxMaxUb[softmaxOffset];
    auto outLSETensor = outputBuff2.Get<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = mqVecSplitInfo.vecDealM * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    PipeBarrier<PIPE_V>();
    Log(outLSETensor, sumTensor, mqVecSplitInfo.vecDealM);
    PipeBarrier<PIPE_V>();
    Add(outLSETensor, outLSETensor, maxTensor, mqVecSplitInfo.vecDealM);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    DataCopyPad(softmaxLseGm[lseOffset], outLSETensor, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::SoftmaxFlashV2Compute(
    const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo, LocalTensor<T>& mmResUb,
    LocalTensor<uint8_t>& softmaxTmpUb, uint32_t startRow, uint32_t mqDealRows, uint32_t mqColumns,
    uint32_t mqActualColumns)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mqVecSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = mqVecInfo.loop % (mqVecConstInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (mqVecInfo.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb[startRow];
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (mqVecInfo.loop - 1) % (mqVecConstInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    if (mqActualColumns != 0) {
        SoftMaxShapeInfo srcShape{mqDealRows, mqColumns, mqDealRows, mqActualColumns};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, SAS_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        uint32_t dealRowCountAlign = SASAlign(mqDealRows, FP32_BLOCK_ELEMENT_NUM);
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCountAlign);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCountAlign);
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::DealBmm1ResBaseBlock(
    const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo, uint32_t startRow, uint32_t mqDealRows,
    uint32_t mqColumns, uint32_t loopId)
{
    uint32_t computeSize = mqDealRows * mqColumns;
    uint64_t inOutGmOffset = (mqVecInfo.loop % mqVecConstInfo.preLoadNum) * mqVecConstInfo.mmResUbSize +
                             (mqVecSplitInfo.nBufferStartM + mqVecSplitInfo.vecStartM + startRow) * mqColumns;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(mqVecInfo, mqVecSplitInfo, startRow, mmResUb, mqDealRows, mqColumns);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(mqVecInfo, mqVecSplitInfo, mmResUb, softmaxTmpUb, startRow, mqDealRows, mqColumns,
                          mqVecInfo.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    pingpongFlag ^= 1; // pingpong 0 1 切换

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ProcessVec1SingleBuf(const RunInfo& mqVecInfo,
                                                                                       const MSplitInfo& mqVecSplitInfo)
{
    if (mqVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = mqVecInfo.actualSingleProcessSInnerSize == 0 ?
                              16 :
                              BASE_BLOCK_MAX_ELEMENT_NUM / mqVecInfo.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mqVecSplitInfo.vecDealM) {
        mSplitSize = mqVecSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mqVecSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mqVecSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;

    SliceAndContactSinksValue((mqVecSplitInfo.nBufferStartM + mqVecSplitInfo.vecStartM) % mqVecConstInfo.qHeadNum,
                              mqVecSplitInfo.vecDealM);

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(mqVecInfo, mqVecSplitInfo, i * mSplitSize, dealSize,
                             mqVecInfo.actualSingleProcessSInnerSizeAlign, i);
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::GetRealS2Idx(
    int64_t s2GmOffset, int64_t& realS2Idx, int64_t topkGmBaseOffset, const RunInfo& mqVecRunInfo,
    const LocalTensor<int32_t>& topkCache, int64_t topkCacheStart, uint32_t topkCacheSize)
{
    int64_t cmpS2Offset = s2GmOffset;
    int64_t topkGmIdx = cmpS2Offset / mqVecConstInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= mqVecConstInfo.sparseBlockCount || s2GmOffset >= mqVecRunInfo.v0S2DealSize)) {
        realS2Idx = -1;
        return;
    }
    int32_t topkValue = 0;
    int64_t topkCacheIdx = topkGmIdx - topkCacheStart;
    if (topkCacheIdx >= 0 && topkCacheIdx < static_cast<int64_t>(topkCacheSize)) {
        topkValue = topkCache.GetValue(static_cast<uint32_t>(topkCacheIdx));
    } else {
        topkValue = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx);
    }
    realS2Idx = topkValue * static_cast<int64_t>(mqVecConstInfo.sparseBlockSize) +
                static_cast<int64_t>(cmpS2Offset % mqVecConstInfo.sparseBlockSize);
}

template <typename SAST>
__aicore__ inline int64_t KvQuantSparseFlashMlaCsaBlockVector<SAST>::GetKeyGmOffset(
    int64_t realS2Idx, const RunInfo& mqVecRunInfo, int64_t s2IdLimit, const LocalTensor<int32_t>& blockTableCache,
    uint32_t blockTableCacheSize)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyGmOffset = 0;
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_BSND || KV_LAYOUT_T == SAS_LAYOUT::PA_BNSD) {
        int64_t blkTableIdx = realS2Idx / mqVecConstInfo.paCmpBlockSize;
        int64_t blkTableOffset = realS2Idx % mqVecConstInfo.paCmpBlockSize;
        int64_t blockId =
            blkTableIdx < static_cast<int64_t>(blockTableCacheSize) ?
                blockTableCache.GetValue(static_cast<uint32_t>(blkTableIdx)) :
                cmpBlockTableGm_.GetValue(mqVecRunInfo.bIdx * mqVecConstInfo.cmpMaxBlockNumPerBatch + blkTableIdx);
        if (mqVecConstInfo.kvQuantMode == 3) {
            realKeyGmOffset = blockId * static_cast<int64_t>(mqVecConstInfo.cmpKvStride0) +
                              blkTableOffset * static_cast<int64_t>(TQ4_SLOT_BYTES);
        } else {
            realKeyGmOffset = blockId * static_cast<int64_t>(mqVecConstInfo.cmpKvStride0) +
                              blkTableOffset * static_cast<int64_t>(mqVecConstInfo.kvHeadNum) *
                                  static_cast<int64_t>(mqVecConstInfo.headDim);
        }

    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::BSND) {
        realKeyGmOffset =
            mqVecRunInfo.bIdx * mqVecConstInfo.kvSeqSize / mqVecConstInfo.cmpRatio * mqVecConstInfo.kvHeadNum +
            realS2Idx * mqVecConstInfo.kvHeadNum;
    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        realKeyGmOffset =
            (mqVecRunInfo.tensorCmpBOffset + realS2Idx * mqVecConstInfo.kvHeadNum * mqVecConstInfo.headDim) /
            mqVecConstInfo.headDim;
    }
    return realKeyGmOffset;
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::DequantTq4Rows(LocalTensor<uint8_t> slots,
                                                                                 LocalTensor<KV_T> output,
                                                                                 int64_t rowCount)
{
    LocalTensor<float> work =
        TQ4_FAST_BF16 ? outputBuff1.Get<float>() : tmpBuff1.Get<float>()[TQ4_PACKED_BATCH_BYTES / sizeof(float)];
    LocalTensor<half> signedCodes = tmpBuff1.Get<half>()[TQ4_SIGNED_OFFSET / sizeof(half)];
    LocalTensor<int32_t> indices =
        TQ4_FAST_BF16 ? outputBuff1.Get<int32_t>() : tmpBuff1.Get<int32_t>()[TQ4_INDEX_OFFSET / sizeof(int32_t)];
    LocalTensor<uint32_t> unsignedIndices = indices.template ReinterpretCast<uint32_t>();
    LocalTensor<float> centroids = tqCentBuff.Get<float>();
    LocalTensor<int4b_t> packedCodes = slots.template ReinterpretCast<int4b_t>();
    LocalTensor<half> slotHalf = slots.template ReinterpretCast<half>();
    LocalTensor<half> scaleCarriers = signedCodes;
    LocalTensor<half> scalesFp16 = signedCodes[TQ4_SCALE_CARRIER_BYTES / sizeof(half)];
    LocalTensor<float> scalesFp32 = TQ4_FAST_BF16 ? tmpBuff1.Get<float>() : indices.template ReinterpretCast<float>();
    LocalTensor<float> scalesBrcb = scalesFp32[TQ4_SCALE_BROADCAST_OFFSET / sizeof(float)];
    LocalTensor<uint32_t> scaleGatherIndices =
        tqCentBuff.Get<uint32_t>()[TQ4_SCALE_GATHER_INDEX_OFFSET / sizeof(uint32_t)];

    for (int64_t base = 0; base < rowCount; base += TQ4_DEQUANT_BATCH_ROWS) {
        uint32_t currentRows =
            static_cast<uint32_t>(base + TQ4_DEQUANT_BATCH_ROWS <= rowCount ? TQ4_DEQUANT_BATCH_ROWS : rowCount - base);
        uint32_t elementCount = currentRows * TQ4_HEAD_DIM;
        if constexpr (TQ4_FAST_BF16) {
            LocalTensor<uint8_t> compactCodes = tmpBuff1.Get<uint8_t>();
            DataCopyParams compactParams;
            compactParams.blockCount = static_cast<uint16_t>(currentRows);
            compactParams.blockLen = static_cast<uint16_t>(TQ4_PACKED_BYTES / BYTE_BLOCK);
            compactParams.srcStride = static_cast<uint16_t>((TQ4_SLOT_ROW_BYTES - TQ4_PACKED_BYTES) / BYTE_BLOCK);
            compactParams.dstStride = 0;
            DataCopy(compactCodes, slots[base * TQ4_SLOT_ROW_BYTES], compactParams);
            PipeBarrier<PIPE_V>();
            uint32_t packedElementCount = currentRows * TQ4_PACKED_BYTES;
            Cast(signedCodes, compactCodes, RoundMode::CAST_NONE, packedElementCount);
            PipeBarrier<PIPE_V>();
            Cast(indices, signedCodes, RoundMode::CAST_ROUND, packedElementCount);
            PipeBarrier<PIPE_V>();
            ShiftLeft(indices, indices, static_cast<int32_t>(2), packedElementCount);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint32_t> outputPairs = output.template ReinterpretCast<uint32_t>();
            // Gather consumes the indices before outputBuff1 is reused below as the FP32 scale workspace.
            Gather(outputPairs[base * TQ4_PACKED_BYTES], tqByteLutBuff.Get<uint32_t>(), unsignedIndices, 0,
                   packedElementCount);
            PipeBarrier<PIPE_V>();
            Cast(work, output[base * TQ4_HEAD_DIM], RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t row = 0; row < currentRows; ++row) {
                uint32_t sourceRow = static_cast<uint32_t>(base) + row;
                Cast(signedCodes[row * TQ4_HEAD_DIM], packedCodes[sourceRow * TQ4_SLOT_ROW_BYTES * 2U],
                     RoundMode::CAST_NONE, TQ4_HEAD_DIM);
            }
            PipeBarrier<PIPE_V>();
            Adds(signedCodes, signedCodes, static_cast<half>(8.0f), elementCount);
            PipeBarrier<PIPE_V>();
            Muls(signedCodes, signedCodes, static_cast<half>(TQ4_CENTROID_BYTE_STRIDE), elementCount);
            PipeBarrier<PIPE_V>();
            Cast(indices, signedCodes, RoundMode::CAST_ROUND, elementCount);
            PipeBarrier<PIPE_V>();
            Gather(work, centroids, unsignedIndices, 0, elementCount);
            PipeBarrier<PIPE_V>();
        }

        // Extract one aligned carrier per staged row, then broadcast its leading FP16 scale.
        DataCopyParams scaleCopyParams;
        scaleCopyParams.blockCount = static_cast<uint16_t>(currentRows);
        scaleCopyParams.blockLen = 1;
        scaleCopyParams.srcStride = static_cast<uint16_t>(TQ4_SLOT_ROW_BYTES / BYTE_BLOCK - 1U);
        scaleCopyParams.dstStride = 0;
        uint32_t scaleSourceOffset =
            (static_cast<uint32_t>(base) * TQ4_SLOT_ROW_BYTES + TQ4_PACKED_BYTES) / sizeof(half);
        DataCopy(scaleCarriers, slotHalf[scaleSourceOffset], scaleCopyParams);
        PipeBarrier<PIPE_V>();
        Gather(scalesFp16, scaleCarriers, scaleGatherIndices, 0, currentRows);
        PipeBarrier<PIPE_V>();
        Cast(scalesFp32, scalesFp16, RoundMode::CAST_NONE, currentRows);
        PipeBarrier<PIPE_V>();
        uint8_t scaleBrcbRepeats =
            static_cast<uint8_t>((currentRows + FP32_BLOCK_ELEMENT_NUM - 1U) / FP32_BLOCK_ELEMENT_NUM);
        Brcb(scalesBrcb, scalesFp32, scaleBrcbRepeats, {1, FP32_BLOCK_ELEMENT_NUM});
        PipeBarrier<PIPE_V>();
        RowMuls(work, work, scalesBrcb, currentRows, TQ4_HEAD_DIM, TQ4_HEAD_DIM);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<KV_T, bfloat16_t>::value) {
            Cast(output[base * TQ4_HEAD_DIM], work, RoundMode::CAST_RINT, elementCount);
        } else {
            Cast(output[base * TQ4_HEAD_DIM], work, RoundMode::CAST_ROUND, elementCount);
        }
        // The next batch reuses work; the final batch is synchronized by CopyOutMrgeResult's V_MTE3 event.
        if (base + static_cast<int64_t>(currentRows) < rowCount) {
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::CopyInSingleKv(
    int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx, int64_t keyBNBOffset,
    int64_t s2IdLimit, const RunInfo& mqVecRunInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    if (mqVecConstInfo.kvQuantMode == 3) {
        int64_t stagedRow = mte2Size - mte3Size;
        if (stagedRow == 0) {
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        LocalTensor<uint8_t> slot = tmpBuff1.Get<uint8_t>()[TQ4_RAW_OFFSET + mergeMte3Idx % 2 * TQ4_RAW_REGION_BYTES +
                                                            stagedRow * TQ4_SLOT_ROW_BYTES];
        DataCopyExtParams copyParams{1, TQ4_SLOT_BYTES, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{true, 0, static_cast<uint8_t>(TQ4_SLOT_ROW_BYTES - TQ4_SLOT_BYTES), 0};
        DataCopyPad(slot, cmpTqGm_[keyBNBOffset], copyParams, padParams);
        ++mte2Size;
        return;
    }
    int64_t validS2Count = (realS2Idx + mqVecConstInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx :
                                                                                     mqVecConstInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * mqVecConstInfo.headDim * sizeof(KV_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_BSND || KV_LAYOUT_T == SAS_LAYOUT::PA_BNSD) {
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                              (mte2Size - mte3Size) * mqVecConstInfo.headDim],
                    cmpKvGm_[keyBNBOffset], intriParams, padParams);
    } else {
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                              (mte2Size - mte3Size) * mqVecConstInfo.headDim],
                    cmpKvGm_[keyBNBOffset * mqVecConstInfo.headDim], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::CopyInKv(
    int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1, int64_t realS2Idx2,
    const RunInfo& mqVecRunInfo, const LocalTensor<int32_t>& blockTableCache, uint32_t blockTableCacheSize)
{
    int64_t s2IdLimit = mqVecRunInfo.cmpS2IdLimit;

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, mqVecRunInfo, s2IdLimit, blockTableCache, blockTableCacheSize);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, mqVecRunInfo, s2IdLimit, blockTableCache, blockTableCacheSize);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }
    if (mqVecConstInfo.kvQuantMode == 3) {
        int64_t validHitCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        int64_t keySrcStride = keyOffset2 - keyOffset1 - static_cast<int64_t>(TQ4_SLOT_BYTES);
        // A two-burst copy is valid only when increasing GM order matches logical sparse-index order.
        // Descending, duplicate, tail, and stride-limit cases retain the ordered single-slot path.
        if (validHitCount == 2 && keySrcStride >= 0 && keySrcStride < INT32_MAX) {
            int64_t stagedRow = mte2Size - mte3Size;
            if (stagedRow == 0) {
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
            }
            LocalTensor<uint8_t> slot =
                tmpBuff1.Get<uint8_t>()[TQ4_RAW_OFFSET + mergeMte3Idx % 2 * TQ4_RAW_REGION_BYTES +
                                        stagedRow * TQ4_SLOT_ROW_BYTES];
            DataCopyExtParams copyParams{static_cast<uint16_t>(validHitCount), TQ4_SLOT_BYTES,
                                         static_cast<uint32_t>(keySrcStride), 0, 0};
            DataCopyPadExtParams<uint8_t> padParams{true, 0, static_cast<uint8_t>(TQ4_SLOT_ROW_BYTES - TQ4_SLOT_BYTES),
                                                    0};
            DataCopyPad(slot, cmpTqGm_[keyOffset1], copyParams, padParams);
            mte2Size += validHitCount;
        } else {
            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, mqVecRunInfo);
            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, mqVecRunInfo);
        }
        return;
    }

    int64_t keySrcStride = 0;
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_BSND || KV_LAYOUT_T == SAS_LAYOUT::PA_BNSD) {
        int64_t blkTableSrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                                     mqVecConstInfo.sparseBlockSize * mqVecConstInfo.headDim);
        keySrcStride = blkTableSrcStride * sizeof(KV_T);
    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::BSND) {
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                        mqVecConstInfo.sparseBlockSize) *
                       mqVecConstInfo.headDim * sizeof(KV_T);
    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                        mqVecConstInfo.sparseBlockSize) *
                       mqVecConstInfo.headDim * sizeof(KV_T);
    }
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
                 realS2Idx1 + mqVecConstInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + mqVecConstInfo.sparseBlockSize >= s2IdLimit)) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        // 因为需要拷贝两块
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, mqVecRunInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, mqVecRunInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = mqVecConstInfo.sparseBlockSize * mqVecConstInfo.headDim * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_BSND || KV_LAYOUT_T == SAS_LAYOUT::PA_BNSD) {
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                                  (mte2Size - mte3Size) * mqVecConstInfo.headDim],
                        cmpKvGm_[startGmOffset], intriParams, padParams);
        } else {
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                                  (mte2Size - mte3Size) * mqVecConstInfo.headDim],
                        cmpKvGm_[startGmOffset * mqVecConstInfo.headDim], intriParams, padParams);
        }
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * mqVecConstInfo.sparseBlockSize;
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                                    int64_t s2GmStartOffset,
                                                                                    int64_t mergeMte3Idx,
                                                                                    const RunInfo& mqVecRunInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    if (mqVecConstInfo.kvQuantMode == 3) {
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        LocalTensor<uint8_t> slots = tmpBuff1.Get<uint8_t>()[TQ4_RAW_OFFSET + mergeMte3Idx % 2 * TQ4_RAW_REGION_BYTES];
        LocalTensor<KV_T> output = kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T)];
        if constexpr (TQ4_FAST_BF16) {
            // Vec2 may still be copying its result from the shared outputBuff1 workspace.
            WaitFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        }
        DequantTq4Rows(slots, output, mte2Size - mte3Size);
        if constexpr (TQ4_FAST_BF16) {
            SetFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
    } else {
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = mqVecConstInfo.headDim * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(kvMergeGm_[mqVecRunInfo.cmpLoop % MERGE_CACHE_GM_BUF_NUM * 512 * 512 +
                           (s2GmStartOffset + mte3Size) * mqVecConstInfo.headDim],
                kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T)], dataCopyParams);
}

// b s1 k
template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::InitInt8Outputs()
{
    // Preserve full output/LSE initialization for padding and empty queries.
    // inputBuff2 is allocated as two16KiB buffers; no extra UB is required.
    constexpr uint32_t ZERO_BYTES = ConstInfo::MQ_BUFFER_BYTES_16K * 2U;
    constexpr uint32_t OUTPUT_CHUNK_ELEMENTS = ZERO_BYTES / sizeof(OUT_T);
    constexpr uint32_t LSE_CHUNK_ELEMENTS = ZERO_BYTES / sizeof(float);
    static_assert(ZERO_BYTES == ConstInfo::MQ_BUFFER_BYTES_32K);
    auto zeros = inputBuff2.Get<OUT_T>();
    Duplicate(zeros, static_cast<OUT_T>(0), OUTPUT_CHUNK_ELEMENTS);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    uint64_t rows = int8Input.params.qTokens * int8Input.params.heads;
    uint64_t outputElements = rows * 512U;
    uint64_t aivCount = static_cast<uint64_t>(GetBlockNum()) * 2U;
    uint64_t outputStep = aivCount * OUTPUT_CHUNK_ELEMENTS;
    for (uint64_t offset = static_cast<uint64_t>(GetBlockIdx()) * OUTPUT_CHUNK_ELEMENTS; offset < outputElements;
         offset += outputStep) {
        uint32_t count =
            static_cast<uint32_t>(Min(static_cast<uint64_t>(OUTPUT_CHUNK_ELEMENTS), outputElements - offset));
        DataCopyExtParams copy{1U, count * static_cast<uint32_t>(sizeof(OUT_T)), 0U, 0U, 0U};
        DataCopyPad(attentionOutGm[offset], zeros, copy);
    }
    if (int8Input.params.returnLse) {
        // FP16/BF16 positive-zero bits are also FP32 positive zero. The buffer
        // remains immutable until every MTE3 read has completed below.
        auto lseZeros = zeros.template ReinterpretCast<float>();
        uint64_t lseStep = aivCount * LSE_CHUNK_ELEMENTS;
        for (uint64_t offset = static_cast<uint64_t>(GetBlockIdx()) * LSE_CHUNK_ELEMENTS; offset < rows;
             offset += lseStep) {
            uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(LSE_CHUNK_ELEMENTS), rows - offset));
            DataCopyExtParams copy{1U, count * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(softmaxLseGm[offset], lseZeros, copy);
        }
    }
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::PrepareInt8Selection(Int8CsaInput<SAST>& selection,
                                                                                       uint64_t planOffset,
                                                                                       uint32_t& oriRows,
                                                                                       uint32_t& cmpRows)
{
    // Reuse the arch22 input-buffer events and its allocated page/index buffers.
    static_assert((1024U + 128U + 8192U) * sizeof(int32_t) <= ConstInfo::MQ_BUFFER_BYTES_32K * 2U);
    WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    // The scheduler has already computed this query's dynamic range. Resolve
    // it once using that same state; the Vector input object only owns KV handles.
    selection.CacheQueryAddresses(inputBuff1.Get<int32_t>(), v0ValidSizeBuff.Get<uint32_t>(), oriRows, cmpRows);
    SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    preparedInt8 = true;
    preparedOriRows = oriRows;

    // Private two-tile/query experiment only. Vec2 accumulation slot1 is never
    // read or written by computation: ori stores slot0, and final cmp reads slot0.
    // Two separate 64B plan slots occupy the first128B of the unused slot1.
    // Keep each AIV's original 32B payload and all existing synchronization.
    auto counts = v0ValidSizeBuff.Get<float>()[ConstInfo::MQ_BUFFER_BYTES_8K / sizeof(float) - 8U];
    counts.SetValue(0, static_cast<float>(static_cast<int32_t>(oriRows)));
    counts.SetValue(1, static_cast<float>(static_cast<int32_t>(cmpRows)));
    for (uint32_t i = 2; i < 8U; ++i) {
        counts.SetValue(i, 0.0f);
    }
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopy(vec2ResGm[planOffset + GetSubBlockIdx() * 8U], counts, 8U);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::PrepareInt8SelectionGeneric(
    Int8CsaInput<SAST>& selection, uint64_t planOffset, uint32_t& oriRows, uint32_t& cmpRows)
{
    static_assert((1024U + 128U + 8192U) * sizeof(int32_t) <= ConstInfo::MQ_BUFFER_BYTES_32K * 2U);
    WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    selection.CacheQueryAddresses(inputBuff1.Get<int32_t>(), v0ValidSizeBuff.Get<uint32_t>(), oriRows, cmpRows);
    SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    preparedInt8 = true;

    // Only head0's AIV may reuse the MM1 head cacheline. The other AIV still
    // completes its address preparation and participates in plan-ready flag10.
    if (GetSubBlockIdx() == 0U) {
        auto counts = v0ValidSizeBuff.Get<uint32_t>()[ConstInfo::MQ_BUFFER_BYTES_8K / sizeof(uint32_t) - 8U];
        counts.SetValue(0, oriRows);
        counts.SetValue(1, cmpRows);
        for (uint32_t i = 2U; i < 8U; ++i) {
            counts.SetValue(i, 0U);
        }
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        GlobalTensor<uint32_t> plans;
        plans.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(mm1ResGm.GetPhyAddr(0)),
                              mqVecConstInfo.preLoadNum * mqVecConstInfo.mmResUbSize);
        DataCopy(plans[planOffset], counts, 8U);
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ConvertInt8Batch(uint32_t slot, uint32_t staged,
                                                                                   uint64_t outOffset, bool wrapOutput)
{
    constexpr uint32_t DIM = Int8CsaInput<SAST>::HEAD_DIM;
    constexpr uint32_t BATCH_ROWS = 16U;
    constexpr uint32_t RECORD_BYTES = Int8CsaInput<SAST>::RECORD_BYTES;
    constexpr uint32_t NOPE_DIM = Int8CsaInput<SAST>::NOPE_DIM;
    constexpr uint32_t RAW_BYTES = BATCH_ROWS * RECORD_BYTES + BYTE_BLOCK;
    constexpr uint32_t HALF_BYTES = BATCH_ROWS * DIM * sizeof(half);
    constexpr uint32_t SCALE_COUNT = 8U;
    static_assert(RAW_BYTES + HALF_BYTES + BATCH_ROWS * SCALE_COUNT * sizeof(float) <= INPUT1_BUFFER_OFFSET);
    static_assert(BATCH_ROWS * DIM * sizeof(float) <= ConstInfo::MQ_BUFFER_BYTES_32K);
    static_assert(BATCH_ROWS * DIM * sizeof(KV_T) <= INPUT2_BUFFER_OFFSET);
    auto bytes = inputBuff1.Get<int8_t>()[slot * INPUT1_BUFFER_OFFSET];
    auto raw = bytes;
    auto halfValues = bytes.template ReinterpretCast<half>()[RAW_BYTES / sizeof(half)];
    auto scales = bytes.template ReinterpretCast<float>()[(RAW_BYTES + HALF_BYTES) / sizeof(float)];
    auto values = tmpBuff1.Get<float>();
    auto output = inputBuff2.Get<KV_T>()[slot * INPUT2_BUFFER_OFFSET / sizeof(KV_T)];
    WaitFlag<HardEvent::MTE2_V>(slot);
    // A5 processes the same dimension group across multiple records.
    // Use arch22 strided repeats instead of its register-based VF instructions.
    // The final 64 code lanes are ignored and replaced by the BF16 rope values.
    Duplicate(raw.template ReinterpretCast<float>()[staged * RECORD_BYTES / sizeof(float)], 0.0f, 8U);
    PipeBarrier<PIPE_V>();
    UnaryRepeatParams codeParams;
    codeParams.dstBlkStride = 1;
    codeParams.srcBlkStride = 1;
    codeParams.dstRepStride = static_cast<uint8_t>(DIM * sizeof(half) / BYTE_BLOCK);
    codeParams.srcRepStride = static_cast<uint8_t>(RECORD_BYTES / BYTE_BLOCK);
    for (uint32_t part = 0; part < 4U; ++part) {
        Cast(halfValues[part * 128U], raw[128U + part * 128U], RoundMode::CAST_NONE, 128U, static_cast<uint8_t>(staged),
             codeParams);
    }
    PipeBarrier<PIPE_V>();
    Cast(values, halfValues, RoundMode::CAST_NONE, staged * DIM);
    PipeBarrier<PIPE_V>();
    UnaryRepeatParams ropeParams;
    ropeParams.dstBlkStride = 1;
    ropeParams.srcBlkStride = 1;
    ropeParams.dstRepStride = static_cast<uint8_t>(DIM * sizeof(float) / BYTE_BLOCK);
    ropeParams.srcRepStride = static_cast<uint8_t>(RECORD_BYTES / BYTE_BLOCK);
    Cast(values[NOPE_DIM], raw.template ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, 64U,
         static_cast<uint8_t>(staged), ropeParams);
    UnaryRepeatParams scaleCastParams = ropeParams;
    scaleCastParams.dstRepStride = static_cast<uint8_t>(SCALE_COUNT * sizeof(float) / BYTE_BLOCK);
    Cast(scales, raw.template ReinterpretCast<bfloat16_t>()[288], RoundMode::CAST_NONE, SCALE_COUNT,
         static_cast<uint8_t>(staged), scaleCastParams);
    PipeBarrier<PIPE_V>();
    // All record reads are complete. Reuse the raw region for the same
    // block broadcast and Mul stride pattern used by RowMuls below.
    static_assert(BATCH_ROWS * SCALE_COUNT * FP32_BLOCK_ELEMENT_NUM * sizeof(float) <= RAW_BYTES);
    auto scaleBrcb = raw.template ReinterpretCast<float>();
    Brcb(scaleBrcb, scales, static_cast<uint8_t>(staged * SCALE_COUNT / FP32_BLOCK_ELEMENT_NUM),
         {1, FP32_BLOCK_ELEMENT_NUM});
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams scaleParams;
    scaleParams.dstBlkStride = 1;
    scaleParams.src0BlkStride = 1;
    scaleParams.src1BlkStride = 0;
    scaleParams.dstRepStride = FP32_REPEAT_ELEMENT_NUM / FP32_BLOCK_ELEMENT_NUM;
    scaleParams.src0RepStride = FP32_REPEAT_ELEMENT_NUM / FP32_BLOCK_ELEMENT_NUM;
    scaleParams.src1RepStride = 1;
    scaleParams.dstRepStride = DIM * sizeof(float) / BYTE_BLOCK;
    scaleParams.src0RepStride = DIM * sizeof(float) / BYTE_BLOCK;
    scaleParams.src1RepStride = SCALE_COUNT;
    for (uint32_t group = 0; group < 7U; ++group) {
        Mul(values[group * 64U], values[group * 64U], scaleBrcb[group * FP32_BLOCK_ELEMENT_NUM], 64U,
            static_cast<uint8_t>(staged), scaleParams);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(slot + SYNC_INPUT_BUF1_FLAG);
    WaitFlag<HardEvent::MTE3_V>(slot);
    if constexpr (IsSameType<KV_T, bfloat16_t>::value) {
        Cast(output, values, RoundMode::CAST_RINT, staged * DIM);
    } else {
        Cast(output, values, RoundMode::CAST_ROUND, staged * DIM);
    }
    SetFlag<HardEvent::V_MTE3>(slot);
    WaitFlag<HardEvent::V_MTE3>(slot);
    if (!wrapOutput) {
        DataCopy(kvMergeGm_[outOffset], output, staged * DIM);
    } else {
        const uint32_t row = static_cast<uint32_t>(outOffset / DIM) & 511U;
        const uint32_t remaining = 512U - row;
        const uint32_t firstRows = staged > remaining ? remaining : staged;
        DataCopy(kvMergeGm_[static_cast<uint64_t>(row) * DIM], output, firstRows * DIM);
        if (firstRows != staged) {
            DataCopy(kvMergeGm_[0], output[firstRows * DIM], (staged - firstRows) * DIM);
        }
    }
    SetFlag<HardEvent::MTE3_V>(slot);
}

template <typename SAST>
template <bool INT8_FAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ProcessInt8Kv(const RunInfo& info,
                                                                                uint32_t stageRowStart, bool wrapOutput)
{
    constexpr uint32_t DIM = Int8CsaInput<SAST>::HEAD_DIM;
    constexpr uint32_t BATCH_ROWS = 16U;
    constexpr uint32_t RECORD_BYTES = Int8CsaInput<SAST>::RECORD_BYTES;
    // A2/A3: unit2KiB; len6 is below the AIV16KiB cache limit.
    // Start the hint before existing buffer-credit waits; no polling or new barrier.
    AscendC::ICachePreLoad(6);
    WaitFlag<HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);
    // Private cmp512 path: two256-row waves, each AIV produces128 rows.
    const bool wave256 = INT8_FAST && !info.isOriOnly && info.actualSingleProcessSInnerSize == 512U;
    uint32_t halfRows = wave256 ? 128U : (info.actualSingleProcessSInnerSize + 1U) / 2U;
    uint32_t first = GetSubBlockIdx() == 0 ? 0 : halfRows;
    uint32_t end = wave256 ? first + halfRows : (GetSubBlockIdx() == 0 ? halfRows : info.actualSingleProcessSInnerSize);
    uint32_t wave = 0U;
    bool pendingWaveReady = false;
    uint32_t row = preparedInt8 ? first : 0U;
    uint32_t staged = 0U, loadSlot = 0U, pendingSlot = 0U, pendingRows = 0U;
    uint64_t outOffset = static_cast<uint64_t>(stageRowStart + first) * DIM;
    uint64_t pendingOffset = outOffset;
    int64_t candidateStart = info.int8Start + (preparedInt8 ? first : 0U);
    for (int64_t candidate = candidateStart; candidate < info.int8End && row < end; ++candidate) {
        uint32_t physicalRow = preparedInt8 ? v0ValidSizeBuff.Get<uint32_t>().GetValue(
                                                  static_cast<uint32_t>((info.isOriOnly ? 0U : 512U) + candidate)) :
                                              0U;
        int64_t offset = preparedInt8 ? static_cast<int64_t>(physicalRow) * RECORD_BYTES :
                                        int8Input.CandidateOffset(info, candidate);
        if (offset < 0) {
            continue;
        }
        if (row++ < first) {
            continue;
        }
        if (staged == 0U) {
            WaitFlag<HardEvent::V_MTE2>(loadSlot + SYNC_INPUT_BUF1_FLAG);
        }
        auto raw = inputBuff1.Get<int8_t>()[loadSlot * INPUT1_BUFFER_OFFSET + staged * RECORD_BYTES];
        // Prepared original KV stays in logical token order. Coalesce complete
        // pairs within one physical page; boundary pairs keep the existing order.
        uint32_t oriRunRows = 0U;
        if (preparedInt8 && info.isOriOnly) {
            uint32_t blockSize = int8Input.params.oriBlockSize;
            uint32_t pageRows = blockSize - (physicalRow & (blockSize - 1U));
            oriRunRows = Min(BATCH_ROWS - staged, end - row + 1U);
            oriRunRows = Min(oriRunRows, static_cast<uint32_t>(info.int8End - candidate));
            oriRunRows = Min(oriRunRows, pageRows) & ~1U;
        }
        if (oriRunRows != 0U) {
            DataCopyExtParams copy{1U, oriRunRows * RECORD_BYTES, 0U, 0U, 0U};
            DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
            DataCopyPad(raw, int8Input.oriGm[offset], copy, pad);
            candidate += oriRunRows - 1U;
            row += oriRunRows - 1U;
            staged += oriRunRows;
        } else if (preparedInt8 && !info.isOriOnly && candidate + 1 < info.int8End && row < end &&
                   staged + 1U < BATCH_ROWS) {
            // ori cache rows must retain logical order, including descending physical-page transitions.
            // Keep arch22 SMLA's paired transfer for compressed KV and original page boundaries.
            int64_t nextOffset = static_cast<int64_t>(v0ValidSizeBuff.Get<uint32_t>().GetValue(
                                     static_cast<uint32_t>((info.isOriOnly ? 0U : 512U) + candidate + 1))) *
                                 RECORD_BYTES;
            int64_t gap = (offset > nextOffset ? offset - nextOffset : nextOffset - offset) - RECORD_BYTES;
            if (gap >= 0 && gap < 0x7FFFFFFFLL) {
                DataCopyExtParams copy;
                copy.blockCount = 2U;
                copy.blockLen = RECORD_BYTES;
                copy.srcStride = static_cast<uint32_t>(gap);
                copy.dstStride = 0U;
                DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
                int64_t begin = offset < nextOffset ? offset : nextOffset;
                if (info.isOriOnly) {
                    DataCopyPad(raw, int8Input.oriGm[begin], copy, pad);
                } else {
                    DataCopyPad(raw, int8Input.cmpGm[begin], copy, pad);
                }
            } else {
                if (info.isOriOnly) {
                    DataCopy(raw, int8Input.oriGm[offset], RECORD_BYTES);
                    DataCopy(raw[RECORD_BYTES], int8Input.oriGm[nextOffset], RECORD_BYTES);
                } else {
                    DataCopy(raw, int8Input.cmpGm[offset], RECORD_BYTES);
                    DataCopy(raw[RECORD_BYTES], int8Input.cmpGm[nextOffset], RECORD_BYTES);
                }
            }
            ++candidate;
            ++row;
            staged += 2U;
        } else {
            if (info.isOriOnly) {
                DataCopy(raw, int8Input.oriGm[offset], RECORD_BYTES);
            } else {
                DataCopy(raw, int8Input.cmpGm[offset], RECORD_BYTES);
            }
            ++staged;
        }
        if (staged != BATCH_ROWS && row != end) {
            continue;
        }
        SetFlag<HardEvent::MTE2_V>(loadSlot);
        if (pendingRows != 0U) {
            ConvertInt8Batch(pendingSlot, pendingRows, pendingOffset, wrapOutput);
            if (pendingWaveReady) {
                // Ordered after the final16-row MTE3 write of this wave.
                CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(6);
            }
        }
        pendingWaveReady = wave256 && row == end;
        pendingSlot = loadSlot;
        pendingRows = staged;
        pendingOffset = outOffset;
        outOffset += staged * DIM;
        staged = 0U;
        loadSlot ^= 1U;
        if (wave256 && wave == 0U && row == end) {
            // Retain one-batch-ahead16-row prefetch across the wave boundary.
            wave = 1U;
            first += 256U;
            row = first;
            end = first + 128U;
            candidate = info.int8Start + first - 1;
            outOffset = static_cast<uint64_t>(stageRowStart + first) * DIM;
        }
    }
    if (pendingRows != 0U) {
        ConvertInt8Batch(pendingSlot, pendingRows, pendingOffset, wrapOutput);
        if (pendingWaveReady) {
            CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(6);
        }
    }
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
    if (GetSubBlockIdx() == 1 && info.actualSingleProcessSInnerSize < info.actualSingleProcessSInnerSizeAlign) {
        auto output = inputBuff2.Get<KV_T>();
        Duplicate(output, static_cast<KV_T>(0), 16U * DIM);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        uint64_t base = (info.cmpLoop % MERGE_CACHE_GM_BUF_NUM) * mqVecConstInfo.s2BaseSize * DIM;
        for (uint32_t start = info.actualSingleProcessSInnerSize; start < info.actualSingleProcessSInnerSizeAlign;
             start += 16U) {
            uint32_t count = Min(16U, info.actualSingleProcessSInnerSizeAlign - start);
            DataCopy(kvMergeGm_[base + start * DIM], output, count * DIM);
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
    SetFlag<HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
}

template <typename SAST>
template <bool INT8_FAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ProcessVec0L(const RunInfo& mqVecRunInfo)
{
    if constexpr (SAST::int8Kv) {
        if constexpr (!INT8_FAST) {
            const uint32_t stage = (mqVecRunInfo.cmpLoop % MERGE_CACHE_GM_BUF_NUM) * mqVecConstInfo.s2BaseSize;
            ProcessInt8Kv<false>(mqVecRunInfo, stage, false);
            return;
        }
        // Keep queued ori128/online-softmax/two-tile semantics intact. Only
        // the local V0 decoder skips the previous logical window's suffix.
        RunInfo decodeInfo = mqVecRunInfo;
        uint32_t stageRowStart = 512U + ((mqVecRunInfo.loop / 2U) & 1U) * 512U;
        if (mqVecRunInfo.isOriOnly) {
            decodeInfo.int8Start += mqVecRunInfo.oriReuseRows;
            decodeInfo.actualSingleProcessSInnerSize -= mqVecRunInfo.oriReuseRows;
            stageRowStart = mqVecRunInfo.oriCacheWriteRow;
        }
        // Contract: full ori128 and cmp512. Padding a 1-row append would
        // destroy cached rows; leave all original RunInfo dimensions unchanged.
        decodeInfo.actualSingleProcessSInnerSizeAlign = decodeInfo.actualSingleProcessSInnerSize;
        ProcessInt8Kv<true>(decodeInfo, stageRowStart, mqVecRunInfo.isOriOnly);
        return;
    }

    int64_t s2ProcessSize = mqVecRunInfo.v0S2DealSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2 * mqVecConstInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = mqVecRunInfo.topKBaseOffset + mqVecRunInfo.v0S2Start;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    int64_t s2SplitPoint = SASAlign(s2Pair, 2) * mqVecConstInfo.sparseBlockSize;
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : s2SplitPoint;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? s2SplitPoint : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    LocalTensor<int32_t> topkCache = v0ValidSizeBuff.Get<int32_t>();
    LocalTensor<int32_t> blockTableCache =
        v0ValidSizeBuff.Get<int32_t>()[TQ4_BLOCK_TABLE_CACHE_OFFSET / sizeof(int32_t)];
    int64_t topkCacheStart = 0;
    uint32_t topkCacheSize = 0;
    uint32_t blockTableCacheSize = 0;
    bool hasIndexPrefetch = false;
    if (mqVecConstInfo.kvQuantMode == 3) {
        // Prefetch only a complete required range. A zero cache size keeps the original GM scalar fallback.
        topkCacheStart = s2GmStartOffset / mqVecConstInfo.sparseBlockSize;
        int64_t topkCacheEnd = CeilDiv(s2GmLimit, mqVecConstInfo.sparseBlockSize);
        if (topkCacheEnd > static_cast<int64_t>(mqVecConstInfo.sparseBlockCount)) {
            topkCacheEnd = mqVecConstInfo.sparseBlockCount;
        }
        int64_t requiredTopkCount = topkCacheEnd - topkCacheStart;
        if (requiredTopkCount > 0 && requiredTopkCount <= TQ4_TOPK_CACHE_CAPACITY) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(requiredTopkCount * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams;
            DataCopyPad(topkCache, topkGm_[topkGmBaseOffset + topkCacheStart], copyParams, padParams);
            topkCacheSize = static_cast<uint32_t>(requiredTopkCount);
            hasIndexPrefetch = true;
        }
        if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_BSND || KV_LAYOUT_T == SAS_LAYOUT::PA_BNSD) {
            int64_t requiredBlockTableCount =
                mqVecRunInfo.cmpS2IdLimit > 0 ?
                    CeilDiv(mqVecRunInfo.cmpS2IdLimit, static_cast<int64_t>(mqVecConstInfo.paCmpBlockSize)) :
                    0;
            if (requiredBlockTableCount > 0 &&
                requiredBlockTableCount <= static_cast<int64_t>(mqVecConstInfo.cmpMaxBlockNumPerBatch) &&
                requiredBlockTableCount <= TQ4_BLOCK_TABLE_CACHE_CAPACITY) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(requiredBlockTableCount * sizeof(int32_t)), 0, 0,
                                             0};
                DataCopyPadExtParams<int32_t> padParams;
                DataCopyPad(blockTableCache,
                            cmpBlockTableGm_[mqVecRunInfo.bIdx * mqVecConstInfo.cmpMaxBlockNumPerBatch], copyParams,
                            padParams);
                blockTableCacheSize = static_cast<uint32_t>(requiredBlockTableCount);
                hasIndexPrefetch = true;
            }
        }
        if (hasIndexPrefetch) {
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    // 处理两个基本块
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * mqVecConstInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, mqVecRunInfo, topkCache, topkCacheStart,
                     topkCacheSize);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, mqVecRunInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + mqVecConstInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, mqVecRunInfo,
                     topkCache, topkCacheStart, topkCacheSize);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, mqVecRunInfo, blockTableCache,
                 blockTableCacheSize);
        if ((mte2Size - mte3Size + 2 * mqVecConstInfo.sparseBlockSize > TQ4_MAX_STAGED_ROWS) ||
            s2GmOffsetArray + 2 * mqVecConstInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, mqVecRunInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }
    return;
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ProcessVec1L(const RunInfo& mqVecInfo)
{
    uint32_t nBufferLoopTimes =
        (mqVecInfo.actMBaseSize + mqVecConstInfo.nBufferMBaseSize - 1) / mqVecConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail = mqVecInfo.actMBaseSize - (nBufferLoopTimes - 1) * mqVecConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mqVecSplitInfo;
        mqVecSplitInfo.nBufferIdx = i;
        mqVecSplitInfo.nBufferStartM = i * mqVecConstInfo.nBufferMBaseSize;
        mqVecSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? mqVecConstInfo.nBufferMBaseSize : nBufferTail;

        mqVecSplitInfo.vecDealM = (mqVecSplitInfo.nBufferDealM <= 16) ?
                                      mqVecSplitInfo.nBufferDealM :
                                      (((mqVecSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mqVecSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mqVecSplitInfo.vecStartM = mqVecSplitInfo.vecDealM;
            mqVecSplitInfo.vecDealM = mqVecSplitInfo.nBufferDealM - mqVecSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(mqVecConstInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(mqVecInfo, mqVecSplitInfo);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(mqVecConstInfo.syncV1C2);

        // move lse for flash decode or FA
        if (mqVecConstInfo.returnSoftmaxLse && mqVecInfo.s2Idx == mqVecInfo.curSInnerLoopTimes - 1) {
            ProcessLse(mqVecInfo, mqVecSplitInfo);
        }
    }
}

template <typename SAST>
__aicore__ inline uint64_t KvQuantSparseFlashMlaCsaBlockVector<SAST>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ProcessVec2SingleBuf(const RunInfo& mqVecInfo,
                                                                                       const MSplitInfo& mqVecSplitInfo)
{
    if (mqVecSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(mqVecInfo, mqVecSplitInfo, 0, mqVecSplitInfo.vecDealM);
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ProcessVec2L(const RunInfo& mqVecInfo)
{
    uint32_t nBufferLoopTimes =
        (mqVecInfo.actMBaseSize + mqVecConstInfo.nBufferMBaseSize - 1) / mqVecConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail = mqVecInfo.actMBaseSize - (nBufferLoopTimes - 1) * mqVecConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mqVecSplitInfo;
        mqVecSplitInfo.nBufferIdx = i;
        mqVecSplitInfo.nBufferStartM = i * mqVecConstInfo.nBufferMBaseSize;
        mqVecSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? mqVecConstInfo.nBufferMBaseSize : nBufferTail;

        mqVecSplitInfo.vecDealM = (mqVecSplitInfo.nBufferDealM <= 16) ?
                                      mqVecSplitInfo.nBufferDealM :
                                      (((mqVecSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mqVecSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mqVecSplitInfo.vecStartM = mqVecSplitInfo.vecDealM;
            mqVecSplitInfo.vecDealM = mqVecSplitInfo.nBufferDealM - mqVecSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(mqVecConstInfo.syncC2V2);
        ProcessVec2SingleBuf(mqVecInfo, mqVecSplitInfo);
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::ProcessVec2Inner(const RunInfo& mqVecInfo,
                                                                                   const MSplitInfo& mqVecSplitInfo,
                                                                                   uint32_t mStartRow,
                                                                                   uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / mqVecConstInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(mqVecInfo, mqVecSplitInfo, i * mSplitSize + mStartRow, dealSize, mqVecConstInfo.headDim,
                             mqVecConstInfo.headDim);
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::Bmm2FDDataCopyOut(
    const RunInfo& mqVecInfo, LocalTensor<T>& mqBmm2ResultUb, uint32_t mqWorkspaceRow, uint32_t mqDealRows,
    uint32_t mqColumns, uint32_t mqActualColumns)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, mqBmm2ResultUb, mqColumns * mqDealRows);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t mqAccumOutputCount = CalcAccumOffset(mqVecInfo.bIdx, mqVecInfo.gS1Idx);
    uint64_t offset = mqAccumOutputCount * mqVecConstInfo.kvHeadNum * mqVecConstInfo.mBaseSize *
                          mqVecConstInfo.headDim + // taskoffset
                      mqVecInfo.tndCoreStartKVSplitPos * mqVecConstInfo.kvHeadNum * mqVecConstInfo.mBaseSize *
                          mqVecConstInfo.headDim +      // 份数offset
                      mqWorkspaceRow * mqActualColumns; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (mqVecInfo.actualSingleProcessSInnerSize == 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mqDealRows;
        dataCopyParams.blockLen = mqActualColumns * sizeof(T);
        dataCopyParams.srcStride = (mqColumns - mqActualColumns) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, mqDealRows * mqActualColumns, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::Bmm2DataCopyOutTrans(
    const RunInfo& mqVecInfo, LocalTensor<OUT_T>& attenOutUb, uint32_t mqWorkspaceRow, uint32_t mqDealRows,
    uint32_t mqColumns, uint32_t mqActualColumns)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mqDealRows;
    dataCopyParams.blockLen = mqActualColumns * sizeof(OUT_T);
    dataCopyParams.srcStride = (mqColumns - mqActualColumns) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[mqVecInfo.attenOutOffset + mqWorkspaceRow * mqActualColumns], attenOutUb,
                dataCopyParams);
    return;
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::Bmm2CastAndCopyOut(
    const RunInfo& mqVecInfo, LocalTensor<T>& mqBmm2ResultUb, uint32_t mqWorkspaceRow, uint32_t mqDealRows,
    uint32_t mqColumns, uint32_t mqActualColumns)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, mqBmm2ResultUb, AscendC::RoundMode::CAST_RINT, mqDealRows * mqColumns);
    } else {
        Cast(tmpBmm2ResCastTensor, mqBmm2ResultUb, AscendC::RoundMode::CAST_ROUND, mqDealRows * mqColumns);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(mqVecInfo, tmpBmm2ResCastTensor, mqWorkspaceRow, mqDealRows, mqColumns, mqActualColumns);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::Bmm2ResCopyOut(
    const RunInfo& mqVecInfo, LocalTensor<T>& mqBmm2ResultUb, uint32_t mqWorkspaceRow, uint32_t mqDealRows,
    uint32_t mqColumns, uint32_t mqActualColumns)
{
    if constexpr (FLASH_DECODE) {
        if (mqVecInfo.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(mqVecInfo, mqBmm2ResultUb, mqWorkspaceRow, mqDealRows, mqColumns, mqActualColumns);
        } else {
            Bmm2CastAndCopyOut(mqVecInfo, mqBmm2ResultUb, mqWorkspaceRow, mqDealRows, mqColumns, mqActualColumns);
        }
    } else {
        Bmm2CastAndCopyOut(mqVecInfo, mqBmm2ResultUb, mqWorkspaceRow, mqDealRows, mqColumns, mqActualColumns);
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::DealBmm2ResBaseBlock(
    const RunInfo& mqVecInfo, const MSplitInfo& mqVecSplitInfo, uint32_t startRow, uint32_t mqDealRows,
    uint32_t mqColumns, uint32_t mqActualColumns)
{
    uint32_t vec2ComputeSize = mqDealRows * mqColumns;
    uint32_t mStart = mqVecSplitInfo.nBufferStartM + mqVecSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset =
        (mqVecInfo.loop % mqVecConstInfo.preLoadNum) * mqVecConstInfo.bmm2ResUbSize + mStart * mqColumns;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    LocalTensor<T> mqBmm2ResultUb = tmpBuff1.Get<T>();
    mqBmm2ResultUb.SetSize(vec2ComputeSize);
    DataCopy(mqBmm2ResultUb, tmpBmm2ResUb, vec2ComputeSize);
    if constexpr (SAST::int8Kv) {
        // The first KV tile can immediately copy this UB result to outUb.
        PipeBarrier<PIPE_V>();
    }
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    pingpongFlag ^= 1; // pingpong 0 1切换

    uint32_t inOutBaseOffset = mStart * mqColumns;
    uint32_t baseOffset = mqVecSplitInfo.nBufferStartM / 2 + startRow;

    // 除第一个循环外，均需要更新中间计算结果
    if (!mqVecInfo.isFirstSInnerLoop) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);

        LocalTensor<MM2_OUT_T> bmm2ResPreUb = inputBuff1.Get<MM2_OUT_T>();
        bmm2ResPreUb = bmm2ResPreUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

        uint64_t vec2ResGmOffset =
            ((mqVecInfo.loop - 1) % mqVecConstInfo.preLoadNum) * mqVecConstInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);

        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

        uint32_t idx = mqVecInfo.loop % (mqVecConstInfo.preLoadNum);
        LocalTensor<T> expUb = v0ValidSizeBuff.Get<T>()[V0_VEC2_TEMP_OFFSET / sizeof(T)];
        Brcb(expUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (mqDealRows + 7) / 8,
             {1, 8});
        PipeBarrier<PIPE_V>();

        RowMuls(bmm2ResPreUb, bmm2ResPreUb, expUb, mqDealRows, mqColumns, mqActualColumns);
        AscendC::PipeBarrier<PIPE_V>();
        Add(mqBmm2ResultUb, mqBmm2ResultUb, bmm2ResPreUb, vec2ComputeSize);
        AscendC::PipeBarrier<PIPE_V>();

        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
        pingpongFlag ^= 1; // pingpong 0 1 切换
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (mqVecInfo.isLastS2Loop) {
        uint32_t idx = mqVecInfo.loop % (mqVecConstInfo.preLoadNum);
        LocalTensor<T> tmpSumUb = v0ValidSizeBuff.Get<T>()[V0_VEC2_TEMP_OFFSET / sizeof(T)];
        Brcb(tmpSumUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (mqDealRows + 7) / 8,
             {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(mqBmm2ResultUb, mqBmm2ResultUb, tmpSumUb, mqDealRows, mqColumns, mqActualColumns);
        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(mqVecInfo, mqBmm2ResultUb, mStart, mqDealRows, mqColumns, mqActualColumns);
    } else {
        LocalTensor<T> outUb = outputBuff1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(outUb, mqBmm2ResultUb, mqDealRows * mqColumns);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        uint64_t vec2ResGmOffset =
            (mqVecInfo.loop % mqVecConstInfo.preLoadNum) * mqVecConstInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset], outUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::RowDivs(LocalTensor<float> dstUb,
                                                                          LocalTensor<float> src0Ub,
                                                                          LocalTensor<float> src1Ub,
                                                                          uint32_t mqDealRows, uint32_t mqColumns,
                                                                          uint32_t mqActualColumns)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = mqActualColumns / dtypeMask;
    uint32_t dRemain = mqActualColumns % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = mqColumns / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = mqColumns / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= mqDealRows) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, mqDealRows, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < mqDealRows; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += mqColumns;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, mqDealRows, repeatParamsDiv);
    }
}

template <typename SAST>
__aicore__ inline void KvQuantSparseFlashMlaCsaBlockVector<SAST>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub,
                                                                          LocalTensor<T> src1Ub, uint32_t mqDealRows,
                                                                          uint32_t mqColumns, uint32_t mqActualColumns)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = mqActualColumns / repeatElementNum;
    uint32_t dRemain = mqActualColumns % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (mqColumns < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = mqColumns / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = mqColumns / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= mqDealRows) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, mqDealRows, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < mqDealRows; i++) {
                Mul(dstUb[i * mqColumns], src0Ub[i * mqColumns], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                    columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, mqDealRows,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < mqDealRows; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * mqColumns], src0Ub[i * mqColumns], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * mqColumns + dLoop * repeatElementNum], src0Ub[i * mqColumns + dLoop * repeatElementNum],
                    src1Ub[i * blockElementNum], dRemain, 1, repeatParams);
            }
        }
    }
}
} // namespace SASKernel
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_TQ_CSA_BLOCK_VECTOR_H
