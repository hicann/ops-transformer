/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_aiv_mode_util.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_UTIL
#define MATMUL_REDUCE_SCATTER_AIV_MODE_UTIL
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"

using namespace AscendC;
using namespace matmulReduceScatterV2_aivmode_tiling;

namespace matmulReduceScatterV2_util {
#define PADDING_ARGS_CALL() \
    transA, transB, alignedA, alignedB, matrixAM, matrixAK, matrixBK, matrixBN, matrixAMAlign, matrixAKAlign, \
        matrixBKAlign, matrixBNAlign, gmA, gmB, gmAAlign, gmBAlign, castBias, biasLength, gmBias, gmBiasCast

#define DEQUANT_ARGS_CALL() \
    rowNum, colNum, perChannelScale, perTokenScale, workspace, reinterpret_cast<GM_ADDR>(peerMem), \
        reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(biasptr), tileM0, tileN0, pValue, swizzlDirect, \
        swizzlCount, coreIdx, coreNum, rankIdx, rankSize, calIdx, resource, needPerChannel, needPerToken

#define DEQUANT_ARGS_FUN() \
    uint32_t rowNum, uint32_t colNum, __gm__ float32_t *perChannelScale, __gm__ float32_t *perTokenScale, \
        __gm__ int32_t *workspace, GM_ADDR peerMem, GM_ADDR output, GM_ADDR biasptr, uint32_t tileM0, uint32_t tileN0, \
        uint32_t pValue, uint32_t swizzlDirect, uint32_t swizzlCount, uint32_t coreIdx, uint32_t coreNum, \
        uint32_t rankIdx, uint32_t rankSize, uint32_t calIdx, Arch::Resource<Arch::AtlasA2> resource, \
        bool needPerChannel = false, bool needPerToken = false

constexpr int32_t MAX_BLOCK_COUNT = 2;
constexpr int32_t MAX_BLOCK_COUNT_SM = 4;
constexpr int32_t FLAG_ZERO_IDX = 0;
constexpr int32_t FLAG_ONE_IDX = 1;
constexpr int32_t FLAG_VALUE = 1;
constexpr int32_t FLAG_OFFSET = 180 * 1024 * 1024 / sizeof(int32_t);
constexpr int32_t USED_UB_SIZE = 160 * 1024;
constexpr int32_t AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID = 12;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t BLOCK_SIZE_16 = 16;
constexpr uint32_t BASE_BLOCK_SIZE_32 = 32;
constexpr uint32_t BASE_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASE_BLOCK_SIZE_512 = 512;
constexpr uint32_t TILE_SHAPE_64 = 64;
constexpr uint32_t TILE_SHAPE_128 = 128;
constexpr uint32_t TILE_SHAPE_256 = 256;
constexpr uint32_t TILE_SHAPE_512 = 512;
constexpr uint32_t UB_BUFFER_NUM = 2;
constexpr uint32_t AIV_CROSS_CORE_SYNC_MODE = 0;
constexpr uint32_t AIC_CROSS_CORE_SYNC_MODE = 2;
constexpr uint32_t RAND_BASE = 3;

template <typename T, size_t SIZE>
struct BaseBlock {
    static_assert((SIZE & (SIZE - 1)) == 0, "Invalid block size");
    static constexpr size_t size = SIZE / sizeof(T);

    static inline __aicore__ size_t Count(size_t len)
    {
        return (len + size - 1) / size;
    }

    static inline __aicore__ bool IsAligned(size_t len)
    {
        return len % size == 0;
    }

    static inline __aicore__ size_t AlignUp(size_t len)
    {
        return (len + size - 1) & ~(size - 1);
    }

    static inline __aicore__ size_t AlignDown(size_t len)
    {
        return len & ~(size - 1);
    }
};

template <typename T>
using Block32B = BaseBlock<T, BASE_BLOCK_SIZE_32>;

template <typename T>
using Block256B = BaseBlock<T, BASE_BLOCK_SIZE_256>;

template <typename T>
using Block512B = BaseBlock<T, BASE_BLOCK_SIZE_512>;

__aicore__ inline int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

__aicore__ inline void GetSwizzledBlockIdx(int32_t loop_idx, int32_t m_loop, int32_t n_loop, int32_t swizzl_direction,
                                           int32_t swizzl_count, int64_t &m_idx, int64_t &n_idx)
{
    uint32_t in_batch_idx = loop_idx % (m_loop * n_loop);
    if (swizzl_direction == 0) { // Zn
        uint32_t tile_block_loop = (m_loop + swizzl_count - 1) / swizzl_count;
        uint32_t tile_block_idx = in_batch_idx / (swizzl_count * n_loop);
        uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * n_loop);

        uint32_t n_row = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_row = m_loop - swizzl_count * tile_block_idx;
        }
        m_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_row;
        n_idx = in_tile_block_idx / n_row;
        if (tile_block_idx % UB_BUFFER_NUM != 0) {
            n_idx = n_loop - n_idx - 1;
        }
    } else if (swizzl_direction == 1) { // Nz
        uint32_t tile_block_loop = (n_loop + swizzl_count - 1) / swizzl_count;
        uint32_t tile_block_idx = in_batch_idx / (swizzl_count * m_loop);
        uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * m_loop);

        uint32_t n_col = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - swizzl_count * tile_block_idx;
        }
        m_idx = in_tile_block_idx / n_col;
        n_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_col;
        if (tile_block_idx % UB_BUFFER_NUM != 0) {
            m_idx = m_loop - m_idx - 1;
        }
    }
}

class CommBase {
public:
    __aicore__ explicit CommBase(){};

    __aicore__ inline void SetArgs(MatmulReduceScatterV2AivModeTilingData tilingData)
    {
        block_id = AscendC::GetBlockIdx();
        core_num = AscendC::GetBlockNum();
        aiv_idx = GetSubBlockIdx();
        core_idx = block_id / GetTaskRation();

        SetTiling(tilingData);
        if (is910C) {
            __gm__ HcclOpResParam *winContext_{nullptr};
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            winContext_ = (__gm__ HcclOpResParam *)contextGM0;
            rank = winContext_->localUsrRankId;
            rank_size = winContext_->rankSize;
            for (int i = 0; i < rank_size; i++) {
                buff[i] =
                    (GM_ADDR)((i == rank) ?
                                  winContext_->localWindowsIn :
                                  ((HcclRankRelationResV2 *)(winContext_->remoteRes[i].nextDevicePtr))->windowsIn);
            }
        } else {
            __gm__ HcclA2CombineOpParam *winContext_{nullptr};
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
            rank = winContext_->rankId;
            rank_size = winContext_->rankNum;
            for (int i = 0; i < rank_size; i++) {
                buff[i] = (GM_ADDR)winContext_->windowsIn[i];
            }
        }
        k_loop = CeilDev(k, k0);
        n_loop = CeilDev(n, n0);
        int32_t tail_m = (m / rank_size) % m0;
        m_loop = m / rank_size / m0;
        if (tail_m) {
            m_loop += 1;
        }
        m_loop *= rank_size;
        nAlign16 = n % BLOCK_SIZE_16 == 0;
        loop_num_per_comm = p_value * core_num;
        other_rank = (core_idx < rank_size) ? core_idx : -1;
        gm_c_pingpong_size = m0 * n0 * loop_num_per_comm;
        if ASCEND_IS_AIV {
            TPipe pipe;
            pipe.InitBuffer(uBuf_, USED_UB_SIZE);
        }
    }

    __aicore__ inline void SetTiling(MatmulReduceScatterV2AivModeTilingData &tilingData)
    {
        m0 = tilingData.cocTiling.m0;
        k0 = tilingData.cocTiling.k0;
        n0 = tilingData.cocTiling.n0;
        swizzl_count = tilingData.cocTiling.swizzlCount;
        swizzl_direct = tilingData.cocTiling.swizzlDirect;
        p_value = tilingData.cocTiling.pValue;
        max_ub_single_dma_size = tilingData.cocTiling.ubMoveNum;
        comm_npu_split = tilingData.cocTiling.commNpuSplit;
        comm_data_split = tilingData.cocTiling.commDataSplit;
        len_per_loop = tilingData.cocTiling.lenPerLoop;

        m = tilingData.matmulReduceScatterV2AivModeInfo.M;
        k = tilingData.matmulReduceScatterV2AivModeInfo.K;
        n = tilingData.matmulReduceScatterV2AivModeInfo.N;
        aAlignSize = tilingData.matmulReduceScatterV2AivModeInfo.aAlignSize;
        bAlignSize = tilingData.matmulReduceScatterV2AivModeInfo.bAlignSize;
        dequantSize = tilingData.matmulReduceScatterV2AivModeInfo.dequantSize;
        biasCastSize = tilingData.matmulReduceScatterV2AivModeInfo.biasCastSize;
        hasAAlign = tilingData.matmulReduceScatterV2AivModeInfo.hasAAlign;
        hasBAlign = tilingData.matmulReduceScatterV2AivModeInfo.hasBAlign;
        dequant_type = tilingData.matmulReduceScatterV2AivModeInfo.dequant_type;
        is910C = tilingData.matmulReduceScatterV2AivModeInfo.is910C;
        isX2ScaleTypeInt64 = tilingData.matmulReduceScatterV2AivModeInfo.isX2ScaleTypeInt64;
    }

    __aicore__ inline void AlignJudge(bool trans_a, bool trans_b, int32_t m, int32_t k, int32_t n, int32_t m_align,
                                      int32_t k_align, int32_t n_align, int32_t &aligned_a, int32_t &aligned_b)
    {
        if (!trans_a) {
            aligned_a = k != k_align;
        } else {
            aligned_a = (m != m_align && m != 1);
        }

        if (!trans_b) {
            aligned_b = (n != n_align);
        } else {
            aligned_b = (k != k_align);
        }
    }

    __aicore__ inline void ResetIpcFlags(int32_t num_flags)
    {
        for (int32_t idx = 0; idx < num_flags; ++idx) {
            if (core_idx == 0 && aiv_idx == 0) {
                SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + idx, 0);
            }
        }
    }

    __aicore__ inline void ResetIpcFlags(int32_t num_flags, int32_t val)
    {
        if (core_idx == 0 && aiv_idx == 1) {
            for (int32_t idx = 0; idx < num_flags; ++idx) {
                SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + idx, val);
            }
        }
    };

    __aicore__ inline void SetBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
        LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
        ubTensor(0) = flag + RAND_BASE;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    }

    __aicore__ inline void SetBuffFlagByAdd(__gm__ int32_t *buff, int32_t flag)
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
        ubTensor(0) = flag;
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<int32_t>();
        PipeBarrier<PIPE_ALL>();
        CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CheckBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
        while (true) {
            CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3); // Scalar等MTE2
            if (ubTensor(0) == flag + RAND_BASE) {
                break;
            }
        }
        uBuf_.FreeTensor<int32_t>(ubTensor);
    }

    __aicore__ inline void CheckBuffFlagV2(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();

        while (true) {
            CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3); // Scalar等MTE2
            if (ubTensor(0) > flag + RAND_BASE) {
                uint32_t cut_flag = ubTensor(0);
                break;
            }
        }
        uBuf_.FreeTensor<int32_t>(ubTensor);
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbufAlignB16(LocalTensor<T> ubTensor, __gm__ T *src, uint16_t nBurst,
                                                uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,    // blockCount
                                         lenBurst,  // blockLen
                                         srcStride, // srcStride
                                         dstStride, // dstStride
                                         0);
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(src);
        DataCopyPadExtParams<T> padParams;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyUbufToGmAlignB16(__gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst,
                                                uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,    // blockCount
                                         lenBurst,  // blockLen
                                         srcStride, // srcStride
                                         dstStride, // dstStride
                                         0);
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(dst);
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbuf(LocalTensor<T> ubTensor, __gm__ T *src, uint16_t nBurst, uint32_t lenBurst,
                                        uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyParams dataCopyParams(nBurst,    // blockCount
                                      lenBurst,  // blockLen
                                      srcStride, // srcStride
                                      dstStride  // dstStride
        );
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(src);
        DataCopy(ubTensor, gmTensor, dataCopyParams);
    }

    template <typename T>
    __aicore__ inline void CopyUbufToGm(__gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst, uint16_t lenBurst,
                                        uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyParams dataCopyParams(nBurst,    // blockCount
                                      lenBurst,  // blockLen
                                      srcStride, // srcStride
                                      dstStride  // dstStride
        );
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(dst);
        DataCopy(gmTensor, ubTensor, dataCopyParams);
    }

    template <typename T>
    __aicore__ inline void CopyUbufToGmUnknown(bool nAlign16, __gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst,
                                               uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        if (nAlign16) {
            CopyUbufToGm(dst, ubTensor, nBurst, lenBurst / 32, srcStride, dstStride / 32);
        } else {
            CopyUbufToGmAlignB16(dst, ubTensor, nBurst, lenBurst, srcStride, dstStride);
        }
    }
    template <pipe_t pipe, uint64_t mode>
    inline __aicore__ void FFTSCrossCoreSync(uint64_t flag_id)
    {
        AscendC::CrossCoreSetFlag<mode, pipe>(flag_id);
    }

    __aicore__ inline void SetAndWaitAivSync(uint64_t flag_idx, int32_t pipe_depth = 2)
    {
        FFTSCrossCoreSync<PIPE_MTE3, AIV_CROSS_CORE_SYNC_MODE>(flag_idx + pipe_depth);
        WaitEvent(flag_idx + pipe_depth);
    }

    __aicore__ inline void SetAicSync(uint64_t flag_idx)
    {
        FFTSCrossCoreSync<PIPE_MTE3, AIC_CROSS_CORE_SYNC_MODE>(flag_idx);
    }

    __aicore__ inline void CrossRankSyncV1(int32_t flag_idx, int32_t flag_data)
    {
        if (core_idx == rank && aiv_idx == 0) {
            SetBuffFlagByAdd((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + flag_idx, FLAG_VALUE);
        } else if (core_idx < rank_size && aiv_idx == 0) {
            CheckBuffFlag((__gm__ int32_t *)buff[core_idx] + FLAG_OFFSET + flag_idx, FLAG_VALUE * flag_data);
        }
    }

    __aicore__ inline void CrossRankSyncV2(int32_t flag_idx, int32_t flag_data)
    {
        if (aiv_idx == 0 && core_idx < rank_size) {
            SetBuffFlagByAdd((__gm__ int32_t *)buff[core_idx] + FLAG_OFFSET + flag_idx, FLAG_VALUE);
        }
        if (aiv_idx == 0 && core_idx == rank) {
            CheckBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + flag_idx, FLAG_VALUE * rank_size * flag_data);
        }
    }

public:
    int32_t block_id;
    int32_t core_num;
    int32_t aiv_idx;
    int32_t core_idx;
    int32_t rank;
    int32_t rank_size;
    GM_ADDR buff[8];
    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t swizzl_count;
    int32_t swizzl_direct;
    int32_t p_value;
    int32_t max_ub_single_dma_size;
    int32_t comm_npu_split;
    int32_t comm_data_split;
    int32_t len_per_loop;

    int32_t m;
    int32_t k;
    int32_t n;

    int32_t m_loop;
    int32_t k_loop;
    int32_t n_loop;

    int32_t loop_num_per_comm;
    int32_t other_rank;
    int32_t max_ub_ping_pong_size;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR biasSrcGM_;
    GM_ADDR biasCastGM_;
    GM_ADDR perChannelScaleGM_;
    GM_ADDR perTokenScaleGM_;
    uint64_t aAlignSize;
    uint64_t bAlignSize;
    uint64_t dequantSize;
    uint64_t biasCastSize;
    bool hasAAlign;
    bool hasBAlign;
    bool nAlign16;
    int32_t gm_c_pingpong_size;
    DequantType dequant_type;
    bool is910C;
    bool isX2ScaleTypeInt64;
};

class CommColumnSplitter {
private:
    int32_t m_batchSize;       // 一次任务的批量计算块数（原loopNumPerComm）
    int32_t m_unitBlocksM;     // 单位任务块的行数（原mLoops）
    int32_t m_unitBlocksN;     // 单位任务块的列数（原nLoops）
    int32_t m_totalUnitBlocks; // 单位任务块的总数（m_unitBlocksM * m_unitBlocksN）
    int32_t m_n_max;

public:
    __aicore__ explicit CommColumnSplitter(int32_t batchSize, int32_t unitBlocksM, int32_t unitBlocksN, int32_t n_max)
        : m_batchSize(batchSize),
          m_unitBlocksM(unitBlocksM),
          m_unitBlocksN(unitBlocksN),
          m_totalUnitBlocks(unitBlocksM * unitBlocksN),
          m_n_max(n_max)
    {
    }

    /**
     * 计算指定任务索引在n维度上的起始位置
     * @param taskIndex 任务索引（从0开始）
     * @return n维度上的起始位置
     */
    __aicore__ inline int32_t GetCulumnStartPos(int32_t taskIndex) const
    {
        // 计算覆盖的单位任务块数量
        int32_t coveredUnitBlocks = (taskIndex * m_batchSize) / m_totalUnitBlocks;

        // 计算并返回n维度上的起始位置
        return coveredUnitBlocks * m_unitBlocksN;
    }

    __aicore__ inline int32_t GetCulumnEndPos(int32_t taskIndex) const
    {
        // 计算覆盖的单位任务块数量
        int32_t coveredUnitBlocks = ((taskIndex + 1) * m_batchSize) / m_totalUnitBlocks;
        int32_t ret = coveredUnitBlocks * m_unitBlocksN;
        // 计算并返回n维度上的起始位置
        return ret < m_n_max ? ret : m_n_max;
    }

    __aicore__ inline int32_t getBatchSize() const
    {
        return m_batchSize;
    }

    __aicore__ inline int32_t getUnitBlocksM() const
    {
        return m_unitBlocksM;
    }

    __aicore__ inline int32_t getUnitBlocksN() const
    {
        return m_unitBlocksN;
    }
};

} // namespace matmulReduceScatterV2_util
#endif
