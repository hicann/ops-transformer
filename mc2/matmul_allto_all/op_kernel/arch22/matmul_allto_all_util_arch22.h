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
 * \file matmul_allto_all_util_arch22.h
 * \brief
 */
#ifndef MATMUL_ALLTO_ALL_UTIL_ARCH22
#define MATMUL_ALLTO_ALL_UTIL_ARCH22

#include "matmul_allto_all_tiling.h"

namespace {
constexpr static uint32_t BUFFER_NUM = 2U;     // 多buf
constexpr static uint32_t STATE_OFFSET = 512U; // 状态空间偏移地址
constexpr static uint32_t BLOCK_SIZE = 32U;
constexpr static uint32_t B32_PER_BLOCK = 8U;
constexpr static uint32_t B64_PER_BLOCK = 4U;
constexpr static int32_t CUBE_MATRIX_SIZE_B16 = 256;        // 16 * 16
constexpr static int32_t L0AB_PINGPONG_BUFFER_SIZE = 32768; // 32 KB
constexpr static int32_t MAX_BLOCK_COUNT = 2;
constexpr static int32_t FLAG_ZERO_IDX = 0;
constexpr static int32_t FLAG_ONE_IDX = 1;
constexpr static int32_t FLAG_VALUE = 1;
constexpr static int32_t USED_UB_SIZE = 160 * 1024;
constexpr static int32_t FLAG_OFFSET = 180 * 1024 * 1024 / sizeof(int32_t);
constexpr static uint32_t UB_OFFSET = 97440 / sizeof(int16_t); // 2 是 size of T
} // namespace

template <typename T, size_t SIZE>
struct BaseBlock {
    static_assert((SIZE & (SIZE - 1)) == 0, "Invalid block size");
    static constexpr size_t size = SIZE / sizeof(T);

    static __aicore__ inline size_t Count(size_t len)
    {
        return (len + size - 1) / size;
    }

    static __aicore__ inline bool IsAligned(size_t len)
    {
        return len % size == 0;
    }

    static __aicore__ inline size_t AlignUp(size_t len)
    {
        return (len + size - 1) & ~(size - 1);
    }

    static __aicore__ inline size_t AlignDown(size_t len)
    {
        return len & ~(size - 1);
    }
};

class CommBase {
public:
    __aicore__ explicit CommBase(){};

    __aicore__ inline void SetArgs(uint32_t *rank, int32_t rank_size, MatmulAlltoAllTilingData info)
    {
        block_id = GetBlockIdx();
        core_num = GetBlockNum();
        aiv_idx = GetSubBlockIdx();
        core_idx = block_id / GetTaskRation();
        this->rank_size = rank_size;

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
        *rank = static_cast<uint32_t>(winContext_->rankId);
        this->rank = winContext_->rankId;

        for (int i = 0; i < rank_size; i++) {
            buff[i] = (GM_ADDR)winContext_->windowsIn[i];
        }

        SetTiling(info);

        if ASCEND_IS_AIV {
            TPipe pipe;
            pipe.InitBuffer(uBuf_, USED_UB_SIZE);
        }
    }

    __aicore__ inline void SetTiling(MatmulAlltoAllTilingData &info)
    {
        m0 = info.cocTiling.m0;
        k0 = info.cocTiling.k0;
        n0 = info.cocTiling.n0;
        p_value = info.cocTiling.pValue;
        max_ub_ping_pong_size = info.cocTiling.ubMoveNum;
        max_ub_ping_pong_size = max_ub_ping_pong_size / MAX_BLOCK_COUNT; // double buffer
        max_ub_ping_pong_size = max_ub_ping_pong_size / n0 * n0;

        m = info.matmulAlltoAllInfo.M;
        k = info.matmulAlltoAllInfo.K;
        n = info.matmulAlltoAllInfo.N;

        m_loop = DivCeil(m, m0);
        n_loop = DivCeil(n, n0);
        k_loop = DivCeil(k, k0);
        gm_a_pingpong_size = static_cast<int64_t>(m0) * n * p_value;
    }

    template <pipe_t pipe, uint64_t mode>
    __aicore__ inline void FFTSCrossCoreSync(uint64_t flag_id)
    {
        AscendC::CrossCoreSetFlag<mode, pipe>(flag_id);
    }

    __aicore__ inline void SetAndWaitAivSync(uint64_t flag_idx, int32_t pipe_depth = 2)
    {
        FFTSCrossCoreSync<PIPE_MTE3, 0>(flag_idx + pipe_depth);
        WaitEvent(flag_idx + pipe_depth);
    }

    __aicore__ inline void SetAicSync(uint64_t flag_idx)
    {
        FFTSCrossCoreSync<PIPE_MTE3, 2>(flag_idx);
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

    __aicore__ inline void CheckBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        LocalTensor<int32_t> ubTensor = uBuf_.template Get<int32_t>();
        while (true) {
            CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3); // Scalar等MTE2
            if (ubTensor(0) == flag) {
                break;
            }
        }
    }

    __aicore__ inline void SetBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
        LocalTensor<int32_t> ubTensor = uBuf_.template Get<int32_t>();
        ubTensor(0) = flag;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    }

    template <typename T>
    __aicore__ inline void CopyGMToGM(__gm__ T *gm_src, __gm__ T *gm_dst, int64_t copy_size)
    {
        LocalTensor<T> ubTensor = uBuf_.template Get<T>();
        LocalTensor<T> copyTensor0 = ubTensor;
        LocalTensor<T> copyTensor1 = ubTensor[UB_OFFSET];
        int64_t interm_offset = 0;
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int32_t move_idx = 0; interm_offset < copy_size; ++move_idx) {
            uint32_t data_size =
                interm_offset + max_ub_ping_pong_size < copy_size ? max_ub_ping_pong_size : copy_size - interm_offset;
            auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T> copyTensor = (move_idx & 1) ? copyTensor0 : copyTensor1;
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            CopyGmToUbufAlignB16(copyTensor, gm_src + interm_offset, 1, data_size * sizeof(T), 0, 0);
            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            CopyUbufToGmAlignB16(gm_dst + interm_offset, copyTensor, 1, data_size * sizeof(T), 0, 0);
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
            interm_offset += data_size;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void ResetIpcFlags(int32_t num_flags)
    {
        for (int32_t idx = 0; idx < num_flags; ++idx) {
            if (core_idx == 0 && aiv_idx == 0) {
                SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + idx, 0);
            }
        }
    }

    __aicore__ inline void CrossRankSyncV1(int32_t flag_idx, int32_t flag_data)
    {
        if (core_idx == 0 && aiv_idx == 0) {
            SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + flag_idx, flag_data);
        }
        if (core_idx < rank_size && aiv_idx == 0) {
            CheckBuffFlag((__gm__ int32_t *)buff[core_idx] + FLAG_OFFSET + flag_idx, flag_data);
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
    __gm__ HcclA2CombineOpParam *winContext_{nullptr};
    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t p_value;
    int32_t max_ub_ping_pong_size;
    int64_t gm_a_pingpong_size;
    int32_t len_per_loop;

    int32_t m;
    int32_t k;
    int32_t n;

    int32_t m_loop;
    int32_t k_loop;
    int32_t n_loop;
};
#endif
