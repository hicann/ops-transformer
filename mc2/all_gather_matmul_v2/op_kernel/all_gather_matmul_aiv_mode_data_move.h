/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Data movement and synchronization implementation for AllGatherMatmulAIVMode.
#pragma once

#include "all_gather_matmul_aiv_mode.h"

namespace AllGatherMatmulAIVModeImpl {

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::CrossRankSyncV1(int32_t flag_idx, int32_t flag_data)
{
    if (aivIdx == 0 && blockIdx == rankId) {
        Mc2AivSync::SetBuffFlagByAdd((__gm__ int32_t *)stateAddrPerRank[rankId] + FLAG_OFFSET + flag_idx, uBuf_,
                                     FLAG_VALUE);
    } else if (aivIdx == 0 && blockIdx < worldSize) {
        CheckBuffFlag((__gm__ int32_t *)stateAddrPerRank[blockIdx] + FLAG_OFFSET + flag_idx, uBuf_,
                      FLAG_VALUE * flag_data);
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::CrossRankSyncV2(int32_t flag_idx, int32_t flag_data)
{
    if (aivIdx == 0 && blockIdx < worldSize) {
        Mc2AivSync::SetBuffFlagByAdd((__gm__ int32_t *)stateAddrPerRank[blockIdx] + FLAG_OFFSET + flag_idx, uBuf_,
                                     FLAG_VALUE);
    }
    if (aivIdx == 0 && blockIdx == rankId) {
        CheckBuffFlag((__gm__ int32_t *)stateAddrPerRank[rankId] + FLAG_OFFSET + flag_idx, uBuf_,
                      FLAG_VALUE * worldSize * flag_data);
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::Padding()
{
    if (!aligned_a && !aligned_b) {
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreFlag flagAivFinishPadding{AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID};
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        return;
    }
    bool transA = false; // 当前暂未支持A矩阵转置
    bool transB = TB;
    bool alignedA = aligned_a;
    bool alignedB = aligned_b;
    uint32_t matrixAM = m;
    uint32_t matrixAK = k;
    uint32_t matrixBK = k;
    uint32_t matrixBN = n;
    uint32_t matrixAMAlign = m_align;
    uint32_t matrixAKAlign = k_align;
    uint32_t matrixBKAlign = k_align;
    uint32_t matrixBNAlign = n_align;
    GM_ADDR gmA = reinterpret_cast<GM_ADDR>(aGM_);
    GM_ADDR gmB = reinterpret_cast<GM_ADDR>(bGM_);
    GM_ADDR gmAAlign = reinterpret_cast<GM_ADDR>(gm_a_align);
    GM_ADDR gmBAlign = reinterpret_cast<GM_ADDR>(gm_b_align);
    PaddingRunner<X1Type, X2Type> padding_runner;
    padding_runner.Run(PADDING_ARGS_CALL());
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveResultFromSrcToDst(__gm__ supportX1Type *gm_src,
                                                                                        __gm__ supportX1Type *gm_dst,
                                                                                        int32_t len)
{
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
    MoveResultToDst(gm_src, gm_dst, len);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::AllGatherPerTokenScale(int64_t buff_st)
{
    if (!needPerToken) {
        return;
    }

    int32_t multi = sizeof(float32_t) / sizeof(supportX1Type);
    int32_t scale_size = m * multi;
    int32_t scale_st = rankId * scale_size;
    __gm__ supportX1Type *scale = reinterpret_cast<__gm__ supportX1Type *>(x1ScaleGM_);
    __gm__ supportX1Type *scaleOut = reinterpret_cast<__gm__ supportX1Type *>(gm_scale_workspace);
    Mc2AivSync::SetAndWaitAivSync(FLAG_VALUE);
    // 将本卡的scale拷贝到buff中
    if (aivIdx == 0 && rankId == blockIdx) {
        MoveResultFromSrcToDst(
            scale, reinterpret_cast<__gm__ supportX1Type *>(stateAddrPerRank[rankId]) + buff_st + scale_st, scale_size);
    }
    CrossRankSyncV1(FLAG_TWO_IDX, FLAG_VALUE);
    Mc2AivSync::SetAndWaitAivSync(FLAG_VALUE);
    // 将其他卡的scale拷贝到buff中
    scale_st = blockIdx * scale_size;
    if (aivIdx == 0 && blockIdx < worldSize) {
        MoveResultFromSrcToDst(
            reinterpret_cast<__gm__ supportX1Type *>(stateAddrPerRank[blockIdx]) + buff_st + scale_st,
            scaleOut + scale_st, scale_size);
    }
    Mc2AivSync::SetAndWaitAivSync(FLAG_VALUE);
    CrossRankSyncV2(FLAG_THREE_IDX, FLAG_VALUE);
    Mc2AivSync::SetAndWaitAivSync(FLAG_VALUE);
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::Dequant(int32_t cal_idx)
{
    // per token 反量化实现
    if (!needPerChannel && !needPerToken) {
        return;
    }

    uint32_t rowNum = cal_idx == cal_count - 1 ? m - cal_idx * m0 * pValue : m0 * pValue;
    uint32_t colNum = n;
    uint32_t tileM0 = m0;
    uint32_t tileN0 = n0;

    int64_t blockSt = static_cast<int64_t>(cal_idx) * m0 * pValue * n;
    int64_t blockSize = static_cast<int64_t>(m) * n;
    int64_t blockStInWorkspace = static_cast<int64_t>(cal_idx % MAX_BLOCK_COUNT) * worldSize * m0 * pValue * n;
    int64_t blockSizeInWorkspace = static_cast<int64_t>(m0) * pValue * n;

    if (!accumWorkSpacePingPong) {
        blockStInWorkspace = blockSt;
        blockSizeInWorkspace = blockSize;
    }

    __gm__ float32_t *perChannelScale = needPerChannel ? reinterpret_cast<__gm__ float32_t *>(x2ScaleGM_) : nullptr;
    __gm__ float32_t *perTokenScale = needPerToken ? reinterpret_cast<__gm__ float32_t *>(gm_scale_workspace) : nullptr;
    __gm__ int32_t *workspace = needPerChannel ? reinterpret_cast<__gm__ int32_t *>(gm_accum) : nullptr;
    __gm__ YType *output = reinterpret_cast<__gm__ YType *>(cGM_);

    // 当 X1Type 为 int4_t 时，只让 subblockIdx == 1 的核参与计算
    constexpr bool isInt4Type = std::is_same_v<X1Type, AscendC::int4b_t>;
    if (isInt4Type && aivIdx != 1) {
        return;
    }

    dequantRunner.Run(DEQUANT_ARGS_CALL());
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveToOtherRankWithSkip(
    __gm__ supportX1Type *gm_src, int64_t rank_offset, int32_t len, int32_t rank_st, int32_t skip_num,
    int32_t group_num, int32_t rank_scope)
{
    LocalTensor<supportX1Type> ubTensor = uBuf_.AllocTensor<supportX1Type>();
    LocalTensor<supportX1Type> copyTensor0 = ubTensor;
    LocalTensor<supportX1Type> copyTensor1 = ubTensor[UB_OFFSET];
    int32_t ping_pong_move_count = (len + max_ub_ping_pong_size - 1) / max_ub_ping_pong_size;
    for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
        int32_t actual_move_size = max_ub_ping_pong_size;
        if (move_idx == ping_pong_move_count - 1) {
            actual_move_size = len - move_idx * max_ub_ping_pong_size;
        }
        int32_t block_len = actual_move_size * Catlass::SizeOfBits<X1Type>::value / 8;
        auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<supportX1Type> copyTensor = (move_idx & 1) ? copyTensor0 : copyTensor1;
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        Mc2AivDataCopy::CopyGmToUbufAlignB16(copyTensor, gm_src, 1, block_len, 0, 0);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        int32_t dst_rank = rank_st % rank_scope;
        for (int32_t cycle_idx = 0; cycle_idx < group_num; ++cycle_idx) {
            if (dst_rank != rankId && dst_rank < worldSize) {
                if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                    Mc2AivDataCopy::CopyUbufToGmAlignB16((__gm__ int8_t *)stateAddrPerRank[dst_rank] + rank_offset / 2,
                                                         copyTensor, 1, block_len, 0, 0);
                } else {
                    Mc2AivDataCopy::CopyUbufToGmAlignB16((__gm__ X1Type *)stateAddrPerRank[dst_rank] + rank_offset,
                                                         copyTensor, 1, block_len, 0, 0);
                }
            }
            dst_rank = (dst_rank + skip_num) % rank_scope;
        }
        if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
            gm_src += (max_ub_ping_pong_size / 2);
        } else {
            gm_src += max_ub_ping_pong_size;
        }
        rank_offset += max_ub_ping_pong_size;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
    }
    uBuf_.FreeTensor<supportX1Type>(ubTensor);
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveResultFromPeerMemToOut(
    __gm__ supportX1Type *gm_src, __gm__ supportX1Type *gm_dst, int32_t actual_m)
{
    LocalTensor<supportX1Type> ubTensor = uBuf_.AllocTensor<supportX1Type>();
    LocalTensor<supportX1Type> copyTensor0 = ubTensor;
    LocalTensor<supportX1Type> copyTensor1 = ubTensor[UB_OFFSET];
    max_move_m = max_ub_ping_pong_size > max_move_k ? max_ub_ping_pong_size / max_move_k : 1;
    int32_t ping_pong_move_count = (actual_m + max_move_m - 1) / max_move_m;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
    for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
        int32_t actual_move_m = max_move_m;
        if (move_idx == ping_pong_move_count - 1) {
            actual_move_m = actual_m - move_idx * max_move_m;
        }
        auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<supportX1Type> ub_buff_st = (move_idx & 1) ? copyTensor0 : copyTensor1;
        int32_t k_move_count = (k_align + max_move_k - 1) / max_move_k;
        for (int32_t k_move_idx = 0; k_move_idx < k_move_count; ++k_move_idx) {
            int32_t actual_k_move_num_in_peer_mem = max_move_k;
            int32_t actual_k_move_num_in_out = max_move_k;
            if (k_move_idx == k_move_count - 1) {
                actual_k_move_num_in_peer_mem = k_align - k_move_idx * max_move_k;
                actual_k_move_num_in_out = k - k_move_idx * max_move_k;
            }
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            int64_t gm_src_offset_k_align =
                static_cast<int64_t>(move_idx) * max_move_m * k_align + k_move_idx * max_move_k;
            if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                gm_src_offset_k_align = gm_src_offset_k_align / 2;
            }
            Mc2AivDataCopy::CopyGmToUbuf(
                ub_buff_st, gm_src + gm_src_offset_k_align, actual_move_m,
                actual_k_move_num_in_peer_mem * Catlass::SizeOfBits<X1Type>::value / (8 * 32),
                (k_align - actual_k_move_num_in_peer_mem) * Catlass::SizeOfBits<X1Type>::value / (8 * 32), 0);
            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            int64_t gm_src_offset = static_cast<int64_t>(move_idx) * max_move_m * k + k_move_idx * max_move_k;
            if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                gm_src_offset = gm_src_offset / 2;
            }
            Mc2AivDataCopy::CopyUbufToGmAlignB16(
                gm_dst + gm_src_offset, ub_buff_st, actual_move_m,
                actual_k_move_num_in_out * Catlass::SizeOfBits<X1Type>::value / 8,
                (actual_k_move_num_in_peer_mem - actual_k_move_num_in_out) * Catlass::SizeOfBits<X1Type>::value /
                    (8 * 32),
                (k - actual_k_move_num_in_out) * Catlass::SizeOfBits<X1Type>::value / (8 * 32));
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveResultToDst(__gm__ supportX1Type *gm_src,
                                                                                 __gm__ supportX1Type *gm_dst,
                                                                                 int32_t len)
{
    LocalTensor<supportX1Type> ubTensor = uBuf_.AllocTensor<supportX1Type>();
    LocalTensor<supportX1Type> copyTensor0 = ubTensor;
    LocalTensor<supportX1Type> copyTensor1 = ubTensor[UB_OFFSET];
    int32_t ping_pong_move_count = (len + max_ub_ping_pong_size - 1) / max_ub_ping_pong_size;
    for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
        int32_t actual_move_size = max_ub_ping_pong_size;
        if (move_idx == ping_pong_move_count - 1) {
            actual_move_size = len - move_idx * max_ub_ping_pong_size;
        }
        auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<supportX1Type> copyTensor = (move_idx & 1) ? copyTensor0 : copyTensor1;
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        Mc2AivDataCopy::CopyGmToUbufAlignB16(copyTensor, gm_src, 1, actual_move_size * sizeof(supportX1Type), 0, 0);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        Mc2AivDataCopy::CopyUbufToGmAlignB16(gm_dst, copyTensor, 1, actual_move_size * sizeof(supportX1Type), 0, 0);
        gm_src += max_ub_ping_pong_size;
        gm_dst += max_ub_ping_pong_size;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
    }
    uBuf_.FreeTensor<supportX1Type>(ubTensor);
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveWithSplit(__gm__ supportX1Type *gm_src,
                                                                               int64_t rank_offset, int64_t len)
{
    int64_t data_split = DivCeil(len, static_cast<int64_t>(len_per_loop));
    int32_t data_block = len_per_loop; // 每份数据量 len_per_loop = 2560
    int32_t rank_st = blockIdx;
    int32_t group_num = DivCeil(worldSize, comm_npu_split); // 1？ comm_npu_split=worldSize?
    int32_t scope = comm_npu_split * group_num;             // worldSize？
    int64_t data_offset = -data_block;                      // 当前份数据的起始位置
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);               // MTE2等MTE3
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);               // MTE2等MTE3
    for (int64_t data_block_idx = 0; data_block_idx < data_split; ++data_block_idx) {
        data_offset += data_block; // 当前份数据的起始位置
        data_block = data_block_idx == data_split - 1 ? static_cast<int32_t>(len - data_offset) : data_block;
        int32_t num_per_core = DivCeil(data_block, comm_data_split); // 2560

        int64_t data_src = data_offset + (blockIdx / comm_npu_split) * num_per_core;
        int32_t data_len = static_cast<int32_t>(data_block + data_offset - data_src);
        data_len = data_len >= num_per_core ? num_per_core : data_len;
        // npu 方向：一份数据先发送到所有目标卡，再发送下一份数据，以此类推
        if (comm_direct) { // comm_direct=0？
            if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                MoveToOtherRankWithSkip(gm_src + data_src / 2, rank_offset + data_src, data_len, rank_st,
                                        comm_npu_split, group_num, scope);
            } else {
                MoveToOtherRankWithSkip(gm_src + data_src, rank_offset + data_src, data_len, rank_st, comm_npu_split,
                                        group_num, scope);
            }
            continue;
        }
        // data len 方向：所有的数据先发送到目标卡0，再发送到目标卡1，以此类推
        int32_t dst_rank = rank_st % scope;
        for (int32_t rank_group_idx = 0; rank_group_idx < group_num; ++rank_group_idx) {
            if (dst_rank != rankId && dst_rank < worldSize) {
                if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                    MoveResultToDst(gm_src + data_src / 2,
                                    (__gm__ int8_t *)stateAddrPerRank[dst_rank] + (rank_offset + data_src) / 2,
                                    data_len / 2);
                } else {
                    MoveResultToDst(gm_src + data_src,
                                    (__gm__ X1Type *)stateAddrPerRank[dst_rank] + rank_offset + data_src, data_len);
                }
            }
            dst_rank = (dst_rank + comm_npu_split) % scope;
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
}

} // namespace AllGatherMatmulAIVModeImpl
