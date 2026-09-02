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
 * \file cube_op.h
 * \brief
 */
#pragma once
#include "kernel_operator.h"
#include "nsa_selected_attention_grad_common_header.h"
using namespace AscendC;

namespace NSAG_BASIC {

template <typename NSAGT>
class CubeOp {
    using TILING_CLASS = typename NSAGT::tiling_class;
    using T1 = typename NSAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = NSAGT::atten_enable;

public:
    /* clang-format off */
    __aicore__ inline CubeOp(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                GM_ADDR atten_mask, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                const TILING_CLASS *__restrict tilingData, TPipe *pipe);

    __aicore__ inline __attribute__((always_inline)) void cube12Process(const int64_t queryGmOffset,
                                                                       const int64_t keyGmOffset,
                                                                       const int64_t dyGmOffset,
                                                                       const int64_t valueGmOffset,
                                                                       const int64_t indicesGmOffset,
                                                                       const int64_t mm12GmOffset,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx);

    __aicore__ inline __attribute__((always_inline)) void cube345Process(const int64_t queryGmOffset,
                                                                         const int64_t keyGmOffset,
                                                                         const int64_t dyGmOffset,
                                                                         const int64_t valueGmOffset,
                                                                         const int64_t indicesGmOffset,
                                                                         const int64_t mm345GmOffset,
                                                                         const int32_t blkCntOffset,
                                                                         const int32_t mmPingPongIdx);
private:
    __aicore__ inline __attribute__((always_inline)) void cube1Process(const int64_t queryGmOffset,
                                                                       const int64_t keyGmOffset,
                                                                       const int64_t indicesGmOffset,
                                                                       const int64_t outGmOffset,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx);

    __aicore__ inline __attribute__((always_inline)) void cube2Process(const int64_t dyGmOffset,
                                                                       const int64_t valueGmOffset,
                                                                       const int64_t indicesGmOffset,
                                                                       const int64_t outGmOffset,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx);

    __aicore__ inline __attribute__((always_inline)) void cube3Process(const int64_t dsGmOffset,
                                                                       const int64_t keyGmOffset,
                                                                       const int64_t indicesGmOffset,
                                                                       const int64_t outGmOffset,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx);

    __aicore__ inline __attribute__((always_inline)) void cube4Process(const int64_t dsGmOffset,
                                                                       const int64_t queryGmOffset,
                                                                       const int64_t indicesGmOffset,
                                                                       const int64_t outGmOffset,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx);

    __aicore__ inline __attribute__((always_inline)) void cube5Process(const int64_t pGmOffset,
                                                                       const int64_t dyGmOffset,
                                                                       const int64_t indicesGmOffset,
                                                                       const int64_t outGmOffset,
                                                                       const int32_t blkCntOffset,
                                                                       const int32_t mmPingPongIdx);
    __aicore__ inline __attribute__((always_inline)) void LoadBData(const int64_t dsGmOffset,
                                                                    const int64_t keyGmOffset,
                                                                    const int64_t indicesGmOffset,
                                                                    const int64_t outGmOffset,
                                                                    const int32_t blkCntOffset,
                                                                    const int32_t mmPingPongIdx);

    __aicore__ inline __attribute__((always_inline)) void Cube4LoadAData(const int64_t dsGmOffset,
                                                                         const int64_t queryGmOffset,
                                                                         const int64_t indicesGmOffset,
                                                                         const int64_t outGmOffset,
                                                                         const int32_t blkCntOffset);

    __aicore__ inline __attribute__((always_inline)) void Cube5LoadAData(const int64_t pGmOffset,
                                                                         const int64_t dyGmOffset,
                                                                         const int64_t indicesGmOffset,
                                                                         const int64_t outGmOffset,
                                                                         const int32_t blkCntOffset);

private:
    __aicore__ inline void InitGMBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                        GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                        GM_ADDR topk_indices, GM_ADDR workspace);
    __aicore__ inline void InitLocalTensor();

    AscendC::Nd2NzParams commonNd2NzParams {
        1,
        0,
        0,
        0,
        128,
        128,
        1,
        0
    };
    AscendC::Nd2NzParams cube1Nd2NzParams{
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0
    };
    AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose {
        0,
        128,
        128,
        0,
        0,
        false,
        0
    };

    AscendC::LoadData2dParams commonLoadData2dParamsTrans {
        0,
        128,
        128,
        0,
        0,
        true,
        0
    };
    /* clang-format on */
    // Gm Addr
    GlobalTensor<T1> queryGm;
    GlobalTensor<T1> keyGm;
    GlobalTensor<T1> valueGm;
    GlobalTensor<T1> attentionGm;
    GlobalTensor<T1> attentionGradGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<T1> dqGm;
    GlobalTensor<T1> dkGm;
    GlobalTensor<T1> dvGm;
    // workspace
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> pWorkspaceGm;
    GlobalTensor<T1> dsWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> dvWorkspaceGm;
    // workspace
    uint32_t mm12WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;

    uint32_t ping_pong_flag_l1_a_ = 0;
    uint32_t ping_pong_flag_l1_b_ = 0;
    // L1
    LocalTensor<T1> l1_query_ping_tensor;
    LocalTensor<T1> l1_query_pong_tensor;
    LocalTensor<T1> l1_key_ping_tensor;
    LocalTensor<T1> l1_key_pong_tensor;
    LocalTensor<T1> l1_dy_ping_tensor;
    LocalTensor<T1> l1_dy_pong_tensor;
    LocalTensor<T1> l1_ds_tensor;
    LocalTensor<T1> l1_tmp_tensor;
    // L0
    LocalTensor<T1> l0_a_cube1_tensor;
    LocalTensor<T1> l0_a_cube2_tensor;
    LocalTensor<T1> l0_a_cube3_tensor;
    LocalTensor<T1> l0_a_cube4_ping_tensor;
    LocalTensor<T1> l0_a_cube4_pong_tensor;
    LocalTensor<T1> l0_a_cube5_ping_tensor;
    LocalTensor<T1> l0_a_cube5_pong_tensor;

    LocalTensor<T1> l0_b_ding_tensor;
    LocalTensor<T1> l0_b_dong_tensor;
    LocalTensor<T1> l0_b_dung_tensor;

    LocalTensor<float> l0_c_ping_tensor;
    LocalTensor<float> l0_c_pong_tensor;
    LocalTensor<float> l0_c_ding_tensor;
    LocalTensor<float> l0_c_dong_tensor;
    LocalTensor<float> l0_c_dung_tensor;

    TBuf<AscendC::TPosition::CO1> L0CBuffer;
    TBuf<AscendC::TPosition::A1> L1Buffer;
    constexpr static int32_t SIZE_16 = 16;
    constexpr static int32_t BLOCK_SIZE = 16 * 16; // 一个分形的数据量
    constexpr static uint32_t BLOCK_WORKSPACE = 16 * 128 * 128;
    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t SINGLE_BLOCK_CNT = 8; // 单次最多处理的selcted_block_cnt
    uint32_t pingPongIdx = 0;
    uint64_t globalBlockOffset = 0;

    int64_t dimN2;
    int64_t dimG;
    int64_t dimGAlign;
    int64_t dimDqk;
    int64_t dimDv;
    int64_t selectedBlockCount;
    int64_t selectedBlockSize;
    int64_t selectedS2;
    int32_t processBS1ByCore;
    uint32_t usedCoreNum;
    uint32_t eventIdPing = 4;
    uint32_t eventIdPong = 5;
};

template <typename NSAGT>
__aicore__ inline void CubeOp<NSAGT>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                           GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                           GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                           GM_ADDR atten_mask, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                           const TILING_CLASS *__restrict tilingData, TPipe *pipe)
{
    dimG = tilingData->opInfo.G;
    dimN2 = tilingData->opInfo.N2;
    dimDqk = tilingData->opInfo.D;
    dimDv = tilingData->opInfo.D2;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    selectedS2 = selectedBlockCount * selectedBlockSize;
    dimGAlign = AlignTo<int64_t>(dimG, SIZE_16);
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    dvWorkspaceLen = tilingData->opInfo.dvWorkspaceLen;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    pipe->InitBuffer(L0CBuffer, HardwareInfo<ArchType::ASCEND_V220>::l0CSize);
    pipe->InitBuffer(L1Buffer, HardwareInfo<ArchType::ASCEND_V220>::l1Size);
    AscendC::SetLoadDataPaddingValue<uint64_t>(0);
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    InitGMBuffer(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                 workspace);
    InitLocalTensor();
}

template <typename NSAGT>
__aicore__ inline __attribute__((always_inline)) void CubeOp<NSAGT>::cube12Process(
    const int64_t queryGmOffset, const int64_t keyGmOffset, const int64_t dyGmOffset, const int64_t valueGmOffset,
    const int64_t indicesGmOffset, const int64_t mm12GmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    cube1Process(queryGmOffset, keyGmOffset, indicesGmOffset, mm12GmOffset, blkCntOffset, mmPingPongIdx);
    cube2Process(dyGmOffset, valueGmOffset, indicesGmOffset, mm12GmOffset, blkCntOffset, mmPingPongIdx);
}

template <typename NSAGT>
__aicore__ inline __attribute__((always_inline)) void CubeOp<NSAGT>::cube345Process(
    const int64_t queryGmOffset, const int64_t keyGmOffset, const int64_t dyGmOffset, const int64_t valueGmOffset,
    const int64_t indicesGmOffset, const int64_t mm345GmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    cube5Process(mm345GmOffset, dyGmOffset, indicesGmOffset, valueGmOffset, blkCntOffset, mmPingPongIdx);
    cube4Process(mm345GmOffset, queryGmOffset, indicesGmOffset, keyGmOffset, blkCntOffset, mmPingPongIdx);
    WAIT_FLAG(FIX, M, eventIdPing);
    WAIT_FLAG(FIX, M, eventIdPong);
    cube3Process(mm345GmOffset, keyGmOffset, indicesGmOffset, queryGmOffset, blkCntOffset, mmPingPongIdx);
    SET_FLAG(FIX, M, eventIdPing);
    SET_FLAG(FIX, M, eventIdPong);
}

template <typename NSAGT>
__aicore__ inline void CubeOp<NSAGT>::InitGMBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                                   GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                                   GM_ADDR topk_indices, GM_ADDR workspace)
{
    // 必选输入初始化
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    attentionGm.SetGlobalBuffer((__gm__ T1 *)attention_out);
    attentionGradGm.SetGlobalBuffer((__gm__ T1 *)attention_out_grad);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);

    int64_t usedWorkspaceLen = 0;
    uint32_t blockIdx = GetBlockIdx();
    /*
     * mm1 与 p 复用workspace
     */
    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + blockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t pAddr = usedWorkspaceLen / sizeof(T1) + blockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    /*
     * mm2 与 ds 复用workspace
     */
    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + blockIdx * mm12WorkspaceLen / sizeof(float);
    int64_t dsAddr = usedWorkspaceLen / sizeof(T1) + blockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;

    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    int64_t dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm2Addr);
    pWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + pAddr);
    dsWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + dsAddr);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    dvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvAddr);
}

template <typename NSAGT>
__aicore__ inline void CubeOp<NSAGT>::InitLocalTensor()
{
    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;
    /*
     * Init L1 Tensor
     */
    int32_t l1Offset = 0;
    // query
    l1_query_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += dimGAlign * dimDqk * sizeof(T1);
    l1_query_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += dimGAlign * dimDqk * sizeof(T1);
    // dy
    l1_dy_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += dimGAlign * dimDv * sizeof(T1);
    l1_dy_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += dimGAlign * dimDv * sizeof(T1);
    // key
    l1_key_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += 512 * dimDqk * sizeof(T1);
    l1_key_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += 512 * dimDqk * sizeof(T1);
    // ds
    l1_ds_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);
    l1Offset += dimGAlign * 512 * sizeof(T1);
    // value/p
    l1_tmp_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, T1>(l1Offset);

    /*
     * Init L0 Tensor
     */
    int32_t l0AOffset = 0;
    l0_a_cube1_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, T1>(l0AOffset);
    l0AOffset += SIZE_16 * dimDqk * sizeof(T1);
    l0_a_cube2_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, T1>(l0AOffset);
    l0AOffset += SIZE_16 * dimDv * sizeof(T1);
    l0_a_cube3_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, T1>(l0AOffset);
    l0AOffset += SIZE_16 * 512 * sizeof(T1);
    l0_a_cube4_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, T1>(l0AOffset);
    l0AOffset += SIZE_16 * selectedBlockSize * sizeof(T1);
    l0_a_cube4_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, T1>(l0AOffset);
    l0AOffset += SIZE_16 * selectedBlockSize * sizeof(T1);
    l0_a_cube5_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, T1>(l0AOffset);
    l0AOffset += SIZE_16 * selectedBlockSize * sizeof(T1);
    l0_a_cube5_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, T1>(l0AOffset);
    l0AOffset += SIZE_16 * selectedBlockSize * sizeof(T1);

    l0_b_ding_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, T1>(0);
    l0_b_dong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, T1>(24 * 1024);
    l0_b_dung_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, T1>(48 * 1024);

    l0_c_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(64 * 1024);
    l0_c_ding_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_dong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(48 * 1024);
    l0_c_dung_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(96 * 1024);
}

#include "./cube_modules/cube1.h"
#include "./cube_modules/cube2.h"
#include "./cube_modules/cube3.h"
#include "./cube_modules/cube4.h"
#include "./cube_modules/cube5.h"
} // namespace NSAG_BASIC
