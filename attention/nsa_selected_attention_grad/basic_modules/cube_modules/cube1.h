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
 * \file cube1.h
 * \brief Formula: s = q * k^T
 * l0_a： dimG * dimDqk * sizof(T1) (0-6k)
 * l0_b： selectedBlockSize * dimDqk * sizof(T1) (0-24k / 24k-48k)
 * l0_c： dimG * selectedBlockSize * sizof(T1) (0-4k / 64k-68k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube1Process(
    const int64_t queryGmOffset, const int64_t keyGmOffset, const int64_t indicesGmOffset, const int64_t outGmOffset,
    const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    LocalTensor<T1> l1_query_tensor = mmPingPongIdx ? l1_query_ping_tensor : l1_query_pong_tensor;
    LocalTensor<T1> l1_key_tensor = mmPingPongIdx ? l1_key_ping_tensor : l1_key_pong_tensor;
    LocalTensor<T1> l0_a_tensor = l0_a_cube1_tensor;
    LocalTensor<T1> l0_b_tensor;
    LocalTensor<float> l0_c_tensor;

    cube1Nd2NzParams.nValue = dimG;
    cube1Nd2NzParams.dValue = dimDqk;
    cube1Nd2NzParams.srcDValue = dimDqk;
    cube1Nd2NzParams.dstNzC0Stride = dimGAlign;
    DataCopy(l1_query_tensor, queryGm[queryGmOffset], cube1Nd2NzParams);

    SET_FLAG(MTE2, MTE1, EVENT_ID0);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

    commonLoadData2dParamsNoTranspose.repeatTimes = dimDqk / SIZE_16;
    commonLoadData2dParamsNoTranspose.srcStride = dimGAlign / SIZE_16;
    for (int32_t i = 0; i < dimGAlign / SIZE_16; i++) {
        LoadData(l0_a_tensor[i * dimDqk * SIZE_16], l1_query_tensor[i * BLOCK_SIZE], commonLoadData2dParamsNoTranspose);
    }

    for (int32_t nIdx = blkCntOffset; nIdx < blkCntOffset + 8; nIdx++) {
        bool flag = (nIdx % 2 == 0);
        int32_t topkIdx = topkIndicesGm[indicesGmOffset].GetValue(nIdx);
        int32_t startS2Idx = topkIdx * selectedBlockSize * dimN2 * dimDqk;
        int32_t keyL1Offset = (nIdx - blkCntOffset) * selectedBlockSize * dimDqk;
        uint32_t eventId = flag ? eventIdPing : eventIdPong;
        l0_b_tensor = flag ? l0_b_ding_tensor : l0_b_dong_tensor;
        l0_c_tensor = flag ? l0_c_ping_tensor : l0_c_pong_tensor;

        commonNd2NzParams.nValue = selectedBlockSize;
        commonNd2NzParams.dValue = dimDqk;
        commonNd2NzParams.srcDValue = dimDqk * dimN2;
        commonNd2NzParams.dstNzC0Stride = selectedBlockSize;

        DataCopy(l1_key_tensor[keyL1Offset], keyGm[keyGmOffset + startS2Idx], commonNd2NzParams);

        SET_FLAG(MTE2, MTE1, eventId);
        WAIT_FLAG(MTE2, MTE1, eventId);
        WAIT_FLAG(M, MTE1, eventId);

        commonLoadData2dParamsNoTranspose.repeatTimes = selectedBlockSize * dimDqk / 256;
        commonLoadData2dParamsNoTranspose.srcStride = 1;
        LoadData(l0_b_tensor, l1_key_tensor[keyL1Offset], commonLoadData2dParamsNoTranspose);

        SET_FLAG(MTE1, M, eventId);
        WAIT_FLAG(MTE1, M, eventId);
        WAIT_FLAG(FIX, M, eventId);

        MmadParams madParams;
        madParams.m = (dimG == 1 ? 2 : dimG);
        madParams.n = selectedBlockSize;
        madParams.k = dimDqk;
        Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);

        SET_FLAG(M, FIX, eventId);
        WAIT_FLAG(M, FIX, eventId);
        SET_FLAG(M, MTE1, eventId);

        FixpipeParamsV220 fixParams;
        fixParams.mSize = dimG;
        fixParams.nSize = selectedBlockSize;
        fixParams.srcStride = dimGAlign;
        fixParams.dstStride = 512;
        Fixpipe<float, float, CFG_ROW_MAJOR>(mm1WorkspaceGm[outGmOffset + (nIdx - blkCntOffset) * selectedBlockSize],
                                             l0_c_tensor, fixParams);
        SET_FLAG(FIX, M, eventId);
    }
}
