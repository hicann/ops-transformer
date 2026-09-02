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
 * \file  cube2.h
 * \brief Formula: dp = dy * v^T
 * l0_a： dimG * dimDv * sizof(T1) (6k-10k)
 * l0_b： selectedBlockSize * dimDv * sizof(T1) (0-16k / 24k-40k)
 * l0_c： dimG * selectedBlockSize * sizof(T1) (0-4k / 64k-68k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube2Process(
    const int64_t dyGmOffset, const int64_t valueGmOffset, const int64_t indicesGmOffset, const int64_t outGmOffset,
    const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    LocalTensor<T1> l1_dy_tensor = mmPingPongIdx ? l1_dy_ping_tensor : l1_dy_pong_tensor;
    LocalTensor<T1> l1_value_tensor = l1_tmp_tensor;
    LocalTensor<T1> l0_a_tensor = l0_a_cube2_tensor;
    LocalTensor<T1> l0_b_tensor;
    LocalTensor<float> l0_c_tensor;

    cube1Nd2NzParams.nValue = dimG;
    cube1Nd2NzParams.dValue = dimDv;
    cube1Nd2NzParams.srcDValue = dimDv;
    cube1Nd2NzParams.dstNzC0Stride = dimGAlign;
    DataCopy(l1_dy_tensor, attentionGradGm[dyGmOffset], cube1Nd2NzParams);

    SET_FLAG(MTE2, MTE1, EVENT_ID0);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

    commonLoadData2dParamsNoTranspose.repeatTimes = dimDv / SIZE_16;
    commonLoadData2dParamsNoTranspose.srcStride = dimGAlign / SIZE_16;
    for (int32_t i = 0; i < dimGAlign / SIZE_16; i++) {
        LoadData(l0_a_tensor[i * dimDv * SIZE_16], l1_dy_tensor[i * BLOCK_SIZE], commonLoadData2dParamsNoTranspose);
    }

    for (int32_t nIdx = blkCntOffset; nIdx < blkCntOffset + 8; nIdx++) {
        bool flag = (nIdx % 2 == 0);
        int32_t topkIdx = topkIndicesGm[indicesGmOffset].GetValue(nIdx);
        int32_t startS2Idx = topkIdx * selectedBlockSize * dimN2 * dimDv;
        int32_t l1Offset = ((nIdx - blkCntOffset) % 5) * selectedBlockSize * dimDv;
        uint32_t eventId = flag ? eventIdPing : eventIdPong;
        l0_b_tensor = flag ? l0_b_ding_tensor : l0_b_dong_tensor;
        l0_c_tensor = flag ? l0_c_ping_tensor : l0_c_pong_tensor;

        if ((blkCntOffset + 8 - nIdx) <= 3) {
            WAIT_FLAG(M, MTE2, EVENT_ID0);
        }
        commonNd2NzParams.nValue = selectedBlockSize;
        commonNd2NzParams.dValue = dimDv;
        commonNd2NzParams.srcDValue = dimDv * dimN2;
        commonNd2NzParams.dstNzC0Stride = selectedBlockSize;
        DataCopy(l1_value_tensor[l1Offset], valueGm[valueGmOffset + startS2Idx], commonNd2NzParams);

        SET_FLAG(MTE2, MTE1, eventId);
        WAIT_FLAG(MTE2, MTE1, eventId);
        WAIT_FLAG(M, MTE1, eventId);

        commonLoadData2dParamsNoTranspose.repeatTimes = selectedBlockSize * dimDv / 256;
        commonLoadData2dParamsNoTranspose.srcStride = 1;
        LoadData(l0_b_tensor, l1_value_tensor[l1Offset], commonLoadData2dParamsNoTranspose);

        SET_FLAG(MTE1, M, eventId);
        WAIT_FLAG(MTE1, M, eventId);
        WAIT_FLAG(FIX, M, eventId);

        MmadParams madParams;
        madParams.m = (dimG == 1 ? 2 : dimG);
        madParams.n = selectedBlockSize;
        madParams.k = dimDv;
        Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);

        if ((nIdx - blkCntOffset) < 3) {
            SET_FLAG(M, MTE2, EVENT_ID0);
        }
        SET_FLAG(M, FIX, eventId);
        WAIT_FLAG(M, FIX, eventId);
        SET_FLAG(M, MTE1, eventId);

        FixpipeParamsV220 fixParams;
        fixParams.mSize = dimG;
        fixParams.nSize = selectedBlockSize;
        fixParams.srcStride = dimGAlign;
        fixParams.dstStride = 512;
        Fixpipe<float, float, CFG_ROW_MAJOR>(mm2WorkspaceGm[outGmOffset + (nIdx - blkCntOffset) * selectedBlockSize],
                                             l0_c_tensor, fixParams);
        SET_FLAG(FIX, M, eventId);
    }
}
