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
 * \file cube4.h
 * \brief Formula: dk = s^T * q
 * l0_a： dimG * selectedBlockSize * sizof(T1) (26-28k/28-30k)
 * l0_b： dimG * 512 * sizof(T1) (0k-16k)
 * l0_c： dimG * dimDqk * sizof(flot) (0-12k / 64k-76k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::Cube4LoadAData(const int64_t dsGmOffset,
                                                                                 const int64_t queryGmOffset,
                                                                                 const int64_t indicesGmOffset,
                                                                                 const int64_t outGmOffset,
                                                                                 const int32_t blkCntOffset)
{
    LocalTensor<T1> l0_a_tensor;
    LocalTensor<T1> l0_b_tensor = l0_b_ding_tensor;
    LocalTensor<float> l0_c_tensor;

    cube1Nd2NzParams.nValue = dimG;
    cube1Nd2NzParams.dValue = 512;
    cube1Nd2NzParams.srcDValue = 512;
    cube1Nd2NzParams.dstNzC0Stride = dimGAlign;
    DataCopy(l1_ds_tensor, dsWorkspaceGm[dsGmOffset], cube1Nd2NzParams);

    SET_FLAG(MTE2, MTE1, EVENT_ID0);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

    for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + 8; mIdx++) {
        bool flag = (mIdx % 2 == 0);
        int32_t topkIdx = topkIndicesGm[indicesGmOffset].GetValue(mIdx);
        int32_t startS2Idx = topkIdx * selectedBlockSize * dimN2 * dimDqk;
        int32_t l1Offset = (mIdx - blkCntOffset) * selectedBlockSize * dimGAlign;
        uint32_t eventId = flag ? eventIdPing : eventIdPong;
        l0_a_tensor = flag ? l0_a_cube4_ping_tensor : l0_a_cube4_pong_tensor;
        l0_c_tensor = flag ? l0_c_ping_tensor : l0_c_pong_tensor;

        WAIT_FLAG(M, MTE1, eventId);

        commonLoadData2dParamsTrans.repeatTimes = selectedBlockSize / SIZE_16;
        commonLoadData2dParamsTrans.srcStride = dimGAlign / SIZE_16;
        for (int32_t i = 0; i < dimGAlign / SIZE_16; i++) {
            LoadData(l0_a_tensor[i * selectedBlockSize * SIZE_16], l1_ds_tensor[l1Offset + i * BLOCK_SIZE],
                     commonLoadData2dParamsTrans);
        }

        SET_FLAG(MTE1, M, eventId);
        WAIT_FLAG(MTE1, M, eventId);
        WAIT_FLAG(FIX, M, eventId);

        MmadParams madParams;
        madParams.m = selectedBlockSize;
        madParams.n = dimDqk;
        madParams.k = dimG;
        Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);

        SET_FLAG(M, FIX, eventId);
        WAIT_FLAG(M, FIX, eventId);
        SET_FLAG(M, MTE1, eventId);

        FixpipeParamsV220 fixParams;
        fixParams.mSize = selectedBlockSize;
        fixParams.nSize = dimDqk;
        fixParams.srcStride = selectedBlockSize;
        fixParams.dstStride = dimDqk * dimN2;
        AscendC::SetAtomicType<float>();
        Fixpipe<float, float, CFG_ROW_MAJOR>(dkWorkspaceGm[outGmOffset + startS2Idx], l0_c_tensor, fixParams);
        AscendC::SetAtomicNone();

        SET_FLAG(FIX, M, eventId);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube4Process(
    const int64_t dsGmOffset, const int64_t queryGmOffset, const int64_t indicesGmOffset, const int64_t outGmOffset,
    const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    LocalTensor<T1> l1_query_tensor = mmPingPongIdx ? l1_query_ping_tensor : l1_query_pong_tensor;
    LocalTensor<T1> l0_b_tensor = l0_b_ding_tensor;

    commonLoadData2dParamsTrans.repeatTimes = dimGAlign * dimDqk / 256;
    commonLoadData2dParamsTrans.srcStride = 1;
    LoadData(l0_b_tensor, l1_query_tensor, commonLoadData2dParamsTrans);

    Cube4LoadAData(dsGmOffset, queryGmOffset, indicesGmOffset, outGmOffset, blkCntOffset);
}
