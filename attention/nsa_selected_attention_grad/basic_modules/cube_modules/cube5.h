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
 * \file cube5.h
 * \brief Formula: dv = p^T * dy
 * l0_a： dimG * dimDqk * sizof(T1) (30k-32k/32-34k)
 * l0_b： dimG * 512 * sizof(T1) (48-64k)
 * l0_c： dimG * dimDqk * sizof(float) (0-12k / 64k-76k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::Cube5LoadAData(const int64_t pGmOffset,
                                                                                 const int64_t dyGmOffset,
                                                                                 const int64_t indicesGmOffset,
                                                                                 const int64_t outGmOffset,
                                                                                 const int32_t blkCntOffset)
{
    LocalTensor<T1> l0_a_tensor;
    LocalTensor<T1> l0_b_tensor = l0_b_dung_tensor;
    LocalTensor<float> l0_c_tensor;

    for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + 8; mIdx++) {
        bool flag = (mIdx % 2 == 0);
        int32_t topkIdx = topkIndicesGm[indicesGmOffset].GetValue(mIdx);
        int32_t startS2Idx = topkIdx * selectedBlockSize * dimN2 * dimDv;
        int32_t l1Offset = (mIdx - blkCntOffset) * selectedBlockSize * dimGAlign;
        uint32_t eventId = flag ? eventIdPing : eventIdPong;
        l0_a_tensor = flag ? l0_a_cube5_ping_tensor : l0_a_cube5_pong_tensor;
        l0_c_tensor = flag ? l0_c_ping_tensor : l0_c_pong_tensor;

        commonNd2NzParams.nValue = dimG;
        commonNd2NzParams.dValue = selectedBlockSize;
        commonNd2NzParams.srcDValue = 512;
        commonNd2NzParams.dstNzC0Stride = dimGAlign;
        DataCopy(l1_tmp_tensor[l1Offset], pWorkspaceGm[pGmOffset + (mIdx - blkCntOffset) * selectedBlockSize],
                 commonNd2NzParams);

        SET_FLAG(MTE2, MTE1, eventId);
        WAIT_FLAG(MTE2, MTE1, eventId);
        WAIT_FLAG(M, MTE1, eventId);

        commonLoadData2dParamsTrans.repeatTimes = selectedBlockSize / SIZE_16;
        commonLoadData2dParamsTrans.srcStride = dimGAlign / SIZE_16;
        for (int32_t i = 0; i < dimGAlign / SIZE_16; i++) {
            LoadData(l0_a_tensor[i * selectedBlockSize * SIZE_16], l1_tmp_tensor[l1Offset + i * BLOCK_SIZE],
                     commonLoadData2dParamsTrans);
        }

        SET_FLAG(MTE1, M, eventId);
        WAIT_FLAG(MTE1, M, eventId);
        WAIT_FLAG(FIX, M, eventId);

        MmadParams madParams;
        madParams.m = selectedBlockSize;
        madParams.n = dimDv;
        madParams.k = dimG;
        Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);

        SET_FLAG(M, FIX, eventId);
        WAIT_FLAG(M, FIX, eventId);
        SET_FLAG(M, MTE1, eventId);

        FixpipeParamsV220 fixParams;
        fixParams.mSize = selectedBlockSize;
        fixParams.nSize = dimDv;
        fixParams.srcStride = selectedBlockSize;
        fixParams.dstStride = dimDv * dimN2;

        AscendC::SetAtomicType<float>();
        Fixpipe<float, float, CFG_ROW_MAJOR>(dvWorkspaceGm[outGmOffset + startS2Idx], l0_c_tensor, fixParams);
        AscendC::SetAtomicNone();

        SET_FLAG(FIX, M, eventId);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube5Process(
    const int64_t pGmOffset, const int64_t dyGmOffset, const int64_t indicesGmOffset, const int64_t outGmOffset,
    const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    LocalTensor<T1> l1_dy_tensor = mmPingPongIdx ? l1_dy_ping_tensor : l1_dy_pong_tensor;
    LocalTensor<T1> l0_b_tensor = l0_b_dung_tensor;

    commonLoadData2dParamsTrans.repeatTimes = dimGAlign * dimDv / 256;
    commonLoadData2dParamsTrans.srcStride = 1;
    LoadData(l0_b_tensor, l1_dy_tensor, commonLoadData2dParamsTrans);

    Cube5LoadAData(pGmOffset, dyGmOffset, indicesGmOffset, outGmOffset, blkCntOffset);
}
