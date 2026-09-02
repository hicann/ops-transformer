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
 * \file cube3.h
 * \brief Formula: dq = ds * k
 * l0_a： dimG * 512 * sizof(T1) (10k-26k)
 * l0_b： selectedBlockSize * dimDv * sizof(T1) (0-24k / 24k-48k)
 * l0_c： dimG * selectedBlockSize * sizof(T1) (0-4k / 48k-52k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::LoadBData(
    const int64_t dsGmOffset, const int64_t keyGmOffset, const int64_t indicesGmOffset, const int64_t outGmOffset,
    const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    LocalTensor<T1> l1_key_tensor = mmPingPongIdx ? l1_key_ping_tensor : l1_key_pong_tensor;
    LocalTensor<T1> l0_a_tensor = l0_a_cube3_tensor;
    LocalTensor<T1> l0_b_tensor;
    LocalTensor<float> l0_c_tensor = l0_c_ping_tensor;

    WAIT_FLAG(M, MTE1, eventIdPing);
    WAIT_FLAG(M, MTE1, eventIdPong);
    SET_FLAG(M, MTE1, eventIdPing);
    SET_FLAG(M, MTE1, eventIdPong);

    for (int32_t nIdx = blkCntOffset; nIdx < blkCntOffset + 8; nIdx++) {
        bool flag = (nIdx % 2 == 0);
        int32_t topkIdx = topkIndicesGm[indicesGmOffset].GetValue(nIdx);
        int32_t startS2Idx = topkIdx * selectedBlockSize * dimN2 * dimDqk;
        int32_t l1Offset = (nIdx - blkCntOffset) * selectedBlockSize * dimDqk;
        int32_t l0AOffset = (nIdx - blkCntOffset) * dimGAlign * selectedBlockSize;
        bool isFirstLoop = (nIdx == blkCntOffset);
        bool isLastLoop = (nIdx == blkCntOffset + 8 - 1);
        uint32_t eventId = flag ? eventIdPing : eventIdPong;
        l0_b_tensor = flag ? l0_b_ding_tensor : l0_b_dong_tensor;

        WAIT_FLAG(M, MTE1, eventId);

        commonLoadData2dParamsTrans.repeatTimes = dimDqk / 16;
        commonLoadData2dParamsTrans.srcStride = selectedBlockSize / 16;
        for (int32_t i = 0; i < selectedBlockSize / 16; i++) {
            LoadData(l0_b_tensor[i * dimDqk * 16], l1_key_tensor[l1Offset + i * 256], commonLoadData2dParamsTrans);
        }

        SET_FLAG(MTE1, M, eventId);
        WAIT_FLAG(MTE1, M, eventId);

        MmadParams madParams;
        madParams.m = (dimG == 1 ? 2 : dimG);
        madParams.n = dimDqk;
        madParams.k = selectedBlockSize;
        madParams.cmatrixInitVal = isFirstLoop ? true : false;
        Mmad(l0_c_tensor, l0_a_tensor[l0AOffset], l0_b_tensor, madParams);

        SET_FLAG(M, MTE1, eventId);

        if (isLastLoop) {
            SET_FLAG(M, FIX, eventId);
            WAIT_FLAG(M, FIX, eventId);

            FixpipeParamsV220 fixParams;
            fixParams.mSize = dimG;
            fixParams.nSize = dimDqk;
            fixParams.srcStride = dimGAlign;
            fixParams.dstStride = dimDqk;
            AscendC::SetAtomicType<float>();
            Fixpipe<float, float, CFG_ROW_MAJOR>(dqWorkspaceGm[outGmOffset], l0_c_tensor, fixParams);
            AscendC::SetAtomicNone();
        }
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube3Process(
    const int64_t dsGmOffset, const int64_t keyGmOffset, const int64_t indicesGmOffset, const int64_t outGmOffset,
    const int32_t blkCntOffset, const int32_t mmPingPongIdx)
{
    commonLoadData2dParamsNoTranspose.repeatTimes = 512 / SIZE_16;
    commonLoadData2dParamsNoTranspose.srcStride = dimGAlign / SIZE_16;
    for (int32_t i = 0; i < dimGAlign / SIZE_16; i++) {
        LoadData(l0_a_cube3_tensor[i * 512 * SIZE_16], l1_ds_tensor[i * BLOCK_SIZE], commonLoadData2dParamsNoTranspose);
        LoadBData(dsGmOffset, keyGmOffset, indicesGmOffset, outGmOffset, blkCntOffset, mmPingPongIdx);
    }
}
