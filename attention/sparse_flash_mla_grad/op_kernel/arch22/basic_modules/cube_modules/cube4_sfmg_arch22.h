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
 * \file cube4_sfmg_arch22.h
 * \brief Formula: dk = s^T * q
 * l0_a： dimG * selectedBlockSize * sizof(T1) (26-28k/28-30k)
 * l0_b： dimG * 512 * sizof(T1) (0k-16k)
 * l0_c： dimG * dimDqk * sizof(flot) (0-12k / 64k-76k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube4Process(const int64_t dsGmOffset,
                                                                               const int64_t queryGmOffset,
                                                                               const int32_t blkCntOffset,
                                                                               const int32_t mmPingPongIdx,
                                                                               const RunInfo &runInfo)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;

    uint32_t blockOffset = M_SPLIT_SIZE / selectedBlockSize; // 128 / 1 = 128

    MMParam mmParam;
    mmParam.singleM = selectedBlockSize * blockOffset;
    mmParam.singleK = runInfo.curS1g;
    mmParam.isFixOut = true;
    mmParam.isLeftTranspose = true;
    mmParam.isRightTranspose = true;
    mmParam.dstStride = dimDTotal * dimN2;
    int64_t dsL1Offset = 0;
    int64_t dsL1CachedSize = 0;

    int64_t mm4ResOutBaseOffset;
    if constexpr (MODE == SMLAG_SCFA_MODE) {
        if (!runInfo.isOri) {
            mm4ResOutBaseOffset = runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * dimDTotal +
                                  cBlockIdx * selectedBlockCount * dimDTotal;
            mm4ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm4ResAddr);
        } else {
            mm4ResOutBaseOffset = runInfo.selectedKGmOffset - blkCntOffset * selectedBlockSize * dimDTotal;
            mm4ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + usedWorkspaceLen / sizeof(float));
        }
    } else {
        mm4ResOutBaseOffset = runInfo.isOri ?
                                  runInfo.selectedKGmOffset - blkCntOffset * selectedBlockSize * dimDTotal :
                                  runInfo.selectedKGmOffset - blkCntOffset * selectedBlockSize * dimDTotal + dOriKvSize;
    }
    dSL1TotalSize = selectedCntOffset * selectedBlockSize * AlignTo<int64_t>(mmParam.singleK, SIZE_16);
    if (dSL1TotalSize <= 128 * 512) { // 此时dS可全载
        CopyGmToL1(l1_ds_tensor, dsWorkspaceGm[dsGmOffset], mmParam.singleK, selectedCntOffset * selectedBlockSize,
                   singleN);
    }
    for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + selectedCntOffset; mIdx += blockOffset) {
        int64_t l1Offset =
            static_cast<int64_t>(mIdx - blkCntOffset) * selectedBlockSize * AlignTo<int64_t>(mmParam.singleK, SIZE_16);

        mmParam.singleN = perLoopDSize;
        mmParam.singleM = min(selectedBlockSize * blockOffset,
                              selectedCntOffset * selectedBlockSize - (mIdx - blkCntOffset) * selectedBlockSize);

        LocalTensor<T1> current_l1_ds_tensor, l1_query_tensor;
        if (dSL1TotalSize <= 128 * 512) {
            current_l1_ds_tensor = l1_ds_tensor[l1Offset];
        } else {
            if (l1Offset >= dsL1CachedSize) {
                SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT);
                WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT);
                CopyGmToL1(l1_ds_tensor, dsWorkspaceGm[dsGmOffset + (mIdx - blkCntOffset)], mmParam.singleK, 512,
                           singleN); // 重新搬运singleK * 512
                dsL1CachedSize += 512 * AlignTo<int64_t>(mmParam.singleK, SIZE_16);
                dsL1Offset = l1Offset;
            }
            current_l1_ds_tensor = l1_ds_tensor[l1Offset - dsL1Offset];
        }
        int64_t currentQueryOffset;
        int64_t mm4ResOutOffset = mm4ResOutBaseOffset + mIdx * selectedBlockSize * dimDTotal;

        for (uint32_t dIdx = 0; dIdx < dLoopTimes - 1; dIdx++) {
            LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
            l1_query_tensor = l1_query_tensors[ping_pong_flag_l1_query_];

            currentQueryOffset = queryGmOffset + dIdx * N_SPLIT_SIZE;
            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
            CopyGmToL1(l1_query_tensor, queryGm[currentQueryOffset], mmParam.singleK, N_SPLIT_SIZE, dimDqk);

            int64_t currentOutGmOffset = mm4ResOutOffset + dIdx * perLoopDSize;

            // l0a复用
            uint32_t l0a_ping_pong_flag = ping_pong_flag_l0a_;
            if (runInfo.isOri) {
                MmadInnerWithSync<T1, true>(l0cTensor, current_l1_ds_tensor, l1_query_tensor, aL0TensorPingPong,
                                            bL0TensorPingPong, mmParam, l0a_ping_pong_flag, ping_pong_flag_l0b_,
                                            ping_pong_flag_l0c_, dIdx == 0, mm4ResWorkspaceGm[currentOutGmOffset]);
            } else {
                MmadInnerWithSync<T1, MODE != SMLAG_SCFA_MODE>(
                    l0cTensor, current_l1_ds_tensor, l1_query_tensor, aL0TensorPingPong, bL0TensorPingPong, mmParam,
                    l0a_ping_pong_flag, ping_pong_flag_l0b_, ping_pong_flag_l0c_, dIdx == 0,
                    mm4ResWorkspaceGm[currentOutGmOffset]);
            }
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
            UpdatePingPongFlag(ping_pong_flag_l0c_);
            UpdatePingPongFlag(ping_pong_flag_l1_query_);
        }

        mmParam.singleN = tailLoopDSize;
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
        l1_query_tensor = l1_query_tensors[ping_pong_flag_l1_query_];

        currentQueryOffset = queryGmOffset + (dLoopTimes - 1) * K_SPLIT_SIZE;
        GlobalTensor<T1> srcGm = queryGm[currentQueryOffset];
        int64_t destStride = dimDqk;
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
        CopyGmToL1(l1_query_tensor, srcGm, mmParam.singleK, tailLoopDSize, destStride);

        int64_t currentOutGmOffset = mm4ResOutOffset + (dLoopTimes - 1) * perLoopDSize;

        if (runInfo.isOri) {
            MmadInnerWithSync<T1, true>(l0cTensor, current_l1_ds_tensor, l1_query_tensor, aL0TensorPingPong,
                                        bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                        ping_pong_flag_l0c_, true, mm4ResWorkspaceGm[currentOutGmOffset]);
        } else {
            MmadInnerWithSync<T1, MODE != SMLAG_SCFA_MODE>(
                l0cTensor, current_l1_ds_tensor, l1_query_tensor, aL0TensorPingPong, bL0TensorPingPong, mmParam,
                ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true,
                mm4ResWorkspaceGm[currentOutGmOffset]);
        }
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
        UpdatePingPongFlag(ping_pong_flag_l0c_);
        UpdatePingPongFlag(ping_pong_flag_l1_query_);
    }
}
