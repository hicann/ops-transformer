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
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube2Process(const int64_t dyGmOffset, const int64_t valueGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    uint32_t dLoopTimes = (dimDqk + 127) / K_SPLIT_SIZE;
    uint32_t perLoopDSize = K_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDqk - (dLoopTimes - 1) * perLoopDSize;
    uint32_t blockOffset = N_SPLIT_SIZE / selectedBlockSize; // 128 / 1 = 128

    MMParam mmParam;
    mmParam.singleM = runInfo.curS1g;
    mmParam.singleN = MODE == SMLAG_SCFA_MODE ? selectedBlockSize * blockOffset : selectedCntOffset;
    mmParam.singleK = perLoopDSize;
    mmParam.isFixOut = false;
    mmParam.dstStride = singleN;

    LocalTensor<T1> l1_dy_tensor = l1_dy_tensors[mmPingPongIdx];
    CopyGmToL1(l1_dy_tensor, attentionGradGm[dyGmOffset], mmParam.singleM, dimDv, dimDv); // [Align(s1g, 16) * (dimDv, 16), 16]
    for (int32_t nIdx = blkCntOffset; nIdx < blkCntOffset + selectedCntOffset; nIdx+=blockOffset) {
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_ & 1];
        int64_t mm2WorkspaceGmOffset =  outGmOffset + (nIdx - blkCntOffset) * selectedBlockSize;

        LocalTensor<T1> current_l1_dy_tensor, l1_v_tensor;
        int64_t currentDyOffset = 0;
        int64_t currentVOffset;
        // query node @ key node
        for (int32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
            mmParam.isOutKFisrt = dIdx == 0;
            mmParam.isFixOut = dIdx == dLoopTimes - 1;
            l1_v_tensor = l1_common_tensors[ping_pong_flag_l1_common_];
            
            currentDyOffset = dIdx * AlignTo<int64_t>(mmParam.singleM, SIZE_16) * K_SPLIT_SIZE;
            currentVOffset = valueGmOffset + (nIdx - blkCntOffset) * selectedBlockSize * dimDTotal + dIdx * K_SPLIT_SIZE;
            
            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            if constexpr (MODE == SMLAG_SCFA_MODE) {
                if (!runInfo.isOri) {
                    CopyGmToL1(l1_v_tensor, selectedKWorkspaceGm[currentVOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
                } else {
                    CopyGmToL1(l1_v_tensor, oriKvGm[currentVOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
                }
            } else if (runInfo.isOri) {
                CopyGmToL1(l1_v_tensor, oriKvGm[currentVOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
            } else {
                CopyGmToL1(l1_v_tensor, cmpKvGm[currentVOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
            }
            current_l1_dy_tensor = l1_dy_tensor[currentDyOffset];

            MmadInnerWithSync<T1>(l0cTensor, current_l1_dy_tensor, l1_v_tensor,
                    aL0TensorPingPong, bL0TensorPingPong,
                    mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, mm2WorkspaceGm[mm2WorkspaceGmOffset]);
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}