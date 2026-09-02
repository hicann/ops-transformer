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
 * \file cube5_sfag.h
 * \brief Formula: dv = p^T * dy
 * l0_a： dimG * dimDqk * sizof(T1) (30k-32k/32-34k)
 * l0_b： dimG * 512 * sizof(T1) (48-64k)
 * l0_c： dimG * dimDqk * sizof(float) (0-12k / 64k-76k)
 */
template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube5Process(
    const int64_t pGmOffset, const int64_t dyGmOffset, const int64_t indicesGmOffset, const int64_t outGmOffset,
    const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    uint32_t dLoopTimes = (dimDv + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t blockOffset = M_SPLIT_SIZE / selectedBlockSize; // 128 / 1 = 128

    MMParam mmParam;
    mmParam.singleN = perLoopDSize;
    mmParam.singleK = dimG;
    mmParam.isFixOut = true;
    mmParam.isLeftTranspose = true;
    mmParam.isRightTranspose = true;
    mmParam.dstStride = dimDv * dimN2;

    int64_t mm5ResOutBaseOffset =
        runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSize * dimDv +
        cBlockIdx * selectedBlockCount * selectedBlockSize * dimDv;
    const bool reloadDy = !runInfo.noReload && runInfo.isLastBasicBlock;

    uint32_t totalSel = selectedCntOffset * selectedBlockSize;
    if (runInfo.isLastBasicBlock) {
        totalSel = totalSel - selectedBlockSize + runInfo.lastBlockSize;
    }

    const bool useNonOptimizedSmallS2Path = !enableOptimizedScatter && runInfo.isSmallS2;
    int64_t currentPGmOffset = useNonOptimizedSmallS2Path ? runInfo.mm345GmOffset : pGmOffset;
    for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + selectedCntOffset; mIdx += blockOffset) {
        LocalTensor<T1> current_l1_dy_tensor, l1_p_tensor;
        l1_p_tensor = l1_p_tensors[ping_pong_flag_l1_p_];
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_P_EVENT[ping_pong_flag_l1_p_]);

        mmParam.singleM = min(selectedBlockSize * blockOffset, totalSel - (mIdx - blkCntOffset) * selectedBlockSize);

        int64_t mm5ResOutOffset = mm5ResOutBaseOffset + mIdx * selectedBlockSize * dimDv;
        CopyGmToL1(l1_p_tensor, pWorkspaceGm[currentPGmOffset + (mIdx - blkCntOffset) * selectedBlockSize], dimG,
                   mmParam.singleM, PER_LOOP_BLOCK_SIZE);
        for (int32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
            if (unlikely(reloadDy)) {
                // last block reload dy
                WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
                current_l1_dy_tensor = l1_common_tensors[ping_pong_flag_l1_common_];
                int64_t currentDyGmOffset =
                    (useNonOptimizedSmallS2Path ? runInfo.dyGmOffset : dyGmOffset) + dIdx * perLoopDSize;
                CopyGmToL1(current_l1_dy_tensor, attentionGradGm[currentDyGmOffset], dimG, perLoopDSize, dimDv);
            } else {
                current_l1_dy_tensor = l1_dy_tensor[dIdx * perLoopDSize * dimGAlign];
            }
            LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_ & 1];

            // l0a复用
            uint32_t l0a_ping_pong_flag = ping_pong_flag_l0a_;
            if (!enableOptimizedScatter) {
                int64_t currentOutGmOffset = mm5ResOutOffset + dIdx * perLoopDSize;
                MmadInnerWithSync<T1>(l0cTensor, l1_p_tensor, current_l1_dy_tensor, aL0TensorPingPong,
                                      bL0TensorPingPong, mmParam, l0a_ping_pong_flag, ping_pong_flag_l0b_,
                                      ping_pong_flag_l0c_, dIdx == 0, mm5ResWorkspaceGm[currentOutGmOffset]);
            } else {
                int64_t currentOutGmOffset = mm5ResOutOffset + dIdx * perLoopDSize * mmParam.singleM;
                MmadInnerWithSyncFixpNzOut<T1>(l0cTensor, l1_p_tensor, current_l1_dy_tensor, aL0TensorPingPong,
                                               bL0TensorPingPong, mmParam, l0a_ping_pong_flag, ping_pong_flag_l0b_,
                                               ping_pong_flag_l0c_, dIdx == 0, mm5ResWorkspaceGm[currentOutGmOffset]);
            }
            UpdatePingPongFlag(ping_pong_flag_l0c_);
            if (unlikely(reloadDy)) {
                SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
                UpdatePingPongFlag(ping_pong_flag_l1_common_);
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_P_EVENT[ping_pong_flag_l1_p_]);
        UpdatePingPongFlag(ping_pong_flag_l1_p_);
        UpdatePingPongFlag(ping_pong_flag_l0a_);
    }
}
