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
 * \file cube4_sfag.h
 * \brief Formula: dk = s^T * q
 * l0_a： dimG * selectedBlockSize * sizof(T1) (26-28k/28-30k)
 * l0_b： dimG * 512 * sizof(T1) (0k-16k)
 * l0_c： dimG * dimDqk * sizof(flot) (0-12k / 64k-76k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube4Process(
    const int64_t dsGmOffset, const int64_t queryGmOffset, const int64_t queryRopeGmOffset,
    const int64_t indicesGmOffset, const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx,
    const RunInfo &runInfo)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;

    uint32_t blockOffset = M_SPLIT_SIZE / selectedBlockSize; // 128 / 1 = 128

    MMParam mmParam;
    mmParam.singleK = dimG;
    mmParam.isFixOut = true;
    mmParam.isLeftTranspose = true;
    mmParam.isRightTranspose = true;
    mmParam.dstStride = dimDTotal * dimN2;

    int64_t mm4ResOutBaseOffset = runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDtotal +
                                  cBlockIdx * selectedBlockCount * selectedBlockSizeDtotal;
    const bool reloadQuery = !runInfo.noReload && runInfo.isLastBasicBlock;

    LocalTensor<T1> l1_ds_tensor = l1_ds_tensors[ping_pong_flag_l1_ds_];
    uint32_t totalSel = selectedCntOffset * selectedBlockSize;
    if (runInfo.isLastBasicBlock) {
        totalSel = totalSel - selectedBlockSize + runInfo.lastBlockSize;
    }

    const bool useNonOptimizedSmallS2Path = !enableOptimizedScatter && runInfo.isSmallS2;
    int64_t currentDsGmOffset = useNonOptimizedSmallS2Path ? runInfo.mm345GmOffset : dsGmOffset;
    CopyGmToL1(l1_ds_tensor, dsWorkspaceGm[currentDsGmOffset], dimG, totalSel, TOTAL_BLOCK_SIZE);
    for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + selectedCntOffset; mIdx += blockOffset) {
        int32_t l1Offset = (mIdx - blkCntOffset) * selectedBlockSize * dimGAlign;

        mmParam.singleN = perLoopDSize;
        mmParam.singleM = min(selectedBlockSize * blockOffset, totalSel - (mIdx - blkCntOffset) * selectedBlockSize);

        LocalTensor<T1> current_l1_ds_tensor, current_l1_query_tensor;
        current_l1_ds_tensor = l1_ds_tensor[l1Offset];
        int64_t currentQueryOffset;
        int64_t mm4ResOutOffset = mm4ResOutBaseOffset + mIdx * selectedBlockSizeDtotal;

        for (uint32_t dIdx = 0; dIdx < dLoopTimes - 1; dIdx++) {
            LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
            if (unlikely(reloadQuery)) {
                // last block reload query
                WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
                current_l1_query_tensor = l1_common_tensors[ping_pong_flag_l1_common_];
                currentQueryOffset =
                    (useNonOptimizedSmallS2Path ? runInfo.queryGmOffset : queryGmOffset) + dIdx * perLoopDSize;
                CopyGmToL1(current_l1_query_tensor, queryGm[currentQueryOffset], dimG, perLoopDSize, dimDqk);
            } else {
                current_l1_query_tensor = l1_query_tensor[dIdx * dimGAlign * perLoopDSize];
            }

            // l0a复用
            uint32_t l0a_ping_pong_flag = ping_pong_flag_l0a_;
            if (!enableOptimizedScatter) {
                int64_t currentOutGmOffset = mm4ResOutOffset + dIdx * perLoopDSize;
                MmadInnerWithSync<T1>(l0cTensor, current_l1_ds_tensor, current_l1_query_tensor, aL0TensorPingPong,
                                      bL0TensorPingPong, mmParam, l0a_ping_pong_flag, ping_pong_flag_l0b_,
                                      ping_pong_flag_l0c_, dIdx == 0, mm4ResWorkspaceGm[currentOutGmOffset]);
            } else {
                int64_t currentOutGmOffset = mm4ResOutOffset + dIdx * perLoopDSize * mmParam.singleM;
                MmadInnerWithSyncFixpNzOut<T1>(l0cTensor, current_l1_ds_tensor, current_l1_query_tensor,
                                               aL0TensorPingPong, bL0TensorPingPong, mmParam, l0a_ping_pong_flag,
                                               ping_pong_flag_l0b_, ping_pong_flag_l0c_, dIdx == 0,
                                               mm4ResWorkspaceGm[currentOutGmOffset]);
            }
            if (unlikely(reloadQuery)) {
                SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
                UpdatePingPongFlag(ping_pong_flag_l1_common_);
            }
            UpdatePingPongFlag(ping_pong_flag_l0c_);
        }

        mmParam.singleN = tailLoopDSize;
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
        if (unlikely(reloadQuery)) {
            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            current_l1_query_tensor = l1_common_tensors[ping_pong_flag_l1_common_];
            if (useNonOptimizedSmallS2Path) {
                currentQueryOffset =
                    HAS_ROPE ? runInfo.queryRopeGmOffset : runInfo.queryGmOffset + (dLoopTimes - 1) * perLoopDSize;
            } else {
                currentQueryOffset = HAS_ROPE ? queryRopeGmOffset : queryGmOffset + (dLoopTimes - 1) * perLoopDSize;
            }
            GlobalTensor<T1> srcGm = HAS_ROPE ? queryRopeGm[currentQueryOffset] : queryGm[currentQueryOffset];
            int64_t destStride = HAS_ROPE ? dimRope : dimDqk;
            CopyGmToL1(current_l1_query_tensor, srcGm, dimG, tailLoopDSize, destStride);
        } else {
            current_l1_query_tensor = l1_query_tensor[(dLoopTimes - 1) * dimGAlign * perLoopDSize];
        }

        if (!enableOptimizedScatter) {
            int64_t currentOutGmOffset = mm4ResOutOffset + (dLoopTimes - 1) * perLoopDSize;
            MmadInnerWithSync<T1>(l0cTensor, current_l1_ds_tensor, current_l1_query_tensor, aL0TensorPingPong,
                                  bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                  ping_pong_flag_l0c_, true, mm4ResWorkspaceGm[currentOutGmOffset]);
        } else {
            int64_t currentOutGmOffset = mm4ResOutOffset + (dLoopTimes - 1) * perLoopDSize * mmParam.singleM;
            MmadInnerWithSyncFixpNzOut<T1>(l0cTensor, current_l1_ds_tensor, current_l1_query_tensor, aL0TensorPingPong,
                                           bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                           ping_pong_flag_l0c_, true, mm4ResWorkspaceGm[currentOutGmOffset]);
        }
        if (unlikely(reloadQuery)) {
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}
