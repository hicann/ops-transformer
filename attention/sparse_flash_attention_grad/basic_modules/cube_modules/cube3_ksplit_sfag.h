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
 * \file cube3_ksplit_sfag.h
 * \brief K-split dq = ds * K. Overwrite this core's private partial. No atomic.
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube3ProcessKsplitSparse(
    const int64_t dsGmOffset, const int64_t keyGmOffset, const int32_t blkCntOffset, const int64_t lastBlockSize,
    const bool isLastBasicBlock, const AscendC::GlobalTensor<float>& dqPartialGm)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;
    uint32_t blockOffset = K_SPLIT_SIZE / selectedBlockSize;

    MMParam mmParam;
    mmParam.singleM = dimG;
    mmParam.singleN = perLoopDSize;
    mmParam.isRightTranspose = true;
    mmParam.dstStride = dimDTotal;

    LocalTensor<T1> current_l1_ds_tensor, l1_key_tensor;
    uint32_t totalSel = selectedCntOffset * selectedBlockSize;
    if (isLastBasicBlock) {
        totalSel = totalSel - selectedBlockSize + lastBlockSize;
    }
    for (int32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_ & 1];

        if (dIdx == dLoopTimes - 1) {
            mmParam.singleN = tailLoopDSize;
        }
        int64_t currentOutGmOffset = dIdx * perLoopDSize;

        for (int32_t nIdx = blkCntOffset; nIdx < blkCntOffset + selectedCntOffset; nIdx += blockOffset) {
            int32_t l1Offset = (nIdx - blkCntOffset) * selectedBlockSize * dimGAlign;
            bool isFirstLoop = (nIdx == blkCntOffset);
            bool isLastLoop = (nIdx + blockOffset >= blkCntOffset + selectedCntOffset);

            mmParam.singleK =
                min(selectedBlockSize * blockOffset, totalSel - (nIdx - blkCntOffset) * selectedBlockSize);

            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            current_l1_ds_tensor = l1_ds_tensors[ping_pong_flag_l1_ds_][l1Offset];
            l1_key_tensor = l1_common_tensors[ping_pong_flag_l1_common_];
            int64_t currentKeyOffset;
            int64_t srcDstride;
            if constexpr (HAS_ROPE) {
                if (dIdx == dLoopTimes - 1) {
                    currentKeyOffset =
                        keyGmOffset + PER_LOOP_BLOCK_SIZE * dimDqk + (nIdx - blkCntOffset) * selectedBlockSizeDrope;
                    srcDstride = dimRope;
                } else {
                    currentKeyOffset = keyGmOffset + (nIdx - blkCntOffset) * selectedBlockSizeDqk + dIdx * perLoopDSize;
                    srcDstride = dimDqk;
                }
            } else {
                currentKeyOffset = keyGmOffset + (nIdx - blkCntOffset) * selectedBlockSizeDqk + dIdx * perLoopDSize;
                srcDstride = dimDqk;
            }

            CopyGmToL1(l1_key_tensor, selectedKWorkspaceGm[currentKeyOffset], mmParam.singleK, mmParam.singleN,
                       srcDstride);

            mmParam.isOutKFisrt = isFirstLoop;
            mmParam.isFixOut = isLastLoop;

            MmadInnerWithSync<T1, false>(l0cTensor, current_l1_ds_tensor, l1_key_tensor, aL0TensorPingPong,
                                         bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                         ping_pong_flag_l0c_, true, dqPartialGm[currentOutGmOffset]);
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube345ProcessKsplit(
    const RunInfo& runInfo, const int32_t blkCntOffset, const int32_t mmPingPongIdx,
    const AscendC::GlobalTensor<float>& dqPartialGm)
{
    selectedCntOffset = runInfo.actualSelCntOffset;
    cube5Process(runInfo.mm345GmOffset, runInfo.dyGmOffset, runInfo.indicesGmOffset, runInfo.mm5OutGmOffset,
                 blkCntOffset, mmPingPongIdx, runInfo);

    WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT[ping_pong_flag_l1_ds_]);
    cube4Process(runInfo.mm345GmOffset, runInfo.queryGmOffset, runInfo.queryRopeGmOffset, runInfo.indicesGmOffset,
                 runInfo.mm4OutGmOffset, blkCntOffset, mmPingPongIdx, runInfo);
    cube3ProcessKsplitSparse(runInfo.mm345GmOffset, runInfo.selectedKGmOffset, blkCntOffset, runInfo.lastBlockSize,
                             runInfo.isLastBasicBlock, dqPartialGm);
    SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT[ping_pong_flag_l1_ds_]);
    UpdatePingPongFlag(ping_pong_flag_l1_ds_);
}
