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
 * \file cube3_sfag.h
 * \brief Formula: dq = ds * k
 * l0_a： dimG * 512 * sizof(T1) (10k-26k)
 * l0_b： selectedBlockSize * dimDv * sizof(T1) (0-24k / 24k-48k)
 * l0_c： dimG * selectedBlockSize * sizof(T1) (0-4k / 48k-52k)
 */
template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube3ProcessSparse(
    const int64_t dsGmOffset, const int64_t keyGmOffset, const int64_t indicesGmOffset, const int64_t outGmOffset,
    const int32_t blkCntOffset, const int32_t mmPingPongIdx, const int64_t lastBlockSize, const bool isLastBasicBlock)
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
        int64_t currentOutGmOffset = outGmOffset + dIdx * perLoopDSize;

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

            if (blkCntOffset == 0) {
                MmadInnerWithSync<T1, false>(l0cTensor, current_l1_ds_tensor, l1_key_tensor, aL0TensorPingPong,
                                             bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                             ping_pong_flag_l0c_, true, dqWorkspaceGm[currentOutGmOffset]);
            } else {
                MmadInnerWithSync<T1, true>(l0cTensor, current_l1_ds_tensor, l1_key_tensor, aL0TensorPingPong,
                                            bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                            ping_pong_flag_l0c_, true, dqWorkspaceGm[currentOutGmOffset]);
            }
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube3ProcessDense(const int32_t blkCntOffset,
                                                                                    const int32_t mmPingPongIdx,
                                                                                    const int64_t lastBlockSize,
                                                                                    const bool isLastBasicBlock,
                                                                                    const RunInfo &runInfo)
{
    const int64_t dsGmOffset = runInfo.mm345GmOffset;
    const int64_t keyGmOffset = runInfo.keyGmOffset;
    const int64_t indicesGmOffset = runInfo.indicesGmOffset;
    const int64_t outGmOffset = runInfo.mm3OutGmOffset;

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
        int64_t currentOutGmOffset = outGmOffset + dIdx * perLoopDSize;
        for (int32_t nIdx = blkCntOffset; nIdx < blkCntOffset + selectedCntOffset; nIdx += blockOffset) {
            int32_t l1Offset = (nIdx - blkCntOffset) * selectedBlockSize * dimGAlign;
            bool isFirstLoop = (nIdx == blkCntOffset);
            bool isLastLoop = (nIdx + blockOffset >= blkCntOffset + selectedCntOffset);

            mmParam.singleK =
                min(selectedBlockSize * blockOffset, totalSel - (nIdx - blkCntOffset) * selectedBlockSize);

            current_l1_ds_tensor = l1_ds_tensors[ping_pong_flag_l1_ds_][l1Offset];
            l1_key_tensor = l1_common_tensors[ping_pong_flag_l1_common_];
            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);

            int64_t currentKeyOffset = 0;
            if (dIdx != dLoopTimes - 1) {
                currentKeyOffset = keyGmOffset + (blkCntOffset * dimN2 + nIdx - blkCntOffset) * selectedBlockSizeDqk +
                                   dIdx * perLoopDSize;
                CopyGmToL1(l1_key_tensor, keyGm[currentKeyOffset], mmParam.singleK, mmParam.singleN, dimDqk);
            } else {
                if constexpr (HAS_ROPE) {
                    currentKeyOffset =
                        runInfo.keyRopeGmOffset + (blkCntOffset * dimN2 + nIdx - blkCntOffset) * selectedBlockSizeDrope;
                    CopyGmToL1(l1_key_tensor, keyRopeGm[currentKeyOffset], mmParam.singleK, mmParam.singleN, dimRope);
                } else {
                    currentKeyOffset = keyGmOffset +
                                       (blkCntOffset * dimN2 + nIdx - blkCntOffset) * selectedBlockSizeDqk +
                                       dIdx * perLoopDSize;
                    CopyGmToL1(l1_key_tensor, keyGm[currentKeyOffset], mmParam.singleK, mmParam.singleN, dimDqk);
                }
            }

            mmParam.isOutKFisrt = isFirstLoop;
            mmParam.isFixOut = isLastLoop;

            if (blkCntOffset == 0) {
                MmadInnerWithSync<T1, false>(l0cTensor, current_l1_ds_tensor, l1_key_tensor, aL0TensorPingPong,
                                             bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                             ping_pong_flag_l0c_, true, dqWorkspaceGm[currentOutGmOffset]);
            } else {
                MmadInnerWithSync<T1, true>(l0cTensor, current_l1_ds_tensor, l1_key_tensor, aL0TensorPingPong,
                                            bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                            ping_pong_flag_l0c_, true, dqWorkspaceGm[currentOutGmOffset]);
            }
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube3Process(
    const int64_t dsGmOffset, const int64_t keyGmOffset, const int64_t indicesGmOffset, const int64_t outGmOffset,
    const int32_t blkCntOffset, const int32_t mmPingPongIdx, const int64_t lastBlockSize, const bool isLastBasicBlock,
    const RunInfo &runInfo)
{
    if (!runInfo.isSmallS2) {
        cube3ProcessSparse(dsGmOffset, keyGmOffset, indicesGmOffset, outGmOffset, blkCntOffset, mmPingPongIdx,
                           lastBlockSize, isLastBasicBlock);
    } else {
        cube3ProcessDense(blkCntOffset, mmPingPongIdx, lastBlockSize, isLastBasicBlock, runInfo);
    }
}
