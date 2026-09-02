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
 * \file cube3_sfmg_arch22.h
 * \brief Formula: dq = ds * k
 * l0_a： dimG * 512 * sizof(T1) (10k-26k)
 * l0_b： selectedBlockSize * dimDv * sizof(T1) (0-24k / 24k-48k)
 * l0_c： dimG * selectedBlockSize * sizof(T1) (0-4k / 48k-52k)
 */
template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube3Process(
    const int64_t keyGmOffset, const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx,
    const int64_t lastBlockSize, const bool isLastBasicBlock, const RunInfo &runInfo)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;
    uint32_t blockOffset = K_SPLIT_SIZE / selectedBlockSize; // 128 / 1 = 128

    MMParam mmParam;
    mmParam.singleM = runInfo.curS1g;
    mmParam.singleN = perLoopDSize;
    mmParam.singleK = selectedBlockSize * blockOffset;
    mmParam.isRightTranspose = true;
    mmParam.dstStride = dimDTotal;
    int64_t dsL1Offset = 0;
    int64_t dsL1CachedSize = 0;

    LocalTensor<T1> current_l1_ds_tensor, l1_key_tensor;
    // l1_ds_tensor已经在mm4搬完，mm3不需要再搬（当dSL1TotalSize可全载时）
    // 切N
    for (int32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_ & 1];

        if (dIdx == dLoopTimes - 1) {
            mmParam.singleN = tailLoopDSize;
        }
        int64_t currentOutGmOffset = outGmOffset + dIdx * perLoopDSize;
        dsL1CachedSize = 0;
        // selectedBlockSize较小时，nIdx需要+2或者更多
        for (int32_t nIdx = blkCntOffset; nIdx < blkCntOffset + selectedCntOffset; nIdx += blockOffset) {
            int64_t l1Offset = static_cast<int64_t>(nIdx - blkCntOffset) * selectedBlockSize *
                               AlignTo<int64_t>(mmParam.singleM, SIZE_16);
            bool isFirstLoop = (nIdx == blkCntOffset);
            bool isLastLoop = (nIdx + blockOffset >= blkCntOffset + selectedCntOffset);

            uint32_t totalSel = selectedCntOffset * selectedBlockSize;
            if (isLastBasicBlock && isLastLoop) {
                totalSel = totalSel - selectedBlockSize + lastBlockSize;
            }
            mmParam.singleK =
                min(selectedBlockSize * blockOffset, totalSel - (nIdx - blkCntOffset) * selectedBlockSize);

            if (dSL1TotalSize <= 128 * 512) {
                current_l1_ds_tensor = l1_ds_tensor[l1Offset];
            } else {
                if (l1Offset >= dsL1CachedSize) {
                    SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT);
                    WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT);
                    CopyGmToL1(l1_ds_tensor, dsWorkspaceGm[runInfo.mm345GmOffset + (nIdx - blkCntOffset)],
                               mmParam.singleM, 512, singleN);
                    dsL1CachedSize += 512 * AlignTo<int64_t>(mmParam.singleM, SIZE_16);
                    dsL1Offset = l1Offset;
                }
                current_l1_ds_tensor = l1_ds_tensor[l1Offset - dsL1Offset];
            }
            l1_key_tensor = l1_common_tensors[ping_pong_flag_l1_common_];
            int64_t currentKeyOffset =
                keyGmOffset + (nIdx - blkCntOffset) * selectedBlockSize * dimDTotal + dIdx * perLoopDSize;

            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            if constexpr (MODE == SMLAG_SCFA_MODE) {
                if (!runInfo.isOri) {
                    CopyGmToL1(l1_key_tensor, selectedKWorkspaceGm[currentKeyOffset], mmParam.singleK, mmParam.singleN,
                               dimDTotal);
                } else {
                    CopyGmToL1(l1_key_tensor, oriKvGm[currentKeyOffset], mmParam.singleK, mmParam.singleN, dimDTotal);
                }
            } else if (runInfo.isOri) {
                CopyGmToL1(l1_key_tensor, oriKvGm[currentKeyOffset], mmParam.singleK, mmParam.singleN, dimDTotal);
            } else {
                CopyGmToL1(l1_key_tensor, cmpKvGm[currentKeyOffset], mmParam.singleK, mmParam.singleN, dimDTotal);
            }

            mmParam.isOutKFisrt = isFirstLoop;
            mmParam.isFixOut = isLastLoop;

            MmadInnerWithSync<T1, true>(l0cTensor, current_l1_ds_tensor, l1_key_tensor, aL0TensorPingPong,
                                        bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                        ping_pong_flag_l0c_, true, dqWorkspaceGm[currentOutGmOffset]);
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}
