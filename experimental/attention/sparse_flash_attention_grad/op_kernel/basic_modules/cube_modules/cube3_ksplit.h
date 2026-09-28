/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube3_ksplit.h
 * \brief ksplit 确定性路径专用：mm3（dq = ds * k）写本核私有 partial [G, Dtotal] fp32，
 *        非原子覆盖写（每行整 buffer 重写，无需 atomic add）。
 *        仅被 sparse_flash_attention_grad_ksplit_det.h 使用，不影响 1000~1111 既有路径。
 */

// mm3 切 K 私有 partial 版（sparse 数据源：selectedK workspace）
template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube3ProcessKsplitSparse(
    const int64_t dsGmOffset, const int64_t keyGmOffset, const int32_t blkCntOffset, const int64_t lastBlockSize,
    const bool isLastBasicBlock, const GlobalTensor<float> &dqPartialGm)
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

            // 非原子覆盖写本核私有 partial（每行整 buffer 重写）
            MmadInnerWithSync<T1, false>(l0cTensor, current_l1_ds_tensor, l1_key_tensor, aL0TensorPingPong,
                                         bL0TensorPingPong, mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_,
                                         ping_pong_flag_l0c_, true, dqPartialGm[currentOutGmOffset]);
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}

// mm3 切 K 私有 partial 版（dense 数据源：isSmallS2，直接从 keyGm 连续读）
template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube3ProcessKsplitDense(
    const int32_t blkCntOffset, const int64_t lastBlockSize, const bool isLastBasicBlock, const RunInfo &runInfo,
    const GlobalTensor<float> &dqPartialGm)
{
    const int64_t keyGmOffset = runInfo.keyGmOffset;

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

            // 非原子覆盖写本核私有 partial（每行整 buffer 重写）
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
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube3ProcessKsplit(
    const int64_t dsGmOffset, const int64_t keyGmOffset, const int32_t blkCntOffset, const int64_t lastBlockSize,
    const bool isLastBasicBlock, const RunInfo &runInfo, const GlobalTensor<float> &dqPartialGm)
{
    if (!runInfo.isSmallS2) {
        cube3ProcessKsplitSparse(dsGmOffset, keyGmOffset, blkCntOffset, lastBlockSize, isLastBasicBlock, dqPartialGm);
    } else {
        cube3ProcessKsplitDense(blkCntOffset, lastBlockSize, isLastBasicBlock, runInfo, dqPartialGm);
    }
}

// cube345 切 K 变体：mm5/mm4 与既有路径一致（写 mm4Res/mm5Res workspace），
// 仅 mm3 改写本核私有 dq partial（dqPartialGm 已定位到本核本 parity 的 [G, Dtotal] 起点）。
template <typename T1>
__aicore__ inline __attribute__((always_inline)) void CubeOp<T1>::cube345ProcessKsplit(
    const RunInfo &runInfo, const int32_t blkCntOffset, const int32_t mmPingPongIdx,
    const GlobalTensor<float> &dqPartialGm)
{
    selectedCntOffset = runInfo.actualSelCntOffset;
    cube5Process(runInfo.mm345GmOffset, runInfo.dyGmOffset, runInfo.indicesGmOffset, runInfo.mm5OutGmOffset,
                 blkCntOffset, mmPingPongIdx, runInfo);

    WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT[ping_pong_flag_l1_ds_]);
    cube4Process(runInfo.mm345GmOffset, runInfo.queryGmOffset, runInfo.queryRopeGmOffset, runInfo.indicesGmOffset,
                 runInfo.mm4OutGmOffset, blkCntOffset, mmPingPongIdx, runInfo);
    cube3ProcessKsplit(runInfo.mm345GmOffset, runInfo.selectedKGmOffset, blkCntOffset, runInfo.lastBlockSize,
                       runInfo.isLastBasicBlock, runInfo, dqPartialGm);
    SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DS_EVENT[ping_pong_flag_l1_ds_]);
    UpdatePingPongFlag(ping_pong_flag_l1_ds_);
}
