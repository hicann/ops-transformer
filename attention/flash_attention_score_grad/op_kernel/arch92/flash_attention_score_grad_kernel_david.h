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
 * \file flash_attention_score_grad_kernel_david.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_DAVID_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_DAVID_H

#include "flash_attention_score_grad_kernel_base_david.h"
#include "flash_attention_score_grad_common_david.h"
#include "flash_attention_score_grad_tiling_data_david.h"

namespace FagBaseApi {

template <typename CubeBlockType, typename VecBlockType>

class FlashAttentionScoreGradKernel
    : public FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                               CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseClass = FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                                        CubeBlockType, VecBlockType>;
    __aicore__ inline void SetUniqueRunInfo(FagRunInfo& runInfo);
    __aicore__ inline void SetUniqueConstInfo(FagConstInfo& constInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBn2MultiBlk();
    static constexpr uint64_t SYNC_MODE = 4;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueRunInfo(FagRunInfo& runInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueConstInfo(
    FagConstInfo& constInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::Process()
{
    if constexpr (IS_BN2_MULTIBLK) {
        ProcessBn2MultiBlk();
    } else {
        if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] == 0) {
            return;
        }
        int64_t taskId = 0, preRunId = -1, runId = 0, nextRunId = 1;
        bool isLastLoop = false;
        FagRunInfo runInfos[3]; // for ping pong
        int8_t ubBufIdx = 0;
        int64_t innerMm12LoopIdx = 0;
        int64_t nextValidBlockInnerIdx = 0;
        int64_t curLoopIdx = 0; // just for continuous split core
        LocalTensor<T2> mm1ResTensor;
        LocalTensor<T2> mm2ResTensor;
        int64_t blockInnerIdx = 0;

        // Pre-Set
        nextValidBlockInnerIdx =
            this->GetNextValidIdx(runInfos[runId], taskId,
                                  this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx], curLoopIdx);
        blockInnerIdx = nextValidBlockInnerIdx;

        if (blockInnerIdx != -1) {
            this->SetRunInfo(runInfos[runId], taskId, blockInnerIdx);
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_RELEASE[0]);
            }
            if ASCEND_IS_AIV {
                CrossCoreSetFlag<SYNC_MODE, PIPE_V>(this->QKDXV_RES_RELEASE[0]);
            }
            // Pre-Copy
            this->vecBlock.ProcessVec1(runInfos[runId], this->constInfo);
        }
        while (true) {
            isLastLoop = (blockInnerIdx == -1);
            if (!isLastLoop) {
                nextValidBlockInnerIdx =
                    this->GetNextValidIdx(runInfos[nextRunId], taskId, blockInnerIdx + 1, curLoopIdx + 1);
                if (nextValidBlockInnerIdx != -1) {
                    this->SetRunInfo(runInfos[nextRunId], taskId + 1, nextValidBlockInnerIdx);
                }
                if constexpr (IS_SWIZZLE) {
                    curLoopIdx++;
                }
                blockInnerIdx = nextValidBlockInnerIdx;
                uint32_t innerLoopNum =
                    (runInfos[runId].commonRunInfo.s1RealSize + this->VECTOR_BASEM - 1) / this->VECTOR_BASEM;
                for (uint32_t innerLoopIdx = 0; innerLoopIdx < innerLoopNum; innerLoopIdx++) {
                    mm1ResTensor = this->mm1ResBuf[ubBufIdx].template Get<T2>();
                    mm2ResTensor = this->mm2ResBuf[ubBufIdx].template Get<T2>();
                    if ASCEND_IS_AIC {
                        if (innerMm12LoopIdx >= this->UB_BUF_NUM) {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_RELEASE[ubBufIdx]);
                        }
                    }
                    this->cubeBlock.IterateMmQKAndDyV(mm1ResTensor, mm2ResTensor, runInfos[runId], runInfos[nextRunId],
                                                      this->constInfo, innerLoopNum, innerLoopIdx, ubBufIdx);
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_READY[ubBufIdx]);
                    }
                    this->vecBlock.ProcessVec2(runInfos[runId], runInfos[nextRunId], this->constInfo,
                                               blockInnerIdx == -1, mm1ResTensor, mm2ResTensor, innerLoopNum,
                                               innerLoopIdx, ubBufIdx);
                    ubBufIdx = (ubBufIdx + 1 == this->UB_BUF_NUM) ? 0 : ubBufIdx + 1;
                    innerMm12LoopIdx += 1;
                }
            }

            if (taskId > 0) {
                if ASCEND_IS_AIC {
                    runInfos[preRunId].nextS2IdxNoChange = isLastLoop ? false : runInfos[runId].isS2IdxNoChange;
                    this->cubeBlock.IterateMmDsK(runInfos[preRunId], this->constInfo, this->dqWorkSpaceGm); // bmm3
                    this->cubeBlock.IterateMmDsQ(runInfos[preRunId], this->constInfo, this->dkWorkSpaceGm); // bmm4
                    if constexpr (SPLIT_AXIS == BN2) {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->ALL_DONE + 1);
                    }
                    this->cubeBlock.IterateMmPDy(runInfos[preRunId], this->constInfo, this->dvWorkSpaceGm); // bmm5
                    if (isLastLoop) {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->ALL_DONE);
                    }
                }
                if ASCEND_IS_AIV {
                    if constexpr (SPLIT_AXIS == BN2) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(this->ALL_DONE + 1);
                        this->vecBlock.PostCompute(runInfos[preRunId], this->constInfo, this->dqWorkSpaceGm,
                                                   this->dkWorkSpaceGm); // vf5 6
                    }
                }
            }
            if (isLastLoop) {
                break;
            }
            taskId++;
            preRunId = runId;
            runId = nextRunId;
            nextRunId = nextRunId + 1 == 3 ? 0 : nextRunId + 1;
        }
        if (taskId == 0) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->ALL_DONE);
            }
        }
        this->vecBlock.FreeAsync(this->constInfo);
        this->cubeBlock.FreeAlloc(innerMm12LoopIdx);
        this->cubeBlock.FreeBuffer();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessBn2MultiBlk()
{
    if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] == 0) {
        return;
    }
    int64_t taskId = 0, preRunId = -1, runId = 0, nextRunId = 1;
    bool isLastLoop = false;
    FagRunInfo runInfos[3]; // for ping pong
    int8_t ubBufIdx = 0;
    int64_t innerMm12LoopIdx = 0;
    int64_t nextValidBlockInnerIdx = 0;
    int64_t curLoopIdx = 0; // just for continuous split core
    LocalTensor<T2> mm1ResTensor;
    LocalTensor<T2> mm2ResTensor;
    int64_t blockInnerIdx = 0;
    bool hasDqReader = false;

    // Pre-Set
    nextValidBlockInnerIdx = this->GetNextValidIdx(
        runInfos[runId], taskId, this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx], curLoopIdx);
    blockInnerIdx = nextValidBlockInnerIdx;

    if (blockInnerIdx != -1) {
        this->SetRunInfo(runInfos[runId], taskId, blockInnerIdx);
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_RELEASE[0]);
        }
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_V>(this->QKDXV_RES_RELEASE[0]);
        }
        // Pre-Copy
        this->vecBlock.ProcessVec1(runInfos[runId], this->constInfo);
    }
    while (true) {
        isLastLoop = (blockInnerIdx == -1);
        if (!isLastLoop) {
            nextValidBlockInnerIdx =
                this->GetNextValidIdx(runInfos[nextRunId], taskId + 1, blockInnerIdx + 1, curLoopIdx + 1);
            if (nextValidBlockInnerIdx != -1) {
                this->SetRunInfo(runInfos[nextRunId], taskId + 1, nextValidBlockInnerIdx);
            }
            if constexpr (IS_SWIZZLE) {
                curLoopIdx++;
            }
            blockInnerIdx = nextValidBlockInnerIdx;
            uint32_t innerLoopNum =
                (runInfos[runId].commonRunInfo.s1RealSize + this->VECTOR_BASEM - 1) / this->VECTOR_BASEM;
            for (uint32_t innerLoopIdx = 0; innerLoopIdx < innerLoopNum; innerLoopIdx++) {
                mm1ResTensor = this->mm1ResBuf[ubBufIdx].template Get<T2>();
                mm2ResTensor = this->mm2ResBuf[ubBufIdx].template Get<T2>();
                if ASCEND_IS_AIC {
                    if (innerMm12LoopIdx >= this->UB_BUF_NUM) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_RELEASE[ubBufIdx]);
                    }
                }
                this->cubeBlock.IterateMmQKAndDyV(mm1ResTensor, mm2ResTensor, runInfos[runId], runInfos[nextRunId],
                                                  this->constInfo, innerLoopNum, innerLoopIdx, ubBufIdx);
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->QKDXV_RES_READY[ubBufIdx]);
                }
                this->vecBlock.ProcessVec2(runInfos[runId], runInfos[nextRunId], this->constInfo, blockInnerIdx == -1,
                                           mm1ResTensor, mm2ResTensor, innerLoopNum, innerLoopIdx, ubBufIdx);
                ubBufIdx = (ubBufIdx + 1 == this->UB_BUF_NUM) ? 0 : ubBufIdx + 1;
                innerMm12LoopIdx += 1;
            }
        }

        if (taskId > 0) {
            runInfos[preRunId].nextS2IdxNoChange = isLastLoop ? false : runInfos[runId].isS2IdxNoChange;
            if ASCEND_IS_AIC {
                if (runInfos[preRunId].isFirstS2Outer && hasDqReader) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(this->BN2_BLK_V3_TO_C3);
                }
            }
            this->cubeBlock.IterateMmDsK(runInfos[preRunId], this->constInfo, this->dqWorkSpaceGm); // bmm3
            if (runInfos[preRunId].isLastS2Outer) {
                hasDqReader = true;
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->BN2_BLK_C3_TO_V3);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(this->BN2_BLK_C3_TO_V3);
                }
                this->vecBlock.template ProcessMulsAndCast<T2, DQ_IDX>(this->dqWorkSpaceGm, this->constInfo,
                                                                       runInfos[preRunId]);
                if ASCEND_IS_AIV {
                    if (!isLastLoop) {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(this->BN2_BLK_V3_TO_C3);
                    }
                }
            }
            this->cubeBlock.IterateMmDsQ(runInfos[preRunId], this->constInfo, this->dkWorkSpaceGm); // bmm4
            if (!runInfos[preRunId].nextS2IdxNoChange) {
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->BN2_BLK_C4_TO_V3);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(this->BN2_BLK_C4_TO_V3);
                }
                this->vecBlock.template ProcessMulsAndCast<T2, DK_IDX>(this->dkWorkSpaceGm, this->constInfo,
                                                                       runInfos[preRunId]);
            }
            this->cubeBlock.IterateMmPDy(runInfos[preRunId], this->constInfo, this->dvWorkSpaceGm); // bmm5
            if ASCEND_IS_AIC {
                if (isLastLoop) {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->ALL_DONE);
                }
            }
            if (!runInfos[preRunId].nextS2IdxNoChange) {
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(this->BN2_BLK_C5_TO_V3);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(this->BN2_BLK_C5_TO_V3);
                }
                this->vecBlock.template ProcessMulsAndCast<T2, DV_IDX>(this->dvWorkSpaceGm, this->constInfo,
                                                                       runInfos[preRunId]);
            }
        }
        if (isLastLoop) {
            break;
        }
        taskId++;
        preRunId = runId;
        runId = nextRunId;
        nextRunId = nextRunId + 1 == 3 ? 0 : nextRunId + 1;
    }
    this->vecBlock.FreeAsync(this->constInfo);
    this->cubeBlock.FreeAlloc(innerMm12LoopIdx);
    this->cubeBlock.FreeBuffer();
}
} // namespace FagBaseApi

#endif
