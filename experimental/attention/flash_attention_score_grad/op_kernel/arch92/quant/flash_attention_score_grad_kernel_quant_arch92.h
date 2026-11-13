/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_QUANT_ARCH92_H_
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_QUANT_ARCH92_H_

#include "flash_attention_score_grad_quant_context_arch92.h"

namespace FagBaseApi {
namespace quant {
// Full 128x128 S/dP ping-pong with independent quantized P/dS outputs.
// Short-KV balanced scheduling retains deterministic gradient ownership.
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreGradKernelQuant : public FlashAttentionScoreGradQuantContext<CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;

    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift,
                                GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen, GM_ADDR deqScaleQ,
                                GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dsScale,
                                GM_ADDR pScale, GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                GM_ADDR dpse, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR workspace,
                                FagTilingType ordTilingData, TPipe* pipeIn)
    {
        if ASCEND_IS_AIV {
            this->vBlockIdx = GetBlockIdx();
            this->cBlockIdx = this->vBlockIdx;
            this->vSubBlockIdx = 0;
        } else {
            this->cBlockIdx = GetBlockIdx();
        }
        this->tilingData = ordTilingData;
        this->pipe = pipeIn;
        this->InitCVCommonGlobalBuffer(dq, dk, dv, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, dsScale, pScale,
                                       workspace);
        this->SetConstInfo();
        coreCount = ordTilingData->s1s2BNGS1S2BaseParams.coreNum;
        uint64_t bn = this->constInfo.bSize * this->constInfo.n2Size;
        balanced = FagBaseApi::quant::UseBalanced(bn, this->constInfo.commonConstInfo.s1Size,
                                                  this->constInfo.commonConstInfo.s2Size, coreCount);
        localBarrier = balanced;
        if (balanced) {
            uint64_t total = bn * this->constInfo.s1Outer;
            taskBegin = FagBaseApi::quant::TaskBegin(total, this->cBlockIdx, coreCount);
            taskEnd = FagBaseApi::quant::TaskBegin(total, this->cBlockIdx + 1, coreCount);
            // Both tiling keys preserve the original quant path's ordered accumulation.
            uint64_t offset = FagBaseApi::quant::PartialOffset(ordTilingData->postTilingData.sfmgWorkSpaceOffset, bn,
                                                               this->constInfo.commonConstInfo.s1Size);
            partialK.SetGlobalBuffer((__gm__ float*)(workspace + offset));
            partialV.SetGlobalBuffer((__gm__ float*)(workspace + offset) +
                                     coreCount * 2 * this->constInfo.commonConstInfo.s2Size * 128);
        }
        if ASCEND_IS_AIC {
            this->constInfo.pScaleD = 1.0f / this->pScaleGm.GetValue(0);
            this->constInfo.dsScaleD = 1.0f / this->dsScaleGm.GetValue(0);
        }
        InitPipelineBuffer();
        this->vecBlock.SetVecBlockParams(pipeIn, ordTilingData, this->vBlockIdx, this->cBlockIdx, 0, this->constInfo);
        this->vecBlock.InitUbBuffer(this->commonTensors);
        this->vecBlock.InitStatsBuffer(this->commonTensors);
        this->vecBlock.InitGlobalBuffer(value, dy, y, softmaxMax, softmaxSum, deqScaleQ, deqScaleK, deqScaleV,
                                        deqScaleDy, dq, dk, dv, workspace,
                                        ordTilingData->postTilingData.sfmgWorkSpaceOffset);
        this->cubeBlock.SetCubeBlockParams(pipeIn, ordTilingData, &this->l1BufferManager);
        this->cubeBlock.InitCubeBuffer(this->constInfo);
        this->cubeBlock.InitGlobalBuffer(query, key, value, dy);
    }

    __aicore__ inline void Process()
    {
        InitPipelineEvents();
        if ASCEND_IS_AIV {
            this->vecBlock.AllocEventID();
        } else {
            this->cubeBlock.AllocEventID();
        }

        int64_t loopMax =
            balanced ?
                Ceil<int64_t>(this->constInfo.bSize * this->constInfo.n2Size * this->constInfo.s1Outer, coreCount) :
                this->CalDeterMaxLoopNum();
        InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[0]);
        InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[1]);
        int64_t taskId = 0;
        int64_t nextRound;
        int64_t nextBlock;
        FagRunInfo runInfos[2]{};
        {
            if (this->constInfo.s2Outer == 1 && !balanced) {
                ++this->cBlockIdx;
                InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0,
                                   this->nextCoreFirstBlockCoordinateInfo);
                LookupTask(0, loopMax, nextRound, nextBlock, taskId, this->nextCoreFirstBlockCoordinateInfo,
                           runInfos[0]);
                --this->cBlockIdx;
            }
        }
        LookupTask(0, loopMax, nextRound, nextBlock, taskId, this->coordinateInfos[0], runInfos[0]);

        // Four logical phases per round, with vector-side completion bridges
        // retaining the original ordered gradient accumulation.
        // The extra round drains the last P column and all four pending dQ rows.
        for (int64_t round = 0; round <= loopMax; ++round) {
            FagRunInfo& current = runInfos[taskId & 1];
            FagRunInfo& previous = runInfos[(taskId + 1) & 1];
            bool hasCurrent = round < loopMax && round == nextRound && nextBlock >= 0;
            if (hasCurrent) {
                LookupTask(round + 1, loopMax, nextRound, nextBlock, taskId, this->coordinateInfos[(taskId + 1) & 1],
                           previous);
                this->SetRunInfo(current, previous, taskId, this->coordinateInfos[taskId & 1],
                                 this->coordinateInfos[(taskId + 1) & 1]);
                CompleteRunInfo(current);
                if ASCEND_IS_AIV {
                    this->vecBlock.PrepareStats(taskId, current.maxsumOffset, current.commonRunInfo.s1RealSize);
                }
                current.quantRunInfo.isDqCompleted = false;
                current.quantRunInfo.isDkvCompleted = false;
            }

            for (uint32_t column = 0; column < FagBaseApi::quant::QUANT_INNER_LOOP_NUM; ++column) {
                if ASCEND_IS_AIV {
                    if (!localBarrier)
                        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
                }
                ProcessColumn(current, previous, hasCurrent, column);
                if ASCEND_IS_AIV {
                    // Includes dQ/dV/dK Fixpipe completion before the next ordered phase.
                    if (!localBarrier)
                        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
                }
            }

            if (hasCurrent) {
                ++taskId;
            }
        }

        if ASCEND_IS_AIV {
            this->vecBlock.FreeEventID();
        } else {
            this->cubeBlock.FreeEventID();
        }
        DrainPipelineEvents();
    }

private:
    __aicore__ inline void LookupTask(uint32_t round, uint32_t maxRounds, int64_t& nextRound, int64_t& nextBlock,
                                      int64_t task, CoordinateInfo& coordinate, FagRunInfo& run)
    {
        if (!balanced) {
            this->CalDeterIndex(round, maxRounds, nextRound, nextBlock, task, coordinate, run);
            return;
        }
        uint64_t id = taskBegin + round;
        if (id >= taskEnd) {
            coordinate.batchId = -1;
            nextBlock = -1;
            nextRound = maxRounds;
            return;
        }
        uint64_t head = id / this->constInfo.s1Outer;
        coordinate.batchId = head / this->constInfo.n2Size;
        coordinate.n2Idx = head % this->constInfo.n2Size;
        coordinate.gIdx = 0;
        coordinate.s1Idx = id % this->constInfo.s1Outer;
        coordinate.s2Idx = 0;
        nextBlock = id;
        nextRound = round;
    }

    __aicore__ inline void CompleteRunInfo(FagRunInfo& run)
    {
        if ASCEND_IS_AIC {
            int64_t qScaleOffset = this->GetDeqScaleQOffset(run);
            auto q = this->deqScaleQGm.GetValue(qScaleOffset);
            int64_t kScaleOffset = this->GetDeqScaleKOffset(run);

            auto k = this->deqScaleKGm.GetValue(kScaleOffset);
            auto dy = this->deqScaleDyGm.GetValue(qScaleOffset);
            run.quantRunInfo.dsScaleDMulDeqScaleK = this->constInfo.dsScaleD * k;
            run.quantRunInfo.dsScaleDMulDeqScaleQ = this->constInfo.dsScaleD * q;
            run.quantRunInfo.pScaleDMulDeqScaleDy = this->constInfo.pScaleD * dy;
            auto& next = this->nextCoreFirstBlockCoordinateInfo;
            run.quantRunInfo.kvNeedAtomic =
                (this->constInfo.s2Outer != 1 && run.isValueReuse) ||
                (this->constInfo.s2Outer == 1 && !run.isFirstProcessBlock &&
                 ((run.commonRunInfo.boIdx == next.batchId && run.commonRunInfo.n2oIdx == next.n2Idx) ||
                  run.isValueReuse));
        }

        if (balanced) {
            uint64_t head = run.commonRunInfo.boIdx * this->constInfo.n2Size + run.commonRunInfo.n2oIdx;
            uint64_t first = Max<uint64_t>(taskBegin, head * this->constInfo.s1Outer);
            uint64_t id = head * this->constInfo.s1Outer + run.commonRunInfo.s1oIdx;
            // First task overwrites this core/head slot; later tasks accumulate in core order.
            run.quantRunInfo.kvNeedAtomic = id != first;
        }
    }
    static constexpr uint64_t SYNC_MODE = 4;
    // Full: SP free 0/1, gradient free 2/3/4, SP ready 5/6, gradient ready 13/14/15.
    // L1 credits/ready and the all-AIV barrier retain baseline IDs 8..12.
    static constexpr uint8_t GRAD_FREE_BASE = 2;
    static constexpr uint8_t GRAD_READY_BASE = 13;
    static_assert(GRAD_READY_BASE + FagBaseApi::quant::GRAD_CREDITS <= 16, "Quant pipeline flag ID overflow");
    LocalTensor<CALC_TYPE> spSlots[FagBaseApi::quant::SP_SLOTS];
    LocalTensor<CALC_TYPE> dpSlots[FagBaseApi::quant::SP_SLOTS];
    uint32_t gradSlot = 0;
    bool balanced = false, localBarrier = false;
    uint64_t coreCount = 0, taskBegin = 0, taskEnd = 0;
    GlobalTensor<float> partialK, partialV;
    LocalTensor<INPUT_TYPE> pOutputs[FagBaseApi::quant::SP_SLOTS], dsOutputs[FagBaseApi::quant::SP_SLOTS];

    __aicore__ inline void InitPipelineBuffer()
    {
        this->l1BufferManager.Init(this->pipe, L1_MAX_SIZE);
        this->pL1Buf.Init(this->l1BufferManager, FagBaseApi::quant::QUANT_SP_SLAB_ELEM * sizeof(INPUT_TYPE));
        for (uint32_t i = 0; i < FagBaseApi::quant::QUANT_INNER_LOOP_NUM; ++i) {
            for (uint32_t j = 0; j < FagBaseApi::quant::QUANT_INNER_LOOP_NUM; ++j) {
                this->dsL1Buf[i][j].Init(this->l1BufferManager,
                                         FagBaseApi::quant::QUANT_SP_SLAB_ELEM * sizeof(INPUT_TYPE));
            }
        }
        this->pipe->InitBuffer(this->vecQue, FagBaseApi::quant::MAIN_UB_BYTES);
        uint32_t offset = 0;
        for (uint32_t slot = 0; slot < FagBaseApi::quant::SP_SLOTS; ++slot) {
            constexpr uint32_t spViewElems = FagBaseApi::quant::SP_ELEMS;
            spSlots[slot] = this->vecQue.template GetWithOffset<CALC_TYPE>(spViewElems, offset);
            offset += FagBaseApi::quant::SP_ELEMS * 4;
            dpSlots[slot] = this->vecQue.template GetWithOffset<CALC_TYPE>(FagBaseApi::quant::SP_ELEMS, offset);
            offset += FagBaseApi::quant::SP_ELEMS * 4;
        }
        {
            for (uint32_t slot = 0; slot < FagBaseApi::quant::SP_SLOTS; ++slot) {
                pOutputs[slot] =
                    this->vecQue.template GetWithOffset<INPUT_TYPE>(FagBaseApi::quant::PDS_OUTPUT_ELEMS, offset);
                offset += FagBaseApi::quant::PDS_OUTPUT_ELEMS;
                dsOutputs[slot] =
                    this->vecQue.template GetWithOffset<INPUT_TYPE>(FagBaseApi::quant::PDS_OUTPUT_ELEMS, offset);
                offset += FagBaseApi::quant::PDS_OUTPUT_ELEMS;
            }
        }
        this->commonTensors =
            this->vecQue.template GetWithOffset<CALC_TYPE>(FagBaseApi::quant::STATS_BYTES / 4, offset);
        offset += FagBaseApi::quant::STATS_BYTES;
    }

    __aicore__ inline void InitPipelineEvents()
    {
        if ASCEND_IS_AIV {
            for (uint32_t slot = 0; slot < FagBaseApi::quant::SP_SLOTS; ++slot) {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(slot);
            }
            {
                for (uint32_t slot = 0; slot < FagBaseApi::quant::GRAD_CREDITS; ++slot) {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(GRAD_FREE_BASE + slot);
                }
            }
            if (!localBarrier)
                CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
        } else {
            for (uint32_t i = 0; i < 2; ++i) {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_P_FLAG);
            }
            for (uint32_t i = 0; i < 8U; ++i) {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_DS_FLAG);
            }
        }
    }

    __aicore__ inline void DrainPipelineEvents()
    {
        if ASCEND_IS_AIC {
            for (uint32_t slot = 0; slot < FagBaseApi::quant::SP_SLOTS; ++slot) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(slot);
            }
            {
                for (uint32_t slot = 0; slot < FagBaseApi::quant::GRAD_CREDITS; ++slot) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(GRAD_FREE_BASE + slot);
                }
            }
        } else {
            for (uint32_t i = 0; i < 2; ++i) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_UB2L1_P_FLAG);
            }
            for (uint32_t i = 0; i < 8U; ++i) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_UB2L1_DS_FLAG);
            }
            if (!localBarrier)
                CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
        }
    }

    __aicore__ inline LocalTensor<INPUT_TYPE> GetDs(FagRunInfo& runInfo, uint32_t row, uint32_t column)
    {
        if (runInfo.commonRunInfo.taskIdMod2 == 0) {
            return this->dsL1Buf[row][column].Get().template GetTensor<INPUT_TYPE>();
        }
        return this->dsL1Buf[column][row].Get().template GetTensor<INPUT_TYPE>();
    }

    __aicore__ inline void ProduceSP(FagRunInfo& runInfo, uint32_t row, uint32_t column)
    {
        runInfo.quantRunInfo.s1Idx = row;
        runInfo.quantRunInfo.s2Idx = column;
        bool valid = row < runInfo.quantRunInfo.innerS1LoopNum && column < runInfo.quantRunInfo.innerS2LoopNum;
        {
            uint32_t slot = row & 1;
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(slot);
                if (valid)
                    this->cubeBlock.IterateMmDsP(spSlots[slot], dpSlots[slot], this->constInfo, runInfo);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(5 + slot);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(5 + slot);
                {
                    auto p = this->pL1Buf.Get().template GetTensor<INPUT_TYPE>();
                    if (valid) {
                        this->vecBlock.ProcessFullOutput(p, GetDs(runInfo, row, column), spSlots[slot], dpSlots[slot],
                                                         pOutputs[slot], dsOutputs[slot], slot, this->constInfo,
                                                         runInfo);
                    } else {
                        // Invalid/tail rows still return one input credit and consume L1 credits.
                        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(slot);
                        if (slot == 0) {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_UB2L1_P_FLAG);
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_UB2L1_DS_FLAG);
                        }
                    }
                }
            }
        }
        if ASCEND_IS_AIV {
            if (row & 1)
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_PDS_TO_DKV_FLAG);
            if (column == 3)
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_PDS_TO_DKV_FLAG_TAIL);
            if (column == 3 && row == 3)
                this->vecBlock.FinishStats();
        }
    }

    __aicore__ inline void AccumulateDQ(FagRunInfo& runInfo, uint32_t row, uint32_t part)
    {
        runInfo.quantRunInfo.s1Idx = row;
        runInfo.quantRunInfo.s2Idx = part;
        if ASCEND_IS_AIC {
            if (part == 2) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_PDS_TO_DKV_FLAG_TAIL);
            }
            if (row < runInfo.quantRunInfo.innerS1LoopNum && part < runInfo.quantRunInfo.innerS2LoopNum) {
                runInfo.quantRunInfo.isDqFixOut = part == 2;
                // MMAD writes L0C; TransferGradient writes GM after the matching completion credit.
                this->cubeBlock.IterateMmDsK(GetDs(runInfo, row, part), GetDs(runInfo, row, part + 1), this->constInfo,
                                             runInfo);
            }
            // The old column-3 dK still owns this pair; its first MMAD releases it.
            if (!(part == 2 && row == 0)) {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_DS_FLAG);
            }
        }
    }

    template <bool isDK>
    __aicore__ inline void AccumulateDKV(FagRunInfo& runInfo, uint32_t column, uint32_t part)
    {
        runInfo.quantRunInfo.s1Idx = part;
        runInfo.quantRunInfo.s2Idx = column;
        if ASCEND_IS_AIC {
            bool valid = column < runInfo.quantRunInfo.innerS2LoopNum && part < runInfo.quantRunInfo.innerS1LoopNum;
            if constexpr (!isDK) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_PDS_TO_DKV_FLAG);
                LocalTensor<INPUT_TYPE> p0 = this->pL1Buf.Get().template GetTensor<INPUT_TYPE>();
                LocalTensor<INPUT_TYPE> p1 = this->pL1Buf.Get().template GetTensor<INPUT_TYPE>();
                if (valid) {
                    runInfo.quantRunInfo.isDvFixOut = part == 2;
                    this->cubeBlock.IterateMmPDy(p0, p1, this->constInfo, runInfo);
                }
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_P_FLAG);
            } else {
                if (valid) {
                    runInfo.quantRunInfo.isDkFixOut = part == 2;
                    this->cubeBlock.IterateMmDsQ(GetDs(runInfo, part, column), GetDs(runInfo, part + 1, column),
                                                 this->constInfo, runInfo);
                }
                if (part == 0 && column == 3) {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_DS_FLAG);
                }
            }
        }
    }

    template <uint64_t mode>
    __aicore__ inline void TransferGradient(FagRunInfo& run)
    {
        bool valid = mode == 0 ? run.quantRunInfo.s1Idx < run.quantRunInfo.innerS1LoopNum :
                                 run.quantRunInfo.s2Idx < run.quantRunInfo.innerS2LoopNum;
        bool privatePartial = balanced && mode != 0;
        int64_t offset = mode == 0 ? run.quantRunInfo.qInnerOffset[run.quantRunInfo.s1Idx] :
                                     run.quantRunInfo.kvInnerOffset[run.quantRunInfo.s2Idx];
        auto gm = mode == 0 ? this->dqWorkSpaceGm : (mode == 1 ? this->dvWorkSpaceGm : this->dkWorkSpaceGm);
        uint64_t stride = this->constInfo.commonConstInfo.n2GD;
        if (privatePartial) {
            uint64_t head = run.commonRunInfo.boIdx * this->constInfo.n2Size + run.commonRunInfo.n2oIdx;
            uint64_t segment = this->cBlockIdx * 2 + head - taskBegin / this->constInfo.s1Outer;
            offset = segment * this->constInfo.commonConstInfo.s2Size * 128 + run.quantRunInfo.s2Idx * 128 * 128;
            gm = mode == 1 ? partialV : partialK;
            stride = 128;
        }
        bool atomic = mode == 0 ? this->constInfo.s2Outer != 1 : run.quantRunInfo.kvNeedAtomic;

        {
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(GRAD_FREE_BASE + gradSlot);
                if (valid) {
                    uint32_t rows = mode == 0 ? run.quantRunInfo.innerS1RealSize[run.quantRunInfo.s1Idx] :
                                                run.quantRunInfo.innerS2RealSize[run.quantRunInfo.s2Idx];
                    float scale = mode == 0 ? run.quantRunInfo.dsScaleDMulDeqScaleK :
                                              (mode == 1 ? run.quantRunInfo.pScaleDMulDeqScaleDy :
                                                           run.quantRunInfo.dsScaleDMulDeqScaleQ);
                    if constexpr (mode == 0 && IsSameType<OUTDTYPE, bfloat16_t>::value) {
                        if (FagBaseApi::quant::UseDqFixBf16(this->constInfo.s2Outer)) {
                            // A single KV block owns the complete dQ sum; no atomic or
                            // FP32 workspace round trip is needed for this output.
                            this->cubeBlock.CopyDqFixBf16(this->dqGm, offset, rows, stride,
                                                          scale * static_cast<float>(this->constInfo.scaleValue));
                        } else {
                            this->cubeBlock.template CopyGradientGM<mode>(gm, offset, rows, stride, scale, atomic);
                        }
                    } else {
                        this->cubeBlock.template CopyGradientGM<mode>(gm, offset, rows, stride, scale, atomic);
                    }
                }
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(GRAD_READY_BASE + gradSlot);
            } else {
                // Retain one receiving pipeline per flag and bridge FIX completion
                // through V before acknowledging the GM write on MTE3.
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(GRAD_READY_BASE + gradSlot);
                // Bridge FIX completion through V to the GM-write stage/barrier.
                SetFlag<HardEvent::V_MTE3>(4 + gradSlot);
                WaitFlag<HardEvent::V_MTE3>(4 + gradSlot);
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(GRAD_FREE_BASE + gradSlot);
            }
            gradSlot = (gradSlot + 1) % FagBaseApi::quant::GRAD_CREDITS;
        }
    }

    __aicore__ inline void ProcessColumn(FagRunInfo& current, FagRunInfo& previous, bool hasCurrent, uint32_t column)
    {
        bool hasDQ = !previous.quantRunInfo.isDqCompleted;
        bool hasDKV = column == 0 ? !previous.quantRunInfo.isDkvCompleted : hasCurrent;
        FagRunInfo& kvRun = column == 0 ? previous : current;
        uint32_t kvColumn = column == 0 ? 3 : column - 1;

        if (hasCurrent) {
            ProduceSP(current, 0, column);
        }
        if (hasDQ) {
            AccumulateDQ(previous, column, 0);
        }
        if (hasCurrent) {
            ProduceSP(current, 1, column);
        }
        if (hasDQ) {
            AccumulateDQ(previous, column, 2);
            TransferGradient<0>(previous); // No delayed dQ Fixpipe.
            previous.quantRunInfo.isDqCompleted = column == 3;
        }

        if (hasDKV) {
            AccumulateDKV<false>(kvRun, kvColumn, 0);
        }
        if (hasCurrent) {
            ProduceSP(current, 2, column);
        }
        if (hasDKV) {
            AccumulateDKV<false>(kvRun, kvColumn, 2);
            TransferGradient<1>(kvRun); // dV leaves L0C before either dK MMAD.
        }
        if (hasCurrent) {
            ProduceSP(current, 3, column);
        }
        if (hasDKV) {
            AccumulateDKV<true>(kvRun, kvColumn, 0);
            AccumulateDKV<true>(kvRun, kvColumn, 2);
            TransferGradient<2>(kvRun);
            if (column == 0) {
                previous.quantRunInfo.isDkvCompleted = true;
            }
        }
    }
};
} // namespace quant
} // namespace FagBaseApi
#endif
