/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_KERNEL_INT8_H_
#define QUANT_FLASH_ATTN_KERNEL_INT8_H_
#include "quant_flash_attn_block_cube_int8.h"
#include "quant_flash_attn_block_vec_int8.h"
#include "quant_flash_attn_scheduler_int8.h"

using namespace AscendC;

namespace BaseApi {
using namespace Int8GqaTnd;
// PATH is fixed by the tiling key. Keep layout scheduling and scale-cache policies at compile time while sharing
// task decoding and the pipeline.
template <bool RETURN_LSE, Int8GqaPath PATH, uint16_t CONFIG, bool D64_CROSS_Q = false, uint16_t D64_VEC_FEATURES = 0U,
          bool D64_REUSE_MASK = false, bool D64_OPT = false, bool LONG_TND_CROSS_Q = false,
          bool LONG_TND_STATIC_D256 = false, bool WINDOWED = false>
class QuantFlashAttnKernelGqaInt8 {
    static_assert(!D64_CROSS_Q || (PATH != Int8GqaPath::BNSD && CONFIG == QFA_INT8_CFG_D64_M512),
                  "D64 cross-Q requires the M512/N512 pipeline");
    static_assert(!LONG_TND_CROSS_Q || (PATH == Int8GqaPath::TND &&
                                        (CONFIG == QFA_INT8_CFG_RESIDENT_Q || CONFIG == QFA_INT8_CFG_D128_M512)),
                  "Long TND overlap requires the validated D128/D256 pipeline");
    static constexpr Int8GqaCubeStrategy CUBE_STRATEGY =
        CONFIG == QFA_INT8_CFG_SHARED_QP || CONFIG == QFA_INT8_CFG_ISOLATED_SHARED_QP ? Int8GqaCubeStrategy::SHARED_QP :
        CONFIG == QFA_INT8_CFG_RESIDENT_Q ? Int8GqaCubeStrategy::RESIDENT_Q :
                                            Int8GqaCubeStrategy::M512;
    using Cube = QuantFlashAttnBlockCubeGqaInt8<
        PATH, CUBE_STRATEGY,
        CONFIG == QFA_INT8_CFG_D96_M512_FIXPIPE || CONFIG == QFA_INT8_CFG_D96_M256_CAUSAL_MHA || D64_OPT,
        D64_OPT || (PATH == Int8GqaPath::TND && CONFIG == QFA_INT8_CFG_D64_M512 && D64_CROSS_Q),
        LONG_TND_STATIC_D256 ? QFA_HEAD_DIM_256 : 0U>;
    static constexpr uint32_t STATIC_HEAD_DIM =
        CONFIG == QFA_INT8_CFG_D96_M512_FIXPIPE || CONFIG == QFA_INT8_CFG_D96_M256_CAUSAL_MHA ? QFA_HEAD_DIM_96 :
        LONG_TND_STATIC_D256                                                                  ? QFA_HEAD_DIM_256 :
        CONFIG == QFA_INT8_CFG_D64_M512                                                       ? QFA_HEAD_DIM_64 :
        CONFIG == QFA_INT8_CFG_D128_M512                                                      ? QFA_HEAD_DIM_128 :
                                                                                                0U;
    static constexpr uint32_t STATIC_QUERY_TILE =
        CONFIG == QFA_INT8_CFG_D96_M256_CAUSAL_MHA ? QFA_PV_OVERLAY_MIN_QUERY_TILE :
        CONFIG >= QFA_INT8_CFG_D64_M512 && CONFIG != QFA_INT8_CFG_ISOLATED_SHARED_QP &&
                CONFIG != QFA_INT8_CFG_DYNAMIC_MHA ?
                                                     QFA_LARGE_QUERY_TILE_ROWS :
                                                     0U;
    using Vector = QuantFlashAttnBlockVecGqaInt8<
        RETURN_LSE, STATIC_HEAD_DIM, STATIC_QUERY_TILE, PATH,
        CONFIG == QFA_INT8_CFG_D96_M512_FIXPIPE || CONFIG == QFA_INT8_CFG_D96_M256_CAUSAL_MHA, D64_VEC_FEATURES,
        D64_REUSE_MASK, PATH == Int8GqaPath::TND && CONFIG == QFA_INT8_CFG_RESIDENT_Q, WINDOWED>;
    // Every path that can retain K scales must share the same producer ownership protocol, including the generic
    // causal M512 pipelines.
    static constexpr bool K_SCALE_PROTOCOL =
        CONFIG == QFA_INT8_CFG_SHARED_QP || CONFIG == QFA_INT8_CFG_RESIDENT_Q || D64_CROSS_Q ||
        (PATH != Int8GqaPath::BNSD &&
         (CONFIG == QFA_INT8_CFG_D64_M512 || CONFIG == QFA_INT8_CFG_D128_M512 || CONFIG == QFA_INT8_CFG_GENERIC_M512 ||
          CONFIG == QFA_INT8_CFG_D96_M512_FIXPIPE || CONFIG == QFA_INT8_CFG_DYNAMIC_MHA ||
          CONFIG == QFA_INT8_CFG_D96_M256_CAUSAL_MHA));
    // Each cross-core FIFO stage reserves one flag per pipeline slot.
    static constexpr uint32_t K_SCALE_FREE_FLAG = 0U;
    static constexpr uint32_t MM2_READY_FLAG_BASE = 1U;
    static constexpr uint32_t MM1_READY_FLAG_BASE = MM2_READY_FLAG_BASE + QFA_PIPELINE_SLOTS;
    static constexpr uint32_t VEC1_READY_FLAG_BASE = MM1_READY_FLAG_BASE + QFA_PIPELINE_SLOTS;
    static constexpr uint32_t K_SCALE_READY_FLAG = VEC1_READY_FLAG_BASE + QFA_PIPELINE_SLOTS;
    static constexpr uint32_t SYNC_MODE = 2U; // Synchronize one AIC with its paired AIVs.
    static constexpr uint32_t VEC2_STAGE_DELAY = QFA_PIPELINE_SLOTS - 1U;
    struct RunInfo {
        bool isValid, isFirst, isLast;
        uint32_t batch, kvHead, mStart, mSize;
        uint32_t sq, sk, qStart, kStart;
        uint32_t nStart, nSize, dataSlot, flagSlot;
        uint32_t initialState, sumBank;
    };

public:
    __aicore__ static inline void Process(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dequantScaleQuery,
                                          GM_ADDR dequantScaleKey, GM_ADDR dequantScaleValue, GM_ADDR pScale,
                                          GM_ADDR cuSeqLensQ, GM_ADDR cuSeqLensKv, GM_ADDR metadata, GM_ADDR attnOut,
                                          GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
    {
        constexpr bool PACKED = PATH == Int8GqaPath::TND;
        const auto* payload = reinterpret_cast<const __gm__ QuantFlashAttnTilingPayloadInt8*>(tiling);
        GET_TILING_DATA_WITH_STRUCT(QuantFlashAttnTilingDataInt8, cfg, tiling);
        if constexpr (!WINDOWED)
            cfg.winLeft = cfg.winRight = QFA_UNBOUNDED_WINDOW;

        uint32_t coreId = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId /= QFA_VECTOR_CORES_PER_CUBE;
        }
        if (cfg.hasMetadata) {
            const auto* schedule = reinterpret_cast<const __gm__ uint32_t*>(metadata);
            cfg.usedCoreNum = schedule[Int8GqaTnd::QFA_METADATA_USED_CORES];
            cfg.blockM = schedule[Int8GqaTnd::QFA_METADATA_BLOCK_M];
            cfg.blockN = schedule[Int8GqaTnd::QFA_METADATA_BLOCK_N];
        }
        if (coreId < cfg.usedCoreNum) {
            TPipe pipe;
            __gm__ uint8_t* workspaceCore =
                GetUserWorkspace(workspace) + static_cast<uint64_t>(coreId) * cfg.coreWorkspaceBytes;
            GlobalTensor<int32_t> metaGm;
            metaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(metadata));
            const uint32_t dataSlotCount =
                cfg.blockM >= QFA_PV_OVERLAY_MIN_QUERY_TILE ? QFA_PV_OVERLAY_SLOTS : QFA_PIPELINE_SLOTS;
            const uint32_t tasksPerBatch = cfg.numHeadsKv * cfg.totalTilesM;
            // Match host tilesForM: causal BNSD tasks end at the query-head boundary, so one head's last query
            // cannot expand the next head's early queries to the full KV range. No new tiling ABI is needed.
            const bool headAligned = PATH == Int8GqaPath::BNSD && cfg.layoutQ == QFA_LAYOUT_BNSD &&
                                     cfg.layoutKv == QFA_LAYOUT_BNSD && cfg.layoutOut == QFA_LAYOUT_BNSD &&
                                     cfg.maskMode != QFA_MASK_NONE &&
                                     QfaUseHeadAlignedTiles(cfg.seqLenQ, cfg.seqLenKv, cfg.headDim, cfg.blockM);
            const uint32_t queryTilesPerHead = (static_cast<uint64_t>(cfg.seqLenQ) + cfg.blockM - 1U) / cfg.blockM;
            // Retain each path's task order and causal work balancing.
            const bool tokenMajor =
                PATH == Int8GqaPath::BNSD ?
                    false :
                    (PATH == Int8GqaPath::BSND ? cfg.layoutQ == QFA_LAYOUT_BSND : cfg.layoutQ != QFA_LAYOUT_BNSD);
            BaseApi::Int8GqaTnd::TndTaskScheduler packedScheduler;
            uint64_t taskBegin, taskEnd;
            uint32_t taskStep;
            if (cfg.hasMetadata && cfg.layoutQ <= QFA_LAYOUT_BSND) {
                const uint32_t offset =
                    BaseApi::Int8GqaTnd::QfaMetadataRangeOffset(cfg.batchSize) + QFA_METADATA_CORE_RANGE_WORDS * coreId;
                taskBegin = metaGm.GetValue(offset);
                taskEnd = metaGm.GetValue(offset + 1U);
                taskStep = metaGm.GetValue(BaseApi::Int8GqaTnd::QFA_METADATA_TASK_STEP);
            } else {
                const auto range = BaseApi::Int8GqaTnd::QfaDenseTaskRange(cfg, coreId);
                taskBegin = range.begin;
                taskEnd = range.end;
                taskStep = range.step;
                if (cfg.layoutQ <= QFA_LAYOUT_BSND && payload->useCoreRanges != QFA_CORE_RANGES_NONE) {
                    taskBegin = payload->coreTaskBegin[coreId];
                    taskEnd = payload->coreTaskBegin[coreId + 1U];
                    taskStep = 1U;
                }
                if constexpr (PACKED) {
                    if (cfg.layoutQ == QFA_LAYOUT_TND) {
                        packedScheduler.Init(cfg, cuSeqLensQ, cuSeqLensKv, metadata, coreId, tiling);
                        taskBegin = packedScheduler.Begin();
                        taskEnd = packedScheduler.End();
                        taskStep = 1U;
                    }
                }
            }
            // Generic tasks overlap only with independent P/PV slots. The two-slot overlaid path keeps its existing
            // ownership schedule. The D96 M512 configuration is the cache-free BSND D96 long-prefill specialization.
            const bool crossTaskPipeline =
                (((CONFIG == QFA_INT8_CFG_SHARED_QP || CONFIG == QFA_INT8_CFG_RESIDENT_Q) &&
                  cfg.blockM < QFA_PV_OVERLAY_MIN_QUERY_TILE) ||
                 LONG_TND_CROSS_Q || (D64_CROSS_Q && CONFIG == QFA_INT8_CFG_D64_M512) ||
                 (PATH == Int8GqaPath::BNSD && CONFIG != QFA_INT8_CFG_ISOLATED_SHARED_QP &&
                  CONFIG != QFA_INT8_CFG_DYNAMIC_MHA) ||
                 (PATH == Int8GqaPath::BSND &&
                  (CONFIG == QFA_INT8_CFG_D96_M512_FIXPIPE || CONFIG == QFA_INT8_CFG_D96_M256_CAUSAL_MHA)) ||
                 (PATH != Int8GqaPath::BNSD &&
                  (CONFIG == QFA_INT8_CFG_D64_M512 || CONFIG == QFA_INT8_CFG_D128_M512 ||
                   CONFIG == QFA_INT8_CFG_GENERIC_M512) &&
                  cfg.maskMode != QFA_MASK_NONE && cfg.groupSize == 1U &&
                  cfg.headDimAlign <= QFA_LARGE_FP32_HEAD_DIM_THRESHOLD)) &&
                taskBegin < taskEnd && taskEnd - taskBegin > taskStep;
            Cube cube;
            Vector vec;
            if ASCEND_IS_AIC {
                cube.Init(query, key, value, dequantScaleValue, pScale, workspaceCore, cfg, &pipe);
            } else {
                vec.Init(workspaceCore, attnOut, dequantScaleQuery, dequantScaleKey, dequantScaleValue, pScale,
                         softmaxLse, cfg, &pipe);
            }
            RunInfo taskCache[QFA_PIPELINE_SLOTS];
            if (crossTaskPipeline) {
                for (uint32_t i = 0U; i < QFA_PIPELINE_SLOTS; ++i)
                    taskCache[i].isValid = false;
            }
            uint64_t pipelineLoop = 0U;
            uint32_t initialState = 0U;
            for (uint64_t task = taskBegin; task < taskEnd; task += taskStep) {
                uint64_t taskId = task;
                if constexpr (CONFIG == QFA_INT8_CFG_D96_M512_FIXPIPE || CONFIG == QFA_INT8_CFG_D96_M256_CAUSAL_MHA ||
                              D64_OPT) {
                    const bool mappedTask = cfg.hasMetadata ?
                                                metaGm.GetValue(BaseApi::Int8GqaTnd::QFA_METADATA_TASK_MODE) ==
                                                    BaseApi::Int8GqaTnd::QFA_METADATA_MAPPED :
                                                payload->useCoreRanges == QFA_CORE_RANGES_TASK_MAP;
                    if (mappedTask && !(PACKED && cfg.layoutQ == QFA_LAYOUT_TND)) {
                        if (cfg.hasMetadata) {
                            taskId = metaGm.GetValue(BaseApi::Int8GqaTnd::QfaMetadataMapOffset(
                                                         cfg.batchSize, QfaUnpackMetadataCapacity(cfg.hasMetadata)) +
                                                     task);
                        } else {
                            const auto* mapped =
                                reinterpret_cast<const __gm__ QuantFlashAttnTilingQBlockPayloadInt8*>(tiling);
                            taskId = mapped->taskIds[task];
                        }
                    }
                }
                uint32_t b = 0U, kvHead = 0U, mStart = 0U;
                uint32_t headLocal = 0U, tokenStart = 0U;
                uint32_t sq = cfg.seqLenQ, sk = cfg.seqLenKv, qStart = 0U, kStart = 0U;
                if (PACKED && cfg.layoutQ == QFA_LAYOUT_TND) {
                    if (!packedScheduler.Decode(taskId, b, kvHead, mStart, sq, sk, qStart, kStart))
                        continue;
                } else {
                    b = taskId / tasksPerBatch;
                    kvHead = (taskId / cfg.totalTilesM) % cfg.numHeadsKv;
                    const uint32_t localTile = taskId % cfg.totalTilesM;
                    if (headAligned) {
                        headLocal = localTile / queryTilesPerHead;
                        tokenStart = (localTile % queryTilesPerHead) * cfg.blockM;
                    } else {
                        mStart = localTile * cfg.blockM;
                    }
                    if (cfg.hasMetadata) {
                        qStart =
                            metaGm.GetValue(BaseApi::Int8GqaTnd::QfaMetadataSequenceOffset(b) + QFA_METADATA_Q_START);
                        kStart =
                            metaGm.GetValue(BaseApi::Int8GqaTnd::QfaMetadataSequenceOffset(b) + QFA_METADATA_K_START);
                        sq = metaGm.GetValue(BaseApi::Int8GqaTnd::QfaMetadataSequenceOffset(b) + QFA_METADATA_Q_LENGTH);
                        sk = metaGm.GetValue(BaseApi::Int8GqaTnd::QfaMetadataSequenceOffset(b) + QFA_METADATA_K_LENGTH);
                    }
                }
                if (sq > cfg.seqLenQ || sk > cfg.seqLenKv || (sq > 0U && sk == 0U))
                    continue;
                if (cfg.layoutQ == QFA_LAYOUT_TND && (static_cast<uint64_t>(qStart) + sq > cfg.totalTokensQ ||
                                                      static_cast<uint64_t>(kStart) + sk > cfg.totalTokensKv))
                    continue;
                uint32_t mSize;
                if (headAligned) {
                    if (tokenStart >= sq)
                        continue;
                    mStart = headLocal * sq + tokenStart;
                    mSize = cfg.blockM < sq - tokenStart ? cfg.blockM : sq - tokenStart;
                } else {
                    if (mStart >= cfg.groupSize * sq)
                        continue;
                    mSize = cfg.blockM < cfg.groupSize * sq - mStart ? cfg.blockM : cfg.groupSize * sq - mStart;
                }
                uint32_t nTiles;
                {
                    uint32_t visible = sk;
                    if (cfg.maskMode != QFA_MASK_NONE) {
                        const uint32_t first = mStart % sq;
                        const uint32_t last = tokenMajor ? (mStart + mSize - 1U) / cfg.groupSize :
                                                           (first + mSize > sq ? sq - 1U : first + mSize - 1U);
                        const int64_t limit = static_cast<int64_t>(last) + 1 + sk - sq;
                        visible = limit <= 0 ? 0U : static_cast<uint32_t>(limit);
                    }
                    nTiles = (visible + cfg.blockN - 1U) / cfg.blockN;
                }
                if (crossTaskPipeline) {
                    if (nTiles == 0U) {
                        DrainTasks(pipelineLoop, taskCache, cube, vec);
                        if ASCEND_IS_AIV {
                            vec.SetSequence(sq, sk, qStart, kStart);
                            vec.BeginTask(b, kvHead, mStart, mSize);
                            vec.WriteZero();
                        }
                        continue;
                    }
                    for (uint32_t n = 0U; n < nTiles; ++n) {
                        const uint32_t nStart = n * cfg.blockN;
                        taskCache[pipelineLoop % QFA_PIPELINE_SLOTS] = {
                            true,
                            n == 0U,
                            n + 1U == nTiles,
                            b,
                            kvHead,
                            mStart,
                            mSize,
                            sq,
                            sk,
                            qStart,
                            kStart,
                            nStart,
                            cfg.blockN < sk - nStart ? cfg.blockN : sk - nStart,
                            static_cast<uint32_t>(pipelineLoop % dataSlotCount),
                            static_cast<uint32_t>(pipelineLoop % QFA_PIPELINE_SLOTS),
                            initialState,
                            initialState ^ (nTiles & 1U)};
                        ExecuteTask(pipelineLoop, taskCache, cube, vec);
                        ++pipelineLoop;
                    }
                    initialState ^= nTiles & 1U;
                } else {
                    if ASCEND_IS_AIC {
                        cube.SetSequence(sq, sk, qStart, kStart);
                        cube.BeginTask(b, kvHead, mStart, mSize);
                    } else {
                        vec.SetSequence(sq, sk, qStart, kStart);
                        vec.BeginTask(b, kvHead, mStart, mSize);
                    }
                    if (nTiles == 0U) {
                        if ASCEND_IS_AIV
                            vec.WriteZero();
                        continue;
                    }
                    for (uint32_t step = 0U; step < nTiles + VEC2_STAGE_DELAY; ++step) {
                        if (step < nTiles) {
                            const uint32_t nStart = step * cfg.blockN;
                            const uint32_t nSize = cfg.blockN < sk - nStart ? cfg.blockN : sk - nStart;
                            if ASCEND_IS_AIC {
                                if (PATH == Int8GqaPath::BNSD || cube.NeedKScale(nStart))
                                    CrossCoreWaitFlag(K_SCALE_READY_FLAG);
                                cube.ComputeMm1(nStart, nSize, step % dataSlotCount);
                                if ((PATH == Int8GqaPath::BNSD || !cube.CachedKScales(nStart)) &&
                                    step + dataSlotCount < nTiles)
                                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(K_SCALE_FREE_FLAG);
                                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(MM1_READY_FLAG_BASE + step % QFA_PIPELINE_SLOTS);
                            } else {
                                if (PATH == Int8GqaPath::BNSD || vec.NeedKScale(nStart)) {
                                    if ((PATH == Int8GqaPath::BNSD || !vec.CachedKScales(nStart)) &&
                                        step >= dataSlotCount +
                                                    (PATH == Int8GqaPath::TND && CONFIG == QFA_INT8_CFG_RESIDENT_Q ?
                                                         QFA_K_SCALE_PREFIX_PANELS :
                                                         0U))
                                        CrossCoreWaitFlag(K_SCALE_FREE_FLAG);
                                    vec.PrepareKScale(nStart, nSize, step % dataSlotCount);
                                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(K_SCALE_READY_FLAG);
                                }
                            }
                        }
                        if (step > 0U && step - 1U < nTiles) {
                            const uint32_t prev = step - 1U;
                            const uint32_t nStart = prev * cfg.blockN;
                            const uint32_t nSize = cfg.blockN < sk - nStart ? cfg.blockN : sk - nStart;
                            if ASCEND_IS_AIC {
                                cube.PreloadV(nStart, nSize);
                                CrossCoreWaitFlag(VEC1_READY_FLAG_BASE + prev % QFA_PIPELINE_SLOTS);
                                cube.ComputeMm2(nStart, nSize, prev % dataSlotCount, true);
                                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(MM2_READY_FLAG_BASE + prev % QFA_PIPELINE_SLOTS);
                            } else {
                                CrossCoreWaitFlag(MM1_READY_FLAG_BASE + prev % QFA_PIPELINE_SLOTS);
                                vec.Vec1(nStart, nSize, prev % dataSlotCount, prev == 0U);
                                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(VEC1_READY_FLAG_BASE +
                                                                       prev % QFA_PIPELINE_SLOTS);
                            }
                        }
                        if ASCEND_IS_AIV {
                            if (step >= VEC2_STAGE_DELAY && step - VEC2_STAGE_DELAY < nTiles) {
                                const uint32_t prev = step - VEC2_STAGE_DELAY;
                                CrossCoreWaitFlag(MM2_READY_FLAG_BASE + prev % QFA_PIPELINE_SLOTS);
                                vec.Vec2(prev % dataSlotCount, prev + 1U == nTiles);
                            }
                        }
                    }
                }
            }
            if (crossTaskPipeline) {
                DrainTasks(pipelineLoop, taskCache, cube, vec);
            }
            if ASCEND_IS_AIC {
                cube.Finish();
            }
        }
    }

private:
    // Advance MM1 ahead of Vec1/MM2/Vec2 across Q-task boundaries.
    __aicore__ static inline void ExecuteTask(uint64_t loop, RunInfo taskCache[QFA_PIPELINE_SLOTS], Cube& cube,
                                              Vector& vec)
    {
        RunInfo& current = taskCache[loop % QFA_PIPELINE_SLOTS];
        RunInfo& previous = taskCache[(loop + QFA_PIPELINE_SLOTS - 1U) % QFA_PIPELINE_SLOTS];
        RunInfo& older = taskCache[(loop + 1U) % QFA_PIPELINE_SLOTS];
        if (current.isValid) {
            if constexpr (K_SCALE_PROTOCOL) {
                // Before changing sequence/head, retire the previous MM1's reads from the shared scale table.
                if (current.isFirst && previous.isValid &&
                    (current.batch != previous.batch || current.kvHead != previous.kvHead)) {
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(K_SCALE_FREE_FLAG);
                    } else {
                        CrossCoreWaitFlag(K_SCALE_FREE_FLAG);
                    }
                }
            }
            if ASCEND_IS_AIC {
                cube.SetSequence(current.sq, current.sk, current.qStart, current.kStart);
                if (current.isFirst) {
                    cube.BeginTask(current.batch, current.kvHead, current.mStart, current.mSize, false);
                } else {
                    cube.SelectTask(current.batch, current.kvHead, current.mStart, current.mSize, false);
                }
                if (!K_SCALE_PROTOCOL || cube.NeedKScale(current.nStart))
                    CrossCoreWaitFlag(K_SCALE_READY_FLAG);
                cube.ComputeMm1(current.nStart, current.nSize, current.dataSlot);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(MM1_READY_FLAG_BASE + current.flagSlot);
            } else {
                // K conversion is a producer stage, independent of the live softmax/accumulator state of the
                // preceding Q task.
                vec.SetSequence(current.sq, current.sk, current.qStart, current.kStart);
                vec.SelectKTask(current.batch, current.kvHead);
                if (!K_SCALE_PROTOCOL || vec.NeedKScale(current.nStart)) {
                    vec.PrepareKScale(current.nStart, current.nSize, current.dataSlot);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(K_SCALE_READY_FLAG);
                }
            }
        }
        if (previous.isValid) {
            if ASCEND_IS_AIC {
                cube.SetSequence(previous.sq, previous.sk, previous.qStart, previous.kStart);
                cube.SelectTask(previous.batch, previous.kvHead, previous.mStart, previous.mSize);
                cube.PreloadV(previous.nStart, previous.nSize);
                CrossCoreWaitFlag(VEC1_READY_FLAG_BASE + previous.flagSlot);
                cube.ComputeMm2(previous.nStart, previous.nSize, previous.dataSlot, true);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(MM2_READY_FLAG_BASE + previous.flagSlot);
            } else {
                vec.SetSequence(previous.sq, previous.sk, previous.qStart, previous.kStart);
                if (previous.isFirst) {
                    vec.BeginTask(previous.batch, previous.kvHead, previous.mStart, previous.mSize, false, false);
                    // The first tile writes the other sum bank, preserving the preceding task's final denominator
                    // for its Vec2.
                    vec.SetSoftmaxState(previous.initialState);
                } else {
                    vec.SelectTask(previous.batch, previous.kvHead, previous.mStart, previous.mSize, false);
                }
                CrossCoreWaitFlag(MM1_READY_FLAG_BASE + previous.flagSlot);
                {
                    vec.Vec1(previous.nStart, previous.nSize, previous.dataSlot, previous.isFirst);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(VEC1_READY_FLAG_BASE + previous.flagSlot);
                }
            }
        }
        if (older.isValid) {
            if ASCEND_IS_AIV {
                vec.SetSequence(older.sq, older.sk, older.qStart, older.kStart);
                vec.SelectTask(older.batch, older.kvHead, older.mStart, older.mSize);
                CrossCoreWaitFlag(MM2_READY_FLAG_BASE + older.flagSlot);
                vec.Vec2Task(older.dataSlot, older.isFirst, older.isLast, older.sumBank);
            }
            older.isValid = false;
        }
    }

    __aicore__ static inline void DrainTasks(uint64_t& loop, RunInfo taskCache[QFA_PIPELINE_SLOTS], Cube& cube,
                                             Vector& vec)
    {
        for (uint32_t drain = 0U; drain < VEC2_STAGE_DELAY; ++drain) {
            taskCache[loop % QFA_PIPELINE_SLOTS].isValid = false;
            ExecuteTask(loop, taskCache, cube, vec);
            ++loop;
        }
    }
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_KERNEL_INT8_H_
