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
 * \file attention_worker_combine.h
 * \brief Ascend950 nonquant kernel that shares one implementation across BS/K/H tiling strategies.
 */

#ifndef ATTENTION_WORKER_COMBINE_REGBASE_H_
#define ATTENTION_WORKER_COMBINE_REGBASE_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "attention_worker_combine_tiling_struct.h"
#include "../attention_worker_combine_common_utils.h"

namespace AttentionWorkerCombineRegbase {
using namespace AscendC;
using optiling::AttentionWorkerCombineRegbaseTilingData;

constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int64_t BLOCK_SIZE = 32;

enum class NonquantSplit {
    BS,
    K,
    H
};

template <typename T, NonquantSplit Split>
class KernelAttentionWorkerCombine {
public:
    __aicore__ inline void Init(GM_ADDR schedule_context, GM_ADDR expert_scales, GM_ADDR layer_id, GM_ADDR y,
                                GM_ADDR next_layer_id, GM_ADDR workspace,
                                const AttentionWorkerCombineRegbaseTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        contextGm0.SetGlobalBuffer((__gm__ uint32_t *)schedule_context);
        contextGm1.SetGlobalBuffer((__gm__ uint64_t *)schedule_context);

        if (tilingData->needSchedule == 1) {
            uint32_t micro_batch_num = contextGm0(GET_OFFSET_B32(ScheduleContext, common.micro_batch_num));
            micro_batch_id =
                (contextGm0(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id)) + 1) % micro_batch_num;
        } else {
            micro_batch_id = contextGm0(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id));
        }

        uint64_t token_data_addr = contextGm1(GET_OFFSET_B64(ScheduleContext, attention.token_data_buf));
        uint64_t token_info_addr = contextGm1(GET_OFFSET_B64(ScheduleContext, attention.token_info_buf));

        srcTokenGm.SetGlobalBuffer((__gm__ T *)(token_data_addr + micro_batch_id * tilingData->BS *
                                                                      (tilingData->K + 1) * tilingData->H * sizeof(T)));
        srcTokenInfoGm.SetGlobalBuffer((__gm__ int32_t *)(token_info_addr + micro_batch_id * tilingData->BS *
                                                                                (tilingData->K + 1) * sizeof(int32_t)));
        srcScalesGm.SetGlobalBuffer((__gm__ float *)expert_scales);
        srcLayerIdGm.SetGlobalBuffer((__gm__ int32_t *)layer_id);
        dstGm.SetGlobalBuffer((__gm__ T *)y);
        dstNextLayerIdGm.SetGlobalBuffer((__gm__ int32_t *)next_layer_id);

        const int64_t inputRows =
            Split == NonquantSplit::BS ? tilingData->K + 1 : (Split == NonquantSplit::K ? tilingData->KSplitFactor : 1);
        pipe->InitBuffer(tokenDataQue, DOUBLE_BUFFER, inputRows * tilingData->HSplitFactor * sizeof(T));
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, tilingData->HSplitFactor * sizeof(T));
        pipe->InitBuffer(tmpBuf, tilingData->HSplitFactor * sizeof(float) * DOUBLE_BUFFER);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockId = GetBlockIdx();
        const int64_t hCores = Split == NonquantSplit::H ? tilingData->HSplitCoreNum : 1;
        const int64_t bsCore = blockId / hCores;
        const int64_t hCore = blockId % hCores;
        const int64_t bsStart = bsCore * tilingData->mainCoreBsLoopNum;
        const int64_t bsCount =
            bsCore == tilingData->BsSplitCoreNum - 1 ? tilingData->tailCoreBsLoopNum : tilingData->mainCoreBsLoopNum;
        const int64_t flagStart = bsStart * (tilingData->K + 1);
        const int64_t flagCount = bsCount * (tilingData->K + 1);

        if (blockId == 0) {
            dstNextLayerIdGm.SetValue(0, srcLayerIdGm.GetValue(0) + 1);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(dstNextLayerIdGm);
        }
        if (tilingData->needSchedule == 1) {
            WaitForTokensReady(flagStart, flagCount);
        }

        for (int64_t row = bsStart; row < bsStart + bsCount; ++row) {
            if constexpr (Split == NonquantSplit::H) {
                const int64_t hStart = hCore * tilingData->mainCoreHLoopNum * tilingData->HSplitFactor;
                const int64_t hLoops =
                    hCore == hCores - 1 ? tilingData->tailCoreHLoopNum : tilingData->mainCoreHLoopNum;
                for (int64_t loop = 0; loop < hLoops; ++loop) {
                    const int64_t h = hStart + loop * tilingData->HSplitFactor;
                    const int64_t count =
                        tilingData->H - h < tilingData->HSplitFactor ? tilingData->H - h : tilingData->HSplitFactor;
                    ComputeTile(row, h, count);
                }
            } else {
                ComputeTile(row, 0, tilingData->H);
            }
        }

        if (tilingData->needSchedule == 1) {
            PipeBarrier<PIPE_ALL>();
            if constexpr (Split == NonquantSplit::H) {
                SyncAll();
            }
            if (hCore == 0) {
                ClearTokenInfo(flagStart, flagCount);
            }
            PipeBarrier<PIPE_ALL>();
            SyncAll();
            if (blockId == 0) {
                contextGm0.SetValue(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id), micro_batch_id);
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(
                    contextGm0[GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id)]);
            }
        }
    }

private:
    __aicore__ inline void WaitForTokensReady(int64_t offset, int64_t count)
    {
        int32_t sumRes = 0;
        int32_t expectedSum = count;

        while (sumRes != expectedSum) {
            sumRes = ScanTokenInfo(offset, count);
        }
    }

    __aicore__ inline int32_t ScanTokenInfo(int64_t tokenInfoStart, int64_t kSize)
    {
        // 复用 token/output 队列，按较小的输出容量分块，避免短 H、大 BS 时越界。
        int64_t flagCapacity = tilingData->HSplitFactor * sizeof(T) / sizeof(int32_t);
        int32_t sum = 0;
        for (int64_t offset = 0; offset < kSize; offset += flagCapacity) {
            int64_t count = kSize - offset < flagCapacity ? kSize - offset : flagCapacity;
            LocalTensor<int32_t> srcLocal = tokenDataQue.AllocTensor<int32_t>();
            DataCopyExtParams copyParams(1, count * sizeof(int32_t), 0, 0, 0);
            DataCopyPadExtParams<int32_t> padParams(false, 0, 0, 0);
            DataCopyPad(srcLocal, srcTokenInfoGm[tokenInfoStart + offset], copyParams, padParams);
            tokenDataQue.EnQue(srcLocal);
            srcLocal = tokenDataQue.DeQue<int32_t>();
            for (int64_t i = 0; i < count; i++) {
                sum += srcLocal.GetValue(i);
            }
            tokenDataQue.FreeTensor(srcLocal);
        }
        return sum;
    }

    __aicore__ inline void ClearTokenInfo(int64_t offset, int64_t kSize)
    {
        int64_t flagCapacity = tilingData->HSplitFactor * sizeof(T) / sizeof(int32_t);
        for (int64_t cleared = 0; cleared < kSize; cleared += flagCapacity) {
            int64_t count = kSize - cleared < flagCapacity ? kSize - cleared : flagCapacity;
            LocalTensor<int32_t> dstLocal = yQue.AllocTensor<int32_t>();
            Duplicate(dstLocal, static_cast<int32_t>(0), count);
            yQue.EnQue(dstLocal);
            dstLocal = yQue.DeQue<int32_t>();
            DataCopyExtParams copyParams(1, count * sizeof(int32_t), 0, 0, 0);
            DataCopyPad(srcTokenInfoGm[offset + cleared], dstLocal, copyParams);
            yQue.FreeTensor(dstLocal);
        }
    }

    __aicore__ inline void ComputeTile(int64_t row, int64_t h, int64_t count)
    {
        auto values = tmpBuf.Get<float>();
        auto sum = values[tilingData->HSplitFactor];
        Duplicate(sum, 0.0f, count);
        PipeBarrier<PIPE_V>();

        const int64_t expertRows = tilingData->K + 1;
        const int64_t kTile =
            Split == NonquantSplit::BS ? expertRows : (Split == NonquantSplit::K ? tilingData->KSplitFactor : 1);
        for (int64_t kStart = 0; kStart < expertRows; kStart += kTile) {
            const int64_t rows = expertRows - kStart < kTile ? expertRows - kStart : kTile;
            CopyIn((row * expertRows + kStart) * tilingData->H + h, rows, count);
            auto input = tokenDataQue.DeQue<T>();
            for (int64_t i = 0; i < rows; ++i) {
                Cast(values, input[i * tilingData->HSplitFactor], RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                const int64_t expert = kStart + i;
                if (expert < tilingData->K) {
                    Muls(values, values, srcScalesGm.GetValue(row * tilingData->K + expert), count);
                    PipeBarrier<PIPE_V>();
                }
                // The final shared expert is unscaled. Preserve the original K operand order.
                if constexpr (Split == NonquantSplit::K) {
                    if (expert == tilingData->K) {
                        Add(sum, values, sum, count);
                    } else {
                        Add(sum, sum, values, count);
                    }
                } else {
                    Add(sum, sum, values, count);
                }
                PipeBarrier<PIPE_V>();
            }
            tokenDataQue.FreeTensor(input);
        }
        CopyOut(row * tilingData->H + h, sum, count);
    }

    __aicore__ inline void CopyIn(int64_t offset, int64_t rows, int64_t count)
    {
        auto input = tokenDataQue.AllocTensor<T>();
        const int64_t alignedBytes = Ops::Base::CeilAlign<int64_t>(count * sizeof(T), BLOCK_SIZE);
        // BS/K load contiguous complete GM rows into HSplitFactor-spaced UB rows; H loads one row.
        DataCopyExtParams copyParams(rows, count * sizeof(T), 0,
                                     (tilingData->HSplitFactor * sizeof(T) - alignedBytes) / BLOCK_SIZE, 0);
        DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
        DataCopyPad(input, srcTokenGm[offset], copyParams, padParams);
        tokenDataQue.EnQue(input);
    }

    __aicore__ inline void CopyOut(int64_t offset, LocalTensor<float> sum, int64_t count)
    {
        auto output = yQue.AllocTensor<T>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(output, sum, RoundMode::CAST_RINT, count);
        } else {
            Cast(output, sum, RoundMode::CAST_NONE, count);
        }
        yQue.EnQue(output);
        output = yQue.DeQue<T>();
        DataCopyExtParams copyParams(1, count * sizeof(T), 0, 0, 0);
        DataCopyPad(dstGm[offset], output, copyParams);
        yQue.FreeTensor(output);
    }

    TPipe *pipe = nullptr;
    const AttentionWorkerCombineRegbaseTilingData *tilingData = nullptr;

    GlobalTensor<uint32_t> contextGm0;
    GlobalTensor<uint64_t> contextGm1;
    GlobalTensor<T> srcTokenGm;
    GlobalTensor<int32_t> srcTokenInfoGm;
    GlobalTensor<float> srcScalesGm;
    GlobalTensor<int32_t> srcLayerIdGm;
    GlobalTensor<T> dstGm;
    GlobalTensor<int32_t> dstNextLayerIdGm;

    TQue<QuePosition::VECIN, 1> tokenDataQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TBuf<TPosition::VECCALC> tmpBuf;

    uint32_t micro_batch_id = 0;
};

} // namespace AttentionWorkerCombineRegbase

#endif // ATTENTION_WORKER_COMBINE_REGBASE_H_
