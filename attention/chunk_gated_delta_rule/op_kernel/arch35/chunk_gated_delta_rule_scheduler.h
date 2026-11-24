/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "utils/chunk_gated_delta_rule_context.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// Scheduler 只产出任务，不持有/调用 BasicBlock，也不改写计算层的 Context。
// 调度单位是核组：一个 AIC 与两个 AIV 必须得到相同的任务及顺序。
// 同一 head 的所有 chunk 属于一个完整任务，保留状态递推的串行依赖。
template <typename stateType>
class Scheduler {
public:
    __aicore__ inline explicit Scheduler(const KernelParams<stateType>& params)
        : params_(params)
    {
        // 将 AIC/AIV 的硬件编号归一为同一个核组编号，保证三者独立枚举出相同任务。
        if ASCEND_IS_AIC {
            nextWork_ = static_cast<uint32_t>(GetBlockIdx());
        } else {
            nextWork_ = static_cast<uint32_t>(GetBlockIdx()) / static_cast<uint32_t>(GetSubBlockNum());
        }
        totalWork_ = params.batchCount * params.nv;
    }

    // 当前策略保持循环分配：workId=groupId+n*aiCoreNum。
    // 后续按长度调整整条head的分配/顺序只需替换任务枚举；Block不感知策略。
    __aicore__ inline bool Next(WorkItem& task)
    {
        if (nextWork_ >= totalWork_) {
            return false;
        }
        const uint64_t batchId = nextWork_ / params_.nv;
        // 当前workId单调递增，只累加新经过的batch。若改为乱序调度，必须一起替换此定位逻辑。
        while (scannedBatch_ < batchId) {
            tokenStart_ = ClampToTokenCount(tokenStart_ + SequenceLength(scannedBatch_));
            ++scannedBatch_;
        }
        // 空序列仍产出任务，由Block完成initialState -> finalState复制。
        // tokenStart/seqLength 均钳制在 T 内(host 已校验 T=sum(L_i) 与各项 [0,T];
        // 图模式取不到 host 数据时,单项超 T 或累计和越界在此兜底,保证 cube/vector
        // 的一切 tokenStart*... GM 偏移不越出 T 维)。
        const uint64_t seqLength = ClampToTokenCount(tokenStart_ + SequenceLength(batchId)) - tokenStart_;
        task = {batchId, tokenStart_, seqLength, nextWork_ - batchId * params_.nv};
        nextWork_ += params_.aiCoreNum;
        return true;
    }

private:
    // T 维合法上界,前缀累加后统一经此钳制,防止单项超 T 或累计和越过 T 时越界寻址。
    __aicore__ inline uint64_t ClampToTokenCount(uint64_t tokenPos) const
    {
        return tokenPos > params_.tokenCount ? params_.tokenCount : tokenPos;
    }

    __aicore__ inline uint64_t SequenceLength(uint64_t batchId) const
    {
        const int32_t length = params_.actualSeqLengths.GetValue(batchId);
        // 接口要求长度非负；防御性地将负值按空序列处理，避免转为uint64后回绕。
        // 任务长度与tokenStart前缀累加必须使用同一规则。
        return length > 0 ? static_cast<uint64_t>(length) : 0;
    }

    const KernelParams<stateType>& params_;
    uint64_t nextWork_;
    uint64_t totalWork_;
    uint64_t scannedBatch_ = 0;
    uint64_t tokenStart_ = 0;
};

} // namespace ChunkGatedDeltaRule
