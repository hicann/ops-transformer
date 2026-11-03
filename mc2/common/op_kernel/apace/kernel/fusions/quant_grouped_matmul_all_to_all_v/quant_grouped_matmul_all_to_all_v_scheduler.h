/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_all_to_all_v_scheduler.h
 * \brief apace 通路 GMM→AllToAllV 流水调度器（结构参考 gmm_a2av_scheduler.h）。
 *
 * 仅负责流水编排与同步点，计算/通信/共享专家实现由模板参数注入：
 *   - ComputationOpType：路由专家 GMM（接口 BatchCount/GetBatchTokenCount/
 *     Process(batch, tokenPrefix)）
 *   - CommOpType：AIC 核内 AllToAllV（接口 Launch(batch)/WaitAll()/Finalize()）
 *   - SharedComputationOpType：共享专家（接口 Process()）
 *   - IsNeedMM：是否存在共享专家（编译期，对应 TILINGKEY_COMPUTE_MATMUL）
 *
 * 流水（参考 gmm_a2av_scheduler.h 的批循环调度）：批内 GMM 计算 → 核间同步 →
 * 提交本批非阻塞 AllToAllV（通信由引擎异步执行，与后续批次计算重叠）；共享
 * 专家在批循环后执行（与末批通信重叠）；尾部统一 Wait 全部 handle → 排空
 * 管线 → 核间汇聚 → Finalize。
 *
 * workspace 表区布局（由各算子 Init 从同一 tableGM 基址按相同公式切分，
 * 复用原 gmmComputeWS 区域）：
 *   [0, e*R*8)  groupList（LENGTH 计数，每批覆写）
 *   [+, e*R*8)  C 输出偏移表（rank-major，每批覆写）
 *   [+, e*R*8)  B 权重偏移表（批内 expert 粒度）
 *   [+, e*R*8)  ScaleB 偏移表（批内 expert 粒度）
 *   [+, 8)      共享专家 mmGroupList（[0] = BS）
 */

#ifndef APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_SCHEDULER_H__
#define APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_SCHEDULER_H__

// 本头仅含流水编排模板，AscendC 核间同步原语（CrossCoreSetFlag/WaitFlag、
// PipeBarrier）由使用方经先序包含 hcomm_impl.h（blaze/hccl 头链）提供

namespace Apace {

// AIC 全核屏障事件 ID：同仓 AIC-only apace 实现（matmul_reduce_scatter_v2、
// all_to_all_quant_matmul 960 同步修复）均使用该值，避开 Blaze 内部 0-7 与
// SyncAll 保留 11-14
static constexpr uint16_t GMM_A2AV_AIC_SYNC_EVENT_ID = 3;

template <typename CommOpType, typename ComputationOpType, typename SharedComputationOpType, bool IsNeedMM>
class GmmA2avHcommScheduler {
public:
    __aicore__ inline GmmA2avHcommScheduler(CommOpType& commOp, ComputationOpType& computeOp,
                                            SharedComputationOpType& sharedOp)
        : commOp_(commOp),
          computeOp_(computeOp),
          sharedOp_(sharedOp)
    {}

    __aicore__ inline void Process()
    {
        uint64_t tokenPrefix = 0;
        for (uint32_t batch = 0; batch < computeOp_.BatchCount(); ++batch) {
            const uint64_t batchTokens = computeOp_.GetBatchTokenCount(batch);
            if (batchTokens != 0) {
                computeOp_.Process(batch, tokenPrefix);
            }
            // AIC 核间同步：本批 sendBuf 多核写全部落 GM 后才可提交通信；
            // 空批也照常提交，保证各 rank 集合通信次数对齐。
            // 注意不能用默认 SyncAll()（isAIVOnly=true 时 AIC 核为空操作，
            // 先到核会提前提交非阻塞 AlltoAllV，读到其他核未落 GM 的陈旧段）
            SyncAicCores();
            commOp_.Launch(batch);
            // 排空本核数据搬移管线（含 block0 侧参数区拷贝）后再复用参数数组
            AscendC::PipeBarrier<PIPE_ALL>();
            tokenPrefix += batchTokens;
        }

        // 共享专家位于批循环后、Wait 之前（与末批在途通信重叠执行）
        if constexpr (IsNeedMM) {
            sharedOp_.Process();
        }

        // 尾部统一等待全部批次完成（全核，按提交顺序）；Wait 后排空管线再
        // 汇聚、Finalize
        commOp_.WaitAll();
        AscendC::PipeBarrier<PIPE_ALL>();
        SyncAicCores();
        commOp_.Finalize();
    }

private:
    // AIC-only 任务下的 AIC 全核屏障。默认 SyncAll()（isAIVOnly=true）在
    // AIC 核上是空操作，无法保证本批 GMM 输出（40 核各写各段）全部落 GM；
    // 必须用 CrossCore 原语做全核对齐后再提交非阻塞 AlltoAllV，否则通信
    // 引擎会读到陈旧 sendBuf 段导致收端 rank 精度偶发错误。
    __aicore__ inline void SyncAicCores()
    {
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(GMM_A2AV_AIC_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(GMM_A2AV_AIC_SYNC_EVENT_ID);
    }

    CommOpType& commOp_;
    ComputationOpType& computeOp_;
    SharedComputationOpType& sharedOp_;
};

} // namespace Apace

#endif // APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_SCHEDULER_H__
