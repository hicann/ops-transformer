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
 * \file allto_all_matmul_add_bias_pipeline.h
 * \brief arch35 FP 非量化 + 独立加 bias 流水线模板
 *
 * 软件流水: add_bias(i-1) 延后到下一轮的 comm(i)+trans(i) 之后, 与 matmul(i) 并行
 *
 * 每轮 task 流水 (i >= 1):
 *   AIV: comm(i) → SyncAll → trans(i) → SyncAll[A] → add_bias(i-1) → SyncAll[B]
 *   AIC:                                SyncAll[A] → matmul(i)      → SyncAll[B]
 *   add_bias(i-1) 与 matmul(i) 在不同核上并行执行 (写不同 y 区域, 无冲突)
 *
 * 首轮 (i=0): AIV 无 add_bias 可做, 在 SyncAll[B] 等待 matmul(0) 完成
 * 尾部: 循环结束后 AIV 执行 add_bias(n-1)
 */

#ifndef ALLTO_ALL_MATMUL_ADD_BIAS_PIPELINE_H
#define ALLTO_ALL_MATMUL_ADD_BIAS_PIPELINE_H

#include "../allto_all_matmul_pipeline.h"
#include "../../../common/op_kernel/mc2_templates/computation/math/mc2_vec_add_bias.h"

namespace Mc2Kernel {

// arch35 FP 非量化 + 独立加 bias 专用上下文
// 继承原始上下文, 追加 addBiasContext 字段
template <typename ComputationContextType>
struct AlltoAllMmAddBiasPipelineContext : public AlltoAllMmPipelineContext<ComputationContextType> {
    MC2KernelTemplate::MC2AddBiasContext *addBiasContext = nullptr;
};

// 通信转置计算 + 独立加 bias 模板
template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
class AlltoAllMatmulAddBiasPipeLine {
public:
    __aicore__ inline AlltoAllMatmulAddBiasPipeLine(CommunicationType *commStage, TransposeType *transStage,
                                                    ComputationType *computeStage, AddBiasType *addBiasStage)
        : commStage_(commStage),
          transStage_(transStage),
          computeStage_(computeStage),
          addBiasStage_(addBiasStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType *context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    CommunicationType *commStage_;  // 通信节点
    TransposeType *transStage_;     // 转置计算的计算节点
    ComputationType *computeStage_; // 矩阵乘的计算节点
    AddBiasType *addBiasStage_;     // 加 bias 的计算节点
};

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void
AlltoAllMatmulAddBiasPipeLine<CommunicationType, TransposeType, ComputationType, AddBiasType, ContextType>::Init()
{
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void AlltoAllMatmulAddBiasPipeLine<CommunicationType, TransposeType, ComputationType, AddBiasType,
                                                     ContextType>::GetContext(ContextType *context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
    context->addBiasContext = addBiasStage_->GetContextPtr();
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void AlltoAllMatmulAddBiasPipeLine<CommunicationType, TransposeType, ComputationType, AddBiasType,
                                                     ContextType>::Process(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    for (uint32_t index = 0; index < taskCnt; index++) {
        // 阶段1/2: AIV 通信 + 转置
        if ASCEND_IS_AIV {
            commStage_->Process(index);
            AscendC::SyncAll<true>();
            transStage_->Process(index);
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(8);
            if (index > 0) {
                AscendC::CrossCoreWaitFlag(10);
                addBiasStage_->Process(index - 1);
                AscendC::SyncAll<true>();
            }
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(8);
            computeStage_->Process(index);
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(9);
            AscendC::CrossCoreWaitFlag(9);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(10);
        }
    }
    // 尾部: 最后一轮的 add_bias(n-1) (matmul(n-1) 已在循环末尾完成)
    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(10);
        addBiasStage_->Process(taskCnt - 1);
        AscendC::SyncAll<true>();
    }
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void
AlltoAllMatmulAddBiasPipeLine<CommunicationType, TransposeType, ComputationType, AddBiasType, ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}
}; // namespace Mc2Kernel

#endif
