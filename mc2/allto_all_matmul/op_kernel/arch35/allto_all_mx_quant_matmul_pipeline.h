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
 * \file allto_all_mx_quant_matmul_pipeline.h
 * \brief
 */

#ifndef ALLTO_ALL_MX_QUANT_MATMUL_PIPELINE_H
#define ALLTO_ALL_MX_QUANT_MATMUL_PIPELINE_H

#include "../../../common/op_kernel/mc2_templates/scheduler/pipeline_builder.h"

// 流水线模板
namespace Mc2Kernel {
template <typename ComputationContextType>
struct AlltoAllMXQmmPipelineContext {
    // computation
    ComputationContextType *computationContext;
    // transpose
    MC2KernelTemplate::MC2TransposeContext *transposeContext;
    // scaleTranspose
    MC2KernelTemplate::MC2TransposeContext *scaleTransposeContext;
    // communication
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
};

// 通信转置计算模板
template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
class AlltoAllMXQuantMatmulPipeLine {
public:
    __aicore__ inline AlltoAllMXQuantMatmulPipeLine(CommunicationType *commStage, TransposeType *transStage,
                                                    ScaleTransposeType *scaleTransStage, ComputationType *computeStage)
        : commStage_(commStage),
          transStage_(transStage),
          scaleTransStage_(scaleTransStage),
          computeStage_(computeStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType *context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void ProcessScale();

    __aicore__ inline void End();

private:
    CommunicationType *commStage_;        // 通信节点
    TransposeType *transStage_;           // 转置计算的计算节点
    ScaleTransposeType *scaleTransStage_; // 缩放转置的计算节点
    ComputationType *computeStage_;       // 矩阵乘的计算节点
};

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void AlltoAllMXQuantMatmulPipeLine<CommunicationType, TransposeType, ScaleTransposeType,
                                                     ComputationType, ContextType>::Init()
{
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void AlltoAllMXQuantMatmulPipeLine<CommunicationType, TransposeType, ScaleTransposeType,
                                                     ComputationType, ContextType>::GetContext(ContextType *context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->scaleTransposeContext = scaleTransStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void AlltoAllMXQuantMatmulPipeLine<CommunicationType, TransposeType, ScaleTransposeType,
                                                     ComputationType, ContextType>::Process(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    for (uint32_t index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIV {
            commStage_->Process(index);
            AscendC::SyncAll<true>();
            transStage_->Process(index);
            // 核间使用软同步，当前搭配的matmul没有使用软同步标识索引8和9，后续如果matmul有变化需要同步调整
            AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(9);
            computeStage_->Process(index);
        }
    }
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void AlltoAllMXQuantMatmulPipeLine<CommunicationType, TransposeType, ScaleTransposeType,
                                                     ComputationType, ContextType>::ProcessScale()
{
    if ASCEND_IS_AIV {
        commStage_->PrepareAll(1);
        commStage_->Process(0);
        AscendC::SyncAll<true>();
        scaleTransStage_->Process(0);
    }
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void
AlltoAllMXQuantMatmulPipeLine<CommunicationType, TransposeType, ScaleTransposeType, ComputationType, ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}
}; // namespace Mc2Kernel

#endif
