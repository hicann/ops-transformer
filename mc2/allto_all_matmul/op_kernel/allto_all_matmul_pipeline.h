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
 * \file allto_all_matmul_pipeline.h
 * \brief
 */

#ifndef ALLTO_ALL_MATMUL_PIPELINE_H
#define ALLTO_ALL_MATMUL_PIPELINE_H

#include "../../common/op_kernel/mc2_templates/communication/comm_stage.h"
#include "../../common/op_kernel/mc2_templates/computation/math/mc2_vec_transpose.h"
#include "../../common/op_kernel/mc2_templates/computation/matmul/mc2_fp_matmul.h"

// 流水线模板
namespace Mc2Kernel {
template <typename ComputationContextType>
struct AlltoAllMmPipelineContext {
    // computation
    ComputationContextType *computationContext;
    // transpose
    MC2KernelTemplate::MC2TransposeContext *transposeContext;
    // communication
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
};

// 通信转置计算模板
template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
class AlltoAllMatmulPipeLine {
public:
    __aicore__ inline AlltoAllMatmulPipeLine(CommunicationType *commStage, TransposeType *transStage,
                                             ComputationType *computeStage)
        : commStage_(commStage),
          transStage_(transStage),
          computeStage_(computeStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType *context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    CommunicationType *commStage_;  // 通信节点
    TransposeType *transStage_;     // 转置计算的计算节点
    ComputationType *computeStage_; // 矩阵乘的计算节点
};

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, ContextType>::Init()
{
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void
AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, ContextType>::GetContext(ContextType *context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, ContextType>::Process(
    uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    uint32_t index;
    for (index = 0; index < taskCnt; index++) {
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

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}
}; // namespace Mc2Kernel

#endif
