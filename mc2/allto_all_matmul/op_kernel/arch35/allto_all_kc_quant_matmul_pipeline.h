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
 * \file allto_all_kc_quant_matmul_pipeline.h
 * \brief
 */

#ifndef ALLTO_ALL_KC_QUANT_MATMUL_H
#define ALLTO_ALL_KC_QUANT_MATMUL_H

#include "../../../common/op_kernel/mc2_templates/scheduler/pipeline_builder.h"

// 流水线模板
namespace Mc2Kernel {
// 后续可以按节点拆成对应的上下文复用
template <typename ComputationContextType>
struct AlltoAllKCQmmPipelineContext {
    // computation
    ComputationContextType *computationContext;
    // communication
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
    // quantization
    MC2KernelTemplate::MC2PertokenDQuantContext *quantizationContext;
};

// 通信转置计算模板
template <typename CommunicationType, typename TransposeAndQuantizeType, typename ComputationType, typename ContextType>
class AlltoAllKCQuantMatmulPipeLine {
public:
    __aicore__ inline AlltoAllKCQuantMatmulPipeLine(CommunicationType *commStage,
                                                    TransposeAndQuantizeType *transAndQuantStage,
                                                    ComputationType *computeStage)
        : commStage_(commStage),
          transAndQuantStage_(transAndQuantStage),
          computeStage_(computeStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType *context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    CommunicationType *commStage_;                 // 通信节点
    TransposeAndQuantizeType *transAndQuantStage_; // 进行动态量化的节点
    ComputationType *computeStage_;                // 矩阵乘的计算节点
};

template <typename CommunicationType, typename TransposeAndQuantizeType, typename ComputationType, typename ContextType>
__aicore__ inline void
AlltoAllKCQuantMatmulPipeLine<CommunicationType, TransposeAndQuantizeType, ComputationType, ContextType>::Init()
{
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename TransposeAndQuantizeType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllKCQuantMatmulPipeLine<CommunicationType, TransposeAndQuantizeType, ComputationType,
                                                     ContextType>::GetContext(ContextType *context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->quantizationContext = transAndQuantStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
}

template <typename CommunicationType, typename TransposeAndQuantizeType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllKCQuantMatmulPipeLine<CommunicationType, TransposeAndQuantizeType, ComputationType,
                                                     ContextType>::Process(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    uint32_t index;
    for (index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIV {
            commStage_->Process(index);
            AscendC::SyncAll<true>();
            transAndQuantStage_->Process(index);
        }
        AscendC::SyncAll<false>();
        computeStage_->Process(index);
    }
}

template <typename CommunicationType, typename TransposeAndQuantizeType, typename ComputationType, typename ContextType>
__aicore__ inline void
AlltoAllKCQuantMatmulPipeLine<CommunicationType, TransposeAndQuantizeType, ComputationType, ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}
}; // namespace Mc2Kernel

#endif
