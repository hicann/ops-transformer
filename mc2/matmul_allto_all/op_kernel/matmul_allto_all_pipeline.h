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
 * \file matmul_allto_all_pipeline.h
 * \brief
 */

#ifndef MATMUL_ALLTO_ALL_PIPELINE_H
#define MATMUL_ALLTO_ALL_PIPELINE_H

#include "../../common/op_kernel/mc2_templates/scheduler/pipeline_builder.h"

// 流水线模板
namespace Mc2Kernel {
template <typename ComputationContextType>
struct MatmulAlltoAllPipelineContext {
    // computation
    ComputationContextType *computationContext;
    // transpose
    MC2KernelTemplate::MC2TransposeContext *transposeContext;
    // communication
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
};

// 计算转置通信模板
template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType,
          bool IsKCQuantMode = false>
class MatmulAlltoAllPipeLine {
public:
    __aicore__ inline MatmulAlltoAllPipeLine(ComputationType *computeStage, TransposeType *transStage,
                                             CommunicationType *commStage)
        : computeStage_(computeStage),
          transStage_(transStage),
          commStage_(commStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType *context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    ComputationType *computeStage_; // 矩阵乘的计算节点
    TransposeType *transStage_;     // 转置计算的计算节点
    CommunicationType *commStage_;  // 通信节点
    __aicore__ inline void ProcessKC(uint32_t taskCnt);
    __aicore__ inline void ProcessDefault(uint32_t taskCnt);
};

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType,
          bool IsKCQuantMode>
__aicore__ inline void
MatmulAlltoAllPipeLine<ComputationType, TransposeType, CommunicationType, ContextType, IsKCQuantMode>::Init()
{
    computeStage_->Init();
    commStage_->Init();
}

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType,
          bool IsKCQuantMode>
__aicore__ inline void MatmulAlltoAllPipeLine<ComputationType, TransposeType, CommunicationType, ContextType,
                                              IsKCQuantMode>::GetContext(ContextType *context)
{
    context->computationContext = computeStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->communicationContext = commStage_->GetContextPtr();
}

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType,
          bool IsKCQuantMode>
__aicore__ inline void MatmulAlltoAllPipeLine<ComputationType, TransposeType, CommunicationType, ContextType,
                                              IsKCQuantMode>::Process(uint32_t taskCnt)
{
    if constexpr (IsKCQuantMode) {
        ProcessKC(taskCnt);
    } else {
        ProcessDefault(taskCnt);
    }
}

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType,
          bool IsKCQuantMode>
__aicore__ inline void MatmulAlltoAllPipeLine<ComputationType, TransposeType, CommunicationType, ContextType,
                                              IsKCQuantMode>::ProcessDefault(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    for (uint32_t index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIC {
            computeStage_->Process(index);
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(9);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(9);
            transStage_->Process(index);
            AscendC::SyncAll<true>();
            commStage_->Process(index);
        }
    }
}

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType,
          bool IsKCQuantMode>
__aicore__ inline void MatmulAlltoAllPipeLine<ComputationType, TransposeType, CommunicationType, ContextType,
                                              IsKCQuantMode>::ProcessKC(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    for (uint32_t index = 0; index < taskCnt; index++) {
        computeStage_->Process(index);
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            transStage_->Process(index);
            AscendC::SyncAll<true>();
            commStage_->Process(index);
        }
        AscendC::SyncAll<false>();
    }
}

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename ContextType,
          bool IsKCQuantMode>
__aicore__ inline void
MatmulAlltoAllPipeLine<ComputationType, TransposeType, CommunicationType, ContextType, IsKCQuantMode>::End()
{
    computeStage_->End();
    commStage_->End();
}
}; // namespace Mc2Kernel

#endif
