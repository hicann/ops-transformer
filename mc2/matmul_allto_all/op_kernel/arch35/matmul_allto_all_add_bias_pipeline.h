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
 * \file matmul_allto_all_add_bias_pipeline.h
 * \brief matmul_allto_all 独立加 bias 流水线模板
 *
 * 数据流: matmul → tempComputeOutGM_ → add_bias(in-place) → trans → comm → y_
 *
 * 各轮 matmul 写 tempComputeOutGM_ 的不同 tile 区域, 无地址重叠,
 * AIC 的 matmul 可连续执行, 不需等待 AIV 的 add_bias/trans/comm 完成。
 *
 * 每轮 task 流水 (i):
 *   AIC: matmul(i) → flag8(AIC全核同步) → flag9(通知AIV)
 *   AIV: waitFlag(9) → add_bias(i) → trans(i) → SyncAll → comm(i)
 *   matmul(i+1) 与 add_bias(i)/trans(i)/comm(i) 并行 (不同 tile, 无冲突)
 */

#ifndef MATMUL_ALLTO_ALL_ADD_BIAS_PIPELINE_H
#define MATMUL_ALLTO_ALL_ADD_BIAS_PIPELINE_H

#include "../matmul_allto_all_pipeline.h"
#include "../../../common/op_kernel/mc2_templates/computation/math/mc2_vec_add_bias.h"

namespace Mc2Kernel {

template <typename ComputationContextType>
struct MatmulAlltoAllAddBiasPipelineContext : public MatmulAlltoAllPipelineContext<ComputationContextType> {
    MC2KernelTemplate::MC2AddBiasContext *addBiasContext = nullptr;
};

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename AddBiasType,
          typename ContextType>
class MatmulAlltoAllAddBiasPipeLine {
public:
    __aicore__ inline MatmulAlltoAllAddBiasPipeLine(ComputationType *computeStage, TransposeType *transStage,
                                                    CommunicationType *commStage, AddBiasType *addBiasStage)
        : computeStage_(computeStage),
          transStage_(transStage),
          commStage_(commStage),
          addBiasStage_(addBiasStage){};

    __aicore__ inline void Init();
    __aicore__ inline void GetContext(ContextType *context);
    __aicore__ inline void Process(uint32_t taskCnt);
    __aicore__ inline void End();

private:
    ComputationType *computeStage_;
    TransposeType *transStage_;
    CommunicationType *commStage_;
    AddBiasType *addBiasStage_;
};

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void
MatmulAlltoAllAddBiasPipeLine<ComputationType, TransposeType, CommunicationType, AddBiasType, ContextType>::Init()
{
    computeStage_->Init();
    commStage_->Init();
}

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void MatmulAlltoAllAddBiasPipeLine<ComputationType, TransposeType, CommunicationType, AddBiasType,
                                                     ContextType>::GetContext(ContextType *context)
{
    context->computationContext = computeStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->communicationContext = commStage_->GetContextPtr();
    context->addBiasContext = addBiasStage_->GetContextPtr();
}

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void MatmulAlltoAllAddBiasPipeLine<ComputationType, TransposeType, CommunicationType, AddBiasType,
                                                     ContextType>::Process(uint32_t taskCnt)
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
            addBiasStage_->Process(index);
            AscendC::SyncAll<true>();
            transStage_->Process(index);
            AscendC::SyncAll<true>();
            commStage_->Process(index);
        }
    }
}

template <typename ComputationType, typename TransposeType, typename CommunicationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void
MatmulAlltoAllAddBiasPipeLine<ComputationType, TransposeType, CommunicationType, AddBiasType, ContextType>::End()
{
    computeStage_->End();
    commStage_->End();
}

}; // namespace Mc2Kernel

#endif
