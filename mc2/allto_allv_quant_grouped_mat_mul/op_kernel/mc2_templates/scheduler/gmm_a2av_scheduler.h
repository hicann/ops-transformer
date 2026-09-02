/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gmm_a2av_scheduler.h
 * \brief
 */

#ifndef MC2_GMMA2AV_PIPELINE_TEMPLATE_COMM_COMPUTE_H
#define MC2_GMMA2AV_PIPELINE_TEMPLATE_COMM_COMPUTE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "basic_api/kernel_basic_intf.h"
#include "../a2av_gmm_utils.h"

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename CommOpType, typename ComputationOpType, typename SharedComputationOpType, bool IsNeedMM>
class GmmA2avScheduler {
public:
    __aicore__ inline GmmA2avScheduler(CommOpType &hcclOp, ComputationOpType &computeOp,
                                       SharedComputationOpType &shareComputeOp, TaskTilingInfo *taskTilingInfo)
        : hcclOp_(hcclOp),
          computeOp_(computeOp),
          shareComputeOp_(shareComputeOp),
          taskTilingInfo_(taskTilingInfo){};

    __aicore__ inline void Process()
    {
        uint32_t e = taskTilingInfo_->e;
        uint32_t expertNum = taskTilingInfo_->expertNum;

        for (uint32_t start = 0; start < e; start += expertNum) {
            uint32_t actualExpertNum = (start + expertNum > e) ? (e - start) : expertNum;
            if ASCEND_IS_AIC {
                computeOp_.Process(start, actualExpertNum);
                AscendC::CrossCoreSetFlag<0, PIPE_FIX>(8);
                AscendC::CrossCoreWaitFlag(8);
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(9);
            }
            if ASCEND_IS_AIV {
                AscendC::CrossCoreWaitFlag(9);
                hcclOp_.LaunchCommData(start, actualExpertNum);
            }
        }

        if (IsNeedMM) {
            shareComputeOp_.Process(0, 1);
        }

        for (uint32_t start = 0; start < e; start += expertNum) {
            hcclOp_.Wait(start);
        }

        this->End();
    }

    __aicore__ inline void End()
    {
        hcclOp_.End();
    }

private:
    CommOpType &hcclOp_;
    ComputationOpType &computeOp_;
    SharedComputationOpType &shareComputeOp_;
    const TaskTilingInfo *taskTilingInfo_;
};
} // namespace MC2KernelTemplate
#endif
