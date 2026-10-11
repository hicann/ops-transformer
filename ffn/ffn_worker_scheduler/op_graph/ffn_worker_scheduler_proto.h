/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attention_worker_scheduler_proto.h
 * \brief
 */

#ifndef FFN_WORKER_SCHEDULER_PROTO_H
#define FFN_WORKER_SCHEDULER_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Data Scanning Operator for the FFN Side in Decoupled Attention-FFN Deployment Scenarios，This operator polls
data from AttentionToFFN to confirm its readiness state.
*@par Inputs:
*@li  schedule_context: A tensor. The type support int8. Its shape must be 1D (1024). Format: ND

*@par Attributes
*@li sync_group_size: Sessions handled per group, with a default value of 1. The type support int32.
*@li execute_mode: Kernel execution mode. Currently only supports value 0. The type support int32.

*@par Outputs:
*schedule_context: A ND Tensor that holds the new value of schedule_context after the value has been assigned.
*Has the same shape and dtype and format as the input "schedule_context". \n
*/
REG_OP(FfnWorkerScheduler)
    .INPUT(schedule_context, TensorType({DT_INT8}))
    .OUTPUT(schedule_context, TensorType({DT_INT8}))
    .ATTR(sync_group_size, Int, 1)
    .ATTR(execute_mode, Int, 0) // 1: loop, 0: once
    .OP_END_FACTORY_REG(FfnWorkerScheduler)
} // namespace ge

#endif // ATTENTION_WORKER_SCHEDULER_PROTO_H
