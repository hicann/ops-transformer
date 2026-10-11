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
 * \file mhc_pre_sinkhorn_proto.h
 * \brief MhcPreSinkhorn operator proto definition
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_SINKHORN_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_SINKHORN_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief MhcPreSinkhorn operator performs Sinkhorn normalization with hierarchical combination
 */
REG_OP(MhcPreSinkhorn)
    .INPUT(x, "T1")
    .INPUT(phi, "T2")
    .INPUT(alpha, "T2")
    .INPUT(bias, "T2")
    .OUTPUT(hin, "T1")
    .OUTPUT(hPost, "T2")
    .OUTPUT(hRes, "T2")
    .OUTPUT(hPre, "T2")
    .OUTPUT(hcBeforeNorm, "T2")
    .OUTPUT(invRms, "T2")
    .OUTPUT(sumOut, "T2")
    .OUTPUT(normOut, "T2")
    .DATATYPE(T1, TensorType({DT_BF16, DT_FLOAT16}))
    .DATATYPE(T2, TensorType({DT_FLOAT}))
    .ATTR(hc_mult, Int, 4)
    .ATTR(num_iters, Int, 20)
    .ATTR(hc_eps, Float, 1e-6f)
    .ATTR(norm_eps, Float, 1e-6f)
    .ATTR(need_backward, Bool, true)
    .OP_END_FACTORY_REG(MhcPreSinkhorn)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_SINKHORN_PROTO_H_
