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
 * \file mhc_sinkhorn_proto.h
 * \brief mhc_sinkhorn
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_MHC_SINKHORN_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MHC_SINKHORN_H_
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Perform the Sinkhorn normalization on the input tensor h_res to generate
 * a doubly stochastic matrix, and compute related normalization and summation results.
 * @par Description:
 * This operator implements the Sinkhorn-Knopp algorithm to iteratively normalize
 * the rows and columns of the input matrix h_res, making it close to a doubly stochastic matrix.
 * It outputs the normalized matrix, normalization factor, and summation result based on configuration.
 * @par Inputs:
 * @li h_res: A Tensor of type Float32. Input matrix to be normalized by Sinkhorn algorithm.
 * @par Outputs:
 * @li y: A Tensor of type Float32. The doubly stochastic matrix generated after Sinkhorn normalization.
 * @li norm_out: A Tensor of type Float32. Normalization factor tensor computed during the iteration process.
 * @li sum_out: A Tensor of type Float32. Summation result tensor of the normalized matrix rows/columns.
 * @par Attributes:
 * @li eps: Float type, default value 1.00e-06. A small epsilon value to prevent division by zero during normalization.
 * @li num_iters: Int type, default value 20. Number of iteration steps for the Sinkhorn-Knopp algorithm. Range: [1,
 * 100].
 * @li out_flag: Int type, default value 0. Control flag for output mode (0: basic output, 1: extended output,
 * etc.).Currently, only value 0 is supported in this version.
 */
REG_OP(MhcSinkhorn)
    .INPUT(h_res, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(norm_out, TensorType({DT_FLOAT}))
    .OUTPUT(sum_out, TensorType({DT_FLOAT}))
    .ATTR(eps, Float, 1.00e-06)
    .ATTR(num_iters, Int, 20)
    .ATTR(out_flag, Int, 0)
    .OP_END_FACTORY_REG(MhcSinkhorn)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_MHC_SINKHORN_H_
