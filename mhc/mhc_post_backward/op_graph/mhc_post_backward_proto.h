

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
 * \file mhc_post_backward_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MHC_POST_BACKWARD_H_
#define OPS_OP_PROTO_INC_MHC_POST_BACKWARD_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Backward operator for mhc_post. Computes gradients of grad_output w.r.t.
 * x, h_res, h_out and h_post.
 * @par Inputs:
 * @li grad_y: A Tensor. Gradient of output. Type is: BFloat16 or Float16. \n
 * Dataformat:ND. Supports 3D (Shape:[T, n, D]) or 4D (Shape:[B, S, n, D]) tensors.
 * @li x: A Tensor. Input data in the mHC layer. Type is: BFloat16 or Float16. \n
 * Dataformat:ND. Supports 3D (Shape:[T, n, D]) or 4D (Shape:[B, S, n, D]) tensors.
 * @li h_res: A Tensor. The h_res transformation matrix. Type is: Float32. \n
 * Dataformat:ND. Supports 3D (Shape:[T, n, n]) or 4D (Shape:[B, S, n, n]) tensors.
 * @li h_out: A Tensor. Output of the Atten/MLP layer. Type is: BFloat16 or Float16. \n
 * Dataformat:ND. Supports 2D (Shape:[T, D]) or 3D (Shape:[B, S, D]) tensors.
 * @li h_post: A Tensor. The h_post gating vector. Type is: Float32. \n
 * Dataformat:ND. Supports 2D (Shape:[T, n]) or 3D (Shape:[B, S, n]) tensors.
 * @par Outputs:
 * @li grad_x: A Tensor. Gradient of x. Type is: BFloat16 or Float16. \n
 * Dataformat:ND. Supports 3D (Shape:[T, n, D]) or 4D (Shape:[B, S, n, D]) tensors.
 * @li grad_h_res: A Tensor. Gradient of h_res. Type is: Float32. \n
 * Dataformat:ND. Supports 3D (Shape:[T, n, n]) or 4D (Shape:[B, S, n, n]) tensors.
 * @li grad_h_out: A Tensor. Gradient of h_out. Type is: BFloat16 or Float16. \n
 * Dataformat:ND. Supports 2D (Shape:[T, D]) or 3D (Shape:[B, S, D]) tensors.
 * @li grad_h_post: A Tensor. Gradient of h_post. Type is: Float32. \n
 * Dataformat:ND. Supports 2D (Shape:[T, n]) or 3D (Shape:[B, S, n]) tensors.
 */
REG_OP(MhcPostBackward)
    .INPUT(grad_y, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(h_res, TensorType({DT_FLOAT}))
    .INPUT(h_out, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(h_post, TensorType({DT_FLOAT}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_h_res, TensorType({DT_FLOAT}))
    .OUTPUT(grad_h_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_h_post, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(MhcPostBackward)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MHC_POST_BACKWARD_H_
