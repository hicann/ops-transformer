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
 * \file dense_lightning_indexer_softmax_lse_v2_proto.h
 * \brief
 */
#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_PROTO_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(DenseLightningIndexerSoftmaxLseV2)
    .INPUT(query_index, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key_index, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(cu_seq_lens_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seq_lens_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seq_used_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seq_used_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cmp_residual_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(metadata, TensorType({DT_INT32}))
    .OUTPUT(softmax_lse, TensorType({DT_FLOAT}))
    .ATTR(layoutQ, String, "BSND")
    .ATTR(layoutK, String, "BSND")
    .ATTR(maskMode, Int, 0)
    .ATTR(cmpRatio, Int, 1)
    .OP_END_FACTORY_REG(DenseLightningIndexerSoftmaxLseV2)

} // namespace ge

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_PROTO_H
