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
 * \file dense_lightning_indexer_kl_loss_grad_metadata_aicpu.h
 * \brief
 */
#ifndef DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_H
#define DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include "log.h"
#include "status.h"
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "../../common/op_kernel/aicpu_common.h"

namespace aicpu {

template <typename T>
inline T CeilDiv(T num, T rnd)
{
    if (rnd == 0) {
        return 0;
    }
    return (num + rnd - 1) / rnd;
}

inline bool IsTensorValid(Tensor *tensor)
{
    return tensor != nullptr && tensor->GetData() != nullptr && tensor->GetTensorShape() != nullptr;
}

inline int64_t AbsDiff(int64_t lhs, int64_t rhs)
{
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

} // namespace aicpu

#endif // DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_H
