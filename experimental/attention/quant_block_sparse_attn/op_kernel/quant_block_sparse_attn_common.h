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
 * \file quant_block_sparse_attn_common.h
 * \brief QuantBlockSparseAttn kernel common declarations.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_COMMON_H
#define QUANT_BLOCK_SPARSE_ATTN_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_block_sparse_attn_const.h"

using namespace AscendC;

enum class QBSADType : uint32_t {
    FP8_E4M3FN = 0,
};

enum class QBSAMaskMode : uint32_t {
    NONE = 0,
    CAUSAL = 3,
};

constexpr uint32_t QBSA_BLOCK_SIZE = 128U;
constexpr uint32_t QBSA_HEAD_DIM = 128U;
constexpr float QBSA_EMPTY_LSE_VALUE = -3.4028234663852886e38F;

#endif // QUANT_BLOCK_SPARSE_ATTN_COMMON_H
