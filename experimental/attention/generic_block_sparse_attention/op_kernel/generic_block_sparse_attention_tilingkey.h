/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_TILINGKEY_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_TILINGKEY_H

#include "kernel_tiling/kernel_tiling.h"

// W8A8 pseudo-quantization (arch22 / ascend910b only): FP16/BF16 query + INT8 KV.
// Key values are kept identical to the original GenericBlockSparseAttention op
// for debugging parity.  TILING_KEY_IS needs integer literals (expression
// macros like base+offset are dropped from fatbin).
#define GSA_BASE_ARCH22_TILING 40000
#define GSA_FP16_INT8_ARCH22_TILING 40006
#define GSA_BF16_INT8_ARCH22_TILING 40007

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_TILINGKEY_H
