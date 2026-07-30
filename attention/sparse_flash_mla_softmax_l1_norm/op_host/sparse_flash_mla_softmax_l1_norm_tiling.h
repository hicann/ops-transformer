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
 * \file sparse_flash_mla_softmax_l1_norm_tiling.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_H_
#define SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
namespace smlasoftmaxl1norm {

BEGIN_TILING_DATA_DEF(SmlaSoftmaxL1NormTilingData)
TILING_DATA_FIELD_DEF(int64_t, b)
TILING_DATA_FIELD_DEF(int64_t, sq)
TILING_DATA_FIELD_DEF(int64_t, sk)
TILING_DATA_FIELD_DEF(int64_t, g)
TILING_DATA_FIELD_DEF(int64_t, d)
TILING_DATA_FIELD_DEF(int64_t, t1)
TILING_DATA_FIELD_DEF(int64_t, t2)
TILING_DATA_FIELD_DEF(int64_t, max_seqlen_k)
TILING_DATA_FIELD_DEF(int64_t, k_length)
TILING_DATA_FIELD_DEF(int64_t, cmp_ratio)
TILING_DATA_FIELD_DEF(int64_t, init_per_core_num)
TILING_DATA_FIELD_DEF(int64_t, init_total_num)
TILING_DATA_FIELD_DEF(float, softmax_scale)
TILING_DATA_FIELD_DEF(bool, has_seqused_q)
TILING_DATA_FIELD_DEF(bool, has_seqused_k)
TILING_DATA_FIELD_DEF(bool, has_topk_length)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseFlashMlaSoftmaxL1Norm, SmlaSoftmaxL1NormTilingData)

struct SparseFlashMlaSoftmaxL1NormCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
};

struct AiCoreParams {
    uint64_t ubSize = 0;
    uint64_t numBlocks = 0;
    uint64_t aicNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
};
} // namespace smlasoftmaxl1norm
} // namespace optiling

#endif // SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_H_
