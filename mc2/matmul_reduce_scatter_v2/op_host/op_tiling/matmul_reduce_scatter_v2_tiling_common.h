/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_TILING_COMMON_H
#define MATMUL_REDUCE_SCATTER_V2_TILING_COMMON_H

/*!
 * \file matmul_reduce_scatter_v2_tiling_common.h
 * \brief
 */
#include "graph/utils/type_utils.h"

namespace optiling {
ge::graphStatus MatmulReduceScatterTilingV2Func(gert::TilingContext* context);
} // namespace optiling

#endif // MATMUL_REDUCE_SCATTER_V2_TILING_COMMON_H
