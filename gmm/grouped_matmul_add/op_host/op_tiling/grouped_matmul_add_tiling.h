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
 * \file grouped_matmul_add_tiling.h
 * \brief
 */

#ifndef __GROUPED_MATMUL_ADD_TILING_H__
#define __GROUPED_MATMUL_ADD_TILING_H__
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_LIST_SIZE = 128;
struct GroupedMatmulAddCompileInfo {};
BEGIN_TILING_DATA_DEF(GmmBaseParams)
TILING_DATA_FIELD_DEF(int64_t, groupNum);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, groupType); // 分组类型，仅支持2 --- K分组
TILING_DATA_FIELD_DEF(int64_t, groupListType);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GmmBaseParamsOp, GmmBaseParams)

BEGIN_TILING_DATA_DEF(GroupedMatmulAddTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GmmBaseParams, gmmBaseParams)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GroupedMatmulAdd, GroupedMatmulAddTilingData)

} // namespace optiling

#endif // __GROUPED_MATMUL_ADD_TILING_H__
