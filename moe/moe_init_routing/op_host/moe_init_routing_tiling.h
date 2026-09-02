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
 * \file moe_init_routing_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_H
#include <cmath>
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {
BEGIN_TILING_DATA_DEF(VBSComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(VBSComputeTilingDataOp, VBSComputeTilingData)

BEGIN_TILING_DATA_DEF(VMSMiddleComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(VMSMiddleComputeTilingDataOp, VMSMiddleComputeTilingData)

BEGIN_TILING_DATA_DEF(SortOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SortOutComputeTilingDataOp, SortOutComputeTilingData)

BEGIN_TILING_DATA_DEF(GatherOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, activateRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreK);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopK);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopK);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreK);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopK);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopK);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopRows);
TILING_DATA_FIELD_DEF(int64_t, maxColsOneLoop);
TILING_DATA_FIELD_DEF(int64_t, splitFlag);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GatherOutComputeTilingDataOp, GatherOutComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF_STRUCT(VBSComputeTilingData, vbsComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(VMSMiddleComputeTilingData, vmsMiddleComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(SortOutComputeTilingData, sortOutComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(GatherOutComputeTilingData, srcToDstComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(GatherOutComputeTilingData, gatherOutComputeParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRouting, MoeInitRoutingTilingData)
struct MoeInitRoutingCompileInfo {};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_H
