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
 * \file quant_grouped_mat_mul_allto_allv_tiling.cpp
 * \brief
 */

#include <string>
#include <numeric>
#include <climits>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "op_host/op_tiling/hccl_formulaic_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/tiling_templates_registry.h"
#include "mc2_hcom_topo_info.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_base.h"
#include "register/op_impl_registry.h"
#include "context_util.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace Mc2Tiling {
static ge::graphStatus QuantGroupedMatMulAlltoAllvTilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct QuantGroupedMatMulAlltoAllvCompileInfo {};
static ge::graphStatus TilingParseForQuantGroupedMatMulAlltoAllv(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantGroupedMatMulAlltoAllv)
    .Tiling(QuantGroupedMatMulAlltoAllvTilingFunc)
    .TilingParse<QuantGroupedMatMulAlltoAllvCompileInfo>(TilingParseForQuantGroupedMatMulAlltoAllv);
} // namespace Mc2Tiling
