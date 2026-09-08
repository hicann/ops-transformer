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
 * \file grouped_mat_mul_allto_allv_tiling_a3.cpp
 */

#include "grouped_mat_mul_allto_allv_tiling_a3.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
REGISTER_OPS_TILING_TEMPLATE(GroupedMatMulAlltoAllv, GroupedMatmulAllToAllvTiling, 1);
}
