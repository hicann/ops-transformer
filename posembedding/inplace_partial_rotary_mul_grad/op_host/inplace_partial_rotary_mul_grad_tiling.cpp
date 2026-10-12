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
 * \file inplace_partial_rotary_mul_grad_tiling.cpp
 * \brief
 */
#include "inplace_partial_rotary_mul_grad_regbase_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
ge::graphStatus Tiling4InplacePartialRotaryMulGrad(gert::TilingContext* context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForInplacePartialRotaryMulGrad(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplacePartialRotaryMulGrad)
    .Tiling(Tiling4InplacePartialRotaryMulGrad)
    .TilingParse<InplacePartialRotaryMulGradCompileInfo>(TilingPrepareForInplacePartialRotaryMulGrad);
} // namespace optiling
