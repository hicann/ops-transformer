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
 * \file moe_distribute_combine_add_rms_norm_tiling.cpp
 * \brief
 */

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "../../../moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.h"
#include "moe_distribute_combine_add_rms_norm_tiling_base.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {
ge::graphStatus MoeDistributeCombineAddRmsNormTilingFuncBase::MoeDistributeCombineAddRmsNormTilingFunc(
    gert::TilingContext *context)
{
    ge::graphStatus ret = MoeDistributeCombineV2TilingFunc(context);
    return ret;
}

} // namespace optiling
