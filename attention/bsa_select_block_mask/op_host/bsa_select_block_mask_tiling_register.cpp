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
 * \file bsa_select_block_mask_tiling_register.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/bsa_select_block_mask_tiling_data.h"
#include "../op_kernel/bsa_select_block_mask_tiling_key.h"
#include "bsa_select_block_mask_tiling_base.h"

namespace optiling {

static ge::graphStatus BSASelectBlockMaskTilingFunc(gert::TilingContext* context)
{
    BSASelectBlockMaskTilingBase bsaTiling(context);
    bsaTiling.DoTiling();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBSASelectBlockMask([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start parsing tiling for BSASelectBlockMask.");
    auto compileInfoPtr = context->GetCompiledInfo<BSASelectBlockMaskCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BSASelectBlockMask)
    .Tiling(BSASelectBlockMaskTilingFunc)
    .TilingParse<BSASelectBlockMaskCompileInfo>(TilingParseForBSASelectBlockMask);
} // namespace optiling
