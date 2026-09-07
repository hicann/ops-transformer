/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * /

/*!
 * \file distribute_barrier_extend_tiling.cc
 * \brief
 */

#include "../../../distribute_barrier/op_host/op_tiling/distribute_barrier_tiling_helper.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static ge::graphStatus DistributeBarrierExtendTilingFunc(gert::TilingContext *context)
{
    DistributeBarrierConfig config;
    config.contextIndex = INPUT_CONTEXT_EXTEND_INDEX; // 根据disributeBarrierExtend算子原型标志位初始化context索引
    config.xRefIndex = INPUT_X_EXTEND_INDEX;     // 根据disributeBarrierExtend算子原型标志位初始化xRef索引
    config.timeOutIndex = TIME_OUT_EXTEND_INDEX; // 根据disributeBarrierExtend算子原型标志位初始化timeOut索引
    config.elasticInfoIndex =
        ELASTIC_INFO_EXTEND_INDEX;        // 根据disributeBarrierExtend算子原型标志位初始化elasticInfo索引
    config.xRefOutIndex = OUTPUT_X_INDEX; // 根据disributeBarrierExtend算子原型标志位设置xRefOut索引
    config.attrGroupIndex = ATTR_GROUP_INDEX; // 根据disributeBarrierExtend算子原型标志位初始化attrGroupIndex索引
    config.attrWorldSizeIndex =
        ATTR_WORLD_SIZE_INDEX; // 根据disributeBarrierExtend算子原型标志位设置attrWorldSizeIndex索引
    config.isMc2Context = true;
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter DistributeBarrierExtend tiling");
    ge::graphStatus ret = optiling::DistributeBarrierExtendTilingFuncNew(context, config);
    return ret;
}

struct DistributeBarrierExtendCompileInfo {};
ge::graphStatus TilingParseForDistributeBarrierExtend(gert::TilingParseContext *context)
{
    const gert::TilingParseContext *const_context = context;
    // 避免未使用变量警告
    (void)const_context;
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DistributeBarrierExtend)
    .Tiling(DistributeBarrierExtendTilingFunc)
    .TilingParse<DistributeBarrierExtendCompileInfo>(TilingParseForDistributeBarrierExtend);
} // end of namespace optiling
