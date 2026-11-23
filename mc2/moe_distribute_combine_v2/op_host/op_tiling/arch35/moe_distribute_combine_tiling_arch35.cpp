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
 * \file moe_distribute_combine_tiling_arch35.cpp
 * \brief
 */

#include "moe_distribute_combine_tiling_arch35.h"

#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <sys/types.h>
#include <queue>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <unistd.h>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../../op_kernel/moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_tiling_arch35.h"
using namespace Mc2Tiling;

namespace {
const std::string OP_NAME = "MoeDistributeCombineA5";
} // namespace

namespace optiling {

uint64_t MoeDistributeCombineV2TilingFuncA5::CalTilingKey(uint32_t commQuantMode, bool isLayered)
{
    uint32_t quantMode = TILINGKEY_NO_QUANT;
    uint32_t hierarchy = TILINGKEY_TPL_MTE;
    if (commQuantMode >= TILINGKEY_INT8_QUANT) {
        quantMode = commQuantMode;
    }
    uint32_t archTag = isA6_ ? TILINGKEY_TPL_A6 : TILINGKEY_TPL_A5;
    uint64_t tilingKey = GET_TPL_TILING_KEY(quantMode, hierarchy, archTag);
    return tilingKey;
}

ge::graphStatus MoeDistributeCombineV2TilingFuncA5::MoeDistributeCombineTilingFuncImpl(gert::TilingContext* context,
                                                                                       const CombineV2Config& config)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE(OP_NAME, "Fail to get tiling context."), return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE(OP_NAME, "Fail to get nodeName."), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>((config.attrCommAlgIndex)));
    // 检查 commAlg 参数合法性校验
    bool isNullOrEmpty = (commAlgPtr == nullptr) || (std::strlen(commAlgPtr) == 0);
    OP_TILING_CHECK(!(isNullOrEmpty || config.isRoce),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "commAlg", commAlgPtr != nullptr ? commAlgPtr : "null",
                                              "expected empty; hierarchy requires the A5 V3 Context entry"),
                    return ge::GRAPH_FAILED);
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    if (npuArch == Ops::Base::DAV_9201 || npuArch == Ops::Base::DAV_9202) {
        isA6_ = true;
    }
    // 默认空指针和空字符走 MTE 方式，MTE 调用 A3 tiling 实现
    return MoeDistributeCombineA3TilingFuncImpl(context, config);
}

static ge::graphStatus MoeDistributeCombineV2TilingFuncImplA5(gert::TilingContext* context)
{
    MoeDistributeCombineV2TilingFuncA5 impl;
    return impl.MoeDistributeCombineV2TilingFunc(context);
}

struct MoeDistributeCombineCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeCombineV2(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineV2)
    .Tiling(MoeDistributeCombineV2TilingFuncImplA5)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombineV2);

} // namespace optiling
