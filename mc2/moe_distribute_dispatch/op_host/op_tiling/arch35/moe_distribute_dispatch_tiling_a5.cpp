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
 * \file moe_distribute_dispatch_tiling_a5.cpp
 * \brief
 */

#include "moe_distribute_dispatch_tiling_a5.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
namespace optiling {

ge::graphStatus MoeDistributeDispatchTilingA5::MoeDistributeDispatchTilingFunc(gert::TilingContext* context)
{
    ge::graphStatus ret = MoeDistributeDispatchA3A5TilingFuncImpl(context);
    return ret;
}

ge::graphStatus MoeDistributeDispatchTilingA5::CheckEpWorldSizeAttrs(const char* nodeName,
                                                                     MoeDistributeDispatchTilingData& tilingData)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchInfo.epWorldSize;

    // 为支持在 A5 上的验证，放开 epWorldSize 为 2 或 4 的校验
    // 检验epWorldSize是否是2的倍数
    OP_TILING_CHECK(
        epWorldSize % 2 != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(), "divisible by 2"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK((256 % epWorldSize != 0) && (epWorldSize % 144 != 0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
                                              "2, 4, 8, 16, 32, 64, 128, 144, 256, or 288"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// A5 采用 MTE 方式通信，复用 A3 tiling
REGISTER_OPS_TILING_TEMPLATE(MoeDistributeDispatch, MoeDistributeDispatchTilingA5, 0);

} // namespace optiling
