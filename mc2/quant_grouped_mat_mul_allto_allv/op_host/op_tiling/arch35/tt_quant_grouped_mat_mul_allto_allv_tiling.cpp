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
 * \file tt_quant_grouped_mat_mul_allto_allv_tiling.cpp
 * \brief
 */

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "tt_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "../quant_grouped_mat_mul_allto_allv_tiling_adapter.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include "mc2_comm_utils.h"
#include <numeric>

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace Mc2Tiling {

bool TTQuantGroupedMatmulAllToAllvTiling::IsCapable()
{
    QuantModePair mode = GetQuantMode(context_, opName_);
    OP_TILING_CHECK(mode == QUANT_PAIR_ERROR,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "quantMode", "QUANT_PAIR_ERROR", "valid quant mode"),
                    return false);
    if (mode == QUANT_PAIR_TT) {
        OP_LOGI(opName_, "TTQuantGroupedMatmulAllToAllvTiling TT mode capable.");
        return true;
    }
    OP_LOGI(opName_, "Skip TTQuantGroupedMatmulAllToAllvTiling TT.");
    return false;
}

ge::graphStatus TTQuantGroupedMatmulAllToAllvTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspaces[0] = workSpaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t TTQuantGroupedMatmulAllToAllvTiling::GetTilingKey() const
{
    uint8_t commMode = 0;
    if (QuantGroupedMatmulAllToAllvTilingCommon::QuantGetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans,
                                                  localParams_.isMmWeightTrans, commMode);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, commMode, tilingKey);
    return tilingKey;
}

// 注册tiling类
REGISTER_OPS_TILING_TEMPLATE(QuantGroupedMatMulAlltoAllv, TTQuantGroupedMatmulAllToAllvTiling, 0);

} // namespace Mc2Tiling
