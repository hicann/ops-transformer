/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fag_feature_checker.h"

namespace optiling {

// GS1S2 uses layout-specific newDeter schedules, including native packed TND.
// BN2 single-block already has one writer per head. Routing sends deterministic
// BN2 MultiBlk to GS1S2 and DecideSwizzle always selects the non-swizzle path.
ge::graphStatus FagDeterministicChecker::Check(FagCheckCtx&)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
