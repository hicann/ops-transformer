/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MC2_HCCL_TILING_REPORT_H
#define MC2_HCCL_TILING_REPORT_H
#include <cstdlib>
#include "graph/types.h"
#include "mc2_tiling_failure_report.h"
namespace Mc2Error {
// Caller validates the group length and all communication configuration fields.
template <typename Config, typename Init, typename Cc>
ge::graphStatus GetValidatedHcclTiling(Config& config, Init& init, Cc& cc, const char* opName)
{
    const char* ascendHome = std::getenv("ASCEND_HOME_PATH");
    if (ascendHome == nullptr || ascendHome[0] == '\0') {
        // Formal environment-code registration is unavailable in this host target.
        OP_LOGE(opName, "%s: required environment variable ASCEND_HOME_PATH is absent or empty", opName);
        return ge::GRAPH_FAILED;
    }
    const auto initRet = config.GetTiling(init);
    if (initRet != 0) {
        ReportTilingFailure(
            opName, "Mc2CcTilingConfig::GetTiling(init) failed to load the required rtGetSocVersion entry; rc=" +
                        std::to_string(initRet) + ", library=" + ascendHome +
                        "/lib64/libruntime.so; check the CANN runtime installation");
        return ge::GRAPH_FAILED;
    }
    const auto ccRet = config.GetTiling(cc);
    if (ccRet != 0) {
        OP_LOGE_WITHOUT_REPORT(opName, "Validated communication tiling configuration failed, rc=%u", ccRet);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace Mc2Error
#endif
