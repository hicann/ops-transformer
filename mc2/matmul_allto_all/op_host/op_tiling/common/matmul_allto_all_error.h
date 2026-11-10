/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MATMUL_ALLTO_ALL_ERROR_H
#define MATMUL_ALLTO_ALL_ERROR_H

#include <cstdlib>
#include <string>
#include <vector>
#include "op_common/log/log.h"

namespace MC2Tiling {
inline void ReportMatmulAlltoAllTilingFailure(const char* opName, const std::string& reason)
{
    OP_LOGE(opName, "%s", reason.c_str());
}

template <typename Config, typename Init, typename Cc>
ge::graphStatus GetMatmulAlltoAllHcclTiling(Config& config, Init& init, Cc& cc, const char* opName)
{
    const char* ascendHome = std::getenv("ASCEND_HOME_PATH");
    if (ascendHome == nullptr || ascendHome[0] == '\0') {
        OP_LOGE(opName, "Cannot initialize communication tiling: ASCEND_HOME_PATH is absent or empty");
        return ge::GRAPH_FAILED;
    }
    const auto initRet = config.GetTiling(init);
    if (initRet != 0) {
        ReportMatmulAlltoAllTilingFailure(
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
} // namespace MC2Tiling
#endif
