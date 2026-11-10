/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MC2_TILING_FAILURE_REPORT_H
#define MC2_TILING_FAILURE_REPORT_H
#include <string>
#include <vector>
#include "mc2_log.h"
namespace Mc2Error {
inline void ReportTilingFailure(const char* opName, const std::string& reason)
{
    const char* name = opName == nullptr ? "Mc2Tiling" : opName;
    OP_LOGE(name, "%s", reason.c_str());
}
} // namespace Mc2Error
#endif
