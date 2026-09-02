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
 * \file moe_log.h
 * \brief
 */

#ifndef MOE_LOG_H
#define MOE_LOG_H
#include <stdarg.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "base/err_msg.h"
#include "log/log.h"
#include "securec.h"

#if __has_include("error_manager/error_manager.h")
#include "error_manager/error_manager.h"
#else
#include "err_mgr.h"
#endif

inline const char *MoeConvertStringToCstr(const std::string &str)
{
    return str.c_str();
}

namespace ops {
#define MOE_OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret) \
    if ((ptr) == nullptr) { \
        const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
        OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr); \
        REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr); \
        return ret; \
    }
} // namespace ops

namespace optiling {
#define MOE_VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) \
    do { \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
        REPORT_INNER_ERR_MSG("E89999", "op[%s], " err_msg, MoeConvertStringToCstr(Ops::Base::GetOpInfo(op_name)), \
                             ##__VA_ARGS__); \
    } while (0)
#define MOE_OP_TILING_CHECK(cond, log_func, expr) \
    do { \
        if (cond) { \
            log_func; \
            expr; \
        } \
    } while (0)
} // namespace optiling
#endif
