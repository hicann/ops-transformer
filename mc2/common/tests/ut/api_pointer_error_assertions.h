/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_API_POINTER_ERROR_ASSERTIONS_H
#define MC2_API_POINTER_ERROR_ASSERTIONS_H

#include "formal_error_assertions.h"
#include "opdev/op_errno.h"

namespace Mc2ErrorCodeUT {
inline void ExpectRequiredApiPointerError(const char* functionName, const char* parameterName)
{
    const auto reports = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_EQ(reports[0].error_id, "EZ1011");
    ASSERT_EQ(reports[0].args_map.count("functionName"), 1U);
    ASSERT_EQ(reports[0].args_map.count("paraName"), 1U);
    EXPECT_EQ(reports[0].args_map.at("functionName"), functionName);
    EXPECT_EQ(reports[0].args_map.at("paraName"), parameterName);
    ASSERT_EQ(ErrorManager::GetInstance().SetRawErrorMessages(reports), 0);
    const auto message = ErrorManager::GetInstance().GetErrorMessage();
    EXPECT_NE(message.find(functionName), std::string::npos);
    EXPECT_NE(message.find(parameterName), std::string::npos);
}
} // namespace Mc2ErrorCodeUT
#endif
