/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_INTERNAL_FALLBACK_ASSERTIONS_H
#define MC2_INTERNAL_FALLBACK_ASSERTIONS_H
#include <string>
#include <gtest/gtest.h>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
namespace Mc2ErrorCodeUT {
// This checks the scope-constrained internal fallback, not formal-code closure.
inline void ExpectInternalFallbackReport(const std::string& reason)
{
    auto& manager = ErrorManager::GetInstance();
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ9999");
    const auto metadata = errors[0].error_message.find("[FUNC:");
    ASSERT_NE(metadata, std::string::npos);
    EXPECT_EQ(errors[0].error_message.substr(0, metadata), reason);
    ASSERT_EQ(manager.SetRawErrorMessages(errors), 0);
    const auto formatted = manager.GetErrorMessage();
    EXPECT_NE(formatted.find("EZ9999"), std::string::npos);
    EXPECT_NE(formatted.find(reason), std::string::npos);
}
} // namespace Mc2ErrorCodeUT
#endif
