/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_KDA_FWD_TESTS_UT_OP_API_STUB_CHUNK_KDA_LAUNCHER_TEST_HOOK_H
#define CHUNK_KDA_FWD_TESTS_UT_OP_API_STUB_CHUNK_KDA_LAUNCHER_TEST_HOOK_H

#include <cstdlib>
#include <string>

#include "aclnn/aclnn_base.h"

namespace chunk_kda_ut {
constexpr const char* LAUNCHER_STATUS_ENV = "CHUNK_KDA_UT_LAUNCHER_STATUS";
constexpr const char* CONVERT_FAIL_CALL_ENV = "CHUNK_KDA_UT_CONVERT_FAIL_CALL";

inline void SetNextLauncherStatus(aclnnStatus status)
{
    const std::string value = std::to_string(static_cast<int32_t>(status));
    (void)setenv(LAUNCHER_STATUS_ENV, value.c_str(), 1);
}

inline aclnnStatus ConsumeLauncherStatus()
{
    const char* value = std::getenv(LAUNCHER_STATUS_ENV);
    if (value == nullptr) {
        return ACLNN_SUCCESS;
    }
    const auto status = static_cast<aclnnStatus>(std::strtol(value, nullptr, 10));
    (void)unsetenv(LAUNCHER_STATUS_ENV);
    return status;
}

inline bool HasPendingLauncherStatus()
{
    return std::getenv(LAUNCHER_STATUS_ENV) != nullptr;
}

inline void ClearLauncherStatus()
{
    (void)unsetenv(LAUNCHER_STATUS_ENV);
}

inline void SetConvertToTensorFailCall(int32_t failCall)
{
    const std::string value = std::to_string(failCall);
    (void)setenv(CONVERT_FAIL_CALL_ENV, value.c_str(), 1);
}

inline bool ShouldFailConvertToTensor()
{
    const char* value = std::getenv(CONVERT_FAIL_CALL_ENV);
    if (value == nullptr) {
        return false;
    }
    const int failCall = static_cast<int>(std::strtol(value, nullptr, 10));
    if (failCall <= 1) {
        (void)unsetenv(CONVERT_FAIL_CALL_ENV);
        return failCall == 1;
    }
    const std::string nextValue = std::to_string(failCall - 1);
    (void)setenv(CONVERT_FAIL_CALL_ENV, nextValue.c_str(), 1);
    return false;
}

inline void ClearConvertToTensorFailCall()
{
    (void)unsetenv(CONVERT_FAIL_CALL_ENV);
}
} // namespace chunk_kda_ut

#endif // CHUNK_KDA_FWD_TESTS_UT_OP_API_STUB_CHUNK_KDA_LAUNCHER_TEST_HOOK_H
