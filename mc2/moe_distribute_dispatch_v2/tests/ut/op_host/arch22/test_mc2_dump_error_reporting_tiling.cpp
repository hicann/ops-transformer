/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "mc2_exception_dump.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
namespace {
template <typename Action>
void ExpectDumpPreservesFirstError(Action action)
{
    auto& manager = ErrorManager::GetInstance();
    (void)manager.GetRawErrorMessages();
    OP_LOGE_FOR_INVALID_VALUE("OriginalOperator", "ep_world_size", "3", "2, 4 or 8");
    const auto original = manager.GetRawErrorMessages();
    ASSERT_EQ(original.size(), 1U);
    ASSERT_EQ(original[0].error_id, "EZ0024");
    ASSERT_EQ(manager.SetRawErrorMessages(original), 0);
    action();
    const auto after = manager.GetRawErrorMessages();
    ASSERT_EQ(after.size(), original.size());
    EXPECT_EQ(after[0].error_id, original[0].error_id);
    EXPECT_EQ(after[0].args_map, original[0].args_map);
}
} // namespace

TEST(Mc2DumpErrorReporting, EmptyDumpDirectoryPreservesFirstError)
{
    ExpectDumpPreservesFirstError([] { EXPECT_EQ(Mc2Exception::DumpToFile("", "unused", 0, nullptr, 0), -1); });
}

TEST(Mc2DumpErrorReporting, NullA2DumpGroupPreservesFirstError)
{
    ExpectDumpPreservesFirstError([] {
        std::vector<uint8_t> buffer;
        EXPECT_EQ(Mc2Exception::ProcessArgsForA2(nullptr, buffer), -1);
    });
}
#endif
