/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "gtest/gtest.h"
#include "tiling_case_executor.h"

namespace {
using Ops::Transformer::AnyValue;

struct TilingCase {
    const char* name;
    ge::DataType queryDtype;
    int64_t compressBlockSize;
    const char* errorCode;
    const char* parameter;
    const char* actualValue;
    const char* constraint;
};

class NsaCompressAttentionInferTilingTest : public testing::TestWithParam<TilingCase> {};

TEST_P(NsaCompressAttentionInferTilingTest, ReportsInvalidInput)
{
    const auto& test = GetParam();
    int64_t cmpSeqLengths[] = {128, 128, 128, 128};
    // The context builder requires a compile-info slot even though this tiling reads the platform slot.
    int unusedCompileInfo = 0;
    gert::TilingContextPara para("NsaCompressAttentionInfer",
                                 {
                                     {{{4, 16, 64}, {4, 16, 64}}, test.queryDtype, ge::FORMAT_ND},
                                     {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{}, ge::DT_BOOL, ge::FORMAT_ND},
                                     {{{4, 4}, {4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, cmpSeqLengths},
                                     {{}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{}, ge::DT_BOOL, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{4, 16, 64}, {4, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{4, 1, 1}, {4, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {"numHeads", AnyValue::CreateFrom<int64_t>(16)},
                                     {"numKeyValueHeads", AnyValue::CreateFrom<int64_t>(1)},
                                     {"selectBlockSize", AnyValue::CreateFrom<int64_t>(64)},
                                     {"selectBlockCount", AnyValue::CreateFrom<int64_t>(1)},
                                     {"compressBlockSize", AnyValue::CreateFrom<int64_t>(test.compressBlockSize)},
                                     {"compressStride", AnyValue::CreateFrom<int64_t>(16)},
                                     {"scaleValue", AnyValue::CreateFrom<float>(1.0F)},
                                     {"layout", AnyValue::CreateFrom<std::string>("TND")},
                                     {"pageBlockSize", AnyValue::CreateFrom<int64_t>(128)},
                                     {"sparseMode", AnyValue::CreateFrom<int64_t>(0)},
                                 },
                                 &unusedCompileInfo);

    auto& errorManager = ErrorManager::GetInstance();
    (void)errorManager.GetRawErrorMessages();

    ExecuteTestCase(para, ge::GRAPH_FAILED);

    const auto errors = errorManager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().error_id, test.errorCode);
    EXPECT_NE(errors.front().error_message.find(test.parameter), std::string::npos);
    EXPECT_NE(errors.front().error_message.find(test.actualValue), std::string::npos);
    EXPECT_NE(errors.front().error_message.find(test.constraint), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidTilingInput, NsaCompressAttentionInferTilingTest,
    testing::Values(TilingCase{"QueryDtype", ge::DT_FLOAT, 32, "EZ0020", "query", "incorrect dtype DT_FLOAT.",
                               "Only float16 and bfloat16 are supported"},
                    TilingCase{"UnalignedCompressBlock", ge::DT_FLOAT16, 18, "EZ0026", "compressBlockSize",
                               "incorrect value 18.", "Must be a multiple of 16"}),
    [](const testing::TestParamInfo<TilingCase>& info) { return info.param.name; });
} // namespace
