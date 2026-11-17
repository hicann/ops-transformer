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
#include "infer_shape_case_executor.h"

namespace {
using Ops::Transformer::AnyValue;

gert::InfershapeContextPara MakeContext(const gert::StorageShape& query, const std::string& layout)
{
    return gert::InfershapeContextPara("NsaCompressAttentionInfer",
                                       {
                                           {query, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{16, 128, 256}, {16, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{16, 128, 128}, {16, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       },
                                       {
                                           {query, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {query, ge::DT_INT32, ge::FORMAT_ND},
                                       },
                                       {
                                           {"numHeads", AnyValue::CreateFrom<int64_t>(16)},
                                           {"numKeyValueHeads", AnyValue::CreateFrom<int64_t>(2)},
                                           {"selectBlockSize", AnyValue::CreateFrom<int64_t>(64)},
                                           {"selectBlockCount", AnyValue::CreateFrom<int64_t>(4)},
                                           {"compressBlockSize", AnyValue::CreateFrom<int64_t>(32)},
                                           {"compressStride", AnyValue::CreateFrom<int64_t>(16)},
                                           {"scaleValue", AnyValue::CreateFrom<float>(1.0F)},
                                           {"layout", AnyValue::CreateFrom<std::string>(layout)},
                                       });
}

TEST(NsaCompressAttentionInferShapeTest, InfersTndAttentionAndTopKOutputs)
{
    auto para = MakeContext({{8, 16, 128}, {8, 16, 128}}, "TND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{8, 16, 64}, {8, 2, 4}});
}

TEST(NsaCompressAttentionInferShapeTest, InfersBsndAttentionAndTopKOutputs)
{
    auto para = MakeContext({{4, 2, 16, 128}, {4, 2, 16, 128}}, "BSND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{4, 2, 16, 64}, {4, 2, 2, 4}});
}

TEST(NsaCompressAttentionInferShapeTest, RejectsUnsupportedLayout)
{
    auto para = MakeContext({{8, 16, 128}, {8, 16, 128}}, "BNSD");
    auto& errorManager = ErrorManager::GetInstance();
    (void)errorManager.GetRawErrorMessages();

    ExecuteTestCase(para, ge::GRAPH_FAILED);

    const auto errors = errorManager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().error_id, "EZ0026");
    EXPECT_NE(errors.front().error_message.find("layoutOptional"), std::string::npos);
    EXPECT_NE(errors.front().error_message.find("BNSD"), std::string::npos);
    EXPECT_NE(errors.front().error_message.find("Only TND and BSND are supported"), std::string::npos);
}
} // namespace
