/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>

#include "../../../framework/npu_moe_init_routing_onnx_plugin.cpp"

TEST(OnnxMoeInitRoutingPluginTest, MissingAttributeEnvelope)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, EmptyAttributeArray)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, MissingAttributeArray)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, ParsesJsonValue0)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":2,"i":0}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeInitRoutingPluginTest, ParsesJsonValue1)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":2,"i":1}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 1);
}

TEST(OnnxMoeInitRoutingPluginTest, ParsesJsonValue8)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":2,"i":8}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 8);
}

TEST(OnnxMoeInitRoutingPluginTest, UnrelatedAttributeUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"unrelated","type":2,"i":5}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, WrongOnnxTypeUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":1,"i":7}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, MissingIntegerFieldDefaultsToZero)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":2}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeInitRoutingPluginTest, ParsesNegativeOne)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":2,"i":-1}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, -1);
}

TEST(OnnxMoeInitRoutingPluginTest, MalformedJsonFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"(not-json)"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, WrongRootTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"([])"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, WrongArrayTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":{}})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, WrongEntryTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[4]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, StringIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":2,"i":"2"}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, FloatIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":2,"i":2.5}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, OutOfRangeIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute",
                ge::AscendString(R"({"attribute":[{"name":"active_num","type":2,"i":18446744073709551615}]})"));
    dest.SetAttr("active_num", int64_t{9});

    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeInitRoutingPluginTest, DuplicateRequiredAttributeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute",
                ge::AscendString(
                    R"({"attribute":[{"name":"active_num","type":2,"i":1},{"name":"active_num","type":2,"i":2}]})"));
    dest.SetAttr("active_num", int64_t{9});
    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, int64_t{9});
}

TEST(OnnxMoeInitRoutingPluginTest, UnrelatedAttributeDoesNotHideRequiredAttribute)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr(
        "attribute",
        ge::AscendString(R"({"attribute":[{"name":"other","type":2,"i":3},{"name":"active_num","type":2,"i":2}]})"));
    dest.SetAttr("active_num", int64_t{9});
    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, int64_t{2});
}

TEST(OnnxMoeInitRoutingPluginTest, PreservesInt64Value)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":2,"i":2147483648}]})"));
    dest.SetAttr("active_num", int64_t{9});
    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, int64_t{2147483648});
}

TEST(OnnxMoeInitRoutingPluginTest, MalformedEntryAfterValueDoesNotChangeDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"active_num","type":2,"i":2},4]})"));
    dest.SetAttr("active_num", int64_t{9});
    ASSERT_EQ(domi::ParseParamsNpuMoeInitRouting(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("active_num", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, int64_t{9});
}
