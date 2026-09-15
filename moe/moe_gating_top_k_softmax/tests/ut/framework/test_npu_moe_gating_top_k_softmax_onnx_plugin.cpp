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

#include "../../../framework/npu_moe_gating_top_k_softmax_onnx_plugin.cpp"

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, MissingAttributeEnvelope)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, EmptyAttributeArray)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, MissingAttributeArray)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, ParsesJsonValue0)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"k","type":2,"i":0}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, ParsesJsonValue1)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"k","type":2,"i":1}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 1);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, ParsesJsonValue8)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"k","type":2,"i":8}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 8);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, UnrelatedAttributeUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"unrelated","type":2,"i":5}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, WrongOnnxTypeUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"k","type":1,"i":7}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, MissingIntegerFieldDefaultsToZero)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"k","type":2}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, NegativeOneFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"k","type":2,"i":-1}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, MalformedJsonFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"(not-json)"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, WrongRootTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"([])"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, WrongArrayTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":{}})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, WrongEntryTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[4]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, StringIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"k","type":2,"i":"2"}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, FloatIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"k","type":2,"i":2.5}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeGatingTopKSoftmaxPluginTest, OutOfRangeIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"k","type":2,"i":18446744073709551615}]})"));
    dest.SetAttr("k", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeGatingTopKSoftmax(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("k", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}
