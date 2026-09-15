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

#include "../../../framework/npu_moe_finalize_routing_v2_onnx_plugin.cpp"

TEST(OnnxMoeFinalizeRoutingV2PluginTest, MissingAttributeEnvelope)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, EmptyAttributeArray)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, MissingAttributeArray)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, ParsesJsonValue0)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"drop_pad_mode","type":2,"i":0}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, ParsesJsonValue1)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"drop_pad_mode","type":2,"i":1}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 1);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, ParsesJsonValue2)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"drop_pad_mode","type":2,"i":2}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 2);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, ParsesJsonValue3)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"drop_pad_mode","type":2,"i":3}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 3);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, UnrelatedAttributeUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"unrelated","type":2,"i":5}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, WrongOnnxTypeUsesMissingPolicy)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"drop_pad_mode","type":1,"i":7}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, MissingIntegerFieldDefaultsToZero)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"drop_pad_mode","type":2}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::SUCCESS);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 0);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, MalformedJsonFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"(not-json)"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, WrongRootTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"([])"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, WrongArrayTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":{}})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, WrongEntryTypeFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[4]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, StringIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"drop_pad_mode","type":2,"i":"2"}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, FloatIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"drop_pad_mode","type":2,"i":2.5}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}

TEST(OnnxMoeFinalizeRoutingV2PluginTest, OutOfRangeIntegerFailsWithoutChangingDestination)
{
    ge::Operator src("src", "TestOp");
    ge::Operator dest("dest", "TestOp");
    src.SetAttr("attribute",
                ge::AscendString(R"({"attribute":[{"name":"drop_pad_mode","type":2,"i":18446744073709551615}]})"));
    dest.SetAttr("drop_pad_mode", int64_t{9});

    ASSERT_EQ(domi::ParseParamsMoeFinalizeRoutingV2(src, dest), domi::FAILED);
    int64_t value = -1;
    ASSERT_EQ(dest.GetAttr("drop_pad_mode", value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(value, 9);
}
