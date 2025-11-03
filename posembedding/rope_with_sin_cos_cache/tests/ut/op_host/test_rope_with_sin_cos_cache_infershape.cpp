/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RopeWithSinCosCache : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RopeWithSinCosCache Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RopeWithSinCosCache Proto Test TearDown" << std::endl;
    }
};

TEST_F(RopeWithSinCosCache, rope_with_sin_cos_cache_bf16_true)
{
    std::vector<int64_t> mropeParams{0, 0, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "RopeWithSinCosCache",
        {
            // input info
            {{{48}, {48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(mropeParams)},
            {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RopeWithSinCosCache, rope_with_sin_cos_cache_fp16_true)
{
    std::vector<int64_t> mropeParams{0, 0, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "RopeWithSinCosCache",
        {
            // input info
            {{{48}, {48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(mropeParams)},
            {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RopeWithSinCosCache, rope_with_sin_cos_cache_fp32_true)
{
    std::vector<int64_t> mropeParams{0, 0, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "RopeWithSinCosCache",
        {
            // input info
            {{{48}, {48}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{48, 256}, {48, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{48, 512}, {48, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{48, 128}, {48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"numQHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"numKHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"headSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"mropeSection", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(mropeParams)},
            {"qstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"kstride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"isNeoxStyle", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}