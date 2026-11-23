/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "gtest/gtest.h"
#include "flash_attn_param.h"
#include "infer_shape_case_executor.h"

namespace FlashAttnUT {
namespace {
using TD = gert::InfershapeContextPara::TensorDescription;
using Ops::Transformer::AnyValue;

std::vector<TD> Inputs()
{
    return {TD(gert::StorageShape({2, 128, 8, 128}, {2, 128, 8, 128}), ge::DT_FLOAT16, ge::FORMAT_ND),
            TD(gert::StorageShape({2, 256, 2, 128}, {2, 256, 2, 128}), ge::DT_FLOAT16, ge::FORMAT_ND),
            TD(gert::StorageShape({2, 256, 2, 128}, {2, 256, 2, 128}), ge::DT_FLOAT16, ge::FORMAT_ND)};
}

std::vector<TD> Outputs()
{
    return {TD(gert::StorageShape({2, 128, 8, 128}, {2, 128, 8, 128}), ge::DT_FLOAT16, ge::FORMAT_ND),
            TD(gert::StorageShape({}, {}), ge::DT_UNDEFINED, ge::FORMAT_NULL)};
}

// omitLayout: 省略该必选属性, 使 GetAttrPointer 返回 nullptr
std::vector<gert::InfershapeContextPara::OpAttr> Attrs(const std::string& omitLayout)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"softmax_scale", AnyValue::CreateFrom<float>(0.125F)},
        {"mask_mode", AnyValue::CreateFrom<int64_t>(0)},
        {"win_left", AnyValue::CreateFrom<int64_t>(-1)},
        {"win_right", AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_q", AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_kv", AnyValue::CreateFrom<int64_t>(-1)},
        {"layout_q", AnyValue::CreateFrom<std::string>("BSND")},
        {"layout_kv", AnyValue::CreateFrom<std::string>("BSND")},
        {"layout_out", AnyValue::CreateFrom<std::string>("BSND")},
        {"return_softmax_lse", AnyValue::CreateFrom<int64_t>(0)},
    };
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        if (it->attrName_ == omitLayout) {
            attrs.erase(it);
            break;
        }
    }
    return attrs;
}

// checkParamName: faker 按位置存属性, 省略非末位属性会使后续索引前移,
// 此时只能断言正式码与首错唯一性, 不能断言具体参数名
void RunAndExpect(const std::string& omitLayout, const std::string& expectCode, bool checkParamName)
{
    gert::InfershapeContextPara para("FlashAttn", Inputs(), Outputs(), Attrs(omitLayout), {1, 1, 1}, {1, 1});
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().error_id, expectCode);
    if (checkParamName) {
        EXPECT_NE(errors.front().error_message.find(omitLayout), std::string::npos);
    }
}

// 基线: 全部必选属性齐备时 InferShape 应成功且不上报任何错误码
TEST(FlashAttnInferShapeErrorCodeTest, BaselineSucceedsWithoutError)
{
    gert::InfershapeContextPara para("FlashAttn", Inputs(), Outputs(), Attrs(""), {1, 1, 1}, {1, 1});
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 128, 8, 128}});
    EXPECT_EQ(ErrorManager::GetInstance().GetRawErrorMessages().size(), 0U);
}

// 必选 layout 属性缺失: 属用户可构造的参数错误, 必须首报 EZ0037 而非 EZ9999
TEST(FlashAttnInferShapeErrorCodeTest, ReportsEz0037WhenLayoutQMissing)
{
    RunAndExpect("layout_q", "EZ0037", false);
}

TEST(FlashAttnInferShapeErrorCodeTest, ReportsEz0037WhenLayoutKvMissing)
{
    RunAndExpect("layout_kv", "EZ0037", false);
}

TEST(FlashAttnInferShapeErrorCodeTest, ReportsEz0037WhenLayoutOutMissing)
{
    RunAndExpect("layout_out", "EZ0037", false);
}

TEST(FlashAttnInferShapeErrorCodeTest, ReportsEz0037WhenReturnSoftmaxLseMissing)
{
    RunAndExpect("return_softmax_lse", "EZ0037", true);
}

} // namespace
} // namespace FlashAttnUT
