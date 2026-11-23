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
#include "flash_mla_param.h"
#include "infer_shape_case_executor.h"

namespace FlashMlaUT {
namespace {
using TD = gert::InfershapeContextPara::TensorDescription;
using Ops::Transformer::AnyValue;

// 合法基线取自 test_flash_mla_shape_infershape.csv 首条 SUCCESS 用例
// InferShape_BSND_PA_BBND_FLOAT16_Lse0; FlashMla InferShape 只读 INPUT_IDX_Q(0) 与输出 0/1
std::vector<TD> Inputs()
{
    return {TD(gert::StorageShape({2, 2, 3, 576}, {2, 2, 3, 576}), ge::DT_FLOAT16, ge::FORMAT_ND)};
}

std::vector<TD> Outputs()
{
    return {TD(gert::StorageShape({2, 2, 3, 512}, {2, 2, 3, 512}), ge::DT_FLOAT16, ge::FORMAT_ND),
            TD(gert::StorageShape({0}, {0}), ge::DT_FLOAT, ge::FORMAT_ND)};
}

// 属性顺序须与 flash_mla_def.cpp 一致(GetAttrPointer 按下标取): head_dim_v=0 ... layout_q=7 ...
// return_softmax_lse=10。omit 指定要省略的必选属性, 使其 GetAttrPointer 返回 nullptr
std::vector<gert::InfershapeContextPara::OpAttr> Attrs(const std::string& omit)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"head_dim_v", AnyValue::CreateFrom<int64_t>(512)},
        {"softmax_scale", AnyValue::CreateFrom<float>(0.041666666666666664F)},
        {"mask_mode", AnyValue::CreateFrom<int64_t>(0)},
        {"win_left", AnyValue::CreateFrom<int64_t>(-1)},
        {"win_right", AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_q", AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_kv", AnyValue::CreateFrom<int64_t>(-1)},
        {"layout_q", AnyValue::CreateFrom<std::string>("BSND")},
        {"layout_kv", AnyValue::CreateFrom<std::string>("PA_BBND")},
        {"layout_out", AnyValue::CreateFrom<std::string>("BSND")},
        {"return_softmax_lse", AnyValue::CreateFrom<int64_t>(0)},
    };
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        if (it->attrName_ == omit) {
            attrs.erase(it);
            break;
        }
    }
    return attrs;
}

// checkParamName: faker 按位置存属性, 省略非末位属性会使后续下标前移,
// 此时只能断言正式码与首错唯一性, 不能断言具体参数名(与 flash_attn 侧同处理)
void RunAndExpect(const std::string& omit, bool checkParamName)
{
    gert::InfershapeContextPara para("FlashMla", Inputs(), Outputs(), Attrs(omit));
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().error_id, "EZ0037");
    if (checkParamName) {
        EXPECT_NE(errors.front().error_message.find(omit), std::string::npos)
            << "actual message: " << errors.front().error_message;
    }
}

// 基线: 必选属性齐备时 InferShape 应成功且不上报任何错误码
TEST(FlashMlaInferShapeErrorCodeTest, BaselineSucceedsWithoutError)
{
    gert::InfershapeContextPara para("FlashMla", Inputs(), Outputs(), Attrs(""));
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 2, 3, 512}, {0}});
    EXPECT_EQ(ErrorManager::GetInstance().GetRawErrorMessages().size(), 0U);
}

// 用户可构造: 必选属性缺失 -> EZ0037 (flash_mla_infershape.cpp:48/49/50/53/57)
TEST(FlashMlaInferShapeErrorCodeTest, ReportsEz0037WhenLayoutQMissing)
{
    RunAndExpect("layout_q", false);
}

TEST(FlashMlaInferShapeErrorCodeTest, ReportsEz0037WhenLayoutKvMissing)
{
    RunAndExpect("layout_kv", false);
}

TEST(FlashMlaInferShapeErrorCodeTest, ReportsEz0037WhenLayoutOutMissing)
{
    RunAndExpect("layout_out", false);
}

TEST(FlashMlaInferShapeErrorCodeTest, ReportsEz0037WhenHeadDimVMissing)
{
    RunAndExpect("head_dim_v", false);
}

TEST(FlashMlaInferShapeErrorCodeTest, ReportsEz0037WhenReturnSoftmaxLseMissing)
{
    RunAndExpect("return_softmax_lse", true);
}

} // namespace
} // namespace FlashMlaUT
