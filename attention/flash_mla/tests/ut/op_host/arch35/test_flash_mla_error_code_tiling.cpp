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
#include "../flash_mla_param.h"
#include "tiling_case_executor.h"

namespace FlashMlaUT {
namespace {
using TD = gert::TilingContextPara::TensorDescription;
using Ops::Transformer::AnyValue;

// 合法基线取自 test_flash_mla_tiling.csv 首条 SUCCESS 用例
// BSND_PA_BBND_FLOAT16_N1_Block16_Lse0_Mask0
constexpr int64_t HEAD_DIM_V = 512;
constexpr int64_t ROPE_DIM = 64;

TD Q(int64_t lastDim = HEAD_DIM_V + ROPE_DIM, int64_t batch = 2)
{
    return TD(gert::StorageShape({batch, 2, 1, lastDim}, {batch, 2, 1, lastDim}), ge::DT_FLOAT16, ge::FORMAT_ND);
}
TD KCache(int64_t lastDim = HEAD_DIM_V + ROPE_DIM)
{
    return TD(gert::StorageShape({4, 16, 1, lastDim}, {4, 16, 1, lastDim}), ge::DT_FLOAT16, ge::FORMAT_ND);
}

struct Para {
    TD q = Q();
    TD kCache = KCache();
    TD attnMask = TD_DEFAULT;
    int64_t maskMode = 0;
};

gert::TilingContextPara MakePara(const Para& p)
{
    // static: TilingContextPara 只保存指针, 局部变量会在返回后悬垂(实测段错误)
    static optiling::flash_mla::FlashMlaCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"head_dim_v", AnyValue::CreateFrom<int64_t>(HEAD_DIM_V)},
        {"softmax_scale", AnyValue::CreateFrom<float>(0.041666666666666664F)},
        {"mask_mode", AnyValue::CreateFrom<int64_t>(p.maskMode)},
        {"win_left", AnyValue::CreateFrom<int64_t>(-1)},
        {"win_right", AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_q", AnyValue::CreateFrom<int64_t>(2)},
        {"max_seqlen_kv", AnyValue::CreateFrom<int64_t>(-1)},
        {"layout_q", AnyValue::CreateFrom<std::string>("BSND")},
        {"layout_kv", AnyValue::CreateFrom<std::string>("PA_BBND")},
        {"layout_out", AnyValue::CreateFrom<std::string>("BSND")},
        {"return_softmax_lse", AnyValue::CreateFrom<int64_t>(0)},
    };
    TD blockTable(gert::StorageShape({2, 2}, {2, 2}), ge::DT_INT32, ge::FORMAT_ND);
    TD cacheSeqlens(gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND);
    TD metadata(gert::StorageShape({4096}, {4096}), ge::DT_INT32, ge::FORMAT_ND);
    TD attnOut(gert::StorageShape({2, 2, 1, HEAD_DIM_V}, {2, 2, 1, HEAD_DIM_V}), ge::DT_FLOAT16, ge::FORMAT_ND);
    TD softmaxLse(gert::StorageShape({0}, {0}), ge::DT_FLOAT, ge::FORMAT_ND);
    // 实例数须与 GetTensorGE 口径一致: 未提供的可选输入填 0, 否则 faker 会用 TD_DEFAULT
    // (空 shape + DT_UNDEFINED) 建张量而崩溃
    const uint32_t maskInstance = (p.attnMask.dtype_ == ge::DT_UNDEFINED) ? 0U : 1U;
    return gert::TilingContextPara(
        "FlashMla", {p.q, p.kCache, blockTable, cacheSeqlens, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, p.attnMask, metadata},
        {attnOut, softmaxLse}, attrs, {1, 1, 1, 1, 0, 0, 0, maskInstance, 1}, {1, 1}, &compileInfo, "Ascend950", 64,
        262144, 8192);
}

void ExpectSingleError(const std::string& expectCode, const std::string& expectField)
{
    const auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().error_id, expectCode);
    if (!expectField.empty()) {
        EXPECT_NE(errors.front().error_message.find(expectField), std::string::npos)
            << "actual message: " << errors.front().error_message;
    }
}

void RunAndExpect(const Para& p, const std::string& expectCode, const std::string& expectField)
{
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    ExecuteTestCase(MakePara(p), ge::GRAPH_FAILED, UINT64_MAX, "", {}, 0, true);
    ExpectSingleError(expectCode, expectField);
}

TEST(FlashMlaErrorCodeTest, BaselineSucceedsWithoutError)
{
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    ExecuteTestCase(MakePara(Para()), ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, 0, true);
    EXPECT_EQ(ErrorManager::GetInstance().GetRawErrorMessages().size(), 0U);
}

// 用户可构造: q 某维为 0 -> EZ0015 (flash_mla_tiling_info_parser.cpp GetEmptyTensorFlag)
TEST(FlashMlaErrorCodeTest, ReportsEz0015WhenQueryHasEmptyDim)
{
    Para p;
    p.q = Q(HEAD_DIM_V + ROPE_DIM, 0);
    RunAndExpect(p, "EZ0015", "q");
}

// 用户可构造: q 末维 != head_dim_v + rope(64) -> EZ0015 (GetQkNopeHeadDim)
TEST(FlashMlaErrorCodeTest, ReportsEz0015WhenQueryLastDimMismatch)
{
    Para p;
    p.q = Q(HEAD_DIM_V + ROPE_DIM - 1);
    RunAndExpect(p, "EZ0015", "q");
}

// 用户可构造: k_cache 末维 != head_dim_v + rope(64) -> EZ0015 (GetValueHeadDim)
TEST(FlashMlaErrorCodeTest, ReportsEz0015WhenKCacheLastDimMismatch)
{
    Para p;
    p.kCache = KCache(HEAD_DIM_V + ROPE_DIM - 1);
    RunAndExpect(p, "EZ0015", "k_cache");
}

} // namespace
} // namespace FlashMlaUT
