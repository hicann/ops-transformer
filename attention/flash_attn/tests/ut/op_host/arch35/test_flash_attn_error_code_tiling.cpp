/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "gtest/gtest.h"
#include "../flash_attn_param.h"
#include "tiling_case_executor.h"

namespace FlashAttnUT {
namespace {
using TD = gert::TilingContextPara::TensorDescription;
using Ops::Transformer::AnyValue;

constexpr uint64_t CORE_NUM = 64;
constexpr uint64_t UB_SIZE = 262144;
constexpr uint64_t TILING_DATA_SIZE = 8192;

TD Q()
{
    return TD(gert::StorageShape({2, 128, 8, 128}, {2, 128, 8, 128}), ge::DT_FLOAT16, ge::FORMAT_ND);
}
TD Kv()
{
    return TD(gert::StorageShape({2, 256, 2, 128}, {2, 256, 2, 128}), ge::DT_FLOAT16, ge::FORMAT_ND);
}
TD AttnOut()
{
    return TD(gert::StorageShape({2, 128, 8, 128}, {2, 128, 8, 128}), ge::DT_FLOAT16, ge::FORMAT_ND);
}
TD Meta()
{
    return TD(gert::StorageShape({1}, {1}), ge::DT_INT32, ge::FORMAT_ND);
}

// 可覆参数 builder: 未指定的字段取 BSND/FP16 合法基线(与 test_flash_attn_tiling.csv 首条用例一致)
struct Para {
    TD q = Q();
    TD metadata = Meta();
    std::string layoutQ = "BSND";
    std::string layoutKv = "BSND";
    std::string layoutOut = "BSND";
    int64_t maskMode = 0;
    TD attnMask = TD_DEFAULT;
    uint64_t coreNum = CORE_NUM;
};

gert::TilingContextPara MakePara(const Para& p)
{
    // static: TilingContextPara 只保存指针, 局部变量会在返回后悬垂
    static FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"softmax_scale", AnyValue::CreateFrom<float>(0.125F)},
        {"mask_mode", AnyValue::CreateFrom<int64_t>(p.maskMode)},
        {"win_left", AnyValue::CreateFrom<int64_t>(-1)},
        {"win_right", AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_q", AnyValue::CreateFrom<int64_t>(-1)},
        {"max_seqlen_kv", AnyValue::CreateFrom<int64_t>(-1)},
        {"layout_q", AnyValue::CreateFrom<std::string>(p.layoutQ)},
        {"layout_kv", AnyValue::CreateFrom<std::string>(p.layoutKv)},
        {"layout_out", AnyValue::CreateFrom<std::string>(p.layoutOut)},
        {"return_softmax_lse", AnyValue::CreateFrom<int64_t>(0)},
    };
    return gert::TilingContextPara("FlashAttn",
                                   {p.q, Kv(), Kv(), TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT, TD_DEFAULT,
                                    TD_DEFAULT, p.attnMask, p.metadata},
                                   {AttnOut(), TD_DEFAULT}, attrs, &compileInfo, "Ascend950", p.coreNum, UB_SIZE,
                                   TILING_DATA_SIZE);
}

// 断言"一次故障只有一个 ErrorManager 首错", 且首错为期望正式码并携带关键诊断字段
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
    ExecuteTestCase(MakePara(p), ge::GRAPH_FAILED);
    ExpectSingleError(expectCode, expectField);
}

// 基线: 合法输入应 tiling 成功且零上报
TEST(FlashAttnErrorCodeTest, BaselineSucceedsWithoutError)
{
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    ExecuteTestCase(MakePara(Para()), ge::GRAPH_SUCCESS, UINT64_MAX, "", {}, 0, true);
    EXPECT_EQ(ErrorManager::GetInstance().GetRawErrorMessages().size(), 0U);
}

// 用户可构造: 某维长度为 0 -> EZ0015 (flash_attn_tiling_info_parser.cpp GetEmptyTensorFlag)
TEST(FlashAttnErrorCodeTest, ReportsEz0015WhenQueryHasEmptyDim)
{
    Para p;
    p.q = TD(gert::StorageShape({0, 128, 8, 128}, {0, 128, 8, 128}), ge::DT_FLOAT16, ge::FORMAT_ND);
    RunAndExpect(p, "EZ0015", "q");
}

// 用户可构造: 必选输入 metadata 缺失 -> EZ0004 (checkers/metadata_checker.cpp CheckSinglePara)
TEST(FlashAttnErrorCodeTest, ReportsEz0004WhenMetadataMissing)
{
    Para p;
    p.metadata = TD_DEFAULT;
    RunAndExpect(p, "EZ0004", "metadata");
}

// 注: layout_kv/layout_out 组合约束(common_checker.cpp CheckMultiParaLayout)的 EZ0026 分支
// 无法由单参数扰动触发 —— 实测 layout_kv="BNSD" 时更早的 shape 关系检查先首报 EZ0010
// ("N of query must be an integer multiple of the same axis of key", 属既有已整改站点),
// 该实验同时反证了"一次故障只有一个 ErrorManager 首错"成立(errors.size()==1)。

// 用户可构造: PA 场景下 k/v 传入非连续 stride, 首个非连续维下标超出该 layout 允许范围
// -> EZ0026 (checkers/paged_attention_checker.cpp CheckMultiPara)
// 复用同目录 test_flash_attn_tiling_stride.csv 中既有的 FAILED 用例配置(含 k_stride/v_stride),
// 这些用例此前只断言 GRAPH_FAILED, 未断言错误码
TEST(FlashAttnErrorCodeTest, ReportsEz0026WhenPaNonContiguousDimOutOfRange)
{
    const auto csvPath = std::filesystem::path(__FILE__).replace_filename("test_flash_attn_tiling_stride.csv").string();
    auto cases = GetCasesFromCsv<FlashAttnTilingUtParam>(csvPath);
    size_t checked = 0;
    for (auto& param : cases) {
        // 只取 PA 布局 + 非连续维越界的既有用例(Dim0 是合法场景, 需排除)
        if (param.expectResult != ge::GRAPH_FAILED || param.case_name.find("PA_") == std::string::npos ||
            param.case_name.find("NonContiguousDim0") != std::string::npos) {
            continue;
        }
        gert::TilingContextPara tilingContextPara(
            "FlashAttn",
            {param.q, param.k, param.v, param.block_table, param.cu_seqlens_q, param.cu_seqlens_kv, param.seqused_q,
             param.seqused_kv, param.sinks, param.attn_mask, param.metadata},
            {param.attn_out, param.softmax_lse},
            {
                {"softmax_scale", AnyValue::CreateFrom<float>(param.softmax_scale)},
                {"mask_mode", AnyValue::CreateFrom<int64_t>(param.mask_mode)},
                {"win_left", AnyValue::CreateFrom<int64_t>(param.win_left)},
                {"win_right", AnyValue::CreateFrom<int64_t>(param.win_right)},
                {"max_seqlen_q", AnyValue::CreateFrom<int64_t>(param.max_seqlen_q)},
                {"max_seqlen_kv", AnyValue::CreateFrom<int64_t>(param.max_seqlen_kv)},
                {"layout_q", AnyValue::CreateFrom<std::string>(param.layout_q)},
                {"layout_kv", AnyValue::CreateFrom<std::string>(param.layout_kv)},
                {"layout_out", AnyValue::CreateFrom<std::string>(param.layout_out)},
                {"return_softmax_lse", AnyValue::CreateFrom<int64_t>(param.return_softmax_lse)},
            },
            param.inputInstance, param.outputInstance, &param.compileInfo, "Ascend950", 64, UB_SIZE, TILING_DATA_SIZE);

        (void)ErrorManager::GetInstance().GetRawErrorMessages();
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, UINT64_MAX, "", {}, 0, true);
        const auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
        ASSERT_GE(errors.size(), 1U) << "case: " << param.case_name;
        EXPECT_EQ(errors.front().error_id, "EZ0026") << "case: " << param.case_name;
        EXPECT_EQ(errors.size(), 1U) << "case: " << param.case_name << " 首错应唯一";
        ++checked;
    }
    EXPECT_GT(checked, 0U) << "未从 stride CSV 命中任何 PA 非连续维越界用例, 断言实际未生效";
}

// 上一条用例中 k_stride 与 v_stride 同时非法, 而 key 先于 value 检查, 故只证明了 key 分支。
// 本用例把 k 的 stride 改回连续、只留 v 非法, 以单独证明 value 分支(paged_attention_checker.cpp:155)
TEST(FlashAttnErrorCodeTest, ReportsEz0026WhenPaValueNonContiguousDimOutOfRange)
{
    const auto csvPath = std::filesystem::path(__FILE__).replace_filename("test_flash_attn_tiling_stride.csv").string();
    auto cases = GetCasesFromCsv<FlashAttnTilingUtParam>(csvPath);
    auto it = std::find_if(cases.begin(), cases.end(), [](const FlashAttnTilingUtParam& c) {
        return c.case_name == "View_PA_BBND_FP16_NonContiguousDim1";
    });
    ASSERT_NE(it, cases.end()) << "stride CSV 缺少基线用例 View_PA_BBND_FP16_NonContiguousDim1";
    auto param = *it;
    // k 恢复连续 stride(shape {16,128,2,128} 的连续步长为 32768 256 128 1), v 保持 dim1 非连续
    param.k.stride_ = ParseStride("32768 256 128 1", param.k.shape_);
    param.k.hasStride_ = true;

    gert::TilingContextPara tilingContextPara(
        "FlashAttn",
        {param.q, param.k, param.v, param.block_table, param.cu_seqlens_q, param.cu_seqlens_kv, param.seqused_q,
         param.seqused_kv, param.sinks, param.attn_mask, param.metadata},
        {param.attn_out, param.softmax_lse},
        {
            {"softmax_scale", AnyValue::CreateFrom<float>(param.softmax_scale)},
            {"mask_mode", AnyValue::CreateFrom<int64_t>(param.mask_mode)},
            {"win_left", AnyValue::CreateFrom<int64_t>(param.win_left)},
            {"win_right", AnyValue::CreateFrom<int64_t>(param.win_right)},
            {"max_seqlen_q", AnyValue::CreateFrom<int64_t>(param.max_seqlen_q)},
            {"max_seqlen_kv", AnyValue::CreateFrom<int64_t>(param.max_seqlen_kv)},
            {"layout_q", AnyValue::CreateFrom<std::string>(param.layout_q)},
            {"layout_kv", AnyValue::CreateFrom<std::string>(param.layout_kv)},
            {"layout_out", AnyValue::CreateFrom<std::string>(param.layout_out)},
            {"return_softmax_lse", AnyValue::CreateFrom<int64_t>(param.return_softmax_lse)},
        },
        param.inputInstance, param.outputInstance, &param.compileInfo, "Ascend950", 64, UB_SIZE, TILING_DATA_SIZE);

    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, UINT64_MAX, "", {}, 0, true);
    const auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().error_id, "EZ0026");
    EXPECT_NE(errors.front().error_message.find("value"), std::string::npos)
        << "actual message: " << errors.front().error_message;
}

} // namespace
} // namespace FlashAttnUT
