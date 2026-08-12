/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_INT4_quant_arch35_pergroup_tiling.cpp
 * \brief Host tiling UT for GroupedS4S4Quant (INT4×INT4) pergroup groupSize 约束。
 *
 * 验证 GMM_A5_S4S4_PERGROUPSIZE_CONSTRAINT.md §3 偶数规格：
 *   - groupSize ∈ {偶数} 且整除 K -> tiling 成功，quantGroupNum = K/groupSize，baseK=128
 *     覆盖 256/512/128（128 倍数）、192/64（32 倍数非 128）、2（最小偶数非 32 倍数）。
 *   - groupSize 奇数 -> tiling 失败（A1：grouped_s4s4_quant_tiling.cpp
 *     的 (k/G) %S4S4_QUANT_GROUP_SIZE 校验，S4S4_QUANT_GROUP_SIZE=2）。
 *   - perchannel（scale 2D）对照：quantGroupNum=1。
 *
 * 路由：TilingGMM 在 npuArch==DAV_3510 && xDType==DT_INT4 && weightDtype==DT_INT4 时走
 *       GroupedS4S4IntQuantTiling（grouped_matmul_tiling.cpp:2256）。
 *
 * 输入契约（GroupedMatmul 9 输入）：x(0) weight(1) bias(2) scale(3) offset(4)
 *   antiquantScale(5) antiquantOffset(6) groupList(7) perTokenScale(8)。
 *  S4S4：x [M,K] INT4，weight [E,K,N] INT4，scale [E,G,N] UINT64(pergroup)/[E,N] UINT64(perchannel)，
 *   groupList [E] INT64(cumsum, 末值=M)，perTokenScale [M] FLOAT，y [M,N] FP16。
 *
 * 注：本测试只覆盖 host tiling 层。kernel 端到端精度 UT 受S4S4_BLAZE_ENABLED 未定义阻塞，
 *     见约束文档 §6 deferred TODO。
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.h"
#include "../../../op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h"  // AIC_AIV_CORE_RATIO
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {

constexpr uint32_t DEFAULT_AIC_NUM = 32;

optiling::GMMCompileInfo MakeAscend950CompileInfo()
{
    return {
        DEFAULT_AIC_NUM,                                      // aicNum
        DEFAULT_AIC_NUM * optiling::AIC_AIV_CORE_RATIO,       // aivNum = 2*aic
        262144,                                               // ubSize
        524288,                                               // l1Size
        134217728,                                            // l2Size
        262144,                                               // l0CSize
        65536,                                                // l0ASize
        65536,                                                // l0BSize
        platform_ascendc::SocVersion::ASCEND950,              // socVersion
        NpuArch::DAV_3510,
    };
}

gert::StorageShape MakeEmptyShape()
{
    return gert::StorageShape();
}

gert::TilingContextPara::TensorDescription MakeTensorDesc(const vector<int64_t> &dims, ge::DataType dtype,
                                                          ge::Format format = ge::FORMAT_ND)
{
    return {ops::ut::MakeGertStorageShape(dims, dims), dtype, format};
}

vector<gert::TilingContextPara::OpAttr> GetS4S4Attrs(bool transposeWeight)
{
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    };
}

bool RunS4S4Tiling(int64_t m, int64_t k, int64_t n, int64_t e, int64_t groupSize, bool isPergroup,
                   TilingInfo &tilingInfo)
{
    auto compileInfo = MakeAscend950CompileInfo();
    const int64_t G = isPergroup ? (k / groupSize) : 0;

    vector<int64_t> groupList(static_cast<size_t>(e));
    const int64_t per = m / e;
    for (int64_t i = 0; i < e; ++i) {
        groupList[static_cast<size_t>(i)] = (i + 1) * per;
    }
    groupList[static_cast<size_t>(e - 1)] = m;

    gert::StorageShape scaleShape = isPergroup
        ? ops::ut::MakeGertStorageShape({e, G, n}, {e, G, n})
        : ops::ut::MakeGertStorageShape({e, n}, {e, n});

    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({m, k}, ge::DT_INT4),
        MakeTensorDesc({e, k, n}, ge::DT_INT4),
        {MakeEmptyShape(), ge::DT_FLOAT16, ge::FORMAT_ND},
        {scaleShape, ge::DT_UINT64, ge::FORMAT_ND},
        {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},
        {MakeEmptyShape(), ge::DT_FLOAT16, ge::FORMAT_ND},
        {MakeEmptyShape(), ge::DT_FLOAT16, ge::FORMAT_ND},
        MakeTensorDesc({e}, ge::DT_INT64),
        MakeTensorDesc({m}, ge::DT_FLOAT),
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({m, n}, ge::DT_FLOAT16),
    };

    gert::TilingContextPara ctx(
        "GroupedMatmul", inputDescs, outputDescs, GetS4S4Attrs(false),
        &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);
    return ExecuteTiling(ctx, tilingInfo);
}

const GroupedMatmulTilingData::GMMBaseParamsS4S4 *GetS4S4Params(const TilingInfo &info)
{
    return &reinterpret_cast<const GroupedMatmulTilingData::GMMS4S4IntQuantTilingData *>(info.tilingData.get())
                ->gmmS4S4Params;
}

} // namespace

class TestGroupedS4S4QuantPergroupArch35Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, pergroupGroupSize256Accepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunS4S4Tiling(/*m=*/128, /*k=*/1024, /*n=*/256, /*e=*/2, /*groupSize=*/256, true, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMS4S4IntQuantTilingData));
    const auto *p = GetS4S4Params(info);
    EXPECT_EQ(p->quantGroupNum, 4U);
    EXPECT_EQ(p->baseK, 128U);
    EXPECT_EQ(p->k, 1024U);
    EXPECT_EQ(p->n, 256U);
}

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, pergroupGroupSize512Accepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunS4S4Tiling(128, 1024, 256, 2, /*groupSize=*/512, true, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMS4S4IntQuantTilingData));
    const auto *p = GetS4S4Params(info);
    EXPECT_EQ(p->quantGroupNum, 2U);
    EXPECT_EQ(p->baseK, 128U);
}

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, pergroupGroupSize128BaselineAccepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunS4S4Tiling(128, 1024, 256, 2, /*groupSize=*/128, true, info));
    const auto *p = GetS4S4Params(info);
    EXPECT_EQ(p->quantGroupNum, 8U);
    EXPECT_EQ(p->baseK, 128U);
}

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, pergroupGroupSize192EvenAccepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunS4S4Tiling(128, /*k=*/960, 256, 2, /*groupSize=*/192, true, info));
    const auto *p = GetS4S4Params(info);
    EXPECT_EQ(p->quantGroupNum, 5U);
    EXPECT_EQ(p->baseK, 128U);
}

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, pergroupGroupSize64EvenAccepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunS4S4Tiling(128, /*k=*/512, 256, 2, /*groupSize=*/64, true, info));
    const auto *p = GetS4S4Params(info);
    EXPECT_EQ(p->quantGroupNum, 8U);   // G = 512/64
    EXPECT_EQ(p->baseK, 128U);
}

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, pergroupGroupSize2MinEvenAccepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunS4S4Tiling(128, /*k=*/1024, 256, 2, /*groupSize=*/2, true, info));
    const auto *p = GetS4S4Params(info);
    EXPECT_EQ(p->quantGroupNum, 512U);   // G = 1024/2
    EXPECT_EQ(p->baseK, 128U);
}

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, pergroupGroupSize3OddRejected)
{
    TilingInfo info;
    EXPECT_FALSE(RunS4S4Tiling(128, /*k=*/192, 256, 2, /*groupSize=*/3, true, info));
}

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, pergroupGroupSize129OddRejected)
{
    TilingInfo info;
    EXPECT_FALSE(RunS4S4Tiling(128, /*k=*/4128, 256, 2, /*groupSize=*/129, true, info));
}

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, perchannelScale2DQuantGroupNumOne)
{
    TilingInfo info;
    ASSERT_TRUE(RunS4S4Tiling(128, 1024, 256, 2, /*groupSize=*/0, /*isPergroup=*/false, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMS4S4IntQuantTilingData));
    const auto *p = GetS4S4Params(info);
    EXPECT_EQ(p->quantGroupNum, 1U);   // perchannel
    EXPECT_EQ(p->baseK, 128U);         // min(1024,128)
}

TEST_F(TestGroupedS4S4QuantPergroupArch35Tiling, perchannelKZeroRejected)
{
    TilingInfo info;
    EXPECT_FALSE(RunS4S4Tiling(128, /*k=*/0, 256, 2, /*groupSize=*/0, /*isPergroup=*/false, info));
}
