/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../allto_all_matmul_v2_host_ut_param.h"
#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace AlltoAllMatmulV2UT {

static const std::string OP_NAME = "AlltoAllMatmulV2";

struct AlltoAllMatmulV2CompileInfo {
} compileInfo;

class AlltoAllMatmulV2Arch35TilingTest : public testing::TestWithParam<AlltoAllMatmulV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulV2Arch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulV2Arch35TilingTest TearDown." << std::endl;
    }
};

TEST_P(AlltoAllMatmulV2Arch35TilingTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {param.context, param.x1, param.x2, param.bias, param.x1Scale, param.x2Scale});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({param.y, param.all2allOut});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.worldSize)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSize)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.yDtypeAttr)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantMode)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2QuantMode)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantDtype)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX1)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX2)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)},
         {"precision_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.precisionMode)}});

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.soc, param.coreNum);
    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, "", {});
}

INSTANTIATE_TEST_SUITE_P(
    AlltoAllMatmulV2TilingUT, AlltoAllMatmulV2Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllMatmulV2TilingUtParam>);

// Cover the rawTilingData capacity check (L651-652 and L656-657 in tiling_base.h)
// by creating TilingContextPara with artificially small/zero tilingDataSize.
TEST(AlltoAllMatmulV2SmallTilingDataTest, CapacityTooSmall)
{
    AlltoAllMatmulV2CompileInfo compileInfo;
    gert::StorageShape ctxShape = GetStorageShape("1 2052");
    gert::StorageShape x1Shape = GetStorageShape("57086 1536");
    gert::StorageShape x2Shape = GetStorageShape("9216 3072");
    gert::StorageShape x1ScaleShape = GetStorageShape("57086 24 2");
    gert::StorageShape x2ScaleShape = GetStorageShape("9216 48 2");
    gert::StorageShape yShape = GetStorageShape("28543 9216");
    gert::StorageShape all2allOutShape = GetStorageShape("28543 3072");

    gert::TilingContextPara::TensorDescription context(ctxShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1Scale(x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2Scale(x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription y(yShape, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription all2allOut(all2allOutShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc(
        {context, x1, x2, TD_DEFAULT, x1Scale, x2Scale});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc({y, all2allOut});

    std::vector<gert::TilingContextPara::OpAttr> attrs(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1073741824)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(27)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4295032864)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("urma")},
         {"precision_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    // Case 1: tilingDataSize=1 → covers L656-657 (capacity too small)
    {
        gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo,
                                                  "3510", 64);
        tilingContextPara.tilingDataSize_ = 1;
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, UINT64_MAX, "", {});
    }
    // Case 2: tilingDataSize=0 → CreateCap(0) may return null → covers L651-652
    {
        gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo,
                                                  "3510", 64);
        tilingContextPara.tilingDataSize_ = 0;
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, UINT64_MAX, "", {});
    }
}

// Cover TilingParseForAlltoAllMatmulV2 (L48/L50/L51 in tiling.cpp) which
// is registered but never called by the tiling UT.
TEST(AlltoAllMatmulV2TilingParseTest, CallTilingParse)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl(OP_NAME.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    // TilingParseForAlltoAllMatmulV2 ignores the context parameter ((void)context),
    // so passing nullptr is safe and covers the 3-line function body.
    auto ret = opImpl->tiling_parse(nullptr);
    EXPECT_EQ(ge::GRAPH_SUCCESS, static_cast<ge::graphStatus>(ret));
}

} // namespace AlltoAllMatmulV2UT
