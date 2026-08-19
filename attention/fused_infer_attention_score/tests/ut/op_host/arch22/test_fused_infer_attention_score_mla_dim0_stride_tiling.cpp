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

#include <initializer_list>

#include "../../../../op_host/arch22/fia_tiling_nonquant_mla_arch22.h"
#include "../../../../op_host/checkers/paged_attention_checker_fused_infer.h"
#include "../../../../op_host/fused_infer_attention_score_tiling_info_parser.h"

namespace FusedInferAttentionScoreUT {
namespace {

void SetStride(gert::Stride &stride, std::initializer_list<int64_t> values)
{
    stride.SetDimNum(static_cast<uint32_t>(values.size()));
    uint32_t index = 0;
    for (int64_t value : values) {
        stride.SetStride(index++, value);
    }
}

optiling::FiaTilingInfo MakeArch22MlaD512Info(int32_t keyDim = -1, int32_t valueDim = -1,
                                              int32_t keyRopeDim = -1)
{
    static const char layout[] = "TND";
    optiling::FiaTilingInfo fiaInfo;
    fiaInfo.opName = "FiaMlaD512Dim0StrideTest";
    fiaInfo.npuArch = NpuArch::DAV_2201;
    fiaInfo.socVersion = platform_ascendc::SocVersion::ASCEND910B;
    fiaInfo.kvStorageMode = optiling::KvStorageMode::PAGE_ATTENTION;
    fiaInfo.pageAttentionFlag = true;
    fiaInfo.hasViewStride = true;
    fiaInfo.quantMode = optiling::FiaQuantMode::NO_QUANT;
    fiaInfo.ropeMode = optiling::RopeMode::ROPE_SPLIT;
    fiaInfo.mlaMode = optiling::MlaMode::ROPE_SPLIT_D512;
    fiaInfo.inputQType = ge::DT_FLOAT16;
    fiaInfo.inputKvType = ge::DT_FLOAT16;
    fiaInfo.outputType = ge::DT_FLOAT16;
    fiaInfo.qkHeadDim = 512;
    fiaInfo.vHeadDim = 512;
    fiaInfo.ropeHeadDim = 64;
    fiaInfo.n2Size = 1;
    fiaInfo.s1Size = 1;
    fiaInfo.blockSize = 128;
    fiaInfo.kvLayout = optiling::FiaLayout::BnNBsD;
    fiaInfo.sparseMode = optiling::SPARSE_MODE_NO_MASK;
    fiaInfo.opParamInfo.layOut = layout;
    fiaInfo.keyNonContigDim = keyDim;
    fiaInfo.valueNonContigDim = valueDim;
    fiaInfo.keyRopeNonContigDim = keyRopeDim;
    fiaInfo.keyBnStride = keyDim == 0 ? 131072U : 0U;
    fiaInfo.valueBnStride = valueDim == 0 ? 196608U : 0U;
    fiaInfo.kRopeBnStride = keyRopeDim == 0 ? 16384U : 0U;
    return fiaInfo;
}

ge::graphStatus CheckNonContiguousInfo(const optiling::FiaTilingInfo &fiaInfo)
{
    optiling::PagedAttentionChecker checker(true, false, false);
    return checker.CheckMultiParaConsistency(fiaInfo);
}

TEST(FiaMlaD512StrideParser, ExtractsThreeIndependentDim0Strides)
{
    gert::StorageShape keyShape({8, 1, 128, 512}, {8, 1, 128, 512});
    gert::StorageShape valueShape({8, 1, 128, 512}, {8, 1, 128, 512});
    gert::StorageShape keyRopeShape({8, 1, 128, 64}, {8, 1, 128, 64});
    gert::Tensor keyRopeTensor(keyRopeShape, gert::StorageFormat(), ge::DT_FLOAT16);
    gert::Stride keyStride;
    gert::Stride valueStride;
    gert::Stride keyRopeStride;
    SetStride(keyStride, {131072, 65536, 512, 1});
    SetStride(valueStride, {196608, 65536, 512, 1});
    SetStride(keyRopeStride, {16384, 8192, 64, 1});

    optiling::FiaInfoParser parser(nullptr);
    parser.opName_ = "FiaMlaD512Dim0StrideTest";
    parser.hasViewStride_ = true;
    parser.kvStorageMode_ = optiling::KvStorageMode::PAGE_ATTENTION;
    parser.kvLayout_ = optiling::FiaLayout::BnNBsD;
    parser.kCache_ = {&keyShape};
    parser.vCache_ = {&valueShape};
    parser.kStrideCache_ = {&keyStride};
    parser.vStrideCache_ = {&valueStride};
    parser.keyStrides_ = &keyStride;
    parser.valueStrides_ = &valueStride;
    parser.kRopeStrides_ = &keyRopeStride;
    parser.opParamInfo_.keyRope.tensor = &keyRopeTensor;

    parser.GetKvIsContiguous();
    ASSERT_EQ(parser.GetKvStrideValues(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(parser.keyNonContigDim_, 0);
    EXPECT_EQ(parser.valueNonContigDim_, 0);
    EXPECT_EQ(parser.keyRopeNonContigDim_, 0);
    EXPECT_EQ(parser.keyBnStride_, 131072U);
    EXPECT_EQ(parser.valueBnStride_, 196608U);
    EXPECT_EQ(parser.kRopeBnStride_, 16384U);
}

TEST(FiaMlaD512StrideParser, RejectsForwardedStrideRankTooSmall)
{
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {131072, 65536, 512, 1});
    SetStride(valueStride, {196608});

    optiling::FiaInfoParser parser(nullptr);
    parser.opName_ = "FiaMlaD512Dim0StrideTest";
    parser.hasViewStride_ = true;
    parser.kvStorageMode_ = optiling::KvStorageMode::PAGE_ATTENTION;
    parser.kvLayout_ = optiling::FiaLayout::BnNBsD;
    parser.keyStrides_ = &keyStride;
    parser.valueStrides_ = &valueStride;

    EXPECT_EQ(parser.GetKvStrideValues(), ge::GRAPH_FAILED);
}

TEST(FiaMlaD512StrideChecker, AcceptsEachCacheAloneAndAllCachesTogether)
{
    EXPECT_EQ(CheckNonContiguousInfo(MakeArch22MlaD512Info(0, -1, -1)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(CheckNonContiguousInfo(MakeArch22MlaD512Info(-1, 0, -1)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(CheckNonContiguousInfo(MakeArch22MlaD512Info(-1, -1, 0)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(CheckNonContiguousInfo(MakeArch22MlaD512Info(0, 0, 0)), ge::GRAPH_SUCCESS);
}

TEST(FiaMlaD512StrideChecker, RejectsInnerAxisAndZeroForwardedStride)
{
    auto dim1Info = MakeArch22MlaD512Info(1, -1, -1);
    dim1Info.keyBnStride = 131072;
    EXPECT_EQ(CheckNonContiguousInfo(dim1Info), ge::GRAPH_FAILED);

    auto zeroStrideInfo = MakeArch22MlaD512Info(0, -1, -1);
    zeroStrideInfo.keyBnStride = 0;
    EXPECT_EQ(CheckNonContiguousInfo(zeroStrideInfo), ge::GRAPH_FAILED);
}

TEST(FiaMlaD512StrideChecker, RejectsNonTargetMlaGeometry)
{
    auto d128Info = MakeArch22MlaD512Info(0, -1, -1);
    d128Info.mlaMode = optiling::MlaMode::ROPE_SPLIT_D128;
    d128Info.qkHeadDim = 128;
    d128Info.vHeadDim = 128;
    EXPECT_EQ(CheckNonContiguousInfo(d128Info), ge::GRAPH_FAILED);

    auto valueDimInfo = MakeArch22MlaD512Info(0, -1, -1);
    valueDimInfo.vHeadDim = 128;
    EXPECT_EQ(CheckNonContiguousInfo(valueDimInfo), ge::GRAPH_FAILED);

    auto ropeDimInfo = MakeArch22MlaD512Info(0, -1, -1);
    ropeDimInfo.ropeHeadDim = 32;
    EXPECT_EQ(CheckNonContiguousInfo(ropeDimInfo), ge::GRAPH_FAILED);

    auto kvHeadInfo = MakeArch22MlaD512Info(0, -1, -1);
    kvHeadInfo.n2Size = 2;
    EXPECT_EQ(CheckNonContiguousInfo(kvHeadInfo), ge::GRAPH_FAILED);

    auto noMlaInfo = MakeArch22MlaD512Info(0, -1, -1);
    noMlaInfo.ropeMode = optiling::RopeMode::NO_ROPE;
    noMlaInfo.mlaMode = optiling::MlaMode::NO_MLA;
    EXPECT_EQ(CheckNonContiguousInfo(noMlaInfo), ge::GRAPH_FAILED);

    auto antiquantInfo = MakeArch22MlaD512Info(0, -1, -1);
    antiquantInfo.quantMode = optiling::FiaQuantMode::ANTI_QUANT;
    EXPECT_EQ(CheckNonContiguousInfo(antiquantInfo), ge::GRAPH_FAILED);
}

TEST(FiaMlaD512StrideChecker, RejectsFallbackFeatureAndNonPa)
{
    auto pseInfo = MakeArch22MlaD512Info(0, -1, -1);
    pseInfo.pseShiftFlag = true;
    EXPECT_EQ(CheckNonContiguousInfo(pseInfo), ge::GRAPH_FAILED);

    auto nonPaInfo = MakeArch22MlaD512Info(-1, -1, 0);
    nonPaInfo.kvStorageMode = optiling::KvStorageMode::BATCH_CONTINUOUS;
    nonPaInfo.pageAttentionFlag = false;
    EXPECT_EQ(CheckNonContiguousInfo(nonPaInfo), ge::GRAPH_FAILED);
}

TEST(FiaMlaD512StrideTiling, WritesOnlyActuallyNonContiguousCacheStrides)
{
    optiling::FusedInferAttentionKvStrideParams params;
    auto fiaInfo = MakeArch22MlaD512Info(0, -1, 0);
    fiaInfo.valueBnStride = 65536; // Dense metadata must remain the zero sentinel in TilingData.

    optiling::FillArch22MlaKvStrideParams(params, fiaInfo);

    EXPECT_EQ(params.get_keyBnStride(), 131072U);
    EXPECT_EQ(params.get_valueBnStride(), 0U);
    EXPECT_EQ(params.get_keyRopeBnStride(), 16384U);
}

} // namespace
} // namespace FusedInferAttentionScoreUT
