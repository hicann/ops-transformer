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
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../quant_block_sparse_attn_host_ut_param.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace QuantBlockSparseAttnUT {

static const std::string OP_NAME = "QuantBlockSparseAttn";

struct QuantBlockSparseAttnCompileInfo {
} compileInfo;

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

const TensorDesc EmptyInput(gert::StorageShape({0}, {0}), ge::DT_INT32, ge::FORMAT_ND);
const TensorDesc DefaultMetadata(gert::StorageShape({1}, {1}), ge::DT_INT32, ge::FORMAT_ND);

static TensorDesc NormalizeTd(const TensorDesc& td)
{
    if (td.dtype_ == ge::DT_UNDEFINED) {
        return EmptyInput;
    }
    return td;
}

static void SetStride(TensorDesc& td, uint64_t stride0, uint64_t stride1, uint64_t stride2, uint64_t stride3)
{
    td.stride_.SetStride(0U, stride0);
    td.stride_.SetStride(1U, stride1);
    td.stride_.SetStride(2U, stride2);
    td.stride_.SetStride(3U, stride3);
    td.stride_.SetDimNum(4U);
    td.hasStride_ = true;
}

static bool HasPositiveDims(const gert::Shape& shape)
{
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

static void SetFp8PaStrides(TensorDesc& key, TensorDesc& value, TensorDesc& kDescale)
{
    const gert::Shape& keyShape = key.shape_.GetStorageShape();
    const gert::Shape& valueShape = value.shape_.GetStorageShape();
    const gert::Shape& kDescaleShape = kDescale.shape_.GetStorageShape();
    SetStride(key, 1U, 1U, 1U, 1U);
    SetStride(value, 1U, 1U, 1U, 1U);
    SetStride(kDescale, 1U, 1U, 1U, 1U);
    if (keyShape.GetDimNum() != 4U || valueShape.GetDimNum() != 4U || !HasPositiveDims(keyShape) ||
        !HasPositiveDims(valueShape)) {
        return;
    }

    const uint64_t keyN = static_cast<uint64_t>(keyShape.GetDim(1U));
    const uint64_t keyBlockSize = static_cast<uint64_t>(keyShape.GetDim(2U));
    const uint64_t keyD = static_cast<uint64_t>(keyShape.GetDim(3U));
    const uint64_t valueN = static_cast<uint64_t>(valueShape.GetDim(1U));
    const uint64_t valueBlockSize = static_cast<uint64_t>(valueShape.GetDim(2U));
    const uint64_t valueD = static_cast<uint64_t>(valueShape.GetDim(3U));
    uint64_t paBlockStride = keyN * keyBlockSize * keyD + valueN * valueBlockSize * valueD;
    if (kDescaleShape.GetDimNum() == 4U && HasPositiveDims(kDescaleShape)) {
        const uint64_t kDescaleN = static_cast<uint64_t>(kDescaleShape.GetDim(1U));
        const uint64_t kDescaleBlockSize = static_cast<uint64_t>(kDescaleShape.GetDim(2U));
        const uint64_t kDescaleD = static_cast<uint64_t>(kDescaleShape.GetDim(3U));
        paBlockStride += kDescaleN * kDescaleBlockSize * kDescaleD * sizeof(float);
        SetStride(kDescale, paBlockStride / sizeof(float), kDescaleBlockSize * kDescaleD, kDescaleD, 1U);
    }

    SetStride(key, paBlockStride, keyBlockSize * keyD, keyD, 1U);
    SetStride(value, paBlockStride, valueBlockSize * valueD, valueD, 1U);
}

static bool ExecuteCase(QuantBlockSparseAttnTilingUtParam param, TilingInfo& tilingInfo,
                        const std::string& socVersion = "Ascend950")
{
    TensorDesc key = NormalizeTd(param.key);
    TensorDesc value = NormalizeTd(param.value);
    TensorDesc kDescale = NormalizeTd(param.kDescale);
    TensorDesc metadata = NormalizeTd(param.metadata);
    if (param.quant_mode == 1) {
        if (param.provideFp8Strides) {
            SetFp8PaStrides(key, value, kDescale);
            if (param.mismatchValueStride) {
                value.stride_.SetStride(0U, value.stride_.GetStride(0U) + 1U);
            }
        }
    }
    if (param.provideFp8Metadata && param.metadata.dtype_ == ge::DT_UNDEFINED) {
        metadata = DefaultMetadata;
    } else if (!param.provideFp8Metadata) {
        metadata = param.metadata;
    }
    std::vector<TensorDesc> inputTensorDesc({
        NormalizeTd(param.query),
        key,
        value,
        NormalizeTd(param.qDescale),
        kDescale,
        NormalizeTd(param.vDescale),
        NormalizeTd(param.pScale),
        NormalizeTd(param.cuSeqlensQ),
        NormalizeTd(param.cuSeqlensKv),
        NormalizeTd(param.sequsedQ),
        NormalizeTd(param.sequsedKv),
        NormalizeTd(param.sparseIndices),
        NormalizeTd(param.sparseSeqLen),
        NormalizeTd(param.blockTable),
        NormalizeTd(param.attenMask),
        metadata,
    });

    std::vector<TensorDesc> outputTensorDesc({
        NormalizeTd(param.attentionOut),
        NormalizeTd(param.softmaxLse),
    });

    std::vector<OpAttr> attrs({
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_q_block_size)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_kv_block_size)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_sparse_indices)},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_mode)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)},
    });

    gert::TilingContextPara para(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo, socVersion, 64,
                                 262144, 65536);
    return ExecuteTiling(para, tilingInfo);
}

class QuantBlockSparseAttnTilingArch35Test : public testing::TestWithParam<QuantBlockSparseAttnTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantBlockSparseAttnTilingArch35Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantBlockSparseAttnTilingArch35Test TearDown" << std::endl;
    }
};

TEST_P(QuantBlockSparseAttnTilingArch35Test, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;
    TilingInfo tilingInfo;
    bool ok = ExecuteCase(param, tilingInfo);

    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(ok, true);
        EXPECT_GT(tilingInfo.blockNum, 0U);
        EXPECT_GT(tilingInfo.tilingDataSize, 0U);
    } else {
        EXPECT_EQ(ok, false);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuantBlockSparseAttnTiling, QuantBlockSparseAttnTilingArch35Test,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnTilingUtParam>);

TEST(QuantBlockSparseAttnArchitecture, Fp8DispatchA5AndA6)
{
    const auto cases = GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__));
    ASSERT_FALSE(cases.empty());
    const auto& param = cases.front(); // Valid FP8 TND case.
    TilingInfo arch35;
    TilingInfo arch92;
    ASSERT_TRUE(ExecuteCase(param, arch35, "Ascend950"));
    ASSERT_TRUE(ExecuteCase(param, arch92, "Ascend960dt"));
    EXPECT_NE(arch35.tilingKey, arch92.tilingKey);
    EXPECT_LE(arch35.blockNum, 36U);
    EXPECT_LE(arch92.blockNum, 64U);
}

static QuantBlockSparseAttnTilingUtParam MakePaddedStrideCase(const std::string& layout, int64_t batch,
                                                              int64_t sequence, int64_t heads)
{
    const auto cases = GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__));
    for (auto param : cases) {
        if (param.case_name != "mxfp8_tiling_bnsd_bns_lse") {
            continue;
        }
        const int64_t dim1 = layout == "BSND" ? sequence : heads;
        const int64_t dim2 = layout == "BSND" ? heads : sequence;
        param.layout_q = layout;
        param.layout_out = layout;
        param.mask_mode = 0;
        param.attenMask = TensorDesc(gert::StorageShape({0}, {0}), ge::DT_UINT8, ge::FORMAT_ND);
        param.query = TensorDesc(gert::StorageShape({batch, dim1, dim2, 128}, {batch, dim1, dim2, 128}),
                                 ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
        param.qDescale = TensorDesc(gert::StorageShape({batch, dim1, dim2, 2, 2}, {batch, dim1, dim2, 2, 2}),
                                    ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
        param.key = TensorDesc(gert::StorageShape({4, heads, 128, 128}, {4, heads, 128, 128}), ge::DT_FLOAT8_E4M3FN,
                               ge::FORMAT_ND);
        param.value = param.key;
        param.kDescale = TensorDesc(gert::StorageShape({4, heads, 128, 2, 2}, {4, heads, 128, 2, 2}),
                                    ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
        param.vDescale = TensorDesc(gert::StorageShape({4, heads, 2, 128, 2}, {4, heads, 2, 128, 2}),
                                    ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
        param.sequsedKv = TensorDesc(gert::StorageShape({batch}, {batch}), ge::DT_INT32, ge::FORMAT_ND);
        param.sparseIndices =
            TensorDesc(gert::StorageShape({batch, heads, 1, 4}, {batch, heads, 1, 4}), ge::DT_INT32, ge::FORMAT_ND);
        param.sparseSeqLen =
            TensorDesc(gert::StorageShape({batch, heads, 1}, {batch, heads, 1}), ge::DT_INT32, ge::FORMAT_ND);
        param.blockTable = TensorDesc(gert::StorageShape({batch, 8}, {batch, 8}), ge::DT_INT32, ge::FORMAT_ND);
        param.attentionOut = TensorDesc(gert::StorageShape({batch, dim1, dim2, 128}, {batch, dim1, dim2, 128}),
                                        ge::DT_BF16, ge::FORMAT_ND);
        param.softmaxLse = TensorDesc(gert::StorageShape({batch, heads, sequence}, {batch, heads, sequence}),
                                      ge::DT_FLOAT, ge::FORMAT_ND);
        for (auto* input : {&param.query, &param.qDescale}) {
            const auto& shape = input->shape_.GetStorageShape();
            int64_t stride = 1;
            for (size_t i = shape.GetDimNum(); i > 0U; --i) {
                input->stride_.SetStride(i - 1U, stride);
                stride *= shape.GetDim(i - 1U);
            }
            input->stride_.SetDimNum(shape.GetDimNum());
            input->hasStride_ = true;
        }
        return param;
    }
    throw std::runtime_error("Missing padded-layout baseline tiling case");
}

TEST(QuantBlockSparseAttnArchitecture, MxFp8RemainsArch35Only)
{
    const auto param = MakePaddedStrideCase("BNSD", 2, 31, 4);
    TilingInfo arch35;
    TilingInfo arch92;
    EXPECT_TRUE(ExecuteCase(param, arch35, "Ascend950"));
    EXPECT_FALSE(ExecuteCase(param, arch92, "Ascend960dt"));
}

TEST(QuantBlockSparseAttnPaddedStrideTest, AcceptsSingletonDimensionStrides)
{
    // Includes LAYOUT006 (B=2, N=1, S=31), singleton B/S, and a non-singleton control.
    const int64_t dimensions[][3] = {{2, 31, 1}, {1, 31, 4}, {2, 1, 4}, {2, 31, 4}};
    for (const std::string layout : {"BSND", "BNSD"}) {
        for (const auto& dims : dimensions) {
            // Exercise Q and QScale independently, then together; mode 0 is canonical.
            for (uint32_t mode = 0U; mode < 4U; ++mode) {
                SCOPED_TRACE(layout + " B=" + std::to_string(dims[0]) + " S=" + std::to_string(dims[1]) +
                             " N=" + std::to_string(dims[2]) + " mode=" + std::to_string(mode));
                auto param = MakePaddedStrideCase(layout, dims[0], dims[1], dims[2]);
                TensorDesc* inputs[] = {&param.query, &param.qDescale};
                for (size_t inputIndex = 0U; inputIndex < 2U; ++inputIndex) {
                    if ((mode & (1U << inputIndex)) == 0U) {
                        continue;
                    }
                    auto& input = *inputs[inputIndex];
                    const auto& shape = input.shape_.GetStorageShape();
                    for (size_t dim = 0U; dim < shape.GetDimNum(); ++dim) {
                        if (shape.GetDim(dim) == 1) {
                            // A singleton index is always zero, so its stride cannot change an address.
                            input.stride_.SetStride(dim, inputIndex == 0U ? 128 : 4);
                        }
                    }
                }
                TilingInfo tilingInfo;
                EXPECT_TRUE(ExecuteCase(param, tilingInfo));
            }
        }
    }
}

TEST(QuantBlockSparseAttnPaddedStrideTest, RejectsNonContiguousInputsAndWrongStrideRank)
{
    for (const std::string layout : {"BSND", "BNSD"}) {
        for (size_t inputIndex = 0U; inputIndex < 2U; ++inputIndex) {
            // Keep N=1 so the relaxed singleton rule must not hide a real stride error.
            auto baseline = MakePaddedStrideCase(layout, 2, 31, 1);
            const auto& shape = (inputIndex == 0U ? baseline.query : baseline.qDescale).shape_.GetStorageShape();
            for (size_t dim = 0U; dim < shape.GetDimNum(); ++dim) {
                if (shape.GetDim(dim) == 1) {
                    continue;
                }
                auto param = baseline;
                auto& input = inputIndex == 0U ? param.query : param.qDescale;
                input.stride_.SetStride(dim, input.stride_.GetStride(dim) + 1);
                TilingInfo tilingInfo;
                EXPECT_FALSE(ExecuteCase(param, tilingInfo));
            }
            auto& input = inputIndex == 0U ? baseline.query : baseline.qDescale;
            input.stride_.SetDimNum(shape.GetDimNum() - 1U);
            TilingInfo tilingInfo;
            EXPECT_FALSE(ExecuteCase(baseline, tilingInfo));
        }
    }
}

} // namespace QuantBlockSparseAttnUT
