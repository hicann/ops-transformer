/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_qsag_validation_tiling.cpp
 * \brief QSAG input restrictions: one case per rule in qsag::Validate.
 *
 * The validator is shared by graph tiling (arch22) and the ACLNN front end, so these cases pin the
 * contract independently of the tiling registry: an unsupported configuration must be rejected by the
 * shared rule with the documented message, never silently reinterpreted as legacy GBSAG.
 */

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/qsag_validation.h"

namespace {
// The validator receives the platform dtype codes from its caller; the exact values are irrelevant.
constexpr int32_t kDtypeFp16 = 1;
constexpr int32_t kDtypeBf16 = 2;
constexpr int32_t kDtypeFp32 = 3;
constexpr int32_t kDtypeInt32 = 4;
constexpr int32_t kDtypeInt64 = 5;
constexpr int32_t kDtypeUnsupported = 6;

constexpr int64_t kBatch = 1;
constexpr int64_t kNq = 8;
constexpr int64_t kNk = 2;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kSq = 16;
constexpr int64_t kSk = 32;
constexpr int64_t kTopK = 4;

qsag::TensorInfo MakeTensor(const std::vector<int64_t>& shape, int32_t dtype)
{
    qsag::TensorInfo tensor;
    tensor.shape = shape;
    tensor.dtype = dtype;
    tensor.present = true;
    return tensor;
}

// QSAG requires metadata (8) and atten_mask (9) to be absent; all other optional inputs are absent
// except the TND cumulative lengths.
std::array<qsag::TensorInfo, 17> MakeTensors(const std::string& layout, int64_t nq = kNq, int64_t nk = kNk,
                                             int64_t headDim = kHeadDim, int64_t sq = kSq, int64_t sk = kSk,
                                             int64_t batch = kBatch, int64_t qcap = kSq, int64_t topk = kTopK)
{
    std::array<qsag::TensorInfo, 17> tensors{};
    std::vector<int64_t> query;
    std::vector<int64_t> key;
    std::vector<int64_t> lse;
    if (layout == "TND") {
        query = {sq, nq, headDim};
        key = {sk, nk, headDim};
        lse = {sq, nq, 1};
    } else if (layout == "BNSD") {
        query = {batch, nq, sq, headDim};
        key = {batch, nk, sk, headDim};
        lse = {batch, nq, sq, 1};
    } else {
        query = {batch, sq, nq, headDim};
        key = {batch, sk, nk, headDim};
        lse = {batch, nq, sq, 1};
    }
    tensors[0] = MakeTensor(query, kDtypeFp16);
    tensors[1] = MakeTensor(key, kDtypeFp16);
    tensors[2] = MakeTensor(key, kDtypeFp16);
    tensors[3] = MakeTensor(query, kDtypeFp16);
    tensors[4] = MakeTensor(query, kDtypeFp16);
    tensors[5] = MakeTensor(lse, kDtypeFp32);
    tensors[6] = MakeTensor({batch, nk, qcap, topk}, kDtypeInt32);
    tensors[7] = MakeTensor({batch, nk, qcap}, kDtypeInt32);
    if (layout == "TND") {
        tensors[10] = MakeTensor({batch + 1}, kDtypeInt64);
        tensors[11] = MakeTensor({batch + 1}, kDtypeInt64);
    }
    tensors[14] = MakeTensor(query, kDtypeFp16);
    tensors[15] = MakeTensor(key, kDtypeFp16);
    tensors[16] = MakeTensor(key, kDtypeFp16);
    return tensors;
}

qsag::Config MakeConfig(const std::string& layout, std::vector<int64_t> block = {1, 64})
{
    qsag::Config config;
    config.layoutQ = layout;
    config.layoutKv = layout;
    config.block = std::move(block);
    config.residual = 1;
    config.mask = 1;
    config.precision = 0;
    config.left = -1;
    config.right = -1;
    config.consistent = false;
    config.scale = 0.125f;
    return config;
}

std::string RunValidate(const std::array<qsag::TensorInfo, 17>& tensors, const qsag::Config& config)
{
    qsag::ShapeInfo shape{};
    return qsag::Validate(tensors, config, kDtypeFp16, kDtypeBf16, kDtypeFp32, kDtypeInt32, kDtypeInt64, shape);
}

struct RestrictionCase {
    const char* name;
    const char* layout;
    std::function<void(std::array<qsag::TensorInfo, 17>&, qsag::Config&)> mutate;
    const char* expectedFragment;
};

using Tensors = std::array<qsag::TensorInfo, 17>;
using Mutator = std::function<void(Tensors&, qsag::Config&)>;

const std::vector<RestrictionCase>& RestrictionCases()
{
    static const std::vector<RestrictionCase> cases = {
        {"required_tensor_missing", "TND", [](Tensors& t, qsag::Config&) { t[6].present = false; },
         "required tensor missing"},
        {"negative_dimension", "TND", [](Tensors& t, qsag::Config&) { t[0].shape[0] = -1; }, "required tensor missing"},
        {"shape_product_overflow", "TND",
         [](Tensors& t, qsag::Config&) { t[0].shape = {1LL << 62, 1LL << 62, kHeadDim}; }, "required tensor missing"},
        {"layout_mismatch", "TND", [](Tensors&, qsag::Config& c) { c.layoutKv = "BSND"; }, "matching TND/BNSD/BSND"},
        {"unsupported_layout", "TND",
         [](Tensors&, qsag::Config& c) {
             c.layoutQ = "PA_BBND";
             c.layoutKv = "PA_BBND";
         },
         "matching TND/BNSD/BSND"},
        {"residual_block_mode", "TND", [](Tensors&, qsag::Config& c) { c.residual = 0; }, "residual_block_mode=1"},
        {"mask_type", "TND", [](Tensors&, qsag::Config& c) { c.mask = 2; }, "mask 0/1"},
        {"softmax_precision", "TND", [](Tensors&, qsag::Config& c) { c.precision = 1; }, "mask 0/1"},
        {"win_left", "TND", [](Tensors&, qsag::Config& c) { c.left = 0; }, "mask 0/1"},
        {"win_right", "TND", [](Tensors&, qsag::Config& c) { c.right = 0; }, "mask 0/1"},
        {"scale_not_finite", "TND",
         [](Tensors&, qsag::Config& c) { c.scale = std::numeric_limits<float>::quiet_NaN(); }, "mask 0/1"},
        {"metadata_present", "TND", [](Tensors& t, qsag::Config&) { t[8] = MakeTensor({8}, kDtypeInt32); },
         "does not consume metadata"},
        {"atten_mask_present", "TND", [](Tensors& t, qsag::Config&) { t[9] = MakeTensor({8}, kDtypeInt32); },
         "does not consume metadata"},
        {"query_rank", "TND",
         [](Tensors& t, qsag::Config&) { t[0] = MakeTensor({kBatch, kSq, kNq, kHeadDim}, kDtypeFp16); },
         "rank does not match layout"},
        {"key_value_shape", "TND",
         [](Tensors& t, qsag::Config&) { t[2] = MakeTensor({kSk + 1, kNk, kHeadDim}, kDtypeFp16); },
         "shapes must match"},
        {"unsupported_dtype", "TND", [](Tensors& t, qsag::Config&) { t[0].dtype = kDtypeUnsupported; },
         "FP16/BF16 only"},
        {"dtype_mismatch", "TND", [](Tensors& t, qsag::Config&) { t[1].dtype = kDtypeBf16; }, "dtypes must match"},
        {"lse_dtype", "TND", [](Tensors& t, qsag::Config&) { t[5].dtype = kDtypeFp16; }, "LSE must be FP32"},
        {"lse_shape", "TND", [](Tensors& t, qsag::Config&) { t[5] = MakeTensor({kSq, kNq, 2}, kDtypeFp32); },
         "LSE must be TN/TN1"},
        {"sparse_rank", "TND", [](Tensors& t, qsag::Config&) { t[6] = MakeTensor({kBatch, kNk, kSq}, kDtypeInt32); },
         "ranks must be 4/3"},
        {"head_dim_zero", "TND", [](Tensors& t, qsag::Config&) { t = MakeTensors("TND", kNq, kNk, 0); }, "positive"},
        {"batch_zero", "BNSD",
         [](Tensors& t, qsag::Config&) { t = MakeTensors("BNSD", kNq, kNk, kHeadDim, kSq, kSk, 0); }, "positive"},
        {"head_count_indivisible", "TND", [](Tensors& t, qsag::Config&) { t = MakeTensors("TND", 7, kNk); },
         "divisible head counts"},
        {"counts_shape", "TND", [](Tensors& t, qsag::Config&) { t[7].shape[2] = kSq - 1; }, "indices [B,N2,Qcap,topK]"},
        {"topk_zero", "TND", [](Tensors& t, qsag::Config&) { t[6].shape[3] = 0; }, "1<=topK"},
        {"dense_batch_mismatch", "BNSD",
         [](Tensors& t, qsag::Config&) {
             const auto key = MakeTensor({2, kNk, kSk, kHeadDim}, kDtypeFp16);
             t[1] = key;
             t[2] = key;
             t[15] = key;
             t[16] = key;
         },
         "dense batch and Qcap"},
        {"dense_qcap_mismatch", "BNSD",
         [](Tensors& t, qsag::Config&) {
             t[6].shape[2] = kSq / 2;
             t[7].shape[2] = kSq / 2;
         },
         "dense batch and Qcap"},
        {"tnd_cu_lengths_missing", "TND", [](Tensors& t, qsag::Config&) { t[10].present = false; },
         "TND requires INT64"},
        {"tnd_cu_lengths_dtype", "TND", [](Tensors& t, qsag::Config&) { t[10].dtype = kDtypeInt32; },
         "TND requires INT64"},
        {"tnd_cu_lengths_shape", "TND", [](Tensors& t, qsag::Config&) { t[11] = MakeTensor({3}, kDtypeInt64); },
         "TND requires INT64"},
        {"seqused_dtype", "TND", [](Tensors& t, qsag::Config&) { t[12] = MakeTensor({kBatch}, kDtypeInt64); },
         "seqused must be INT32"},
        {"seqused_shape", "TND", [](Tensors& t, qsag::Config&) { t[12] = MakeTensor({kBatch + 1}, kDtypeInt32); },
         "seqused must be INT32"},
        {"dense_cu_lengths_rejected", "BNSD",
         [](Tensors& t, qsag::Config&) { t[10] = MakeTensor({kBatch + 1}, kDtypeInt64); },
         "does not accept cu lengths"},
        {"dense_seqused_rejected", "BNSD", [](Tensors& t, qsag::Config&) { t[13] = MakeTensor({kBatch}, kDtypeInt32); },
         "does not accept cu lengths"},
        {"block_shape_size", "TND", [](Tensors&, qsag::Config& c) { c.block = {64}; }, "block_shape must be [1,Bs]"},
        {"block_shape_x", "TND", [](Tensors&, qsag::Config& c) { c.block = {2, 64}; }, "block_shape must be [1,Bs]"},
        {"block_shape_zero", "TND", [](Tensors&, qsag::Config& c) { c.block = {1, 0}; }, "block_shape must be [1,Bs]"},
    };
    return cases;
}
} // namespace

class QsagValidationTilingTest : public testing::Test {};

TEST_F(QsagValidationTilingTest, dense_lse_uses_bns_independently_of_query_layout)
{
    // Nq != Sq distinguishes the producer's BNS order from obsolete BSN.
    for (const std::string layout : {"BSND", "BNSD"}) {
        for (bool singleton : {false, true}) {
            auto tensors = MakeTensors(layout);
            std::vector<int64_t> bns{kBatch, kNq, kSq};
            std::vector<int64_t> bsn{kBatch, kSq, kNq};
            if (singleton) {
                bns.push_back(1);
                bsn.push_back(1);
            }
            tensors[5] = MakeTensor(bns, kDtypeFp32);
            EXPECT_EQ(RunValidate(tensors, MakeConfig(layout)), "");
            tensors[5] = MakeTensor(bsn, kDtypeFp32);
            EXPECT_NE(RunValidate(tensors, MakeConfig(layout)).find("LSE must be TN/TN1"), std::string::npos);
        }
    }
}

TEST_F(QsagValidationTilingTest, accepts_supported_layouts)
{
    for (const char* layout : {"TND", "BNSD", "BSND"}) {
        SCOPED_TRACE(layout);
        const auto error = RunValidate(MakeTensors(layout), MakeConfig(layout));
        EXPECT_EQ(error, "") << "layout " << layout << " rejected: " << error;
    }
}

TEST_F(QsagValidationTilingTest, accepts_group_256)
{
    const auto tensors = MakeTensors("TND", 512, kNk);
    const auto config = MakeConfig("TND", {1, 32});
    qsag::ShapeInfo shape{};
    const auto error =
        qsag::Validate(tensors, config, kDtypeFp16, kDtypeBf16, kDtypeFp32, kDtypeInt32, kDtypeInt64, shape);
    EXPECT_EQ(error, "");
    EXPECT_EQ(shape.group, 256U);
}

TEST_F(QsagValidationTilingTest, accepts_group_4)
{
    const auto tensors = MakeTensors("TND");
    const auto config = MakeConfig("TND");
    qsag::ShapeInfo shape{};
    const auto error =
        qsag::Validate(tensors, config, kDtypeFp16, kDtypeBf16, kDtypeFp32, kDtypeInt32, kDtypeInt64, shape);
    EXPECT_EQ(error, "");
    EXPECT_EQ(shape.group, 4U);
}

TEST_F(QsagValidationTilingTest, accepts_block_shape_equal_to_base_ktile)
{
    // A5 allows Bs == base Ktile: G=32 keeps base Ktile=128, G=129 moves to 64.
    const std::pair<int64_t, int64_t> groups[] = {{32, 128}, {129, 64}};
    for (const auto& [group, blockSize] : groups) {
        SCOPED_TRACE(group);
        const auto tensors = MakeTensors("TND", group, 1);
        const auto config = MakeConfig("TND", {1, blockSize});
        const auto error = RunValidate(tensors, config);
        EXPECT_EQ(error, "") << "group " << group << " block " << blockSize << " rejected: " << error;
    }
}

TEST_F(QsagValidationTilingTest, accepts_block_y_8_and_head_dim_192)
{
    for (const char* layout : {"TND", "BNSD", "BSND"}) {
        SCOPED_TRACE(layout);
        const auto tensors = MakeTensors(layout, 8, 1, 192);
        const auto config = MakeConfig(layout, {1, 8});
        qsag::ShapeInfo shape{};
        const auto error =
            qsag::Validate(tensors, config, kDtypeFp16, kDtypeBf16, kDtypeFp32, kDtypeInt32, kDtypeInt64, shape);
        EXPECT_EQ(error, "") << layout << " rejected: " << error;
        EXPECT_EQ(shape.bs, 8U);
        EXPECT_EQ(shape.d, 192U);
    }
}

TEST_F(QsagValidationTilingTest, rejects_every_documented_restriction)
{
    for (const auto& testCase : RestrictionCases()) {
        SCOPED_TRACE(testCase.name);
        auto tensors = MakeTensors(testCase.layout);
        auto config = MakeConfig(testCase.layout);
        testCase.mutate(tensors, config);
        const auto error = RunValidate(tensors, config);
        EXPECT_NE(error.find(testCase.expectedFragment), std::string::npos)
            << "expected a message containing \"" << testCase.expectedFragment << "\", got: " << error;
    }
}

TEST_F(QsagValidationTilingTest, accepts_both_consistent_topk_hints)
{
    for (bool consistent : {false, true}) {
        auto config = MakeConfig("TND");
        config.consistent = consistent;
        EXPECT_EQ(RunValidate(MakeTensors("TND"), config), "");
    }
}

TEST_F(QsagValidationTilingTest, validates_head_dim_and_block_size_at_shared_entry)
{
    for (const char* layout : {"TND", "BNSD", "BSND"}) {
        SCOPED_TRACE(layout);
        EXPECT_EQ(RunValidate(MakeTensors(layout, 128, 1, 256), MakeConfig(layout, {1, 128})), "");
        EXPECT_EQ(RunValidate(MakeTensors(layout, 129, 1, 256), MakeConfig(layout, {1, 64})), "");
        EXPECT_NE(RunValidate(MakeTensors(layout, 128, 1, 257), MakeConfig(layout, {1, 128})).find("D<=256"),
                  std::string::npos);
        EXPECT_NE(RunValidate(MakeTensors(layout, 128, 1, 256), MakeConfig(layout, {1, 129})).find("Bs<=128"),
                  std::string::npos);
        EXPECT_NE(RunValidate(MakeTensors(layout, 129, 1, 256), MakeConfig(layout, {1, 65})).find("Bs<=64"),
                  std::string::npos);
    }
}

TEST_F(QsagValidationTilingTest, leaves_group_limit_to_tiling)
{
    EXPECT_EQ(RunValidate(MakeTensors("TND", 514, 2), MakeConfig("TND")), "");
}
