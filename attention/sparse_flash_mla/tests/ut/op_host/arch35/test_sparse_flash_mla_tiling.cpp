/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../test_sparse_flash_mla_tiling.h"

using namespace std;

// DAV_3510 (Ascend950) tiling cases for SparseFlashMla
class SparseFlashMlaTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashMlaTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashMlaTilingArch35 TearDown" << std::endl;
    }
};

namespace {
constexpr uint64_t SKIP_TILING_KEY = UINT64_MAX;
constexpr int64_t kBatchSize = 4;
constexpr int64_t kNumHeadsKv = 1;
constexpr int64_t kMetadataSize = optiling::SMLA_META_SIZE;

struct SmlaCase {
    std::string soc = "Ascend950";
    uint64_t coreNum = 56;
    uint64_t ubSize = 262144;
    std::vector<int64_t> oriKvStride = {};
    std::vector<int64_t> cmpKvStride = {};
    ge::DataType qType = ge::DT_FLOAT16;
    std::vector<int64_t> qShape = {512, 64, 512};
    std::vector<int64_t> oriKvShape = {128, 128, 1, 512};
    std::vector<int64_t> cmpKvShape = {};
    std::vector<int64_t> oriSparseIndicesShape = {};
    std::vector<int64_t> cmpSparseIndicesShape = {};
    std::vector<int64_t> oriBlockTableShape = {4, 32};
    std::vector<int64_t> cmpBlockTableShape = {};
    std::vector<int64_t> cuSeqLensQShape = {5};
    std::vector<int64_t> cuSeqLensOriKvShape = {};
    std::vector<int64_t> cuSeqLensCmpKvShape = {};
    std::vector<int64_t> sequsedQShape = {};
    std::vector<int64_t> sequsedOriKvShape = {4};
    std::vector<int64_t> sequsedCmpKvShape = {};
    std::vector<int64_t> cmpResidualKvShape = {};
    std::vector<int64_t> oriTopkLengthShape = {};
    std::vector<int64_t> cmpTopkLengthShape = {};
    ge::DataType oriTopkLengthType = ge::DT_INT32;
    ge::DataType cmpTopkLengthType = ge::DT_INT32;
    std::vector<int64_t> softmaxLseShape = {1};
    std::vector<int64_t> attnOutShape = {};
    ge::DataType softmaxLseType = ge::DT_FLOAT;
    bool returnSoftmaxLse = false;
    std::vector<int64_t> metadataShape = {1024};
    std::vector<int64_t> sinksShape = {64};
    ge::DataType sinksType = ge::DT_FLOAT;
    ge::DataType cuSeqLensQType = ge::DT_INT32;
    ge::DataType cuSeqLensOriKvType = ge::DT_INT32;
    ge::DataType cuSeqLensCmpKvType = ge::DT_INT32;
    ge::DataType oriSparseIndicesType = ge::DT_INT32;
    ge::DataType cmpSparseIndicesType = ge::DT_INT32;
    ge::DataType blockTableType = ge::DT_INT32;
    ge::DataType metadataType = ge::DT_INT32;
    std::string layoutQ = "TND";
    std::string layoutKv = "PA_BBND";
    int64_t cmpRatio = 1;
    int64_t oriMaskMode = 4;
    int64_t cmpMaskMode = 0;
    int64_t oriWinLeft = 127;
    int64_t oriWinRight = 0;
};

gert::StorageShape ToStorageShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    if (dims.empty()) {
        return shape;
    }
    shape.MutableShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        shape.MutableShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

gert::TilingContextPara::TensorDescription Desc(const std::vector<int64_t> &dims, ge::DataType dtype)
{
    return gert::TilingContextPara::TensorDescription(ToStorageShape(dims), dtype, ge::FORMAT_ND);
}

gert::TilingContextPara::TensorDescription DescWithStride(const std::vector<int64_t> &dims, ge::DataType dtype,
                                                          const std::vector<int64_t> &strides)
{
    auto desc = gert::TilingContextPara::TensorDescription(ToStorageShape(dims), dtype, ge::FORMAT_ND);
    desc.hasStride_ = true;
    desc.stride_.SetDimNum(static_cast<uint32_t>(strides.size()));
    for (size_t i = 0; i < strides.size(); i++) {
        desc.stride_.SetStride(static_cast<uint32_t>(i), strides[i]);
    }
    return desc;
}

void RunSmlaTilingCase(const SmlaCase &c, ge::graphStatus expect, uint64_t expectTilingKey = SKIP_TILING_KEY)
{
    struct SMLACompileInfo {
    } compileInfo;
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t seqUsedCmpKvData[] = {4096, 4096, 4096, 4096};
    int64_t cmpResidualKvData[] = {0, 0, 0, 0};
    int64_t metadataData[kMetadataSize] = {0};
    smla_ut::InitMetadataGm(reinterpret_cast<int32_t *>(metadataData), static_cast<uint32_t>(kBatchSize),
                            static_cast<uint32_t>(kNumHeadsKv));
    auto oriKvDesc =
        c.oriKvStride.empty() ? Desc(c.oriKvShape, c.qType) : DescWithStride(c.oriKvShape, c.qType, c.oriKvStride);
    auto cmpKvDesc =
        c.cmpKvStride.empty() ? Desc(c.cmpKvShape, c.qType) : DescWithStride(c.cmpKvShape, c.qType, c.cmpKvStride);
    const auto &attnOutShape = c.attnOutShape.empty() ? c.qShape : c.attnOutShape;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {Desc(c.qShape, c.qType), oriKvDesc, cmpKvDesc, Desc(c.oriSparseIndicesShape, c.oriSparseIndicesType),
         Desc(c.cmpSparseIndicesShape, c.cmpSparseIndicesType), Desc(c.oriBlockTableShape, c.blockTableType),
         Desc(c.cmpBlockTableShape, c.blockTableType),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.cuSeqLensQShape), c.cuSeqLensQType, ge::FORMAT_ND,
                                                    true, cuSeqLensQData),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.cuSeqLensOriKvShape), c.cuSeqLensOriKvType,
                                                    ge::FORMAT_ND),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.cuSeqLensCmpKvShape), c.cuSeqLensCmpKvType,
                                                    ge::FORMAT_ND),
         Desc(c.sequsedQShape, ge::DT_INT32),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.sequsedOriKvShape), ge::DT_INT32, ge::FORMAT_ND,
                                                    true, seqUsedOriKvData),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.sequsedCmpKvShape), ge::DT_INT32, ge::FORMAT_ND,
                                                    true, seqUsedCmpKvData),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.cmpResidualKvShape), ge::DT_INT32, ge::FORMAT_ND,
                                                    true, cmpResidualKvData),
         Desc(c.oriTopkLengthShape, c.oriTopkLengthType), Desc(c.cmpTopkLengthShape, c.cmpTopkLengthType),
         Desc(c.sinksShape, c.sinksType),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.metadataShape), c.metadataType, ge::FORMAT_ND,
                                                    true, metadataData)},
        {Desc(attnOutShape, c.qType), Desc(c.softmaxLseShape, c.softmaxLseType)},
        {{"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.cmpRatio)},
         {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.oriMaskMode)},
         {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.cmpMaskMode)},
         {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.oriWinLeft)},
         {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.oriWinRight)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(c.layoutQ)},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(c.layoutKv)},
         {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(c.returnSoftmaxLse)}},
        &compileInfo, c.soc, c.coreNum, c.ubSize);
    ExecuteTestCase(tilingContextPara, expect, expectTilingKey);
}
} // namespace

// SWA only, ori_kv fp16, TND/PA_BBND success on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_swa_only_ori_kv_fp16_tnd_pa_nd)
{
    SmlaCase c;
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdSwaTilingKey);
}

// SWA only, ori_kv bf16, TND/PA_BBND success on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_swa_only_ori_kv_bf16_tnd_pa_nd)
{
    SmlaCase c;
    c.qType = ge::DT_BF16;
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdSwaTilingKey);
}

// HCA with ori and cmp kv, TND/PA_BBND success on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_hca_ori_and_cmp_kv_fp16_tnd_pa_nd)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 128;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdHcaTilingKey);
}

// CSA with sparse indices, TND/PA_BBND success on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_csa_with_sparse_indices_fp16_tnd_pa_nd)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.oriSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// q head num must be multiple of 64 on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_n1_not_64_failed)
{
    SmlaCase c;
    c.qShape = {512, 32, 512};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// ori_kv must be provided
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_ori_kv_null_failed)
{
    SmlaCase c;
    c.oriKvShape = {};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// cmp_sparse_indices requires cmp_kv
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_cmp_sparse_indices_without_cmp_kv_failed)
{
    SmlaCase c;
    c.cmpSparseIndicesShape = {512, 1, 512};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// unsupported dtype should fail on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_unsupported_dtype_failed)
{
    SmlaCase c;
    c.qType = ge::DT_FLOAT;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// sinks must not be provided when ori_mask_mode is not the sinks mode
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_sinks_with_non_sink_mask_failed)
{
    SmlaCase c;
    c.oriMaskMode = 0;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// return_softmax_lse=true on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_swa_return_lse_success)
{
    SmlaCase c;
    // softmax_lse output shape {kvHeadNum, T, N1/kvHeadNum} handled by infershape; tiling only needs valid inputs
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdSwaTilingKey);
}

// BSND q + PA_BBND ori_kv success on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_swa_bsnd_q_success)
{
    SmlaCase c;
    c.qShape = {4, 128, 64, 512};
    c.cuSeqLensQShape = {};
    c.layoutQ = "BSND";
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// q head dimension must be 512
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_q_head_dim_failed)
{
    SmlaCase c;
    c.qShape = {512, 64, 256};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// q head num must be in [1, 128]
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_q_heads_over_limit_failed)
{
    SmlaCase c;
    c.qShape = {512, 129, 512};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// metadata is required on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_metadata_missing_failed)
{
    SmlaCase c;
    c.metadataShape = {};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// seqused_ori_kv is required for PA_BBND layout on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_seqused_ori_kv_missing_failed)
{
    SmlaCase c;
    c.sequsedOriKvShape = {};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// cu_seqlens_q is required for TND layout on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_cu_seqlens_q_missing_failed)
{
    SmlaCase c;
    c.cuSeqLensQShape = {};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// ori_block_table is required for PA_BBND layout on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_ori_block_table_missing_failed)
{
    SmlaCase c;
    c.oriBlockTableShape = {};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// ori_block_table must be 2 dims
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_ori_block_table_dim_failed)
{
    SmlaCase c;
    c.oriBlockTableShape = {4, 32, 2};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// TND q + TND ori_kv success on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_swa_tnd_kv_success)
{
    SmlaCase c;
    c.oriKvShape = {512, 1, 512};
    c.oriBlockTableShape = {};
    c.cuSeqLensOriKvShape = {5};
    c.layoutKv = "TND";
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// BSND q + BSND ori_kv success on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_swa_bsnd_kv_success)
{
    SmlaCase c;
    c.qShape = {4, 128, 64, 512};
    c.oriKvShape = {4, 128, 1, 512};
    c.oriBlockTableShape = {};
    c.cuSeqLensQShape = {};
    c.layoutQ = "BSND";
    c.layoutKv = "BSND";
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// dtype of sinks only supports float
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_sinks_dtype_failed)
{
    SmlaCase c;
    c.sinksType = ge::DT_FLOAT16;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// sinks must be 1 dim
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_sinks_dim_failed)
{
    SmlaCase c;
    c.sinksShape = {64, 2};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// metadata size must be 1024 on Ascend950
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_metadata_size_failed)
{
    SmlaCase c;
    c.metadataShape = {512};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// dtype of ori_block_table only supports int32
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_block_table_dtype_failed)
{
    SmlaCase c;
    c.blockTableType = ge::DT_INT64;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// SWA with return_softmax_lse=true, TND: softmax_lse shape {kvHeadNum, T, group}
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_swa_return_lse_valid_success)
{
    SmlaCase c;
    c.returnSoftmaxLse = true;
    c.softmaxLseShape = {1, 512, 64};
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdSwaTilingKey);
}

// SWA with return_softmax_lse=true, BSND: softmax_lse shape {B, kvHeadNum, S, group}
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_swa_bsnd_return_lse_valid_success)
{
    SmlaCase c;
    c.qShape = {4, 128, 64, 512};
    c.cuSeqLensQShape = {};
    c.layoutQ = "BSND";
    c.returnSoftmaxLse = true;
    c.softmaxLseShape = {4, 1, 128, 64};
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// dtype of softmax_lse only supports float when return_softmax_lse=true
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_lse_dtype_failed)
{
    SmlaCase c;
    c.returnSoftmaxLse = true;
    c.softmaxLseShape = {1, 512, 64};
    c.softmaxLseType = ge::DT_FLOAT16;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// shape of softmax_lse must match {kvHeadNum, T, group} when return_softmax_lse=true
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_lse_shape_failed)
{
    SmlaCase c;
    c.returnSoftmaxLse = true;
    c.softmaxLseShape = {1, 512, 32};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// CSA with ori_topk_length and cmp_topk_length provided (TND): shape (q_t, kv_n)
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_csa_topk_length_success)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.oriSparseIndicesShape = {512, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    c.oriTopkLengthShape = {512, 1};
    c.cmpTopkLengthShape = {512, 1};
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// dtype of ori_topk_length only supports int32
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_topk_length_dtype_failed)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.oriSparseIndicesShape = {512, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    c.oriTopkLengthShape = {512, 1};
    c.cmpTopkLengthShape = {512, 1};
    c.oriTopkLengthType = ge::DT_INT64;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// TND shape of ori_topk_length must be (q_t, kv_n=1)
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_topk_length_shape_failed)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.oriSparseIndicesShape = {512, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    c.oriTopkLengthShape = {512, 2};
    c.cmpTopkLengthShape = {512, 1};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// CSA BSND layout: sparse_indices shape (b, q_s, kv_n, topk) and topk_length (b, q_s, kv_n)
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_csa_bsnd_success)
{
    SmlaCase c;
    c.qShape = {4, 128, 64, 512};
    c.cuSeqLensQShape = {};
    c.layoutQ = "BSND";
    c.cmpKvShape = {32, 128, 1, 512};
    c.oriSparseIndicesShape = {4, 128, 1, 512};
    c.cmpSparseIndicesShape = {4, 128, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    c.oriTopkLengthShape = {4, 128, 1};
    c.cmpTopkLengthShape = {4, 128, 1};
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// last dim of cmp_sparse_indices must be greater than 0
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_cmp_sparse_indices_topk_zero_failed)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 0};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// TND shape of ori_sparse_indices must be (q_t, kv_n=1, topk)
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_ori_sparse_indices_shape_failed)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.oriSparseIndicesShape = {512, 2, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// ORI_SPARSE mode: ori_mask_mode=0 + ori_topk_length present allows omitting seqused_ori_kv (PA)
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_ori_sparse_omit_seqused_success)
{
    SmlaCase c;
    c.oriSparseIndicesShape = {512, 1, 512};
    c.oriTopkLengthShape = {512, 1};
    c.oriMaskMode = 0;
    c.oriWinLeft = -1;
    c.oriWinRight = -1;
    c.sequsedOriKvShape = {};
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// CSA with cmp_mask_mode=0 + cmp_topk_length present allows omitting seqused_cmp_kv
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_csa_omit_seqused_cmp_success)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpTopkLengthShape = {512, 1};
    c.cmpBlockTableShape = {4, 8};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 0;
    c.sequsedCmpKvShape = {};
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// length of seqused_ori_kv must match batch size
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_seqused_ori_kv_length_failed)
{
    SmlaCase c;
    c.sequsedOriKvShape = {3};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// dtype of cu_seqlens_q only supports int32
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_cu_seqlens_q_dtype_failed)
{
    SmlaCase c;
    c.cuSeqLensQType = ge::DT_INT64;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// second dimension of ori_block_table must be greater than 0
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_block_table_dim1_zero_failed)
{
    SmlaCase c;
    c.oriBlockTableShape = {4, 0};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// PA_BBND with non-contiguous stride on axis 0 of ori_kv is allowed
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_pa_ori_kv_stride0_success)
{
    SmlaCase c;
    c.oriKvStride = {131072, 512, 512, 1};
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdSwaTilingKey);
}

// non-contiguous stride on axis 1 of ori_kv should fail
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_ori_kv_stride1_failed)
{
    SmlaCase c;
    c.oriKvStride = {65536, 600, 512, 1};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// CSA with non-contiguous stride on axis 0 of cmp_kv
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_csa_cmp_kv_stride0_success)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpKvStride = {131072, 512, 512, 1};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// sinks must be provided
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_sinks_missing_failed)
{
    SmlaCase c;
    c.sinksShape = {};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// shape size of cu_seqlens_q must be bSize + 1 (TND)
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_cu_seqlens_q_size_failed)
{
    SmlaCase c;
    c.cuSeqLensQShape = {4};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// shape size of seqused_q must be bSize (TND)
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_seqused_q_size_failed)
{
    SmlaCase c;
    c.sequsedQShape = {3};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// shape size of seqused_cmp_kv must be bSize
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_seqused_cmp_kv_size_failed)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {3};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// shape size of cmp_residual_kv must be bSize
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_cmp_residual_kv_size_failed)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {3};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// invalid layout_q value should fail
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_layout_q_invalid_failed)
{
    SmlaCase c;
    c.layoutQ = "XXX";
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// invalid layout_kv value should fail
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_layout_kv_invalid_failed)
{
    SmlaCase c;
    c.layoutKv = "XXX";
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// second dimension of ori_block_table must not be negative
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_block_table_dim1_negative_failed)
{
    SmlaCase c;
    c.oriBlockTableShape = {4, -1};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// dimension of cmp_block_table must be 2
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_cmp_block_table_dim_failed)
{
    SmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8, 2};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// CSA with TND kv requires cu_seqlens_cmp_kv
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_csa_tnd_kv_cu_seqlens_cmp_missing_failed)
{
    SmlaCase c;
    c.oriKvShape = {512, 1, 512};
    c.oriBlockTableShape = {};
    c.cuSeqLensOriKvShape = {5};
    c.layoutKv = "TND";
    c.cmpKvShape = {512, 1, 512};
    c.cmpKvStride = {};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

namespace {
// Base of a valid Ascend910B TND/PA_BBND SWA case (mirrors legacy 910b tests)
SmlaCase Make910bCase()
{
    SmlaCase c;
    c.soc = "Ascend910B";
    c.coreNum = 40;
    c.ubSize = 196608;
    c.oriMaskMode = 4;
    return c;
}
} // namespace

// 910B HCA: cmp_ratio must be in [1, 128]
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_hca_cmp_ratio_over_limit_failed)
{
    SmlaCase c = Make910bCase();
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 200;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B SWA: dtype of sinks only supports float
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_sinks_dtype_failed)
{
    SmlaCase c = Make910bCase();
    c.sinksType = ge::DT_FLOAT16;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B SWA: metadata size must be 1024
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_metadata_size_failed)
{
    SmlaCase c = Make910bCase();
    c.metadataShape = {512};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B TND: dtype of cu_seqlens_q only supports int32
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_cu_seqlens_q_dtype_failed)
{
    SmlaCase c = Make910bCase();
    c.cuSeqLensQType = ge::DT_INT64;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B PA: dtype of cu_seqlens_ori_kv only supports int32 when provided
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_cu_seqlens_ori_kv_dtype_failed)
{
    SmlaCase c = Make910bCase();
    c.cuSeqLensOriKvShape = {5};
    c.cuSeqLensOriKvType = ge::DT_INT64;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B HCA: dtype of cu_seqlens_cmp_kv only supports int32 when provided
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_cu_seqlens_cmp_kv_dtype_failed)
{
    SmlaCase c = Make910bCase();
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 128;
    c.cmpMaskMode = 3;
    c.cuSeqLensCmpKvShape = {5};
    c.cuSeqLensCmpKvType = ge::DT_INT64;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B DSpark: ori_sparse_indices requires ori_mask_mode=0
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_ori_sparse_mask_mode_failed)
{
    SmlaCase c = Make910bCase();
    c.oriSparseIndicesShape = {512, 1, 512};
    c.oriTopkLengthShape = {512, 1};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B DSpark: dtype of ori_sparse_indices only supports int32
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_ori_sparse_indices_dtype_failed)
{
    SmlaCase c = Make910bCase();
    c.oriSparseIndicesShape = {512, 1, 512};
    c.oriSparseIndicesType = ge::DT_INT64;
    c.oriTopkLengthShape = {512, 1};
    c.oriMaskMode = 0;
    c.oriWinLeft = -1;
    c.oriWinRight = -1;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B DSpark: shape of ori_sparse_indices must match q (TND)
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_ori_sparse_indices_shape_failed)
{
    SmlaCase c = Make910bCase();
    c.oriSparseIndicesShape = {256, 1, 512};
    c.oriTopkLengthShape = {512, 1};
    c.oriMaskMode = 0;
    c.oriWinLeft = -1;
    c.oriWinRight = -1;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B CSA: dtype of cmp_sparse_indices only supports int32
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_cmp_sparse_indices_dtype_failed)
{
    SmlaCase c = Make910bCase();
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpSparseIndicesType = ge::DT_INT64;
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B PA: shape size of seqused_ori_kv must be bSize
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_seqused_ori_kv_size_failed)
{
    SmlaCase c = Make910bCase();
    c.sequsedOriKvShape = {3};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B HCA: shape size of seqused_cmp_kv must be bSize
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_seqused_cmp_kv_size_failed)
{
    SmlaCase c = Make910bCase();
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {3};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 128;
    c.cmpMaskMode = 3;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B TND: shape size of seqused_q must be bSize
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_seqused_q_size_failed)
{
    SmlaCase c = Make910bCase();
    c.sequsedQShape = {3};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B PA: dim 0 of ori_block_table must be bSize
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_block_table_dim0_failed)
{
    SmlaCase c = Make910bCase();
    c.oriBlockTableShape = {3, 32};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B SWA without cmp_kv: cmp_ratio must be 1
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_swa_cmp_ratio_not_1_failed)
{
    SmlaCase c = Make910bCase();
    c.cmpRatio = 4;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B DSpark: dtype of ori_topk_length only supports int32
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_ori_topk_length_dtype_failed)
{
    SmlaCase c = Make910bCase();
    c.oriSparseIndicesShape = {512, 1, 512};
    c.oriTopkLengthShape = {512, 1};
    c.oriTopkLengthType = ge::DT_INT64;
    c.oriMaskMode = 0;
    c.oriWinLeft = -1;
    c.oriWinRight = -1;
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 910B: cmp_topk_length is reserved and must be empty
TEST_F(SparseFlashMlaTilingArch35, test_tiling_910b_cmp_topk_length_reserved_failed)
{
    SmlaCase c = Make910bCase();
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    c.cmpTopkLengthShape = {512, 1};
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// unsupported npu arch (Ascend310P) should fail
TEST_F(SparseFlashMlaTilingArch35, test_tiling_unsupported_arch_failed)
{
    SmlaCase c = Make910bCase();
    c.soc = "Ascend310P";
    RunSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// 950 BSND q with optional seqused_q provided
TEST_F(SparseFlashMlaTilingArch35, test_tiling_950_bsnd_seqused_q_success)
{
    SmlaCase c;
    c.qShape = {4, 128, 64, 512};
    c.cuSeqLensQShape = {};
    c.layoutQ = "BSND";
    c.sequsedQShape = {4};
    RunSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}
