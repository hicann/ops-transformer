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
#include "../../../../sparse_flash_mla/tests/ut/op_host/test_sparse_flash_mla_tiling.h"

using namespace std;

// Ascend950 tiling cases for QuantSparseFlashMla
class QuantSparseFlashMlaTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantSparseFlashMlaTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantSparseFlashMlaTiling TearDown" << std::endl;
    }
};

namespace {
constexpr uint64_t SKIP_TILING_KEY = UINT64_MAX;
constexpr int64_t kBatchSize = 4;
constexpr int64_t kNumHeadsKv = 1;
constexpr int64_t kMetadataSize = optiling::SMLA_META_SIZE;

struct QSmlaCase {
    std::vector<int64_t> qShape = {512, 64, 512};
    std::vector<int64_t> oriKvShape = {128, 128, 1, 512};
    std::vector<int64_t> cmpKvShape = {};
    std::vector<int64_t> qDescaleShape = {1};
    std::vector<int64_t> oriKvDescaleShape = {1};
    std::vector<int64_t> cmpKvDescaleShape = {};
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
    std::vector<int64_t> sinksShape = {64};
    std::vector<int64_t> metadataShape = {1024};
    ge::DataType qType = ge::DT_HIFLOAT8;
    ge::DataType kvType = ge::DT_HIFLOAT8;
    ge::DataType descaleType = ge::DT_FLOAT;
    ge::DataType outType = ge::DT_BF16;
    ge::DataType sinksType = ge::DT_FLOAT;
    std::string layoutQ = "TND";
    std::string layoutKv = "PA_BBND";
    int64_t quantMode = 1;
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

void RunQSmlaTilingCase(const QSmlaCase &c, ge::graphStatus expect, uint64_t expectTilingKey = SKIP_TILING_KEY)
{
    struct QSMLACompileInfo {
    } compileInfo;
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t seqUsedCmpKvData[] = {4096, 4096, 4096, 4096};
    int64_t cmpResidualKvData[] = {0, 0, 0, 0};
    int64_t metadataData[kMetadataSize] = {0};
    smla_ut::InitMetadataGm(reinterpret_cast<int32_t *>(metadataData), static_cast<uint32_t>(kBatchSize),
                            static_cast<uint32_t>(kNumHeadsKv));
    gert::TilingContextPara tilingContextPara(
        "QuantSparseFlashMla",
        {Desc(c.qShape, c.qType),
         Desc(c.oriKvShape, c.kvType),
         Desc(c.cmpKvShape, c.kvType),
         Desc(c.qDescaleShape, c.descaleType),
         Desc(c.oriKvDescaleShape, c.descaleType),
         Desc(c.cmpKvDescaleShape, c.descaleType),
         Desc(c.oriSparseIndicesShape, ge::DT_INT32),
         Desc(c.cmpSparseIndicesShape, ge::DT_INT32),
         Desc(c.oriBlockTableShape, ge::DT_INT32),
         Desc(c.cmpBlockTableShape, ge::DT_INT32),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.cuSeqLensQShape), ge::DT_INT32, ge::FORMAT_ND,
                                                    true, cuSeqLensQData),
         Desc(c.cuSeqLensOriKvShape, ge::DT_INT32),
         Desc(c.cuSeqLensCmpKvShape, ge::DT_INT32),
         Desc(c.sequsedQShape, ge::DT_INT32),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.sequsedOriKvShape), ge::DT_INT32, ge::FORMAT_ND,
                                                    true, seqUsedOriKvData),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.sequsedCmpKvShape), ge::DT_INT32, ge::FORMAT_ND,
                                                    true, seqUsedCmpKvData),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.cmpResidualKvShape), ge::DT_INT32, ge::FORMAT_ND,
                                                    true, cmpResidualKvData),
         Desc(c.oriTopkLengthShape, ge::DT_INT32),
         Desc(c.cmpTopkLengthShape, ge::DT_INT32),
         Desc(c.sinksShape, c.sinksType),
         gert::TilingContextPara::TensorDescription(ToStorageShape(c.metadataShape), ge::DT_INT32, ge::FORMAT_ND, true,
                                                    metadataData)},
        {Desc(c.qShape, c.outType), Desc({1}, ge::DT_FLOAT)},
        {{"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.quantMode)},
         {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.cmpRatio)},
         {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.oriMaskMode)},
         {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.cmpMaskMode)},
         {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.oriWinLeft)},
         {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.oriWinRight)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(c.layoutQ)},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(c.layoutKv)},
         {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, expect, expectTilingKey);
}
} // namespace

// SWA only, ori_kv hif8, TND/PA_BBND success
TEST_F(QuantSparseFlashMlaTiling, test_tiling_swa_only_ori_kv_tnd_pa_nd_success)
{
    QSmlaCase c;
    RunQSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// HCA with ori and cmp kv, TND/PA_BBND success
TEST_F(QuantSparseFlashMlaTiling, test_tiling_hca_ori_and_cmp_kv_tnd_pa_nd_success)
{
    QSmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpKvDescaleShape = {1};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 128;
    c.cmpMaskMode = 3;
    RunQSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// CSA with sparse indices, TND/PA_BBND success
TEST_F(QuantSparseFlashMlaTiling, test_tiling_csa_with_sparse_indices_tnd_pa_nd_success)
{
    QSmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    c.cmpKvDescaleShape = {1};
    c.oriSparseIndicesShape = {512, 1, 512};
    c.cmpBlockTableShape = {4, 8};
    c.sequsedCmpKvShape = {4};
    c.cmpResidualKvShape = {4};
    c.cmpRatio = 4;
    c.cmpMaskMode = 3;
    RunQSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// TND q + TND ori_kv success
TEST_F(QuantSparseFlashMlaTiling, test_tiling_swa_tnd_kv_success)
{
    QSmlaCase c;
    c.oriKvShape = {512, 1, 512};
    c.oriBlockTableShape = {};
    c.cuSeqLensOriKvShape = {5};
    c.layoutKv = "TND";
    RunQSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// quant_mode only supports 1
TEST_F(QuantSparseFlashMlaTiling, test_tiling_quant_mode_invalid_failed)
{
    QSmlaCase c;
    c.quantMode = 2;
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// q_descale is required when quant_mode=1
TEST_F(QuantSparseFlashMlaTiling, test_tiling_q_descale_missing_failed)
{
    QSmlaCase c;
    c.qDescaleShape = {};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// ori_kv_descale is required when quant_mode=1
TEST_F(QuantSparseFlashMlaTiling, test_tiling_ori_kv_descale_missing_failed)
{
    QSmlaCase c;
    c.oriKvDescaleShape = {};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// cmp_kv_descale must be present exactly when cmp_kv is present
TEST_F(QuantSparseFlashMlaTiling, test_tiling_cmp_kv_descale_missing_failed)
{
    QSmlaCase c;
    c.cmpKvShape = {32, 128, 1, 512};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// descale dtype only supports float
TEST_F(QuantSparseFlashMlaTiling, test_tiling_descale_dtype_failed)
{
    QSmlaCase c;
    c.descaleType = ge::DT_FLOAT16;
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// descale shape size must be 1
TEST_F(QuantSparseFlashMlaTiling, test_tiling_descale_shape_failed)
{
    QSmlaCase c;
    c.qDescaleShape = {2};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// unsupported q dtype should fail
TEST_F(QuantSparseFlashMlaTiling, test_tiling_unsupported_q_dtype_failed)
{
    QSmlaCase c;
    c.qType = ge::DT_FLOAT16;
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// ori_kv must be provided
TEST_F(QuantSparseFlashMlaTiling, test_tiling_ori_kv_null_failed)
{
    QSmlaCase c;
    c.oriKvShape = {};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// q head num must be multiple of 64
TEST_F(QuantSparseFlashMlaTiling, test_tiling_n1_not_64_failed)
{
    QSmlaCase c;
    c.qShape = {512, 32, 512};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// metadata is required
TEST_F(QuantSparseFlashMlaTiling, test_tiling_metadata_missing_failed)
{
    QSmlaCase c;
    c.metadataShape = {};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// cu_seqlens_q is required when layout_q is TND
TEST_F(QuantSparseFlashMlaTiling, test_tiling_cu_seqlens_q_missing_failed)
{
    QSmlaCase c;
    c.cuSeqLensQShape = {};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// seqused_ori_kv is required for PA_BBND layout
TEST_F(QuantSparseFlashMlaTiling, test_tiling_seqused_ori_kv_missing_failed)
{
    QSmlaCase c;
    c.sequsedOriKvShape = {};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// ori_block_table is required for PA_BBND layout
TEST_F(QuantSparseFlashMlaTiling, test_tiling_ori_block_table_missing_failed)
{
    QSmlaCase c;
    c.oriBlockTableShape = {};
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// sinks must not be provided when ori_mask_mode is not the sinks mode
TEST_F(QuantSparseFlashMlaTiling, test_tiling_sinks_with_non_sink_mask_failed)
{
    QSmlaCase c;
    c.oriMaskMode = 0;
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// unsupported attn_out dtype should fail
TEST_F(QuantSparseFlashMlaTiling, test_tiling_out_dtype_failed)
{
    QSmlaCase c;
    c.outType = ge::DT_FLOAT16;
    RunQSmlaTilingCase(c, ge::GRAPH_FAILED);
}

// BSND q + PA_BBND ori_kv success
TEST_F(QuantSparseFlashMlaTiling, test_tiling_swa_bsnd_q_success)
{
    QSmlaCase c;
    c.qShape = {4, 128, 64, 512};
    c.cuSeqLensQShape = {};
    c.layoutQ = "BSND";
    RunQSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}

// BSND q + BSND ori_kv success
TEST_F(QuantSparseFlashMlaTiling, test_tiling_swa_bsnd_kv_success)
{
    QSmlaCase c;
    c.qShape = {4, 128, 64, 512};
    c.oriKvShape = {4, 128, 1, 512};
    c.oriBlockTableShape = {};
    c.cuSeqLensQShape = {};
    c.layoutQ = "BSND";
    c.layoutKv = "BSND";
    RunQSmlaTilingCase(c, ge::GRAPH_SUCCESS);
}
