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
 * \file test_sparse_flash_mla_tiling.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "register/tilingdata_base.h"
#include "test_sparse_flash_mla_tiling.h"
#include "../../../op_host/checkers/checker_adapter.h"
#include "../../../op_host/checkers/mask_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/paged_attention_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/seq_len_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/softmax_lse_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/sparse_compression_checker.h"
#include "../../../op_host/checkers/metadata_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/sinks_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/sparse_flash_mla_checker.h"
#include "../../../op_host/checkers/checker_runner.h"

using namespace std;

struct SMLACompileInfo {
    int64_t core_num;
};

namespace {
constexpr int64_t kBatchSize = 4;
constexpr int64_t kNumHeadsKv = 1;
constexpr int64_t kMetadataSize = optiling::SMLA_META_SIZE;

void FillMockSmlaMetadata(int64_t *metadataData)
{
    smla_ut::InitMetadataGm(reinterpret_cast<int32_t *>(metadataData), static_cast<uint32_t>(kBatchSize),
                            static_cast<uint32_t>(kNumHeadsKv));
}
} // namespace

class SparseFlashMlaTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashMlaTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashMlaTiling TearDown" << std::endl;
    }
};

TEST_F(SparseFlashMlaTiling, test_tiling_swa_only_ori_kv_fp16_tnd_pa_nd)
{
    SMLACompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[kMetadataSize] = {0};
    FillMockSmlaMetadata(metadataData);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdSwaTilingKey);
}

TEST_F(SparseFlashMlaTiling, test_tiling_swa_only_ori_kv_bf16_tnd_pa_nd)
{
    SMLACompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[kMetadataSize] = {0};
    FillMockSmlaMetadata(metadataData);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdSwaTilingKey);
}

TEST_F(SparseFlashMlaTiling, test_tiling_hca_ori_and_cmp_kv_fp16_tnd_pa_nd)
{
    SMLACompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t seqUsedCmpKvData[] = {4096, 4096, 4096, 4096};
    int64_t cmpResidualKvData[] = {0, 0, 0, 0};
    int64_t metadataData[kMetadataSize] = {0};
    FillMockSmlaMetadata(metadataData);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedCmpKvData},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, cmpResidualKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdHcaTilingKey);
}

TEST_F(SparseFlashMlaTiling, test_tiling_csa_with_sparse_indices_fp16_tnd_pa_nd)
{
    SMLACompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t seqUsedCmpKvData[] = {4096, 4096, 4096, 4096};
    int64_t cmpResidualKvData[] = {0, 0, 0, 0};
    int64_t metadataData[kMetadataSize] = {0};
    FillMockSmlaMetadata(metadataData);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{512, 1, 512}, {512, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedCmpKvData},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, cmpResidualKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdCsaTilingKey);
}

TEST_F(SparseFlashMlaTiling, test_tiling_csa_with_sparse_indices_bf16_tnd_pa_nd)
{
    SMLACompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t seqUsedCmpKvData[] = {4096, 4096, 4096, 4096};
    int64_t cmpResidualKvData[] = {0, 0, 0, 0};
    int64_t metadataData[kMetadataSize] = {0};
    FillMockSmlaMetadata(metadataData);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{32, 128, 1, 512}, {32, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{512, 1, 512}, {512, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedCmpKvData},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, cmpResidualKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, smla_ut::kTndPaBnbdCsaTilingKey);
}

TEST_F(SparseFlashMlaTiling, test_tiling_n1_not_64_failed)
{
    SMLACompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[kMetadataSize] = {0};
    FillMockSmlaMetadata(metadataData);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 32, 512}, {512, 32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 32, 512}, {512, 32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SparseFlashMlaTiling, test_tiling_ori_kv_null_failed)
{
    SMLACompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[kMetadataSize] = {0};
    FillMockSmlaMetadata(metadataData);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SparseFlashMlaTiling, test_tiling_cmp_sparse_indices_without_cmp_kv_failed)
{
    SMLACompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[kMetadataSize] = {0};
    FillMockSmlaMetadata(metadataData);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{512, 1, 512}, {512, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(SparseFlashMlaTiling, test_tiling_unsupported_dtype_failed)
{
    SMLACompileInfo compileInfo = {};
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    int64_t metadataData[kMetadataSize] = {0};
    FillMockSmlaMetadata(metadataData);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData},
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
            {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 40, 196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Tiling data classes are registered for the op
TEST_F(SparseFlashMlaTiling, SparseFlashMla_tiling_data_class_registered)
{
    auto &factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashMla"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashMlaSwaParamsOp"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashMlaCmpParamsOp"), nullptr);
}

// ---- checker unit tests migrated from test_sparse_flash_mla_checker.cpp ----
namespace {
using optiling::sparse_mla_checker::CheckContext;
using optiling::sparse_mla_checker::Layout;
using optiling::sparse_mla_checker::MaskChecker;
using optiling::sparse_mla_checker::OperatorVariant;
using optiling::sparse_mla_checker::PagedAttentionChecker;
using optiling::sparse_mla_checker::SeqLenChecker;
using optiling::sparse_mla_checker::SoftmaxLseChecker;
using optiling::sparse_mla_checker::SparseCompressionChecker;

struct ShapeOnlyOptionalParam {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::Tensor *tensor = nullptr;
    const gert::StorageShape *shape = nullptr;
};

CheckContext MakeSparseContext()
{
    CheckContext context;
    context.opName = "SparseFlashMla";
    context.variant = OperatorVariant::SPARSE;
    context.cmpRatio = 1;
    context.oriMaskMode = 0;
    context.cmpMaskMode = 0;
    context.oriWinLeft = -1;
    context.oriWinRight = -1;
    return context;
}
} // namespace

TEST(SparseFlashMlaChecker, AcceptsPagedAttentionSequsedWithShapeButNoTensorPointer)
{
    gert::StorageShape shape = {{2}, {2}};
    ShapeOnlyOptionalParam param;
    param.shape = &shape;

    const auto tensorParam = optiling::sparse_mla_checker::MakeOptionalTensor(param);
    EXPECT_TRUE(tensorParam.present);
    ASSERT_NE(tensorParam.shape, nullptr);
    EXPECT_EQ(tensorParam.shape->GetDimNum(), 1U);
    EXPECT_EQ(tensorParam.shape->GetDim(0), 2);

    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.oriBlockTable.present = true;
    context.sequsedOriKv = tensorParam;

    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, TreatsMissingOptionalInputAsAbsent)
{
    const ShapeOnlyOptionalParam param;

    const auto tensorParam = optiling::sparse_mla_checker::MakeOptionalTensor(param);

    EXPECT_FALSE(tensorParam.present);
    EXPECT_EQ(tensorParam.shape, nullptr);
}

TEST(SparseFlashMlaChecker, AcceptsOriSparseIndicesWithMatchingTopkLength)
{
    CheckContext context = MakeSparseContext();
    context.oriSparseIndices.present = true;
    context.oriTopkLength.present = true;

    SparseCompressionChecker compressionChecker;
    MaskChecker maskChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(maskChecker.CheckFeature(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, RejectsOriSparseIndicesWithoutTopkLength)
{
    CheckContext context = MakeSparseContext();
    context.oriSparseIndices.present = true;

    SparseCompressionChecker compressionChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_FAILED);
}

TEST(SparseFlashMlaChecker, RejectsCmpSparseIndicesWithoutTopkLength)
{
    CheckContext context = MakeSparseContext();
    context.cmpKv.present = true;
    context.cmpSparseIndices.present = true;

    SparseCompressionChecker compressionChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_FAILED);
}

TEST(SparseFlashMlaChecker, AcceptsTopkLengthAsSequsedReplacementInPaModeWithoutSparseIndices)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.oriTopkLength.present = true;
    context.oriBlockTable.present = true;

    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsOriCmpSparseTopkLengthPairs)
{
    CheckContext context = MakeSparseContext();
    context.cmpKv.present = true;
    context.oriSparseIndices.present = true;
    context.cmpSparseIndices.present = true;
    context.oriTopkLength.present = true;
    context.cmpTopkLength.present = true;

    SparseCompressionChecker compressionChecker;
    MaskChecker maskChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(maskChecker.CheckFeature(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, SkipsSoftmaxLseChecksWhenReturnIsDisabled)
{
    CheckContext context = MakeSparseContext();
    context.returnSoftmaxLse = false;
    context.softmaxLse.present = true;

    SoftmaxLseChecker checker;
    EXPECT_EQ(checker.CheckSinglePara(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckMultiPara(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsTopkLengthsWithoutSequsedKvInOriCmpSparsePaMode)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.cmpKv.present = true;
    context.oriSparseIndices.present = true;
    context.cmpSparseIndices.present = true;
    context.oriTopkLength.present = true;
    context.cmpTopkLength.present = true;
    context.oriBlockTable.present = true;
    context.cmpBlockTable.present = true;

    SparseCompressionChecker compressionChecker;
    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsOriTopkLengthWithoutSequsedOriKvInOriSparsePaMode)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.oriSparseIndices.present = true;
    context.oriTopkLength.present = true;
    context.oriBlockTable.present = true;

    SparseCompressionChecker compressionChecker;
    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsCmpTopkLengthAsSequsedReplacementInPaMode)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.cmpKv.present = true;
    context.cmpSparseIndices.present = true;
    context.cmpTopkLength.present = true;
    context.oriBlockTable.present = true;
    context.cmpBlockTable.present = true;
    context.sequsedOriKv.present = true;

    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsTopkLengthsAsSequsedReplacementRegardlessOfMaskMode)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.cmpKv.present = true;
    context.oriSparseIndices.present = true;
    context.cmpSparseIndices.present = true;
    context.oriTopkLength.present = true;
    context.cmpTopkLength.present = true;
    context.oriBlockTable.present = true;
    context.cmpBlockTable.present = true;
    context.cmpMaskMode = 3;

    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, RejectsInvalidMaskModeAndWindow)
{
    MaskChecker maskChecker;
    {
        CheckContext context = MakeSparseContext();
        context.oriMaskMode = 1;
        EXPECT_EQ(maskChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.cmpMaskMode = 2;
        EXPECT_EQ(maskChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.oriWinLeft = -2;
        EXPECT_EQ(maskChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.oriMaskMode = 3;
        context.cmpMaskMode = 3;
        context.oriWinLeft = 0;
        context.oriWinRight = 64;
        EXPECT_EQ(maskChecker.CheckSinglePara(context), ge::GRAPH_SUCCESS);
    }
}

TEST(SparseFlashMlaChecker, RejectsMissingMetadata)
{
    optiling::sparse_mla_checker::MetadataChecker metadataChecker;
    {
        CheckContext context = MakeSparseContext();
        EXPECT_EQ(metadataChecker.CheckSinglePara(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(metadataChecker.CheckParaExistence(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.metadata.present = true;
        EXPECT_EQ(metadataChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
}

TEST(SparseFlashMlaChecker, RejectsInvalidSinks)
{
    optiling::sparse_mla_checker::SinksChecker sinksChecker;
    {
        CheckContext context = MakeSparseContext();
        EXPECT_EQ(sinksChecker.CheckSinglePara(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(sinksChecker.CheckParaExistence(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.sinks.present = true;
        EXPECT_EQ(sinksChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
    {
        gert::StorageShape shape = {{2}, {2}};
        ShapeOnlyOptionalParam param;
        param.shape = &shape;
        CheckContext context = MakeSparseContext();
        context.sinks = optiling::sparse_mla_checker::MakeOptionalTensor(param);
        context.qNumHeads = 4;
        EXPECT_EQ(sinksChecker.CheckMultiPara(context), ge::GRAPH_FAILED);
    }
}

TEST(SparseFlashMlaChecker, RejectsInvalidBlockTableUsage)
{
    PagedAttentionChecker pagedAttentionChecker;
    {
        CheckContext context = MakeSparseContext();
        context.oriBlockTable.present = true;
        EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.kvLayout = Layout::PA_BBND;
        EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.kvLayout = Layout::PA_BBND;
        context.oriBlockTable.present = true;
        context.cmpBlockTable.present = true;
        EXPECT_EQ(pagedAttentionChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
    {
        gert::StorageShape shape = {{2}, {2}};
        ShapeOnlyOptionalParam param;
        param.shape = &shape;
        CheckContext context = MakeSparseContext();
        context.kvLayout = Layout::PA_BBND;
        context.oriBlockTable = optiling::sparse_mla_checker::MakeOptionalTensor(param);
        context.cmpBlockTable = optiling::sparse_mla_checker::MakeOptionalTensor(param);
        context.bSize = 4;
        EXPECT_EQ(pagedAttentionChecker.CheckMultiPara(context), ge::GRAPH_FAILED);
    }
}

TEST(SparseFlashMlaChecker, RejectsInvalidSoftmaxLse)
{
    optiling::sparse_mla_checker::SoftmaxLseChecker softmaxLseChecker;
    {
        CheckContext context = MakeSparseContext();
        context.returnSoftmaxLse = true;
        EXPECT_EQ(softmaxLseChecker.CheckSinglePara(context), ge::GRAPH_SUCCESS);
    }
    {
        CheckContext context = MakeSparseContext();
        context.returnSoftmaxLse = true;
        context.softmaxLse.present = true;
        EXPECT_EQ(softmaxLseChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
}

TEST(SparseFlashMlaChecker, RejectsInvalidCompressionParams)
{
    SparseCompressionChecker compressionChecker;
    {
        CheckContext context = MakeSparseContext();
        context.cmpRatio = 0;
        EXPECT_EQ(compressionChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.topkValueMode = 2;
        EXPECT_EQ(compressionChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.oriSparseIndices.present = true;
        EXPECT_EQ(compressionChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
    {
        CheckContext context = MakeSparseContext();
        context.oriTopkLength.present = true;
        EXPECT_EQ(compressionChecker.CheckSinglePara(context), ge::GRAPH_FAILED);
    }
}

// CheckerRunner runs all registered checkers; empty context is rejected by the first failing check
TEST(SparseFlashMlaChecker, CheckerRunnerProcessesRegisteredCheckers)
{
    CheckContext context = MakeSparseContext();
    optiling::sparse_mla_checker::CheckerRunner runner;
    optiling::sparse_mla_checker::RegisterCommonCheckers(runner);
    EXPECT_EQ(runner.Process(context), ge::GRAPH_FAILED);
}

// SparseFlashMlaChecker::Process builds the context from SMLATilingInfo and runs the checkers
TEST(SparseFlashMlaChecker, SparseFlashMlaCheckerProcessRejectsEmptyInfo)
{
    optiling::SMLATilingInfo info;
    info.opName = "SparseFlashMla";
    optiling::SparseFlashMlaChecker checker(info);
    EXPECT_EQ(checker.Process(), ge::GRAPH_FAILED);
}

// Top-level (soc-agnostic) case with mocked Ascend950 platform: drives the DAV_3510 checker
// framework of TilingForSparseFlashMla in arch22 builds as well, keeping the checker sources
// covered on every soc report. SWA only, ori_kv fp16, TND/PA_BBND success.
TEST_F(SparseFlashMlaTiling, test_tiling_mock_950_swa_only_ori_kv_fp16_tnd_pa_bbnd)
{
    struct SMLACompileInfo {
    } compileInfo;
    int64_t cuSeqLensQData[] = {0, 128, 256, 384, 512};
    int64_t seqUsedOriKvData[] = {4096, 4096, 4096, 4096};
    constexpr int64_t kMetadataSize = optiling::SMLA_META_SIZE;
    int64_t metadataData[kMetadataSize] = {0};
    smla_ut::InitMetadataGm(reinterpret_cast<int32_t *>(metadataData), 4, 1);
    gert::TilingContextPara tilingContextPara(
        "SparseFlashMla",
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // q
            {{{128, 128, 1, 512}, {128, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // ori_kv
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // cmp_kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // ori_sparse_indices
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // cmp_sparse_indices
            {{{4, 32}, {4, 32}}, ge::DT_INT32, ge::FORMAT_ND},                         // ori_block_table
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // cmp_block_table
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqLensQData},           // cu_seqlens_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // cu_seqlens_ori_kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // cu_seqlens_cmp_kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // seqused_q
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsedOriKvData},         // seqused_ori_kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // seqused_cmp_kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // cmp_residual_kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // ori_topk_length
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                   // cmp_topk_length
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // sinks
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, metadataData}        // metadata
        },
        {
            {{{512, 64, 512}, {512, 64, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attn_out
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}                          // softmax_lse
        },
        {{"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417381615906f)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
         {"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"topk_value_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX);
}
