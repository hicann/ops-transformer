/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STEM_INDEXER_PARAM_H
#define STEM_INDEXER_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace StemIndexerUT {

struct StemIndexerHostUtParamBase : public HostUtParamBase {
    bool causal;
    int64_t stem_block_size;
    int64_t stem_stride;
    float alpha;
    int64_t initial_blocks;
    int64_t window_size;
    float k_block_num_rate_medium;
    int64_t k_block_num_bias_medium;
    float k_block_num_rate_large;
    int64_t k_block_num_bias_large;
    int64_t topk_score_precision;

    StemIndexerHostUtParamBase(const csv_map &csvMap) : HostUtParamBase(csvMap)
    {
        this->causal = std::stoi(ReadMap(csvMap, "causal", "1"));
        this->stem_block_size = std::stoll(ReadMap(csvMap, "stem_block_size", "128"));
        this->stem_stride = std::stoll(ReadMap(csvMap, "stem_stride", "16"));
        this->alpha = std::stof(ReadMap(csvMap, "alpha", "1.0"));
        this->initial_blocks = std::stoll(ReadMap(csvMap, "initial_blocks", "4"));
        this->window_size = std::stoll(ReadMap(csvMap, "window_size", "4"));
        this->k_block_num_rate_medium = std::stof(ReadMap(csvMap, "k_block_num_rate_medium", "0.2"));
        this->k_block_num_bias_medium = std::stoll(ReadMap(csvMap, "k_block_num_bias_medium", "30"));
        this->k_block_num_rate_large = std::stof(ReadMap(csvMap, "k_block_num_rate_large", "0.1"));
        this->k_block_num_bias_large = std::stoll(ReadMap(csvMap, "k_block_num_bias_large", "30"));
        this->topk_score_precision = std::stoll(ReadMap(csvMap, "topk_score_precision", "1"));
    }
};

struct StemIndexerTilingUtParam : public StemIndexerHostUtParamBase {
    gert::TilingContextPara::TensorDescription qflat = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription kflat = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription vbias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription q_seq_lens = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription kv_seq_lens = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription num_prompt_tokens = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription metadata = TD_DEFAULT;

    gert::TilingContextPara::TensorDescription sparse_indices = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription sparse_seq_len = TD_DEFAULT;

    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    StemIndexerTilingUtParam(const csv_map &csvMap) : StemIndexerHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "qflat_shape", "qflat_dtype", "qflat_format", this->qflat));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "kflat_shape", "kflat_dtype", "kflat_format", this->kflat));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "vbias_shape", "vbias_dtype", "vbias_format", this->vbias));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_seq_lens_shape", "q_seq_lens_dtype", "q_seq_lens_format", this->q_seq_lens));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "kv_seq_lens_shape", "kv_seq_lens_dtype", "kv_seq_lens_format", this->kv_seq_lens));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "num_prompt_tokens_shape", "num_prompt_tokens_dtype",
                                                     "num_prompt_tokens_format", this->num_prompt_tokens));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "metadata_shape", "metadata_dtype", "metadata_format", this->metadata));

        this->outputInstance.emplace_back(GetTensorGE(csvMap, "sparse_indices_shape", "sparse_indices_dtype",
                                                      "sparse_indices_format", this->sparse_indices));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "sparse_seq_len_shape", "sparse_seq_len_dtype",
                                                      "sparse_seq_len_format", this->sparse_seq_len));

        this->expectTilingKey = UINT64_MAX;
        this->expectTilingDataHash = "";
        if (this->expectResult == ge::GRAPH_SUCCESS) {
            std::string tilingKeyStr = ReadMap(csvMap, "expectTilingKey");
            if (!tilingKeyStr.empty()) {
                this->expectTilingKey = std::stoull(tilingKeyStr);
            }
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};

struct StemIndexerInferShapeUtParam : public StemIndexerHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription qflat = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription kflat = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription vbias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription q_seq_lens = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription kv_seq_lens = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription num_prompt_tokens = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription metadata = ID_DEFAULT;

    gert::InfershapeContextPara::TensorDescription sparse_indices = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription sparse_seq_len = ID_DEFAULT;

    std::vector<std::vector<int64_t>> expectOutputShape;

    StemIndexerInferShapeUtParam(const csv_map &csvMap) : StemIndexerHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "qflat_shape", "qflat_dtype", "qflat_format", this->qflat));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "kflat_shape", "kflat_dtype", "kflat_format", this->kflat));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "vbias_shape", "vbias_dtype", "vbias_format", this->vbias));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_seq_lens_shape", "q_seq_lens_dtype", "q_seq_lens_format", this->q_seq_lens));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "kv_seq_lens_shape", "kv_seq_lens_dtype", "kv_seq_lens_format", this->kv_seq_lens));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "num_prompt_tokens_shape", "num_prompt_tokens_dtype",
                                                     "num_prompt_tokens_format", this->num_prompt_tokens));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "metadata_shape", "metadata_dtype", "metadata_format", this->metadata));

        this->outputInstance.emplace_back(GetTensorGE(csvMap, "sparse_indices_shape", "sparse_indices_dtype",
                                                      "sparse_indices_format", this->sparse_indices));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "sparse_seq_len_shape", "sparse_seq_len_dtype",
                                                      "sparse_seq_len_format", this->sparse_seq_len));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape = {GetShapeArr(ReadMap(csvMap, "sparse_indices_shape")),
                                       GetShapeArr(ReadMap(csvMap, "sparse_seq_len_shape"))};
        }
    }
};

struct StemIndexerInferDTypeUtParam : public StemIndexerHostUtParamBase {
    ge::DataType qflat_dtype = ge::DT_UNDEFINED;
    ge::DataType kflat_dtype = ge::DT_UNDEFINED;
    ge::DataType vbias_dtype = ge::DT_UNDEFINED;
    ge::DataType q_seq_lens_dtype = ge::DT_UNDEFINED;
    ge::DataType kv_seq_lens_dtype = ge::DT_UNDEFINED;
    ge::DataType num_prompt_tokens_dtype = ge::DT_UNDEFINED;
    ge::DataType metadata_dtype = ge::DT_UNDEFINED;

    ge::DataType sparse_indices_dtype = ge::DT_UNDEFINED;
    ge::DataType sparse_seq_len_dtype = ge::DT_UNDEFINED;

    StemIndexerInferDTypeUtParam(const csv_map &csvMap) : StemIndexerHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "qflat_dtype", this->qflat_dtype));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "kflat_dtype", this->kflat_dtype));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "vbias_dtype", this->vbias_dtype));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "q_seq_lens_dtype", this->q_seq_lens_dtype));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "kv_seq_lens_dtype", this->kv_seq_lens_dtype));
        this->inputInstance.emplace_back(
            GetDataTypeGE(csvMap, "num_prompt_tokens_dtype", this->num_prompt_tokens_dtype));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "metadata_dtype", this->metadata_dtype));

        this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "sparse_indices_dtype", this->sparse_indices_dtype));
        this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "sparse_seq_len_dtype", this->sparse_seq_len_dtype));
    }
};

} // namespace StemIndexerUT

#endif // STEM_INDEXER_PARAM_H
