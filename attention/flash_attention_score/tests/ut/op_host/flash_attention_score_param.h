/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_SCORE_PARAM_H
#define FLASH_ATTENTION_SCORE_PARAM_H

#include <memory>
#include <sstream>
#include "op_host_csv_case_loader.h"
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "../../../op_host/flash_attention_score_tiling_common.h"

namespace FlashAttentionScoreUT {

using optiling::FlashAttentionScoreCompileInfo;

struct FlashAttentionScoreHostUtParamBase : public HostUtParamBase {
    float scale_value = 0.0f;
    float keep_prob = 1.0f;
    int64_t pre_tockens = 0;
    int64_t next_tockens = 0;
    int64_t head_num = 1;
    std::string input_layout = "BSH";
    int64_t inner_precise = 0;
    int64_t sparse_mode = 0;
    int64_t pse_type = 0;
    int64_t seed = 0;
    int64_t offset = 0;
    int64_t out_dtype = 0;
    std::string softmax_out_layout = "";

    FlashAttentionScoreHostUtParamBase(const csv_map &csvMap)
        : HostUtParamBase(csvMap)
    {
        scale_value = std::stof(ReadMap(csvMap, "scale_value", "0.0"));
        keep_prob = std::stof(ReadMap(csvMap, "keep_prob", "1.0"));
        pre_tockens = std::stoll(ReadMap(csvMap, "pre_tockens", "0"));
        next_tockens = std::stoll(ReadMap(csvMap, "next_tockens", "0"));
        head_num = std::stoll(ReadMap(csvMap, "head_num", "1"));
        input_layout = ReadMap(csvMap, "input_layout", "BSH");
        inner_precise = std::stoll(ReadMap(csvMap, "inner_precise", "0"));
        sparse_mode = std::stoll(ReadMap(csvMap, "sparse_mode", "0"));
        pse_type = std::stoll(ReadMap(csvMap, "pse_type", "0"));
        seed = std::stoll(ReadMap(csvMap, "seed", "0"));
        offset = std::stoll(ReadMap(csvMap, "offset", "0"));
        out_dtype = std::stoll(ReadMap(csvMap, "out_dtype", "0"));
        softmax_out_layout = ReadMap(csvMap, "softmax_out_layout", "");
    }
};

static void ApplyConstData(const csv_map &csvMap, const std::string &name,
                           gert::TilingContextPara::TensorDescription &desc, std::shared_ptr<std::vector<int64_t>> &buf)
{
    std::string dataStr = ReadMap(csvMap, name + "_data");
    if (!dataStr.empty()) {
        buf = std::make_shared<std::vector<int64_t>>();
        std::istringstream iss(dataStr);
        int64_t val;
        while (iss >> val)
            buf->push_back(val);
        desc.isConst_ = true;
        desc.constValue_ = buf->data();
    }
}

static void ApplyConstDataIS(const csv_map &csvMap, const std::string &name,
                             gert::InfershapeContextPara::TensorDescription &desc,
                             std::shared_ptr<std::vector<int64_t>> &buf)
{
    std::string dataStr = ReadMap(csvMap, name + "_data");
    if (!dataStr.empty()) {
        buf = std::make_shared<std::vector<int64_t>>();
        std::istringstream iss(dataStr);
        int64_t val;
        while (iss >> val)
            buf->push_back(val);
        desc.isConst_ = true;
        desc.constValue_ = buf->data();
    }
}

static std::vector<size_t> ParseWorkspaces(const std::string &s)
{
    std::vector<size_t> ws;
    if (s.empty())
        return ws;
    std::istringstream iss(s);
    size_t val;
    while (iss >> val)
        ws.push_back(val);
    return ws;
}

struct FlashAttentionScoreTilingUtParam : public FlashAttentionScoreHostUtParamBase {
    gert::TilingContextPara::TensorDescription query = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription real_shift = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription drop_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription padding_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription atten_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription prefix = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription actual_seq_qlen = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription actual_seq_kvlen = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription q_start_idx = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription kv_start_idx = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription dScaleQ = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription dScaleK = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription dScaleV = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription queryRope = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription keyRope = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription sink = TD_DEFAULT;

    gert::TilingContextPara::TensorDescription softmaxMax = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription softmaxSum = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription softmaxOut = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription attentionOut = TD_DEFAULT;

    uint64_t expectTilingKey = 0;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;

    std::shared_ptr<std::vector<int64_t>> actual_seq_qlen_data;
    std::shared_ptr<std::vector<int64_t>> actual_seq_kvlen_data;

    FlashAttentionScoreTilingUtParam(const csv_map &csvMap)
        : FlashAttentionScoreHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "query_shape", "query_dtype", "query_format", this->query));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_shape", "key_dtype", "key_format", this->key));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "value_shape", "value_dtype", "value_format", this->value));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "real_shift_shape", "real_shift_dtype", "real_shift_format", this->real_shift));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "drop_mask_shape", "drop_mask_dtype", "drop_mask_format", this->drop_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "padding_mask_shape", "padding_mask_dtype", "padding_mask_format", this->padding_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "atten_mask_shape", "atten_mask_dtype", "atten_mask_format", this->atten_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "prefix_shape", "prefix_dtype", "prefix_format", this->prefix));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_seq_qlen_shape", "actual_seq_qlen_dtype",
                                                     "actual_seq_qlen_format", this->actual_seq_qlen));
        ApplyConstData(csvMap, "actual_seq_qlen", this->actual_seq_qlen, this->actual_seq_qlen_data);
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_seq_kvlen_shape", "actual_seq_kvlen_dtype",
                                                     "actual_seq_kvlen_format", this->actual_seq_kvlen));
        ApplyConstData(csvMap, "actual_seq_kvlen", this->actual_seq_kvlen, this->actual_seq_kvlen_data);
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_start_idx_shape", "q_start_idx_dtype", "q_start_idx_format", this->q_start_idx));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "kv_start_idx_shape", "kv_start_idx_dtype", "kv_start_idx_format", this->kv_start_idx));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "dScaleQ_shape", "dScaleQ_dtype", "dScaleQ_format", this->dScaleQ));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "dScaleK_shape", "dScaleK_dtype", "dScaleK_format", this->dScaleK));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "dScaleV_shape", "dScaleV_dtype", "dScaleV_format", this->dScaleV));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "queryRope_shape", "queryRope_dtype", "queryRope_format", this->queryRope));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "keyRope_shape", "keyRope_dtype", "keyRope_format", this->keyRope));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "sink_shape", "sink_dtype", "sink_format", this->sink));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmaxMax_shape", "softmaxMax_dtype", "softmaxMax_format", this->softmaxMax));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmaxSum_shape", "softmaxSum_dtype", "softmaxSum_format", this->softmaxSum));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmaxOut_shape", "softmaxOut_dtype", "softmaxOut_format", this->softmaxOut));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "attentionOut_shape", "attentionOut_dtype", "attentionOut_format", this->attentionOut));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            expectTilingKey = std::stoull(ReadMap(csvMap, "expectTilingKey", "0"));
            expectTilingData = ReadMap(csvMap, "expectTilingData", "");
            expectWorkspaces = ParseWorkspaces(ReadMap(csvMap, "expectWorkspaces", ""));
        }
    }
};

struct FlashAttentionScoreInferShapeUtParam : public FlashAttentionScoreHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription query = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription value = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription real_shift = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription drop_mask = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription padding_mask = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription atten_mask = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription prefix = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription actual_seq_qlen = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription actual_seq_kvlen = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription q_start_idx = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription kv_start_idx = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription dScaleQ = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription dScaleK = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription dScaleV = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription queryRope = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription keyRope = ID_DEFAULT;

    gert::InfershapeContextPara::TensorDescription softmaxMax = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription softmaxSum = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription softmaxOut = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription attentionOut = ID_DEFAULT;

    std::vector<std::vector<int64_t>> expectOutputShape;

    std::shared_ptr<std::vector<int64_t>> actual_seq_qlen_data;
    std::shared_ptr<std::vector<int64_t>> actual_seq_kvlen_data;

    FlashAttentionScoreInferShapeUtParam(const csv_map &csvMap)
        : FlashAttentionScoreHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "query_shape", "query_dtype", "query_format", this->query));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_shape", "key_dtype", "key_format", this->key));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "value_shape", "value_dtype", "value_format", this->value));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "real_shift_shape", "real_shift_dtype", "real_shift_format", this->real_shift));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "drop_mask_shape", "drop_mask_dtype", "drop_mask_format", this->drop_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "padding_mask_shape", "padding_mask_dtype", "padding_mask_format", this->padding_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "atten_mask_shape", "atten_mask_dtype", "atten_mask_format", this->atten_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "prefix_shape", "prefix_dtype", "prefix_format", this->prefix));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_seq_qlen_shape", "actual_seq_qlen_dtype",
                                                     "actual_seq_qlen_format", this->actual_seq_qlen));
        ApplyConstDataIS(csvMap, "actual_seq_qlen", this->actual_seq_qlen, this->actual_seq_qlen_data);
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_seq_kvlen_shape", "actual_seq_kvlen_dtype",
                                                     "actual_seq_kvlen_format", this->actual_seq_kvlen));
        ApplyConstDataIS(csvMap, "actual_seq_kvlen", this->actual_seq_kvlen, this->actual_seq_kvlen_data);
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_start_idx_shape", "q_start_idx_dtype", "q_start_idx_format", this->q_start_idx));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "kv_start_idx_shape", "kv_start_idx_dtype", "kv_start_idx_format", this->kv_start_idx));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "dScaleQ_shape", "dScaleQ_dtype", "dScaleQ_format", this->dScaleQ));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "dScaleK_shape", "dScaleK_dtype", "dScaleK_format", this->dScaleK));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "dScaleV_shape", "dScaleV_dtype", "dScaleV_format", this->dScaleV));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "queryRope_shape", "queryRope_dtype", "queryRope_format", this->queryRope));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "keyRope_shape", "keyRope_dtype", "keyRope_format", this->keyRope));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmaxMax_shape", "softmaxMax_dtype", "softmaxMax_format", this->softmaxMax));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmaxSum_shape", "softmaxSum_dtype", "softmaxSum_format", this->softmaxSum));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmaxOut_shape", "softmaxOut_dtype", "softmaxOut_format", this->softmaxOut));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "attentionOut_shape", "attentionOut_dtype", "attentionOut_format", this->attentionOut));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            std::string shapeStr = ReadMap(csvMap, "expect_softmaxMax_shape", "");
            if (!shapeStr.empty()) {
                std::vector<int64_t> shape;
                std::stringstream ss(shapeStr);
                int64_t dim;
                while (ss >> dim)
                    shape.push_back(dim);
                expectOutputShape.push_back(shape);
            }
            shapeStr = ReadMap(csvMap, "expect_softmaxSum_shape", "");
            if (!shapeStr.empty()) {
                std::vector<int64_t> shape;
                std::stringstream ss(shapeStr);
                int64_t dim;
                while (ss >> dim)
                    shape.push_back(dim);
                expectOutputShape.push_back(shape);
            }
            shapeStr = ReadMap(csvMap, "expect_softmaxOut_shape", "");
            if (!shapeStr.empty()) {
                std::vector<int64_t> shape;
                std::stringstream ss(shapeStr);
                int64_t dim;
                while (ss >> dim)
                    shape.push_back(dim);
                expectOutputShape.push_back(shape);
            }
            shapeStr = ReadMap(csvMap, "expect_attentionOut_shape", "");
            if (!shapeStr.empty()) {
                std::vector<int64_t> shape;
                std::stringstream ss(shapeStr);
                int64_t dim;
                while (ss >> dim)
                    shape.push_back(dim);
                expectOutputShape.push_back(shape);
            }
        }
    }
};

struct FlashAttentionScoreInferDTypeUtParam : public FlashAttentionScoreHostUtParamBase {
    ge::DataType input_dtype = ge::DT_UNDEFINED;
    ge::DataType softmaxMax_dtype = ge::DT_UNDEFINED;
    ge::DataType softmaxSum_dtype = ge::DT_UNDEFINED;
    ge::DataType softmaxOut_dtype = ge::DT_UNDEFINED;
    ge::DataType attentionOut_dtype = ge::DT_UNDEFINED;

    FlashAttentionScoreInferDTypeUtParam(const csv_map &csvMap)
        : FlashAttentionScoreHostUtParamBase(csvMap)
    {
        GetDataTypeGE(csvMap, "input_dtype", this->input_dtype);

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            GetDataTypeGE(csvMap, "softmaxMax_dtype", this->softmaxMax_dtype);
            GetDataTypeGE(csvMap, "softmaxSum_dtype", this->softmaxSum_dtype);
            GetDataTypeGE(csvMap, "softmaxOut_dtype", this->softmaxOut_dtype);
            GetDataTypeGE(csvMap, "attentionOut_dtype", this->attentionOut_dtype);
        }
    }
};

} // namespace FlashAttentionScoreUT

#endif // FLASH_ATTENTION_SCORE_PARAM_H
