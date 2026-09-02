/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_PARAM_H
#define FLASH_ATTN_PARAM_H

#include <string>
#include <vector>
#include <sstream>
#include "op_host_csv_case_loader.h"
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "../../../op_host/flash_attn_tiling_common.h"

namespace FlashAttnUT {

using optiling::flash_attn::FlashAttnCompileInfo;

struct FlashAttnHostUtParamBase : public HostUtParamBase {
    float softmax_scale = 0.0f;
    int64_t mask_mode = 0;
    int64_t win_left = -1;
    int64_t win_right = -1;
    int64_t max_seqlen_q = -1;
    int64_t max_seqlen_kv = -1;
    std::string layout_q = "BSND";
    std::string layout_kv = "BSND";
    std::string layout_out = "BSND";
    int64_t return_softmax_lse = 0;

    FlashAttnHostUtParamBase(const csv_map &csvMap)
        : HostUtParamBase(csvMap)
    {
        softmax_scale = std::stof(ReadMap(csvMap, "softmax_scale", "0.0"));
        mask_mode = std::stoll(ReadMap(csvMap, "mask_mode", "0"));
        win_left = std::stoll(ReadMap(csvMap, "win_left", "-1"));
        win_right = std::stoll(ReadMap(csvMap, "win_right", "-1"));
        max_seqlen_q = std::stoll(ReadMap(csvMap, "max_seqlen_q", "-1"));
        max_seqlen_kv = std::stoll(ReadMap(csvMap, "max_seqlen_kv", "-1"));
        layout_q = ReadMap(csvMap, "layout_q", "BSND");
        layout_kv = ReadMap(csvMap, "layout_kv", "BSND");
        layout_out = ReadMap(csvMap, "layout_out", "BSND");
        return_softmax_lse = std::stoll(ReadMap(csvMap, "return_softmax_lse", "0"));
    }
};

struct FlashAttnTilingUtParam : public FlashAttnHostUtParamBase {
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;

    gert::TilingContextPara::TensorDescription q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription k = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription v = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription block_table = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cu_seqlens_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cu_seqlens_kv = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription seqused_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription seqused_kv = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription sinks = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription attn_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription metadata = TD_DEFAULT;

    gert::TilingContextPara::TensorDescription attn_out = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription softmax_lse = TD_DEFAULT;

    uint64_t expectTilingKey = 0;
    std::string expectTilingDataHash;

    FlashAttnCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};

    FlashAttnTilingUtParam(const csv_map &csvMap)
        : FlashAttnHostUtParamBase(csvMap)
    {
        inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        inputInstance.emplace_back(GetTensorGE(csvMap, "k_shape", "k_dtype", "k_format", this->k));
        inputInstance.emplace_back(GetTensorGE(csvMap, "v_shape", "v_dtype", "v_format", this->v));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "block_table_shape", "block_table_dtype", "block_table_format", this->block_table));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "cu_seqlens_q_shape", "cu_seqlens_q_dtype", "cu_seqlens_q_format", this->cu_seqlens_q));
        inputInstance.emplace_back(GetTensorGE(csvMap, "cu_seqlens_kv_shape", "cu_seqlens_kv_dtype",
                                               "cu_seqlens_kv_format", this->cu_seqlens_kv));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "seqused_q_shape", "seqused_q_dtype", "seqused_q_format", this->seqused_q));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "seqused_kv_shape", "seqused_kv_dtype", "seqused_kv_format", this->seqused_kv));
        inputInstance.emplace_back(GetTensorGE(csvMap, "sinks_shape", "sinks_dtype", "sinks_format", this->sinks));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_mask_shape", "attn_mask_dtype", "attn_mask_format", this->attn_mask));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "metadata_shape", "metadata_dtype", "metadata_format", this->metadata));

        outputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_out_shape", "attn_out_dtype", "attn_out_format", this->attn_out));
        outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            expectTilingKey = std::stoull(ReadMap(csvMap, "expectTilingKey", "0"));
            expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash", "");
        }
    }
};

struct FlashAttnInferShapeUtParam : public FlashAttnHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription k = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription v = ID_DEFAULT;

    gert::InfershapeContextPara::TensorDescription attn_out = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription softmax_lse = ID_DEFAULT;

    std::vector<std::vector<int64_t>> expectOutputShape;

    FlashAttnInferShapeUtParam(const csv_map &csvMap)
        : FlashAttnHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "k_shape", "k_dtype", "k_format", this->k));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "v_shape", "v_dtype", "v_format", this->v));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_out_shape", "attn_out_dtype", "attn_out_format", this->attn_out));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            std::string shapeStr = ReadMap(csvMap, "expect_attn_out_shape", "");
            if (!shapeStr.empty()) {
                std::vector<int64_t> shape;
                std::stringstream ss(shapeStr);
                int64_t dim;
                while (ss >> dim) {
                    shape.push_back(dim);
                }
                expectOutputShape.push_back(shape);
            }
            shapeStr = ReadMap(csvMap, "expect_softmax_lse_shape", "");
            if (!shapeStr.empty()) {
                std::vector<int64_t> shape;
                std::stringstream ss(shapeStr);
                int64_t dim;
                while (ss >> dim) {
                    shape.push_back(dim);
                }
                expectOutputShape.push_back(shape);
            }
        }
    }
};

struct FlashAttnInferDTypeUtParam : public FlashAttnHostUtParamBase {
    ge::DataType q_dtype = ge::DT_UNDEFINED;
    ge::DataType k_dtype = ge::DT_UNDEFINED;
    ge::DataType v_dtype = ge::DT_UNDEFINED;

    ge::DataType expect_attn_out_dtype = ge::DT_UNDEFINED;
    ge::DataType expect_softmax_lse_dtype = ge::DT_UNDEFINED;

    FlashAttnInferDTypeUtParam(const csv_map &csvMap)
        : FlashAttnHostUtParamBase(csvMap)
    {
        GetDataTypeGE(csvMap, "q_dtype", this->q_dtype);
        GetDataTypeGE(csvMap, "k_dtype", this->k_dtype);
        GetDataTypeGE(csvMap, "v_dtype", this->v_dtype);

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            GetDataTypeGE(csvMap, "expect_attn_out_dtype", this->expect_attn_out_dtype);
            GetDataTypeGE(csvMap, "expect_softmax_lse_dtype", this->expect_softmax_lse_dtype);
        }
    }
};

} // namespace FlashAttnUT

#endif // FLASH_ATTN_PARAM_H
