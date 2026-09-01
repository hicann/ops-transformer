/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTERV2_HOST_UT_PARAM_H
#define MATMUL_REDUCE_SCATTERV2_HOST_UT_PARAM_H

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "op_host_csv_case_loader.h"

namespace matmul_reduce_scatter_v2_ut {

struct MatmulReduceScatterV2HostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group;
    std::string reduce_op;
    bool is_trans_a;
    bool is_trans_b;
    int64_t comm_turn;
    int64_t rank_size;
    int64_t block_size;
    int64_t group_size;
    bool is_amax_out;
    ge::DataType y_dtype;
    std::string comm_mode;
    ge::graphStatus expectResult;

    MatmulReduceScatterV2HostUtParamBase(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->reduce_op = ReadMap(csvMap, "reduce_op");
        this->is_trans_a = stoi(ReadMap(csvMap, "is_trans_a"));
        this->is_trans_b = stoi(ReadMap(csvMap, "is_trans_b"));
        this->comm_turn = stoll(ReadMap(csvMap, "comm_turn"));
        this->rank_size = stoll(ReadMap(csvMap, "rank_size"));
        this->block_size = stoll(ReadMap(csvMap, "block_size"));
        this->group_size = stoll(ReadMap(csvMap, "group_size"));
        this->is_amax_out = stoi(ReadMap(csvMap, "is_amax_out"));
        GetDataTypeGE(csvMap, "y_dtype", y_dtype);
        this->comm_mode = ReadMap(csvMap, "comm_mode");
        this->expectResult = ReadMap(csvMap, "expectResult") == "SUCCESS" ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    }
};

inline std::ostream &operator<<(std::ostream &os, const MatmulReduceScatterV2HostUtParamBase &param)
{
    return os << param.case_name;
}

template <typename T>
inline std::string GetCaseInfoString(const testing::TestParamInfo<T> &info)
{
    return info.param.case_name;
}

const gert::TilingContextPara::TensorDescription TD_DEFAULT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};
struct MatmulReduceScatterV2TilingUtParam : public MatmulReduceScatterV2HostUtParamBase {
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x1_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription quant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription amax_out = TD_DEFAULT;
    std::string soc;
    uint64_t coreNum;
    uint64_t ubsize;
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    MatmulReduceScatterV2TilingUtParam(const csv_map &csvMap)
        : MatmulReduceScatterV2HostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", x2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", bias));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format", x1_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format", x2_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "quant_scale_shape", "quant_scale_dtype", "quant_scale_format", quant_scale));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "output_y_shape", "output_y_dtype", "output_y_format", y));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "amax_out_shape", "amax_out_dtype", "amax_out_format", amax_out));
        this->soc = ReadMap(csvMap, "soc");
        this->coreNum = stoull(ReadMap(csvMap, "core_num"));
        this->ubsize = stoull(ReadMap(csvMap, "ubsize"));
        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey"));
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};
} // namespace matmul_reduce_scatter_v2_ut

#endif // MATMUL_REDUCESCATTER_V2_HOST_UT_PARAM_H
