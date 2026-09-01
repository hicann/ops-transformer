/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MATMUL_HOST_UT_PARAM_H
#define ALL_GATHER_MATMUL_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace AllGatherMatmulUT {

struct AllGatherMatmulHostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group;
    bool is_trans_a;
    bool is_trans_b;
    int64_t gather_index;
    int64_t comm_turn;
    int64_t rank_size;
    bool is_gather_out;
    ge::graphStatus expectResult;

    AllGatherMatmulHostUtParamBase(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->is_trans_a = stoi(ReadMap(csvMap, "is_trans_a"));
        this->is_trans_b = stoi(ReadMap(csvMap, "is_trans_b"));
        this->gather_index = stoll(ReadMap(csvMap, "gather_index"));
        this->comm_turn = stoll(ReadMap(csvMap, "comm_turn"));
        this->rank_size = stoll(ReadMap(csvMap, "rank_size"));
        this->is_gather_out = stoi(ReadMap(csvMap, "is_gather_out"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
    }
};

inline std::ostream &operator<<(std::ostream &os, const AllGatherMatmulHostUtParamBase &param)
{
    return os << param.case_name;
}

struct AllGatherMatmulTilingUtParam : public AllGatherMatmulHostUtParamBase {
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gather_out = TD_DEFAULT;
    std::string soc;
    int64_t topo_type;
    uint64_t coreNum;
    uint64_t ubsize;
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    AllGatherMatmulTilingUtParam(const csv_map &csvMap)
        : AllGatherMatmulHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", x2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", bias));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "output_y_shape", "output_y_dtype", "output_y_format", y));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "output_gather_out_shape", "output_gather_out_dtype",
                                                      "output_gather_out_format", gather_out));

        this->soc = ReadMap(csvMap, "soc");
        this->topo_type = stoll(ReadMap(csvMap, "topo_type"));
        this->coreNum = stoull(ReadMap(csvMap, "core_num"));
        this->ubsize = stoull(ReadMap(csvMap, "ubsize"));
        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey"));
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};

struct AllGatherMatmulInferShapeUtParam : public AllGatherMatmulHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription x1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription bias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription gather_out = ID_DEFAULT;
    std::vector<std::vector<int64_t>> expectOutputShape;

    AllGatherMatmulInferShapeUtParam(const csv_map &csvMap)
        : AllGatherMatmulHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", x2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", bias));

        this->outputInstance.emplace_back(1);
        this->outputInstance.emplace_back(1);

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape = {GetShapeArr(ReadMap(csvMap, "expectOutputShape"))};
        }
    }
};

struct AllGatherMatmulInferDataTypeUtParam : public AllGatherMatmulHostUtParamBase {
    ge::DataType x1 = ge::DT_UNDEFINED;
    ge::DataType x2 = ge::DT_UNDEFINED;
    ge::DataType bias = ge::DT_UNDEFINED;
    ge::DataType y = ge::DT_UNDEFINED;
    ge::DataType gather_out = ge::DT_UNDEFINED;

    AllGatherMatmulInferDataTypeUtParam(const csv_map &csvMap)
        : AllGatherMatmulHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x1_dtype", x1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x2_dtype", x2));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "bias_dtype", bias));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "expect_y_dtype", y));
            this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "expect_gather_out_dtype", gather_out));
        }
    }
};

} // namespace AllGatherMatmulUT

#endif // ALL_GATHER_MATMUL_HOST_UT_PARAM_H
