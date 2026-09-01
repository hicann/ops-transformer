/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_HOST_UT_PARAM_H
#define ALLTO_ALL_MATMUL_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace AlltoAllMatmulUT {

struct AlltoAllMatmulHostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group;
    int64_t world_size;
    std::vector<int64_t> all2all_axes;
    ge::DataType y_dtype;
    int64_t x1_quant_mode;
    int64_t x2_quant_mode;
    int64_t comm_quant_mode;
    ge::DataType x1_quant_dtype;
    ge::DataType comm_quant_dtype;
    bool transpose_x1;
    bool transpose_x2;
    int64_t group_size;
    bool alltoall_out_flag;
    ge::graphStatus expectResult;

    explicit AlltoAllMatmulHostUtParamBase(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->world_size = stoll(ReadMap(csvMap, "world_size"));
        this->y_dtype = Str2DTypeGE(ReadMap(csvMap, "y_dtype"));
        this->x1_quant_mode = stoll(ReadMap(csvMap, "x1_quant_mode"));
        this->x2_quant_mode = stoll(ReadMap(csvMap, "x2_quant_mode"));
        this->comm_quant_mode = stoll(ReadMap(csvMap, "comm_quant_mode"));
        this->x1_quant_dtype = Str2DTypeGE(ReadMap(csvMap, "x1_quant_dtype"));
        this->comm_quant_dtype = Str2DTypeGE(ReadMap(csvMap, "comm_quant_dtype"));
        this->transpose_x1 = stoi(ReadMap(csvMap, "transpose_x1"));
        this->transpose_x2 = stoi(ReadMap(csvMap, "transpose_x2"));
        this->group_size = stoll(ReadMap(csvMap, "group_size"));
        this->alltoall_out_flag = stoi(ReadMap(csvMap, "alltoall_out_flag"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));

        // Parse all2all_axes as vector
        std::string axesStr = ReadMap(csvMap, "all2all_axes");
        if (axesStr != "UNDEFINED") {
            std::istringstream iss(axesStr);
            std::string token;
            while (std::getline(iss, token, ' ')) {
                this->all2all_axes.emplace_back(stoll(token));
            }
        }
    }
};

inline std::ostream &operator<<(std::ostream &os, const AlltoAllMatmulHostUtParamBase &param)
{
    return os << param.case_name;
}

struct AlltoAllMatmulInferShapeUtParam : public AlltoAllMatmulHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription x1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;
    std::vector<std::vector<int64_t>> expectOutputShape;

    explicit AlltoAllMatmulInferShapeUtParam(const csv_map &csvMap)
        : AlltoAllMatmulHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", x2));

        // Output tensors - use default values, shapes will be inferred
        this->outputInstance.emplace_back(1);
        this->outputInstance.emplace_back(1); // alltoallout output

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            std::string outputShapeStr = ReadMap(csvMap, "expectOutputShape");
            // Parse multiple output shapes separated by |
            std::istringstream iss(outputShapeStr);
            std::string token;
            while (std::getline(iss, token, '|')) {
                this->expectOutputShape.emplace_back(GetShapeArr(token));
            }
        }
    }
};

struct AlltoAllMatmulInferDataTypeUtParam : public AlltoAllMatmulHostUtParamBase {
    ge::DataType x1 = ge::DT_UNDEFINED;
    ge::DataType x2 = ge::DT_UNDEFINED;
    ge::DataType bias = ge::DT_UNDEFINED;
    ge::DataType x1_scale = ge::DT_UNDEFINED;
    ge::DataType x2_scale = ge::DT_UNDEFINED;
    ge::DataType comm_scale = ge::DT_UNDEFINED;
    ge::DataType x1_offset = ge::DT_UNDEFINED;
    ge::DataType x2_offset = ge::DT_UNDEFINED;
    ge::DataType y = ge::DT_UNDEFINED;
    ge::DataType alltoallout = ge::DT_UNDEFINED;

    explicit AlltoAllMatmulInferDataTypeUtParam(const csv_map &csvMap)
        : AlltoAllMatmulHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x1_dtype", x1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x2_dtype", x2));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "bias_dtype", bias));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x1_scale_dtype", x1_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x2_scale_dtype", x2_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "comm_scale_dtype", comm_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x1_offset_dtype", x1_offset));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x2_offset_dtype", x2_offset));

        this->outputInstance.emplace_back(1);
        this->outputInstance.emplace_back(1); // alltoallout output

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->y = Str2DTypeGE(ReadMap(csvMap, "expect_y_dtype"));
            this->alltoallout = Str2DTypeGE(ReadMap(csvMap, "expect_alltoallout_dtype"));
        }
    }
};

} // namespace AlltoAllMatmulUT

#endif // ALLTO_ALL_MATMUL_HOST_UT_PARAM_H
