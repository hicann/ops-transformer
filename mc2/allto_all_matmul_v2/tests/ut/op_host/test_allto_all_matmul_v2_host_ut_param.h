/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_V2_HOST_UT_PARAM_V2_H
#define ALLTO_ALL_MATMUL_V2_HOST_UT_PARAM_V2_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace AlltoAllMatmulV2UT {

struct AlltoAllMatmulV2HostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group;
    int64_t world_size;
    int64_t y_dtype_attr;
    bool transpose_x2;
    ge::graphStatus expectResult;

    explicit AlltoAllMatmulV2HostUtParamBase(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->world_size = stoll(ReadMap(csvMap, "world_size", "2"));
        this->y_dtype_attr = stoll(ReadMap(csvMap, "y_dtype_attr", "27"));
        this->transpose_x2 = StrToBoolIgnoreCase(ReadMap(csvMap, "transpose_x2", "true"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
    }
};

inline std::ostream &operator<<(std::ostream &os, const AlltoAllMatmulV2HostUtParamBase &param)
{
    return os << param.case_name;
}

struct AlltoAllMatmulV2InferShapeUtParam : public AlltoAllMatmulV2HostUtParamBase {
    gert::InfershapeContextPara::TensorDescription context = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription bias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x1Scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2Scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription all2allOut = ID_DEFAULT;
    std::vector<std::vector<int64_t>> expectOutputShape;

    explicit AlltoAllMatmulV2InferShapeUtParam(const csv_map &csvMap)
        : AlltoAllMatmulV2HostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "context_shape", "context_dtype", "context_format", context));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", x2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", bias));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format", x1Scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format", x2Scale));

        this->y = gert::InfershapeContextPara::TensorDescription({}, Str2DTypeGE(ReadMap(csvMap, "y_dtype", "BF16")),
                                                                 ge::FORMAT_ND);
        this->all2allOut = gert::InfershapeContextPara::TensorDescription(
            {}, Str2DTypeGE(ReadMap(csvMap, "all2all_out_dtype", "FLOAT8_E4M3FN")), ge::FORMAT_ND);

        this->outputInstance.emplace_back(1);
        this->outputInstance.emplace_back(1);

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape.emplace_back(GetShapeArr(ReadMap(csvMap, "expect_y_shape")));
        }
    }
};

struct AlltoAllMatmulV2InferDataTypeUtParam : public AlltoAllMatmulV2HostUtParamBase {
    ge::DataType context = ge::DT_INT32;
    ge::DataType x1 = ge::DT_UNDEFINED;
    ge::DataType x2 = ge::DT_UNDEFINED;
    ge::DataType bias = ge::DT_UNDEFINED;
    ge::DataType x1_scale = ge::DT_UNDEFINED;
    ge::DataType x2_scale = ge::DT_UNDEFINED;
    ge::DataType y = ge::DT_UNDEFINED;
    ge::DataType all2all_out = ge::DT_UNDEFINED;

    explicit AlltoAllMatmulV2InferDataTypeUtParam(const csv_map &csvMap)
        : AlltoAllMatmulV2HostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(1); // context always exists
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x1_dtype", x1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x2_dtype", x2));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "bias_dtype", bias));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x1_scale_dtype", x1_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x2_scale_dtype", x2_scale));

        this->outputInstance.emplace_back(1);
        this->outputInstance.emplace_back(1);

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->y = Str2DTypeGE(ReadMap(csvMap, "expect_y_dtype"));
            this->all2all_out = Str2DTypeGE(ReadMap(csvMap, "expect_all2all_out_dtype"));
        }
    }
};

} // namespace AlltoAllMatmulV2UT

#endif // ALLTO_ALL_MATMUL_V2_HOST_UT_PARAM_V2_H
