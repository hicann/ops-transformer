/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INPLACE_MATMUL_ALL_REDUCE_ADD_RMS_NORM_HOST_UT_PARAM_H
#define INPLACE_MATMUL_ALL_REDUCE_ADD_RMS_NORM_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace InplaceMatmulAllReduceAddRmsNormUT {

struct InplaceMatmulAllReduceAddRmsNormInferShapeUtParam {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    gert::InfershapeContextPara::TensorDescription x1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription bias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription residual = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription gamma = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_offset = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription dequant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription norm_out = ID_DEFAULT;
    std::string group;
    std::string reduce_op;
    bool is_trans_a;
    bool is_trans_b;
    int64_t comm_turn;
    int64_t antiquant_group_size;
    float epsilon;
    uint64_t ranksize;
    ge::graphStatus expectResult;
    std::vector<std::vector<int64_t>> expectOutputShape;

    InplaceMatmulAllReduceAddRmsNormInferShapeUtParam(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->reduce_op = ReadMap(csvMap, "reduce_op");
        this->is_trans_a = stoi(ReadMap(csvMap, "is_trans_a"));
        this->is_trans_b = stoi(ReadMap(csvMap, "is_trans_b"));
        this->comm_turn = stoll(ReadMap(csvMap, "comm_turn"));
        this->antiquant_group_size = stoll(ReadMap(csvMap, "antiquant_group_size"));
        this->epsilon = stof(ReadMap(csvMap, "epsilon"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));

        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", this->x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", this->x2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", this->bias));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "residual_shape", "residual_dtype", "residual_format", this->residual));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "gamma_shape", "gamma_dtype", "gamma_format", this->gamma));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "antiquant_scale_shape", "antiquant_scale_dtype",
                                                     "antiquant_scale_format", this->antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "antiquant_offset_shape", "antiquant_offset_dtype",
                                                     "antiquant_offset_format", this->antiquant_offset));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "dequant_scale_shape", "dequant_scale_dtype",
                                                     "dequant_scale_format", this->dequant_scale));

        this->outputInstance.emplace_back(GetTensorGE(csvMap, "out_y_shape", "out_y_dtype", "out_y_format", this->y));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "out_norm_shape", "out_norm_dtype", "out_norm_format", this->norm_out));

        this->ranksize = stoull(ReadMap(csvMap, "ranksize"));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            std::string yShapeStr = ReadMap(csvMap, "expect_y_shape");
            std::string normShapeStr = ReadMap(csvMap, "expect_norm_shape");
            if (!yShapeStr.empty()) {
                this->expectOutputShape.emplace_back(GetShapeArr(yShapeStr));
            }
            if (!normShapeStr.empty()) {
                this->expectOutputShape.emplace_back(GetShapeArr(normShapeStr));
            }
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const InplaceMatmulAllReduceAddRmsNormInferShapeUtParam& param)
{
    return os << param.case_name;
}

struct InplaceMatmulAllReduceAddRmsNormInferDataTypeUtParam {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    ge::DataType x1 = ge::DT_UNDEFINED;
    ge::DataType x2 = ge::DT_UNDEFINED;
    ge::DataType bias = ge::DT_UNDEFINED;
    ge::DataType residual = ge::DT_UNDEFINED;
    ge::DataType gamma = ge::DT_UNDEFINED;
    ge::DataType antiquant_scale = ge::DT_UNDEFINED;
    ge::DataType antiquant_offset = ge::DT_UNDEFINED;
    ge::DataType dequant_scale = ge::DT_UNDEFINED;
    std::string group;
    std::string reduce_op;
    bool is_trans_a;
    bool is_trans_b;
    int64_t comm_turn;
    int64_t antiquant_group_size;
    float epsilon;
    uint64_t ranksize;
    ge::graphStatus expectResult;
    ge::DataType expect_y_dtype;
    ge::DataType expect_norm_dtype;

    InplaceMatmulAllReduceAddRmsNormInferDataTypeUtParam(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->reduce_op = ReadMap(csvMap, "reduce_op");
        this->is_trans_a = stoi(ReadMap(csvMap, "is_trans_a"));
        this->is_trans_b = stoi(ReadMap(csvMap, "is_trans_b"));
        this->comm_turn = stoll(ReadMap(csvMap, "comm_turn"));
        this->antiquant_group_size = stoll(ReadMap(csvMap, "antiquant_group_size"));
        this->epsilon = stof(ReadMap(csvMap, "epsilon"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));

        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x1_dtype", this->x1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "x2_dtype", this->x2));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "bias_dtype", this->bias));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "residual_dtype", this->residual));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "gamma_dtype", this->gamma));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "antiquant_scale_dtype", this->antiquant_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "antiquant_offset_dtype", this->antiquant_offset));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "dequant_scale_dtype", this->dequant_scale));

        this->outputInstance.emplace_back(1);
        this->outputInstance.emplace_back(1);

        this->ranksize = stoull(ReadMap(csvMap, "ranksize"));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expect_y_dtype = Str2DTypeGE(ReadMap(csvMap, "expect_y_dtype"));
            this->expect_norm_dtype = Str2DTypeGE(ReadMap(csvMap, "expect_norm_dtype"));
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const InplaceMatmulAllReduceAddRmsNormInferDataTypeUtParam& param)
{
    return os << param.case_name;
}

} // namespace InplaceMatmulAllReduceAddRmsNormUT

#endif // INPLACE_MATMUL_ALL_REDUCE_ADD_RMS_NORM_HOST_UT_PARAM_H
