/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_ALL_GATHER_MATMUL_V3_HOST_UT_PARAM_H
#define TEST_ALL_GATHER_MATMUL_V3_HOST_UT_PARAM_H

#include <sstream>
#include "all_gather_matmul_v3_host_ut_param.h"

namespace AllGatherMatmulV3UT {

struct AllGatherMatmulV3InferShapeUtParam : public AllGatherMatmulV3HostUtParamBase {
    gert::InfershapeContextPara::TensorDescription context = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription bias = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x1Scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription x2Scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription y = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription gatherOut = ID_DEFAULT;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::vector<std::vector<int64_t>> expectOutputShape;

    explicit AllGatherMatmulV3InferShapeUtParam(const csv_map &csvMap)
        : AllGatherMatmulV3HostUtParamBase(csvMap)
    {
        // def 输入顺序: context(0) x1(1) x2(2) bias(3) x1_scale(4) x2_scale(5)，context 恒实例化
        this->context = gert::InfershapeContextPara::TensorDescription(gert::StorageShape({100}, {100}), ge::DT_INT32,
                                                                       ge::FORMAT_ND);
        this->inputInstance.emplace_back(1);
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", this->x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", this->x2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", this->bias));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format", this->x1Scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format", this->x2Scale));
        this->y = gert::InfershapeContextPara::TensorDescription({}, Str2DTypeGE(ReadMap(csvMap, "y_dtype", "BF16")),
                                                                 ge::FORMAT_ND);
        this->gatherOut = gert::InfershapeContextPara::TensorDescription(
            {}, Str2DTypeGE(ReadMap(csvMap, "gather_out_dtype", "FLOAT8_E4M3FN")), ge::FORMAT_ND);
        // def 输出顺序: y(0) gather_out(1) amax_out(2)，amax_out 未实现不实例化
        this->outputInstance = {1, 1, 0};
        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape.emplace_back(GetShapeArr(ReadMap(csvMap, "expect_y_shape")));
            this->expectOutputShape.emplace_back(GetShapeArr(ReadMap(csvMap, "expect_gather_out_shape")));
        }
    }
};

struct AllGatherMatmulV3InferDataTypeUtParam : public AllGatherMatmulV3HostUtParamBase {
    ge::DataType x1 = ge::DT_UNDEFINED;
    ge::DataType x2 = ge::DT_UNDEFINED;
    ge::DataType x1Scale = ge::DT_UNDEFINED;
    ge::DataType x2Scale = ge::DT_UNDEFINED;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    ge::DataType expectYDtype = ge::DT_UNDEFINED;
    ge::DataType expectGatherOutDtype = ge::DT_UNDEFINED;

    explicit AllGatherMatmulV3InferDataTypeUtParam(const csv_map &csvMap)
        : AllGatherMatmulV3HostUtParamBase(csvMap)
    {
        GetDataTypeGE(csvMap, "x1_dtype", this->x1);
        GetDataTypeGE(csvMap, "x2_dtype", this->x2);
        GetDataTypeGE(csvMap, "x1_scale_dtype", this->x1Scale);
        GetDataTypeGE(csvMap, "x2_scale_dtype", this->x2Scale);
        // def 输入顺序: context(0) x1(1) x2(2) bias(3) x1_scale(4) x2_scale(5)，bias 不实例化
        this->inputInstance = {1, 1, 1, 0, 1, 1};
        // def 输出顺序: y(0) gather_out(1) amax_out(2)，amax_out 未实现不实例化
        this->outputInstance = {1, 1, 0};
        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectYDtype = Str2DTypeGE(ReadMap(csvMap, "expect_y_dtype"));
            this->expectGatherOutDtype = Str2DTypeGE(ReadMap(csvMap, "expect_gather_out_dtype"));
        }
    }
};

} // namespace AllGatherMatmulV3UT

#endif // TEST_ALL_GATHER_MATMUL_V3_HOST_UT_PARAM_H
