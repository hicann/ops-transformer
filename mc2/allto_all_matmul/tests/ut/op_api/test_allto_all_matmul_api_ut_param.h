/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_API_UT_PARAM_H
#define ALLTO_ALL_MATMUL_API_UT_PARAM_H

#include <sstream>
#include <vector>
#include <cstdlib>
#include "op_api_csv_case_loader.h"
#include "op_api_ut_common/array_desc.h"

namespace AlltoAllMatmulUT {

struct AlltoAllMatmulApiUtParam {
    std::string case_name;
    TensorDesc x1;
    TensorDesc x2;
    TensorDesc bias;
    TensorDesc x1Scale;
    TensorDesc x2Scale;
    TensorDesc commScale;
    TensorDesc x1Offset;
    TensorDesc x2Offset;
    aclIntArray *alltoAllAxes;
    TensorDesc output;
    TensorDesc alltoAllOut;
    std::string group;
    int64_t x1QuantMode;
    int64_t x2QuantMode;
    int64_t commQuantMode;
    int64_t commQuantDtype;
    int64_t x1QuantDtype;
    int64_t groupSize;
    bool transposeX1;
    bool transposeX2;
    op::SocVersion soc;
    aclnnStatus expectResult;

    explicit AlltoAllMatmulApiUtParam(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->x1 = GetTensorACL(csvMap, "x1_shape", "x1_dtype", "x1_format");
        this->x2 = GetTensorACL(csvMap, "x2_shape", "x2_dtype", "x2_format");
        this->bias = GetTensorACL(csvMap, "bias_shape", "bias_dtype", "bias_format");
        this->x1Scale = GetTensorACL(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format");
        this->x2Scale = GetTensorACL(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format");
        this->commScale = GetTensorACL(csvMap, "comm_scale_shape", "comm_scale_dtype", "comm_scale_format");
        this->x1Offset = GetTensorACL(csvMap, "x1_offset_shape", "x1_offset_dtype", "x1_offset_format");
        this->x2Offset = GetTensorACL(csvMap, "x2_offset_shape", "x2_offset_dtype", "x2_offset_format");
        std::vector<int64_t> axesAcl = GetShapeArr(ReadMap(csvMap, "allto_all_axes"));
        this->alltoAllAxes = aclCreateIntArray(axesAcl.data(), axesAcl.size());
        this->x1QuantMode = stoll(ReadMap(csvMap, "x1_quant_mode", "0"));
        this->x2QuantMode = stoll(ReadMap(csvMap, "x2_quant_mode", "0"));
        this->commQuantMode = stoll(ReadMap(csvMap, "comm_quant_mode", "0"));
        this->commQuantDtype = stoll(ReadMap(csvMap, "comm_quant_dtype", "0"));
        this->x1QuantDtype = stoll(ReadMap(csvMap, "x1_quant_dtype", "0"));
        this->output = GetTensorACL(csvMap, "output_shape", "output_dtype", "output_format");
        this->alltoAllOut = GetTensorACL(csvMap, "allto_all_out_shape", "allto_all_out_dtype", "allto_all_out_format");
        this->group = ReadMap(csvMap, "group");
        this->groupSize = stoll(ReadMap(csvMap, "group_size", "0"));
        this->transposeX1 = stoi(ReadMap(csvMap, "transpose_x1"));
        this->transposeX2 = stoi(ReadMap(csvMap, "transpose_x2"));
        this->soc = GetCaseSocVersion(csvMap, "soc");
        this->expectResult = GetAclnnRet(csvMap, "expect_result");
    }
};

inline std::ostream &operator<<(std::ostream &os, const AlltoAllMatmulApiUtParam &param)
{
    return os << param.case_name;
}

} // namespace AlltoAllMatmulUT

#endif // ALLTO_ALL_MATMUL_API_UT_PARAM_H
