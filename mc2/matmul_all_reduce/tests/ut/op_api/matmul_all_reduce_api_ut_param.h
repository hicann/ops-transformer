/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_ALL_REDUCE_API_UT_PARAM_H
#define MATMUL_ALL_REDUCE_API_UT_PARAM_H

#include <sstream>
#include "op_api_csv_case_loader.h"

namespace MatmulAllReduceUT {

struct MatmulAllReduceApiUtParam {
    std::string case_name;
    TensorDesc x1;
    TensorDesc x2;
    TensorDesc bias;
    TensorDesc x3;
    TensorDesc dequantScale;
    TensorDesc pertokenScale;
    TensorDesc x1Scale;
    TensorDesc x2Scale;
    TensorDesc commQuantScale1;
    TensorDesc commQuantScale2;
    TensorDesc antiquantScale;
    TensorDesc antiquantOffset;
    TensorDesc output;
    std::string group;
    std::string reduceOp;
    std::string commMode;
    int64_t commTurn;
    int64_t streamMode;
    int64_t groupSize;
    int64_t commQuantMode;
    op::SocVersion soc;
    aclnnStatus expectResult;

    MatmulAllReduceApiUtParam(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->x1 = GetTensorACL(csvMap, "x1_shape", "x1_dtype", "x1_format");
        this->x2 = GetTensorACL(csvMap, "x2_shape", "x2_dtype", "x2_format");
        this->bias = GetTensorACL(csvMap, "bias_shape", "bias_dtype", "bias_format");
        this->x3 = GetTensorACL(csvMap, "x3_shape", "x3_dtype", "x3_format");
        this->dequantScale = GetTensorACL(csvMap, "dequant_scale_shape", "dequant_scale_dtype", "dequant_scale_format");
        this->pertokenScale =
            GetTensorACL(csvMap, "pertoken_scale_shape", "pertoken_scale_dtype", "pertoken_scale_format");
        this->x1Scale = GetTensorACL(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format");
        this->x2Scale = GetTensorACL(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format");
        this->commQuantScale1 =
            GetTensorACL(csvMap, "comm_quant_scale1_shape", "comm_quant_scale1_dtype", "comm_quant_scale1_format");
        this->commQuantScale2 =
            GetTensorACL(csvMap, "comm_quant_scale2_shape", "comm_quant_scale2_dtype", "comm_quant_scale2_format");
        this->antiquantScale =
            GetTensorACL(csvMap, "antiquant_scale_shape", "antiquant_scale_dtype", "antiquant_scale_format");
        this->antiquantOffset =
            GetTensorACL(csvMap, "antiquant_offset_shape", "antiquant_offset_dtype", "antiquant_offset_format");
        this->output = GetTensorACL(csvMap, "output_shape", "output_dtype", "output_format");
        this->group = ReadMap(csvMap, "group");
        this->reduceOp = ReadMap(csvMap, "reduce_op");
        this->commMode = ReadMap(csvMap, "comm_mode", "ai_cpu");
        if (this->commMode == "_") {
            this->commMode = "ai_cpu";
        }
        this->commTurn = stoll(ReadMap(csvMap, "comm_turn", "0"));
        this->streamMode = stoll(ReadMap(csvMap, "stream_mode", "0"));
        this->groupSize = stoll(ReadMap(csvMap, "group_size", "0"));
        this->commQuantMode = stoll(ReadMap(csvMap, "comm_quant_mode", "0"));
        this->soc = GetCaseSocVersion(csvMap, "soc");
        this->expectResult = GetAclnnRet(csvMap, "expect_result");
    }
};

inline std::ostream &operator<<(std::ostream &os, const MatmulAllReduceApiUtParam &param)
{
    return os << param.case_name;
}

} // namespace MatmulAllReduceUT

#endif // MATMUL_ALL_REDUCE_API_UT_PARAM_H
