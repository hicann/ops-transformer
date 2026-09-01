/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MATMUL_API_UT_PARAM_H
#define ALL_GATHER_MATMUL_API_UT_PARAM_H

#include <sstream>
#include "op_api_csv_case_loader.h"

namespace AllGatherMatmulUT {

struct AllGatherMatmulApiUtParam {
    std::string case_name;
    TensorDesc x1;
    TensorDesc x2;
    TensorDesc bias;
    TensorDesc out;
    TensorDesc gatherOut;
    std::string group;
    int64_t gatherIndex;
    int64_t commTurn;
    int64_t streamMode;
    op::SocVersion soc;
    aclnnStatus expectResult;

    AllGatherMatmulApiUtParam(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->x1 = GetTensorACL(csvMap, "x1_shape", "x1_dtype", "x1_format");
        this->x2 = GetTensorACL(csvMap, "x2_shape", "x2_dtype", "x2_format");
        this->bias = GetTensorACL(csvMap, "bias_shape", "bias_dtype", "bias_format");
        this->out = GetTensorACL(csvMap, "out_shape", "out_dtype", "out_format");
        this->gatherOut = GetTensorACL(csvMap, "gather_out_shape", "gather_out_dtype", "gather_out_format");
        this->group = ReadMap(csvMap, "group");
        this->gatherIndex = stoll(ReadMap(csvMap, "gather_index", "0"));
        this->commTurn = stoll(ReadMap(csvMap, "comm_turn", "0"));
        this->streamMode = stoll(ReadMap(csvMap, "stream_mode", "0"));
        this->soc = GetCaseSocVersion(csvMap, "soc");
        this->expectResult = GetAclnnRet(csvMap, "expect_result");
    }
};

inline std::ostream &operator<<(std::ostream &os, const AllGatherMatmulApiUtParam &param)
{
    return os << param.case_name;
}

} // namespace AllGatherMatmulUT

#endif // ALL_GATHER_MATMUL_API_UT_PARAM_H
