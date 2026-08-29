/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_V2_API_UT_PARAM_H
#define ALLTO_ALL_MATMUL_V2_API_UT_PARAM_H

#include <sstream>
#include <vector>
#include <cstdlib>
#include "op_api_csv_case_loader.h"
#include "op_api_ut_common/array_desc.h"

namespace AlltoAllMatmulV2UT {

struct AlltoAllMatmulV2ApiUtParam {
    std::string case_name;
    TensorDesc context;
    TensorDesc x1;
    TensorDesc x2;
    TensorDesc bias;
    TensorDesc x1Scale;
    TensorDesc x2Scale;
    TensorDesc output;
    TensorDesc alltoAllOut;
    std::string group;
    int64_t worldSize;
    int64_t hcclBufferSize;
    int64_t x1QuantMode;
    int64_t x2QuantMode;
    int64_t groupSize;
    std::string commMode;
    int64_t precisionMode;
    op::SocVersion soc;
    aclnnStatus expectResult;

    explicit AlltoAllMatmulV2ApiUtParam(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->context = GetTensorACL(csvMap, "context_shape", "context_dtype", "context_format");
        this->x1 = GetTensorACL(csvMap, "x1_shape", "x1_dtype", "x1_format");
        this->x2 = GetTensorACL(csvMap, "x2_shape", "x2_dtype", "x2_format");
        {
            // 非连续转置 x2：view 为 [N, K]，storage 为 [K, N]，用于覆盖 aclnn 层 TransX2Tensor
            std::string x2StrideStr = ReadMap(csvMap, "x2_stride");
            if (!x2StrideStr.empty()) {
                aclDataType dtype = ReadMap(ACL_DTYPE, ReadMap(csvMap, "x2_dtype"), ACL_DT_UNDEFINED);
                aclFormat format = ReadMap(ACL_FORMAT, ReadMap(csvMap, "x2_format"), ACL_FORMAT_UNDEFINED);
                this->x2 = TensorDesc(GetShapeArr(ReadMap(csvMap, "x2_shape")), dtype, format, GetShapeArr(x2StrideStr),
                                      0, GetShapeArr(ReadMap(csvMap, "x2_storage_shape")));
            }
        }
        this->bias = GetTensorACL(csvMap, "bias_shape", "bias_dtype", "bias_format");
        this->x1Scale = GetTensorACL(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format");
        this->x2Scale = GetTensorACL(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format");
        this->output = GetTensorACL(csvMap, "output_shape", "output_dtype", "output_format");
        this->alltoAllOut = GetTensorACL(csvMap, "all2all_out_shape", "all2all_out_dtype", "all2all_out_format");
        this->group = ReadMap(csvMap, "group");
        this->worldSize = stoll(ReadMap(csvMap, "world_size", "2"));
        this->hcclBufferSize = stoll(ReadMap(csvMap, "hccl_buffer_size", "0"));
        this->x1QuantMode = stoll(ReadMap(csvMap, "x1_quant_mode", "6"));
        this->x2QuantMode = stoll(ReadMap(csvMap, "x2_quant_mode", "6"));
        this->groupSize = stoll(ReadMap(csvMap, "group_size", "0"));
        this->commMode = ReadMap(csvMap, "comm_mode", "urma");
        this->precisionMode = stoll(ReadMap(csvMap, "precision_mode", "0"));
        this->soc = GetCaseSocVersion(csvMap, "soc");
        this->expectResult = GetAclnnRet(csvMap, "expect_result");
    }
};

inline std::ostream &operator<<(std::ostream &os, const AlltoAllMatmulV2ApiUtParam &param)
{
    return os << param.case_name;
}

} // namespace AlltoAllMatmulV2UT

#endif // ALLTO_ALL_MATMUL_V2_API_UT_PARAM_H
