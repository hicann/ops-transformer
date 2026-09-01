/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALLV_QUANT_GROUPED_MAT_MUL_API_UT_PARAM_H
#define ALLTO_ALLV_QUANT_GROUPED_MAT_MUL_API_UT_PARAM_H

#include <sstream>
#include "op_api_csv_case_loader.h"

namespace AlltoAllvQuantGroupedMatMulV2UT {

struct AlltoAllvQuantGroupedMatMulV2ApiUtParam {
    std::string case_name;
    TensorDesc gmmX;
    TensorDesc gmmWeight;
    TensorDesc gmmXScale;
    TensorDesc gmmWeightScale;
    TensorDesc sendCountsTensorOptional;
    TensorDesc recvCountsTensorOptional;
    TensorDesc mmXOptional;
    TensorDesc mmWeightOptional;
    TensorDesc mmXScaleOptional;
    TensorDesc mmWeightScaleOptional;
    int64_t gmmXQuantMode;
    int64_t gmmWeightQuantMode;
    int64_t mmXQuantMode;
    int64_t mmWeightQuantMode;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    aclIntArray *sendCounts;
    aclIntArray *recvCounts;
    int32_t transGmmWeight;
    int32_t transMmWeight;
    int32_t permuteOutFlag;
    TensorDesc gmmY;
    TensorDesc mmYOptional;
    TensorDesc permuteOutOptional;
    std::string group = "group";
    std::string commMode = "ccu";
    op::SocVersion soc;
    aclnnStatus expectResult;

    AlltoAllvQuantGroupedMatMulV2ApiUtParam(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->gmmX = GetTensorACL(csvMap, "gmmx_shape", "gmmx_dtype", "gmmx_format");
        this->gmmWeight = GetTensorACL(csvMap, "gmm_weight_shape", "gmm_weight_dtype", "gmm_weight_format");
        this->gmmXScale = GetTensorACL(csvMap, "gmmx_scale_shape", "gmmx_scale_dtype", "gmmx_scale_format");
        this->gmmWeightScale =
            GetTensorACL(csvMap, "gmm_weight_scale_shape", "gmm_weight_scale_dtype", "gmm_weight_scale_format");
        this->mmXOptional = GetTensorACL(csvMap, "mmx_shape", "mmx_dtype", "mmx_format");
        this->mmWeightOptional = GetTensorACL(csvMap, "mm_weight_shape", "mm_weight_dtype", "mm_weight_format");
        this->mmXScaleOptional = GetTensorACL(csvMap, "mmx_scale_shape", "mmx_scale_dtype", "mmx_scale_format");
        this->mmWeightScaleOptional =
            GetTensorACL(csvMap, "mm_weight_scale_shape", "mm_weight_scale_dtype", "mm_weight_scale_format");
        this->gmmY = GetTensorACL(csvMap, "gmmy_shape", "gmmy_dtype", "gmmy_format");
        this->mmYOptional = GetTensorACL(csvMap, "mmy_shape", "mmy_dtype", "mmy_format");
        this->permuteOutOptional = GetTensorACL(csvMap, "permuteout_shape", "permuteout_dtype", "permuteout_format");
        this->gmmXQuantMode = stoll(ReadMap(csvMap, "gmmx_quant_mode"));
        this->gmmWeightQuantMode = stoll(ReadMap(csvMap, "gmmx_quant_mode"));
        this->mmXQuantMode = stoll(ReadMap(csvMap, "mmx_quant_mode"));
        this->mmWeightQuantMode = stoll(ReadMap(csvMap, "gmmx_quant_mode"));
        this->transGmmWeight = stol(ReadMap(csvMap, "trans_gmm_weight"));
        this->transGmmWeight = stol(ReadMap(csvMap, "trans_mm_weight"));
        this->permuteOutFlag = stol(ReadMap(csvMap, "permuteout_flag"));
        auto sendCountsShape = GetShapeArr(ReadMap(csvMap, "send_counts"));
        std::vector<int64_t> sendCountsList(sendCountsShape[0], sendCountsShape[1]);
        this->sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
        auto recvCountsShape = GetShapeArr(ReadMap(csvMap, "recv_counts"));
        std::vector<int64_t> recvCountsList(sendCountsShape[0], sendCountsShape[1]);
        this->recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
        this->commMode = ReadMap(csvMap, "comm_mode", "");
        this->soc = GetCaseSocVersion(csvMap, "soc");
        this->expectResult = GetAclnnRet(csvMap, "expect_result");
    }
};

inline std::ostream &operator<<(std::ostream &os, const AlltoAllvQuantGroupedMatMulV2ApiUtParam &param)
{
    return os << param.case_name;
}

} // namespace AlltoAllvQuantGroupedMatMulV2UT

#endif // ALLTO_ALLV_QUANT_GROUPED_MAT_MUL_API_UT_PARAM_H
