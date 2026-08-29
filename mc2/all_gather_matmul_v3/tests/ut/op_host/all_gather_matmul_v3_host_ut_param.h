/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MATMUL_V3_HOST_UT_PARAM_H
#define ALL_GATHER_MATMUL_V3_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace AllGatherMatmulV3UT {

struct AllGatherMatmulV3HostUtParamBase {
    std::string case_name;
    ge::graphStatus expectResult;
    std::string group;
    int64_t hcclBufferSize;
    bool isTransA;
    bool isTransB;
    int64_t rankSize;
    int64_t groupSize;
    int64_t yDtypeAttr;
    std::string commMode;
    uint64_t rankNum;

    explicit AllGatherMatmulV3HostUtParamBase(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
        this->group = ReadMap(csvMap, "group", "group");
        this->hcclBufferSize = stoll(ReadMap(csvMap, "hccl_buffer_size", "200"));
        this->isTransA = StrToBoolIgnoreCase(ReadMap(csvMap, "is_trans_a", "false"));
        this->isTransB = StrToBoolIgnoreCase(ReadMap(csvMap, "is_trans_b", "true"));
        this->rankSize = stoll(ReadMap(csvMap, "rank_size", "4"));
        this->groupSize = stoll(ReadMap(csvMap, "group_size", "4295032864"));
        this->yDtypeAttr = stoll(ReadMap(csvMap, "y_dtype_attr", "27"));
        this->commMode = ReadMap(csvMap, "comm_mode", "urma");
        this->rankNum = static_cast<uint64_t>(stoull(ReadMap(csvMap, "rank_num", "4")));
    }
};

inline std::ostream &operator<<(std::ostream &os, const AllGatherMatmulV3HostUtParamBase &param)
{
    return os << param.case_name;
}

struct AllGatherMatmulV3TilingUtParam : public AllGatherMatmulV3HostUtParamBase {
    gert::TilingContextPara::TensorDescription context = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x1Scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2Scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gatherOut = TD_DEFAULT;
    std::string soc;
    uint64_t expectTilingKey;

    explicit AllGatherMatmulV3TilingUtParam(const csv_map &csvMap)
        : AllGatherMatmulV3HostUtParamBase(csvMap)
    {
        GetTensorGE(csvMap, "context_shape", "context_dtype", "context_format", this->context);
        GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", this->x1);
        GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", this->x2);
        GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", this->bias);
        GetTensorGE(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format", this->x1Scale);
        GetTensorGE(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format", this->x2Scale);
        GetTensorGE(csvMap, "y_shape", "y_dtype", "y_format", this->y);
        GetTensorGE(csvMap, "gather_out_shape", "gather_out_dtype", "gather_out_format", this->gatherOut);
        this->soc = ReadMap(csvMap, "soc_version", "3510");
        std::string tilingKeyStr = ReadMap(csvMap, "expectTilingKey");
        this->expectTilingKey = tilingKeyStr.empty() ? UINT64_MAX : stoull(tilingKeyStr);
    }
};

} // namespace AllGatherMatmulV3UT

#endif // ALL_GATHER_MATMUL_V3_HOST_UT_PARAM_H
