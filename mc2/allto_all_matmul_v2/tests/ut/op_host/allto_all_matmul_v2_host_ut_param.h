/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_V2_HOST_UT_PARAM_H
#define ALLTO_ALL_MATMUL_V2_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace AlltoAllMatmulV2UT {
struct AlltoAllMatmulV2TilingUtParam {
    std::string case_name;
    gert::TilingContextPara::TensorDescription context = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x1Scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2Scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription all2allOut = TD_DEFAULT;
    std::string group;
    int64_t worldSize;
    int64_t hcclBufferSize;
    int64_t yDtypeAttr;
    int64_t x1QuantMode;
    int64_t x2QuantMode;
    int64_t x1QuantDtype;
    bool transposeX1;
    bool transposeX2;
    int64_t groupSize;
    std::string commMode;
    int64_t precisionMode;
    std::string soc;
    uint64_t coreNum;
    ge::graphStatus expectResult;
    uint64_t expectTilingKey;

    explicit AlltoAllMatmulV2TilingUtParam(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        GetTensorGE(csvMap, "context_shape", "context_dtype", "context_format", this->context);
        GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", this->x1);
        GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", this->x2);
        GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", this->bias);
        GetTensorGE(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format", this->x1Scale);
        GetTensorGE(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format", this->x2Scale);
        GetTensorGE(csvMap, "y_shape", "y_dtype", "y_format", this->y);
        GetTensorGE(csvMap, "all2all_out_shape", "all2all_out_dtype", "all2all_out_format", this->all2allOut);
        this->group = ReadMap(csvMap, "group");
        this->worldSize = stoll(ReadMap(csvMap, "world_size", "0"));
        this->hcclBufferSize = stoll(ReadMap(csvMap, "hccl_buffer_size", "0"));
        this->yDtypeAttr = stoll(ReadMap(csvMap, "y_dtype_attr", "28"));
        this->x1QuantMode = stoll(ReadMap(csvMap, "x1_quant_mode", "6"));
        this->x2QuantMode = stoll(ReadMap(csvMap, "x2_quant_mode", "6"));
        this->x1QuantDtype = stoll(ReadMap(csvMap, "x1_quant_dtype", "28"));
        this->transposeX1 = StrToBoolIgnoreCase(ReadMap(csvMap, "transpose_x1", "false"));
        this->transposeX2 = StrToBoolIgnoreCase(ReadMap(csvMap, "transpose_x2", "true"));
        this->groupSize = stoll(ReadMap(csvMap, "group_size", "0"));
        this->commMode = ReadMap(csvMap, "comm_mode", "urma");
        this->precisionMode = stoll(ReadMap(csvMap, "precision_mode", "0"));
        this->soc = ReadMap(csvMap, "soc_version", "3510");
        this->coreNum = stoull(ReadMap(csvMap, "core_num", "64"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
        std::string tilingKeyStr = ReadMap(csvMap, "expectTilingKey");
        if (!tilingKeyStr.empty()) {
            this->expectTilingKey = stoull(tilingKeyStr);
        } else {
            this->expectTilingKey = UINT64_MAX; // Skip validation marker
        }
    }
};

inline std::ostream &operator<<(std::ostream &os, const AlltoAllMatmulV2TilingUtParam &param)
{
    return os << param.case_name;
}

} // namespace AlltoAllMatmulV2UT

#endif // ALLTO_ALL_MATMUL_V2_HOST_UT_PARAM_H
