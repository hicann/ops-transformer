/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLTO_ALLV_QUANT_GROUPED_MAT_MUL_HOST_UT_PARAM_H
#define ALLTO_ALLV_QUANT_GROUPED_MAT_MUL_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace AlltoAllvQuantGroupedMatMulUT {

struct AlltoAllvQuantGroupedMatMulTilingUTParamBase {
    std::string case_name;
    std::vector<int64_t> sendCounts;
    std::vector<int64_t> recvCounts;
    // Attributes
    int32_t gmm_x_quant_mode;
    int32_t gmm_weight_quant_mode;
    int32_t mm_x_quant_mode;
    int32_t mm_weight_quant_mode;
    bool trans_gmm_weight_flag;
    bool trans_mm_weight_flag;
    bool permute_out_flag;
    bool mm_out_flag;
    int64_t world_size;
    std::string comm_mode;
    int64_t ep_world_size;
    int64_t graph_type;
    int64_t group_size;
    // Expected tiling key
    uint64_t expectTilingKey;
    // Expected result
    ge::graphStatus expectedStatus;

    AlltoAllvQuantGroupedMatMulTilingUTParamBase(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->gmm_x_quant_mode = stoi(ReadMap(csvMap, "gmm_x_quant_mode"));
        this->gmm_weight_quant_mode = stoi(ReadMap(csvMap, "gmm_weight_quant_mode"));
        this->mm_x_quant_mode = stoi(ReadMap(csvMap, "mm_x_quant_mode"));
        this->mm_weight_quant_mode = stoi(ReadMap(csvMap, "mm_weight_quant_mode"));
        this->trans_gmm_weight_flag = stoi(ReadMap(csvMap, "trans_gmm_weight_flag"));
        this->trans_mm_weight_flag = stoi(ReadMap(csvMap, "trans_mm_weight_flag"));
        this->permute_out_flag = stoi(ReadMap(csvMap, "permute_out_flag"));
        this->mm_out_flag = stoi(ReadMap(csvMap, "mm_out_flag"));
        this->world_size = stoll(ReadMap(csvMap, "world_size"));
        this->comm_mode = ReadMap(csvMap, "comm_mode", "ccu");
        this->ep_world_size = stoll(ReadMap(csvMap, "ep_world_size"));
        this->graph_type = stoll(ReadMap(csvMap, "graph_type"));
        this->group_size = stoll(ReadMap(csvMap, "group_size"));
        this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey"));
        this->expectedStatus = Str2StatusGE(ReadMap(csvMap, "expectedStatus"));
        auto sendCountsShape = GetShapeArr(ReadMap(csvMap, "send_counts"));
        this->sendCounts = std::vector<int64_t>(sendCountsShape[0], sendCountsShape[1]);
        auto recvCountsShape = GetShapeArr(ReadMap(csvMap, "recv_counts"));
        this->recvCounts = std::vector<int64_t>(recvCountsShape[0], recvCountsShape[1]);
    }
};

inline std::ostream &operator<<(std::ostream &os, const AlltoAllvQuantGroupedMatMulTilingUTParamBase &param)
{
    return os << param.case_name;
}

struct AlltoAllvQuantGroupedMatMulTilingUTParam : public AlltoAllvQuantGroupedMatMulTilingUTParamBase {
    // input
    gert::TilingContextPara::TensorDescription gmm_x = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gmm_weight = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gmm_x_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gmm_weight_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mm_x = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mm_x_weight = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mm_x_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mm_x_weight_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription send_counts_tensor = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription recv_counts_tensor = TD_DEFAULT;
    // output: expected output tensor
    gert::TilingContextPara::TensorDescription gmm_y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription mm_y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription permute_out = TD_DEFAULT;

    AlltoAllvQuantGroupedMatMulTilingUTParam(const csv_map &csvMap)
        : AlltoAllvQuantGroupedMatMulTilingUTParamBase(csvMap)
    {
        GetTensorGE(csvMap, "gmm_x_shape", "gmm_x_dtype", "gmm_x_format", gmm_x);
        GetTensorGE(csvMap, "gmm_weight_shape", "gmm_weight_dtype", "gmm_weight_format", gmm_weight);
        GetTensorGE(csvMap, "gmm_x_scale_shape", "gmm_x_scale_dtype", "gmm_x_scale_format", gmm_x_scale);
        GetTensorGE(csvMap, "gmm_weight_scale_shape", "gmm_weight_scale_dtype", "gmm_weight_scale_format",
                    gmm_weight_scale);
        GetTensorGE(csvMap, "mm_x_shape", "mm_x_dtype", "mm_x_format", mm_x);
        GetTensorGE(csvMap, "mm_x_weight_shape", "mm_x_weight_dtype", "mm_x_weight_format", mm_x_weight);
        GetTensorGE(csvMap, "mm_x_scale_shape", "mm_x_scale_dtype", "mm_x_scale_format", mm_x_scale);
        GetTensorGE(csvMap, "mm_x_weight_scale_shape", "mm_x_weight_scale_dtype", "mm_x_weight_scale_format",
                    mm_x_weight_scale);
        GetTensorGE(csvMap, "gmm_y_shape", "gmm_y_dtype", "gmm_y_format", gmm_y);
        GetTensorGE(csvMap, "mm_y_shape", "mm_y_dtype", "mm_y_format", mm_y);
        GetTensorGE(csvMap, "permute_out_shape", "permute_out_dtype", "permute_out_format", permute_out);
        GetTensorGE(csvMap, "send_counts_tensor_shape", "send_counts_tensor_dtype", "send_counts_tensor_format",
                    send_counts_tensor);
        GetTensorGE(csvMap, "recv_counts_tensor_shape", "recv_counts_tensor_dtype", "recv_counts_tensor_format",
                    recv_counts_tensor);
    }
};

} // namespace AlltoAllvQuantGroupedMatMulUT

#endif // ALLTO_ALLV_QUANT_GROUPED_MAT_MUL_HOST_UT_PARAM_H
