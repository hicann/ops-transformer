/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_HOST_UT_PARAM_H
#define MOE_DISTRIBUTE_DISPATCH_V2_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace MoeDistributeDispatchV2UT {
struct MoeDistributeDispatchV2HostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group_ep;
    int64_t ep_world_size;
    int64_t ep_rank_id;
    int64_t moe_expert_num;
    std::string group_tp;
    int64_t tp_world_size;
    int64_t tp_rank_id;
    int64_t expert_shard_type;
    int64_t shared_expert_num;
    int64_t shared_expert_rank_num;
    int64_t quant_mode;
    int64_t global_bs;
    int64_t expert_token_nums_type;
    std::string comm_alg;
    int64_t zero_expert_num;
    int64_t copy_expert_num;
    int64_t const_expert_num;
    ge::DataType y_dtype;
    ge::graphStatus expectResult;

    MoeDistributeDispatchV2HostUtParamBase(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group_ep = ReadMap(csvMap, "group_ep");
        this->ep_world_size = stoll(ReadMap(csvMap, "ep_world_size"));
        this->ep_rank_id = stoll(ReadMap(csvMap, "ep_rank_id"));
        this->moe_expert_num = stoll(ReadMap(csvMap, "moe_expert_num"));
        this->group_tp = ReadMap(csvMap, "group_tp");
        this->tp_world_size = stoll(ReadMap(csvMap, "tp_world_size"));
        this->tp_rank_id = stoll(ReadMap(csvMap, "tp_rank_id"));
        this->expert_shard_type = stoll(ReadMap(csvMap, "expert_shard_type"));
        this->shared_expert_num = stoll(ReadMap(csvMap, "shared_expert_num"));
        this->shared_expert_rank_num = stoll(ReadMap(csvMap, "shared_expert_rank_num"));
        this->quant_mode = stoll(ReadMap(csvMap, "quant_mode"));
        this->global_bs = stoll(ReadMap(csvMap, "global_bs"));
        this->expert_token_nums_type = stoll(ReadMap(csvMap, "expert_token_nums_type"));
        this->comm_alg = ReadMap(csvMap, "comm_alg");
        this->zero_expert_num = stoll(ReadMap(csvMap, "zero_expert_num"));
        this->copy_expert_num = stoll(ReadMap(csvMap, "copy_expert_num"));
        this->const_expert_num = stoll(ReadMap(csvMap, "const_expert_num"));
        this->y_dtype = Str2DTypeGE(ReadMap(csvMap, "y_dtype"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
    }
};

inline std::ostream &operator<<(std::ostream &os, const MoeDistributeDispatchV2HostUtParamBase &param)
{
    return os << param.case_name;
}

struct MoeDistributeDispatchV2TilingUtParam : public MoeDistributeDispatchV2HostUtParamBase {
    gert::TilingContextPara::TensorDescription x = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription expert_ids = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription scales = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x_active_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription expert_scales = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription elastic_info = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription performance_info = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription expand_x = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription dynamic_scales = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription assist_info_for_combine = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription expert_token_nums = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription ep_recv_count = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription tp_recv_count = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription expand_scales = TD_DEFAULT;
    std::string soc;
    uint64_t coreNum;
    uint64_t ubsize;
    uint64_t ranksize;
    uint64_t expectTilingKey;

    MoeDistributeDispatchV2TilingUtParam(const csv_map &csvMap)
        : MoeDistributeDispatchV2HostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x_shape", "x_dtype", "x_format", x));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "expert_ids_shape", "expert_ids_dtype", "expert_ids_format", expert_ids));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "scales_shape", "scales_dtype", "scales_format", scales));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x_active_mask_shape", "x_active_mask_dtype", "x_active_mask_format", x_active_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "expert_scales_shape", "expert_scales_dtype", "expert_scales_format", expert_scales));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "elastic_info_shape", "elastic_info_dtype", "elastic_info_format", elastic_info));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "performance_info_shape", "performance_info_dtype",
                                                     "performance_info_format", performance_info));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "expand_x_shape", "expand_x_dtype", "expand_x_format", expand_x));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "dynamic_scales_shape", "dynamic_scales_dtype",
                                                      "dynamic_scales_format", dynamic_scales));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "assist_info_for_combine_shape",
                                                      "assist_info_for_combine_dtype", "assist_info_for_combine_format",
                                                      assist_info_for_combine));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "expert_token_nums_shape", "expert_token_nums_dtype",
                                                      "expert_token_nums_format", expert_token_nums));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "ep_recv_count_shape", "ep_recv_count_dtype", "ep_recv_count_format", ep_recv_count));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "tp_recv_count_shape", "tp_recv_count_dtype", "tp_recv_count_format", tp_recv_count));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "expand_scales_shape", "expand_scales_dtype", "expand_scales_format", expand_scales));

        this->soc = ReadMap(csvMap, "soc");
        this->coreNum = stoull(ReadMap(csvMap, "core_num"));
        this->ubsize = stoull(ReadMap(csvMap, "ubsize"));
        this->ranksize = stoull(ReadMap(csvMap, "ranksize"));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey"));
        }
    }
};

} // namespace MoeDistributeDispatchV2UT

#endif // MOE_DISTRIBUTE_DISPATCH_V2_HOST_UT_PARAM_H
