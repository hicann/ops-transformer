/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_HOST_UT_PARAM_H
#define MOE_DISTRIBUTE_COMBINE_V2_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace MoeDistributeCombineV2UT {

struct MoeDistributeCombineV2HostUtParamBase {
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
    int64_t global_bs;
    int64_t out_dtype;
    int64_t comm_quant_mode;
    int64_t group_list_type;
    std::string comm_alg;
    int64_t zero_expert_num;
    int64_t copy_expert_num;
    int64_t const_expert_num;
    ge::graphStatus expectResult;

    MoeDistributeCombineV2HostUtParamBase(const csv_map &csvMap)
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
        this->global_bs = stoll(ReadMap(csvMap, "global_bs"));
        this->out_dtype = stoll(ReadMap(csvMap, "out_dtype"));
        this->comm_quant_mode = stoll(ReadMap(csvMap, "comm_quant_mode"));
        this->group_list_type = stoll(ReadMap(csvMap, "group_list_type"));
        this->comm_alg = ReadMap(csvMap, "comm_alg");
        this->zero_expert_num = stoll(ReadMap(csvMap, "zero_expert_num"));
        this->copy_expert_num = stoll(ReadMap(csvMap, "copy_expert_num"));
        this->const_expert_num = stoll(ReadMap(csvMap, "const_expert_num"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
    }
};

inline std::ostream &operator<<(std::ostream &os, const MoeDistributeCombineV2HostUtParamBase &param)
{
    return os << param.case_name;
}

struct MoeDistributeCombineV2TilingUtParam : public MoeDistributeCombineV2HostUtParamBase {
    gert::TilingContextPara::TensorDescription expand_x = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription expert_ids = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription assist_info_for_combine = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription ep_send_counts = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription expert_scales = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription tp_send_counts = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x_active_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription activation_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription weight_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription group_list = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription expand_scales = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription shared_expert_x = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription elastic_info = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription ori_x = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription const_expert_alpha_1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription const_expert_alpha_2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription const_expert_v = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription performance_info = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x = TD_DEFAULT;
    std::string soc;
    uint64_t coreNum;
    uint64_t ubsize;
    uint64_t ranksize;
    uint64_t expectTilingKey;

    MoeDistributeCombineV2TilingUtParam(const csv_map &csvMap)
        : MoeDistributeCombineV2HostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "expand_x_shape", "expand_x_dtype", "expand_x_format", expand_x));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "expert_ids_shape", "expert_ids_dtype", "expert_ids_format", expert_ids));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "assist_info_for_combine_shape",
                                                     "assist_info_for_combine_dtype", "assist_info_for_combine_format",
                                                     assist_info_for_combine));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "ep_send_counts_shape", "ep_send_counts_dtype",
                                                     "ep_send_counts_format", ep_send_counts));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "expert_scales_shape", "expert_scales_dtype", "expert_scales_format", expert_scales));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "tp_send_counts_shape", "tp_send_counts_dtype",
                                                     "tp_send_counts_format", tp_send_counts));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x_active_mask_shape", "x_active_mask_dtype", "x_active_mask_format", x_active_mask));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "activation_scale_shape", "activation_scale_dtype",
                                                     "activation_scale_format", activation_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "weight_scale_shape", "weight_scale_dtype", "weight_scale_format", weight_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "group_list_shape", "group_list_dtype", "group_list_format", group_list));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "expand_scales_shape", "expand_scales_dtype", "expand_scales_format", expand_scales));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "shared_expert_x_shape", "shared_expert_x_dtype",
                                                     "shared_expert_x_format", shared_expert_x));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "elastic_info_shape", "elastic_info_dtype", "elastic_info_format", elastic_info));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "ori_x_shape", "ori_x_dtype", "ori_x_format", ori_x));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "const_expert_alpha_1_shape", "const_expert_alpha_1_dtype",
                                                     "const_expert_alpha_1_format", const_expert_alpha_1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "const_expert_alpha_2_shape", "const_expert_alpha_2_dtype",
                                                     "const_expert_alpha_2_format", const_expert_alpha_2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "const_expert_v_shape", "const_expert_v_dtype",
                                                     "const_expert_v_format", const_expert_v));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "performance_info_shape", "performance_info_dtype",
                                                     "performance_info_format", performance_info));

        this->outputInstance.emplace_back(GetTensorGE(csvMap, "x_shape", "x_dtype", "x_format", x));

        this->soc = ReadMap(csvMap, "soc");
        this->coreNum = stoull(ReadMap(csvMap, "core_num"));
        this->ubsize = stoull(ReadMap(csvMap, "ubsize"));
        this->ranksize = stoull(ReadMap(csvMap, "ranksize"));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey"));
        }
    }
};

} // namespace MoeDistributeCombineV2UT

#endif // MOE_DISTRIBUTE_COMBINE_V2_HOST_UT_PARAM_H
