/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fallback_matmul_all_reduce.cpp
 * \brief
 */
#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"

namespace fallback {
inline const char* kInnerDebug = "MmAllReduceFallback";

struct MatmulParas {
    /* const aclTensor *会导致Release重载方法匹配不上，造成内存泄漏 */
    aclTensor* x1_acl;
    aclTensor* x2_acl;
    const gert::Tensor* bias;
};

inline ge::graphStatus GetMatmulPara(const gert::OpExecuteContext* host_api_ctx, size_t x1_idx, size_t x2_idx,
                                     size_t bias_idx, MatmulParas& para)
{
    const auto x1 = host_api_ctx->GetInputTensor(x1_idx);
    OPS_CHECK(x1 == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x1"), return ge::GRAPH_FAILED);

    const auto x2 = host_api_ctx->GetInputTensor(x2_idx);
    OPS_CHECK(x2 == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x2"), return ge::GRAPH_FAILED);

    para.bias = host_api_ctx->GetOptionalInputTensor(bias_idx);

    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "attrs"),
              return ge::GRAPH_FAILED);

    para.x1_acl = ConvertMmType(x1, false);
    OPS_CHECK(para.x1_acl == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x1_acl"),
              return ge::GRAPH_FAILED);

    const bool* trans_x2_ptr = attrs->GetBool(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_TRANS_X2));
    const bool x2_trans = (trans_x2_ptr != nullptr ? *trans_x2_ptr : false);
    para.x2_acl = ConvertMmType(x2, x2_trans, true);
    OPS_CHECK(para.x2_acl == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x2_acl"),
              return ge::GRAPH_FAILED);

    return ge::SUCCESS;
}

enum class Mc2QuantType : size_t {
    K_NONE_QUANT,
    K_WEIGHT_QUANT,
    K_QUANT,
    K_INVALID
};

struct QuantParas {
    aclTensor* antiquant_scale_acl;
    aclTensor* antiquant_offset_acl = nullptr;
    const gert::Tensor* dequant_scale;
    const gert::Tensor* pertoken_scale = nullptr;
    const gert::Tensor* comm_quant_scale_1 = nullptr;
    const gert::Tensor* comm_quant_scale_2 = nullptr;
    int64_t antiquant_group_size;
    Mc2QuantType type;
};

inline ge::graphStatus GetQuantPara(const gert::OpExecuteContext* host_api_ctx, size_t scale_idx, size_t offset_idx,
                                    size_t dequant_idx, size_t pertoken_idx, size_t comm_quant_scale_1_idx,
                                    size_t comm_quant_scale_2_idx, QuantParas& para)
{
    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "attrs"),
              return ge::GRAPH_FAILED);

    const auto antiquant_scale = host_api_ctx->GetOptionalInputTensor(scale_idx);
    if (antiquant_scale != nullptr) {
        const int64_t* p = attrs->GetInt(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_ANTIQUANT_GROUP_SIZE));
        para.antiquant_group_size = (p != nullptr ? *p : 0);

        const bool* trans_x2_ptr = attrs->GetBool(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_TRANS_X2));
        const bool trans_x2 = (trans_x2_ptr != nullptr ? *trans_x2_ptr : false);

        para.antiquant_scale_acl = ConvertMmType(antiquant_scale, trans_x2);
        OPS_CHECK(para.antiquant_scale_acl == nullptr,
                  OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "antiquant_scale"), return ge::GRAPH_FAILED);

        const auto antiquant_offset = host_api_ctx->GetOptionalInputTensor(offset_idx);
        if (antiquant_offset != nullptr) {
            para.antiquant_offset_acl = ConvertMmType(antiquant_offset, trans_x2);
            OPS_CHECK(para.antiquant_offset_acl == nullptr,
                      OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "antiquant_offset"),
                      return ge::GRAPH_FAILED);
        }
        para.type = Mc2QuantType::K_WEIGHT_QUANT;
    } else if ((para.dequant_scale = host_api_ctx->GetOptionalInputTensor(dequant_idx)) != nullptr) {
        if (comm_quant_scale_1_idx != static_cast<size_t>(-1) && comm_quant_scale_2_idx != static_cast<size_t>(-1)) {
            para.comm_quant_scale_1 = host_api_ctx->GetOptionalInputTensor(comm_quant_scale_1_idx);
            para.comm_quant_scale_2 = host_api_ctx->GetOptionalInputTensor(comm_quant_scale_2_idx);
        }
        if (pertoken_idx != static_cast<size_t>(-1)) {
            para.pertoken_scale = host_api_ctx->GetOptionalInputTensor(pertoken_idx);
        }
        para.type = Mc2QuantType::K_QUANT;
    } else {
        para.type = Mc2QuantType::K_NONE_QUANT;
    }

    return ge::SUCCESS;
}

struct CommParas {
    const char* group;
    const char* op;
    const int64_t stream_mode = 1; // STOP_ON_FAILURE
    int64_t comm_turn;
};

inline ge::graphStatus GetCommPara(const gert::OpExecuteContext* host_api_ctx, CommParas& para)
{
    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "attrs"),
              return ge::GRAPH_FAILED);

    para.group = attrs->GetStr(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_GROUP));
    OPS_CHECK(para.group == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "group"),
              return ge::GRAPH_FAILED);

    para.op = attrs->GetStr(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_OP));
    para.op = (para.op == nullptr ? "sum" : para.op);

    const int64_t* comm_turn_ptr = attrs->GetInt(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_COMM_TURN));
    para.comm_turn = (comm_turn_ptr != nullptr ? *comm_turn_ptr : 0);

    return ge::SUCCESS;
}

struct AddRmsNormParas {
    const gert::Tensor* residual;
    const gert::Tensor* gamma;
    double epsilon;
};

inline ge::graphStatus GetAddRmsNormPara(const gert::OpExecuteContext* host_api_ctx, AddRmsNormParas& para)
{
    para.residual = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2AddRmsNormInputIdx::K_RESIDUAL));
    OPS_CHECK(para.residual == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "residual"),
              return ge::GRAPH_FAILED);

    para.gamma = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2AddRmsNormInputIdx::K_GAMMA));
    OPS_CHECK(para.gamma == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "gamma"),
              return ge::GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "attrs"),
              return ge::GRAPH_FAILED);
    const float* epsilon_ptr = attrs->GetFloat(static_cast<size_t>(ops::MmAllReduceAddRmsNormAttrIdx::K_EPSILON));
    para.epsilon = (epsilon_ptr != nullptr ? static_cast<double>(*epsilon_ptr) : 1e-06);

    return ge::SUCCESS;
}

ge::graphStatus MatmulAllreduceExecuteFuncAddRmsNorm(gert::OpExecuteContext* host_api_ctx)
{
    OP_LOGD("Start to fallback for matmul all reduce add rms norm.");
    OPS_CHECK(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT(kInnerDebug, "host_api_ctx"),
              return ge::GRAPH_FAILED);

    MatmulParas mm_para;
    ge::graphStatus retPara = GetMatmulPara(host_api_ctx, static_cast<size_t>(ops::MC2AddRmsNormInputIdx::K_X1),
                                            static_cast<size_t>(ops::MC2AddRmsNormInputIdx::K_X2),
                                            static_cast<size_t>(ops::MC2AddRmsNormInputIdx::K_BIAS), mm_para);
    OPS_CHECK(retPara != ge::SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get matmul paras."),
              return ge::GRAPH_FAILED);

    QuantParas quant_para;
    retPara = GetQuantPara(host_api_ctx, static_cast<size_t>(ops::MC2AddRmsNormInputIdx::K_SCALE),
                           static_cast<size_t>(ops::MC2AddRmsNormInputIdx::K_OFFSET),
                           static_cast<size_t>(ops::MC2AddRmsNormInputIdx::K_DEQUANT), static_cast<size_t>(-1),
                           static_cast<size_t>(-1), static_cast<size_t>(-1), quant_para);
    OPS_CHECK(retPara != ge::SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get quant paras."),
              return ge::GRAPH_FAILED);

    CommParas comm_para;
    OPS_CHECK(GetCommPara(host_api_ctx, comm_para) != ge::SUCCESS,
              OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get comm paras."), return ge::GRAPH_FAILED);

    AddRmsNormParas add_rms_norm_para;
    OPS_CHECK(GetAddRmsNormPara(host_api_ctx, add_rms_norm_para) != ge::SUCCESS,
              OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get add rms norm paras."), return ge::GRAPH_FAILED);

    const auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MC2AddRmsNormOutputIdx::K_Y));
    OPS_CHECK(y == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "y"), return ge::GRAPH_FAILED);

    const auto norm = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MC2AddRmsNormOutputIdx::K_NORM_OUT));
    OPS_CHECK(norm == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "norm"),
              return ge::GRAPH_FAILED);

    switch (quant_para.type) {
        case Mc2QuantType::K_NONE_QUANT:
            return EXEC_OPAPI_CMD(aclnnMatmulAllReduceAddRmsNorm, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias,
                                  add_rms_norm_para.residual, add_rms_norm_para.gamma, add_rms_norm_para.epsilon,
                                  comm_para.group, comm_para.op, comm_para.comm_turn, comm_para.stream_mode, y, norm);
        case Mc2QuantType::K_WEIGHT_QUANT:
            return EXEC_OPAPI_CMD(aclnnWeightQuantMatmulAllReduceAddRmsNorm, mm_para.x1_acl, mm_para.x2_acl,
                                  mm_para.bias, quant_para.antiquant_scale_acl, quant_para.antiquant_offset_acl,
                                  add_rms_norm_para.residual, add_rms_norm_para.gamma, add_rms_norm_para.epsilon,
                                  comm_para.group, comm_para.op, comm_para.comm_turn, comm_para.stream_mode,
                                  quant_para.antiquant_group_size, y, norm);
        case Mc2QuantType::K_QUANT:
            return EXEC_OPAPI_CMD(aclnnQuantMatmulAllReduceAddRmsNorm, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias,
                                  quant_para.dequant_scale, add_rms_norm_para.residual, add_rms_norm_para.gamma,
                                  add_rms_norm_para.epsilon, comm_para.group, comm_para.op, comm_para.comm_turn,
                                  comm_para.stream_mode, y, norm);
        default:
            return ge::GRAPH_FAILED;
    }
}
IMPL_OP(MatmulAllReduceAddRmsNorm).OpExecuteFunc(MatmulAllreduceExecuteFuncAddRmsNorm);
} // namespace fallback
