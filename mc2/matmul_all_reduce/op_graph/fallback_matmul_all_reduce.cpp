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
                  OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "antiquantScale"), return ge::GRAPH_FAILED);

        const auto antiquant_offset = host_api_ctx->GetOptionalInputTensor(offset_idx);
        if (antiquant_offset != nullptr) {
            para.antiquant_offset_acl = ConvertMmType(antiquant_offset, trans_x2);
            OPS_CHECK(para.antiquant_offset_acl == nullptr,
                      OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "antiquantOffset"),
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
    const char* comm_mode{nullptr};
};

static inline ge::graphStatus GetCommParaBase(const gert::OpExecuteContext* host_api_ctx, CommParas& para)
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

    para.comm_mode = attrs->GetStr(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_COMM_MODE));

    return ge::SUCCESS;
}
struct AllReduceParas {
    int64_t group_size;
    int64_t y_dtype;
    int64_t comm_quant_mode;
};

ge::graphStatus GetAllReducePara(const gert::OpExecuteContext* host_api_ctx, AllReduceParas& para)
{
    const auto attrs = host_api_ctx->GetAttrs();

    OPS_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "attrs"),
              return ge::GRAPH_FAILED);

    const int64_t* group_size_ptr = attrs->GetInt(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_GROUP_SIZE));
    para.group_size = (group_size_ptr != nullptr ? *group_size_ptr : 0);

    const int64_t* y_dtype_ptr = attrs->GetInt(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_DTYPE_Y));
    para.y_dtype = (y_dtype_ptr != nullptr ? *y_dtype_ptr : static_cast<uint64_t>(ge::DataType::DT_UNDEFINED));

    const int64_t* comm_quant_mode_ptr = attrs->GetInt(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_COMM_QUANT_MODE));
    para.comm_quant_mode = (comm_quant_mode_ptr != nullptr ? *comm_quant_mode_ptr : 0);
    return ge::SUCCESS;
}

ge::graphStatus MatmulAllreduceExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OP_LOGD(kInnerDebug, "Start to fallback for matmul all reduce.");
    OPS_CHECK(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT(kInnerDebug, "host_api_ctx"),
              return ge::GRAPH_FAILED);

    MatmulParas mm_para;
    ge::graphStatus retPara = GetMatmulPara(host_api_ctx, static_cast<size_t>(ops::MC2InputIdx::K_X1),
                                            static_cast<size_t>(ops::MC2InputIdx::K_X2),
                                            static_cast<size_t>(ops::MC2InputIdx::K_BIAS), mm_para);
    OPS_CHECK(retPara != ge::SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get matmul paras."),
              return ge::GRAPH_FAILED);

    QuantParas quant_para;
    retPara = GetQuantPara(
        host_api_ctx, static_cast<size_t>(ops::MC2InputIdx::K_SCALE), static_cast<size_t>(ops::MC2InputIdx::K_OFFSET),
        static_cast<size_t>(ops::MC2InputIdx::K_DEQUANT), static_cast<size_t>(ops::MC2InputIdx::K_PERTOKEN),
        static_cast<size_t>(ops::MC2InputIdx::K_COMMQUANTSCALE1),
        static_cast<size_t>(ops::MC2InputIdx::K_COMMQUANTSCALE2), quant_para);
    OPS_CHECK(retPara != ge::SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get quant paras."),
              return ge::GRAPH_FAILED);

    CommParas comm_para;
    OPS_CHECK(GetCommParaBase(host_api_ctx, comm_para) != ge::SUCCESS,
              OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get comm paras."), return ge::GRAPH_FAILED);

    AllReduceParas all_reduce_para;
    OPS_CHECK(GetAllReducePara(host_api_ctx, all_reduce_para) != ge::SUCCESS,
              OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get all reduce paras."), return ge::GRAPH_FAILED);

    const auto x3 = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2InputIdx::K_X3));
    const auto comm_quant_scale_1 =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2InputIdx::K_COMMQUANTSCALE1));
    const auto comm_quant_scale_2 =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2InputIdx::K_COMMQUANTSCALE2));
    const auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MC2OutputIdx::K_Y));
    OPS_CHECK(y == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "y"), return ge::GRAPH_FAILED);

    bool isCommMode = false;
    if (comm_para.comm_mode != nullptr && std::strcmp(comm_para.comm_mode, "ccu") == 0) {
        OP_LOGD(kInnerDebug, "isCommMode is true");
        isCommMode = true;
    } else {
        OP_LOGD(kInnerDebug, "isCommMode is false");
    }

    switch (quant_para.type) {
        case Mc2QuantType::K_NONE_QUANT:
            if (isCommMode) {
                return EXEC_OPAPI_CMD(aclnnMatmulAllReduceV3, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, x3,
                                      comm_para.group, comm_para.op, comm_para.comm_mode, comm_para.comm_turn,
                                      comm_para.stream_mode, y);
            } else {
                if (x3 != nullptr) {
                    return EXEC_OPAPI_CMD(aclnnMatmulAllReduceV2, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, x3,
                                          comm_para.group, comm_para.op, comm_para.comm_turn, comm_para.stream_mode, y);
                } else {
                    return EXEC_OPAPI_CMD(aclnnMatmulAllReduce, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias,
                                          comm_para.group, comm_para.op, comm_para.comm_turn, comm_para.stream_mode, y);
                }
            }
        case Mc2QuantType::K_WEIGHT_QUANT:
            if (isCommMode) {
                return EXEC_OPAPI_CMD(aclnnWeightQuantMatmulAllReduceV2, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias,
                                      quant_para.antiquant_scale_acl, quant_para.antiquant_offset_acl, x3,
                                      comm_para.group, comm_para.op, comm_para.comm_mode, comm_para.comm_turn,
                                      comm_para.stream_mode, quant_para.antiquant_group_size, y);
            }
            return EXEC_OPAPI_CMD(aclnnWeightQuantMatmulAllReduce, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias,
                                  quant_para.antiquant_scale_acl, quant_para.antiquant_offset_acl, x3, comm_para.group,
                                  comm_para.op, comm_para.comm_turn, comm_para.stream_mode,
                                  quant_para.antiquant_group_size, y);
        case Mc2QuantType::K_QUANT:
            if (all_reduce_para.y_dtype != static_cast<uint64_t>(ge::DataType::DT_UNDEFINED)) {
                if (isCommMode) {
                    return EXEC_OPAPI_CMD(aclnnQuantMatmulAllReduceV5, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, x3,
                                          quant_para.pertoken_scale, quant_para.dequant_scale,
                                          quant_para.comm_quant_scale_1, quant_para.comm_quant_scale_2, comm_para.group,
                                          comm_para.op, comm_para.comm_mode, comm_para.comm_turn, comm_para.stream_mode,
                                          all_reduce_para.group_size, all_reduce_para.comm_quant_mode, y);
                }
                return EXEC_OPAPI_CMD(aclnnQuantMatmulAllReduceV4, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, x3,
                                      quant_para.pertoken_scale, quant_para.dequant_scale,
                                      quant_para.comm_quant_scale_1, quant_para.comm_quant_scale_2, comm_para.group,
                                      comm_para.op, comm_para.comm_turn, comm_para.stream_mode,
                                      all_reduce_para.group_size, all_reduce_para.comm_quant_mode, y);
            } else if ((comm_quant_scale_1 != nullptr) && (comm_quant_scale_2 != nullptr)) {
                if (isCommMode) {
                    return EXEC_OPAPI_CMD(aclnnQuantMatmulAllReduceV5, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, x3,
                                          quant_para.pertoken_scale, quant_para.dequant_scale,
                                          quant_para.comm_quant_scale_1, quant_para.comm_quant_scale_2, comm_para.group,
                                          comm_para.op, comm_para.comm_mode, comm_para.comm_turn, comm_para.stream_mode,
                                          all_reduce_para.group_size, all_reduce_para.comm_quant_mode, y);
                }
                return EXEC_OPAPI_CMD(aclnnQuantMatmulAllReduceV3, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, x3,
                                      quant_para.dequant_scale, quant_para.pertoken_scale,
                                      quant_para.comm_quant_scale_1, quant_para.comm_quant_scale_2, comm_para.group,
                                      comm_para.op, comm_para.comm_turn, comm_para.stream_mode, y);
            } else {
                if (isCommMode) {
                    return EXEC_OPAPI_CMD(aclnnQuantMatmulAllReduceV5, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, x3,
                                          quant_para.pertoken_scale, quant_para.dequant_scale,
                                          quant_para.comm_quant_scale_1, quant_para.comm_quant_scale_2, comm_para.group,
                                          comm_para.op, comm_para.comm_mode, comm_para.comm_turn, comm_para.stream_mode,
                                          all_reduce_para.group_size, all_reduce_para.comm_quant_mode, y);
                }
                return EXEC_OPAPI_CMD(aclnnQuantMatmulAllReduceV2, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, x3,
                                      quant_para.dequant_scale, quant_para.pertoken_scale, comm_para.group,
                                      comm_para.op, comm_para.comm_turn, comm_para.stream_mode, y);
            }
        default:
            return ge::GRAPH_FAILED;
    }
}
IMPL_OP(MatmulAllReduce).OpExecuteFunc(MatmulAllreduceExecuteFunc);
} // namespace fallback
