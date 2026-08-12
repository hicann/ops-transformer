/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fallback_allto_all_matmul.cpp
 * \brief 动态shape图回调aclnn
 */
#include "common/utils/op_mc2.h"
#include "fallback/fallback.h"
#include "mc2_common_log.h"

namespace fallback {

constexpr size_t INDEX_IN_X1 = 0;
constexpr size_t INDEX_IN_X2 = 1;
constexpr size_t INDEX_IN_BIAS = 2;
constexpr size_t INDEX_IN_X1_SCALE = 3;
constexpr size_t INDEX_IN_X2_SCALE = 4;
constexpr size_t INDEX_IN_COMM_SCALE = 5;
constexpr size_t INDEX_IN_X1_OFFSET = 6;
constexpr size_t INDEX_IN_X2_OFFSET = 7;
constexpr size_t INDEX_ATTR_GROUP = 0;
constexpr size_t INDEX_ATTR_WORLD_SIZE = 1;
constexpr size_t INDEX_ATTR_ALL2ALL_AXES = 2;
constexpr size_t INDEX_ATTR_Y_DTYPE = 3;
constexpr size_t INDEX_ATTR_X1_QUANT_MODE = 4;
constexpr size_t INDEX_ATTR_X2_QUANT_MODE = 5;
constexpr size_t INDEX_ATTR_COMMON_QUANT_MODE = 6;
constexpr size_t INDEX_ATTR_X1_QUANT_DTYPE = 7;

constexpr size_t INDEX_ATTR_COMMON_QUANT_DTYPE = 8;
constexpr size_t INDEX_ATTR_TRANS_X1 = 9;
constexpr size_t INDEX_ATTR_TRANS_X2 = 10;
constexpr size_t INDEX_ATTR_GROUP_SIZE = 11;
constexpr size_t INDEX_ATTR_COMM_MODE = 12;
constexpr size_t INDEX_ATTR_ALLTOALL_OUT_FLAG = 13;
constexpr size_t INDEX_OUT = 0;
constexpr size_t INDEX_OUT_ALL2ALL_OUT = 1;
// kc量化模式
constexpr uint64_t X1_DYN_PERTOKEN_QUANT_MODE_NUM = 7;
constexpr uint64_t X2_PERCHANNEL_QUANT_MODE_NUM = 2;
// mx量化模式
constexpr uint64_t X1_MX_QUANT_MODE_NUM = 6;
constexpr uint64_t X2_MX_QUANT_MODE_NUM = 6;

const char *AlltoAllMatmulInfo = "AlltoAllMatmulFallback";

// 公共输入参数结构体
struct CommonMatmulParas {
    /* const aclTensor *会导致Release重载方法匹配不上，造成内存泄漏 */
    aclTensor *x1Acl;
    aclTensor *x2Acl;
    const gert::Tensor *bias;
};

// 量化输入参数结构体
struct QuantMatmulParas {
    aclTensor *x1ScaleAcl = nullptr;
    aclTensor *x2ScaleAcl = nullptr;
};

// Attr参数结构体
struct AttrParas {
    aclTensor *commScaleOptional = nullptr;
    aclTensor *x1OffsetOptional = nullptr;
    aclTensor *x2OffsetOptional = nullptr;
    const char *group;
    gert::TypedContinuousVector<int64_t> *alltoAllAxesOptional;
    int64_t commQuantMode;
    int64_t x1QuantDtype;
    int64_t commQuantDtype;
    bool transposeX1;
    bool transposeX2;
    int64_t groupSize = 0;
    const char *commMode;
    bool alltoAllOutFlag;
};

/**
 * @brief 获取公共Matmul输入参数
 * @param host_api_ctx
 * @param para
 */
static ge::graphStatus GetCommonMatmulInputPara(const gert::OpExecuteContext *host_api_ctx,
                                                const gert::RuntimeAttrs *attrs, CommonMatmulParas &para)
{
    const auto x1 = host_api_ctx->GetInputTensor(INDEX_IN_X1);
    OPS_CHECK(x1 == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x1"), return ge::GRAPH_FAILED);

    const auto x2 = host_api_ctx->GetInputTensor(INDEX_IN_X2);
    OPS_CHECK(x2 == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x2"), return ge::GRAPH_FAILED);

    para.bias = host_api_ctx->GetOptionalInputTensor(INDEX_IN_BIAS);

    para.x1Acl = ConvertMmType(x1, false);
    OPS_CHECK(para.x1Acl == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x1Acl"),
              return ge::GRAPH_FAILED);

    // 获取x1_quant_mode和x2_quant_mode
    const int64_t *x1QuantModePtr = attrs->GetInt(INDEX_ATTR_X1_QUANT_MODE);
    const int64_t x1QuantMode = (x1QuantModePtr != nullptr ? *x1QuantModePtr : 0);
    const int64_t *x2QuantModePtr = attrs->GetInt(INDEX_ATTR_X2_QUANT_MODE);
    const int64_t x2QuantMode = (x2QuantModePtr != nullptr ? *x2QuantModePtr : 0);

    if (x1QuantMode == X1_MX_QUANT_MODE_NUM && x2QuantMode == X2_MX_QUANT_MODE_NUM) {
        // 适配fusion pass的.t()场景，这里固定传false
        para.x2Acl = ConvertMmType(x2, false);
    } else {
        // 针对以下场景：
        // x1QuantMode == 0 && x2QuantMode == 0
        // x1QuantMode == X1_DYN_PERTOKEN_QUANT_MODE_NUM && x2QuantMode == X2_PERCHANNEL_QUANT_MODE_NUM
        const bool *transX2Ptr = attrs->GetBool(static_cast<size_t>(INDEX_ATTR_TRANS_X2));
        const bool transX2 = (transX2Ptr != nullptr ? *transX2Ptr : false);
        para.x2Acl = ConvertMmType(x2, transX2);
    }
    OPS_CHECK(para.x2Acl == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x2Acl"),
              return ge::GRAPH_FAILED);

    return ge::SUCCESS;
}

static ge::graphStatus ParseRecvCounts(const gert::TypedContinuousVector<int64_t> *sendCounts,
                                       std::vector<int64_t> &actSendCountsSeqArray)
{
    const size_t sendLen = static_cast<size_t>(sendCounts->GetSize());
    const int64_t *actSendSeqData = sendCounts->GetData();
    for (size_t i = 0UL; i < sendLen; i++) {
        actSendCountsSeqArray.push_back(actSendSeqData[i]);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取attr参数
 * @param host_api_ctx
 * @param para
 */
static ge::graphStatus GetAttrPara(const gert::OpExecuteContext *host_api_ctx, const gert::RuntimeAttrs *attrs,
                                   AttrParas &para)
{
    para.group = attrs->GetStr(INDEX_ATTR_GROUP);
    OPS_CHECK(para.group == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "group"),
              return ge::GRAPH_FAILED);
    para.commMode = attrs->GetStr(INDEX_ATTR_COMM_MODE);
    OPS_CHECK(para.commMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "commMode"),
              return ge::GRAPH_FAILED);

    const bool *transX2Ptr = attrs->GetBool(INDEX_ATTR_TRANS_X2);
    const bool transX2 = (transX2Ptr != nullptr ? *transX2Ptr : false);
    const auto commScaleOptional = host_api_ctx->GetOptionalInputTensor(INDEX_IN_COMM_SCALE);
    if (commScaleOptional != nullptr) {
        para.commScaleOptional = ConvertMmType(commScaleOptional, transX2);
        OPS_CHECK(para.commScaleOptional == nullptr,
                  OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "commScaleOptional"),
                  return ge::GRAPH_FAILED);
    }
    const auto x1OffsetOptional = host_api_ctx->GetOptionalInputTensor(INDEX_IN_X1_OFFSET);
    if (x1OffsetOptional != nullptr) {
        para.x1OffsetOptional = ConvertMmType(x1OffsetOptional, false);
        OPS_CHECK(para.x1OffsetOptional == nullptr,
                  OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x1OffsetOptional"), return ge::GRAPH_FAILED);
    }
    const auto x2OffsetOptional = host_api_ctx->GetOptionalInputTensor(INDEX_IN_X2_OFFSET);
    if (x2OffsetOptional != nullptr) {
        para.x2OffsetOptional = ConvertMmType(x2OffsetOptional, false);
        OPS_CHECK(para.x2OffsetOptional == nullptr,
                  OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x2OffsetOptional"), return ge::GRAPH_FAILED);
    }
    const int64_t *commQuantModePtr = attrs->GetInt(INDEX_ATTR_COMMON_QUANT_MODE);
    para.commQuantMode = (commQuantModePtr != nullptr ? *commQuantModePtr : 0);
    const int64_t *x1QuantDtypePtr = attrs->GetInt(INDEX_ATTR_X1_QUANT_DTYPE);
    para.x1QuantDtype =
        (x1QuantDtypePtr != nullptr ? *x1QuantDtypePtr : static_cast<uint64_t>(ge::DataType::DT_UNDEFINED));
    const int64_t *commQuantDtypePtr = attrs->GetInt(INDEX_ATTR_COMMON_QUANT_DTYPE);
    para.commQuantDtype =
        (commQuantDtypePtr != nullptr ? *commQuantDtypePtr : static_cast<uint64_t>(ge::DataType::DT_UNDEFINED));
    const bool *transX1Ptr = attrs->GetBool(INDEX_ATTR_TRANS_X1);
    para.transposeX1 = (transX1Ptr != nullptr ? *transX1Ptr : false);
    para.transposeX2 = (transX2Ptr != nullptr ? *transX2Ptr : false);
    const int64_t *groupSizePtr = attrs->GetInt(INDEX_ATTR_GROUP_SIZE);
    para.groupSize = (groupSizePtr != nullptr ? *groupSizePtr : 0);

    const bool *allToAllOutFlagPtr = attrs->GetBool(INDEX_ATTR_ALLTOALL_OUT_FLAG);
    para.alltoAllOutFlag = (allToAllOutFlagPtr != nullptr ? *allToAllOutFlagPtr : true);

    return ge::SUCCESS;
}

/**
 * @brief 获取量化Matmul输入参数
 * @param host_api_ctx
 * @param para
 */
static ge::graphStatus GetQuantMatmulPara(const gert::OpExecuteContext *host_api_ctx, const gert::RuntimeAttrs *attrs,
                                          QuantMatmulParas &para)
{
    const auto x1Scale = host_api_ctx->GetOptionalInputTensor(INDEX_IN_X1_SCALE);
    if (x1Scale != nullptr) {
        para.x1ScaleAcl = ConvertMmType(x1Scale, false);
        OPS_CHECK(para.x1ScaleAcl == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x1ScaleAcl"),
                  return ge::GRAPH_FAILED);
    }

    const auto x2Scale = host_api_ctx->GetOptionalInputTensor(INDEX_IN_X2_SCALE);
    OPS_CHECK(x2Scale == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x2Scale"),
              return ge::GRAPH_FAILED);

    // 获取x1_quant_mode和x2_quant_mode
    const int64_t *x2QuantModePtr = attrs->GetInt(INDEX_ATTR_X2_QUANT_MODE);
    const int64_t x2QuantMode = (x2QuantModePtr != nullptr ? *x2QuantModePtr : 0);
    const int64_t *x1QuantModePtr = attrs->GetInt(INDEX_ATTR_X1_QUANT_MODE);
    const int64_t x1QuantMode = (x1QuantModePtr != nullptr ? *x1QuantModePtr : 0);

    if (x1QuantMode == X1_MX_QUANT_MODE_NUM && x2QuantMode == X2_MX_QUANT_MODE_NUM) {
        // 适配fusion pass的.t()场景，这里固定传false
        para.x2ScaleAcl = ConvertMmType(x2Scale, false);
    } else {
        // 针对以下场景：
        // x1QuantMode == 0 && x2QuantMode == 0
        // x1QuantMode == X1_DYN_PERTOKEN_QUANT_MODE_NUM && x2QuantMode == X2_PERCHANNEL_QUANT_MODE_NUM
        const bool *transX2Ptr = attrs->GetBool(INDEX_ATTR_TRANS_X2);
        const bool transX2 = (transX2Ptr != nullptr ? *transX2Ptr : false);
        para.x2ScaleAcl = ConvertMmType(x2Scale, transX2);
    }
    OPS_CHECK(para.x2ScaleAcl == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "x2ScaleAcl"),
              return ge::GRAPH_FAILED);

    return ge::SUCCESS;
}

/**
 * @brief 校验AlltoAllMatmul执行函数
 * @param host_api_ctx
 */
static ge::graphStatus AlltoAllMatmulExecuteFunc(gert::OpExecuteContext *host_api_ctx)
{
    OPS_LOG_D(AlltoAllMatmulInfo, "Start to fallback for allto_all_matmul.");
    OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E(AlltoAllMatmulInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);
    const gert::RuntimeAttrs *attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "attrs"),
              return ge::GRAPH_FAILED);

    CommonMatmulParas matmulParas;
    ge::graphStatus retPara = GetCommonMatmulInputPara(host_api_ctx, attrs, matmulParas);
    OPS_CHECK(retPara != ge::SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get common matmul input paras."),
              return ge::GRAPH_FAILED);

    AttrParas attrParas;
    OPS_CHECK(GetAttrPara(host_api_ctx, attrs, attrParas) != ge::SUCCESS,
              OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get attr paras."), return ge::GRAPH_FAILED);

    const auto output = host_api_ctx->GetOutputTensor(INDEX_OUT);
    OPS_CHECK(output == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "output"),
              return ge::GRAPH_FAILED);

    const auto alltoAllOut = host_api_ctx->GetOutputTensor(INDEX_OUT_ALL2ALL_OUT);
    if (attrParas.alltoAllOutFlag) {
        OPS_CHECK(alltoAllOut == nullptr, OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "alltoAllOut"),
                  return ge::GRAPH_FAILED);
    }

    const auto alltoAllAxesOptional = attrs->GetListInt(INDEX_ATTR_ALL2ALL_AXES);
    std::vector<int64_t> actSeqArray;
    if (alltoAllAxesOptional != nullptr) {
        OPS_CHECK(alltoAllAxesOptional == nullptr,
                  OP_LOGE_WITH_INVALID_INPUT(host_api_ctx->GetNodeName(), "alltoAllAxesOptional"),
                  return ge::GRAPH_FAILED);
        ParseRecvCounts(alltoAllAxesOptional, actSeqArray);
    }

    const int64_t *x1QuantModePtr = attrs->GetInt(INDEX_ATTR_X1_QUANT_MODE);
    const int64_t x1QuantMode = (x1QuantModePtr != nullptr ? *x1QuantModePtr : 0);
    const int64_t *x2QuantModePtr = attrs->GetInt(INDEX_ATTR_X2_QUANT_MODE);
    const int64_t x2QuantMode = (x2QuantModePtr != nullptr ? *x2QuantModePtr : 0);
    if (x1QuantMode == 0 && x2QuantMode == 0) {
        const auto ret = EXEC_OPAPI_CMD(aclnnAlltoAllMatmulV2, matmulParas.x1Acl, matmulParas.x2Acl, matmulParas.bias,
                                        actSeqArray, attrParas.group, attrParas.commMode, attrParas.transposeX1,
                                        attrParas.transposeX2, output, alltoAllOut);
        OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
                   OPS_LOG_E(AlltoAllMatmulInfo, "Aclnn allto all matmul api error code %d", ret),
                   return ge::GRAPH_FAILED);
    } else if ((x1QuantMode == X1_DYN_PERTOKEN_QUANT_MODE_NUM && x2QuantMode == X2_PERCHANNEL_QUANT_MODE_NUM) ||
               (x1QuantMode == X1_MX_QUANT_MODE_NUM && x2QuantMode == X2_MX_QUANT_MODE_NUM)) {
        QuantMatmulParas quantMatmulParas;
        retPara = GetQuantMatmulPara(host_api_ctx, attrs, quantMatmulParas);
        const auto ret = EXEC_OPAPI_CMD(
            aclnnAlltoAllQuantMatmulV2, matmulParas.x1Acl, matmulParas.x2Acl, matmulParas.bias,
            quantMatmulParas.x1ScaleAcl, quantMatmulParas.x2ScaleAcl, attrParas.commScaleOptional,
            attrParas.x1OffsetOptional, attrParas.x2OffsetOptional, attrParas.group, attrParas.commMode, actSeqArray,
            x1QuantMode, x2QuantMode, attrParas.commQuantMode, attrParas.commQuantDtype, attrParas.x1QuantDtype,
            attrParas.groupSize, attrParas.transposeX1, attrParas.transposeX2, output, alltoAllOut);
        OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
                   OPS_LOG_E(AlltoAllMatmulInfo, "Aclnn allto all quant matmul api error code %d", ret),
                   return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AlltoAllMatmul).OpExecuteFunc(AlltoAllMatmulExecuteFunc);
} // namespace fallback
