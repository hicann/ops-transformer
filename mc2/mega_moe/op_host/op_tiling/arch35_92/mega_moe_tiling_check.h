/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TILING_CHECK_H
#define MEGA_MOE_TILING_CHECK_H

#include "mega_moe_dedup_gate.h"
#include "mega_moe_tiling_buffer_calc.h"
namespace optiling {
namespace {

/*
 * 集中校验后续阶段使用的必需属性指针，后续属性处理函数不再重复判空。
 * 涉及的直接属性：moe_expert_num、ep_world_size、ccl_buffer_size、max_recv_token_num、
 * dispatch_quant_mode、dispatch_quant_out_dtype、shared_expert_quant_out_dtype、combine_quant_mode、
 * comm_alg、num_max_tokens_per_rank、topo_type、topk_weights_type、activation 和 activation_params。
 */
static ge::graphStatus CheckRequiredAttrPtrNullptr(const gert::TilingContext* context, const MegaMoeConfig& config)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    auto sharedExpertQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrSharedExpertQuantOutDtypeIndex));
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>((config.attrTopoTypeIndex));
    auto topkWeightsTypePtr = attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);
    auto rankNumPerServerPtr = attrs->GetAttrPointer<int64_t>(config.attrRankNumPerServerIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, moeExpertNumPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, epWorldSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, cclBufferSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxRecvTokenNumPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dispatchQuantModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dispatchQuantOutDtypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, sharedExpertQuantOutDtypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, combineQuantModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, commAlgPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, numMaxTokensPerRankPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, topoTypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsTypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationParamsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, rankNumPerServerPtr);
    const size_t paramCount = activationParamsPtr->GetSize();
    const float* activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();
    if (paramCount != 0U) {
        OP_CHECK_NULL_WITH_CONTEXT(context, activationParams);
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 swiglu/swiglustep 的可选 clamp 参数；该参数表示激活值截断上限。
 * 未配置时使用 float 最大值，等价于不截断（参见 SetActivationAttrParams）。
 */
static ge::graphStatus CheckSwiGluActivationParams(const float* activationParams, size_t paramCount,
                                                   const char* nodeName)
{
    OP_TILING_CHECK(paramCount > 1U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(),
                                              "0 or 1 for swiglu/swiglustep"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 1U) {
        const float activationClamp = activationParams[0];
        OP_TILING_CHECK(activationClamp < 0.0f || std::isnan(activationClamp),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationClamp", std::to_string(activationClamp).c_str(),
                                                  "should be >= 0 and not NAN"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 swigluoai 的 clamp、alpha 和 beta 参数。
 * 三个参数必须同时配置；全部省略时统一使用默认值，不支持部分配置。
 */
static ge::graphStatus CheckSwiGluOaiActivationParams(const float* activationParams, size_t paramCount,
                                                      const char* nodeName)
{
    OP_TILING_CHECK(paramCount != 0U && paramCount != 3U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(),
                                              "0 or 3 for swigluoai [clamp, alpha, beta]"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 3U) {
        const float activationClamp = activationParams[0];
        OP_TILING_CHECK(activationClamp < 0.0f || std::isnan(activationClamp),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationClamp", std::to_string(activationClamp).c_str(),
                                                  ">= 0 and not NAN"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!std::isfinite(activationParams[1]),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationAlpha",
                                                  std::to_string(activationParams[1]).c_str(), "finite"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!std::isfinite(activationParams[2]),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationBeta",
                                                  std::to_string(activationParams[2]).c_str(), "finite"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * situglu 的 beta 必填：A5 仅支持大于 0 的有限值。
 * linear_beta 选填，给了就切到 LINEAR 子模式，同样仅支持大于 0 的有限值。
 * 注意个数校验必须排在取值之前：一个参数都不传时 activationParams 是空指针，
 * 先取 activationParams[0] 会直接段错误，而不是干净地报错返回。
 */
static ge::graphStatus CheckSituGluActivationParams(const float* activationParams, size_t paramCount,
                                                    const char* nodeName)
{
    OP_TILING_CHECK(paramCount < 1U || paramCount > 2U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(),
                                              "should be 1 or 2 for situglu"),
                    return ge::GRAPH_FAILED);
    const float beta = activationParams[0];
    OP_TILING_CHECK(!std::isfinite(beta) || beta <= 0.0f,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_beta", std::to_string(beta).c_str(),
                                              "should be finite and greater than 0"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 2U) {
        const float linearBeta = activationParams[1];
        OP_TILING_CHECK(!std::isfinite(linearBeta) || linearBeta <= 0.0f,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_linear_beta", std::to_string(linearBeta).c_str(),
                                                  "should be finite and greater than 0"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * GELU 仅接受一个 float 最大值作为占位参数，不用于截断计算。
 */
static ge::graphStatus CheckGeluActivationParams(const float* activationParams, size_t paramCount, const char* nodeName)
{
    OP_TILING_CHECK(
        paramCount != 1U,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(), "1 for gelu"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        activationParams[0] != DEFAULT_ACTIVATION_CLAMP,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParams[0]", std::to_string(activationParams[0]).c_str(),
                                  "float maximum placeholder for gelu"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验激活类型、参数个数与取值，并检查激活类型和通信拓扑的兼容性。
 */
static ge::graphStatus CheckActivationParams(const gert::TilingContext* context, const MegaMoeConfig& config,
                                             int64_t topoType, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);
    const size_t paramCount = activationParamsPtr->GetSize();
    const float* activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();

    const bool isSwiGlu = std::strcmp(activationPtr, "swiglu") == 0;
    const bool isSwiGluStep = std::strcmp(activationPtr, "swiglustep") == 0;
    const bool isSwiGluOai = std::strcmp(activationPtr, "swigluoai") == 0;
    const bool isSituGlu = std::strcmp(activationPtr, "situglu") == 0;
    const bool isGelu = std::strcmp(activationPtr, "gelu") == 0;
    OP_TILING_CHECK(!isSwiGlu && !isSwiGluStep && !isSwiGluOai && !isSituGlu && !isGelu,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr,
                                              "one of 'swiglu', 'swiglustep', 'swigluoai', 'situglu' or 'gelu'"),
                    return ge::GRAPH_FAILED);

    // URMA Layered 支持 swiglu、situglu 和逐元素 GELU。
    OP_TILING_CHECK(IsLayeredTopology(topoType) && !isSwiGlu && !isSituGlu && !isGelu,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr,
                                              "'swiglu', 'situglu' or 'gelu' for URMA topology"),
                    return ge::GRAPH_FAILED);

    if (isSwiGlu || isSwiGluStep) {
        return CheckSwiGluActivationParams(activationParams, paramCount, nodeName);
    }
    if (isSwiGluOai) {
        return CheckSwiGluOaiActivationParams(activationParams, paramCount, nodeName);
    }
    if (isGelu) {
        return CheckGeluActivationParams(activationParams, paramCount, nodeName);
    }
    return CheckSituGluActivationParams(activationParams, paramCount, nodeName);
}

/*
 * 校验 EP 通信域规模并写入 tiling data，供专家划分与容量计算使用。
 */
static ge::graphStatus CheckAndSetEpWorldSizeAttr(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                  MegaMoeTilingData* tilingData, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    int64_t epWorldSize = static_cast<int64_t>(*epWorldSizePtr);
    int64_t maxEpWorldSize =
        IsLayeredTopology(tilingData->compute.topoType) ? MAX_URMA_EP_WORLD_SIZE : MAX_MTE_EP_WORLD_SIZE;
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > maxEpWorldSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EP_WORLD_SIZE) + ", " +
                                               std::to_string(maxEpWorldSize) + "] for the selected topology")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    tilingData->compute.epWorldSize = static_cast<uint32_t>(epWorldSize);
    int64_t rankNumPerServer = *attrs->GetAttrPointer<int64_t>(config.attrRankNumPerServerIndex);
    OP_TILING_CHECK(
        IsLayeredTopology(tilingData->compute.topoType) &&
            (rankNumPerServer <= 0 || rankNumPerServer > epWorldSize || epWorldSize % rankNumPerServer != 0),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "rankNumPerServer", std::to_string(rankNumPerServer).c_str(),
                                  "should be in [1, epWorldSize] and divide epWorldSize for URMA"),
        return ge::GRAPH_FAILED);
    tilingData->rankNumPerServer = IsLayeredTopology(tilingData->compute.topoType) ?
                                       static_cast<uint32_t>(rankNumPerServer) :
                                       static_cast<uint32_t>(epWorldSize);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验去重请求属性：combine_comm_mode 只收 0/1，完全由调用方决定（1=强制开启 combine 去重、
 * 0=强制关闭，与内部门控无关）。ccm=1 显式请求撞上能力硬约束（非 MTE 拓扑、量化 combine、
 * 预量化 x 配 tkw=0）直接报错不静默降级；自动请求的 dispatch 位不在此拦——非 MTE 时由
 * GetDedupMode 关位回落。校验通过后落库 tilingData->compute.dedupMode，后续各阶段一律读 tilingData。
 */
static ge::graphStatus CheckDedupRequestAttrs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                              MegaMoeTilingData* tilingData, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    auto ccmPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCombineCommModeIndex));
    int64_t combineCommMode = ccmPtr == nullptr ? COMBINE_COMM_MODE_NO_DEDUP : *ccmPtr;
    OP_TILING_CHECK(
        combineCommMode != COMBINE_COMM_MODE_NO_DEDUP && combineCommMode != COMBINE_COMM_MODE_RANK_DEDUP,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "combine_comm_mode", std::to_string(combineCommMode).c_str(), "0 or 1"),
        return ge::GRAPH_FAILED);
    // ccm 的取值已在上一条拦截，故 GetDedupMode 在此不可能返回无效值。
    int64_t dedupMode = GetDedupMode(context, config);
    int64_t topoType = *attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    // ccm=1 显式请求 + 非 MTE = 报错（自动请求的 dispatch 位已由 GetDedupMode 关位回落）。
    OP_TILING_CHECK(combineCommMode == COMBINE_COMM_MODE_RANK_DEDUP && topoType != TOPO_TYPE_MTE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "combine_comm_mode", "1",
                        "enabling combine dedup requires topo_type=0 (MTE); set combine_comm_mode=0 on other "
                        "topologies"),
                    return ge::GRAPH_FAILED);
    // combine 去重当前只实现非量化 combine 路径；量化 combine 直接报错，不做静默降级。
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>(config.attrCombineQuantModeIndex);
    int64_t combineQuantMode = combineQuantModePtr == nullptr ? COMBINE_QUANT_OUT_TYPE_NO_QUANT : *combineQuantModePtr;
    OP_TILING_CHECK(
        IsCombineDedupOn(dedupMode) && combineQuantMode != COMBINE_QUANT_OUT_TYPE_NO_QUANT,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "combine_comm_mode", "1",
                                              "enabling combine dedup does not support quantized combine; it requires "
                                              "combine_quant_mode=0"),
        return ge::GRAPH_FAILED);
    // 预量化 x 的打包路径（PackPreQuantizedTokenRange）只在 tkw=1 时随记录下发 topk 权重；tkw=0 的
    // combine 去重要靠记录尾段的权重在发送端乘权，该路径尚未接入——放行会得到权重全 0 的静默错误结果。
    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    const bool isPreQuantizedX = IsPreQuantizedXType(context->GetInputDesc(config.xIndex)->GetDataType());
    OP_TILING_CHECK(IsCombineDedupOn(dedupMode) && topkWeightsType == 0 && isPreQuantizedX,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "combine_comm_mode", "1",
                        "enabling combine dedup does not support pre-quantized x together with topk_weights_type=0; "
                        "set topk_weights_type=1, or pass a non-pre-quantized x"),
                    return ge::GRAPH_FAILED);
    // 去重请求位落库。combine 去重两种权重语义并存：tkw=1 沿用 prefetch（激活层乘权、合并与
    // unpermute 纯加）；tkw=0 时合并阶段在发送端按成员权重乘加，权重经记录尾段下发（布局判定
    // 统一按「tkw=1 或 combine 去重开」判定，prefetch 位保持 attr 原值不强拉——强拉会偏离 tkw=0
    // golden 的乘权位置语义）。
    tilingData->compute.dedupMode = static_cast<int32_t>(dedupMode);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验路由专家总数、本 rank 专家数和共享专家数。
 * 路由专家必须在 EP rank 间均分，且派生的本 rank 专家数必须与 weight1 中的专家数一致。
 * 共享专家不参与 EP 路由切分，其数量由 shared weight1 推导，本函数仅校验实现上限。
 */
static ge::graphStatus CheckAndSetExpertCountAttrs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                   const MegaMoeExpertParams& expertParams,
                                                   MegaMoeTilingData* tilingData, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t epWorldSize = static_cast<int64_t>(tilingData->compute.epWorldSize);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    OP_TILING_CHECK((moeExpertNum < epWorldSize || moeExpertNum > MAX_MOE_EXPERT_NUM) || (moeExpertNum % epWorldSize),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(moeExpertNum).c_str(),
                                              (std::string("should in [") + std::to_string(epWorldSize) + ", " +
                                               std::to_string(MAX_MOE_EXPERT_NUM) + "] and mod(..., epWorldSize(" +
                                               std::to_string(epWorldSize) + ")) == 0")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    int64_t moeExpertPerRank = moeExpertNum / epWorldSize;
    OP_TILING_CHECK(moeExpertPerRank < MIN_EXPERT_PER_RANK || moeExpertPerRank > MAX_EXPERT_PER_RANK,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertPerRank", std::to_string(moeExpertPerRank).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EXPERT_PER_RANK) + ", " +
                                               std::to_string(MAX_EXPERT_PER_RANK) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertParams.moe.expertCount != moeExpertPerRank,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "weight1 expert count", std::to_string(expertParams.moe.expertCount).c_str(),
                        (std::string("should equal the local MoE expert count (moeExpertNum / epWorldSize) = ") +
                         std::to_string(moeExpertPerRank))
                            .c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expertParams.shared.expertCount > MAX_SHARED_EXPERT_NUM,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(expertParams.shared.expertCount).c_str(),
                                  "only support 0-4"),
        return ge::GRAPH_FAILED);

    tilingData->compute.moeExpertPerRank = static_cast<uint32_t>(moeExpertPerRank);
    tilingData->compute.sharedExpertNum = static_cast<uint32_t>(expertParams.shared.expertCount);
    return ge::GRAPH_SUCCESS;
}

/*
 * 判断量化输出类型能否映射到当前 kernel 支持的专家量化模式。
 */
static bool IsSupportedExpertQuantDtype(ge::DataType dtype)
{
    return EXPERT_QUANT_MODE_MAP.count(dtype) > 0U;
}

/*
 * 由单个专家组的量化输出类型、权重类型和两个权重的 format 确定 GMM 实现路径。
 * A4W4 ND 是内部保留能力：weight1 为 ND，weight2 仍为 NZ_C0_32；算子原型不因此扩展支持范围。
 */
static uint8_t ResolveExpertGmmMode(ge::DataType quantOutDtype, ge::DataType weightDtype, ge::Format weightOneFormat,
                                    ge::Format weightTwoFormat)
{
    bool isNdFormat = weightOneFormat == ge::FORMAT_ND && weightTwoFormat == ge::FORMAT_ND;
    bool isNzFormat = weightOneFormat == ge::FORMAT_FRACTAL_NZ && weightTwoFormat == ge::FORMAT_FRACTAL_NZ;
    bool isNzC032Format =
        weightOneFormat == ge::FORMAT_FRACTAL_NZ_C0_32 && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;
    bool isNdAndNzC032Format = weightOneFormat == ge::FORMAT_ND && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;
    bool isNzAndNzC032Format =
        weightOneFormat == ge::FORMAT_FRACTAL_NZ && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;

    if (quantOutDtype == ge::DT_FLOAT8_E5M2 && weightDtype == ge::DT_FLOAT8_E5M2 && isNdFormat) {
        return GMM_MODE_A8W8_ND;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT8_E4M3FN && isNdFormat) {
        return GMM_MODE_A8W8_ND;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT8_E4M3FN && isNzFormat) {
        return GMM_MODE_A8W8_NZ;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT4_E2M1 && isNzC032Format) {
        return GMM_MODE_A8W4_NZ;
    }
    if (quantOutDtype == ge::DT_FLOAT4_E2M1 && weightDtype == ge::DT_FLOAT4_E2M1 && isNdAndNzC032Format) {
        return GMM_MODE_A4W4_ND;
    }
    if (quantOutDtype == ge::DT_FLOAT4_E2M1 && weightDtype == ge::DT_FLOAT4_E2M1 && isNzAndNzC032Format) {
        return GMM_MODE_A4W4_NZ;
    }
    return std::numeric_limits<uint8_t>::max();
}

/*
 * 校验 dispatch 和 combine 阶段的量化模式，并将 combine 模式写入 tiling data。
 * dispatch_quant_mode 为 0 表示不在算子内部量化，为 4 表示算子内部执行 MX 量化。
 */
static ge::graphStatus CheckAndSetQuantModeAttrs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                 MegaMoeTilingData* tilingData, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    OP_TILING_CHECK(
        *dispatchQuantModePtr != DISPATCH_QUANT_MODE_PASSTHROUGH && *dispatchQuantModePtr != DISPATCH_QUANT_MODE_MXFP,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "dispatch_quant_mode", std::to_string(*dispatchQuantModePtr).c_str(),
                                  "0 (no internal quantization) or 4 (internal MXFP quantization)"),
        return ge::GRAPH_FAILED);

    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    OP_TILING_CHECK(
        *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_NO_QUANT &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E5M2 &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E4M3FN,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "combineQuantMode", std::to_string(*combineQuantModePtr).c_str(),
                                  "only support no_quant(0), fp8_e5m2(3) and fp8_e4m3fn(4)"),
        return ge::GRAPH_FAILED);
    tilingData->combineQuantMode = static_cast<uint32_t>(*combineQuantModePtr);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验预量化输入的逻辑类型、通信拓扑及共享专家类型约束。
 */
static ge::graphStatus CheckPreQuantizedXAttrs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                               ge::DataType moeQuantOutDtype, ge::DataType sharedQuantOutDtype,
                                               int64_t topoType, const char* nodeName)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    // 预量化输入不再由 x dtype 反推类型，属性与接口补充的逻辑 descriptor 必须严格一致。
    OP_TILING_CHECK(
        xDesc->GetDataType() != moeQuantOutDtype,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "dispatch_quant_out_dtype", std::to_string(moeQuantOutDtype).c_str(),
                                  (std::string("the same dtype as x (") + Ops::Base::ToString(xDesc->GetDataType()) +
                                   ") for pre-quantized input")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topoType != TOPO_TYPE_MTE,
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "topo_type", std::to_string(topoType).c_str(),
                                              "0 (MTE), because pre-quantized x does not support URMA"),
                    return ge::GRAPH_FAILED);

    // 预量化 x 只有一份 data/scales；共享专家显式配置类型时只能与 MoE 专家保持一致。
    const bool hasExplicitSharedQuantDtype = sharedQuantOutDtype != ge::DT_UNDEFINED;
    OP_TILING_CHECK(hasExplicitSharedQuantDtype && sharedQuantOutDtype != moeQuantOutDtype,
                    OP_LOGE_WITH_INVALID_ATTR(
                        nodeName, "shared_expert_quant_out_dtype", std::to_string(sharedQuantOutDtype).c_str(),
                        (std::string("DT_UNDEFINED or the same dtype as dispatch_quant_out_dtype (") +
                         std::to_string(moeQuantOutDtype) + ") for pre-quantized x")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并设置 MoE 与共享专家使用的量化类型。mode 4 下该属性表示内部量化输出类型，
 * mode 0 且类型为 FP8/FP4 时，该属性表示预量化 x 的逻辑类型；同时解析共享专家类型的继承语义。
 */
static ge::graphStatus CheckAndSetQuantOutDtypes(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                 MegaMoeExpertParams& params, int64_t topoType, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    ge::DataType moeQuantOutDtype = static_cast<ge::DataType>(*dispatchQuantOutDtypePtr);

    // 两种 dispatch 模式均由 dispatch_quant_out_dtype 指定 GMM 收到的量化数据类型。
    OP_TILING_CHECK(!IsSupportedExpertQuantDtype(moeQuantOutDtype),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "dispatch_quant_out_dtype",
                                              std::to_string(*dispatchQuantOutDtypePtr).c_str(),
                                              "fp8_e5m2, fp8_e4m3fn or fp4_e2m1"),
                    return ge::GRAPH_FAILED);

    auto sharedQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertQuantOutDtypeIndex);
    ge::DataType sharedQuantOutDtype = static_cast<ge::DataType>(*sharedQuantOutDtypePtr);
    bool hasExplicitSharedQuantDtype = sharedQuantOutDtype != ge::DT_UNDEFINED;
    OP_TILING_CHECK(hasExplicitSharedQuantDtype && !IsSupportedExpertQuantDtype(sharedQuantOutDtype),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "shared_expert_quant_out_dtype",
                                              std::to_string(*sharedQuantOutDtypePtr).c_str(),
                                              "DT_UNDEFINED, fp8_e5m2, fp8_e4m3fn or fp4_e2m1"),
                    return ge::GRAPH_FAILED);

    const bool isPreQuantizedX =
        *dispatchQuantModePtr == DISPATCH_QUANT_MODE_PASSTHROUGH && IsSupportedExpertQuantDtype(moeQuantOutDtype);
    if (isPreQuantizedX) {
        OP_TILING_CHECK(CheckPreQuantizedXAttrs(context, config, moeQuantOutDtype, sharedQuantOutDtype, topoType,
                                                nodeName) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "pre-quantized x attributes are invalid."), return ge::GRAPH_FAILED);
    } else {
        auto xDesc = context->GetInputDesc(config.xIndex);
        OP_TILING_CHECK(
            xDesc->GetDataType() != ge::DT_BF16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                                  "When dispatch_quant_mode is 4, the dtype of x must be DT_BF16"),
            return ge::GRAPH_FAILED);
    }

    params.moe.quantOutDtype = moeQuantOutDtype;
    params.shared.quantOutDtype =
        params.shared.expertCount <= 0 || !hasExplicitSharedQuantDtype ? moeQuantOutDtype : sharedQuantOutDtype;
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验同一专家组的权重类型一致性，并由量化类型、权重类型及格式解析 GMM 模式。
 */
static ge::graphStatus CheckAndSetExpertGmmMode(const gert::TilingContext* context, ExpertParams& expertParams,
                                                const char* nodeName)
{
    auto weightOneDesc = context->GetDynamicInputDesc(expertParams.inputs.weightOne, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(expertParams.inputs.weightTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);

    ge::DataType weightOneDtype = weightOneDesc->GetDataType();
    ge::DataType weightTwoDtype = weightTwoDesc->GetDataType();
    std::string weightNames = std::string(expertParams.expertTypeName) + " weight1, weight2";
    std::string weightDtypes =
        "[" + Ops::Base::ToString(weightOneDtype) + ", " + Ops::Base::ToString(weightTwoDtype) + "]";
    OP_TILING_CHECK(weightOneDtype != weightTwoDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        nodeName, weightNames.c_str(), weightDtypes.c_str(),
                        "weight1 and weight2 within the same expert group must have the same dtype."),
                    return ge::GRAPH_FAILED);

    ge::Format weightOneFormat = weightOneDesc->GetStorageFormat();
    ge::Format weightTwoFormat = weightTwoDesc->GetStorageFormat();
    expertParams.gmmMode =
        ResolveExpertGmmMode(expertParams.quantOutDtype, weightOneDtype, weightOneFormat, weightTwoFormat);
    std::string actualConfig = "[" + Ops::Base::ToString(expertParams.quantOutDtype) + ", " +
                               Ops::Base::ToString(weightOneDtype) + ", " + Ops::Base::ToString(weightOneFormat) +
                               ", " + Ops::Base::ToString(weightTwoFormat) + "]";
    OP_TILING_CHECK(
        expertParams.gmmMode == std::numeric_limits<uint8_t>::max(),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            nodeName, "quant output dtype, weight dtype, weight1 format, weight2 format", actualConfig.c_str(),
            (std::string(expertParams.expertTypeName) + " configuration does not form a supported GMM mode.").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 解析 MoE 与共享专家的 GMM mode，并校验 mode 组合与通信拓扑的兼容性。
 */
static ge::graphStatus CheckAndSetExpertGmmModes(const gert::TilingContext* context, MegaMoeExpertParams& params,
                                                 int64_t topoType, MegaMoeTilingData* tilingData, const char* nodeName)
{
    if (CheckAndSetExpertGmmMode(context, params.moe, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    params.shared.gmmMode = params.moe.gmmMode;
    if (params.shared.expertCount > 0 &&
        CheckAndSetExpertGmmMode(context, params.shared, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool hasDifferentExpertGmmConfigs =
        params.shared.gmmMode != params.moe.gmmMode || params.shared.quantOutDtype != params.moe.quantOutDtype;
    OP_TILING_CHECK(
        topoType != TOPO_TYPE_MTE && hasDifferentExpertGmmConfigs,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "topo_type", std::to_string(topoType).c_str(),
                                  "MTE when MoE and shared experts use different quantization configurations"),
        return ge::GRAPH_FAILED);

    tilingData->moeGmmMode = params.moe.gmmMode;
    tilingData->sharedGmmMode = params.shared.gmmMode;
    tilingData->compute.sharedInputIndependent =
        params.shared.expertCount > 0 && params.shared.quantOutDtype != params.moe.quantOutDtype;
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并默认化单 rank 最大 token 数，将结果写入 tiling data。
 */
static ge::graphStatus CheckAndSetMaxTokensPerRankAttr(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                       MegaMoeTilingData* tilingData, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    int64_t numMaxTokensPerRank = static_cast<int64_t>(*numMaxTokensPerRankPtr);
    OP_TILING_CHECK(IsLayeredTopology(tilingData->compute.topoType) && tilingData->compute.bs == 0U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "bs", "0", "should be greater than 0 for URMA"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(IsLayeredTopology(tilingData->compute.topoType) && numMaxTokensPerRank == 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "numMaxTokensPerRank", "0",
                                              "should be >= bs and identical on all ranks for URMA"),
                    return ge::GRAPH_FAILED);
    // 下游 token 索引仍使用 uint32，因此保留 UINT32_MAX 上界。
    OP_TILING_CHECK(
        numMaxTokensPerRank < 0 || numMaxTokensPerRank > 0xFFFFFFFFLL ||
            (numMaxTokensPerRank != 0 && tilingData->compute.bs > numMaxTokensPerRank),
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "numMaxTokensPerRank", std::to_string(numMaxTokensPerRank).c_str(),
            (std::string("0 or in [bs, UINT32_MAX], bs is ") + std::to_string(tilingData->compute.bs).c_str())),
        return ge::GRAPH_FAILED);
    if (numMaxTokensPerRank == 0) {
        numMaxTokensPerRank = tilingData->compute.bs;
    }
    int64_t routeCapacity = numMaxTokensPerRank * static_cast<int64_t>(tilingData->compute.topK);
    int64_t maxOutputCapacity = numMaxTokensPerRank * static_cast<int64_t>(tilingData->compute.epWorldSize) *
                                std::min(tilingData->compute.topK, tilingData->compute.moeExpertPerRank);
    OP_TILING_CHECK(
        IsLayeredTopology(tilingData->compute.topoType) && (routeCapacity > std::numeric_limits<int32_t>::max() ||
                                                            maxOutputCapacity > std::numeric_limits<int32_t>::max()),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "URMA token capacity", std::to_string(maxOutputCapacity).c_str(),
                                  "route indexes and prefix sums should fit in INT32_MAX"),
        return ge::GRAPH_FAILED);
    tilingData->compute.numMaxTokensPerRank = static_cast<uint32_t>(numMaxTokensPerRank);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 topK weight 预取开关并写入 tiling data。
 */
static ge::graphStatus CheckAndSetTopkWeightsTypeAttr(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                      MegaMoeTilingData* tilingData, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    OP_TILING_CHECK(topkWeightsType != 0 && topkWeightsType != 1,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topkWeightsType", std::to_string(topkWeightsType).c_str(),
                                              "only support 0(disabled) or 1(enabled)"),
                    return ge::GRAPH_FAILED);

    tilingData->topkWeightsPrefetch = static_cast<int32_t>(topkWeightsType);
    return ge::GRAPH_SUCCESS;
}

/*
 * 按已解析的 token 上界、张量维度及量化配置校验 CCL 容量。
 */
template <typename WorkspaceLayoutType>
static ge::graphStatus CheckCclBufferCapacity(const gert::TilingContext* context, const MegaMoeConfig& config,
                                              const MegaMoeTilingData* tilingData,
                                              const WorkspaceLayoutType& workspaceLayout, const char* nodeName,
                                              const CclCommLayoutParams* commOverride = nullptr)
{
    auto attrs = context->GetAttrs();
    const auto& comm = commOverride == nullptr ? tilingData->comm : *commOverride;
    const int64_t cclComputeOffset = CalcCclComputeOffset(comm);
    int64_t cclBufferSize = static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex)));
    OP_TILING_CHECK(workspaceLayout.workspaceSize <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclComputeSize",
                                              std::to_string(workspaceLayout.workspaceSize).c_str(), "positive"),
                    return ge::GRAPH_FAILED);
    int64_t requiredCclBufferSize = cclComputeOffset + workspaceLayout.workspaceSize;
    OP_TILING_CHECK(
        cclBufferSize < requiredCclBufferSize,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclBufferSize", std::to_string(cclBufferSize).c_str(),
                                  (std::string("should >= ") + std::to_string(requiredCclBufferSize)).c_str()),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize=%ld, cclComputeOffset=%ld, cclComputeSize=%ld", cclBufferSize, cclComputeOffset,
            workspaceLayout.workspaceSize);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验接收 token 上限，并据此确定 kernel 的最大输出 token 数。
 */
static ge::graphStatus CheckAndSetMaxRecvTokenNumAttr(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                      MegaMoeTilingData* tilingData, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t maxRecvTokenNum = static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex)));
    OP_TILING_CHECK(maxRecvTokenNum < 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(),
                                              "should be greater than or equal to 0"),
                    return ge::GRAPH_FAILED);

    uint64_t defaultMaxOutputSize = static_cast<uint64_t>(tilingData->compute.numMaxTokensPerRank) *
                                    tilingData->compute.epWorldSize *
                                    std::min(tilingData->compute.topK, tilingData->compute.moeExpertPerRank);
    uint64_t maxOutputSize = maxRecvTokenNum != 0 ? static_cast<uint64_t>(maxRecvTokenNum) : defaultMaxOutputSize;
    OP_TILING_CHECK(maxOutputSize > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxOutputSize", std::to_string(maxOutputSize).c_str(),
                                              "should be less than or equal to UINT32_MAX"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(IsLayeredTopology(tilingData->compute.topoType) &&
                        static_cast<uint64_t>(maxRecvTokenNum) > defaultMaxOutputSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(),
                                              "should not exceed the URMA maximum receive token capacity"),
                    return ge::GRAPH_FAILED);
    tilingData->compute.maxOutputSize = static_cast<uint32_t>(maxOutputSize);
    return ge::GRAPH_SUCCESS;
}

/*
 * 将激活属性规范化后写入 tiling data。默认值必须先于条件分支设置：
 * swiglu 直接使用这些默认值；其他模式按分支覆盖，GELU 的占位参数不用于截断。
 */
static void SetActivationAttrParams(const gert::TilingContext* context, const MegaMoeConfig& config,
                                    MegaMoeTilingData* tilingData)
{
    auto attrs = context->GetAttrs();

    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);

    const size_t paramCount = activationParamsPtr->GetSize();
    const float* activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();
    const bool isGelu = std::strcmp(activationPtr, "gelu") == 0;
    const bool isSituGlu = std::strcmp(activationPtr, "situglu") == 0;
    tilingData->clampLimit = DEFAULT_ACTIVATION_CLAMP;
    if (!isGelu && !isSituGlu && paramCount != 0U) {
        tilingData->clampLimit = activationParams[0];
    }
    tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU);
    tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
    tilingData->activationAlpha = 1.0f;
    tilingData->activationBeta = 1.0f;

    if (isGelu) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::GELU);
    } else if (isSituGlu) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SITU);
        tilingData->activationBeta = activationParams[0];
        if (activationParamsPtr->GetSize() == 2U) {
            tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::LINEAR);
            tilingData->activationAlpha = activationParams[1];
        } else {
            tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
            tilingData->activationAlpha = 0.0f;
        }
    } else if (std::strcmp(activationPtr, "swiglustep") == 0) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU_STEP);
    } else if (std::strcmp(activationPtr, "swigluoai") == 0) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU_OAI);
        tilingData->activationAlpha = paramCount == 0U ? DEFAULT_SWIGLU_OAI_ALPHA : activationParams[1];
        tilingData->activationBeta = paramCount == 0U ? DEFAULT_SWIGLU_OAI_BETA : activationParams[2];
    }
}

/*
 * 集中校验所有必需输入、输出及动态输入首项的指针，后续 tensor 校验不再重复判空。
 */
static ge::graphStatus CheckRequiredTensorPtrNullptr(const gert::TilingContext* context, const MegaMoeConfig& config)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);
    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);
    auto contextShape = context->GetInputShape(config.contextIndex);
    auto xShape = context->GetInputShape(config.xIndex);
    auto topkIdsShape = context->GetInputShape(config.topkIdsIndex);
    auto topkWeightsShape = context->GetInputShape(config.topkWeightsIndex);
    auto weightOneShape = context->GetDynamicInputShape(config.weight1Index, 0);
    auto weightTwoShape = context->GetDynamicInputShape(config.weight2Index, 0);
    auto weightScalesOneShape = context->GetDynamicInputShape(config.weightScales1Index, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(config.weightScales2Index, 0);
    auto yShape = context->GetOutputShape(config.yIndex);
    auto expertTokenNumsShape = context->GetOutputShape(config.expertTokenNumsIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);

    return ge::GRAPH_SUCCESS;
}

/*
 * arch35 当前不支持 xActiveMask，可选输入一旦传入即拒绝。
 */
static ge::graphStatus CheckUnsupportedOptionalInputs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                      const char* nodeName)
{
    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask", "not null", "null"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 单独校验 scales 的传入条件：预量化输入必须传入且 shape 指针有效，其他场景不允许传入。
 */
static ge::graphStatus CheckScalesInput(const gert::TilingContext* context, const MegaMoeConfig& config,
                                        const char* nodeName)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    const bool isPreQuantizedX = *dispatchQuantModePtr == DISPATCH_QUANT_MODE_PASSTHROUGH &&
                                 IsSupportedExpertQuantDtype(static_cast<ge::DataType>(*dispatchQuantOutDtypePtr));
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    if (isPreQuantizedX) {
        OP_TILING_CHECK(scalesDesc == nullptr,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "scales", "null", "not null for pre-quantized x"),
                        return ge::GRAPH_FAILED);
        auto scalesShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context, scalesShape);
    } else {
        OP_TILING_CHECK(scalesDesc != nullptr,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "scales", "not null", "null when x is not pre-quantized"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验单个 tensor 的维数是否符合接口约定。
 */
static ge::graphStatus CheckTensorDimNum(const gert::StorageShape* storageShape, uint32_t expectedDimNum,
                                         const char* tensorName, const char* nodeName)
{
    uint32_t actualDimNum = storageShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(actualDimNum != expectedDimNum,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, tensorName, (std::to_string(actualDimNum) + "D").c_str(),
                                                 (std::to_string(expectedDimNum) + "D").c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 TensorList 指定项的维数及各维大小是否与首项一致。
 */
static ge::graphStatus CheckTensorListEntryMatchesReference(const gert::StorageShape* referenceShape,
                                                            const gert::CompileTimeTensorDesc* referenceDesc,
                                                            const gert::StorageShape* currentShape,
                                                            const gert::CompileTimeTensorDesc* currentDesc,
                                                            const std::string& tensorName, uint32_t tensorIndex,
                                                            const char* nodeName)
{
    const std::string entryName = tensorName + "[" + std::to_string(tensorIndex) + "]";
    const uint32_t referenceDimNum = referenceShape->GetStorageShape().GetDimNum();
    const uint32_t currentDimNum = currentShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(currentDimNum != referenceDimNum,
                    OP_LOGE(nodeName,
                            "%s has %u dimensions, but %s[0] has %u dimensions; "
                            "tensors in the same list must have matching dimensions.",
                            entryName.c_str(), currentDimNum, tensorName.c_str(), referenceDimNum),
                    return ge::GRAPH_FAILED);

    bool shapeMismatch = false;
    for (uint32_t dimIndex = 0; dimIndex < referenceDimNum && !shapeMismatch; ++dimIndex) {
        shapeMismatch =
            currentShape->GetStorageShape().GetDim(dimIndex) != referenceShape->GetStorageShape().GetDim(dimIndex);
    }
    OP_TILING_CHECK(shapeMismatch,
                    OP_LOGE_FOR_INVALID_SHAPE(nodeName, entryName.c_str(),
                                              Ops::Base::ToString(currentShape->GetStorageShape()).c_str(),
                                              Ops::Base::ToString(referenceShape->GetStorageShape()).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        currentDesc->GetDataType() != referenceDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, entryName.c_str(), Ops::Base::ToString(currentDesc->GetDataType()).c_str(),
                                  Ops::Base::ToString(referenceDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(currentDesc->GetStorageFormat() != referenceDesc->GetStorageFormat(),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, entryName.c_str(),
                                               Ops::Base::ToString(currentDesc->GetStorageFormat()).c_str(),
                                               Ops::Base::ToString(referenceDesc->GetStorageFormat()).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验同一 TensorList 内所有 tensor 的维数、shape、dtype 和 format 与首项一致。
 * 以第 0 项为基准，从第 1 项开始逐项比较。
 */
static ge::graphStatus CheckTensorListInternalConsistency(const gert::TilingContext* context, uint32_t inputIndex,
                                                          const char* inputRole, const char* expertTypeName,
                                                          const char* nodeName)
{
    std::string inputName = std::string(inputRole) + " of " + expertTypeName;
    uint32_t tensorCount = GetDynamicInputTensorCount(context, inputIndex);
    OP_TILING_CHECK(tensorCount == 0U, OP_LOGE_WITH_INVALID_INPUT(nodeName, inputName.c_str()),
                    return ge::GRAPH_FAILED);

    auto referenceShape = context->GetDynamicInputShape(inputIndex, 0);
    auto referenceDesc = context->GetDynamicInputDesc(inputIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceDesc);
    for (uint32_t tensorIdx = 1; tensorIdx < tensorCount; ++tensorIdx) {
        auto currentShape = context->GetDynamicInputShape(inputIndex, tensorIdx);
        auto currentDesc = context->GetDynamicInputDesc(inputIndex, tensorIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentDesc);

        if (CheckTensorListEntryMatchesReference(referenceShape, referenceDesc, currentShape, currentDesc, inputName,
                                                 tensorIdx, nodeName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightDimRelation(const gert::StorageShape* shape, uint32_t expectedDimNum,
                                              uint32_t referenceDimNum, const char* tensorName,
                                              const char* referenceName, const char* nodeName)
{
    const uint32_t actualDimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        actualDimNum != expectedDimNum,
        OP_LOGE(nodeName, "%s must have %s %s, but got %u and %u dimensions, respectively.", tensorName,
                expectedDimNum == referenceDimNum ? "the same number of dimensions as" : "one more dimension than",
                referenceName, actualDimNum, referenceDimNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧专家的 weight 与 scale tensor 维数符合当前权重组织形式。
 */
static ge::graphStatus CheckWeightAndScaleDimNum(const gert::TilingContext* context, const ExpertParams& expertParams,
                                                 uint32_t weightDimNum, const char* nodeName)
{
    const auto& inputs = expertParams.inputs;
    const uint32_t scaleDimNum = weightDimNum + 1U;
    const std::string nameSuffix = std::string(" of ") + expertParams.expertTypeName;
    const std::string weightOneName = "weight1[0]" + nameSuffix;
    const std::string weightTwoName = "weight2[0]" + nameSuffix;
    const std::string weightScalesOneName = "weight_scales1[0]" + nameSuffix;
    const std::string weightScalesTwoName = "weight_scales2[0]" + nameSuffix;

    auto weightOneShape = context->GetDynamicInputShape(inputs.weightOne, 0);
    auto weightTwoShape = context->GetDynamicInputShape(inputs.weightTwo, 0);
    auto weightScalesOneShape = context->GetDynamicInputShape(inputs.weightScalesOne, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(inputs.weightScalesTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);

    bool hasInvalidDimNum =
        CheckWeightDimRelation(weightOneShape, weightDimNum, weightDimNum, weightOneName.c_str(),
                               "weight1[0] of MoE expert", nodeName) != ge::GRAPH_SUCCESS ||
        CheckWeightDimRelation(weightTwoShape, weightDimNum, weightDimNum, weightTwoName.c_str(), weightOneName.c_str(),
                               nodeName) != ge::GRAPH_SUCCESS ||
        CheckWeightDimRelation(weightScalesOneShape, scaleDimNum, weightDimNum, weightScalesOneName.c_str(),
                               weightOneName.c_str(), nodeName) != ge::GRAPH_SUCCESS ||
        CheckWeightDimRelation(weightScalesTwoShape, scaleDimNum, weightDimNum, weightScalesTwoName.c_str(),
                               weightTwoName.c_str(), nodeName) != ge::GRAPH_SUCCESS;
    return hasInvalidDimNum ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧两个 weight scale tensor 的末维符合 multi-base 存储约定。
 * 调用前应先通过 CheckWeightAndScaleDimNum，保证目标维存在。
 */
static ge::graphStatus CheckWeightScaleTrailingDim(const gert::TilingContext* context, const ExpertParams& expertParams,
                                                   bool isPerExpertWeightTensor, const char* nodeName)
{
    auto weightScalesOneShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesOne, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);

    const int64_t scaleOneMultiBase =
        GetSingleExpertTensorDimSize(weightScalesOneShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t scaleTwoMultiBase =
        GetSingleExpertTensorDimSize(weightScalesTwoShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const std::string scaleNames = std::string(expertParams.expertTypeName) + " weight_scales1, weight_scales2";
    const std::string trailingDimensions =
        "[" + std::to_string(scaleOneMultiBase) + ", " + std::to_string(scaleTwoMultiBase) + "]";
    OP_TILING_CHECK(
        scaleOneMultiBase != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE || scaleTwoMultiBase != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, scaleNames.c_str(), trailingDimensions.c_str(),
                                               "The trailing dimension of each weight scale tensor must be 2"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧四类 weight/scale 动态输入的 Tensor 数量一致；stacked 布局要求每类仅有一个 Tensor。
 */
static ge::graphStatus CheckWeightAndScaleTensorCounts(const gert::TilingContext* context,
                                                       const ExpertParams& expertParams, bool isPerExpertWeightTensor,
                                                       const char* nodeName)
{
    const auto& inputs = expertParams.inputs;
    uint32_t weightOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightOne);
    uint32_t weightTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightTwo);
    uint32_t weightScalesOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesOne);
    uint32_t weightScalesTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesTwo);
    bool tensorCountsMismatch = weightTwoTensorCount != weightOneTensorCount ||
                                weightScalesOneTensorCount != weightOneTensorCount ||
                                weightScalesTwoTensorCount != weightOneTensorCount;
    OP_TILING_CHECK(
        tensorCountsMismatch || (!isPerExpertWeightTensor && weightOneTensorCount != 1U),
        OP_LOGE(nodeName,
                "weight1, weight2, weight_scales1 and weight_scales2 of %s must contain the same number of tensors; "
                "stacked layout requires exactly one tensor.",
                expertParams.expertTypeName),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 stacked 布局下四类 weight/scale tensor 的专家维大小相互一致。
 */
static ge::graphStatus CheckStackedExpertDimConsistency(const gert::TilingContext* context,
                                                        const ExpertParams& expertParams, const char* nodeName)
{
    const auto& inputs = expertParams.inputs;
    auto weightOneShape = context->GetDynamicInputShape(inputs.weightOne, 0);
    auto weightTwoShape = context->GetDynamicInputShape(inputs.weightTwo, 0);
    auto weightScalesOneShape = context->GetDynamicInputShape(inputs.weightScalesOne, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(inputs.weightScalesTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);

    int64_t expertCount = weightOneShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(weightTwoShape->GetStorageShape().GetDim(0) != expertCount ||
                        weightScalesOneShape->GetStorageShape().GetDim(0) != expertCount ||
                        weightScalesTwoShape->GetStorageShape().GetDim(0) != expertCount,
                    OP_LOGE(nodeName,
                            "Dim0 of weight1, weight2, weight_scales1 and weight_scales2 must have matching expert "
                            "counts for %s inputs.",
                            expertParams.expertTypeName),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧专家四类 weight/scale tensor 的列表长度、列表内部属性及堆叠专家维保持一致。
 */
static ge::graphStatus CheckWeightAndScaleConsistency(const gert::TilingContext* context,
                                                      const ExpertParams& expertParams, uint32_t weightDimNum,
                                                      const char* nodeName)
{
    const auto& inputs = expertParams.inputs;
    const bool isPerExpertWeightTensor = weightDimNum == TWO_DIMS;
    if (CheckWeightAndScaleTensorCounts(context, expertParams, isPerExpertWeightTensor, nodeName) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool inconsistentTensorLists =
        CheckTensorListInternalConsistency(context, inputs.weightOne, "weight1", expertParams.expertTypeName,
                                           nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorListInternalConsistency(context, inputs.weightTwo, "weight2", expertParams.expertTypeName,
                                           nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorListInternalConsistency(context, inputs.weightScalesOne, "weight_scales1",
                                           expertParams.expertTypeName, nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorListInternalConsistency(context, inputs.weightScalesTwo, "weight_scales2",
                                           expertParams.expertTypeName, nodeName) != ge::GRAPH_SUCCESS;
    if (inconsistentTensorLists) {
        return ge::GRAPH_FAILED;
    }
    if (!isPerExpertWeightTensor) {
        return CheckStackedExpertDimConsistency(context, expertParams, nodeName);
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 以 MoE weight1[0] 的维数确定权重输入的组织形式：2D 表示逐专家 TensorList，3D 表示单 tensor 堆叠。
 * 分别校验 MoE 和共享专家的 weight/scale 维数契约与一致性，并记录布局派生信息。
 * 本函数不校验 GMM 矩阵维关系，也不判断 dtype/format 组合是否受支持。
 */
static ge::graphStatus CheckAndSetWeightAndScaleLayout(const gert::TilingContext* context, MegaMoeExpertParams& params,
                                                       MegaMoeTilingData* tilingData, const char* nodeName)
{
    auto weightOneShape = context->GetDynamicInputShape(params.moe.inputs.weightOne, 0);
    uint32_t weightDimNum = weightOneShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        weightDimNum != TWO_DIMS && weightDimNum != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "weight1", (std::to_string(weightDimNum) + "D").c_str(),
                                                 "weight1 must be one stacked 3D tensor or a list of 2D tensors."),
        return ge::GRAPH_FAILED);
    params.isPerExpertWeightTensor = weightDimNum == TWO_DIMS;

    OP_TILING_CHECK(CheckWeightAndScaleDimNum(context, params.moe, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight or scale tensor dimension is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckWeightScaleTrailingDim(context, params.moe, params.isPerExpertWeightTensor, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "MoE expert weight scale trailing dimension is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckWeightAndScaleConsistency(context, params.moe, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight layout is invalid."), return ge::GRAPH_FAILED);
    params.moe.expertCount = GetWeightExpertCount(context, params.moe.inputs.weightOne, params.isPerExpertWeightTensor);

    // 检查共享专家的四个 weight/scale 输入是否同时存在或同时缺省。
    params.shared.expertCount =
        GetWeightExpertCount(context, params.shared.inputs.weightOne, params.isPerExpertWeightTensor);
    bool hasSharedWeightOne = params.shared.expertCount > 0;
    bool hasSharedWeightTwo =
        GetWeightExpertCount(context, params.shared.inputs.weightTwo, params.isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesOne =
        GetWeightExpertCount(context, params.shared.inputs.weightScalesOne, params.isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesTwo =
        GetWeightExpertCount(context, params.shared.inputs.weightScalesTwo, params.isPerExpertWeightTensor) > 0;
    OP_TILING_CHECK(hasSharedWeightOne != hasSharedWeightTwo || hasSharedWeightOne != hasSharedWeightScalesOne ||
                        hasSharedWeightOne != hasSharedWeightScalesTwo,
                    OP_LOGE(nodeName, "Shared expert weights and weight scales must be provided together."),
                    return ge::GRAPH_FAILED);
    if (hasSharedWeightOne) {
        OP_TILING_CHECK(CheckWeightAndScaleDimNum(context, params.shared, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Shared expert weight or scale tensor dimension is invalid."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(CheckWeightScaleTrailingDim(context, params.shared, params.isPerExpertWeightTensor, nodeName) !=
                            ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Shared expert weight scale trailing dimension is invalid."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            CheckWeightAndScaleConsistency(context, params.shared, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Shared expert weight layout is invalid."), return ge::GRAPH_FAILED);
    }
    tilingData->isPerExpertWeightTensor = params.isPerExpertWeightTensor;
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧 weight1、weight2 与 x 的矩阵维关系。
 */
static ge::graphStatus CheckWeightPairShapeRelations(const gert::TilingContext* context,
                                                     const ExpertParams& expertParams, uint32_t xIndex,
                                                     bool isPerExpertWeightTensor, uint8_t actMode,
                                                     const char* nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightOne, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    auto weightTwoStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);

    // 去掉可选的专家维后，weight1 和 weight2 均按单专家二维矩阵读取行数和列数。
    const int64_t weight1RowCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weight1ColumnCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weight2RowCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weight2ColumnCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);

    // 单专家 GMM 形状：weight1=[N, H]，weight2=[H, N / divisor]，x=[BS, H]。
    const gert::StorageShape* xStorageShape = context->GetInputShape(xIndex);
    const int64_t xColumnCount = xStorageShape->GetOriginShape().GetDim(1);
    const std::string commonMatrixDimensionsString = "[" + std::to_string(weight1ColumnCount) + ", " +
                                                     std::to_string(weight2RowCount) + ", " +
                                                     std::to_string(xColumnCount) + "]";
    OP_TILING_CHECK(weight1ColumnCount != weight2RowCount || weight1ColumnCount != xColumnCount,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, expertParams.expertTypeName, commonMatrixDimensionsString.c_str(),
                        "weight1 column count and weight2 row count must equal x column count"),
                    return ge::GRAPH_FAILED);

    const std::string weightRowColumnDimensionsString =
        "[" + std::to_string(weight1RowCount) + ", " + std::to_string(weight2ColumnCount) + "]";
    const int64_t outputDivisor = static_cast<int64_t>(GetActivationOutputDivisor(actMode));
    OP_TILING_CHECK(weight1RowCount % outputDivisor != 0 || weight2ColumnCount != weight1RowCount / outputDivisor,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, expertParams.expertTypeName, weightRowColumnDimensionsString.c_str(),
                        (std::string("weight2 column count must equal weight1 row count divided by ") +
                         std::to_string(outputDivisor))
                            .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 分别校验 MoE 与已启用共享专家的 weight1、weight2 矩阵维关系。
 */
static ge::graphStatus CheckWeightShapeRelations(const gert::TilingContext* context,
                                                 const MegaMoeExpertParams& expertParams, uint32_t xIndex,
                                                 uint8_t actMode, const char* nodeName)
{
    if (CheckWeightPairShapeRelations(context, expertParams.moe, xIndex, expertParams.isPerExpertWeightTensor, actMode,
                                      nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (expertParams.shared.expertCount <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    return CheckWeightPairShapeRelations(context, expertParams.shared, xIndex, expertParams.isPerExpertWeightTensor,
                                         actMode, nodeName);
}

/*
 * 校验输出 y 的维数及 shape 是否与输入 x 一致。
 */
static ge::graphStatus CheckYShape(const gert::TilingContext* context, const MegaMoeConfig& config,
                                   const char* nodeName)
{
    const gert::StorageShape* xStorageShape = context->GetInputShape(config.xIndex);

    int64_t bs = xStorageShape->GetOriginShape().GetDim(0);
    int64_t h = xStorageShape->GetOriginShape().GetDim(1);
    auto yStorageShape = context->GetOutputShape(config.yIndex);
    if (CheckTensorDimNum(yStorageShape, TWO_DIMS, "y", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        yDim0 != bs || yDim1 != h,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "y", (std::string("[") + std::to_string(yDim0) + ", " + std::to_string(yDim1) + "]").c_str(),
            (std::string("The shape of y must be [bs, h] = [") + std::to_string(bs) + ", " + std::to_string(h) + "].")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 expert_token_nums 为一维 tensor，且长度等于本 rank 的 MoE 专家数。
 */
static ge::graphStatus CheckExpertTokenNumsShape(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                 uint32_t moeExpertPerRank, const char* nodeName)
{
    auto expertTokenNumsStorageShape = context->GetOutputShape(config.expertTokenNumsIndex);
    if (CheckTensorDimNum(expertTokenNumsStorageShape, ONE_DIM, "expert_token_nums", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    // expertTokenNums 仅报告 MoE 专家的 token 数，不包含共享专家。
    OP_TILING_CHECK(
        expertTokenNumsDim0 != moeExpertPerRank,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "expertTokenNums", (std::string("dim0=") + std::to_string(expertTokenNumsDim0)).c_str(),
            (std::string("The shape [dim0] of expertTokenNums must be equal to moeExpertPerRank(") +
             std::to_string(moeExpertPerRank) + ").")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一组 weight 与对应 scale 的 shape 关系。每 INPUT_WEIGHT_SCALES_CEIL_ALIGN 个 weight 列
 * 共用一个 scale，因此 scale shape 应为 [weight 行数, ceil(weight 列数 / 对齐值)]。
 * weight1 和 weight2 采用相同规则；scale 自身的 shape 约束已在布局解析阶段校验。
 */
static ge::graphStatus CheckWeightScaleShapeRelation(const gert::StorageShape* weightScaleStorageShape,
                                                     const gert::StorageShape* weightStorageShape,
                                                     bool isPerExpertWeightTensor, const char* expertTypeName,
                                                     const char* weightScaleRole, const char* weightRole,
                                                     const char* nodeName)
{
    const int64_t weightScaleMatrixDimSize =
        GetSingleExpertTensorDimSize(weightScaleStorageShape, WEIGHT_SCALE_MATRIX_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightScaleGroupDimSize =
        GetSingleExpertTensorDimSize(weightScaleStorageShape, WEIGHT_SCALE_GROUP_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightRowCount =
        GetSingleExpertTensorDimSize(weightStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightColumnCount =
        GetSingleExpertTensorDimSize(weightStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);

    OP_TILING_CHECK(weightScaleMatrixDimSize != weightRowCount,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, (std::string(expertTypeName) + " " + weightScaleRole).c_str(),
                        (std::string("matrix dimension=") + std::to_string(weightScaleMatrixDimSize)).c_str(),
                        (std::string("The matrix dimension must equal the row count of ") + expertTypeName + " " +
                         weightRole + "(" + std::to_string(weightRowCount) + ")")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(weightScaleGroupDimSize != ops::CeilDiv(weightColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, (std::string(expertTypeName) + " " + weightScaleRole).c_str(),
                        (std::string("group dimension=") + std::to_string(weightScaleGroupDimSize)).c_str(),
                        (std::string("The group dimension must equal CeilDiv(") + expertTypeName + " " + weightRole +
                         " column count, INPUT_WEIGHT_SCALES_CEIL_ALIGN) = " +
                         std::to_string(ops::CeilDiv(weightColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN)))
                            .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧两组 weight scale 与对应 weight 的 shape 关系。
 */
static ge::graphStatus CheckWeightScaleShapeRelations(const gert::TilingContext* context,
                                                      const ExpertParams& expertParams, bool isPerExpertWeightTensor,
                                                      const char* nodeName)
{
    // CheckAndSetWeightAndScaleLayout 已保证本侧四个 TensorList 的首项 shape 存在且维数合法。
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesOne, 0);
    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesTwo, 0);
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightOne, 0);
    auto weightTwoStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightTwo, 0);

    ge::graphStatus checkStatus =
        CheckWeightScaleShapeRelation(weightScalesOneStorageShape, weightOneStorageShape, isPerExpertWeightTensor,
                                      expertParams.expertTypeName, "weight_scales1", "weight1", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }
    checkStatus =
        CheckWeightScaleShapeRelation(weightScalesTwoStorageShape, weightTwoStorageShape, isPerExpertWeightTensor,
                                      expertParams.expertTypeName, "weight_scales2", "weight2", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验已传入的 x scales 维数及其与逻辑 BS/H 的 shape 关系。
 */
static ge::graphStatus CheckXScalesShape(const gert::TilingContext* context, const MegaMoeConfig& config, int64_t bs,
                                         int64_t h, const char* nodeName)
{
    const auto* scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    if (scalesDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::StorageShape* scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    const auto& scalesShape = scalesStorageShape->GetOriginShape();
    OP_TILING_CHECK(
        scalesShape.GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", (std::to_string(scalesShape.GetDimNum()) + "D").c_str(),
                                     (std::to_string(TWO_DIMS) + "D").c_str()),
        return ge::GRAPH_FAILED);
    const int64_t expectedScaleDim1 = ops::CeilDiv<int64_t>(h, MX_SCALE_GROUP);
    OP_TILING_CHECK(scalesShape.GetDim(0) != bs || scalesShape.GetDim(1) != expectedScaleDim1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "scales",
                        (std::string("[") + std::to_string(scalesShape.GetDim(0)) + ", " +
                         std::to_string(scalesShape.GetDim(1)) + "]")
                            .c_str(),
                        (std::string("For pre-quantized x, scales must be [BS, CeilDiv(H, 32)] = [") +
                         std::to_string(bs) + ", " + std::to_string(expectedScaleDim1) + "]")
                            .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验路由输入的维数、token 数及 topK 维一致性。
 */
static ge::graphStatus CheckRoutingInputShapes(const gert::TilingContext* context, const MegaMoeConfig& config,
                                               int64_t bs, const char* nodeName)
{
    const gert::StorageShape* topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    if (CheckTensorDimNum(topkIdsStorageShape, TWO_DIMS, "topkIds", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape* topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    if (CheckTensorDimNum(topkWeightsStorageShape, TWO_DIMS, "topkWeights", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(bs != topkIdsDim0 || bs != topkWeightsDim0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "x, topkIds, topkWeights",
                        (std::string("[") + std::to_string(bs) + ", " + std::to_string(topkIdsDim0) + ", " +
                         std::to_string(topkWeightsDim0) + "]")
                            .c_str(),
                        "The shape [dim0] of x, topkIds, and topkWeights must be equal."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        topkIdsDim1 != topkWeightsDim1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "topkIds, topkWeights",
            (std::string("[") + std::to_string(topkIdsDim1) + ", " + std::to_string(topkWeightsDim1) + "]").c_str(),
            "The shape [dim1] of topkIds and topkWeights must be equal."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 dispatch 基础输入维数，并按顺序校验 x scales 和路由输入的 shape 关系。
 */
static ge::graphStatus CheckDispatchInputShapes(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                const char* nodeName)
{
    const gert::StorageShape* contextStorageShape = context->GetInputShape(config.contextIndex);
    if (CheckTensorDimNum(contextStorageShape, ONE_DIM, "context", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* xStorageShape = context->GetInputShape(config.xIndex);
    if (CheckTensorDimNum(xStorageShape, TWO_DIMS, "x", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t bs = xStorageShape->GetOriginShape().GetDim(0);
    const int64_t h = xStorageShape->GetOriginShape().GetDim(1);
    OP_TILING_CHECK(CheckXScalesShape(context, config, bs, h, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "x scales shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRoutingInputShapes(context, config, bs, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "routing input shapes are invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 weight1/weight2/x、weight/scale 以及 MoE/shared hiddenDim 之间的 shape 关系。
 */
static ge::graphStatus CheckWeightAndScaleShapeRelations(const gert::TilingContext* context,
                                                         const MegaMoeConfig& config,
                                                         const MegaMoeExpertParams& expertParams, uint8_t actMode,
                                                         const char* nodeName)
{
    OP_TILING_CHECK(
        CheckWeightShapeRelations(context, expertParams, config.xIndex, actMode, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "weight shape relation is invalid."), return ge::GRAPH_FAILED);

    if (expertParams.shared.expertCount > 0) {
        // 两侧 W1 column 和 W2 row 已分别校验为 x.H，且各自满足相同的激活输出宽度关系。
        // 因此跨侧只需保证 W1 row（GMM1 输出维 hiddenDim）一致。
        auto moeWeightOneShape = context->GetDynamicInputShape(expertParams.moe.inputs.weightOne, 0);
        auto sharedWeightOneShape = context->GetDynamicInputShape(expertParams.shared.inputs.weightOne, 0);
        int64_t moeHiddenDim = GetSingleExpertTensorDimSize(moeWeightOneShape, WEIGHT_MATRIX_ROW_DIM_INDEX,
                                                            expertParams.isPerExpertWeightTensor);
        int64_t sharedHiddenDim = GetSingleExpertTensorDimSize(sharedWeightOneShape, WEIGHT_MATRIX_ROW_DIM_INDEX,
                                                               expertParams.isPerExpertWeightTensor);
        OP_TILING_CHECK(sharedHiddenDim != moeHiddenDim,
                        OP_LOGE_FOR_INVALID_VALUE(
                            nodeName, "shared expert hiddenDim", std::to_string(sharedHiddenDim).c_str(),
                            (std::string("must equal MoE expert hiddenDim ") + std::to_string(moeHiddenDim)).c_str()),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(CheckWeightScaleShapeRelations(context, expertParams.moe, expertParams.isPerExpertWeightTensor,
                                                   nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "MoE weight scale shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        expertParams.shared.expertCount > 0 &&
            CheckWeightScaleShapeRelations(context, expertParams.shared, expertParams.isPerExpertWeightTensor,
                                           nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "shared weight scale shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验非权重输入张量的 dtype：context / x / topkIds / topkWeights。
 */
static ge::graphStatus CheckNonWeightInputDataTypes(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                    const char* nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "context",
                                                          Ops::Base::ToString(contextDesc->GetDataType()).c_str(),
                                                          "The dtype of context must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        xDesc->GetDataType() != ge::DT_BF16 && !IsPreQuantizedXType(xDesc->GetDataType()),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "The dtype of x must be DT_BF16, DT_FLOAT8_E5M2, "
                                              "DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1"),
        return ge::GRAPH_FAILED);

    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(scalesDesc != nullptr && scalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "scales", Ops::Base::ToString(scalesDesc->GetDataType()).c_str(),
                        "For pre-quantized x, the dtype of scales must be DT_FLOAT8_E8M0"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkIds",
                                                          Ops::Base::ToString(topkIdsDesc->GetDataType()).c_str(),
                                                          "The dtype of topkIds must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((topkWeightsDesc->GetDataType() != ge::DT_BF16) && (topkWeightsDesc->GetDataType() != ge::DT_FLOAT)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkWeights",
                                              Ops::Base::ToString(topkWeightsDesc->GetDataType()).c_str(),
                                              "The dtype of topkWeights must be DT_FLOAT or DT_BF16."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验输出张量的 dtype：y 与 expertTokenNums。
 */
static ge::graphStatus CheckOutputDataTypes(const gert::TilingContext* context, const MegaMoeConfig& config,
                                            const char* nodeName)
{
    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(
        yDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "y", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "The dtype of y must be DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "expertTokenNums", Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str(),
                        "The dtype of expertTokenNums must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 判断 tensor 的主格式是否属于 FRACTAL_NZ。
 */
static bool IsNzPrimaryFormat(const gert::CompileTimeTensorDesc* desc)
{
    return static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ;
}

/*
 * 校验非权重输入和输出不使用 FRACTAL_NZ 主格式。
 */
static ge::graphStatus CheckNonWeightTensorFormats(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                   const char* nodeName)
{
    const gert::CompileTimeTensorDesc* descs[] = {
        context->GetInputDesc(config.xIndex), context->GetInputDesc(config.topkIdsIndex),
        context->GetInputDesc(config.topkWeightsIndex), context->GetOutputDesc(config.yIndex),
        context->GetOutputDesc(config.expertTokenNumsIndex)};
    const char* names[] = {"x", "topk_ids", "topk_weights", "y", "expert_token_nums"};
    for (uint32_t index = 0; index < sizeof(descs) / sizeof(descs[0]); ++index) {
        OP_TILING_CHECK(IsNzPrimaryFormat(descs[index]),
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, names[index],
                                                   Ops::Base::ToString(descs[index]->GetStorageFormat()).c_str(),
                                                   "a non-FRACTAL_NZ format"),
                        return ge::GRAPH_FAILED);
    }

    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(scalesDesc != nullptr && scalesDesc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(
                        nodeName, "scales", Ops::Base::ToString(scalesDesc->GetStorageFormat()).c_str(), "FORMAT_ND"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧 weight scale 的类型和格式；weight 类型与格式组合由后续 GMM 模式解析校验。
 */
static ge::graphStatus CheckExpertScaleDataTypesAndFormats(const gert::TilingContext* context,
                                                           const ExpertParams& expertParams, const char* nodeName)
{
    auto scaleOneDesc = context->GetDynamicInputDesc(expertParams.inputs.weightScalesOne, 0);
    auto scaleTwoDesc = context->GetDynamicInputDesc(expertParams.inputs.weightScalesTwo, 0);
    const std::string scaleNames = std::string(expertParams.expertTypeName) + " weight_scales1, weight_scales2";

    bool invalidDtype =
        scaleOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0 || scaleTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0;
    const std::string scaleDtypes = "[" + Ops::Base::ToString(scaleOneDesc->GetDataType()) + ", " +
                                    Ops::Base::ToString(scaleTwoDesc->GetDataType()) + "]";
    OP_TILING_CHECK(invalidDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, scaleNames.c_str(), scaleDtypes.c_str(),
                                                           "Both weight scale dtypes must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);

    const std::string scaleFormats = "[" + Ops::Base::ToString(scaleOneDesc->GetStorageFormat()) + ", " +
                                     Ops::Base::ToString(scaleTwoDesc->GetStorageFormat()) + "]";
    OP_TILING_CHECK(
        IsNzPrimaryFormat(scaleOneDesc) || IsNzPrimaryFormat(scaleTwoDesc),
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(nodeName, scaleNames.c_str(), scaleFormats.c_str(),
                                                "Weight scale tensors must use a non-FRACTAL_NZ primary format."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 MoE 和共享专家 weight scale 的数据类型与格式契约。
 */
static ge::graphStatus CheckWeightScaleDataTypesAndFormats(const gert::TilingContext* context,
                                                           const MegaMoeExpertParams& expertParams,
                                                           const char* nodeName)
{
    OP_TILING_CHECK(CheckExpertScaleDataTypesAndFormats(context, expertParams.moe, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight scale properties are invalid."), return ge::GRAPH_FAILED);

    if (expertParams.shared.expertCount > 0) {
        OP_TILING_CHECK(
            CheckExpertScaleDataTypesAndFormats(context, expertParams.shared, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "shared expert weight scale properties are invalid."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 从 x 和 topkIds 提取 bs、H 与 topK，校验取值范围以及按拓扑确定的 H 对齐要求。
 */
static ge::graphStatus CheckAndSetTokenDimensions(const gert::TilingContext* context, MegaMoeTilingData* tilingData,
                                                  const MegaMoeConfig& config, const char* nodeName)
{
    const gert::StorageShape* xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape* topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t bs = xStorageShape->GetOriginShape().GetDim(0);
    int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);

    // 校验 topK 取值范围。
    OP_TILING_CHECK(
        topkIdsDim1 < MIN_TOPK || topkIdsDim1 > MAX_TOPK,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "topK", std::to_string(topkIdsDim1).c_str(), "only support [1, 32]"),
        return ge::GRAPH_FAILED);

    int64_t xDim1 = xStorageShape->GetOriginShape().GetDim(1);
    // 校验 H 取值范围。
    OP_TILING_CHECK(
        xDim1 < MIN_H || xDim1 > MAX_H,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "H", std::to_string(xDim1).c_str(),
            (std::string("should in [") + std::to_string(MIN_H) + ", " + std::to_string(MAX_H) + "]").c_str()),
        return ge::GRAPH_FAILED);
    // MTE 与 URMA/Layered 统一按 64 对齐：满足 W4 NZ_C0_32 的 GMM K 分形要求，
    // A8W8 的 ND/NZ 路径沿用同一口径。
    int64_t requiredHAlignment = IsLayeredTopology(tilingData->compute.topoType) ? URMA_H_ALIGN : MTE_H_ALIGN;
    OP_TILING_CHECK(
        xDim1 % requiredHAlignment != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(xDim1).c_str(),
                                  (std::string("multiple of ") + std::to_string(requiredHAlignment)).c_str()),
        return ge::GRAPH_FAILED);

    tilingData->compute.bs = static_cast<uint32_t>(bs);
    tilingData->compute.h = static_cast<uint32_t>(xDim1);
    tilingData->compute.topK = static_cast<uint32_t>(topkIdsDim1);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并设置 GMM1 输出维 hiddenDim：取值范围与统一对齐要求。
 */
static ge::graphStatus CheckAndSetHiddenDim(const gert::TilingContext* context, MegaMoeTilingData* tilingData,
                                            const MegaMoeExpertParams& expertParams, const char* nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.moe.inputs.weightOne, 0);
    int64_t hiddenDim = GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX,
                                                     expertParams.isPerExpertWeightTensor);
    OP_TILING_CHECK(hiddenDim < MIN_HIDDEN_DIM || hiddenDim > MAX_HIDDEN_DIM,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hiddenDim", std::to_string(hiddenDim).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_HIDDEN_DIM) + ", " +
                                               std::to_string(MAX_HIDDEN_DIM) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hiddenDim % HIDDEN_DIM_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hiddenDim", std::to_string(hiddenDim).c_str(),
                                              (std::string("multiple of ") + std::to_string(HIDDEN_DIM_ALIGN)).c_str()),
                    return ge::GRAPH_FAILED);

    tilingData->compute.hiddenDim = static_cast<uint32_t>(hiddenDim);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验通信拓扑枚举并写入 tiling data，供后续模式和资源规划使用。
 */
static ge::graphStatus CheckAndSetTopoType(const gert::TilingContext* context, const MegaMoeConfig& config,
                                           MegaMoeTilingData* tilingData, const char* nodeName)
{
    int64_t topoType = *context->GetAttrs()->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    OP_TILING_CHECK(
        topoType != TOPO_TYPE_MTE && !IsLayeredTopology(topoType),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "topo_type", std::to_string(topoType).c_str(), "MTE(0), URMA(1) or UBX(2)"),
        return ge::GRAPH_FAILED);
    tilingData->compute.topoType = topoType;
    return ge::GRAPH_SUCCESS;
}

/*
 * 读取并校验平台 AIC/AIV 数量与 UB 容量，同时写入 kernel 分核所需字段。
 */
static ge::graphStatus CheckAndSetPlatformParams(gert::TilingContext* context, MegaMoeTilingData* tilingData,
                                                 uint32_t& aicNum, uint64_t& ubSize, const char* nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    aicNum = ascendcPlatform.GetCoreNumAic();
    OP_TILING_CHECK(aivNum == 0U || aicNum == 0U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "aivNum/aicNum",
                                              (std::to_string(aivNum) + ", " + std::to_string(aicNum)).c_str(),
                                              "should both be > 0"),
                    return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingData->compute.aicNum = aicNum;
    tilingData->blockAivNum = aivNum;
    tilingData->blockNumPerEP = std::max(1U, aicNum / static_cast<uint32_t>(tilingData->compute.epWorldSize));
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);
    return ge::GRAPH_SUCCESS;
}

/*
 * prepass chunk 可行性护栏：floorChunk（保 chunkCount≤DEDUP_PREPASS_MAX_CHUNKS）不得冲破
 * UB 容量上限，否则 prepass member 区越界写（如 k32 且 numMaxTokensPerRank>18432）。
 * 公式与 kernel CalcDedupPrepassChunkTokens 逐项同源（常量单一真源 mega_moe_constants.h）。
 * 超界处置分请求来源：combine 位是显式请求（combine_comm_mode=1），撞硬约束必须显式报错、
 * 不静默降级；dispatch 位是内部自动优化，静默关闭按基线继续。
 */
static ge::graphStatus ApplyDedupPrepassChunkGuard(MegaMoeTilingData* tilingData, const char* nodeName)
{
    if (tilingData->compute.topoType != TOPO_TYPE_MTE || tilingData->compute.dedupMode == 0) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t chunkUbCap =
        DEDUP_MEMBER_UB_BYTES / (tilingData->compute.topK * 3U * static_cast<uint32_t>(sizeof(int32_t)));
    chunkUbCap = std::min(chunkUbCap, 512U);
    chunkUbCap = std::min(chunkUbCap, DEDUP_COUNT_UB_BYTES / static_cast<uint32_t>(sizeof(int32_t)));
    chunkUbCap = std::min(chunkUbCap, DEDUP_ROUTE_LOAD_UB_BYTES / static_cast<uint32_t>(sizeof(int32_t)));
    const uint32_t floorChunk =
        (tilingData->compute.numMaxTokensPerRank + DEDUP_PREPASS_MAX_CHUNKS - 1U) / DEDUP_PREPASS_MAX_CHUNKS;
    if (floorChunk <= chunkUbCap) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(
        IsCombineDedupOn(tilingData->compute.dedupMode),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "num_max_tokens_per_rank", std::to_string(tilingData->compute.numMaxTokensPerRank).c_str(),
            (std::string("enabling combine dedup requires num_max_tokens_per_rank no greater than ") +
             std::to_string(static_cast<int64_t>(chunkUbCap) * DEDUP_PREPASS_MAX_CHUNKS) + " when topk is " +
             std::to_string(tilingData->compute.topK) +
             "; reduce num_max_tokens_per_rank or topk, or set combine_comm_mode=0")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "dedup gate: prepass floorChunk %u exceeds UB chunk cap %u (topK=%ld, numMax=%ld), disable dedup",
            floorChunk, chunkUbCap, tilingData->compute.topK, tilingData->compute.numMaxTokensPerRank);
    tilingData->compute.dedupMode = 0;
    return ge::GRAPH_SUCCESS;
}

/*
 * 把 tilingData 折算成门禁分数的形状输入（口径见 mega_moe_dedup_gate.h）。
 * 以下任一不成立即返回 false，由调用方回落旧判据：topk 权重预取开、每卡共享专家数非 1、
 * 激活非 SwiGLU——这些都是标定批的固定运行配置。
 * 量化模式是否落在标定档内不在这里判，由 IsDedupGateShapeCalibrated 统一解析，避免同一判据两处算。
 */
static bool TryBuildDedupGateShape(const MegaMoeTilingData* tilingData, MegaMoeImpl::DedupGateShape& shape)
{
    if (tilingData->topkWeightsPrefetch != 0 || tilingData->compute.sharedExpertNum != 1U ||
        tilingData->actMode != static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU)) {
        return false;
    }
    shape = MegaMoeImpl::DedupGateShape{
        static_cast<int64_t>(tilingData->compute.epWorldSize),
        static_cast<int64_t>(tilingData->compute.numMaxTokensPerRank),
        static_cast<int64_t>(tilingData->compute.h),
        MegaMoeImpl::GetActivationOutputWidth(static_cast<int64_t>(tilingData->compute.hiddenDim), tilingData->actMode),
        static_cast<int64_t>(tilingData->compute.moeExpertPerRank) *
            static_cast<int64_t>(tilingData->compute.epWorldSize),
        static_cast<int64_t>(tilingData->compute.topK)};
    return true;
}

/*
 * dispatch 去重收益自适应门（纯性能决策；dispatch 去重 bit 级等价，关闭不改变任何数值结果）。
 * 只裁决 dispatch 位：combine 位由 combine_comm_mode 显式决定，开即强制走、不受本门裁决。
 * 标定域内用门禁分数判决；域外沿用旧的节省字节判据，行为与标定前完全一致。
 * 裁决输入只用各卡一致的量（numMaxTokensPerRank/topK/h/hiddenDim/本卡专家数/epWorldSize），不读本卡 bs：
 * 可变 bs 下各卡 bs 不同，而记录是否带权重段、combine 的 canonical 槽协议都要求各卡 dedupMode 一致，
 * 按本卡 bs 裁决会一卡开一卡关、跨卡读记录错位。attr 为 0 时 numMaxTokensPerRank 回落 bs，定长场景行为不变。
 */
static void ApplyDedupBenefitGate(MegaMoeTilingData* tilingData, const char* nodeName)
{
    if (IsDispatchDedupOn(tilingData->compute.dedupMode)) {
        const bool combineDedupOn = IsCombineDedupOn(tilingData->compute.dedupMode);
        MegaMoeImpl::DedupGateShape gateShape{};
        if (TryBuildDedupGateShape(tilingData, gateShape) &&
            MegaMoeImpl::IsDedupGateShapeCalibrated(gateShape, tilingData->moeGmmMode)) {
            const double gateScore = MegaMoeImpl::CalcDedupGateScore(gateShape, tilingData->moeGmmMode, combineDedupOn);
            if (!MegaMoeImpl::IsDispatchDedupBeneficial(gateScore)) {
                OP_LOGI(nodeName,
                        "dedup gate: score %.6f not positive (b=%ld, h=%ld, i=%ld, e=%ld, k=%ld, gmmMode=%u, "
                        "combineDedupOn=%d), disable dispatch dedup for this shape",
                        gateScore, gateShape.maxTokensPerRank, gateShape.hidden, gateShape.intermediateHidden,
                        gateShape.globalMoeExpertNum, gateShape.topK, static_cast<uint32_t>(tilingData->moeGmmMode),
                        static_cast<int32_t>(combineDedupOn));
                // 关掉 dispatch 去重、保留 combine 位的显式请求值：3(双开)->1(仅combine)，2(仅dispatch)->0(全关)。
                tilingData->compute.dedupMode = combineDedupOn ? 1 : 0;
            }
            return;
        }
        double world = static_cast<double>(tilingData->compute.epWorldSize);
        double topKf = static_cast<double>(tilingData->compute.topK);
        // 重复率取均匀路由模型 E[distinct ranks] = world*(1-(1-1/world)^topK)，是真实路由的保守下界。
        double distinctRanks = world * (1.0 - std::pow(1.0 - 1.0 / world, topKf));
        double dupRatio = 1.0 - distinctRanks / topKf;
        // a4 家族(fp4, 2 元素/字节)每行实际发送字节≈h/2：按激活打包口径折算（与 peermem elemsPerByte 同判据）。
        const double actBytesPerElem = IsA4W4GmmMode(tilingData->moeGmmMode) ? 0.5 : 1.0;
        double savedBytes = static_cast<double>(tilingData->compute.numMaxTokensPerRank) * topKf * dupRatio *
                            static_cast<double>(tilingData->compute.h) * actBytesPerElem;
        /*
         * 以下为门禁分数标定域外的回落判据（域内已在上面 return）。
         * 4 卡判据（按去重强制开/关双态逐形状实测 24 个典型形状标定，判定与实测收益 0 错开）：
         * 判别量 = 每个本卡专家摊到的节省字节。prepass/dispatch 的固定成本按 (专家, 来源卡) 槽付，
         * 每卡 64 专家时 bs64~4096 真开恒亏约 50~150us；同样的节省量摊到 12 专家上则净赢 15%~50%。
         * 再按 hiddenDim 分两档：hiddenDim 小则每行计算轻、通信在关键路径上，少量节省即兑现；
         * hiddenDim 大则计算遮住通信，节省量要大到让通信重新成为瓶颈才有收益。
         *   hiddenDim<=2048：亏 0.14MB(bs64) / 赢 0.37MB(h2048 k8 E32 bs1394 -12.5%)，取 0.30MB
         *   hiddenDim> 2048：亏 1.65MB(h6144 k8 E64 bs4096 +3.1%) / 赢 2.37MB(h7168 k6 E32 bs4096 -3.7%)，取 2.10MB
         * 总量判据（节省量 >= 60MB）在同一批实测点上错开 1 个、漏开 11 个（k16 族 bs256~1568 损失
         * 15%~40% 收益），故 4 卡不采用。其它卡数无实测数据，沿用 60MB 总量判据，待标定后再放开。
         */
        constexpr uint32_t DEDUP_GATE_CALIBRATED_WORLD = 4U;
        constexpr uint32_t DEDUP_GATE_COMM_BOUND_MAX_HIDDEN_DIM = 2048U;
        constexpr double DEDUP_GATE_MIN_SAVED_PER_EXPERT_COMM_BOUND = 0.30 * 1024.0 * 1024.0;
        constexpr double DEDUP_GATE_MIN_SAVED_PER_EXPERT_COMPUTE_BOUND = 2.10 * 1024.0 * 1024.0;
        constexpr double DEDUP_GATE_MIN_SAVED_BYTES_UNCALIBRATED = 60.0 * 1024.0 * 1024.0;
        double gateValue = savedBytes;
        double requiredValue = DEDUP_GATE_MIN_SAVED_BYTES_UNCALIBRATED;
        if (tilingData->compute.epWorldSize == DEDUP_GATE_CALIBRATED_WORLD) {
            gateValue = savedBytes / static_cast<double>(tilingData->compute.moeExpertPerRank);
            requiredValue = tilingData->compute.hiddenDim <= DEDUP_GATE_COMM_BOUND_MAX_HIDDEN_DIM ?
                                DEDUP_GATE_MIN_SAVED_PER_EXPERT_COMM_BOUND :
                                DEDUP_GATE_MIN_SAVED_PER_EXPERT_COMPUTE_BOUND;
        }
        if (gateValue < requiredValue) {
            OP_LOGI(nodeName,
                    "dedup gate: value %.0f < required %.0f (numMax=%ld, topK=%ld, h=%ld, hiddenDim=%ld, experts=%ld, "
                    "world=%ld, savedBytes=%.0f), disable dispatch dedup for this shape",
                    gateValue, requiredValue, tilingData->compute.numMaxTokensPerRank, tilingData->compute.topK,
                    tilingData->compute.h, tilingData->compute.hiddenDim, tilingData->compute.moeExpertPerRank,
                    tilingData->compute.epWorldSize, savedBytes);
            // 关掉 dispatch 去重、保留 combine 位的显式请求值：3(双开)->1(仅combine)，2(仅dispatch)->0(全关)。
            tilingData->compute.dedupMode = IsCombineDedupOn(tilingData->compute.dedupMode) ? 1 : 0;
        }
    }
    return;
}

/*
 * combine 去重 UB 容量护栏：独立 combine 阶段需要行环(≥2)+整行合并发送 ping/pong(×2) 共 4×行宽，
 * 外加固定 49152B（成员乒乓/fp32/累加段），装进 [64KiB,184KiB) 要求 h ≤ 9216
 * （4×ceilAlign(2h,32) ≤ 122880-49152，与 kernel InitWaveCombineBuffersDedup 的 fixedBytes 镜像；
 *  desc 批区与权重区在该窗口外的外移段，不占本窗口预算）。超界显式报错、不静默降级。
 */
static ge::graphStatus ApplyDedupCombineCapacityGate(MegaMoeTilingData* tilingData, const char* nodeName)
{
    if (!IsCombineDedupOn(tilingData->compute.dedupMode)) {
        return ge::GRAPH_SUCCESS;
    }
    constexpr int64_t DEDUP_COMBINE_MAX_H = 9216;
    OP_TILING_CHECK(static_cast<int64_t>(tilingData->compute.h) > DEDUP_COMBINE_MAX_H,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "combine_comm_mode", "1",
                                              "combine dedup requires h <= 9216 (UB capacity), "
                                              "set combine_comm_mode=0 for this shape"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace
} // namespace optiling

#endif // MEGA_MOE_TILING_CHECK_H
