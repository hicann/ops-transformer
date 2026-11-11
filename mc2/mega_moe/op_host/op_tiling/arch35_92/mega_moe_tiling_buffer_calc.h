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
 * \file mega_moe_tiling_buffer_calc.h
 * \brief
 */

#ifndef MEGA_MOE_TILING_BUFFER_CALC_H
#define MEGA_MOE_TILING_BUFFER_CALC_H

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <unordered_map>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_exception_dump.h"
#include "../mega_moe_tiling_host.h"
#include "../../../op_kernel/arch35_92/mega_moe_tiling.h"
#include "../../../op_kernel/arch35_92/mega_moe_arch35_92_tiling_key.h"
#include "../../../op_kernel/arch35_92/common/mega_moe_constants.h"
#include "../../../op_kernel/arch35_92/common/mega_moe_gm_layout.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
using namespace MegaMoeImpl;

namespace optiling {
namespace {

// MX 量化的 weight scale 每个专家末尾固定带 2 个 base 分量。
const static int64_t WEIGHT_SCALE_MULTI_BASE_DIM_SIZE = 2LL;
// 当前实现支持的共享专家数量上限。
const static int64_t MAX_SHARED_EXPERT_NUM = 4LL;
const static int64_t UB_BLOCK_SIZE = 32LL;

const static int64_t FOUR_DIMS = 4LL;
const static int64_t THREE_DIMS = 3LL;
const static int64_t TWO_DIMS = 2LL;
const static int64_t ONE_DIM = 1LL;
const static int64_t MIN_TOPK = 1LL;
const static int64_t MAX_TOPK = 32LL;
const static int64_t MIN_EXPERT_PER_RANK = 1LL;
const static int64_t MAX_EXPERT_PER_RANK = MEGA_MOE_MAX_EXPERT_PER_RANK;
const static int64_t MIN_H = 1024LL;
const static int64_t MAX_H = 8LL * 1024LL; // 8K
// MX 量化按每 32 个元素一组，仅用于 scales 的 shape 校验，与 H 的对齐要求无关
const static int64_t MX_SCALE_GROUP = 32LL;
const static int64_t MTE_H_ALIGN = 64LL;
const static int64_t URMA_H_ALIGN = 64LL;
const static int64_t MAX_HIDDEN_DIM = 8LL * 1024LL; // 8K
// hiddenDim 是 GMM1 的完整输出宽度；门控激活按 gate/up 划分，GELU 保留全宽。
// 256 对齐保证门控激活半宽按 128 对齐，下限 512 保证半宽至少覆盖一个完整 tile。
const static int64_t MIN_HIDDEN_DIM = 512LL;
const static int64_t HIDDEN_DIM_ALIGN = 256LL;
const static int64_t MIN_EP_WORLD_SIZE = 2LL;
const static int64_t MAX_MTE_EP_WORLD_SIZE = 1024LL;
const static int64_t MAX_URMA_EP_WORLD_SIZE = 1024LL;
const static int64_t MAX_MOE_EXPERT_NUM = MEGA_MOE_MAX_EXPERT_NUM;
const static int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
constexpr float DEFAULT_ACTIVATION_CLAMP = std::numeric_limits<float>::max();
constexpr float DEFAULT_SWIGLU_OAI_ALPHA = 1.702f;
constexpr float DEFAULT_SWIGLU_OAI_BETA = 1.0f;

constexpr uint32_t GMM_TILE_N = 256U;
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE = 4U;
constexpr int64_t GMM_TILE_STATUS_COUNT_ALIGN = 16LL;
const static uint32_t WEIGHT_MATRIX_ROW_DIM_INDEX = 0U;
const static uint32_t WEIGHT_MATRIX_COLUMN_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MATRIX_DIM_INDEX = 0U;
const static uint32_t WEIGHT_SCALE_GROUP_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MULTI_BASE_DIM_INDEX = 2U;

struct ExpertWeightInputIndices {
    uint32_t weightOne;
    uint32_t weightTwo;
    uint32_t weightScalesOne;
    uint32_t weightScalesTwo;
};

struct ExpertParams {
    ExpertWeightInputIndices inputs;
    const char* expertTypeName;
    int64_t expertCount;
    ge::DataType quantOutDtype;
    uint8_t gmmMode;
};

struct MegaMoeExpertParams {
    ExpertParams moe;
    ExpertParams shared;
    bool isPerExpertWeightTensor;
};

static const std::unordered_map<ge::DataType, int64_t> EXPERT_QUANT_MODE_MAP = {
    {ge::DT_FLOAT8_E5M2, EXPERT_QUANT_OUT_DTYPE_E5M2},
    {ge::DT_FLOAT8_E4M3FN, EXPERT_QUANT_OUT_DTYPE_E4M3FN},
    {ge::DT_FLOAT4_E2M1, EXPERT_QUANT_OUT_DTYPE_E2M1},
};

/*
 * 根据接口索引构造 MoE 与共享专家参数；这里只建立输入角色映射，不读取或校验 tensor。
 */
static MegaMoeExpertParams MakeExpertParams(const MegaMoeConfig& config)
{
    MegaMoeExpertParams params{};
    params.moe.inputs = {config.weight1Index, config.weight2Index, config.weightScales1Index,
                         config.weightScales2Index};
    params.moe.expertTypeName = "MoE expert";
    params.shared.inputs = {config.sharedWeight1Index, config.sharedWeight2Index, config.sharedWeightScales1Index,
                            config.sharedWeightScales2Index};
    params.shared.expertTypeName = "shared expert";
    return params;
}

static bool IsPreQuantizedXType(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT8_E5M2 || dataType == ge::DT_FLOAT8_E4M3FN || dataType == ge::DT_FLOAT4_E2M1;
}

/*
 * 统计指定动态输入 (tensor list) 中的 tensor 数量。
 */
static uint32_t GetDynamicInputTensorCount(const gert::TilingContext* context, uint32_t inputIndex)
{
    uint32_t tensorCount = 0;
    while (context->GetDynamicInputShape(inputIndex, tensorCount) != nullptr) {
        ++tensorCount;
    }
    return tensorCount;
}

/*
 * 按权重布局返回专家数：逐专家布局取 TensorList 长度，堆叠布局取首个 tensor 的 dim0。
 * 可选输入未传入或首维为 0 时返回 0。
 */
static int64_t GetWeightExpertCount(const gert::TilingContext* context, uint32_t inputIndex,
                                    bool isPerExpertWeightTensor)
{
    const auto* firstTensorShape = context->GetDynamicInputShape(inputIndex, 0);
    if (firstTensorShape == nullptr || firstTensorShape->GetStorageShape().GetDimNum() == 0U ||
        firstTensorShape->GetStorageShape().GetDim(0) <= 0) {
        return 0;
    }
    if (isPerExpertWeightTensor) {
        return static_cast<int64_t>(GetDynamicInputTensorCount(context, inputIndex));
    }
    return firstTensorShape->GetStorageShape().GetDim(0);
}

/*
 * 以单专家视图读取指定维度；堆叠布局会自动跳过最外层的专家维。
 */
static int64_t GetSingleExpertTensorDimSize(const gert::StorageShape* tensorShape, uint32_t singleExpertDimIndex,
                                            bool isPerExpertTensor)
{
    uint32_t expertDimOffset = isPerExpertTensor ? 0U : 1U;
    return tensorShape->GetStorageShape().GetDim(expertDimOffset + singleExpertDimIndex);
}

/*
 * 算去重的请求模式，取值含义与 tilingData->compute.dedupMode 相同：
 * 0=全关，1=仅 combine 开，2=仅 dispatch 开，3=双开。
 * dispatch 恒先请求（这个形状开了是否划算，由后面的收益门决定）；combine 完全听
 * combine_comm_mode 属性（1=强制开、0=强制关，门控无权改）。非 MTE 拓扑没有去重能力，
 * 直接返回 0（显式给 ccm=1 撞非 MTE 时由 CheckDedupRequestAttrs 报错）。
 * 无效值返回 -1，由 CheckDedupRequestAttrs 拦截。
 */
static int64_t GetDedupMode(const gert::TilingContext* context, const MegaMoeConfig& config)
{
    auto attrs = context->GetAttrs();
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrTopoTypeIndex));
    if (topoTypePtr != nullptr && *topoTypePtr != TOPO_TYPE_MTE) {
        return 0;
    }
    auto ccmPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCombineCommModeIndex));
    int64_t combineCommMode = ccmPtr == nullptr ? COMBINE_COMM_MODE_NO_DEDUP : *ccmPtr; // 旧序列化图无该属性时按默认 0
    if (combineCommMode != COMBINE_COMM_MODE_NO_DEDUP && combineCommMode != COMBINE_COMM_MODE_RANK_DEDUP) {
        return -1;
    }
    return combineCommMode == COMBINE_COMM_MODE_RANK_DEDUP ? 3 :
                                                             2; // dispatch 恒请求：ccm=1 双开(3)，否则仅 dispatch(2)
}

/*
 * 将专家权重类型、专家量化输出类型和通信拓扑编码为 kernel tiling key。
 */
static uint64_t CalcTilingKey(const gert::TilingContext* context, const MegaMoeExpertParams& expertParams,
                              const MegaMoeTilingData* tilingData)
{
    auto moeWeightDesc = context->GetDynamicInputDesc(expertParams.moe.inputs.weightOne, 0);
    auto sharedWeightDesc = expertParams.shared.expertCount > 0 ?
                                context->GetDynamicInputDesc(expertParams.shared.inputs.weightOne, 0) :
                                moeWeightDesc;

    int64_t topoType = TILINGKEY_TPL_MTE;
    if (IsLayeredTopology(tilingData->compute.topoType)) {
        topoType = TILINGKEY_TPL_URMA;
    }
    return GET_TPL_TILING_KEY(static_cast<int64_t>(moeWeightDesc->GetDataType()),
                              static_cast<int64_t>(sharedWeightDesc->GetDataType()),
                              EXPERT_QUANT_MODE_MAP.at(expertParams.moe.quantOutDtype),
                              EXPERT_QUANT_MODE_MAP.at(expertParams.shared.quantOutDtype), topoType);
}

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
/*
 * 将 MegaMoe 运行时异常转交 MC2 通用异常处理实现。
 */
static inline void MegaMoeExceptionImplWrapper(aclrtExceptionInfo* args, void* userdata)
{
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MegaMoe");
}

/*
 * 在动态库加载时校验版本并注册 MegaMoe 异常处理回调。
 */
static void RegisterMegaMoeException()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;

    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MegaMoe", "Get runtime version failed when register exception func.");
        return;
    }
    if (aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MegaMoe", "Get metadef version failed when register exception func.");
        return;
    }

    if (runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MegaMoe",
                "The runtime(%d) or metadata(%d) version is lower than the version(%d) supporting exception func.",
                runtimeVersionNum, metadefVersionNum, EXCEPTION_DUMP_SUPPORT_VERSION);
        return;
    }

    IMPL_OP(MegaMoe).ExceptionDumpParseFunc(MegaMoeExceptionImplWrapper);
}
#endif

/* 按两段 GMM 的 N 维分核计算每个 wave 的 M 分组数。 */
static void SetWaveParams(MegaMoeTilingData* tilingData, const uint32_t aicNum)
{
    tilingData->compute.mGroupsPerWave =
        static_cast<uint32_t>(CalcMemoryMGroupsPerWave(tilingData->compute.hiddenDim, tilingData->compute.h, aicNum));
}

/*
 * 计算一个 token 的交织量化记录大小。dispatch peermem、SendMask UB 和共享专家量化都必须复用该口径。
 */
static uint32_t CalcQuantTokenAndScaleBytes(const MegaMoeTilingData* tilingData, uint32_t activationElementsPerByte)
{
    uint32_t quantTokenBytes = ops::CeilAlign(static_cast<uint32_t>(tilingData->compute.h) / activationElementsPerByte,
                                              static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes =
        ops::CeilAlign(ops::CeilDiv(static_cast<uint32_t>(tilingData->compute.h), static_cast<uint32_t>(ALIGN_32)) *
                           static_cast<uint32_t>(sizeof(int8_t)),
                       static_cast<uint32_t>(ALIGN_32));
    return quantTokenBytes + quantScaleAlignBytes;
}

/*
 * 计算每个 dispatch ring slot 的量化 token 与 scale 拷贝区字节数。记录带权重段（tkw=1 或 combine
 * 去重开）时再追加，该判定式 host 与 kernel 各用点必须保持同式，否则记录布局两侧错位。
 */
static uint32_t CalcDispatchCopyBufferBytes(const MegaMoeTilingData* tilingData, uint32_t activationElementsPerByte)
{
    uint32_t copyBufferBytes = CalcQuantTokenAndScaleBytes(tilingData, activationElementsPerByte);
    if (tilingData->isTopkWeightsPrefetch || IsCombineDedupOn(tilingData->compute.dedupMode) ||
        IsServerCombineDedupOn(tilingData->compute.dedupMode)) {
        uint32_t weightBytes = ops::CeilAlign(static_cast<uint32_t>(tilingData->compute.topK * sizeof(float)),
                                              static_cast<uint32_t>(ALIGN_32));
        copyBufferBytes += weightBytes;
    }
    return copyBufferBytes;
}

/*
 * 分两阶段确定 dispatch UB 配置：首先以基准 route batch 计算最大 ring 深度，
 * 随后在 ring 深度固定后利用剩余 UB 扩大 route batch，以减少批次数量。
 * 预算减法采用饱和语义以避免无符号回绕；bufferCount 必须依次应用上限和下限约束。
 */
static void SelectDispatchRingAndRouteBatch(MegaMoeDispatchBufferConfig& bufferConfig, uint64_t sendTotalNum,
                                            uint64_t alignedTotalRouteItems, uint32_t fixedBufferBytes,
                                            uint32_t dispatchSlotBytes, uint32_t topkIndexTypeBytes,
                                            uint32_t availableUbBytes)
{
    // 第一阶段：使用基准 batch 确定 ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_RECV_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>(ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));

    // MTE 接收侧只保留一个当前编码宽度的 topK 有效下标 batch。
    uint32_t routeIndexBufferBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * topkIndexTypeBytes;
    uint32_t bytesWithoutDispatchSlots = fixedBufferBytes + routeIndexBufferBytes;
    // dispatchSlotBudgetBytes 是扣除非 ring tensor 后可用于分配 ring slot 的 UB。
    uint32_t dispatchSlotBudgetBytes =
        availableUbBytes > bytesWithoutDispatchSlots ? availableUbBytes - bytesWithoutDispatchSlots : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(dispatchSlotBudgetBytes / dispatchSlotBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_DISPATCH_BUFFER_COUNT);
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_DISPATCH_BUFFER_COUNT);

    // 第二阶段：在 ring 深度固定后使用剩余 UB 扩大 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithDispatchSlots 包含固定 tensor 和已选中的 dispatch slot。
        uint32_t fixedBytesWithDispatchSlots =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * dispatchSlotBytes;
        // routeItemBudgetBytes 是有效下标 tensor 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithDispatchSlots ? availableUbBytes - fixedBytesWithDispatchSlots : 0U;
        uint32_t expandedRouteItems = routeItemBudgetBytes / topkIndexTypeBytes;
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>(
                ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
        }
    }
}

static uint64_t CalcTopkValidIndexRingSlotBytes(uint32_t routeItemsPerBatch, uint32_t topK, uint32_t topkIndexTypeBytes)
{
    // batch 可能从某个 token 的 topK 段中间开始，slot 按该 batch 可跨越的 token 数上界预留。
    uint64_t maxMatchedRouteItems =
        ops::CeilDiv(static_cast<uint64_t>(routeItemsPerBatch) + topK - 1U, static_cast<uint64_t>(topK));
    uint64_t validIndexBytes =
        ops::CeilAlign(maxMatchedRouteItems * topkIndexTypeBytes, static_cast<uint64_t>(ALIGN_32));
    return static_cast<uint64_t>(routeItemsPerBatch) / BITS_PER_BYTE + validIndexBytes;
}

/*
 * 针对给定的单核专家数，计算 topK 有效下标发送阶段的路由分批和 ring 配置。
 */
static MegaMoeSendMaskBufferConfig CalcTopkValidIndexBufferConfig(const MegaMoeTilingData* tilingData,
                                                                  uint32_t fixedBufferBytes, uint32_t ownedExpertCount,
                                                                  uint32_t availableUbBytes)
{
    MegaMoeSendMaskBufferConfig bufferConfig{};
    // 发送批网格按 numMaxTokensPerRank * topK 容量上界划分，kernel 再按实际 bs * topK 裁剪。
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->compute.numMaxTokensPerRank) * tilingData->compute.topK;
    uint64_t alignedTotalRouteItems = ops::CeilAlign(sendTotalNum, static_cast<uint64_t>(ALIGN_256));
    uint32_t topkIndexTypeBytes = static_cast<uint32_t>(
        CalcTopkIndexTypeBytes(tilingData->compute.numMaxTokensPerRank, tilingData->compute.topK));

    // 第一阶段：使用基准 batch 确定 route ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_SEND_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>(ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
    bufferConfig.bufferBytes = static_cast<uint32_t>(CalcTopkValidIndexRingSlotBytes(
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch), tilingData->compute.topK, topkIndexTypeBytes));

    // topkIdsTensor 保持 int32；生成及 gather 后发送的 index 根据容量使用 int16 或 int32。
    uint32_t topkIdsBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t topkIndexBufferBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * topkIndexTypeBytes;
    uint32_t bytesWithoutRouteBuffers = fixedBufferBytes + topkIdsBufferBytes + topkIndexBufferBytes;
    uint32_t routeBufferBudgetBytes =
        availableUbBytes > bytesWithoutRouteBuffers ? availableUbBytes - bytesWithoutRouteBuffers : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(routeBufferBudgetBytes / bufferConfig.bufferBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_SEND_MASK_BUFFER_COUNT);

    uint64_t routePushCount = static_cast<uint64_t>(bufferConfig.routeBatchCount) * ownedExpertCount;
    if (routePushCount > 0U && static_cast<uint64_t>(bufferConfig.bufferCount) > routePushCount) {
        bufferConfig.bufferCount = static_cast<int32_t>(routePushCount);
    }
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);

    // 第二阶段：固定 ring 深度，用剩余 UB 扩大 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        uint64_t fixedBytesWithRoutePadding =
            static_cast<uint64_t>(fixedBufferBytes) +
            static_cast<uint64_t>(bufferConfig.bufferCount) * (ALIGN_32 + 2U * topkIndexTypeBytes);
        uint64_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithRoutePadding ? availableUbBytes - fixedBytesWithRoutePadding : 0U;
        uint64_t expandedRouteItems =
            routeItemBudgetBytes * BITS_PER_BYTE /
            ((sizeof(int32_t) + topkIndexTypeBytes) * BITS_PER_BYTE + static_cast<uint64_t>(bufferConfig.bufferCount) +
             ops::CeilDiv(static_cast<uint64_t>(bufferConfig.bufferCount) * topkIndexTypeBytes * BITS_PER_BYTE,
                          static_cast<uint64_t>(tilingData->compute.topK)));
        expandedRouteItems = expandedRouteItems / ALIGN_256 * ALIGN_256;
        expandedRouteItems = std::min(expandedRouteItems, alignedTotalRouteItems);
        if (expandedRouteItems > static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>(
                ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
            bufferConfig.bufferBytes = static_cast<uint32_t>(CalcTopkValidIndexRingSlotBytes(
                static_cast<uint32_t>(expandedRouteItems), tilingData->compute.topK, topkIndexTypeBytes));
        }
    }
    return bufferConfig;
}

enum class UnpermuteBufferPolicy : uint8_t {
    FUSED_ACCUMULATION,
    SEPARATE_DEQUANTIZATION
};

/*
 * 计算 unpermute 的 slot 与 scale 尺寸：单 token 的 BF16 搬入区 + FP32 计算区，以及 combine quant 的 scale 展开区。
 * dataSlotBytes 与 scaleBytes 经出参带出，供后续两个阶段共用。
 */
static void CalcMteUnpermuteSlotAndScaleBytes(const MegaMoeTilingData* tilingData,
                                              MegaMoeUnpermuteBufferConfig& bufferConfig, uint32_t& dataSlotBytes,
                                              uint32_t& scaleBytes, UnpermuteBufferPolicy bufferPolicy)
{
    uint32_t bf16SlotBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(tilingData->compute.h) * sizeof(uint16_t), static_cast<uint64_t>(ALIGN_32)));
    uint32_t fp32SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->compute.h) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    // dataSlotBytes 是同一 token 的 BF16 搬入区和 FP32 计算区之和。
    dataSlotBytes = bf16SlotBytes + fp32SlotBytes;
    bufferConfig.bf16SlotElementCount = bf16SlotBytes / sizeof(uint16_t);
    bufferConfig.fp32SlotElementCount = fp32SlotBytes / sizeof(float);

    // scaleBytes 是 combine quant 使用的 BF16/FP32 scale 展开区大小。
    scaleBytes = 0U;
    if (bufferPolicy == UnpermuteBufferPolicy::SEPARATE_DEQUANTIZATION &&
        tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleElementCount = (tilingData->compute.h + ALIGN_32 - 1U) / ALIGN_32;
        scaleBytes = static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(uint16_t) * DEQUANT_BF16_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32))) +
                     static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32)));
    }
}

/*
 * 按 weight 元素数量更新 unpermute weight 区大小：始终分配 FP32 主区，
 * 仅在需要数据类型转换时分配转换暂存区。初始配置和扩批后均复用该计算。
 */
static void SetUnpermuteWeightBufferBytes(MegaMoeUnpermuteBufferConfig& bufferConfig, uint32_t weightElementCount,
                                          uint32_t topKWeightsConversionElementBytes,
                                          uint32_t conversionAlignmentBytes = static_cast<uint32_t>(ALIGN_32))
{
    bufferConfig.topKWeightsBufferBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    if (topKWeightsConversionElementBytes > 0U) {
        bufferConfig.topKWeightsConversionBufferBytes = static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * topKWeightsConversionElementBytes,
                           static_cast<uint64_t>(conversionAlignmentBytes)));
    }
}

/*
 * 根据单核 token 数和 topK weight 转换需求，计算 unpermute 阶段的分批及 UB 布局。
 */
static MegaMoeUnpermuteBufferConfig CalcMteUnpermuteBufferConfig(const MegaMoeTilingData* tilingData,
                                                                 uint32_t coreTokenCount,
                                                                 uint32_t topKWeightsConversionElementBytes,
                                                                 uint32_t availableUbBytes,
                                                                 UnpermuteBufferPolicy bufferPolicy)
{
    MegaMoeUnpermuteBufferConfig bufferConfig{};
    if (coreTokenCount == 0U) {
        return bufferConfig;
    }

    // 当前 token 的 rank 表独立放在 UB 尾部，两阶段预算均先扣除该固定区域。
    const uint32_t dataUbBudgetBytes = availableUbBytes - UNPERMUTE_EXPERT_RANKS_UB_BYTES;
    uint32_t dataSlotBytes = 0U;
    uint32_t scaleBytes = 0U;
    CalcMteUnpermuteSlotAndScaleBytes(tilingData, bufferConfig, dataSlotBytes, scaleBytes, bufferPolicy);
    // 累加/输出区固定为 BF16 + FP32；Arch35 输入槽只需保留搬入区。
    const uint32_t inputSlotBytes = bufferPolicy == UnpermuteBufferPolicy::FUSED_ACCUMULATION ?
                                        bufferConfig.bf16SlotElementCount * sizeof(uint16_t) :
                                        dataSlotBytes;

    // 第一阶段：按基准 weight batch 确定每批 token 数和 weight 区，再以剩余 UB 分配输入 ring。
    uint32_t baseTokensPerBatch = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / tilingData->compute.topK;
    bufferConfig.tokensPerBatch = static_cast<int32_t>(std::min(baseTokensPerBatch, coreTokenCount));
    uint32_t weightElementCount = static_cast<uint32_t>(bufferConfig.tokensPerBatch) * tilingData->compute.topK;
    SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);

    // bytesBeforeInputBuffers 包含 weight、scale 和一个累加/输出 data slot。
    uint32_t bytesBeforeInputBuffers = bufferConfig.topKWeightsBufferBytes +
                                       bufferConfig.topKWeightsConversionBufferBytes + scaleBytes + dataSlotBytes;
    uint32_t inputBufferBudgetBytes =
        dataUbBudgetBytes > bytesBeforeInputBuffers ? dataUbBudgetBytes - bytesBeforeInputBuffers : 0U;
    bufferConfig.inputBufferCount = static_cast<int32_t>(inputBufferBudgetBytes / inputSlotBytes);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    int32_t accumulationItemCount =
        bufferConfig.tokensPerBatch *
        static_cast<int32_t>(tilingData->compute.topK + tilingData->compute.sharedExpertNum);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, accumulationItemCount);
    bufferConfig.inputBufferCount = std::max(bufferConfig.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);

    // 第二阶段：ring 深度固定后，使用剩余 UB 扩大 weight batch。
    if (baseTokensPerBatch < coreTokenCount) {
        // fixedBytes 包含 scale、一个累加/输出 slot 和已经选中的所有输入 slot。
        uint32_t fixedBytes =
            scaleBytes + dataSlotBytes + static_cast<uint32_t>(bufferConfig.inputBufferCount) * inputSlotBytes;
        // weightBudgetBytes 是 FP32 weight 及可选转换中转区可使用的 UB。
        uint32_t weightBudgetBytes = dataUbBudgetBytes - fixedBytes - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES;
        uint32_t weightBytesPerToken =
            tilingData->compute.topK * (static_cast<uint32_t>(sizeof(float)) + topKWeightsConversionElementBytes);
        uint32_t expandedTokensPerBatch = std::min(weightBudgetBytes / weightBytesPerToken, coreTokenCount);
        if (expandedTokensPerBatch > static_cast<uint32_t>(bufferConfig.tokensPerBatch)) {
            bufferConfig.tokensPerBatch = static_cast<int32_t>(expandedTokensPerBatch);
            weightElementCount = expandedTokensPerBatch * tilingData->compute.topK;
            SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);
        }
    }
    return bufferConfig;
}

/* 初始化槽尺寸，计算单个输入槽及固定区（输出、累加和 scale 暂存）的字节数。 */
static void CalcUrmaUnpermuteSlotAndScaleBytes(const MegaMoeTilingData* tilingData,
                                               MegaMoeUnpermuteBufferConfig& bufferConfig, uint32_t& inputSlotBytes,
                                               uint32_t& fixedBytes)
{
    const uint32_t bf16SlotBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(tilingData->compute.h) * sizeof(uint16_t), static_cast<uint64_t>(ALIGN_32)));
    const uint32_t fp32SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->compute.h) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    bufferConfig.bf16SlotElementCount = bf16SlotBytes / sizeof(uint16_t);
    bufferConfig.fp32SlotElementCount = fp32SlotBytes / sizeof(float);
    inputSlotBytes = bf16SlotBytes;
    uint32_t scaleBytes = 0U;
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        inputSlotBytes += fp32SlotBytes;
        const uint32_t scaleNum =
            ops::CeilDiv(static_cast<uint32_t>(tilingData->compute.h), static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        constexpr uint32_t INTERLEAVED_REGISTER_COUNT = 2U;
        // 与 layered DeQuantMxFp8 的完整向量写入范围一致，每阶段对齐后再展开两份。
        scaleBytes = static_cast<uint32_t>(ops::CeilAlign(static_cast<uint64_t>(scaleNum) * sizeof(uint16_t),
                                                          static_cast<uint64_t>(ALIGN_256))) *
                         INTERLEAVED_REGISTER_COUNT +
                     static_cast<uint32_t>(
                         ops::CeilAlign(static_cast<uint64_t>(scaleNum) * DEQUANT_BF16_SCALE_EXPANSION * sizeof(float),
                                        static_cast<uint64_t>(ALIGN_256))) *
                         INTERLEAVED_REGISTER_COUNT;
    }
    fixedBytes = bf16SlotBytes + fp32SlotBytes + scaleBytes;
}

/* 输入 ring 已固定，按剩余权重预算扩大 batch，并更新两块权重区大小。 */
static void ExpandUrmaUnpermuteWeightBatch(const MegaMoeTilingData* tilingData,
                                           MegaMoeUnpermuteBufferConfig& bufferConfig, uint32_t coreTokenCount,
                                           uint32_t topKWeightsConversionElementBytes, uint32_t weightBudgetBytes)
{
    const uint32_t weightBytesPerToken =
        tilingData->compute.topK * (static_cast<uint32_t>(sizeof(float)) + topKWeightsConversionElementBytes);
    const uint32_t expandedTokensPerBatch = std::min(weightBudgetBytes / weightBytesPerToken, coreTokenCount);
    if (expandedTokensPerBatch > static_cast<uint32_t>(bufferConfig.tokensPerBatch)) {
        bufferConfig.tokensPerBatch = static_cast<int32_t>(expandedTokensPerBatch);
        SetUnpermuteWeightBufferBytes(bufferConfig, expandedTokensPerBatch * tilingData->compute.topK,
                                      topKWeightsConversionElementBytes, static_cast<uint32_t>(ALIGN_128));
    }
}

/* URMA 先预留基准权重 batch，再分配输入 ring，最后用剩余 UB 扩大权重 batch。 */
static MegaMoeUnpermuteBufferConfig CalcUrmaUnpermuteBufferConfig(const MegaMoeTilingData* tilingData,
                                                                  uint32_t coreTokenCount,
                                                                  uint32_t topKWeightsConversionElementBytes,
                                                                  uint32_t availableUbBytes)
{
    MegaMoeUnpermuteBufferConfig bufferConfig{};
    if (coreTokenCount == 0U) {
        return bufferConfig;
    }

    uint32_t inputSlotBytes = 0U;
    uint32_t fixedBytes = 0U;
    CalcUrmaUnpermuteSlotAndScaleBytes(tilingData, bufferConfig, inputSlotBytes, fixedBytes);
    // Dispatch 已结束，来源就绪缓存占用的 UB 可由本阶段复用。
    const uint32_t usableUbBytes = std::min(availableUbBytes, LAYERED_USABLE_UB_BYTES);

    // 先保证基准权重 batch，再用剩余 UB 选取输入 ring 深度。
    const uint32_t baseTokensPerBatch = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / tilingData->compute.topK;
    bufferConfig.tokensPerBatch = static_cast<int32_t>(std::min(baseTokensPerBatch, coreTokenCount));
    SetUnpermuteWeightBufferBytes(bufferConfig, bufferConfig.tokensPerBatch * tilingData->compute.topK,
                                  topKWeightsConversionElementBytes, static_cast<uint32_t>(ALIGN_128));
    const uint32_t bytesBeforeInputBuffers =
        fixedBytes + bufferConfig.topKWeightsBufferBytes + bufferConfig.topKWeightsConversionBufferBytes;
    const uint32_t inputBudgetBytes =
        usableUbBytes > bytesBeforeInputBuffers ? usableUbBytes - bytesBeforeInputBuffers : 0U;
    const uint32_t accumulationItemCount = static_cast<uint32_t>(bufferConfig.tokensPerBatch) *
                                           (tilingData->compute.topK + tilingData->compute.sharedExpertNum);
    bufferConfig.inputBufferCount = static_cast<int32_t>(
        std::min(inputBudgetBytes / inputSlotBytes, static_cast<uint32_t>(MAX_UNPERMUTE_INPUT_BUFFER_COUNT)));
    bufferConfig.inputBufferCount =
        std::min(bufferConfig.inputBufferCount, static_cast<int32_t>(accumulationItemCount));
    // 当前合法 H 和基准权重 batch 的最坏布局可容纳双 buffer。
    bufferConfig.inputBufferCount = std::max(bufferConfig.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);

    // ring 深度确定后，将剩余 UB 用于扩大权重 batch。
    if (baseTokensPerBatch < coreTokenCount) {
        // FP32 权重按 32B 对齐；BF16 转换输入每轮完整读取 128B，区内覆盖尾读范围。
        const uint32_t weightAlignmentReserveBytes =
            static_cast<uint32_t>(ALIGN_32) +
            (topKWeightsConversionElementBytes > 0U ? static_cast<uint32_t>(ALIGN_128) : 0U);
        const uint32_t fixedUbBytes =
            fixedBytes + bufferConfig.inputBufferCount * inputSlotBytes + weightAlignmentReserveBytes;
        const uint32_t weightBudgetBytes = usableUbBytes > fixedUbBytes ? usableUbBytes - fixedUbBytes : 0U;
        ExpandUrmaUnpermuteWeightBatch(tilingData, bufferConfig, coreTokenCount, topKWeightsConversionElementBytes,
                                       weightBudgetBytes);
    }
    return bufferConfig;
}

/*
 * 计算预量化一个 stage slot 所需的 UB。kernel 的 x/scale/weight 阶段共用同一 slot，
 * stage 中保存完整 dispatch record，token、scale 和 weight 按记录布局相加；BF16 topK weight
 * 的 Cast 输入与输出需要同时保留额外的源数据区。
 */
static uint32_t CalcPreQuantStageBytes(const MegaMoeTilingData* tilingData, uint32_t activationElementsPerByte,
                                       uint32_t tileTokens, ge::DataType topKWeightsDataType)
{
    // 保存完整的 dispatch 记录，使 MTE3 能一次写出整个 tile。
    uint64_t stageBytes =
        static_cast<uint64_t>(CalcDispatchCopyBufferBytes(tilingData, activationElementsPerByte)) * tileTokens;
    if (tilingData->isTopkWeightsPrefetch && topKWeightsDataType == ge::DT_BF16) {
        uint64_t weightSrcRowBytes =
            ops::CeilAlign(static_cast<uint64_t>(tilingData->compute.topK) * ge::GetSizeByDataType(topKWeightsDataType),
                           static_cast<uint64_t>(ALIGN_32));
        stageBytes += weightSrcRowBytes * tileTokens;
    }
    return static_cast<uint32_t>(ops::CeilAlign(stageBytes, static_cast<uint64_t>(ALIGN_512)));
}

/*
 * 计算预量化 tile 的 B 和双 buffer stage 空间。route ring 的核数及 UB 配置
 * 由 SetTopkValidIndexBufferConfigs 统一处理。
 */
static void SetPreQuantTileConfig(MegaMoeTilingData* tilingData, bool isPreQuantizedX,
                                  uint32_t activationElementsPerByte, ge::DataType topKWeightsDataType)
{
    tilingData->preQuantTileTokens = 0U;
    tilingData->preQuantStageBytes = 0U;
    if (!isPreQuantizedX || tilingData->compute.h == 0U) {
        return;
    }

    // PrepareLocalTokens 按 blockAivNum 个 AIV 分工，stage 不需要为 route 上界之外的 token 预留空间。
    uint32_t maxTokensPerAiv =
        ops::CeilDiv(static_cast<uint32_t>(tilingData->compute.bs), std::max(tilingData->blockAivNum, 1U));
    // 仅以单个 token 的 scale 行宽作为 B 的性能基准。
    uint32_t scaleRowBytes = static_cast<uint32_t>(
        ops::CeilAlign(ops::CeilDiv(static_cast<uint64_t>(tilingData->compute.h), static_cast<uint64_t>(ALIGN_32)),
                       static_cast<uint64_t>(ALIGN_32)));
    uint32_t scaleBasedTokens = std::max(1U, static_cast<uint32_t>(ops::CeilDiv(static_cast<uint64_t>(ALIGN_512),
                                                                                static_cast<uint64_t>(scaleRowBytes))));
    uint32_t tileTokens = std::min(scaleBasedTokens, maxTokensPerAiv);
    tileTokens = std::max(tileTokens, 1U);

    tilingData->preQuantTileTokens = tileTokens;
    tilingData->preQuantStageBytes =
        CalcPreQuantStageBytes(tilingData, activationElementsPerByte, tileTokens, topKWeightsDataType);
}

/*
 * 设置 Unpermute 的完整 chunk 与 tail chunk 两套 UB 配置，并记录完整 chunk 对应的 core 数。
 */
static void SetMteUnpermuteBufferConfigs(MegaMoeTilingData* tilingData, ge::DataType topKWeightsDataType,
                                         uint32_t availableUbBytes, UnpermuteBufferPolicy bufferPolicy)
{
    /*
     * 对应 MTE 的 UnpermuteTokens：TilingByJobContext(..., align=1) 按 ceil(bs / blockAivNum)
     * 连续分块，最后一个活跃任务可能只有 tail，后续空任务不进入 token 循环。
     * CreateTokenUnpermuteBuffers 按 jobIndex < fullTokenChunkJobCount 选择 full/tail 配置。
     * 修改分核方式时，需同步更新这里的 chunk 推导和上述配置选择条件。
     */
    uint32_t topKWeightsConversionElementBytes =
        topKWeightsDataType == ge::DT_FLOAT ? 0U : static_cast<uint32_t>(ge::GetSizeByDataType(topKWeightsDataType));
    uint32_t fullTokenChunkSize = ops::CeilDiv(static_cast<uint32_t>(tilingData->compute.bs), tilingData->blockAivNum);
    uint32_t activeCoreCount = ops::CeilDiv(static_cast<uint32_t>(tilingData->compute.bs), fullTokenChunkSize);
    uint32_t tailTokenChunkSize = tilingData->compute.bs - (activeCoreCount - 1U) * fullTokenChunkSize;
    bool tailIsFullTokenChunk = tailTokenChunkSize == fullTokenChunkSize;
    tilingData->unpermuteFullTokenChunkCoreCount = tailIsFullTokenChunk ? activeCoreCount : activeCoreCount - 1U;
    tilingData->unpermuteConfigForFullTokenChunk = CalcMteUnpermuteBufferConfig(
        tilingData, fullTokenChunkSize, topKWeightsConversionElementBytes, availableUbBytes, bufferPolicy);
    tilingData->unpermuteConfigForTailTokenChunk =
        tailIsFullTokenChunk ?
            MegaMoeUnpermuteBufferConfig{} :
            CalcMteUnpermuteBufferConfig(tilingData, tailTokenChunkSize, topKWeightsConversionElementBytes,
                                         availableUbBytes, bufferPolicy);
}

/* 与普通 Layered Unpermute 的 GetBalancedWorkRange 分核对应。 */
static void SetUrmaUnpermuteBufferConfigs(MegaMoeTilingData* tilingData, ge::DataType topKWeightsDataType,
                                          uint32_t availableUbBytes)
{
    const uint32_t topKWeightsConversionElementBytes =
        topKWeightsDataType == ge::DT_FLOAT ? 0U : static_cast<uint32_t>(ge::GetSizeByDataType(topKWeightsDataType));
    const uint32_t baseTokenCount = tilingData->compute.bs / tilingData->blockAivNum;
    const uint32_t extraTokenCoreCount = tilingData->compute.bs % tilingData->blockAivNum;
    const uint32_t fullTokenChunkSize = baseTokenCount + static_cast<uint32_t>(extraTokenCoreCount != 0U);
    tilingData->unpermuteFullTokenChunkCoreCount =
        extraTokenCoreCount != 0U ? extraTokenCoreCount : tilingData->blockAivNum;
    tilingData->unpermuteConfigForFullTokenChunk = CalcUrmaUnpermuteBufferConfig(
        tilingData, fullTokenChunkSize, topKWeightsConversionElementBytes, availableUbBytes);
    tilingData->unpermuteConfigForTailTokenChunk =
        extraTokenCoreCount == 0U ? MegaMoeUnpermuteBufferConfig{} :
                                    CalcUrmaUnpermuteBufferConfig(tilingData, baseTokenCount,
                                                                  topKWeightsConversionElementBytes, availableUbBytes);
}

/* URMA 按均分 token 配置 full/tail，MTE 保留原有分核和缓冲策略。 */
static void SetUnpermuteBufferConfigs(MegaMoeTilingData* tilingData, ge::DataType topKWeightsDataType,
                                      uint32_t availableUbBytes, UnpermuteBufferPolicy bufferPolicy)
{
    if (IsLayeredTopology(tilingData->compute.topoType)) {
        SetUrmaUnpermuteBufferConfigs(tilingData, topKWeightsDataType, availableUbBytes);
    } else {
        SetMteUnpermuteBufferConfigs(tilingData, topKWeightsDataType, availableUbBytes, bufferPolicy);
    }
}

/*
 * 仅申请框架 LibAPI 需要的系统 workspace，算子中间数据使用 CCL。
 */
static ge::graphStatus SetWorkspace(gert::TilingContext* context, const char* nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"), return ge::GRAPH_FAILED);

    workspace[0] = sysWorkspaceSize;

    OP_LOGD(nodeName, "sysWorkspaceSize: %ld \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "mega_moe_tiling workspaceSize: %ld \n", sysWorkspaceSize);

    return ge::GRAPH_SUCCESS;
}

} // namespace
} // namespace optiling

#endif // MEGA_MOE_TILING_BUFFER_CALC_H
