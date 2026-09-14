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
 * \file all_gather_matmul_v2_tiling.cpp
 * \brief
 */

#include "vector"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "mc2_log.h"
#include "tiling_func.h"
#include "../../../op_kernel/all_gather_matmul_aiv_mode_tiling.h"
#include "../../../op_kernel/all_gather_matmul_v2_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace {
const char *K_INNER_DEBUG = "AllGatherMatmulAIVMode Tiling Debug";
constexpr uint32_t ALLGATHER_CORENUM_SIXTEEN = 16;
constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_IS_TRANS_X1 = 1;
constexpr uint32_t ATTR_IS_TRANS_X2 = 2;
constexpr uint64_t INIT_TILINGKEY = 10000U;
constexpr uint32_t TILINGKEY_TRANS_B = 10U;
constexpr uint32_t A_INDEX = 0;
constexpr uint32_t B_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t X1_SCALE_INDEX = 3;
constexpr uint32_t X2_SCALE_INDEX = 4;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t USER_WORKSPACE_A2 = 1 * 1024 * 1024; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
constexpr uint64_t CCL_BUFFER_MIN_BYTES = 200ULL * 1024 * 1024; // 校验HCCL BUFF空间大小
constexpr uint64_t MB_BYTES = 1024ULL * 1024;
} // namespace

namespace optiling {

void AllGatherV2DecodeTilingData(int32_t code, CoCTiling &tilingData)
{
    uint32_t packedCode = static_cast<uint32_t>(code);
    tilingData.commDataSplit = packedCode & COMMDATASPLIT_MASK;
    packedCode >>= COMMDATASPLIT_BNUM;
    tilingData.commNpuSplit = packedCode & COMMNPUSPLIT_MASK;
    packedCode >>= COMMNPUSPLIT_BNUM;
    tilingData.commDirect = packedCode & COMMDIRECT_MASK;
    packedCode >>= COMMDIRECT_BNUM;
    tilingData.ubMoveNum = (packedCode & UBMOVENUM_MASK) * HALF_KBYTE;
    packedCode >>= UBMOVENUM_BNUM;
    tilingData.pValue = packedCode & PVALUE_MASK;
    packedCode >>= PVALUE_BNUM;
    tilingData.swizzlCount = packedCode & SWIZZLCOUNT_MASK;
    packedCode >>= SWIZZLCOUNT_BNUM;
    tilingData.swizzlDirect = packedCode & SWIZZLDIRECT_MASK;
    packedCode >>= SWIZZLDIRECT_BNUM;
    tilingData.m0 = (packedCode & M0_MASK) * DEFAULT_ROW + DEFAULT_ROW;
    tilingData.k0 = DEFAULT_COL;
    tilingData.n0 = tilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
    tilingData.mLoop = CeilDev(tilingData.m, tilingData.m0);
    tilingData.nLoop = CeilDev(tilingData.n, tilingData.n0);
    tilingData.kLoop = CeilDev(tilingData.k, tilingData.k0);
}

int32_t GetValueFromMKNConditionMapAllGather(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
                                             std::map<int, std::vector<std::vector<int>>> conditionMap)
{
    int32_t value = defaultValue;
    for (auto &item : conditionMap) {
        for (auto &condition : item.second) {
            bool inRange = m > condition[CONDITION_M_ST] && m <= condition[CONDITION_M_END] &&
                           k > condition[CONDITION_K_ST] && k <= condition[CONDITION_K_END] &&
                           n > condition[CONDITION_N_ST] && n <= condition[CONDITION_N_END];
            if (inRange) {
                return item.first;
            }
        }
    }
    return value;
}

static void GetTilingKey(uint64_t &tilingKey, const AllGatherMatmulAIVModeInfo &info,
                         const gert::TilingContext *context)
{
    const gert::StorageShape *matrixBias = context->GetOptionalInputShape(BIAS_INDEX);
    bool isBias = (matrixBias == nullptr) ? false : true;
    tilingKey = GET_TPL_TILING_KEY(isBias, info.isTransposeX1, info.isTransposeX2);
}

void SetTilingParam(CoCTiling &cocTilingData, const std::map<int *, TilingValue> &TilingParamMap)
{
    int32_t m = cocTilingData.m;
    int32_t k = cocTilingData.k;
    int32_t n = cocTilingData.n;

    for (auto &item : TilingParamMap) {
        auto value = item.second.value;
        auto conditionMap = item.second.conditionMap;
        if (!conditionMap.empty()) {
            *item.first = GetValueFromMKNConditionMapAllGather(m, k, n, value, conditionMap);
        } else if (value != -1) {
            *item.first = value;
        }
    }

    cocTilingData.ubMoveNum = cocTilingData.ubMoveNum * HALF_KBYTE;
    if (cocTilingData.m0 >= DEFAULT_ROW) {
        cocTilingData.k0 = DEFAULT_COL;
        cocTilingData.n0 = cocTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
        cocTilingData.mLoop = CeilDev(cocTilingData.m, cocTilingData.m0);
        cocTilingData.nLoop = CeilDev(cocTilingData.n, cocTilingData.n0);
        cocTilingData.kLoop = CeilDev(cocTilingData.k, cocTilingData.k0);
    }
}

void DealTilingParamByBuffSize(CoCTiling &cocTilingData)
{
    auto blockCount = MAX_BLOCK_COUNT;
    int64_t maxPeerMemPerRank = (LCAL_BUFF_BYTES - FLAG_BUFF_BYTES) / INPUT_DTYPE / cocTilingData.rankSize / blockCount;
    int64_t maxPValue = maxPeerMemPerRank / cocTilingData.m0 / cocTilingData.k0 / cocTilingData.kLoop;
    cocTilingData.pValue = ClampValue(cocTilingData.pValue, MIN_P_VALUE, maxPValue);

    if (cocTilingData.m0 == DEFAULT_COL &&
        static_cast<int64_t>(cocTilingData.pValue) * cocTilingData.m0 * cocTilingData.k0 * cocTilingData.kLoop >=
            maxPeerMemPerRank) {
        cocTilingData.m0 = DEFAULT_ROW;
        cocTilingData.n0 = DEFAULT_COL;
        cocTilingData.mLoop = CeilDev(cocTilingData.m, cocTilingData.m0);
        cocTilingData.nLoop = CeilDev(cocTilingData.n, cocTilingData.n0);
    }
}

// Tiling Code Function
void AllGatherV2MatmulNPU910BTwoRankINT8Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU910B_TWO_RANK_INT8_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code, {ALLGATHERV2_MATMUL_NPU910B_TWO_RANK_INT8_CODE_DEFAULT, g_allGatherV2MatmulNPU910BTwoRankINT8CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulNPU910BEightRankINT4Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU910B_EIGHT_RANK_INT4_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code,
         {ALLGATHERV2_MATMUL_NPU910B_EIGHT_RANK_INT4_CODE_DEFAULT, g_allGatherV2MatmulNPU910BEightRankINT4CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulNPU910BFourRankINT8Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU910B_FOUR_RANK_INT8_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code,
         {ALLGATHERV2_MATMUL_NPU910B_FOUR_RANK_INT8_CODE_DEFAULT, g_allGatherV2MatmulNPU910BFourRankINT8CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulNPU910BFourRankFP16Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU910B_FOUR_RANK_FP16_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code,
         {ALLGATHERV2_MATMUL_NPU910B_FOUR_RANK_FP16_CODE_DEFAULT, g_allGatherV2MatmulNPU910BFourRankFP16CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulNPU910BEightRankINT8Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU910B_EIGHT_RANK_INT8_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code,
         {ALLGATHERV2_MATMUL_NPU910B_EIGHT_RANK_INT8_CODE_DEFAULT, g_allGatherV2MatmulNPU910BEightRankINT8CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulNPU910BEightRankFP16Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU910B_EIGHT_RANK_FP16_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code,
         {ALLGATHERV2_MATMUL_NPU910B_EIGHT_RANK_FP16_CODE_DEFAULT, g_allGatherV2MatmulNPU910BEightRankFP16CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulNPU91093FourRankINT8Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU91093_FOUR_RANK_INT8_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code,
         {ALLGATHERV2_MATMUL_NPU91093_FOUR_RANK_INT8_CODE_DEFAULT, g_allGatherV2MatmulNPU91093FourRankINT8CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulNPU91093EightRankINT8Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU91093_EIGHT_RANK_INT8_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code,
         {ALLGATHERV2_MATMUL_NPU91093_EIGHT_RANK_INT8_CODE_DEFAULT, g_allGatherV2MatmulNPU91093EightRankINT8CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulNPU91093FourRankFP16Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU91093_FOUR_RANK_FP16_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code,
         {ALLGATHERV2_MATMUL_NPU91093_FOUR_RANK_FP16_CODE_DEFAULT, g_allGatherV2MatmulNPU91093FourRankFP16CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulNPU91093EightRankFP16Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_NPU91093_EIGHT_RANK_FP16_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingParamMap = {
        {&code,
         {ALLGATHERV2_MATMUL_NPU91093_EIGHT_RANK_FP16_CODE_DEFAULT, g_allGatherV2MatmulNPU91093EightRankFP16CodeMap}}};
    SetTilingParam(cocTilingData, TilingParamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    DealTilingParamByBuffSize(cocTilingData);
}

static ge::graphStatus AllGatherMatmulAIVModeCheckAttrAndSetTiling(const gert::TilingContext *context,
                                                                   AllGatherMatmulAIVModeInfo &info)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // Attr相关tilingdata的设置、校验、打印
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto isTransposeX1 = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_X1);
    auto isTransposeX2 = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupPtr);
    if (strlen(groupPtr) == 0) {
        OP_LOGE(context->GetNodeName(), "The length of group string is 0.");
        return GRAPH_FAILED;
    }
    OP_CHECK_NULL_WITH_CONTEXT(context, isTransposeX2);

    info.isTransposeX1 = *isTransposeX1 ? *isTransposeX1 : false;
    info.isTransposeX2 = *isTransposeX2 ? *isTransposeX2 : false;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AllGatherMatmulAIVModeCheckShapeAndSetTiling(const gert::TilingContext *context,
                                                                    AllGatherMatmulAIVModeInfo &info,
                                                                    CoCTiling &coctiling)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI("AllGatherMatmulAIVMode AllGatherMatmulAIVModeCheckShapeAndSetTiling.");

    const auto aShape = context->GetInputShape(A_INDEX);
    const auto bShape = context->GetInputShape(B_INDEX);
    uint32_t M = aShape->GetStorageShape().GetDim(0);
    uint32_t K = aShape->GetStorageShape().GetDim(1);
    uint32_t N = bShape->GetStorageShape().GetDim(1);

    const auto aType = context->GetInputTensor(A_INDEX)->GetDataType();
    const auto bType = context->GetInputTensor(B_INDEX)->GetDataType();
    OP_TILING_CHECK(
        aType != bType,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x1/x2", Ops::Base::ToString(aType).c_str(),
                                              "The dtypes of x1 and x2 must be the same"),
        return GRAPH_FAILED);

    if (aType == ge::DT_INT4 && bType == ge::DT_INT4) {
        OP_TILING_CHECK(K % 2 != 0 || N % 2 != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "K/N",
                            (std::string("K=") + std::to_string(K) + " N=" + std::to_string(N)).c_str(),
                            "The value of K/N must be divisible by 2 for int4"),
                        return GRAPH_FAILED);
    }

    if (aShape->GetStorageShape().GetDim(1) != bShape->GetStorageShape().GetDim(0)) {
        OP_LOGD(nodeName, "A.shape(1) %lu, B.shape(0) %lu, istransB = %d", aShape->GetStorageShape().GetDim(1),
                bShape->GetStorageShape().GetDim(0), info.isTransposeX2);
        N = bShape->GetStorageShape().GetDim(0);
    }

    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(1)->GetStorageFormat()));
    if (formatB == ge::FORMAT_FRACTAL_NZ) {
        const auto cShape = context->GetOutputShape(0);
        N = cShape->GetOriginShape().GetDim(1);
    }

    const gert::StorageShape *matrixBias = context->GetOptionalInputShape(BIAS_INDEX);
    OP_TILING_CHECK(matrixBias != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "bias", "not nullptr",
                                                          "The value of bias must be nullptr in AivMode"),
                    return GRAPH_FAILED);

    // shape相关校验与约束写在这里
    info.M = M;
    info.N = N;
    info.K = K;
    coctiling.m = M;
    coctiling.n = N;
    coctiling.k = K;
    OP_LOGD(K_INNER_DEBUG, "M=%d", info.M);
    OP_LOGD(K_INNER_DEBUG, "K=%d", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%d", info.N);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AllGatherMatmulAIVModeGetPlatformInfoAndSetTiling(const gert::TilingContext *context,
                                                                         AllGatherMatmulAIVModeInfo &info,
                                                                         CoCTiling &coctiling)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;
    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrintfTilingData(AllGatherMatmulAIVModeInfo &info, CoCTiling &coctiling)
{
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData info.M=%u", info.M);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData info.N=%u", info.N);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData info.K=%u", info.K);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData info.aivNum=%u", info.aivNum);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData info.totalUbSize=%u", info.totalUbSize);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData info.isTransposeX2=%d", info.isTransposeX2);

    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData coctiling.k0=%u", coctiling.k0);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData coctiling.n0=%u", coctiling.n0);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData coctiling.mLoop=%u", coctiling.mLoop);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData coctiling.nLoop=%u", coctiling.nLoop);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData coctiling.swizzlCount=%d", coctiling.swizzlCount);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData coctiling.swizzlDirect=%d", coctiling.swizzlDirect);
    OP_LOGD("AllgatherMatmulV2AIVMode", "TilingData coctiling.pValue=%d", coctiling.pValue);
    return ge::GRAPH_SUCCESS;
}

void GetUsrWorkSpaceSize(uint32_t nElemAlign, uint32_t elementSize, uint64_t &userWorkSpaceSize, int64_t rankSize,
                         AllGatherMatmulAIVModeTilingData *tilingData)
{
    auto &info = tilingData->allGatherMatmulInfo;
    const auto &cocTiling = tilingData->cocTiling;
    bool hasAAlign = (!IsMatrixAligned(info.M, info.K, info.isTransposeX1, nElemAlign) && info.M != 1);
    bool hasBAlign = !IsMatrixAligned(info.K, info.N, info.isTransposeX2, nElemAlign);
    uint64_t mAlign = AlignUp<uint64_t>(info.M, nElemAlign);
    uint64_t kAlign = AlignUp<uint64_t>(info.K, nElemAlign);
    uint64_t nAlign = AlignUp<uint64_t>(info.N, nElemAlign);

    info.aAlignSize = 0;
    info.bAlignSize = 0;
    info.hasAAlign = hasAAlign;
    info.hasBAlign = hasBAlign;
    if (info.hasAAlign) {
        if (elementSize != 0) {
            info.aAlignSize =
                (info.isTransposeX1 ? static_cast<uint64_t>(info.K) * mAlign : static_cast<uint64_t>(info.M) * kAlign) *
                elementSize;
        } else {
            info.aAlignSize =
                (info.isTransposeX1 ? static_cast<uint64_t>(info.K) * mAlign : static_cast<uint64_t>(info.M) * kAlign) /
                2;
        }
        userWorkSpaceSize += info.aAlignSize;
    }
    if (info.hasBAlign) {
        if (elementSize != 0) {
            info.bAlignSize =
                (info.isTransposeX2 ? static_cast<uint64_t>(info.N) * kAlign : static_cast<uint64_t>(info.K) * nAlign) *
                elementSize;
        } else {
            info.bAlignSize =
                (info.isTransposeX2 ? static_cast<uint64_t>(info.N) * kAlign : static_cast<uint64_t>(info.K) * nAlign) /
                2;
        }
        userWorkSpaceSize += info.bAlignSize;
    }

    info.accumWorkSpacePingPong = false;
    if (info.quantFlag) {
        info.accumWorkSpacePingPong = (info.M >= MAX_BLOCK_COUNT * cocTiling.m0 * cocTiling.pValue);
        uint32_t workspaceM = info.accumWorkSpacePingPong ? MAX_BLOCK_COUNT * cocTiling.m0 * cocTiling.pValue : info.M;
        userWorkSpaceSize += static_cast<uint64_t>(workspaceM) * info.N * rankSize * sizeof(int32_t);
    }
    if (info.dequantType == DequantType::PER_TOKEN) {
        userWorkSpaceSize += static_cast<uint64_t>(info.M) * rankSize * sizeof(float32_t);
    }
}

static bool CheckDtypeX1(gert::TilingContext *context)
{
    const gert::Tensor *x1Scale = context->GetInputTensor(X1_SCALE_INDEX);
    if (x1Scale == nullptr) {
        return false;
    }
    auto x1Type = x1Scale->GetDataType();
    if (x1Type != ge::DT_FLOAT) {
        return false;
    }
    return true;
}

static bool CheckDtypeX2(const gert::TilingContext *context, AllGatherMatmulAIVModeInfo &info, ge::DataType yType)
{
    const gert::Tensor *x2Scale = context->GetInputTensor(X2_SCALE_INDEX);
    if (x2Scale == nullptr) {
        return false;
    }
    auto x2ScaleType = x2Scale->GetDataType();
    info.isX2ScaleTypeInt64 = false;
    if (x2ScaleType == ge::DT_FLOAT) {
        return true;
    }
    if (yType == ge::DT_FLOAT16 && x2ScaleType == ge::DT_INT64) {
        info.isX2ScaleTypeInt64 = true;
        return true;
    }
    return false;
}

bool SetTilingDataA3(CoCTiling &cocTilingData, const AllGatherMatmulAIVModeInfo &info, int64_t rankSize)
{
    if (rankSize == RANKSIZE_FOUR && info.quantFlag) {
        AllGatherV2MatmulNPU91093FourRankINT8Tiling(cocTilingData);
        return true;
    } else if (rankSize == RANKSIZE_FOUR && !info.quantFlag) {
        AllGatherV2MatmulNPU91093FourRankFP16Tiling(cocTilingData);
        return true;
    } else if (rankSize == RANKSIZE_EIGHT && info.quantFlag) {
        AllGatherV2MatmulNPU91093EightRankINT8Tiling(cocTilingData);
        return true;
    } else if (rankSize == RANKSIZE_EIGHT && !info.quantFlag) {
        AllGatherV2MatmulNPU91093EightRankFP16Tiling(cocTilingData);
        return true;
    }
    return false;
}

bool SetTilingDataA2(CoCTiling &cocTilingData, AllGatherMatmulAIVModeInfo &info, int64_t rankSize, bool isInt4Type)
{
    if (rankSize == RANKSIZE_TWO && info.quantFlag) {
        AllGatherV2MatmulNPU910BTwoRankINT8Tiling(cocTilingData);
        return true;
    } else if (rankSize == RANKSIZE_FOUR && info.quantFlag) {
        AllGatherV2MatmulNPU910BFourRankINT8Tiling(cocTilingData);
        return true;
    } else if (rankSize == RANKSIZE_FOUR && !info.quantFlag) {
        AllGatherV2MatmulNPU910BFourRankFP16Tiling(cocTilingData);
        return true;
    } else if (rankSize == RANKSIZE_EIGHT && isInt4Type) {
        AllGatherV2MatmulNPU910BEightRankINT4Tiling(cocTilingData);
        return true;
    } else if (rankSize == RANKSIZE_EIGHT && info.quantFlag) {
        AllGatherV2MatmulNPU910BEightRankINT8Tiling(cocTilingData);
        return true;
    } else if (rankSize == RANKSIZE_EIGHT && !info.quantFlag) {
        AllGatherV2MatmulNPU910BEightRankFP16Tiling(cocTilingData);
        return true;
    }
    return false;
}

void SetTilingData(CoCTiling &cocTilingData, AllGatherMatmulAIVModeInfo &info, int64_t rankSize, bool isInt4Type)
{
    cocTilingData.rankSize = rankSize;
    if (info.is910C && SetTilingDataA3(cocTilingData, info, rankSize)) {
        return;
    } else if (SetTilingDataA2(cocTilingData, info, rankSize, isInt4Type)) {
        return;
    }
    // Default Tiling Func
    AllGatherV2MatmulNPU910BEightRankFP16Tiling(cocTilingData);
}

ge::graphStatus AllGatherMatmulTilingAIVModeFunc(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI("Enter AllGatherMatmulAIVMode tiling func.");

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batchMode = 1U;
    auto ret = context->SetScheduleMode(batchMode);
    MC2_CHECK_LOG_RET(nodeName, ret);

    // 1. tilingData
    AllGatherMatmulAIVModeTilingData *tilingData = context->GetTilingData<AllGatherMatmulAIVModeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "AllGatherMatmulAIVMode get tilingData.");
    AllGatherMatmulAIVModeInfo &info = tilingData->allGatherMatmulInfo;
    OP_LOGI(nodeName, "AllGatherMatmulAIVMode get tilingData info.");
    CoCTiling &coctiling = tilingData->cocTiling;
    OP_LOGI(nodeName, "AllGatherMatmulAIVMode get CoCTiling info.");

    OP_TILING_CHECK(AllGatherMatmulAIVModeCheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "AllGatherMatmulAIVMode CheckShapeAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(AllGatherMatmulAIVModeCheckShapeAndSetTiling(context, info, coctiling) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "AllGatherMatmulAIVMode CheckAttrAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(AllGatherMatmulAIVModeGetPlatformInfoAndSetTiling(context, info, coctiling) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "AllGatherMatmulAIVMode GetPlatformInfoAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    const char *opName = context->GetNodeName();
    // 校验HCCL BUFF空间大小
    uint64_t hcclBuffSize = 0ULL;
    auto cclRet = mc2tiling::GetCclBufferSize(group, &hcclBuffSize, opName);
    if (cclRet == ge::GRAPH_SUCCESS) {
        OP_TILING_CHECK(hcclBuffSize < CCL_BUFFER_MIN_BYTES,
                        OP_LOGE(opName, "HCCL_BUFFSIZE (%lu Bytes) too small, min required %lu Bytes (%dMB)",
                                hcclBuffSize, CCL_BUFFER_MIN_BYTES, CCL_BUFFER_MIN_BYTES / MB_BYTES),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGW(opName, "Can't get HCCL_BUFFSIZE, skip CCL buffer size validation.");
    }
    int64_t rankSize = 0;
    mc2tiling::GetRankSize(opName, group, rankSize);
    coctiling.rankSize = rankSize;
    OP_TILING_CHECK(
        rankSize != 2 && rankSize != 4 && rankSize != 8,
        OP_LOGE(context->GetNodeName(), "Unsupported rankSize %d. Supported rankSizes are 2, 4 and 8.", rankSize),
        return ge::GRAPH_FAILED);

    // 2. set numBlocks
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(numBlocks);
    coctiling.numBlocks = numBlocks;

    // 3. set tilingKey
    auto aType = context->GetInputTensor(0)->GetDataType();
    auto bType = context->GetInputTensor(1)->GetDataType();
    auto cType = context->GetOutputDesc(0)->GetDataType();
    bool isA4W4 = (aType == ge::DT_INT4) && (bType == ge::DT_INT4);
    bool isA8W8 = (aType == ge::DT_INT8) && (bType == ge::DT_INT8);
    bool isOutputTypeValid = (cType == ge::DT_BF16 || cType == ge::DT_FLOAT16);
    info.quantFlag = (isA4W4 || isA8W8) && isOutputTypeValid;
    if (info.quantFlag) {
        OP_TILING_CHECK(!CheckDtypeX2(context, info, cType),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x2Scale", "invalid dtype",
                                                              "The dtype of x2Scale must be float or int64"),
                        return ge::GRAPH_FAILED);
        info.dequantType = DequantType::PER_CHANNEL;
        if (CheckDtypeX1(context)) {
            info.dequantType = DequantType::PER_TOKEN;
        }
    }

    uint64_t tilingKey = INIT_TILINGKEY;
    GetTilingKey(tilingKey, info, context);
    context->SetTilingKey(tilingKey);
    OP_LOGI(nodeName, "The tilingKey is %lu", tilingKey);

    // 4. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    info.is910C = false;
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    if (socVersion == "Ascend910_93") {
        info.is910C = true;
    }

    // Tiling
    SetTilingData(tilingData->cocTiling, info, rankSize, isA4W4);

    // workspace
    uint32_t elementSize = 0;
    uint32_t nElemAlign = 0;
    if (aType == ge::DT_INT4) {
        nElemAlign = HALF_KBYTE * 2;
    } else {
        elementSize = D_TYPE_SIZE_MAP.at(aType);
        nElemAlign = HALF_KBYTE / elementSize;
    }

    uint64_t userWorkSpaceSize = 0;
    GetUsrWorkSpaceSize(nElemAlign, elementSize, userWorkSpaceSize, rankSize, tilingData);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + userWorkSpaceSize;

    // 5. communication
    if (info.is910C) {
        uint32_t opType = 6;
        std::string algConfig = "AllGather=level0:fullmesh";
        AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
                        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
                        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                        return ge::GRAPH_FAILED);
    } else {
        uint32_t opType = 18;
        std::string algConfig = "MultiPut=level0:fullmesh";
        AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
                        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
                        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                        return ge::GRAPH_FAILED);
    }

    PrintfTilingData(info, coctiling);
    OP_LOGI(nodeName, "Leave AllGatherMatmulAIVMode tiling func.");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
