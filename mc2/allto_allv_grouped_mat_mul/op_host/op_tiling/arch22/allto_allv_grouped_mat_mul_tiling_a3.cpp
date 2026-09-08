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
 * \file allto_allv_grouped_mat_mul_tiling_a3.cpp
 * \brief
 */

#include "allto_allv_grouped_mat_mul_tiling_a3.h"
#include "allto_allv_grouped_mat_mul_aiv_plan.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_aiv_comm.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

#include "platform/platform_infos_def.h"
#include "mc2_comm_utils.h"

namespace {
const char *A_INNER_DEBUG = "AlltoAllvGroupedMatMul Tiling";
constexpr size_t AIV_ATTR_COMM_MODE_INDEX = 7U;
constexpr uint32_t AIV_SCHEDULE_MODE = 1U;
constexpr uint32_t AIV_SYSTEM_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t AIV_ALLTOALL_OP_TYPE = 8U;
constexpr uint32_t AIV_MULTIPUT_OP_TYPE = 18U;
constexpr uint32_t AIV_INPUT_BYTES = 2U;
constexpr int64_t AIV_MAX_TOKEN_NUM = 5000000;
constexpr int64_t AIV_MAX_DIMENSION = 65535;

bool IsShapePresentAiv(const gert::StorageShape *shape)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() > 0;
}

bool IsSupportedAivRankSize(int64_t rankSize, bool is910C)
{
    const bool supportedByA2 = rankSize == 2 || rankSize == 4 || rankSize == 8;
    const bool supportedByA3 = rankSize == 16 || rankSize == 32 || rankSize == 64 || rankSize == 128;
    return supportedByA2 || (is910C && supportedByA3);
}

bool IsSupportedAivDType(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

uint32_t ResolveExpertOverlapMode(const int32_t *recvPrefix, uint32_t rankSize, uint32_t expertPerRank, uint32_t k,
                                  uint32_t n)
{
    const bool protocolSafe =
        AlltoAllvGroupedMatMulAivMode::IsExpertOverlapProtocolSafe(recvPrefix, rankSize, expertPerRank);
    const bool automatic = protocolSafe && AlltoAllvGroupedMatMulAivMode::ShouldUseAutomaticExpertOverlap(
                                               recvPrefix, rankSize, expertPerRank, k, n);
    return automatic ? A2AVGMM_EXPERT_OVERLAP_ENABLED : A2AVGMM_EXPERT_OVERLAP_DISABLED;
}

void SetCocTiling(uint32_t m, uint32_t k, uint32_t n, uint64_t ubSize, AlltoAllvGmmCoCTiling &tiling)
{
    if (m > 64U) {
        tiling.m0 = 128U;
        tiling.k0 = k < 64U ? 32U : 64U;
        tiling.n0 = 128U;
    } else if (k >= 64U && n >= 128U) {
        tiling.m0 = 64U;
        tiling.k0 = 64U;
        tiling.n0 = 128U;
    } else {
        tiling.m0 = 64U;
        tiling.k0 = 32U;
        tiling.n0 = 64U;
    }
    tiling.ubMoveNum = AlltoAllvGroupedMatMulAivMode::CopyMoveCapacity(ubSize);
    tiling.swizzlCount = 1U;
    tiling.swizzlDirect = 0U;
}

bool IsAivMode(const gert::TilingContext *context)
{
    if (context == nullptr || context->GetAttrs() == nullptr) {
        return false;
    }
    const char *commMode = context->GetAttrs()->GetAttrPointer<char>(AIV_ATTR_COMM_MODE_INDEX);
    return commMode != nullptr && std::strcmp(commMode, "aiv") == 0;
}

ge::graphStatus FillAivTiling(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto *raw = context->GetRawTilingData();
    OP_TILING_CHECK(
        raw == nullptr || raw->GetData() == nullptr || raw->GetCapacity() < sizeof(AlltoAllvGmmAivTilingData),
        OP_LOGE(nodeName, "AIV tiling buffer actual=%zu required=%zu.",
                raw == nullptr ? static_cast<size_t>(0U) : raw->GetCapacity(), sizeof(AlltoAllvGmmAivTilingData)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context->SetScheduleMode(AIV_SCHEDULE_MODE) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Failed to set AIV schedule mode."), return ge::GRAPH_FAILED);

    auto *tilingData = context->GetTilingData<AlltoAllvGmmAivTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);
    *tilingData = {};

    const auto *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
    const char *group = attrs->GetAttrPointer<char>(optiling::ATTR_GROUP_INDEX);
    const int64_t *epWorldSize = attrs->GetAttrPointer<int64_t>(optiling::ATTR_EP_WORLD_SIZE_INDEX);
    const auto *sendCountsVector = attrs->GetAttrPointer<gert::ContinuousVector>(optiling::ATTR_SEND_COUNTS_INDEX);
    const auto *recvCountsVector = attrs->GetAttrPointer<gert::ContinuousVector>(optiling::ATTR_RECV_COUNTS_INDEX);
    const bool *transGmmWeight = attrs->GetAttrPointer<bool>(optiling::NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    const bool *transMmWeight = attrs->GetAttrPointer<bool>(optiling::NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    const bool *permuteOutFlag = attrs->GetAttrPointer<bool>(optiling::NON_QUANT_ATTR_PERMUTE_OUT_FLAG_INDEX);
    OP_TILING_CHECK(group == nullptr || epWorldSize == nullptr || sendCountsVector == nullptr ||
                        recvCountsVector == nullptr || transGmmWeight == nullptr || transMmWeight == nullptr ||
                        permuteOutFlag == nullptr,
                    OP_LOGE(nodeName, "AIV attributes are incomplete."), return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(nodeName, "AIV platform information is null."),
                    return ge::GRAPH_FAILED);
    std::string socVersion;
    (void)platformInfo->GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    const bool is910C = socVersion == "Ascend910_93";
    tilingData->is910C = static_cast<uint32_t>(is910C);
    OP_TILING_CHECK(!IsSupportedAivRankSize(*epWorldSize, is910C),
                    OP_LOGE(nodeName,
                            "AIV epWorldSize %lld is unsupported on %s; A2 supports 2/4/8 and A3 supports "
                            "2/4/8/16/32/64/128.",
                            static_cast<long long>(*epWorldSize), is910C ? "A3" : "A2"),
                    return ge::GRAPH_FAILED);

    int64_t rankSize = 0;
    OP_TILING_CHECK(!mc2tiling::GetRankSize(nodeName, group, rankSize),
                    OP_LOGE(nodeName, "Failed to get rank size for group %s.", group), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSize != *epWorldSize,
                    OP_LOGE(nodeName, "epWorldSize %lld does not match HCCL rank size %lld.",
                            static_cast<long long>(*epWorldSize), static_cast<long long>(rankSize)),
                    return ge::GRAPH_FAILED);

    const auto *gmmXShape = context->GetInputShape(optiling::GMM_X_INDEX);
    const auto *gmmWeightShape = context->GetInputShape(optiling::GMM_WEIGHT_INDEX);
    const auto *gmmYShape = context->GetOutputShape(optiling::OUTPUT_GMM_Y_INDEX);
    const auto *gmmXDesc = context->GetInputDesc(optiling::GMM_X_INDEX);
    const auto *gmmWeightDesc = context->GetInputDesc(optiling::GMM_WEIGHT_INDEX);
    const auto *gmmYDesc = context->GetOutputDesc(optiling::OUTPUT_GMM_Y_INDEX);
    OP_TILING_CHECK(gmmXShape == nullptr || gmmWeightShape == nullptr || gmmYShape == nullptr || gmmXDesc == nullptr ||
                        gmmWeightDesc == nullptr || gmmYDesc == nullptr,
                    OP_LOGE(nodeName, "Required GMM tensors are incomplete."), return ge::GRAPH_FAILED);
    const ge::DataType aivDtype = gmmXDesc->GetDataType();
    OP_TILING_CHECK(!IsSupportedAivDType(aivDtype) || gmmWeightDesc->GetDataType() != aivDtype ||
                        gmmYDesc->GetDataType() != aivDtype,
                    OP_LOGE(nodeName, "AIV mode requires matching FP16 or BF16 GMM tensors."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDesc->GetStorageFormat() != ge::FORMAT_ND ||
                        gmmWeightDesc->GetStorageFormat() != ge::FORMAT_ND ||
                        gmmYDesc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE(nodeName, "AIV mode supports ND GMM tensors only."), return ge::GRAPH_FAILED);

    const auto &xShape = gmmXShape->GetStorageShape();
    const auto &weightShape = gmmWeightShape->GetStorageShape();
    const auto &yShape = gmmYShape->GetStorageShape();
    OP_TILING_CHECK(xShape.GetDimNum() != 2 || weightShape.GetDimNum() != 3 || yShape.GetDimNum() != 2,
                    OP_LOGE(nodeName, "AIV GMM expects x/y to be 2D and weight to be 3D."), return ge::GRAPH_FAILED);

    const int64_t inputM = xShape.GetDim(0);
    const int64_t inputK = xShape.GetDim(1);
    const int64_t expertPerRank = weightShape.GetDim(0);
    const int64_t weightK = weightShape.GetDim(*transGmmWeight ? 2 : 1);
    const int64_t weightN = weightShape.GetDim(*transGmmWeight ? 1 : 2);
    const int64_t outputM = yShape.GetDim(0);
    const int64_t outputN = yShape.GetDim(1);
    OP_TILING_CHECK(inputM <= 0 || inputM > AIV_MAX_TOKEN_NUM || outputM <= 0 || outputM > AIV_MAX_TOKEN_NUM,
                    OP_LOGE(nodeName, "AIV BSK/A must be in [1, %lld], but BSK=%lld and A=%lld.",
                            static_cast<long long>(AIV_MAX_TOKEN_NUM), static_cast<long long>(inputM),
                            static_cast<long long>(outputM)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        inputK <= 0 || inputK > AIV_MAX_DIMENSION || weightK <= 0 || weightK > AIV_MAX_DIMENSION || weightN <= 0 ||
            weightN > AIV_MAX_DIMENSION || outputN <= 0 || outputN > AIV_MAX_DIMENSION,
        OP_LOGE(nodeName, "AIV H1/N1 must be in [1, %lld], but H1=%lld/%lld and N1=%lld/%lld.",
                static_cast<long long>(AIV_MAX_DIMENSION), static_cast<long long>(inputK),
                static_cast<long long>(weightK), static_cast<long long>(weightN), static_cast<long long>(outputN)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputK != weightK || outputN != weightN,
                    OP_LOGE(nodeName, "AIV GMM K/N dimensions are inconsistent."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(expertPerRank <= 0 || expertPerRank > static_cast<int64_t>(A2AVGMM_MAX_LOCAL_EXPERT_NUM),
                    OP_LOGE(nodeName, "AIV local expert count must be in [1, %u], but got %lld.",
                            A2AVGMM_MAX_LOCAL_EXPERT_NUM, static_cast<long long>(expertPerRank)),
                    return ge::GRAPH_FAILED);
    const uint32_t rankSizeU32 = static_cast<uint32_t>(rankSize);
    const uint32_t expertPerRankU32 = static_cast<uint32_t>(expertPerRank);
    uint32_t expectedCountNum = 0U;
    OP_TILING_CHECK(!AlltoAllvGroupedMatMulAivPlan::IsValidCountShape(rankSizeU32, expertPerRankU32, expectedCountNum),
                    OP_LOGE(nodeName,
                            "AIV global expert count rankSize * expertPerRank must be <= %u, "
                            "but rankSize=%u and expertPerRank=%u.",
                            A2AVGMM_MAX_COUNT_NUM, rankSizeU32, expertPerRankU32),
                    return ge::GRAPH_FAILED);
    const uint64_t countNum = sendCountsVector->GetSize();
    OP_TILING_CHECK(
        countNum != recvCountsVector->GetSize() || countNum != expectedCountNum || countNum > A2AVGMM_MAX_COUNT_NUM,
        OP_LOGE(nodeName,
                "AIV counts length must equal rankSize * expertPerRank=%u and be <= %u, "
                "but send=%lu and recv=%lu.",
                expectedCountNum, A2AVGMM_MAX_COUNT_NUM, countNum, recvCountsVector->GetSize()),
        return ge::GRAPH_FAILED);
    const auto *sendCounts = static_cast<const int64_t *>(sendCountsVector->GetData());
    const auto *recvCounts = static_cast<const int64_t *>(recvCountsVector->GetData());
    OP_TILING_CHECK(sendCounts == nullptr || recvCounts == nullptr, OP_LOGE(nodeName, "AIV counts data is null."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!AlltoAllvGroupedMatMulAivPlan::BuildInclusivePrefixes(
                        sendCounts, recvCounts, rankSizeU32, expertPerRankU32, static_cast<uint32_t>(inputM),
                        static_cast<uint32_t>(outputM), tilingData->sendPrefix, tilingData->recvPrefix),
                    OP_LOGE(nodeName, "AIV counts must be non-negative int64 values whose sums match BSK/A "
                                      "and whose inclusive prefixes fit int32."),
                    return ge::GRAPH_FAILED);

    uint32_t maxExpertM = 0U;
    for (uint32_t expert = 0U; expert < expertPerRankU32; ++expert) {
        const uint32_t expertEndIndex = expert * rankSizeU32 + rankSizeU32 - 1U;
        const int32_t expertStartPrefix = expert == 0U ? 0 : tilingData->recvPrefix[expertEndIndex - rankSizeU32];
        const uint32_t expertM = static_cast<uint32_t>(tilingData->recvPrefix[expertEndIndex] - expertStartPrefix);
        maxExpertM = std::max(maxExpertM, expertM);
    }

    const auto *mmXShape = context->GetOptionalInputShape(optiling::NON_QUANT_MM_X_INDEX);
    const auto *mmWeightShape = context->GetOptionalInputShape(optiling::NON_QUANT_MM_WEIGHT_INDEX);
    const auto *mmYShape = context->GetOutputShape(optiling::OUTPUT_MM_Y_INDEX);
    const bool hasMmX = IsShapePresentAiv(mmXShape);
    const bool hasMmWeight = IsShapePresentAiv(mmWeightShape);
    const bool hasMmY = IsShapePresentAiv(mmYShape);
    OP_TILING_CHECK((hasMmX || hasMmWeight || hasMmY) && !(hasMmX && hasMmWeight && hasMmY),
                    OP_LOGE(nodeName, "mmX, mmWeight and mmY must be provided together."), return ge::GRAPH_FAILED);

    uint32_t mmM = 0U;
    uint32_t mmK = 0U;
    uint32_t mmN = 0U;
    if (hasMmX) {
        const auto *mmXDesc = context->GetOptionalInputDesc(optiling::NON_QUANT_MM_X_INDEX);
        const auto *mmWeightDesc = context->GetOptionalInputDesc(optiling::NON_QUANT_MM_WEIGHT_INDEX);
        const auto *mmYDesc = context->GetOutputDesc(optiling::OUTPUT_MM_Y_INDEX);
        OP_TILING_CHECK(
            mmXDesc == nullptr || mmWeightDesc == nullptr || mmYDesc == nullptr || mmXDesc->GetDataType() != aivDtype ||
                mmWeightDesc->GetDataType() != aivDtype || mmYDesc->GetDataType() != aivDtype ||
                mmXDesc->GetStorageFormat() != ge::FORMAT_ND || mmWeightDesc->GetStorageFormat() != ge::FORMAT_ND ||
                mmYDesc->GetStorageFormat() != ge::FORMAT_ND,
            OP_LOGE(nodeName, "AIV shared MM requires matching FP16/BF16 ND tensors."), return ge::GRAPH_FAILED);
        const auto &mx = mmXShape->GetStorageShape();
        const auto &mw = mmWeightShape->GetStorageShape();
        const auto &my = mmYShape->GetStorageShape();
        OP_TILING_CHECK(mx.GetDimNum() != 2 || mw.GetDimNum() != 2 || my.GetDimNum() != 2,
                        OP_LOGE(nodeName, "AIV shared MM tensors must be 2D."), return ge::GRAPH_FAILED);
        const int64_t localMmM = mx.GetDim(0);
        const int64_t localMmK = mx.GetDim(1);
        const int64_t localWeightK = mw.GetDim(*transMmWeight ? 1 : 0);
        const int64_t localWeightN = mw.GetDim(*transMmWeight ? 0 : 1);
        OP_TILING_CHECK(localMmM <= 0 || localMmM > AIV_MAX_TOKEN_NUM || localMmK <= 0 ||
                            localMmK > AIV_MAX_DIMENSION || localWeightK <= 0 || localWeightK > AIV_MAX_DIMENSION ||
                            localWeightN <= 0 || localWeightN > AIV_MAX_DIMENSION || localMmK != localWeightK ||
                            my.GetDim(0) != localMmM || my.GetDim(1) != localWeightN,
                        OP_LOGE(nodeName, "AIV shared MM dimensions are inconsistent."), return ge::GRAPH_FAILED);
        mmM = static_cast<uint32_t>(localMmM);
        mmK = static_cast<uint32_t>(localMmK);
        mmN = static_cast<uint32_t>(localWeightN);
    }

    const auto *permuteShape = context->GetOutputShape(optiling::OUTPUT_PERMUTE_OUT_INDEX);
    const bool hasPermute = IsShapePresentAiv(permuteShape);
    OP_TILING_CHECK(*permuteOutFlag != hasPermute, OP_LOGE(nodeName, "permuteOutFlag must match permuteOut presence."),
                    return ge::GRAPH_FAILED);
    if (hasPermute) {
        const auto *permuteDesc = context->GetOutputDesc(optiling::OUTPUT_PERMUTE_OUT_INDEX);
        const auto &shape = permuteShape->GetStorageShape();
        OP_TILING_CHECK(
            permuteDesc == nullptr || permuteDesc->GetDataType() != aivDtype ||
                permuteDesc->GetStorageFormat() != ge::FORMAT_ND || shape.GetDimNum() != 2 ||
                shape.GetDim(0) != outputM || shape.GetDim(1) != inputK,
            OP_LOGE(nodeName, "AIV permuteOut must match the FP16/BF16 input dtype and shape [outputM, K]."),
            return ge::GRAPH_FAILED);
    }

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = platform.GetCoreNumAic();
    const uint32_t aivNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t blockDim = platform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_TILING_CHECK(blockDim == 0U, OP_LOGE(nodeName, "AIV mixed block dimension is zero."), return ge::GRAPH_FAILED);
    context->SetBlockDim(blockDim);

    auto &gmmInfo = tilingData->gmmInfo;
    gmmInfo.M = maxExpertM;
    gmmInfo.K = static_cast<uint32_t>(inputK);
    gmmInfo.N = static_cast<uint32_t>(weightN);
    gmmInfo.rankSize = rankSizeU32;
    gmmInfo.expertPerRank = expertPerRankU32;
    gmmInfo.maxOutputSize = static_cast<uint32_t>(outputM);
    gmmInfo.isTransposeB = static_cast<uint32_t>(*transGmmWeight);
    gmmInfo.hasSharedExpert = static_cast<uint32_t>(hasMmX);
    gmmInfo.hasPermuteOut = static_cast<uint32_t>(hasPermute);
    gmmInfo.aivCoreNum = aivNum;
    gmmInfo.aicCoreNum = aicNum;
    gmmInfo.totalUbSize = static_cast<uint32_t>(ubSize);
    tilingData->expertOverlapMode =
        ResolveExpertOverlapMode(tilingData->recvPrefix, gmmInfo.rankSize, gmmInfo.expertPerRank, gmmInfo.K, gmmInfo.N);

    if (hasMmX) {
        tilingData->mmInfo.M = mmM;
        tilingData->mmInfo.K = mmK;
        tilingData->mmInfo.N = mmN;
        tilingData->mmInfo.rankSize = static_cast<uint32_t>(rankSize);
        tilingData->mmInfo.expertPerRank = 1U;
        tilingData->mmInfo.maxOutputSize = mmM;
        tilingData->mmInfo.isTransposeB = static_cast<uint32_t>(*transMmWeight);
        tilingData->mmInfo.hasSharedExpert = 1U;
        tilingData->mmInfo.aivCoreNum = aivNum;
        tilingData->mmInfo.aicCoreNum = aicNum;
        tilingData->mmInfo.totalUbSize = static_cast<uint32_t>(ubSize);
    }

    SetCocTiling(gmmInfo.M, gmmInfo.K, gmmInfo.N, ubSize, tilingData->gmmCocTiling);
    OP_TILING_CHECK(tilingData->gmmCocTiling.ubMoveNum == 0U,
                    OP_LOGE(nodeName, "UB is too small for two AIV copy buffers and metadata."),
                    return ge::GRAPH_FAILED);
    if (hasMmX) {
        SetCocTiling(mmM, mmK, mmN, ubSize, tilingData->mmCocTiling);
    }
    tilingData->countNum = static_cast<uint32_t>(countNum);

    AlltoAllvGroupedMatMulAiv::WorkspaceLayout workspace = {};
    OP_TILING_CHECK(!AlltoAllvGroupedMatMulAiv::BuildWorkspaceLayout(
                        static_cast<uint64_t>(outputM), static_cast<uint64_t>(inputK), AIV_INPUT_BYTES, workspace),
                    OP_LOGE(nodeName, "Failed to build AIV workspace layout for A=%lld and H1=%lld.",
                            static_cast<long long>(outputM), static_cast<long long>(inputK)),
                    return ge::GRAPH_FAILED);
    tilingData->recvTokenOffset = 0U;
    tilingData->userWorkspaceSize = workspace.totalBytes;

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaceSizes == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"),
                    return ge::GRAPH_FAILED);
    uint64_t totalWorkspaceBytes = 0U;
    OP_TILING_CHECK(
        !AlltoAllvGroupedMatMulAiv::SafeAddU64(AIV_SYSTEM_WORKSPACE_SIZE, workspace.totalBytes, totalWorkspaceBytes) ||
            totalWorkspaceBytes > std::numeric_limits<size_t>::max(),
        OP_LOGE(nodeName, "AIV workspace size overflow."), return ge::GRAPH_FAILED);
    workspaceSizes[0] = static_cast<size_t>(totalWorkspaceBytes);

    AlltoAllvGroupedMatMulAiv::RuntimeControlLayout control = {};
    OP_TILING_CHECK(
        !AlltoAllvGroupedMatMulAiv::BuildRuntimeControlLayout(gmmInfo.rankSize, gmmInfo.expertPerRank, control),
        OP_LOGE(nodeName, "Failed to build AIV runtime control layout."), return ge::GRAPH_FAILED);
    AlltoAllvGroupedMatMulAiv::A2avWindowLayout window = {};
    OP_TILING_CHECK(!AlltoAllvGroupedMatMulAiv::BuildWindowLayout(
                        static_cast<uint64_t>(inputM), static_cast<uint64_t>(inputK), countNum, control, 0U, window),
                    OP_LOGE(nodeName, "Failed to build AIV HCCL window layout."), return ge::GRAPH_FAILED);
    uint64_t actualWindowBytes = 0U;
    const bool windowQuerySucceeded =
        mc2tiling::GetCclBufferSize(group, &actualWindowBytes, nodeName) == ge::GRAPH_SUCCESS;
    if (!windowQuerySucceeded) {
        actualWindowBytes = 0U;
    }
    tilingData->actualWindowBytes = actualWindowBytes;
    tilingData->requiredWindowBytes = window.requiredBytes;
    tilingData->payloadBytes = window.payloadBytes;
    tilingData->countBytes = window.countBytes;
    tilingData->controlBytes = window.controlBytes;
    if (!windowQuerySucceeded) {
        OP_LOGE(nodeName,
                "AIV HCCL window actual=%lu required=%lu payload=%lu count=%lu control=%lu, "
                "BSK=%lld H1=%lld dtypeBytes=2 globalExpertNum=%lu.",
                actualWindowBytes, window.requiredBytes, window.payloadBytes, window.countBytes, window.controlBytes,
                static_cast<long long>(inputM), static_cast<long long>(inputK), countNum);
    } else {
        OP_TILING_CHECK(
            actualWindowBytes < window.requiredBytes,
            OP_LOGE(nodeName,
                    "AIV HCCL window actual=%lu required=%lu payload=%lu count=%lu control=%lu, "
                    "BSK=%lld H1=%lld dtypeBytes=2 globalExpertNum=%lu.",
                    actualWindowBytes, window.requiredBytes, window.payloadBytes, window.countBytes,
                    window.controlBytes, static_cast<long long>(inputM), static_cast<long long>(inputK), countNum),
            return ge::GRAPH_FAILED);
    }

    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(*transGmmWeight, hasMmX ? *transMmWeight : false, Mc2Comm::COMM_MODE_AIV);
    context->SetTilingKey(tilingKey);

    const uint32_t opType = is910C ? AIV_ALLTOALL_OP_TYPE : AIV_MULTIPUT_OP_TYPE;
    const std::string algConfig = is910C ? "AlltoAll=level0:fullmesh;level1:pairwise" : "MultiPut=level0:fullmesh";
    AscendC::Mc2CcTilingConfig commConfig(group, opType, algConfig);
    OP_TILING_CHECK(commConfig.SetCommEngine(Mc2Comm::ENGINE_MTE) != 0U,
                    OP_LOGE(nodeName, "Failed to select MTE communication engine."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        commConfig.GetTiling(tilingData->hcclInitTiling) != 0U || commConfig.GetTiling(tilingData->hcclCcTiling) != 0U,
        OP_LOGE(nodeName, "Failed to generate AIV HCCL tiling."), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName,
            "AIV tiling: rank=%u, expert=%u, maxExpertM=%u, K=%u, N=%u, blockDim=%u, workspace=%lu, is910C=%u.",
            gmmInfo.rankSize, gmmInfo.expertPerRank, gmmInfo.M, gmmInfo.K, gmmInfo.N, blockDim,
            tilingData->userWorkspaceSize, tilingData->is910C);
    raw->SetDataSize(sizeof(AlltoAllvGmmAivTilingData));
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {

std::vector<int64_t> AlltoAllvGmmTilingA3::GetEpWorldSizeOptional() const
{
    return {8, 16, 32, 64, 128};
}

bool AlltoAllvGmmTilingA3::NeedToCheckCounts() const
{
    return true;
}

ge::graphStatus AlltoAllvGmmTilingFuncA3::AlltoAllvGmmOpTilingFunc(gert::TilingContext *context)
{
    AlltoAllvGmmTilingA3 tiling(context);
    OP_TILING_CHECK(tiling.Init(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM tiling init failed."),
                    return ge::GRAPH_FAILED);
    return tiling.RunFusionKernelTiling(context);
}

ge::graphStatus AlltoAllvGmmTilingStructA3::DoOpTiling()
{
    if (IsAivMode(context_)) {
        return FillAivTiling(context_);
    }
    AlltoAllvGmmTilingFuncA3 funcA3;
    return funcA3.AlltoAllvGmmOpTilingFunc(context_);
}

ge::graphStatus AlltoAllvGmmTilingStructA3::GetShapeAttrsInfo()
{
    if (IsAivMode(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    return AlltoAllvGmmTilingBase::GetShapeAttrsInfo();
}

ge::graphStatus AlltoAllvGmmTilingStructA3::DoLibApiTiling()
{
    if (IsAivMode(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    return AlltoAllvGmmTilingBase::DoLibApiTiling();
}

ge::graphStatus AlltoAllvGmmTilingStructA3::GetWorkspaceSize()
{
    if (IsAivMode(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    return AlltoAllvGmmTilingBase::GetWorkspaceSize();
}

ge::graphStatus AlltoAllvGmmTilingStructA3::PostTiling()
{
    if (IsAivMode(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    return AlltoAllvGmmTilingBase::PostTiling();
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, AlltoAllvGmmTilingStructA3, 0);
} // namespace optiling
