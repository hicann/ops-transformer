/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "text_gated_residual_tiling.h"
#include "../op_kernel/catlass_hc_forward_mul_sum_tiling_data.h"
#include "platform/soc_spec.h"

#include <cstdint>
#include <cmath>
#include <limits>
#include <initializer_list>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t HC_ACTIVATION_RANK = 3;
constexpr int64_t HC_WEIGHT_RANK = 2;
constexpr int64_t HC_FEATURE_DIM = 2;
constexpr int64_t HC_DTYPE_BF16 = 2;
constexpr int64_t HC_WORKSPACE_ALIGN_PADDING = 15;
} // namespace

namespace optiling {
namespace {
constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_W_DOWN = 1;
constexpr size_t INPUT_W_UP = 2;
constexpr size_t INPUT_W_GATE = 3;
constexpr size_t INPUT_GAMMA = 4;
constexpr size_t ATTR_CHANNELS = 0;
constexpr size_t ATTR_EPS = 1;
constexpr size_t ATTR_NEED_BACKWARD = 2;
constexpr uint32_t M_TASK_LARGE = HC_MTASK_LARGE;
constexpr uint32_t M_TASK_SMALL = HC_MTASK_SMALL;
constexpr uint32_t L1_M_SMALL = HC_L1M_SMALL;
constexpr uint32_t SMALL_M_THRESH = HC_SMALL_M_THRESH;
// A2：行任务 128 之后，小路径约 0.42µs/行，大路径在核没铺满时停在 430–450µs。
// M=896 小路径更快（407 vs 443），M=1024 大路径更快（433 vs 461）。
constexpr uint32_t SMALL_M_THRESH_A2 = 1024;
constexpr uint32_t L1_N0_LARGE = HC_L1N0_LARGE;
constexpr uint32_t L1_N0_LARGE_A2 = 256;
constexpr uint32_t L1_N1 = HC_L1N1;

bool AddProduct(int64_t& total, std::initializer_list<int64_t> factors)
{
    int64_t product = 1;
    for (int64_t factor : factors) {
        if (factor < 0 || (factor != 0 && product > INT64_MAX / factor)) {
            return false;
        }
        product *= factor;
    }
    if (total > INT64_MAX - product) {
        return false;
    }
    total += product;
    return true;
}

bool CheckedWorkspaceBytes(const HcForwardMulSumTilingData& t, bool isA2, bool fusedSmall, int64_t& bytes)
{
    bytes = 0;
    const int64_t m = t.M, d = t.D, r = t.R, cores = t.usedAic, task = t.mTask;
    const int64_t rawCols = r + t.nWsGate;
    if (fusedSmall) {
        return AddProduct(bytes, {m, d + r, t.szT}) && AddProduct(bytes, {2, m, rawCols, t.szT}) &&
               AddProduct(bytes, {2, m, rawCols, 4, t.splitkFactor}) &&
               (t.needBackward || AddProduct(bytes, {m, d, t.szT}));
    }
    const bool ring = t.smallN == 0;
    if (!AddProduct(bytes, {ring ? cores : 1, ring ? 3 : 1, ring ? task : m, d, t.szT}) ||
        !AddProduct(bytes, {ring ? cores : 1, ring ? 2 : 1, ring ? task : m, r, t.szT})) {
        return false;
    }
    if (isA2) {
        if (!AddProduct(bytes, {ring ? cores : 1, 2, ring ? t.L1M0 : task, r + HcC0GateCols(t), 4})) {
            return false;
        }
        if (t.smallN != 0 && t.splitkFactor > 1 && !AddProduct(bytes, {2, task, HcTilingNWs(t), 4, t.splitkFactor})) {
            return false;
        }
    } else if (!AddProduct(bytes, {cores, t.L1M0, HcA5TilingNWs(t), t.szT, 2})) {
        return false;
    }
    return t.needBackward || AddProduct(bytes, {m, d, t.szT});
}

} // namespace

bool XDtypeOk(ge::DataType dt)
{
    return dt == ge::DT_FLOAT16 || dt == ge::DT_BF16;
}

ge::graphStatus Tiling4TextGatedResidual(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr)
        return ge::GRAPH_FAILED;
    bool isDav2201 = false;
    if (context != nullptr && context->GetPlatformInfo() != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        isDav2201 = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_2201;
    }
    const auto xShapePtr = context->GetInputShape(INPUT_X);
    const auto wDownShapePtr = context->GetInputShape(INPUT_W_DOWN);
    const auto wUpShapePtr = context->GetInputShape(INPUT_W_UP);
    const auto attrs = context->GetAttrs();
    if (xShapePtr == nullptr || wDownShapePtr == nullptr || wUpShapePtr == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& xShape = xShapePtr->GetStorageShape();
    if (xShape.GetDimNum() != HC_ACTIVATION_RANK) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* channelsPtr = attrs->GetAttrPointer<int64_t>(ATTR_CHANNELS);
    const float* epsPtr = attrs->GetAttrPointer<float>(ATTR_EPS);
    const bool hasWGate = context->GetOptionalInputShape(INPUT_W_GATE) != nullptr;
    const bool* needBackwardPtr = attrs->GetAttrPointer<bool>(ATTR_NEED_BACKWARD);
    const uint32_t needBackward = (needBackwardPtr != nullptr && *needBackwardPtr) ? 1u : 0u;
    if (channelsPtr == nullptr || *channelsPtr <= 0) {
        return ge::GRAPH_FAILED;
    }

    const auto& wdShape = wDownShapePtr->GetStorageShape();
    const auto& wuShape = wUpShapePtr->GetStorageShape();
    if (wdShape.GetDimNum() != HC_WEIGHT_RANK || wuShape.GetDimNum() != HC_WEIGHT_RANK || epsPtr == nullptr ||
        !std::isfinite(*epsPtr) || *epsPtr < 0.f)
        return ge::GRAPH_FAILED;
    const int64_t b64 = xShape.GetDim(0), s64 = xShape.GetDim(1), d64 = xShape.GetDim(HC_FEATURE_DIM),
                  r64 = wdShape.GetDim(0);
    if (b64 < 0 || s64 < 0 || d64 <= 0 || r64 <= 0 || b64 > UINT32_MAX || s64 > UINT32_MAX || d64 > UINT32_MAX ||
        r64 > UINT32_MAX || *channelsPtr > UINT32_MAX || (s64 != 0 && b64 > UINT32_MAX / s64) ||
        wdShape.GetDim(1) != d64 || wuShape.GetDim(0) != d64 || wuShape.GetDim(1) != r64)
        return ge::GRAPH_FAILED;
    int64_t tensorBytes = 0;
    if (!AddProduct(tensorBytes, {b64, s64, d64, 4}) || !AddProduct(tensorBytes, {b64, s64, r64, 4}) ||
        !AddProduct(tensorBytes, {r64, d64, 2}) || !AddProduct(tensorBytes, {*channelsPtr, d64, 2})) {
        return ge::GRAPH_FAILED;
    }
    const int64_t B = b64;
    const int64_t S = s64;
    const uint32_t D = static_cast<uint32_t>(xShape.GetDim(HC_FEATURE_DIM));
    const uint32_t C = static_cast<uint32_t>(*channelsPtr);
    if (C == 0 || D % C != 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t H = D / C;
    const uint32_t R = static_cast<uint32_t>(wDownShapePtr->GetStorageShape().GetDim(0));
    const uint32_t M = static_cast<uint32_t>(B * S);
    if (isDav2201) {
        if (epsPtr == nullptr || *epsPtr < 0.f || D == 0 || R == 0) {
            return ge::GRAPH_FAILED;
        }
        const auto xDescEarly = context->GetInputDesc(INPUT_X);
        const auto downDesc = context->GetInputDesc(INPUT_W_DOWN);
        const auto upDesc = context->GetInputDesc(INPUT_W_UP);
        if (xDescEarly == nullptr || downDesc == nullptr || upDesc == nullptr || !XDtypeOk(xDescEarly->GetDataType()) ||
            downDesc->GetDataType() != xDescEarly->GetDataType() ||
            upDesc->GetDataType() != xDescEarly->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
        const auto& wDownShape = wDownShapePtr->GetStorageShape();
        const auto& wUpShape = wUpShapePtr->GetStorageShape();
        if (wDownShape.GetDimNum() != HC_WEIGHT_RANK || wUpShape.GetDimNum() != HC_WEIGHT_RANK ||
            wDownShape.GetDim(1) != static_cast<int64_t>(D) || wUpShape.GetDim(0) != static_cast<int64_t>(D) ||
            wUpShape.GetDim(1) != static_cast<int64_t>(R)) {
            return ge::GRAPH_FAILED;
        }
        const bool useCombineNow = hasWGate;
        if (useCombineNow) {
            const auto wGateShapePtr = context->GetOptionalInputShape(INPUT_W_GATE);
            if (wGateShapePtr == nullptr || wGateShapePtr->GetStorageShape().GetDimNum() != HC_WEIGHT_RANK ||
                wGateShapePtr->GetStorageShape().GetDim(0) != static_cast<int64_t>(C) ||
                wGateShapePtr->GetStorageShape().GetDim(1) != static_cast<int64_t>(D)) {
                return ge::GRAPH_FAILED;
            }
            const auto gateDesc = context->GetInputDesc(INPUT_W_GATE);
            if (gateDesc == nullptr || gateDesc->GetDataType() != xDescEarly->GetDataType()) {
                return ge::GRAPH_FAILED;
            }
        }
        if (B == 0 || S == 0) {
            TextGatedResidualTilingData tiling;
            tiling.set_B(B);
            tiling.set_S(S);
            tiling.set_M(0);
            tiling.set_D(D);
            tiling.set_R(R);
            tiling.set_H(H);
            tiling.set_C(C);
            tiling.set_tilingKey(0);
            tiling.set_eps(epsPtr != nullptr ? *epsPtr : 0.f);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
            context->SetBlockDim(1);
            context->SetTilingKey(0);
            context->GetWorkspaceSizes(1)[0] = 0;
            return ge::GRAPH_SUCCESS;
        }
    }
    const auto xDesc = context->GetInputDesc(INPUT_X);
    if (xDesc == nullptr || !XDtypeOk(xDesc->GetDataType())) {
        return ge::GRAPH_FAILED;
    }
    const auto* wdDesc = context->GetInputDesc(INPUT_W_DOWN);
    const auto* wuDesc = context->GetInputDesc(INPUT_W_UP);
    if (wdDesc == nullptr || wuDesc == nullptr || wdDesc->GetDataType() != xDesc->GetDataType() ||
        wuDesc->GetDataType() != xDesc->GetDataType())
        return ge::GRAPH_FAILED;
    if (hasWGate) {
        const auto* gateShape = context->GetOptionalInputShape(INPUT_W_GATE);
        const auto* gateDesc = context->GetInputDesc(INPUT_W_GATE);
        if (gateShape == nullptr || gateDesc == nullptr || gateDesc->GetDataType() != xDesc->GetDataType() ||
            gateShape->GetStorageShape().GetDimNum() != HC_WEIGHT_RANK || gateShape->GetStorageShape().GetDim(0) != C ||
            gateShape->GetStorageShape().GetDim(1) != D)
            return ge::GRAPH_FAILED;
    }
    uint32_t dtypeId = 1;
    uint32_t szT = 2;
    if (xDesc->GetDataType() == ge::DT_BF16) {
        dtypeId = HC_DTYPE_BF16;
    }
    const uint32_t smallThresh = isDav2201 ? SMALL_M_THRESH_A2 : SMALL_M_THRESH;
    const uint32_t smallN = (M < smallThresh) ? 1u : 0u;
    uint32_t mTask = smallN ? M_TASK_SMALL : M_TASK_LARGE;
    if (smallN != 0u) {
        if (isDav2201) {
            // M>=128 时拆成两段等长任务，行数上取整到 32。M=128/192/320/384 比固定
            // 128 或 256 快 7–13µs；M 更小时半段不足 64 行，单段更快。
            if (M >= 128u) {
                uint32_t half = M / 2u + M % 2u;
                mTask = (half + 31u) & ~31u;
            } else if (M >= 64u) {
                mTask = M_TASK_LARGE;
            }
        }
    }
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    // 大路径 128 行任务不足以覆盖 AIC 时，降到 64 行提高并行度。
    if (smallN == 0u && (M / mTask + (M % mTask != 0u)) < aicNum) {
        mTask = 64u;
    }
    // A5 large kernels keep compile-time L1M=128 for C0 slot addressing even
    // when the task has only 64 valid rows. Host workspace must use that stride.
    const uint32_t l1M = smallN ? L1_M_SMALL : (isDav2201 ? mTask : M_TASK_LARGE);
    const uint32_t l1N0Small = isDav2201 ? 64u : 32u;
    const uint32_t l1N0 = smallN ? l1N0Small : (isDav2201 ? L1_N0_LARGE_A2 : L1_N0_LARGE);
    const uint32_t taskNum = M / mTask + (M % mTask != 0u);
    const uint32_t c1Swizzle = (M < D) ? 1u : 0u;
    const uint32_t useCombine = (hasWGate) ? 1u : 0u;
    // Padding and concatenation remain uint32 tile counts; reject before narrowing or alignment.
    const int64_t paddedR = (static_cast<int64_t>(R) + 31) / 32 * 32;
    const int64_t paddedC = useCombine ? (static_cast<int64_t>(C) + 31) / 32 * 32 : 0;
    const int64_t concat = static_cast<int64_t>(R) + (useCombine ? C : 0u);
    if (paddedR > UINT32_MAX || paddedC > UINT32_MAX || paddedR + paddedC > UINT32_MAX ||
        concat > static_cast<int64_t>(UINT32_MAX) - HC_WORKSPACE_ALIGN_PADDING) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t nWsGate = isDav2201 ? 0u : HcNWsGate(C, useCombine);
    const uint32_t nConcat = isDav2201 ? HcNConcat(R, C, useCombine) : HcNWsDown(R) + nWsGate;
    const uint32_t nWs = isDav2201 ? HcNWs(nConcat) : HcNWsDown(R);

    uint32_t splitkFactor = 1;
    uint32_t usedAic = 1;
    if (smallN && isDav2201) {
        const uint32_t nSliceC0 = nConcat / l1N0 + (nConcat % l1N0 != 0u);
        const uint32_t nSliceC1 = (D + L1_N1 - 1) / L1_N1;
        splitkFactor = HcGetSplitkFactor(mTask, R, D, l1M, l1N0, HcC0L1K(szT), aicNum);
        // C0 每个 wave 的循环数是 mTile×nTile；A2 bf16 的 kernel 侧 L1M0=32。
        // 不计入 mTile 会把 usedAic 压到 nSliceC1=10，C0 峰值被抬到 3 块/核。
        const uint32_t c0TileM = isDav2201 ? 32u : l1M;
        uint32_t c0Loops = nSliceC0 * splitkFactor;
        if (isDav2201) {
            c0Loops *= (mTask + c0TileM - 1u) / c0TileM;
        }
        uint32_t nSliceMax = c0Loops > nSliceC1 ? c0Loops : nSliceC1;
        usedAic = aicNum < nSliceMax ? aicNum : nSliceMax;
    } else if (smallN == 0u) {
        usedAic = aicNum < taskNum ? aicNum : taskNum;
    }
    if (usedAic == 0) {
        usedAic = 1;
    }

    const auto gammaTensor = context->GetOptionalInputTensor(INPUT_GAMMA);
    uint32_t hasGamma = 0u;
    if (gammaTensor != nullptr) {
        const auto gammaShape = context->GetOptionalInputShape(INPUT_GAMMA);
        if (gammaShape == nullptr || gammaShape->GetStorageShape().GetDimNum() != 1 ||
            gammaShape->GetStorageShape().GetShapeSize() != static_cast<int64_t>(D)) {
            return ge::GRAPH_FAILED;
        }
        hasGamma = 1u;
    }
    uint32_t tilingKey = c1Swizzle | (smallN << 1); // bit0 swizzle，bit1 smallN；dtype 由 DTYPE_X

    const bool fusedSmall = ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510 && smallN != 0u && C >= 2u &&
                            C <= 10u && H >= 256u && H <= 4096u && H % 256u == 0u && R >= 64u && R <= 896u &&
                            R % 64u == 0u && aicNum > 0u;
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510 && M > 0u && smallN != 0u && !fusedSmall) {
        return ge::GRAPH_FAILED;
    }
    if (fusedSmall) {
        mTask = M;
        usedAic = aicNum;
        splitkFactor = 14u;
    }

    TextGatedResidualTilingData tiling;
    tiling.set_B(B);
    tiling.set_S(S);
    tiling.set_M(M);
    tiling.set_D(D);
    tiling.set_R(R);
    tiling.set_H(H);
    tiling.set_C(C);
    tiling.set_usedAic(usedAic);
    tiling.set_mTask(mTask);
    tiling.set_taskNum(fusedSmall ? 1u : taskNum);
    tiling.set_L1M0(l1M);
    tiling.set_L1N0(l1N0);
    tiling.set_L1M1(l1M);
    tiling.set_L1N1(L1_N1);
    tiling.set_szT(szT);
    tiling.set_useCombine(useCombine);
    tiling.set_hasGamma(hasGamma);
    tiling.set_tilingKey(HcEncodeTilingKey(dtypeId, useCombine, hasGamma, c1Swizzle, smallN));
    tiling.set_eps(epsPtr != nullptr ? *epsPtr : 1e-6f);
    tiling.set_smallN(smallN);
    tiling.set_splitkFactor(smallN ? splitkFactor : 1u);
    tiling.set_nConcat(nConcat);
    tiling.set_nWs(nWs);
    tiling.set_nWsGate(nWsGate);
    tiling.set_needBackward(needBackward);
    tiling.set_pad0(0);
    tiling.set_pad1(0u);
    tiling.set_pad2(0);
    tiling.set_pad3(0u);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    HcForwardMulSumTilingData ws{};
    ws.M = M;
    ws.D = D;
    ws.R = R;
    ws.usedAic = usedAic;
    ws.mTask = mTask;
    ws.L1M0 = l1M;
    ws.L1N0 = l1N0;
    ws.L1M1 = l1M;
    ws.L1N1 = L1_N1;
    ws.szT = szT;
    ws.smallN = smallN;
    ws.splitkFactor = smallN ? splitkFactor : 1u;
    ws.C = C;
    ws.useCombine = useCombine;
    ws.nConcat = nConcat;
    ws.nWs = nWs;
    ws.nWsGate = nWsGate;
    ws.needBackward = needBackward;

    context->SetBlockDim(usedAic);
    context->SetTilingKey(tilingKey);
    int64_t workspaceBytes = 0;
    if (!CheckedWorkspaceBytes(ws, isDav2201, fusedSmall, workspaceBytes)) {
        return ge::GRAPH_FAILED;
    }
    const auto libBytes = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (libBytes > static_cast<uint64_t>(INT64_MAX) || !AddProduct(workspaceBytes, {static_cast<int64_t>(libBytes)}) ||
        static_cast<uint64_t>(workspaceBytes) > std::numeric_limits<size_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    context->GetWorkspaceSizes(1)[0] = static_cast<size_t>(workspaceBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4TextGatedResidual(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TextGatedResidual)
    .Tiling(Tiling4TextGatedResidual)
    .TilingParse<TextGatedResidualCompileInfo>(TilingPrepare4TextGatedResidual);
} // namespace optiling
