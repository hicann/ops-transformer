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
 * \file quant_grouped_matmul_dequant_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "platform/platform_infos_def.h"
#include "quant_grouped_matmul_dequant_tiling.h"
#include "../op_kernel/quant_grouped_matmul_dequant_config.h"

bool AddWorkspace(gert::TilingContext *context, const size_t workspace)
{
    size_t *workspace_size = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace_size);
    *workspace_size = workspace;
    return true;
}

namespace optiling {

bool QuantMatmulDequantTiling::CheckInOutShapes(gert::TilingContext *context)
{
    uint32_t idx = 0;
    auto x = context->GetInputShape(idx++);
    OP_CHECK_NULL_WITH_CONTEXT(context, x);
    auto shape = x->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != NUMBER_2,
                OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get x shape ndim is not 2, please check."),
                return false);
    _Params.originM = (uint64_t)shape.GetDim(0);
    _Params.originK = (uint64_t)shape.GetDim(1);
    _Params.fracK = (_Params.originK + K_FRACTAL_INT8 - 1) / K_FRACTAL_INT8;

    if (isGrouped) {
        auto quantized_weight = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, quantized_weight);
        shape = quantized_weight->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != NUMBER_5,
                    OP_LOGE(context->GetNodeName(),
                            "QuantMatmulDequant get quantized_weight shape ndim is not 5, please check."),
                    return false);
        _Params.originE = (uint64_t)shape.GetDim(0);
        _Params.fracN = (uint64_t)shape.GetDim(NUMBER_2);
        OP_CHECK_IF(
            (uint64_t)shape.GetDim(1) != _Params.fracK || shape.GetDim(NUMBER_3) != NM_FRACTAL_INT8 ||
                shape.GetDim(NUMBER_4) != K_FRACTAL_INT8,
            OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get quantized_weight shape not (G, K//32, N//16, 16, 32), please check."),
            return false);

        auto weight_scale = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, weight_scale);
        shape = weight_scale->GetStorageShape();
        OP_CHECK_IF(
            shape.GetDimNum() != NUMBER_2,
            OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get weight_scale shape ndim is not 2, please check."),
            return false);
        OP_CHECK_IF(
            _Params.originE != (uint64_t)shape.GetDim(0),
            OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant weight_scale shape[0] and quantized_weight shape[0] not equal please check."),
            return false);
        _Params.originN = shape.GetDim(1);
        OP_CHECK_IF(
            (_Params.originN + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8 != _Params.fracN,
            OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get n-dim of weight_scale and quantized_weight not match, please check."),
            return false);
        auto scale_desc = context->GetInputDesc(idx - 1);
        _Params.scaleIsInt64 = false;
        if (scale_desc != nullptr) {
            _Params.scaleIsInt64 = (scale_desc->GetDataType() == ge::DT_INT64);
        }

        auto group_list = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, group_list);
        shape = group_list->GetStorageShape();
        OP_CHECK_IF(
            shape.GetDimNum() != 1 || (uint64_t)shape.GetDim(0) != _Params.originE,
            OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant group_list shape[0] and quantized_weight shape[0] not equal please check."),
            return false);
    } else {
        auto quantized_weight = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, quantized_weight);
        shape = quantized_weight->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != NUMBER_4,
                    OP_LOGE(context->GetNodeName(),
                            "QuantMatmulDequant get quantized_weight shape ndim is not 4, please check."),
                    return false);
        _Params.fracN = (uint64_t)shape.GetDim(1);
        OP_CHECK_IF((uint64_t)shape.GetDim(0) != _Params.fracK || shape.GetDim(NUMBER_2) != NM_FRACTAL_INT8 ||
                        shape.GetDim(NUMBER_3) != K_FRACTAL_INT8,
                    OP_LOGE(context->GetNodeName(),
                            "QuantMatmulDequant get quantized_weight shape not (K//32, N//16, 16, 32), please check."),
                    return false);

        auto weight_scale = context->GetInputShape(idx++);
        OP_CHECK_NULL_WITH_CONTEXT(context, weight_scale);
        shape = weight_scale->GetStorageShape();
        OP_CHECK_IF(
            shape.GetDimNum() != 1,
            OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get weight_scale shape ndim is not 1, please check."),
            return false);
        _Params.originN = (uint64_t)shape.GetDim(0);
        OP_CHECK_IF(
            (_Params.originN + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8 != _Params.fracN,
            OP_LOGE(context->GetNodeName(),
                    "QuantMatmulDequant get n-dim of weight_scale and quantized_weight not match, please check."),
            return false);
        _Params.originE = 1;
        auto scale_desc = context->GetInputDesc(idx - 1);
        _Params.scaleIsInt64 = false;
        if (scale_desc != nullptr) {
            _Params.scaleIsInt64 = (scale_desc->GetDataType() == ge::DT_INT64);
        }
    }

    auto bias = context->GetOptionalInputShape(idx++);
    OP_CHECK_IF(bias != nullptr,
                OP_LOGE(context->GetNodeName(), "QuantMatmulDequant not support bias for now, please check."),
                return false);

    auto x_scale = context->GetOptionalInputShape(idx++);
    _Params.isXScaleHalf = 0;
    if (x_scale == nullptr) {
        OP_CHECK_IF(_Params.perToken != true,
                    OP_LOGE(context->GetNodeName(),
                            "QuantMatmulDequant only support pertoken in dynamic quant mode, please check."),
                    return false);
        _Params.dynamicQuant = true;
    } else {
        shape = x_scale->GetStorageShape();
        if (_Params.perToken) {
            OP_CHECK_IF(
                shape.GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get x_scale shape ndim is not 1, please check."),
                return false);
            OP_CHECK_IF(_Params.originM != (uint64_t)shape.GetDim(0),
                        OP_LOGE(context->GetNodeName(),
                                "QuantMatmulDequant get m-dim of x_scale and x not match, please check."),
                        return false);
        } else {
            OP_CHECK_IF(shape.GetDimNum() != 1 && shape.GetDimNum() != 0,
                        OP_LOGE(context->GetNodeName(),
                                "QuantMatmulDequant get x_scale shape ndim is not 1 or 0, please check."),
                        return false);
            if (shape.GetDimNum() == 1) {
                OP_CHECK_IF(
                    1 != (uint64_t)shape.GetDim(0),
                    OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get x_scale shape not [1], please check."),
                    return false);
            }
        }
        _Params.dynamicQuant = false;

        auto x_scale_desc = context->GetOptionalInputDesc(idx - 1);
        if (x_scale_desc != nullptr) {
            auto x_scale_dtype = x_scale_desc->GetDataType();
            if (x_scale_dtype == ge::DT_FLOAT16) {
                _Params.isXScaleHalf = 1;
            }
            OP_CHECK_IF(
                x_scale_dtype == ge::DT_FLOAT16 && _Params.perToken,
                OP_LOGE(
                    context->GetNodeName(),
                    "QuantMatmulDequant x_scale_dtype == ge::DT_FLOAT16 only support pertensor mode, please check."),
                return false);
        }
    }

    auto x_offset = context->GetOptionalInputShape(idx++);
    OP_CHECK_IF(x_offset != nullptr,
                OP_LOGE(context->GetNodeName(), "QuantMatmulDequant not support x_offset for now, please check."),
                return false);

    auto smooth_scale = context->GetOptionalInputShape(idx++);
    if (smooth_scale == nullptr) {
        _Params.smoothScale = false;
    } else {
        shape = smooth_scale->GetStorageShape();
        OP_CHECK_IF(
            shape.GetDimNum() != 1,
            OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get smooth_scale shape ndim is not 1, please check."),
            return false);
        OP_CHECK_IF(_Params.originK != (uint64_t)shape.GetDim(0),
                    OP_LOGE(context->GetNodeName(),
                            "QuantMatmulDequant get k-dim of smooth_scale and x not match, please check."),
                    return false);
        _Params.smoothScale = true;
    }

    auto y = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y);
    shape = y->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != NUMBER_2,
                OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get y shape ndim is not 2, please check."),
                return false);
    OP_CHECK_IF(_Params.originM != (uint64_t)shape.GetDim(0),
                OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get m-dim of y and x not match, please check."),
                return false);
    OP_CHECK_IF(
        _Params.originN != (uint64_t)shape.GetDim(1),
        OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get n-dim of y and weight_scale not match, please check."),
        return false);
    return true;
}

bool QuantMatmulDequantTiling::GetPlatformInfo(gert::TilingContext *context)
{
    auto compileInfoPtr = reinterpret_cast<const QuantMatmulDequantCompileInfo *>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    _Params.CoreNum = compileInfoPtr->coreNum;
    _Params.UBSize = compileInfoPtr->ubSize;
    _Params.L1Size = compileInfoPtr->l1Size;
    _Params.L0ASize = compileInfoPtr->l0ASize;
    _Params.L0BSize = compileInfoPtr->l0BSize;
    _Params.L0CSize = compileInfoPtr->l0CSize;
    return true;
}

static bool ShouldUseOptimizedPertokenDynamic(const QuantMatmulDequantParam &params)
{
    const bool hasSmoothFp32Scale = params.smoothScale && !params.scaleIsInt64;

    // QWEN MoE decode: consistently faster in benchmark for E>=128, M<=64.
    if (params.originE >= 128ULL && params.originM <= 64ULL) {
        return true;
    }

    // QWEN 35B prefill-like cases: E=256 is faster across measured prefill shapes.
    if (params.originE >= 256ULL && params.originM >= 1024ULL) {
        return true;
    }

    // Sparse MoE shapes with enough experts and K=2048 benefit from the optimized split-distribution path.
    if (params.originE >= 8ULL && params.originM <= 640ULL && params.originK == 2048ULL && params.originN >= 768ULL) {
        return true;
    }

    // Tiny MoE measured positive; keep this narrow to avoid nearby light-tier regressions.
    if (params.originE >= 4ULL && params.originM <= 32ULL && params.originK <= 512ULL && params.originN == 256ULL) {
        return true;
    }

    // Dense large-N / large-prefill cases with clear or near-clear gains.
    if (params.originE == 1ULL && params.originM >= 512ULL && params.originK == 2048ULL && params.originN >= 4096ULL) {
        return true;
    }
    if (params.originE == 1ULL && params.originM >= 2048ULL && params.originK >= 4096ULL && params.originN >= 4096ULL) {
        return true;
    }
    if (params.originE == 1ULL && params.originN >= 65536ULL) {
        return true;
    }

    // Tall dense case only wins in the measured smooth fp32-scale variant.
    if (params.originE == 1ULL && params.originM >= 512ULL && params.originK >= 4096ULL && params.originN >= 2048ULL &&
        hasSmoothFp32Scale) {
        return true;
    }

    return false;
}

bool QuantMatmulDequantTiling::GetOptimizedPertokenDynamicTilingData(gert::TilingContext *context)
{
    if (!(isGrouped && _Params.dynamicQuant && _Params.perToken)) {
        return true;
    }
    if (!ShouldUseOptimizedPertokenDynamic(_Params)) {
        return true;
    }

    const uint64_t kBytes = _Params.originKAligned32;
    const uint64_t ubSize = _Params.UBSize;
    const uint64_t l0aTotal = _Params.L0ASize / NUMBER_2;
    const uint64_t l0bTotal = _Params.L0BSize / NUMBER_2;
    const uint64_t l0cSize = _Params.L0CSize;
    const uint64_t l1Size = _Params.L1Size;

    uint32_t phaseXChunkUB = NUMBER_128;
    while ((phaseXChunkUB << 1U) <= kBytes && phaseXChunkUB < 2048U) {
        phaseXChunkUB <<= 1U;
    }

    auto ubFits = [&](uint32_t mb, uint32_t nb) -> bool {
        const uint64_t chunk = phaseXChunkUB;
        uint64_t persistent = 2ULL * mb * nb * sizeof(uint16_t) + 2ULL * nb * sizeof(uint64_t) + mb * sizeof(float);
        if (_Params.originE == 1ULL) {
            persistent += static_cast<uint64_t>(mb) * nb * sizeof(uint16_t);
        }
        if (!_Params.scaleIsInt64) {
            persistent += nb * sizeof(uint32_t);
        }
        if (_Params.smoothScale) {
            persistent += kBytes * sizeof(uint16_t) + chunk * sizeof(float);
        }

        const uint64_t mbAlignedPhX = (mb < NM_FRACTAL_INT8) ? NM_FRACTAL_INT8 : mb;
        uint64_t phaseX = mbAlignedPhX * chunk + 2ULL * chunk * sizeof(uint16_t) + chunk * sizeof(uint16_t) +
                          2ULL * chunk * sizeof(float) + chunk * sizeof(int32_t) + chunk * sizeof(uint16_t);

        const uint64_t mbAligned = (mb < NM_FRACTAL_INT8) ? NM_FRACTAL_INT8 : mb;
        uint64_t phaseD = mbAligned * nb * sizeof(uint16_t);
        if (!_Params.scaleIsInt64) {
            phaseD += nb * sizeof(float);
        }

        return persistent + (phaseX > phaseD ? phaseX : phaseD) + 2048ULL <= ubSize;
    };

    auto l0Fits = [&](uint32_t mb, uint32_t nb, uint32_t kb) -> bool {
        return (static_cast<uint64_t>(mb) * kb <= l0aTotal) && (static_cast<uint64_t>(kb) * nb <= l0bTotal) &&
               (static_cast<uint64_t>(mb) * nb * sizeof(int32_t) <= l0cSize);
    };

    auto l1Fits = [&](uint32_t mb, uint32_t nb, uint32_t kb) -> bool {
        const uint64_t mbAligned = (mb < NM_FRACTAL_INT8) ? NM_FRACTAL_INT8 : mb;
        uint64_t xL1 = mbAligned * kBytes;
        uint64_t wL1 = 2ULL * kb * nb;
        return xL1 + wL1 <= l1Size;
    };

    uint32_t bestMb = 16;
    uint32_t bestNb = 16;
    uint64_t bestScore = UINT64_MAX;
    for (uint32_t nbI = 0; nbI < AscendC::NB_CANDIDATE_COUNT; ++nbI) {
        uint32_t nb = AscendC::NB_CANDIDATES[nbI];
        if (nb > _Params.originN) {
            continue;
        }
        for (uint32_t mbI = 0; mbI < AscendC::MB_CANDIDATE_COUNT; ++mbI) {
            uint32_t mb = AscendC::MB_CANDIDATES[mbI];
            if (mb < 32U) {
                continue;
            }
            uint64_t kbRaw =
                std::min<uint64_t>({kBytes, l0aTotal / mb, l0bTotal / nb, static_cast<uint64_t>(AscendC::MMAD_K_MAX)});
            uint32_t kb = static_cast<uint32_t>((kbRaw / K_FRACTAL_INT8) * K_FRACTAL_INT8);
            if (kb == 0 || !l0Fits(mb, nb, kb) || !l1Fits(mb, nb, kb) || !ubFits(mb, nb)) {
                continue;
            }

            const uint64_t weightBytes = static_cast<uint64_t>(_Params.originN) * kBytes;
            const uint64_t iters = (_Params.originM + mb - 1) / mb;
            const uint64_t cbPerIter = (_Params.originN + nb - 1) / nb;
            const uint64_t coreNum = static_cast<uint64_t>(NUM_OF_AICORE);
            const uint64_t perCoreIters = (iters + coreNum - 1) / coreNum;
            const uint64_t kpasses = (kBytes + kb - 1) / kb;
            const uint64_t mte2Bursts = perCoreIters * cbPerIter * kpasses;
            const uint64_t cyclesMte2W = ((weightBytes + coreNum - 1) / coreNum) / 17ULL + 422ULL * mte2Bursts;
            const uint64_t cyclesMte1A = perCoreIters * cbPerIter * kpasses * (static_cast<uint64_t>(mb) * kb) / 512ULL;
            const uint64_t cyclesMte1B = perCoreIters * cbPerIter * kpasses * (static_cast<uint64_t>(kb) * nb) / 256ULL;
            const uint64_t mte1Calls = perCoreIters * cbPerIter * kpasses * (static_cast<uint64_t>(mb) / 16ULL + 1ULL);
            const uint64_t cyclesMte1Issue = 32ULL * mte1Calls;
            const uint64_t cyclesCube =
                perCoreIters * cbPerIter * kpasses * (static_cast<uint64_t>(mb) * nb * kb) / 2048ULL;
            const uint64_t cyclesMte3 = perCoreIters * cbPerIter * (static_cast<uint64_t>(mb) * nb * 2ULL) / 17ULL;
            const uint64_t feedSum = cyclesMte2W + cyclesMte1A + cyclesMte1B + cyclesMte1Issue + cyclesMte3;
            const uint64_t score = (cyclesCube >= feedSum) ? cyclesCube : feedSum;
            if (score < bestScore || (score == bestScore && mb > bestMb) ||
                (score == bestScore && mb == bestMb && nb > bestNb)) {
                bestScore = score;
                bestMb = mb;
                bestNb = nb;
            }
        }
    }

    _Params.Mb = bestMb;
    _Params.Nb = bestNb;
    uint64_t kbRaw =
        std::min<uint64_t>({kBytes, l0aTotal / bestMb, l0bTotal / bestNb, static_cast<uint64_t>(AscendC::MMAD_K_MAX)});
    _Params.Kb = static_cast<uint32_t>((kbRaw / K_FRACTAL_INT8) * K_FRACTAL_INT8);
    _Params.kPasses = static_cast<uint32_t>((kBytes + _Params.Kb - 1) / _Params.Kb);

    const uint64_t mbAlignedXL1 = (_Params.Mb < NM_FRACTAL_INT8) ? NM_FRACTAL_INT8 : _Params.Mb;
    const uint64_t xL1Bytes = mbAlignedXL1 * kBytes;
    const uint64_t wL1OnePair = 2ULL * _Params.Kb * _Params.Nb;
    const uint64_t l1Free = (l1Size > xL1Bytes) ? (l1Size - xL1Bytes) : 0ULL;
    uint32_t wChunk = 1;
    for (uint32_t cand = 2; cand <= _Params.kPasses; cand <<= 1U) {
        if ((_Params.kPasses % cand) != 0 || cand * wL1OnePair > l1Free) {
            break;
        }
        wChunk = cand;
    }
    _Params.wChunkKPasses = wChunk;

    _Params.phaseXChunk = phaseXChunkUB;
    _Params.phaseXFullPairs = static_cast<uint32_t>(kBytes / phaseXChunkUB);
    _Params.phaseXTail = static_cast<uint32_t>(kBytes % phaseXChunkUB);
    _Params.maxOneTurnToken = _Params.Mb;
    _Params.maxWeightColOneTurn = _Params.Nb;
    _Params.smallM = (_Params.Mb <= AscendC::SMALL_M_THRESHOLD);
    _Params.weightBlockN1 = 0;

    if (_Params.originE > 1ULL || _Params.fracM >= NUM_OF_AICORE) {
        _Params.MCoreNum = NUM_OF_AICORE;
        _Params.NCoreNum = 1;
    } else {
        uint32_t chosen = 0;
        uint64_t minCost = (_Params.fracN << NUMBER_3) + _Params.fracM;
        for (uint32_t i = 1; i < NUMBER_4; ++i) {
            uint64_t cost = (_Params.fracN << (NUMBER_3 - i)) + (_Params.fracM << i);
            if (cost < minCost) {
                chosen = i;
                minCost = cost;
            }
        }
        _Params.NCoreNum = 1U << chosen;
        _Params.MCoreNum = NUM_OF_AICORE / _Params.NCoreNum;
    }

    uint64_t totalIters = (_Params.originM + _Params.Mb - 1) / _Params.Mb;
    _Params.perCoreIters = static_cast<uint32_t>((totalIters + _Params.CoreNum - 1) / _Params.CoreNum);
    _Params.remainderIters = static_cast<uint32_t>(totalIters % _Params.CoreNum);

    _Params.l2WeightFits = (_Params.originK * _Params.originN <= L2_HEADROOM_BYTES);
    const uint64_t wL1Bytes = 2ULL * _Params.wChunkKPasses * _Params.Kb * _Params.Nb;
    const uint64_t xRawL1Bytes = mbAlignedXL1 * kBytes * sizeof(uint16_t);
    _Params.hasXRawL1 = (xL1Bytes + wL1Bytes + xRawL1Bytes <= _Params.L1Size);
    _Params.useL2Hint = false;
    _Params.dbL0c = (static_cast<uint64_t>(_Params.Mb) * _Params.Nb * sizeof(int32_t) <= l0cSize / 2ULL);
    const bool stageXZN = (_Params.originE == 1 && _Params.originM >= NM_FRACTAL_INT8 && _Params.originN >= NUMBER_256);
    _Params.useXZNWorkspace = stageXZN;
    const uint64_t alignedM = ((_Params.originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8) * NM_FRACTAL_INT8;
    _Params.xZNWorkspaceBytes = stageXZN ? (alignedM * _Params.originKAligned32) : 0ULL;

    const uint32_t totalColBatch = static_cast<uint32_t>((_Params.originN + _Params.Nb - 1) / _Params.Nb);
    const uint64_t cbBytes = static_cast<uint64_t>(_Params.Nb) * _Params.originK;
    const uint64_t l2EffBytes =
        static_cast<uint64_t>(AscendC::L2_HEADROOM_BYTES) * AscendC::L2_WEIGHT_SHARE_PCT / 100ULL;
    uint32_t nBlockSize = (cbBytes > 0) ? static_cast<uint32_t>(l2EffBytes / cbBytes) : 1;
    if (nBlockSize == 0) {
        nBlockSize = 1;
    }
    if (nBlockSize > totalColBatch) {
        nBlockSize = totalColBatch;
    }
    _Params.nBlockSize = nBlockSize;

    uint32_t flags = QGMMD_FLAG_DYNAMIC_QUANT | QGMMD_FLAG_PER_TOKEN;
    if (_Params.smoothScale)
        flags |= QGMMD_FLAG_HAS_SMOOTH;
    if (_Params.scaleIsInt64)
        flags |= QGMMD_FLAG_SCALE_IS_INT64;
    if (_Params.weightBlockN1 != 0)
        flags |= QGMMD_FLAG_BLOCKED_ZN;
    if (_Params.smallM)
        flags |= QGMMD_FLAG_SMALL_M;
    if (_Params.l2WeightFits)
        flags |= QGMMD_FLAG_L2_WEIGHT_FITS;
    if (_Params.hasXRawL1)
        flags |= QGMMD_FLAG_HAS_X_RAW_L1;
    if (_Params.useL2Hint)
        flags |= QGMMD_FLAG_USE_L2_HINT;
    if (_Params.dbL0c)
        flags |= QGMMD_FLAG_DBL0C;
    if (_Params.useXZNWorkspace)
        flags |= QGMMD_FLAG_USE_XZN_WORKSPACE;
    _Params.flags = flags;
    tilingKey = QuantMatmulDequantTilingKey::GROUPED_DQ_PERTOKEN_OPT;
    return true;
}

bool QuantMatmulDequantTiling::GetTilingData(gert::TilingContext *context)
{
    _Params.originKAligned512 = (_Params.originK + L0_ADDR_ALIGN - 1) / L0_ADDR_ALIGN * L0_ADDR_ALIGN;
    _Params.originKAligned32 = (_Params.originK + K_FRACTAL_INT8 - 1) / K_FRACTAL_INT8 * K_FRACTAL_INT8;
    _Params.fracM = (_Params.originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
    _Params.tailM = (_Params.originM - 1) % NM_FRACTAL_INT8 + 1;
    // 最低的 0 ~ kTailN 位被设置为 1，其余位保持为 0
    _Params.ubKMask = 0;
    uint32_t kTailN = _Params.originKAligned32 - _Params.originK;
    for (uint32_t i = 0; i < K_FRACTAL_INT8; i++) {
        _Params.ubKMask = _Params.ubKMask << 1;
        if (i < kTailN) {
            _Params.ubKMask = (_Params.ubKMask | 1);
        }
    }
    OP_CHECK_IF(_Params.originK % FLOAT16_PERBLOCK != 0,
                OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get k-dim not align to 16, please check."),
                return false);
    OP_CHECK_IF(_Params.originN % FLOAT16_PERBLOCK != 0,
                OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get n-dim not align to 16, please check."),
                return false);
    OP_CHECK_IF(_Params.originM == 0,
                OP_LOGE(context->GetNodeName(), "QuantMatmulDequant get m-dim should not be 0, please check."),
                return false);

    if (isGrouped && (SWIFT_M_MIN <= _Params.originM && _Params.originM <= SWIFT_M_MAX) &&
        (SWIFT_K_MIN <= _Params.originK && _Params.originK <= SWIFT_K_MAX) &&
        (SWIFT_N_MIN <= _Params.originN && _Params.originN <= SWIFT_N_MAX) && (!_Params.perToken) &&
        (_Params.smoothScale)) {
        tilingKey = QuantMatmulDequantTilingKey::SWIFT;
        int32_t tmpThreshold = std::roundf((float)_Params.originK / (float)_Params.originN * GMM_GEMV_THRESHOLD_C1 +
                                           GMM_GEMV_THRESHOLD_C2);
        _Params.swiftGEMVThreshold = std::min(GMM_GMEV_THRESHOLD_MAX, std::max(GMM_GMEV_THRESHOLD_MIN, tmpThreshold));

        _Params.processXKBaseNMax =
            _Params.UBSize /
            (NM_FRACTAL_INT8 * sizeof(int16_t) + NM_FRACTAL_INT8 * sizeof(int16_t) + NM_FRACTAL_INT8 * sizeof(float) +
             (_Params.smoothScale ? (sizeof(int16_t) + sizeof(float)) : 0)) /
            NUMBER_256 * NUMBER_256;
        _Params.processXKloopPerfracM =
            (_Params.originKAligned32 + _Params.processXKBaseNMax - 1) / _Params.processXKBaseNMax;

        // case2
        uint32_t fracMSwift = (SWIFT_M_MAX + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
        uint32_t chosen = 0;
        uint32_t mte2Min = (_Params.fracN << NUMBER_3) + (fracMSwift << 0);
        for (int i = 1; i < NUMBER_4; i++) {
            uint32_t mte2Now = (_Params.fracN << (NUMBER_3 - i)) + (fracMSwift << i);
            if (mte2Now < mte2Min) {
                chosen = i;
                mte2Min = mte2Now;
            }
        }
        _Params.MCoreNum = 1 << (NUMBER_3 - chosen);
        _Params.NCoreNum = 1 << chosen;
        uint32_t singleCoreM = (fracMSwift + _Params.MCoreNum - 1) / _Params.MCoreNum;
        uint32_t singleCoreN = (_Params.fracN + _Params.NCoreNum - 1) / _Params.NCoreNum;
        uint32_t singleCoreNTail = (_Params.fracN - 1) % _Params.NCoreNum + 1;
        uint32_t singleCoreMNFractal = (singleCoreM) * (singleCoreN);
        uint32_t l0CMNFractal = NUMBER_256 - NUMBER_16;
        _Params.baseMNum = 1;
        _Params.baseNNum_2 = 1;
        int32_t baseM = 1;
        int32_t baseN = 1;
        if (singleCoreMNFractal > l0CMNFractal) {
            float MNPieces = (float)singleCoreMNFractal / (float)l0CMNFractal;
            float MDivN = (float)singleCoreM / (float)singleCoreN;
            float tmp = std::sqrt(MNPieces / MDivN);
            if (tmp == 0) {
                return false;
            }
            baseM = static_cast<uint32_t>(floor((float)singleCoreM / (MDivN * tmp)));
            baseN = static_cast<uint32_t>(floor((float)singleCoreN / tmp));

            _Params.baseMNum = (singleCoreM + baseM - 1) / baseM;
            _Params.baseNNum_2 = (singleCoreN + baseN - 1) / baseN;
            if (singleCoreM > singleCoreN) {
                uint32_t BaseMFractalN_ = (singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum + 1;
                while (_Params.baseNNum_2 > 1 &&
                       BaseMFractalN_ * ((singleCoreN + _Params.baseNNum_2 - NUMBER_2) / (_Params.baseNNum_2 - 1)) <=
                           l0CMNFractal) {
                    _Params.baseNNum_2 -= 1;
                }
                uint32_t BaseNFractalN_ = (singleCoreN + _Params.baseNNum_2 - 1) / _Params.baseNNum_2;
                while (_Params.baseMNum > 1 &&
                       BaseNFractalN_ * ((singleCoreM + _Params.baseMNum - NUMBER_2) / (_Params.baseMNum - 1)) <=
                           l0CMNFractal) {
                    _Params.baseMNum -= 1;
                }
            } else {
                uint32_t BaseNFractalN_ = (singleCoreN + _Params.baseNNum_2 - 1) / _Params.baseNNum_2;
                while (_Params.baseMNum > 1 &&
                       BaseNFractalN_ * ((singleCoreM + _Params.baseMNum - NUMBER_2) / (_Params.baseMNum - 1)) <=
                           l0CMNFractal) {
                    _Params.baseMNum -= 1;
                }
                uint32_t BaseMFractalN_ = (singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum + 1;
                while (_Params.baseNNum_2 > 1 &&
                       BaseMFractalN_ * ((singleCoreN + _Params.baseNNum_2 - NUMBER_2) / (_Params.baseNNum_2 - 1)) <=
                           l0CMNFractal) {
                    _Params.baseNNum_2 -= 1;
                }
            }
        }

        uint32_t BaseMNumL0AB = (singleCoreM + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        uint32_t BaseNNumL0AB = (singleCoreN + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        if (_Params.baseMNum < BaseMNumL0AB)
            _Params.baseMNum = BaseMNumL0AB;
        if (_Params.baseNNum_2 < BaseNNumL0AB)
            _Params.baseNNum_2 = BaseNNumL0AB;

        baseM = (singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum;
        _Params.baseN_2 = (singleCoreN + _Params.baseNNum_2 - 1) / _Params.baseNNum_2;

        uint32_t l0AMKFractal = NUMBER_128 / NUMBER_2 / baseM;
        uint32_t l0BNKFractal = NUMBER_128 / NUMBER_2 / _Params.baseN_2;
        uint32_t MKPieces = (_Params.fracK + l0AMKFractal - 1) / l0AMKFractal;
        uint32_t NKPieces = (_Params.fracK + l0BNKFractal - 1) / l0BNKFractal;
        _Params.baseKNum_2 = MKPieces > NKPieces ? MKPieces : NKPieces;
        _Params.baseK_2 = (_Params.fracK + _Params.baseKNum_2 - 1) / _Params.baseKNum_2;
        _Params.baseKTail_2 = (_Params.fracK - 1) % _Params.baseKNum_2 + 1;

        fracMSwift = (SWIFT_M_MIN + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
        chosen = 0;
        mte2Min = (_Params.fracN << NUMBER_3) + (fracMSwift << 0);
        for (int i = 1; i < NUMBER_4; i++) {
            uint32_t mte2Now = (_Params.fracN << (NUMBER_3 - i)) + (fracMSwift << i);
            if (mte2Now < mte2Min) {
                chosen = i;
                mte2Min = mte2Now;
            }
        }
        _Params.MCoreNum = 1 << (NUMBER_3 - chosen);
        _Params.NCoreNum = 1 << chosen;
        singleCoreM = (fracMSwift + _Params.MCoreNum - 1) / _Params.MCoreNum;
        singleCoreN = (_Params.fracN + _Params.NCoreNum - 1) / _Params.NCoreNum;
        singleCoreNTail = (_Params.fracN - 1) % _Params.NCoreNum + 1;
        singleCoreMNFractal = (singleCoreM) * (singleCoreN);
        l0CMNFractal = NUMBER_256 - NUMBER_16;
        _Params.baseMNum = 1;
        _Params.baseNNum = 1;

        BaseMNumL0AB = (singleCoreM + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        BaseNNumL0AB = (singleCoreN + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        if (_Params.baseMNum < BaseMNumL0AB)
            _Params.baseMNum = BaseMNumL0AB;
        if (_Params.baseNNum < BaseNNumL0AB)
            _Params.baseNNum = BaseNNumL0AB;

        baseM = (singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum;
        baseN = (singleCoreN + _Params.baseNNum - 1) / _Params.baseNNum;

        l0AMKFractal = NUMBER_128 / NUMBER_2 / baseM;
        l0BNKFractal = NUMBER_128 / NUMBER_2 / baseN;
        MKPieces = (_Params.fracK + l0AMKFractal - 1) / l0AMKFractal;
        NKPieces = (_Params.fracK + l0BNKFractal - 1) / l0BNKFractal;
        _Params.baseKNum = MKPieces > NKPieces ? MKPieces : NKPieces;
        _Params.baseK = (_Params.fracK + _Params.baseKNum - 1) / _Params.baseKNum;
        _Params.baseKTail = (_Params.fracK - 1) % _Params.baseKNum + 1;
        _Params.singleCoreN = singleCoreN;
        _Params.singleCoreNTail = singleCoreNTail;

        _Params.singleCoreFracN = (_Params.fracN + _Params.CoreNum - 1) / _Params.CoreNum;
        _Params.singleCoreFracNTail = (_Params.fracN - 1) % _Params.CoreNum + 1;
        _Params.baseFracK = NUMBER_16;
        _Params.baseFracN = NUMBER_4;
    } else if (isGrouped) {
        tilingKey = QuantMatmulDequantTilingKey::GROUPED;
        _Params.processXKBaseNMax =
            (_Params.UBSize - (_Params.perToken ? (NM_FRACTAL_INT8 * BLOCK_SIZE) : 0)) /
            (NM_FRACTAL_INT8 * sizeof(int16_t) + NM_FRACTAL_INT8 * sizeof(int16_t) + NM_FRACTAL_INT8 * sizeof(float) +
             (_Params.smoothScale ? (sizeof(int16_t) + sizeof(float)) : 0)) /
            NUMBER_256 * NUMBER_256;
        _Params.singleCoreFracN = (_Params.fracN + _Params.CoreNum - 1) / _Params.CoreNum;
        _Params.singleCoreFracNTail = (_Params.fracN - 1) % _Params.CoreNum + 1;
        _Params.baseFracK = NUMBER_16;
        _Params.baseFracN = NUMBER_4;
    } else if (_Params.originM > GEMV_THRESHOLD ||
               (_Params.originKAligned512 * _Params.originM) > (_Params.L1Size / NUMBER_2)) {
        tilingKey = QuantMatmulDequantTilingKey::NORMAL;
        _Params.fracM = (_Params.originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
        _Params.tailM = (_Params.originM - 1) % NM_FRACTAL_INT8 + 1;

        _Params.processXKBaseNMax =
            (_Params.UBSize - (_Params.perToken ? (NM_FRACTAL_INT8 * BLOCK_SIZE) : 0)) /
            (NM_FRACTAL_INT8 * sizeof(int16_t) + NM_FRACTAL_INT8 * sizeof(int16_t) + NM_FRACTAL_INT8 * sizeof(float) +
             (_Params.smoothScale ? (sizeof(int16_t) + sizeof(float)) : 0)) /
            NUMBER_256 * NUMBER_256;
        _Params.processXKloopPerfracM =
            (_Params.originKAligned32 + _Params.processXKBaseNMax - 1) / _Params.processXKBaseNMax;
        _Params.processXKloop = (_Params.fracM * _Params.processXKloopPerfracM + _Params.CoreNum - 1) / _Params.CoreNum;
        _Params.processXKloopPerfracM = _Params.processXKloop * _Params.CoreNum / _Params.fracM;
        _Params.processXKBaseN =
            (_Params.fracK + _Params.processXKloopPerfracM - 1) / _Params.processXKloopPerfracM * K_FRACTAL_INT8;
        _Params.processXKTailN = _Params.fracK % _Params.processXKloopPerfracM;
        _Params.processXKTailN = _Params.processXKTailN == 0 ? _Params.processXKloopPerfracM : _Params.processXKTailN;

        uint32_t chosen = 0;
        uint32_t mte2Min = (_Params.fracN << NUMBER_3) + (_Params.fracM << 0);
        for (int i = 1; i < NUMBER_4; i++) {
            uint32_t mte2Now = (_Params.fracN << (NUMBER_3 - i)) + (_Params.fracM << i);
            if (mte2Now < mte2Min) {
                chosen = i;
                mte2Min = mte2Now;
            }
        }
        _Params.MCoreNum = 1 << (NUMBER_3 - chosen);
        _Params.NCoreNum = 1 << chosen;
        _Params.singleCoreM = (_Params.fracM + _Params.MCoreNum - 1) / _Params.MCoreNum;
        _Params.singleCoreN = (_Params.fracN + _Params.NCoreNum - 1) / _Params.NCoreNum;
        _Params.singleCoreMTail = (_Params.fracM - 1) % _Params.MCoreNum + 1;
        _Params.singleCoreNTail = (_Params.fracN - 1) % _Params.NCoreNum + 1;
        uint32_t l0CMNFractal = NUMBER_256;
        uint32_t oriBaseMN = static_cast<uint32_t>(floor(std::sqrt((float)l0CMNFractal)));
        uint32_t extraN = _Params.perToken ? 1 : 0;
        uint32_t singleCoreMNFractal = (_Params.singleCoreM + 1) * (_Params.singleCoreN + extraN);
        _Params.baseMNum = 1;
        _Params.baseNNum = 1;
        if (singleCoreMNFractal > l0CMNFractal) {
            _Params.baseMNum = (_Params.singleCoreM + (oriBaseMN - 1) - 1) / (oriBaseMN - 1);
            _Params.baseNNum = (_Params.singleCoreN + (oriBaseMN - extraN) - 1) / (oriBaseMN - extraN);
            if (_Params.singleCoreM > _Params.singleCoreN) {
                uint32_t BaseMFractalN_ = (_Params.singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum + 1;
                while (_Params.baseNNum > 1 &&
                       BaseMFractalN_ * ((_Params.singleCoreN + _Params.baseNNum - NUMBER_2) / (_Params.baseNNum - 1) +
                                         extraN) <=
                           l0CMNFractal) {
                    _Params.baseNNum -= 1;
                }
                uint32_t BaseNFractalN_ = (_Params.singleCoreN + _Params.baseNNum - 1) / _Params.baseNNum + extraN;
                while (_Params.baseMNum > 1 &&
                       BaseNFractalN_ *
                               ((_Params.singleCoreM + _Params.baseMNum - NUMBER_2) / (_Params.baseMNum - 1) + 1) <=
                           l0CMNFractal) {
                    _Params.baseMNum -= 1;
                }
            } else {
                uint32_t BaseNFractalN_ = (_Params.singleCoreN + _Params.baseNNum - 1) / _Params.baseNNum + extraN;
                while (_Params.baseMNum > 1 &&
                       BaseNFractalN_ *
                               ((_Params.singleCoreM + _Params.baseMNum - NUMBER_2) / (_Params.baseMNum - 1) + 1) <=
                           l0CMNFractal) {
                    _Params.baseMNum -= 1;
                }
                uint32_t BaseMFractalN_ = (_Params.singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum + 1;
                while (_Params.baseNNum > 1 &&
                       BaseMFractalN_ * ((_Params.singleCoreN + _Params.baseNNum - NUMBER_2) / (_Params.baseNNum - 1) +
                                         extraN) <=
                           l0CMNFractal) {
                    _Params.baseNNum -= 1;
                }
            }
        }

        uint32_t BaseMNumL0AB = (_Params.singleCoreM + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        uint32_t BaseNNumL0AB = (_Params.singleCoreN + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        if (_Params.baseMNum < BaseMNumL0AB)
            _Params.baseMNum = BaseMNumL0AB;
        if (_Params.baseNNum < BaseNNumL0AB)
            _Params.baseNNum = BaseNNumL0AB;

        uint32_t baseM = (_Params.singleCoreM + _Params.baseMNum - 1) / _Params.baseMNum;
        uint32_t baseN = (_Params.singleCoreN + _Params.baseNNum - 1) / _Params.baseNNum;
        uint32_t l0AMKFractal = NUMBER_128 / NUMBER_2 / baseM;
        uint32_t l0BNKFractal = NUMBER_128 / NUMBER_2 / baseN;
        uint32_t MKPieces = (_Params.fracK + l0AMKFractal - 1) / l0AMKFractal;
        uint32_t NKPieces = (_Params.fracK + l0BNKFractal - 1) / l0BNKFractal;
        _Params.baseKNum = MKPieces > NKPieces ? MKPieces : NKPieces;
    } else {
        tilingKey = QuantMatmulDequantTilingKey::GEMV;
        _Params.singleCoreFracN = (_Params.fracN + _Params.CoreNum - 1) / _Params.CoreNum;
        _Params.singleCoreFracNTail = (_Params.fracN - 1) % _Params.CoreNum + 1;
        _Params.baseFracK = NUMBER_16;
        _Params.baseFracN = NUMBER_4;
        _Params.baseFracNL0C = _Params.UBSize / sizeof(int32_t) / L0C_FRACTAL / L0C_FRACTAL / _Params.originM;
        _Params.ubBaseK = (_Params.UBSize - (_Params.perToken ? _Params.originM * BLOCK_SIZE : 0)) /
                          (_Params.originM * NUMBER_3 + NUMBER_4 + (_Params.smoothScale ? (NUMBER_2 + NUMBER_4) : 0)) /
                          NUMBER_256 * NUMBER_256;
        _Params.ubIterK = (_Params.originK + _Params.ubBaseK - 1) / _Params.ubBaseK;
        _Params.ubBaseKTail = (_Params.originK - 1) % _Params.ubBaseK + 1;
    }

    if (_Params.dynamicQuant) {
        _Params.dynamicBaseK = (_Params.UBSize / NUMBER_2 - BLOCK_SIZE * NUMBER_2 - BLOCK_SIZE * NUMBER_2) /
                               (1 + (_Params.smoothScale ? 1 : 0)) / NUMBER_2 / NUMBER_256 * NUMBER_256;
        _Params.dynamicBaseK = _Params.dynamicBaseK > REDUCE_THRESHOLD ? REDUCE_THRESHOLD : _Params.dynamicBaseK;
        _Params.dynamicIterK = (_Params.originK + _Params.dynamicBaseK - 1) / _Params.dynamicBaseK;
        _Params.dynamicBaseKTail = (_Params.originK - 1) % _Params.dynamicBaseK + 1;
    }

    OP_CHECK_IF(!GetOptimizedPertokenDynamicTilingData(context),
                OP_LOGE(context->GetNodeName(), "get optimized tiling data fail."), return false);

    return true;
}

bool QuantMatmulDequantTiling::SetTilingData(gert::TilingContext *context)
{
    tilingData.set_CoreNum(_Params.CoreNum);
    tilingData.set_perToken(_Params.perToken);
    tilingData.set_dynamicQuant(_Params.dynamicQuant);
    tilingData.set_smoothScale(_Params.smoothScale);
    tilingData.set_originE(_Params.originE);
    tilingData.set_originM(_Params.originM);
    tilingData.set_originN(_Params.originN);
    tilingData.set_originK(_Params.originK);
    tilingData.set_originKAligned32(_Params.originKAligned32);
    tilingData.set_originKAligned512(_Params.originKAligned512);
    tilingData.set_fracN(_Params.fracN);
    tilingData.set_fracK(_Params.fracK);
    // dynamic
    tilingData.set_dynamicBaseK(_Params.dynamicBaseK);
    tilingData.set_dynamicIterK(_Params.dynamicIterK);
    tilingData.set_dynamicBaseKTail(_Params.dynamicBaseKTail);
    // gemv
    tilingData.set_singleCoreFracN(_Params.singleCoreFracN);
    tilingData.set_singleCoreFracNTail(_Params.singleCoreFracNTail);
    tilingData.set_baseFracN(_Params.baseFracN);
    tilingData.set_baseFracK(_Params.baseFracK);
    tilingData.set_baseFracNL0C(_Params.baseFracNL0C);
    tilingData.set_ubBaseK(_Params.ubBaseK);
    tilingData.set_ubIterK(_Params.ubIterK);
    tilingData.set_ubBaseKTail(_Params.ubBaseKTail);
    // normal
    tilingData.set_fracM(_Params.fracM);
    tilingData.set_tailM(_Params.tailM);
    tilingData.set_processXKBaseNMax(_Params.processXKBaseNMax);
    tilingData.set_processXKBaseN(_Params.processXKBaseN);
    tilingData.set_processXKloop(_Params.processXKloop);
    tilingData.set_processXKloopPerfracM(_Params.processXKloopPerfracM);
    tilingData.set_processXKTailN(_Params.processXKTailN);
    tilingData.set_MMmod(_Params.MMmod);
    tilingData.set_MCoreNum(_Params.MCoreNum);
    tilingData.set_NCoreNum(_Params.NCoreNum);
    tilingData.set_singleCoreM(_Params.singleCoreM);
    tilingData.set_singleCoreN(_Params.singleCoreN);
    tilingData.set_singleCoreMTail(_Params.singleCoreMTail);
    tilingData.set_singleCoreNTail(_Params.singleCoreNTail);
    tilingData.set_baseMNum(_Params.baseMNum);
    tilingData.set_baseNNum(_Params.baseNNum);
    tilingData.set_baseKNum(_Params.baseKNum);
    // swift
    tilingData.set_swiftGEMVThreshold(_Params.swiftGEMVThreshold);
    tilingData.set_baseNNum_2(_Params.baseNNum_2);
    tilingData.set_baseKNum_2(_Params.baseKNum_2);
    tilingData.set_baseK(_Params.baseK);
    tilingData.set_baseKTail(_Params.baseKTail);
    tilingData.set_baseK_2(_Params.baseK_2);
    tilingData.set_baseKTail_2(_Params.baseKTail_2);
    tilingData.set_ubKMask(_Params.ubKMask);
    tilingData.set_isXScaleHalf(_Params.isXScaleHalf);
    tilingData.set_coreNum(static_cast<uint32_t>(_Params.CoreNum));
    tilingData.set_M(static_cast<uint32_t>(_Params.originM));
    tilingData.set_K(static_cast<uint32_t>(_Params.originK));
    tilingData.set_N(static_cast<uint32_t>(_Params.originN));
    tilingData.set_E(static_cast<uint32_t>(_Params.originE));
    tilingData.set_Mb(_Params.Mb);
    tilingData.set_Nb(_Params.Nb);
    tilingData.set_Kb(_Params.Kb);
    tilingData.set_kPasses(_Params.kPasses);
    tilingData.set_wChunkKPasses(_Params.wChunkKPasses);
    tilingData.set_phaseXChunk(_Params.phaseXChunk);
    tilingData.set_phaseXFullPairs(_Params.phaseXFullPairs);
    tilingData.set_phaseXTail(_Params.phaseXTail);
    tilingData.set_maxOneTurnToken(_Params.maxOneTurnToken);
    tilingData.set_maxWeightColOneTurn(_Params.maxWeightColOneTurn);
    tilingData.set_perCoreIters(_Params.perCoreIters);
    tilingData.set_remainderIters(_Params.remainderIters);
    tilingData.set_weightBlockN1(_Params.weightBlockN1);
    tilingData.set_nBlockSize(_Params.nBlockSize);
    tilingData.set_flags(_Params.flags);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return true;
}

bool QuantMatmulDequantTiling::SetLaunchInfo(gert::TilingContext *context)
{
    context->SetBlockDim(_Params.CoreNum);

    context->SetTilingKey(static_cast<uint64_t>(tilingKey));

    int64_t workspaceSize = SYS_WORKSPACE_310P;
    if (tilingKey == QuantMatmulDequantTilingKey::GROUPED_DQ_PERTOKEN_OPT) {
        workspaceSize += _Params.xZNWorkspaceBytes;
        if (_Params.useXZNWorkspace) {
            workspaceSize += _Params.CoreNum * BLOCK_SIZE;
            workspaceSize += ((_Params.dynamicQuant && _Params.perToken) ? _Params.originM * BLOCK_SIZE : 0);
        }
    } else {
        workspaceSize += _Params.CoreNum * BLOCK_SIZE +
                         ((_Params.dynamicQuant && _Params.perToken) ? _Params.originM * BLOCK_SIZE : 0);
        if (tilingKey != QuantMatmulDequantTilingKey::GEMV) {
            workspaceSize +=
                (_Params.originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8 * NM_FRACTAL_INT8 * _Params.originKAligned32;
        }
    }
    AddWorkspace(context, workspaceSize);
    return true;
}

bool QuantMatmulDequantTiling::GetCheckAttr(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    //  optional
    const std::string x_quant_mode = static_cast<std::string>(attrs->GetStr(0));
    const bool *transpose_weight = attrs->GetAttrPointer<bool>(1);

    OP_CHECK_IF(
        x_quant_mode != "pertoken" && x_quant_mode != "pertensor",
        OP_LOGE(context->GetNodeName(),
                "QuantMatmulDequant attr x_quant_mode only support \"pertoken\" or \"pertensor\", please check."),
        return false);
    OP_CHECK_IF(
        *transpose_weight == false,
        OP_LOGE(context->GetNodeName(), "QuantMatmulDequant attr transpose_weight only support true, please check."),
        return false);

    _Params.perToken = x_quant_mode == "pertoken" ? true : false;
    return true;
}

ge::graphStatus QuantMatmulDequantTiling::runTiling(gert::TilingContext *context, bool is_grouped)
{
    isGrouped = is_grouped;
    _Params = QuantMatmulDequantParam{};
    // 310P AscendC platformINFO
    OP_CHECK_IF(!GetPlatformInfo(context), OP_LOGE(context->GetNodeName(), "get platforminfo fail."),
                return ge::GRAPH_FAILED);
    // attr
    OP_CHECK_IF(!GetCheckAttr(context), OP_LOGE(context->GetNodeName(), "check attr fail."), return ge::GRAPH_FAILED);
    // shape
    OP_CHECK_IF(!CheckInOutShapes(context), OP_LOGE(context->GetNodeName(), "check shape fail."),
                return ge::GRAPH_FAILED);
    // calculate tilingdata
    OP_CHECK_IF(!GetTilingData(context), OP_LOGE(context->GetNodeName(), "get tiling data fail."),
                return ge::GRAPH_FAILED);

    // tilingdata
    OP_CHECK_IF(!SetTilingData(context), OP_LOGE(context->GetNodeName(), "set tiling data fail."),
                return ge::GRAPH_FAILED);

    // launchinfo: tilingkey, workspace, blockdim
    OP_CHECK_IF(!SetLaunchInfo(context), OP_LOGE(context->GetNodeName(), "set launchinfo fail."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForQuantGroupedMatmulDequant(gert::TilingContext *context)
{
    QuantMatmulDequantTiling tiling_handle;
    context->SetScheduleMode(BATCH_MODE);
    return tiling_handle.runTiling(context, true);
}

ge::graphStatus TilingPrepareForQuantGroupedMatmulDequant(gert::TilingParseContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<QuantMatmulDequantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    std::string val;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    OP_CHECK_IF(!val.empty(), OP_LOGE(context->GetNodeName(), "QuantMatmulDequant support only ASCEND310P for now"),
                return false);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfoPtr->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    OP_CHECK_IF((compileInfoPtr->coreNum == 0 || compileInfoPtr->ubSize == 0 || compileInfoPtr->l1Size == 0 ||
                 compileInfoPtr->l0CSize == 0 || compileInfoPtr->l0ASize == 0 || compileInfoPtr->l0BSize == 0),
                OP_LOGE(context->GetNodeName(),
                        "platform invalid, coreNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
                        compileInfoPtr->coreNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
                        compileInfoPtr->l0CSize, compileInfoPtr->l0ASize, compileInfoPtr->l0BSize),
                return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "Parse compile info success, soc: %d",
            static_cast<int>(compileInfoPtr->socVersion));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantGroupedMatmulDequant)
    .Tiling(TilingForQuantGroupedMatmulDequant)
    .TilingParse<QuantMatmulDequantCompileInfo>(TilingPrepareForQuantGroupedMatmulDequant);

} // namespace optiling
