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
 * \file kda_input_proj_tiling_sigmoid.h
 * \brief Stage2 AIV: Sigmoid on beta. CalcTiling fills elemNum/ubTile;
 *        FillAivSplit is called from WriteTilingResult after blockDim is known.
 */

#ifndef KDA_INPUT_PROJ_TILING_SIGMOID_H
#define KDA_INPUT_PROJ_TILING_SIGMOID_H

#include <algorithm>
#include <cstdint>

#include "kda_input_proj_tiling.h"

namespace optiling {
class KdaInputProjSigmoidTiling {
public:
    explicit KdaInputProjSigmoidTiling(const KdaInputProjTilingInfo &tilingInfo)
        : tilingInfo_(tilingInfo)
    {}

    ge::graphStatus CalcTiling(KdaInputProjSigmoidParams &params) const
    {
        params = {};
        const KdaInputProjBaseParams &bp = tilingInfo_.baseParams;
        const char *opName = tilingInfo_.opName != nullptr ? tilingInfo_.opName : "KdaInputProj";

        const uint64_t elemNumU64 = static_cast<uint64_t>(bp.tSize) * static_cast<uint64_t>(bp.betaSize);
        OP_CHECK_IF(elemNumU64 > UINT32_MAX, OP_LOGE(opName, "Sigmoid elemNum T*N_beta overflow."),
                    return ge::GRAPH_FAILED);
        params.elemNum = static_cast<uint32_t>(elemNumU64);

        uint64_t ubSize = 0;
        if (tilingInfo_.platformInfo != nullptr) {
            platform_ascendc::PlatformAscendC plat(tilingInfo_.platformInfo);
            plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        }
        if (ubSize == 0) {
            ubSize = kDefaultUbSize;
        }
        OP_CHECK_IF(ubSize <= kUbReserve,
                    OP_LOGE(opName, "Sigmoid UB size %lu is too small.", static_cast<unsigned long>(ubSize)),
                    return ge::GRAPH_FAILED);

        const uint64_t avail = ubSize - kUbReserve;
        const uint64_t maxElem = avail / kBytesPerElem;
        params.ubTile = static_cast<uint32_t>((maxElem / kAlignElem) * kAlignElem);
        OP_CHECK_IF(params.ubTile == 0, OP_LOGE(opName, "Sigmoid ubTile is 0."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus FillAivSplit(KdaInputProjSigmoidParams &params, uint32_t launchedAiv) const
    {
        const char *opName = tilingInfo_.opName != nullptr ? tilingInfo_.opName : "KdaInputProj";
        if (params.elemNum == 0) {
            params.aivNum = 0;
            params.blockTile = 0;
            params.ubLoop = 0;
            params.ubTail = 0;
            params.tailUbLoop = 0;
            params.tailUbTail = 0;
            return ge::GRAPH_SUCCESS;
        }
        OP_CHECK_IF(params.ubTile == 0, OP_LOGE(opName, "Sigmoid FillAivSplit ubTile is 0."), return ge::GRAPH_FAILED);

        const uint32_t avail = launchedAiv == 0 ? 1U : launchedAiv;
        const uint64_t bits = static_cast<uint64_t>(params.elemNum) * static_cast<uint64_t>(kFp32Bits);
        uint32_t coreByBytes = static_cast<uint32_t>((bits + static_cast<uint64_t>(kMinTilingBits) - 1ULL) /
                                                     static_cast<uint64_t>(kMinTilingBits));
        if (coreByBytes == 0) {
            coreByBytes = 1;
        }
        uint32_t used = std::min(coreByBytes, avail);
        used = std::min(used, params.elemNum);
        used = std::max(used, 1U);
        params.blockTile = AlignUp(CeilDiv(params.elemNum, used), kAlignElem);
        OP_CHECK_IF(params.blockTile == 0, OP_LOGE(opName, "Sigmoid blockTile is 0."), return ge::GRAPH_FAILED);

        // UB tile 不必大于单核工作量：ubTile 取 min(UB 上限, blockTile)，避免按满 UB 申请。
        // blockTile 已 64 对齐，AlignUp 只为兜底。
        params.ubTile = std::min(params.ubTile, AlignUp(params.blockTile, kAlignElem));
        OP_CHECK_IF(params.ubTile == 0, OP_LOGE(opName, "Sigmoid ubTile is 0 after clamp."), return ge::GRAPH_FAILED);

        const uint32_t blockNum = CeilDiv(params.elemNum, params.blockTile);
        FillUbLoops(params, blockNum);

        OP_LOGI(opName,
                "Sigmoid tiling: elemNum=%u aivNum=%u blockTile=%u ubTile=%u "
                "ubLoop/tailUbLoop=%u/%u ubTail/tailUbTail=%u/%u launchedAiv=%u.",
                params.elemNum, params.aivNum, params.blockTile, params.ubTile, params.ubLoop, params.tailUbLoop,
                params.ubTail, params.tailUbTail, launchedAiv);
        return ge::GRAPH_SUCCESS;
    }

private:
    static constexpr uint32_t kMinTilingBits = 32768; // 4 KiB in bits
    static constexpr uint32_t kFp32Bits = 32;
    static constexpr uint32_t kAlignElem = 64;
    static constexpr uint64_t kUbReserve = 2048;
    static constexpr uint64_t kBytesPerElem = 16; // 2 queues * 2 buffers * 4 B
    static constexpr uint64_t kDefaultUbSize = 253952;

    static uint32_t CeilDiv(uint32_t a, uint32_t b)
    {
        return (b == 0) ? 0 : (a + b - 1U) / b;
    }

    static uint32_t AlignUp(uint32_t a, uint32_t align)
    {
        return (align == 0) ? a : CeilDiv(a, align) * align;
    }

    static void FillUbLoops(KdaInputProjSigmoidParams &params, uint32_t blockNum)
    {
        params.aivNum = blockNum;
        params.ubLoop = CeilDiv(params.blockTile, params.ubTile);
        params.ubTail = params.blockTile - (params.ubLoop - 1U) * params.ubTile;
        const uint32_t blockTail = params.elemNum - (blockNum - 1U) * params.blockTile;
        params.tailUbLoop = CeilDiv(blockTail, params.ubTile);
        params.tailUbTail = blockTail - (params.tailUbLoop - 1U) * params.ubTile;
    }

    const KdaInputProjTilingInfo &tilingInfo_;
};
} // namespace optiling

#endif // KDA_INPUT_PROJ_TILING_SIGMOID_H
