/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_info_parser.h"

#include <string>

#include "../flash_attn_grad_tiling.h" // FlashAttnGradCompileInfo：platform 缺失时的回退来源
#include "../plan/flash_attn_grad_sparse.h"
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

bool OptionalInputExists(gert::TilingContext *context, size_t index)
{
    const auto shape = context->GetOptionalInputShape(index);
    return (shape != nullptr) && (shape->GetStorageShape().GetDimNum() > 0);
}

// attn_mask 只在 mask_mode 为 3/4 时有意义；mask_mode=0 下即便传了也不算数
// （checker 会拒绝这种组合，这里保持与之一致的判定）。
bool HasAttenMask(gert::TilingContext *context, int64_t maskMode)
{
    if (maskMode == MASK_MODE_NO_MASK || (maskMode != MASK_MODE_CAUSAL && maskMode != MASK_MODE_WINDOW)) {
        return false;
    }
    return OptionalInputExists(context, ATTN_MASK_INDEX);
}

// 把两种非 TND 布局折叠成同一种四维紧凑视图 [viewD0, S, viewD2, D]：
//   BSND [B,S,N,D] -> 视图 [B,   S, N, D]，索引 [b,     s, n, 0]
//   BNSD [B,N,S,D] -> 视图 [B*N, S, 1, D]，索引 [b*N+n, s, 0, 0]
//   index0 = b * coefB0 + head * coefN0 ; index2 = head * coefN2
// 于是 kernel 只需一份 pl.load，布局差异退化成运行期算术，不占 tilingkey 位。
// Q 侧有 N1 = N2*G 个 head、KV 侧只有 N2 个，故 view/coefB0 分两套；
// coefN0/coefN2 只取决于布局，两套共用。
void FillLayoutView(FagParsedInfo &info)
{
    if (info.isBnsd) {
        info.viewD0Q = info.b * info.n1;
        info.viewD2Q = 1;
        info.coefB0Q = info.n1;
        info.viewD0KV = info.b * info.n2;
        info.viewD2KV = 1;
        info.coefB0KV = info.n2;
        info.coefN0 = 1;
        info.coefN2 = 0;
    } else {
        info.viewD0Q = info.b;
        info.viewD2Q = info.n1;
        info.coefB0Q = 1;
        info.viewD0KV = info.b;
        info.viewD2KV = info.n2;
        info.coefB0KV = 1;
        info.coefN0 = 0;
        info.coefN2 = 1;
    }
}

ge::graphStatus ParseAttrs(gert::TilingContext *context, FagParsedInfo &info)
{
    const auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const auto scalePtr = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
        if (scalePtr != nullptr) {
            info.scaleValue = *scalePtr;
        }
        const auto maskModePtr = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);
        if (maskModePtr != nullptr) {
            info.maskMode = *maskModePtr;
        }
        const auto winLeftPtr = attrs->GetAttrPointer<int64_t>(ATTR_WIN_LEFT_INDEX);
        if (winLeftPtr != nullptr) {
            info.winLeft = *winLeftPtr;
        }
        const auto winRightPtr = attrs->GetAttrPointer<int64_t>(ATTR_WIN_RIGHT_INDEX);
        if (winRightPtr != nullptr) {
            info.winRight = *winRightPtr;
        }
        const auto maxSeqlenQPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);
        if (maxSeqlenQPtr != nullptr) {
            info.maxSeqlenQ = *maxSeqlenQPtr;
        }
        const auto maxSeqlenKvPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_KV_INDEX);
        if (maxSeqlenKvPtr != nullptr) {
            info.maxSeqlenKv = *maxSeqlenKvPtr;
        }
        const auto layoutQPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_Q_INDEX);
        if (layoutQPtr != nullptr) {
            info.layoutQ = std::string(layoutQPtr);
        }
    }

    info.layout = (info.layoutQ == "TND") ? LAYOUT_TND : LAYOUT_NOT_TND;
    info.isTnd = (info.layout == LAYOUT_TND);
    info.isBnsd = (info.layoutQ == "BNSD");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseShapes(gert::TilingContext *context, FagParsedInfo &info)
{
    const auto qShape = context->GetInputShape(Q_INDEX);
    const auto kShape = context->GetInputShape(K_INDEX);
    const auto vShape = context->GetInputShape(V_INDEX);

    if (info.isTnd) {
        info.s1 = info.maxSeqlenQ > 0 ? info.maxSeqlenQ : qShape->GetStorageShape().GetDim(0);
        info.s2 = info.maxSeqlenKv > 0 ? info.maxSeqlenKv : kShape->GetStorageShape().GetDim(0);
        info.n1 = qShape->GetStorageShape().GetDim(1);
        info.n2 = kShape->GetStorageShape().GetDim(1);
        info.d = qShape->GetStorageShape().GetDim(2);
        info.dv = vShape->GetStorageShape().GetDim(2);
        const auto cuSeqShape = context->GetOptionalInputShape(CU_SEQLENS_Q_INDEX);
        if (cuSeqShape != nullptr && cuSeqShape->GetStorageShape().GetDimNum() > 0) {
            info.b = cuSeqShape->GetStorageShape().GetDim(0) - 1;
        } else {
            info.b = 1;
        }
    } else if (info.isBnsd) {
        // BNSD [B,N,S,D]
        info.b = qShape->GetStorageShape().GetDim(0);
        info.n1 = qShape->GetStorageShape().GetDim(1);
        info.s1 = qShape->GetStorageShape().GetDim(2);
        info.d = qShape->GetStorageShape().GetDim(3);
        info.dv = vShape->GetStorageShape().GetDim(3);
        info.n2 = kShape->GetStorageShape().GetDim(1);
        info.s2 = kShape->GetStorageShape().GetDim(2);
    } else {
        // BSND [B,S,N,D]
        info.b = qShape->GetStorageShape().GetDim(0);
        info.s1 = qShape->GetStorageShape().GetDim(1);
        info.n1 = qShape->GetStorageShape().GetDim(2);
        info.d = qShape->GetStorageShape().GetDim(3);
        info.dv = vShape->GetStorageShape().GetDim(3);
        info.s2 = kShape->GetStorageShape().GetDim(1);
        info.n2 = kShape->GetStorageShape().GetDim(2);
    }

    // GQA：Q/dout/attn_out/dq 带 N1 = N2*G 个 head，K/V/dk/dv 带 N2 个。
    // G 由整除得到，G == 1 即普通 MHA。
    OP_CHECK_IF(info.n2 <= 0 || info.n1 <= 0,
                OP_LOGE(context->GetNodeName(), "head num must be positive, got N1=%ld, N2=%ld.", info.n1, info.n2),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(info.n1 % info.n2 != 0,
                OP_LOGE(context->GetNodeName(),
                        "head num of q (%ld) must be divisible by head num of k/v (%ld) for GQA.", info.n1, info.n2),
                return ge::GRAPH_PARAM_INVALID);
    info.g = info.n1 / info.n2;
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus ParsePlatform(gert::TilingContext *context, FagParsedInfo &info)
{
    const auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        // 图模式下 Tiling 阶段可能拿不到 platform，回退到 TilingParse 缓存的那份。
        // 两路都空才是真失败。
        auto compileInfo = reinterpret_cast<const FlashAttnGradCompileInfo *>(context->GetCompileInfo());
        OP_CHECK_IF(compileInfo == nullptr,
                    OP_LOGE(context->GetNodeName(), "both platform info and compile info are nullptr."),
                    return ge::GRAPH_PARAM_INVALID);
        info.aivNum = compileInfo->aivNum;
        info.aicNum = compileInfo->aicNum;
        info.libapiWorkspaceSize = static_cast<size_t>(compileInfo->libapiWorkspaceSize);
        info.l2CacheSize = compileInfo->l2CacheSize;
        OP_CHECK_IF(info.aicNum == 0 || info.aivNum == 0,
                    OP_LOGE(context->GetNodeName(), "num of core from compile info is 0, aicNum=%u, aivNum=%u.",
                            info.aicNum, info.aivNum),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                    OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                    return ge::GRAPH_PARAM_INVALID);
        return ge::GRAPH_SUCCESS;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    info.aivNum = ascendcPlatform.GetCoreNumAiv();
    info.aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, info.l2CacheSize);
    OP_CHECK_IF(
        info.aicNum == 0 || info.aivNum == 0,
        OP_LOGE(context->GetNodeName(), "num of core obtained is 0, aicNum=%u, aivNum=%u.", info.aicNum, info.aivNum),
        return ge::GRAPH_PARAM_INVALID);
    info.libapiWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    // SOC 是否支持不在这里判：那是"哪个模板类能接"的问题，已下沉到
    // FlashAttnGradTilingRegbase::IsCapable()（D1）。
    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseFlashAttnGradInfo(gert::TilingContext *context, FagParsedInfo &info)
{
    ge::graphStatus ret = ParseAttrs(context, info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ParseShapes(context, info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    info.hasSeqused = OptionalInputExists(context, SEQUSED_Q_INDEX) || OptionalInputExists(context, SEQUSED_KV_INDEX);
    info.hasAttenMask = HasAttenMask(context, info.maskMode);
    info.hasSinks = OptionalInputExists(context, SINKS_INDEX);

    FillLayoutView(info);
    FagFillSparseInfo(info);
    return ge::GRAPH_SUCCESS;
}

bool IsEmptyShape(const FagParsedInfo &info)
{
    return info.b <= 0 || info.s1 <= 0 || info.s2 <= 0 || info.d <= 0 || info.dv <= 0;
}

} // namespace optiling
