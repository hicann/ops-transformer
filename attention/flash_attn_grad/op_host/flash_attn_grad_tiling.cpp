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
 * \file flash_attn_grad_tiling.cpp
 * \brief L0 入口：解析 -> 校验 -> 规划 -> 写回。业务判断都在下层，这里不做决策。
 */

#include "flash_attn_grad_tiling.h"

#include "base/flash_attn_grad_tiling_regbase.h"
#include "checkers/flash_attn_grad_tiling_check.h"
#include "info/flash_attn_grad_tiling_info.h"
#include "info/flash_attn_grad_tiling_info_parser.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus FlashAttnGradTilingFunc(gert::TilingContext *context)
{
    FagParsedInfo info;
    ge::graphStatus ret = ParsePlatform(context, info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = FlashAttnGradCheck::CheckParams(context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "CheckParams failed."),
                return ge::GRAPH_PARAM_INVALID);

    ret = ParseFlashAttnGradInfo(context, info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 只有一个模板类，所以直接构造。将来 TND 拆出独立实现时，这里换成按注册
    // 优先级依次 DoTiling、遇到 GRAPH_PARAM_INVALID 就试下一个。
    FlashAttnGradTilingRegbase tiling(context);
    return tiling.DoTiling(info);
}

// 把平台信息在编译期取一次缓存进 CompileInfo，供图模式下 Tiling 阶段拿不到
// PlatformInfo 时回退使用。取不到 platform 不在这里报错：AOE / 单算子等场景本就
// 可能没有，真正需要时由 ParsePlatform 判断两路都空才失败。
ge::graphStatus TilingParseForFlashAttnGrad(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("FlashAttnGrad", "TilingParseContext is nullptr."),
                return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompiledInfo<FlashAttnGradCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context->GetNodeName(), "compile info is nullptr."),
                return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGW(context->GetNodeName(), "platform info is nullptr at parse stage; tiling will query it directly.");
        return ge::GRAPH_SUCCESS;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo->libapiWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo->l2CacheSize);

    OP_LOGI(context->GetNodeName(), "parsed platform: aicNum=%u, aivNum=%u, l2CacheSize=%lu, libapiWorkspace=%lu.",
            compileInfo->aicNum, compileInfo->aivNum, compileInfo->l2CacheSize, compileInfo->libapiWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlashAttnGrad)
    .Tiling(FlashAttnGradTilingFunc)
    .TilingParse<FlashAttnGradCompileInfo>(TilingParseForFlashAttnGrad);

} // namespace optiling
