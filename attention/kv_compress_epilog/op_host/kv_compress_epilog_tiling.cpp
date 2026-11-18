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
 * \file kv_compress_epilog_tiling.cpp
 * \brief KvCompressEpilog tiling 注册与按 arch 分派入口（架构无关 glue）。
 *        arch22 (DAV_2201: A2 ascend910b / A3 ascend910_93) → quant_mode=1 专用 tiling；
 *        其余平台 (arch35: ascend950/ascend960dt/ascend350) → mode 0/1/2 既有 tiling。
 *        两个架构的实现位于 arch22/ 与 arch35/ 子目录，互不包含；arch35 分派仅在
 *        KV_CE_ENABLE_ARCH35_TILING 定义（CMake 检测到 arch35 源码存在）时编译，
 *        删除 arch35 目录后 arch22 (quant_mode=1) 仍可独立编译运行。
 */

#include "kv_compress_epilog_tiling.h"
#include "arch22/kv_compress_epilog_tiling_arch22.h"
#ifdef KV_CE_ENABLE_ARCH35_TILING
#include "arch35/kv_compress_epilog_tiling_arch35.h"
#endif

using namespace ge;

namespace optiling {

ge::graphStatus TilingForKvCompressEpilog(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KvCompressEpilog", "Tiling context is null"),
                return ge::GRAPH_FAILED);

    // 按 arch 分派: arch22 (DAV_2201, A2 ascend910b / A3 ascend910_93) 走 quant_mode=1 专用 tiling;
    // 其余平台 (arch35: ascend950/ascend960dt/ascend350) 走既有 arch35 tiling (mode 0/1/2)
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_2201) {
            OP_LOGD(context->GetNodeName(), "Using arch22 tiling for KvCompressEpilog");
            KvCompressEpilogTilingArch22 tiling(context);
            return tiling.RunTiling();
        }
    }

#ifdef KV_CE_ENABLE_ARCH35_TILING
    KvCompressEpilogTilingArch35 tiling(context);
    return tiling.RunTiling();
#else
    OP_LOGE(context->GetNodeName(),
            "KvCompressEpilog tiling for this platform is unavailable: arch35 tiling sources are not compiled");
    return ge::GRAPH_FAILED;
#endif
}

ge::graphStatus TilingPrepareForKvCompressEpilog(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCompressEpilog)
    .Tiling(TilingForKvCompressEpilog)
    .TilingParse<KvCompressEpilogCompileInfo>(TilingPrepareForKvCompressEpilog);

} // namespace optiling
