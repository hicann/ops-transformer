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
 * \file allto_all_matmul_tiling.cpp
 * \brief hosttiling
 */
#include <register/op_def_registry.h>
#include <register/op_impl_registry.h>

#include "allto_all_matmul_tiling_base.h"
#include "op_host/util/op_const_def.h"
#include "mc2_exception_dump.h"
#if MC2_DFX_ENABLE
#include "../../op_kernel/arch35/allto_all_matmul_tiling_data.h"
#endif

using namespace ge;
using Ops::Transformer::OpTiling::TilingRegistryArch;
using Ops::Transformer::OpTiling::TilingRegistryNew;

namespace MC2Tiling {

static ge::graphStatus AlltoAllMatmulTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == Ops::Base::DAV_3510) {
        return TilingRegistryArch::GetInstance().DoTilingImpl(context);
    }
    return TilingRegistryNew::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForAlltoAllMatmul(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllMatmulCompileInfo {};

IMPL_OP_OPTILING(AlltoAllMatmul)
    .Tiling(AlltoAllMatmulTilingFunc)
    .TilingParse<AlltoAllMatmulCompileInfo>(TilingParseForAlltoAllMatmul);

#if MC2_DFX_ENABLE
// Register exception dump func
// dump 回调按算子仅注册一次，dfxInfoOffset 取自 AlltoAllMatmulTilingData；
// AlltoAllQuantMatmulTilingData 与 Apace::hcommAllToAllMatmulTilingData 需与其保持相同的
// mc2InitTiling/mc2CcTiling/dumpInfo 前缀（即 dumpInfo 偏移一致），
// 调整任一结构体前缀时必须同步维护该约束
inline void AlltoAllMatmulExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char *socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "AlltoAllMatmul");
    Mc2Exception::Mc2DumpTilingAndWorkspace(args, "AlltoAllMatmul",
                                            12U, // tilingGmArgIdx
                                            11U, // workspaceGmArgIdx
                                            static_cast<uint32_t>(offsetof(AlltoAllMatmulTilingData,
                                                                           dumpInfo))); // dfxInfoOffset
}

__attribute__((constructor)) void RegisterAlltoAllMatmulExceptionFunc()
{
    IMPL_OP(AlltoAllMatmul).ExceptionDumpParseFunc(AlltoAllMatmulExceptionImplWrapper);
}
#endif // MC2_DFX_ENABLE
} // namespace MC2Tiling
