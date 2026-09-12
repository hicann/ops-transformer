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
 * \file matmul_reduce_scatter_v2_tiling_common.cpp
 * \brief
 */
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "matmul_reduce_scatter_v2_tiling_common.h"
#if MC2_DFX_ENABLE
#include "../../../common/utils/mc2_exception_dump.h"
#include "../../op_kernel/matmul_reduce_scatter_v2_c_tiling.h"
#endif

namespace optiling {
ge::graphStatus MatmulReduceScatterTilingV2Func(gert::TilingContext *context);
struct MatmulReduceScatterV2CompileInfo {};
ge::graphStatus TilingParseForMatmulReduceScatterV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulReduceScatterV2)
    .Tiling(MatmulReduceScatterTilingV2Func)
    .TilingParse<MatmulReduceScatterV2CompileInfo>(TilingParseForMatmulReduceScatterV2);

#if MC2_DFX_ENABLE
// Register exception dump func
// dump 回调按算子仅注册一次，dfxInfoOffset 取自 MatmulReduceScatterV2TilingData；
// QuantBatchMatmulV3ReduceScatterTilingData 需与其保持相同的 mc2InitTiling/mc2CcTiling/dumpInfo
// 前缀（即 dumpInfo 偏移一致），调整任一结构体前缀时必须同步维护该约束
inline void MatmulReduceScatterV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char *socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "MatmulReduceScatterV2");

    Mc2Exception::Mc2DumpTilingAndWorkspace(args, "MatmulReduceScatterV2",
                                            10U, // tilingGmArgIdx
                                            9U,  // workspaceGmArgIdx
                                            static_cast<uint32_t>(offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData,
                                                                           dumpInfo))); // dfxInfoOffset
}

__attribute__((constructor)) void RegisterMatmulReduceScatterV2ExceptionFunc()
{
    IMPL_OP(MatmulReduceScatterV2).ExceptionDumpParseFunc(MatmulReduceScatterV2ExceptionImplWrapper);
}
#endif // MC2_DFX_ENABLE

} // namespace optiling
