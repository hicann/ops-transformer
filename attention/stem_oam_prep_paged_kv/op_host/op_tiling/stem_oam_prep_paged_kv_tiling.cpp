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
 * \file stem_oam_prep_paged_kv_tiling.cpp
 * \brief stem oam prep paged kv tiling file
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "arch35/stem_oam_prep_paged_kv_tiling_simd.h"
#include "stem_oam_prep_paged_kv_tiling.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"

namespace optiling {

static ge::graphStatus Tiling4StemOamPrepPagedKv(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "StemOamPrepPagedKv is running.");

    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGD(context->GetNodeName(), "Using arch35 tiling for RegbaseSocVersion");
        return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }

    OP_LOGD(context->GetNodeName(), "StemOamPrepPagedKv Only Support RegbaseSocVersion.");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingPrepare4StemOamPrepPagedKv(gert::TilingParseContext *context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(StemOamPrepPagedKv)
    .Tiling(Tiling4StemOamPrepPagedKv)
    .TilingParse<StemOamPrepPagedKvCompileInfo>(TilingPrepare4StemOamPrepPagedKv)
    .TilingInputsDataDependency({3});
} // namespace optiling
