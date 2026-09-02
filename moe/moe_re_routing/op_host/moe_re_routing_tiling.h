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
 * \file moe_re_routing_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_H_
#include "op_host/tiling_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_type.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MoeReRoutingTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, tokensNum);
TILING_DATA_FIELD_DEF(int64_t, tokensSize);
TILING_DATA_FIELD_DEF(int64_t, rankNum);
TILING_DATA_FIELD_DEF(int64_t, expertNumPerRank);
TILING_DATA_FIELD_DEF(int64_t, hasScale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeReRouting, MoeReRoutingTilingData);

BEGIN_TILING_DATA_DEF(MoeReRoutingReTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokenSize);       // token大小
TILING_DATA_FIELD_DEF(int64_t, scaleSize);       // scale大小
TILING_DATA_FIELD_DEF(int64_t, rankNum);         // rank总数
TILING_DATA_FIELD_DEF(int64_t, expertNum);       // expert总数
TILING_DATA_FIELD_DEF(int64_t, coreNum);         // 使用了多少个核
TILING_DATA_FIELD_DEF(int64_t, blockNumR);       // rank分核个数
TILING_DATA_FIELD_DEF(int64_t, blockFactorR);    // 每个核处理的rank数
TILING_DATA_FIELD_DEF(int64_t, blockNumE);       // expert分核个数
TILING_DATA_FIELD_DEF(int64_t, blockFactorE);    // 每个核处理的expert数
TILING_DATA_FIELD_DEF(int64_t, ubFactor);        // 可用Ub大小
TILING_DATA_FIELD_DEF(int64_t, idxType);         // 属性idxType
TILING_DATA_FIELD_DEF(int64_t, tokenSizeOrigin); // FP4原始token大小
TILING_DATA_FIELD_DEF(int64_t, hasTopkWeight);   // 0=无topkWeight,1=有topkWeight
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeReRouting_200000, MoeReRoutingReTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRouting_200100, MoeReRoutingReTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRouting_200200, MoeReRoutingReTilingData);

BEGIN_TILING_DATA_DEF(MoeReRoutingRTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokenSize);       // token大小
TILING_DATA_FIELD_DEF(int64_t, scaleSize);       // scale大小
TILING_DATA_FIELD_DEF(int64_t, rankNum);         // rank总数
TILING_DATA_FIELD_DEF(int64_t, expertNum);       // expert总数
TILING_DATA_FIELD_DEF(int64_t, coreNum);         // 使用了多少个核
TILING_DATA_FIELD_DEF(int64_t, blockFactor);     // 每个核处理的rank数
TILING_DATA_FIELD_DEF(int64_t, ubFactor);        // 可用Ub大小
TILING_DATA_FIELD_DEF(int64_t, idxType);         // 属性idxType
TILING_DATA_FIELD_DEF(int64_t, tokenSizeOrigin); // FP4原始token大小
TILING_DATA_FIELD_DEF(int64_t, hasTopkWeight);   // 0=无topkWeight,1=有topkWeight
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeReRouting_210000, MoeReRoutingRTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRouting_210100, MoeReRoutingRTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRouting_210200, MoeReRoutingRTilingData);

struct ParamsMoeReRouting {
    ge::DataType tokenDtype = ge::DT_UNDEFINED;
    ge::DataType expertDtype = ge::DT_UNDEFINED;
    ge::DataType scaleDtype = ge::DT_UNDEFINED;
    ge::Shape tokenShape;
    ge::Format format = ge::FORMAT_RESERVED;
};

struct MoeReRoutingCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    platform_ascendc::SocVersion socVersion;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_H_
