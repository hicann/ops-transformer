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
 * \file moe_token_unpermute_with_routing_map.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
namespace optiling {

struct TilingPadParams {
    uint64_t core_num_use = 0;
    uint64_t num_tokens = 0;
    uint64_t num_experts = 0;
    uint64_t capacity = 0;
    uint64_t front_core = 0;
    uint64_t loop_time_each_front_core = 0;
    uint64_t num_tokens_each_front_core = 0;
    uint64_t num_tokens_front_core_each_loop = 0;
    uint64_t num_tokens_front_core_last_loop = 0;
    uint64_t tail_core = 0;
    uint64_t loop_time_each_tail_core = 0;
    uint64_t num_tokens_each_tail_core = 0;
    uint64_t num_tokens_tail_core_each_loop = 0;
    uint64_t num_tokens_tail_core_last_loop = 0;
};

BEGIN_TILING_DATA_DEF(MaskedSelectMTUTilingData)
TILING_DATA_FIELD_DEF(uint64_t, formerNum);
TILING_DATA_FIELD_DEF(uint64_t, formerLength);
TILING_DATA_FIELD_DEF(uint64_t, formertileNum);
TILING_DATA_FIELD_DEF(uint64_t, formertileLength);
TILING_DATA_FIELD_DEF(uint64_t, formerlasttileLength);
TILING_DATA_FIELD_DEF(uint64_t, tailNum);
TILING_DATA_FIELD_DEF(uint64_t, tailLength);
TILING_DATA_FIELD_DEF(uint64_t, tailtileNum);
TILING_DATA_FIELD_DEF(uint64_t, tailtileLength);
TILING_DATA_FIELD_DEF(uint64_t, taillasttileLength);
TILING_DATA_FIELD_DEF(uint64_t, tokenNum);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MaskedSelectMTUTilingDataOp, MaskedSelectMTUTilingData)

BEGIN_TILING_DATA_DEF(MoeTokenUnpermuteWithRoutingMapPadTilingData)
TILING_DATA_FIELD_DEF(uint64_t, core_num_use);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens);
TILING_DATA_FIELD_DEF(uint64_t, num_experts);
TILING_DATA_FIELD_DEF(uint64_t, capacity);
TILING_DATA_FIELD_DEF(uint64_t, front_core);
TILING_DATA_FIELD_DEF(uint64_t, loop_time_each_front_core);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_each_front_core);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_front_core_each_loop);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_front_core_last_loop);
TILING_DATA_FIELD_DEF(uint64_t, tail_core);
TILING_DATA_FIELD_DEF(uint64_t, loop_time_each_tail_core);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_each_tail_core);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_tail_core_each_loop);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_tail_core_last_loop);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeTokenUnpermuteWithRoutingMapPadTilingDataOp, MoeTokenUnpermuteWithRoutingMapPadTilingData)

BEGIN_TILING_DATA_DEF(MoeTokenUnpermuteWithRoutingMapTilingData)
TILING_DATA_FIELD_DEF(int64_t, hidden_size);
TILING_DATA_FIELD_DEF(int64_t, top_k);
TILING_DATA_FIELD_DEF(int64_t, num_out_tokens);

TILING_DATA_FIELD_DEF(int64_t, hidden_splited_length);
TILING_DATA_FIELD_DEF(int64_t, hidden_splited_num);
TILING_DATA_FIELD_DEF(int64_t, hidden_splited_remain);

TILING_DATA_FIELD_DEF(int64_t, tokens_core_length);
TILING_DATA_FIELD_DEF(int64_t, tokens_core_remain);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_length);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_num);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_remain);
TILING_DATA_FIELD_DEF(int64_t, used_core_num);

TILING_DATA_FIELD_DEF(int64_t, buffer_num);
TILING_DATA_FIELD_DEF_STRUCT(MaskedSelectMTUTilingData, maskedSelectParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(MoeTokenUnpermuteWithRoutingMapPadTilingData,
                             moeTokenUnpermuteWithRoutingMapPadTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeTokenUnpermuteWithRoutingMap, MoeTokenUnpermuteWithRoutingMapTilingData)

struct MoeTokenUnpermuteWithRoutingMapCompileInfo {
    uint64_t aivnum = 0;
    uint64_t ubSize = 0;
    uint64_t workSpaceSIZE = 0;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_H
