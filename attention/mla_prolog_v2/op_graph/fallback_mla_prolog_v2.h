/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FALLBACK_MLA_PROLOG_V2_H
#define FALLBACK_MLA_PROLOG_V2_H

#include <vector>
#include "log/log.h"
#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "../../mla_prolog/op_graph/fallback_mla_prolog.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

constexpr size_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;

struct MlaPrologV2FallBackParam : MlaPrologFallBackParam {
    const gert::Tensor* dequantScaleQNope = nullptr;
};

graphStatus GetMlaPrologV2OutputTensor(const OpExecuteContext* ctx, MlaPrologV2FallBackParam& param);
graphStatus MlaV2HostExecuteFunc(OpExecuteContext* host_api_ctx);

} // namespace fallback

#ifdef __cplusplus
}
#endif

#endif // FALLBACK_MLA_PROLOG_V2_H
