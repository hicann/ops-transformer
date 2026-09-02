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
 * \file ffn_worker_batching_tiling.h
 * \brief
 */
#ifndef OP_HOST_FFN_WORKER_BATCHING_TILING_H
#define OP_HOST_FFN_WORKER_BATCHING_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "platform/platform_info.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FfnWorkerBatchingTilingData)
TILING_DATA_FIELD_DEF(int64_t, Y);
TILING_DATA_FIELD_DEF(int64_t, H);
TILING_DATA_FIELD_DEF(int64_t, tokenDtype);
TILING_DATA_FIELD_DEF(int64_t, expertNum);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, sortLoopMaxElement);
TILING_DATA_FIELD_DEF(int64_t, sortNumWorkSpace);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FfnWorkerBatching, FfnWorkerBatchingTilingData)

struct FfnWorkerBatchingCompileInfo {};

class FfnWorkerBatchingTiling {
public:
    explicit FfnWorkerBatchingTiling(gert::TilingContext *context)
        : context_(context) {};
    ge::graphStatus RunFfnWorkerBatchingTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus CheckInputParam();
    ge::graphStatus SetBatchingWorkspaceSize(int64_t expertNum);
    ge::graphStatus GetAttrsInfo();
    void SetBatchingTilingKey(int64_t sortLoopMaxElement);

private:
    FfnWorkerBatchingTilingData tilingData_;
    gert::TilingContext *context_ = nullptr;

    int64_t A_ = 0;
    int64_t BS_ = 0;
    int64_t K_ = 0;
    int64_t Y_ = 0;
    int64_t needSchedule_ = 0;
    int64_t layerNum_ = 0;
    uint32_t ubBlockSize_ = 0;
};

} // namespace optiling
#endif // OP_HOST_FFN_WORKER_BATCHING_TILING_H
