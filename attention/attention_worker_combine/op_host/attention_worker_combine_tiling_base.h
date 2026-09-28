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
 * \file attention_worker_combine_tiling_base.h
 * \brief Common AttentionWorkerCombine tiling lifecycle and data contract shared by platform implementations.
 */
#ifndef OP_HOST_ATTENTION_WORKER_COMBINE_TILING_BASE_H
#define OP_HOST_ATTENTION_WORKER_COMBINE_TILING_BASE_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"

namespace optiling {

inline int64_t CeilDiv(int64_t N, int64_t n)
{
    if (unlikely(n == 0)) {
        return N;
    }
    return ((N + n - 1) / n);
}

inline int64_t AlignUp(int64_t N, int64_t n)
{
    if (unlikely(n == 0)) {
        return N;
    }
    return (((N + n - 1) / n) * n);
}

inline int64_t AlignDown(int64_t N, int64_t n)
{
    if (unlikely(n == 0)) {
        return N;
    }
    return N / n * n;
}

BEGIN_TILING_DATA_DEF(AttentionWorkerCombineTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, BS);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, H);
TILING_DATA_FIELD_DEF(int64_t, needSchedule);
TILING_DATA_FIELD_DEF(int64_t, BsSplitFactor);
TILING_DATA_FIELD_DEF(int64_t, BsSplitCoreNum);
TILING_DATA_FIELD_DEF(int64_t, mainCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, HSplitFactor);
TILING_DATA_FIELD_DEF(int64_t, HSplitTailFactor);
TILING_DATA_FIELD_DEF(int64_t, HSplitCoreNum);
TILING_DATA_FIELD_DEF(int64_t, mainCoreHLoopNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreHLoopNum);
TILING_DATA_FIELD_DEF(int64_t, KSplitFactor);
TILING_DATA_FIELD_DEF(int64_t, KSplitTailFactor);
TILING_DATA_FIELD_DEF(int64_t, KSplitLoopNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AttentionWorkerCombine, AttentionWorkerCombineTilingData)

struct AttentionWorkerCombineCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

class AttentionWorkerCombineTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AttentionWorkerCombineTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~AttentionWorkerCombineTilingBase() override = default;

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    virtual ge::graphStatus DoGetPlatformInfo() = 0;
    virtual ge::graphStatus DoGetShapeAttrsInfo() = 0;
    virtual ge::graphStatus CalcOpTiling() = 0;
    virtual ge::graphStatus CalcTilingKey() = 0;
    virtual void DoPostTiling() = 0;

protected:
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t tokenDtype_ = 0;
    uint64_t tilingKey_ = 0;
};

} // namespace optiling

#endif // OP_HOST_ATTENTION_WORKER_COMBINE_TILING_BASE_H
