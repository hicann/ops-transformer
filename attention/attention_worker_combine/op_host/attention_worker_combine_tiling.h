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
 * \file attention_worker_combine_tiling.h
 * \brief
 */
#ifndef OP_HOST_ATTENTION_WORKER_COMBINE_TILING_H
#define OP_HOST_ATTENTION_WORKER_COMBINE_TILING_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "attention/attention_update/op_host/decode_update_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AttentionWorkerCombineTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);       // 使用的核数
TILING_DATA_FIELD_DEF(int64_t, BS);                // batchSize
TILING_DATA_FIELD_DEF(int64_t, K);                 // TopK 选出的专家数
TILING_DATA_FIELD_DEF(int64_t, H);                 // hiddenSize
TILING_DATA_FIELD_DEF(int64_t, needSchedule);      // 是否需要扫描 schedule_context
TILING_DATA_FIELD_DEF(int64_t, BsSplitFactor);     // bs轴核内切分
TILING_DATA_FIELD_DEF(int64_t, BsSplitCoreNum);    // bs轴核间切分
TILING_DATA_FIELD_DEF(int64_t, mainCoreBsLoopNum); // 主核 bs 循环次数
TILING_DATA_FIELD_DEF(int64_t, tailCoreBsLoopNum); // 尾核 bs 循环次数
TILING_DATA_FIELD_DEF(int64_t, HSplitFactor);      // hidden 轴核内切分
TILING_DATA_FIELD_DEF(int64_t, HSplitTailFactor);  // hidden 轴核内尾块
TILING_DATA_FIELD_DEF(int64_t, HSplitCoreNum);     // hidden 轴核间切分
TILING_DATA_FIELD_DEF(int64_t, mainCoreHLoopNum);  // 主核 H 循环次数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHLoopNum);  // 尾核 H 循环次数
TILING_DATA_FIELD_DEF(int64_t, KSplitFactor);      // k 轴核内切分
TILING_DATA_FIELD_DEF(int64_t, KSplitTailFactor);  // k 轴尾块切分
TILING_DATA_FIELD_DEF(int64_t, KSplitLoopNum);     // k 轴循环次数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AttentionWorkerCombine, AttentionWorkerCombineTilingData)

struct AttentionWorkerCombineCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

class AttentionWorkerCombineTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AttentionWorkerCombineTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~AttentionWorkerCombineTiling() override {}

    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    void DoOpTilingFullK(int64_t batchSize, int64_t bsCoreNum, int64_t hAlign, int64_t kIn);
    void SelectBestHCore(int64_t hAlign, int64_t hInSplitK, int64_t &bsCoreNum, int64_t &lastCoreNum,
                         int64_t &lastBestHCore);

private:
    int64_t tokenDtype_ = 0;
    AttentionWorkerCombineTilingData tilingData_;
};

} // namespace optiling

#endif // OP_HOST_ATTENTION_WORKER_COMBINE_TILING_H
