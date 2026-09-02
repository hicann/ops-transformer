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
 * \file interleave_rope_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_INTERLEAVE_ROPE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_INTERLEAVE_ROPE_H_

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
BEGIN_TILING_DATA_DEF(InterleaveRopeTilingData)
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(int64_t, splitAxis);

TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, numHead);
TILING_DATA_FIELD_DEF(int64_t, seqLength);
TILING_DATA_FIELD_DEF(int64_t, hiddenDim);

TILING_DATA_FIELD_DEF(int64_t, batchsPerBlock);
TILING_DATA_FIELD_DEF(int64_t, batchsLastBlock);
TILING_DATA_FIELD_DEF(int64_t, batchLoops);
TILING_DATA_FIELD_DEF(int64_t, batchPerLoop);
TILING_DATA_FIELD_DEF(int64_t, batchLastLoop);

TILING_DATA_FIELD_DEF(int64_t, hiddenDimCountPerBlock);
TILING_DATA_FIELD_DEF(int64_t, hiddenDimCountLastBlock);
TILING_DATA_FIELD_DEF(int64_t, hiddenDimLoopsPerBlock);
TILING_DATA_FIELD_DEF(int64_t, hiddenDimCountPerLoopPerBlock);
TILING_DATA_FIELD_DEF(int64_t, hiddenDimCountLastLoopPerBlock);
TILING_DATA_FIELD_DEF(int64_t, hiddenDimLoopsLastBlock);
TILING_DATA_FIELD_DEF(int64_t, hiddenDimCountPerLoopLastBlock);
TILING_DATA_FIELD_DEF(int64_t, hiddenDimCountLastLoopLastBlock);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InterleaveRope, InterleaveRopeTilingData)

struct InterleaveRopeCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

class InterleaveRopeTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit InterleaveRopeTiling(gert::TilingContext *p_context_)
        : TilingBaseClass(p_context_)
    {}
    ~InterleaveRopeTiling() override = default;

    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t batchSize_ = 0;
    int64_t numHead_ = 0;
    int64_t seqLength_ = 0;
    int64_t hiddenDim_ = 0;
    int64_t cosB_ = 0;
    int64_t cosN_ = 0;
    int64_t cosS_ = 0;
    int64_t cosD_ = 0;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus SplitBlockForFixBNS();
    ge::graphStatus SplitBlockForBatch();
    ge::graphStatus SplitBlockForNS();
    ge::graphStatus SplitHiddenDim();
    ge::graphStatus SplitHiddenDimInblock(int64_t hiddenDimCount, int64_t &hiddenDimLoops,
                                          int64_t &hiddenDimCountPerLoop, int64_t &hiddenDimCountLastLoop);

private:
    InterleaveRopeTilingData tilingData_;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_INTERLEAVE_ROPE_H_
