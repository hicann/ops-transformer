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
 * \file apply_rotary_pos_emb_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_ROTARY_POS_EMB_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_ROTARY_POS_EMB_GRAD_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "../op_kernel/arch35/apply_rotary_pos_emb_grad_tiling_data.h"
#include "platform/platform_info.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "util/math_util.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ApplyRopeGradRegbaseParams)
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, s);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, nQ);
TILING_DATA_FIELD_DEF(int64_t, nK);
TILING_DATA_FIELD_DEF(int64_t, blockNumB);
TILING_DATA_FIELD_DEF(int64_t, blockFactorB);
TILING_DATA_FIELD_DEF(int64_t, blockNumS);
TILING_DATA_FIELD_DEF(int64_t, blockFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumN);
TILING_DATA_FIELD_DEF(int64_t, ubFactorN);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorN);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, rotaryMode);
TILING_DATA_FIELD_DEF(int64_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, dCosFlag);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyRopeGradRegbaseParamsOp, ApplyRopeGradRegbaseParams)

BEGIN_TILING_DATA_DEF(ApplyRotaryPosEmbGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(ApplyRopeGradRegbaseParams, ropeGradParams);
TILING_DATA_FIELD_DEF(uint32_t, dCosFlag);
TILING_DATA_FIELD_DEF(uint32_t, layout);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmbGrad, ApplyRotaryPosEmbGradTilingData)

struct ApplyRotaryPosEmbGradCompileInfo {
    int64_t numBlocks;
    uint64_t ubSize;
    platform_ascendc::SocVersion socVersion;
    Ops::Base::ReduceOpCompileInfo opInfo;
};

enum class ApplyRopeGradLayout : int64_t {
    BSND = 0,
    SBND = 1,
    BNSD = 2,
    NO_BROADCAST = 3
};

enum class ApplyRopeGradRotaryMode : int64_t {
    HALF = 0
};

enum class ApplyRopeGradDxTilingKey : uint32_t {
    TILING_KEY_BAB = 203,
    TILING_KEY_AB = 204,
    TILING_KEY_A = 205
};

class ApplyRotaryPosEmbGradRegbaseTilingClass : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit ApplyRotaryPosEmbGradRegbaseTilingClass(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        reduceInputFloat_ = false;
    }

private:
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtypeAndAttr();
    ge::graphStatus CheckParam();
    ge::graphStatus CheckShapeLimit();
    ge::graphStatus CheckOptionalInput() const;
    ge::graphStatus CheckShapeDim() const;
    ge::graphStatus ValidateBroadcastByLayout(int64_t attrLayout, const gert::Shape &gqShape,
                                              const gert::Shape &gkShape, const gert::Shape &cosShape);
    ge::graphStatus CheckRotaryModeShapeRelation(const int64_t d);
    ge::graphStatus CheckInPutShapeAllPositive(const int64_t idx) const;
    ge::graphStatus CheckOutPutShapeAllPositive(const int64_t idx) const;
    ge::graphStatus CheckShapeAllPositive() const;

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetReduceOpCompileInfo(Ops::Base::ReduceOpCompileInfo *compileInfo);
    ge::graphStatus GetInputParam(Ops::Base::ReduceOpInputParam &opInput, uint32_t inputIdx, uint32_t axesIdx);
    ge::graphStatus InitTilingData();
    ge::graphStatus TilingReduce();
    ge::graphStatus SetTilingKeyBlockDim(uint32_t dxTilingKey);

    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    const static int64_t MAX_COPY_BLOCK_COUNT = 4095;
    const static int64_t DIM_NUM = 4;
    const static int64_t DIM_NUM_TND = 3;

    int64_t b_{0};
    int64_t s_{0};
    int64_t nQ_{0};
    int64_t nK_{0};
    int64_t d_{0};
    int64_t cosb_{0};
    int64_t usedCoreNum_{0};
    ge::DataType dtype_;
    ApplyRopeGradLayout layout_;
    ApplyRopeGradRotaryMode rotaryMode_;
    ApplyRopeGradTilingData *tilingData_{nullptr};

    gert::Shape gqShape_; // grad_query_embed shape
    gert::Shape cosShape_;
    Ops::Base::ReduceTilingKey key_;
    uint64_t tilingKey_{0};
    int64_t blockSize_;
    int64_t vLength_;
    int64_t dSplitCoef_;
    uint32_t dCosFlag_{0};
    bool reduceInputFloat_{false};
    bool isTndLayout_ = false;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_ROTARY_POS_EMB_GRAD_H
