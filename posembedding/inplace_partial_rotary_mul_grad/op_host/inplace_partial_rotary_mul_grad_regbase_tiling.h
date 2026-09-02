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
 * \file inplace_partial_rotary_mul_grad_regbase_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "platform/platform_info.h"
#include "util/math_util.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(InplacePartialRotaryMulGradRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, s);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, blockNumB);
TILING_DATA_FIELD_DEF(int64_t, blockFactorB);
TILING_DATA_FIELD_DEF(int64_t, blockNumS);
TILING_DATA_FIELD_DEF(int64_t, blockFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubFactorB);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumN);
TILING_DATA_FIELD_DEF(int64_t, ubFactorN);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorN);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, rotaryMode);
TILING_DATA_FIELD_DEF(int64_t, sliceStart);
TILING_DATA_FIELD_DEF(int64_t, sliceEnd);
TILING_DATA_FIELD_DEF(int64_t, sliceLength);
TILING_DATA_FIELD_DEF(int64_t, dSplitCoef);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegbaseTilingData)

BEGIN_TILING_DATA_DEF(InplacePartialRotaryMulGradRegbaseTilingDataAb)
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, s);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, dAlign);
TILING_DATA_FIELD_DEF(int64_t, dSplitCoef);
TILING_DATA_FIELD_DEF(int64_t, blockNumBS);
TILING_DATA_FIELD_DEF(int64_t, blockFactorBS);
TILING_DATA_FIELD_DEF(int64_t, blockTailBS);
TILING_DATA_FIELD_DEF(int64_t, blockNumN);
TILING_DATA_FIELD_DEF(int64_t, blockFactorN);
TILING_DATA_FIELD_DEF(int64_t, blockTailN);
TILING_DATA_FIELD_DEF(int64_t, ubFactorBS);
TILING_DATA_FIELD_DEF(int64_t, ubFactorN);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, rotaryMode);
TILING_DATA_FIELD_DEF(int64_t, sliceStart);
TILING_DATA_FIELD_DEF(int64_t, sliceEnd);
TILING_DATA_FIELD_DEF(int64_t, sliceLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InplacePartialRotaryMulGrad_204, InplacePartialRotaryMulGradRegbaseTilingDataAb)

constexpr uint64_t TILING_KEY_EMPTY = 403;
constexpr static int64_t WORKSPACE_COUNT = 1;
constexpr static size_t RESERVED_WORKSPACE = static_cast<size_t>(16 * 1024 * 1024);
struct InplacePartialRotaryMulGradCompileInfo {};

enum class InplacePartialRotaryMulGradLayout : int64_t {
    NO_BROADCAST = 1,
    BROADCAST_BSN = 2,
    BSND = 3,
    SBND = 4,
    BNSD = 5
};

enum class InplacePartialRotaryMulGradMode : int64_t {
    HALF = 0,
    INTERLEAVE = 1,
    QUARTER = 2,
    INTERLEAVE_HALF = 3
};

class InplacePartialRotaryMulGradRegbaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit InplacePartialRotaryMulGradRegbaseTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

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
    const static int64_t MAX_COPY_BLOCK_COUNT = 4095;
    int64_t b_{0};
    int64_t s_{0};
    int64_t n_{0};
    int64_t d_{0};
    int64_t cosb_{0};
    int64_t usedCoreNum_{0};
    ge::DataType dtype_;
    ge::DataType cosDtype_;
    InplacePartialRotaryMulGradLayout layout_;
    InplacePartialRotaryMulGradMode rotaryMode_;

    gert::Shape dyShape_;
    gert::Shape cosShape_;
    uint64_t tilingKey_{0};
    int64_t blockSize_;
    int64_t dSplitCoef_;
    int64_t sliceStart_{0};
    int64_t sliceEnd_{0};
    int64_t sliceLength_{0};
    bool is1snd_ = false;
    // No-op: empty slice (start == end) or empty dy/cos/sin -> nothing to compute,
    // kernel goes through TILING_KEY_EMPTY and returns, output dx == dy (in-place).
    bool isNoOp_{false};

private:
    bool IsInplacePartialRotaryMulGradMode(const int32_t mode) const;
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckDtypeDyGroup();
    ge::graphStatus CheckDtypeCosGroup();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckParam();
    ge::graphStatus CheckShapeLimit();
    ge::graphStatus CheckShapeDim() const;
    ge::graphStatus JudgeLayoutByShape(const gert::Shape &xShape, const gert::Shape &cosShape);
    ge::graphStatus CheckRotaryModeShapeRelation(const int64_t sliceLen);
    ge::graphStatus CheckInPutShapeAllPositive(const int64_t idx) const;
    ge::graphStatus CheckOutPutShapeAllPositive(const int64_t idx) const;

    ge::graphStatus CheckShapeAllPositive() const;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H
