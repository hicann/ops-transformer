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
 * \file rotary_position_embedding_grad.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_GRAD_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "../op_kernel/arch35/rotary_position_embedding_grad_tiling_data.h"
#include "platform/platform_info.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "util/math_util.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RopeHalfGradParams)
TILING_DATA_FIELD_DEF(uint64_t, layout);
TILING_DATA_FIELD_DEF(uint64_t, xShapeSize);
TILING_DATA_FIELD_DEF(uint64_t, cosShapeSize);
TILING_DATA_FIELD_DEF(uint64_t, dimB);
TILING_DATA_FIELD_DEF(uint64_t, dimS);
TILING_DATA_FIELD_DEF(uint64_t, dimN);
TILING_DATA_FIELD_DEF(uint64_t, dimD);
TILING_DATA_FIELD_DEF(uint64_t, cosDimB);
TILING_DATA_FIELD_DEF(uint64_t, cosDimN);
TILING_DATA_FIELD_DEF(uint64_t, halfDimDAlignNum);

TILING_DATA_FIELD_DEF(uint64_t, coreData);
TILING_DATA_FIELD_DEF(uint64_t, coreLast);
TILING_DATA_FIELD_DEF(uint64_t, copyLoop);
TILING_DATA_FIELD_DEF(uint64_t, copyTail);
TILING_DATA_FIELD_DEF(uint64_t, lastCopyLoop);
TILING_DATA_FIELD_DEF(uint64_t, lastCopyTail);
TILING_DATA_FIELD_DEF(uint64_t, alignUbSize);
TILING_DATA_FIELD_DEF(uint64_t, calcUbSize);
TILING_DATA_FIELD_DEF(uint64_t, coreUsed);
TILING_DATA_FIELD_DEF(uint64_t, coreNum);

TILING_DATA_FIELD_DEF(uint64_t, firstReduce);
TILING_DATA_FIELD_DEF(uint64_t, secondReduce);
TILING_DATA_FIELD_DEF(uint64_t, ubLoopGap);
TILING_DATA_FIELD_DEF(uint64_t, blockLenInner);
TILING_DATA_FIELD_DEF(uint64_t, strideInner);
TILING_DATA_FIELD_DEF(uint64_t, blockLenPadInner);
TILING_DATA_FIELD_DEF(uint64_t, stridePadInner);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RopeHalfGradParamsOp, RopeHalfGradParams)

BEGIN_TILING_DATA_DEF(RopeInterleavedGradParams)
TILING_DATA_FIELD_DEF(uint64_t, batchSize);
TILING_DATA_FIELD_DEF(uint64_t, seqLen);
TILING_DATA_FIELD_DEF(uint64_t, numHeads);
TILING_DATA_FIELD_DEF(uint64_t, headDim);
TILING_DATA_FIELD_DEF(uint64_t, alignHeadDim);
TILING_DATA_FIELD_DEF(uint64_t, padHeadDim);

TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, seqFrontLen);
TILING_DATA_FIELD_DEF(uint64_t, seqTailLen);

TILING_DATA_FIELD_DEF(uint64_t, seqFrontCalcNum);
TILING_DATA_FIELD_DEF(uint64_t, seqFrontCalcLoop);
TILING_DATA_FIELD_DEF(uint64_t, seqFrontCalcTail);

TILING_DATA_FIELD_DEF(uint64_t, seqTailCalcNum);
TILING_DATA_FIELD_DEF(uint64_t, seqTailCalcLoop);
TILING_DATA_FIELD_DEF(uint64_t, seqTailCalcTail);

TILING_DATA_FIELD_DEF(uint64_t, numHeadsLength);
TILING_DATA_FIELD_DEF(uint64_t, numHeadsLoop);
TILING_DATA_FIELD_DEF(uint64_t, numHeadsTail);

TILING_DATA_FIELD_DEF(uint64_t, batchNumHeadsLength);
TILING_DATA_FIELD_DEF(uint64_t, batchNumHeadsLoop);
TILING_DATA_FIELD_DEF(uint64_t, batchNumHeadsTail);

TILING_DATA_FIELD_DEF(uint64_t, layout);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RopeInterleavedGradParamsOp, RopeInterleavedGradParams)

BEGIN_TILING_DATA_DEF(RotaryPositionEmbeddingGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(RopeHalfGradParams, ropeHalfGradParams);
TILING_DATA_FIELD_DEF_STRUCT(RopeInterleavedGradParams, ropeInterleavedGradParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RotaryPositionEmbeddingGrad, RotaryPositionEmbeddingGradTilingData)

struct RotaryPositionEmbeddingGradCompileInfo {
    int64_t numBlocks;
    uint64_t ubSize;
    platform_ascendc::SocVersion socVersion;
    Ops::Base::ReduceOpCompileInfo opInfo;
};

enum class RopeLayout : int64_t {
    NO_BROADCAST = 1,
    BROADCAST_BSN = 2,
    BSND = 3,
    SBND = 4,
    BNSD = 5
};

enum class RotaryPosEmbeddingMode : int64_t {
    HALF = 0,
    INTERLEAVE = 1,
    QUARTER = 2,
    DEEPSEEK_INTERLEAVE = 3
};

enum class ropeGradTilingKey : uint32_t {
    TILING_KEY_ABA = 201,
    TILING_KEY_BA = 202,
    TILING_KEY_BAB = 203,
    TILING_KEY_AB = 204,
    TILING_KEY_A = 205,
    TILING_KEY_B = 206
};

class RotaryPosEmbeddingGradMembaseTilingClass : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RotaryPosEmbeddingGradMembaseTilingClass(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext *context) override
    {
        RotaryPosEmbeddingGradMembaseTilingClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override;

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable() override
    {
        return true;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override;

    uint64_t GetTilingKey() const override
    {
        return context_->GetTilingKey();
    }

protected:
    static const uint32_t MODE_ROTATE_HALF = 0;
    static const uint32_t MODE_ROTATE_INTERLEAVED = 1;
    uint32_t inputMode_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

class RopeGradRegBaseTilingClass : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RopeGradRegBaseTilingClass(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

private:
    bool IsRotaryPosEmbeddingMode(const int32_t mode) const;
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtypeAndAttr();
    ge::graphStatus CheckParam();
    ge::graphStatus CheckShapeLimit();
    ge::graphStatus CheckOptionalInput() const;
    ge::graphStatus CheckShapeDim() const;
    ge::graphStatus JudgeLayoutByShape(const gert::Shape &xShape, const gert::Shape &cosShape);
    bool Is3dBsdBroadcastLayout(const gert::Shape &xShape, const gert::Shape &cosShape) const;
    void Set3dBsdShapeAttrs(const gert::Shape &xShape);
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
    ge::graphStatus SetRotaryXTilingData();
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    const static int64_t MAX_COPY_BLOCK_COUNT = 4095;
    int64_t b_{0};
    int64_t s_{0};
    int64_t n_{0};
    int64_t d_{0};
    int64_t cosb_{0};
    int64_t usedCoreNum_{0};
    ge::DataType dtype_;
    RopeLayout layout_;
    RotaryPosEmbeddingMode rotaryMode_;
    RopeGradTilingData *tilingData_{nullptr};

    gert::Shape dyShape_;
    gert::Shape cosShape_;
    Ops::Base::ReduceTilingKey key_;
    uint64_t tilingKey_{0};
    int64_t blockSize_;
    int64_t vLength_;
    int64_t dSplitCoef_;
    uint32_t dCosFlag_{0};
    bool is1snd_ = false;
    bool isTndLayout_ = false;
    // 3D BSD: dy=(B,S,D), cos/sin=(1,S,D), mapped to BSND with N=1.
    bool is3dBsdLayout_ = false;
    bool isEmptyDy_ = false; // dy/x 为空 tensor（空进空出），kernel 仅对 dcos/dsin 清零
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_GRAD_H
