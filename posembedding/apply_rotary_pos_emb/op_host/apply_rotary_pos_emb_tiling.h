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
 * \file apply_rotary_pos_emb_tiling.h
 * \brief
 */
#ifndef AIR_RUNTIME_V2_OP_IMPL_APPLY_ROTARY_POS_EMB_TILING_H
#define AIR_RUNTIME_V2_OP_IMPL_APPLY_ROTARY_POS_EMB_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "platform/platform_infos_def.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ApplyRotaryPosEmbTilingData)
TILING_DATA_FIELD_DEF(int64_t, useCoreNum);
TILING_DATA_FIELD_DEF(int64_t, lastDim);
TILING_DATA_FIELD_DEF(int64_t, halfNum);
TILING_DATA_FIELD_DEF(int64_t, preCBatchB);
TILING_DATA_FIELD_DEF(int64_t, preCBatchL);
TILING_DATA_FIELD_DEF(int64_t, lastCBatchL);
TILING_DATA_FIELD_DEF(int64_t, comBatchBB);
TILING_DATA_FIELD_DEF(int64_t, comBatchBBL);
TILING_DATA_FIELD_DEF(int64_t, comBatchBLL);
TILING_DATA_FIELD_DEF(int64_t, comBatchLLL);
TILING_DATA_FIELD_DEF(int64_t, qPart1Ub);
TILING_DATA_FIELD_DEF(int64_t, q2q1Part1Ub);
TILING_DATA_FIELD_DEF(int64_t, cosPart1Ub);
TILING_DATA_FIELD_DEF(int64_t, sin1UbSize);
TILING_DATA_FIELD_DEF(int64_t, preCLTimes);
TILING_DATA_FIELD_DEF(int64_t, lastCLTimes);
TILING_DATA_FIELD_DEF(int64_t, preCBBTimes);
TILING_DATA_FIELD_DEF(int64_t, preCBLTimes);
TILING_DATA_FIELD_DEF(int64_t, preCLLTimes);
TILING_DATA_FIELD_DEF(int64_t, qCoreOffset);
TILING_DATA_FIELD_DEF(int64_t, kCoreOffset);
TILING_DATA_FIELD_DEF(int64_t, cosCoreOffset);
TILING_DATA_FIELD_DEF(int64_t, qcdNum);
TILING_DATA_FIELD_DEF(int64_t, kcdNum);
TILING_DATA_FIELD_DEF(int64_t, coscdNum);
TILING_DATA_FIELD_DEF(int64_t, qkcNum);
TILING_DATA_FIELD_DEF(int64_t, mulNum);
TILING_DATA_FIELD_DEF(int64_t, qcdHalfNum);
TILING_DATA_FIELD_DEF(int64_t, dstRepSBr);
TILING_DATA_FIELD_DEF(int64_t, blockLenQ);
TILING_DATA_FIELD_DEF(int64_t, srcStrideK);
TILING_DATA_FIELD_DEF(int64_t, blockLenq2q1);
TILING_DATA_FIELD_DEF(int64_t, mask);
TILING_DATA_FIELD_DEF(int64_t, qcNum);
TILING_DATA_FIELD_DEF(int64_t, kcNum);
TILING_DATA_FIELD_DEF(int64_t, qDim3);
TILING_DATA_FIELD_DEF(int64_t, kDim3);
TILING_DATA_FIELD_DEF(int64_t, blockMoveQ);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmb, ApplyRotaryPosEmbTilingData)

BEGIN_TILING_DATA_DEF(ApplyRotaryPosEmbRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, CosB);
TILING_DATA_FIELD_DEF(int64_t, S);
TILING_DATA_FIELD_DEF(int64_t, D);
TILING_DATA_FIELD_DEF(int64_t, QN);
TILING_DATA_FIELD_DEF(int64_t, KN);
TILING_DATA_FIELD_DEF(int64_t, blockNumB);
TILING_DATA_FIELD_DEF(int64_t, blockFactorB);
TILING_DATA_FIELD_DEF(int64_t, blockNumS);
TILING_DATA_FIELD_DEF(int64_t, blockFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumS);
TILING_DATA_FIELD_DEF(int64_t, ubFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumB);
TILING_DATA_FIELD_DEF(int64_t, ubFactorB);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorB);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumQN);
TILING_DATA_FIELD_DEF(int64_t, ubFactorQN);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorQN);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumKN);
TILING_DATA_FIELD_DEF(int64_t, ubFactorKN);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorKN);
TILING_DATA_FIELD_DEF(int64_t, rotaryMode);
TILING_DATA_FIELD_DEF(int64_t, realDim); // 旋转编码的实际Dim长度
TILING_DATA_FIELD_DEF(bool, isPartialRope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmb_20010, ApplyRotaryPosEmbRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmb_20011, ApplyRotaryPosEmbRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmb_20020, ApplyRotaryPosEmbRegbaseTilingData)

BEGIN_TILING_DATA_DEF(ApplyRotaryPosEmbRegbaseABTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, CosB);
TILING_DATA_FIELD_DEF(int64_t, S);
TILING_DATA_FIELD_DEF(int64_t, D);
TILING_DATA_FIELD_DEF(int64_t, QN);
TILING_DATA_FIELD_DEF(int64_t, KN);
TILING_DATA_FIELD_DEF(int64_t, dAlign);
TILING_DATA_FIELD_DEF(int64_t, dSplitCoef);
TILING_DATA_FIELD_DEF(int64_t, blockNumBS);    // 切分的核数
TILING_DATA_FIELD_DEF(int64_t, blockFactorBS); // 每个核切分的BS大小
TILING_DATA_FIELD_DEF(int64_t, blockTailBS);   // 尾核切分的BS大小
TILING_DATA_FIELD_DEF(int64_t, ubFactorBS);    // 每次循环处理的BS个数
TILING_DATA_FIELD_DEF(int64_t, ubLoopN);       // UB切分的循环次数
TILING_DATA_FIELD_DEF(int64_t, ubFactorN);     // UB切分的整块大小
TILING_DATA_FIELD_DEF(int64_t, ubTailN);       // UB切分的尾块大小
TILING_DATA_FIELD_DEF(int64_t, rotaryMode);
TILING_DATA_FIELD_DEF(int64_t, realDim);
TILING_DATA_FIELD_DEF(bool, isPartialRope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmb_20030, ApplyRotaryPosEmbRegbaseABTilingData)

struct ApplyRotaryPosEmbCompileInfo {
    int64_t numBlocks = 0;
    uint64_t ubSize = 0;
    int64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

enum class ApplyRotaryPosEmbTilingKey : int64_t {
    TILINGKEY_SMALL = 1,
    TILINGKEY_2 = 2,
    TILINGKEY_AB = 3,
    TILINGKEY_AB_CAST = 4,
};

enum class ApplyRotaryPosEmbLayout : int64_t {
    BSND = 1,
    SBND = 2,
    BNSD = 3,
    TND = 4,
};

enum class ApplyRotaryPosEmbRotaryMode : int64_t {
    HALF = 1,
    INTERLEAVE = 2,
    QUARTER = 3,
};

class ApplyRotaryPosEmbRegbaseTilingBaseClass : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit ApplyRotaryPosEmbRegbaseTilingBaseClass(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    virtual ~ApplyRotaryPosEmbRegbaseTilingBaseClass() {}

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

    uint64_t GetTilingKey() const override
    {
        return 0;
    }

    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable() override
    {
        return true;
    }
    platform_ascendc::SocVersion socVersion_;
    int64_t b_;
    int64_t s_;
    int64_t qn_;
    int64_t kn_;
    int64_t d_;
    int64_t cosb_;
    ge::DataType dtype_;
    ApplyRotaryPosEmbLayout layout_;
    ApplyRotaryPosEmbRotaryMode rotaryMode_;
    int64_t dSplitCoef_ = 1;
    int64_t blockSize_;
    int64_t vLength_;
    int64_t reald_;
    bool isPartialRope_ = false;

private:
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtypeAndAttr();
    ge::graphStatus CheckParam();
    ge::graphStatus CheckShapeRelation();
    ge::graphStatus CheckRotaryModeShapeRelation(const int64_t &d);
    ge::graphStatus CheckShapeAllPositive(const int64_t &idx, const gert::Shape &shape);
    ge::graphStatus CheckShapeAllPositive();
    void ConvertRotaryMode();
    void SetBsnd();
    std::string rotaryModeStr_;

    ge::DataType qDataType_;
    ge::DataType kDataType_;
    ge::DataType cosDataType_;
    ge::DataType sinDataType_;
    ge::DataType qOutDataType_;
    ge::DataType kOutDataType_;
    gert::Shape qShape_;
    gert::Shape kShape_;
    gert::Shape cosShape_;
    gert::Shape sinShape_;
    gert::Shape qOutShape_;
    gert::Shape kOutShape_;
};

} // namespace optiling
#endif // AIR_RUNTIME_V2_OP_IMPL_APPLY_ROTARY_POS_EMB_TILING_H
