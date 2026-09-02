/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_finalize_routing_v2_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_FINALIZE_ROUTING_V2_GRAD_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_FINALIZE_ROUTING_V2_GRAD_TILING_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"

namespace optiling {
// 基础tiling数据，为保证兼容性，延用旧tiling
BEGIN_TILING_DATA_DEF(MoeFinalizeRoutingV2GradTilingData)
TILING_DATA_FIELD_DEF(int64_t, initOutNeedCoreNum);      // 初始化输出过程中需要的核数
TILING_DATA_FIELD_DEF(int64_t, initOutEachCoreBatchNum); // 初始化输出过程中每个核需要处理的行数
TILING_DATA_FIELD_DEF(int64_t, initOutModCoreNum);       // 初始化输出过程中需要多处理1行的核数
TILING_DATA_FIELD_DEF(int64_t, computeNeedCoreNum);      // 运算过程中需要的核数
TILING_DATA_FIELD_DEF(int64_t, computeEachCoreBatchNum); // 运算过程中每个核需要处理的行数
TILING_DATA_FIELD_DEF(int64_t, computeModCoreNum);       // 运算过程中需要多处理1行的核数
TILING_DATA_FIELD_DEF(int64_t, dropPadMode);             // drop_pad_mode
TILING_DATA_FIELD_DEF(int64_t, topK);                    // k的大小 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, hidden);                  // H的大小 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, expandedXDim0); // grad_expanded_x除最后一维外，其余维度合成一维后的大小
TILING_DATA_FIELD_DEF(int64_t, hiddenPrePart);    // hidden切分后前块元素个数 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, hiddenInnerLoops); // hidden切分后前块循环次数
TILING_DATA_FIELD_DEF(int64_t, hiddenLastPart);   // hidden切分后尾块元素个数 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, tilingKey);        // 模板
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFinalizeRoutingV2Grad, MoeFinalizeRoutingV2GradTilingData);

BEGIN_TILING_DATA_DEF(MoeFinalizeRoutingV2GradBaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, initOutNeedCoreNum);      // 初始化输出过程中需要的核数
TILING_DATA_FIELD_DEF(int64_t, initOutEachCoreBatchNum); // 初始化输出过程中每个核需要处理的行数
TILING_DATA_FIELD_DEF(int64_t, initOutModCoreNum);       // 初始化输出过程中需要多处理1行的核数
TILING_DATA_FIELD_DEF(int64_t, computeNeedCoreNum);      // 运算过程中需要的核数
TILING_DATA_FIELD_DEF(int64_t, computeEachCoreBatchNum); // 运算过程中每个核需要处理的行数
TILING_DATA_FIELD_DEF(int64_t, computeModCoreNum);       // 运算过程中需要多处理1行的核数
TILING_DATA_FIELD_DEF(int64_t, dropPadMode);             // drop_pad_mode
TILING_DATA_FIELD_DEF(int64_t, topK);                    // k的大小 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, hidden);                  // H的大小 单位元素个数
TILING_DATA_FIELD_DEF(int64_t, expandedXDim0); // grad_expanded_x除最后一维外，其余维度合成一维后的大小
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFinalizeRoutingV2GradBaseTilingDataOp, MoeFinalizeRoutingV2GradBaseTilingData)

BEGIN_TILING_DATA_DEF(MoeFinalizeRoutingV2GradBinaryAddTilingData)
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, binaryAddk);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLastNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFinalizeRoutingV2GradBinaryAddTilingDataOp, MoeFinalizeRoutingV2GradBinaryAddTilingData);

BEGIN_TILING_DATA_DEF(MoeFinalizeRoutingV2GradNotSplitHTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MoeFinalizeRoutingV2GradBaseTilingData, baseParams);        // 基础tiling信息
TILING_DATA_FIELD_DEF(int64_t, hAlign);                                                  // hidden按块对齐
TILING_DATA_FIELD_DEF_STRUCT(MoeFinalizeRoutingV2GradBinaryAddTilingData, binAddParams); // 二分累加信息
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeFinalizeRoutingV2Grad_20011, MoeFinalizeRoutingV2GradNotSplitHTilingData)
REGISTER_TILING_DATA_CLASS(MoeFinalizeRoutingV2Grad_20021, MoeFinalizeRoutingV2GradNotSplitHTilingData)

BEGIN_TILING_DATA_DEF(MoeFinalizeRoutingV2GradSplitHTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MoeFinalizeRoutingV2GradBaseTilingData, baseParams);
TILING_DATA_FIELD_DEF(int64_t, mainFactor);
TILING_DATA_FIELD_DEF(int64_t, mainBlockNum);
TILING_DATA_FIELD_DEF(int64_t, foldFactor);
TILING_DATA_FIELD_DEF(int64_t, foldBlockNum);
TILING_DATA_FIELD_DEF(int64_t, foldTailLen);
TILING_DATA_FIELD_DEF(int64_t, foldTailBlockNum);
TILING_DATA_FIELD_DEF_STRUCT(MoeFinalizeRoutingV2GradBinaryAddTilingData, mainBinAddParams);
TILING_DATA_FIELD_DEF_STRUCT(MoeFinalizeRoutingV2GradBinaryAddTilingData, foldBinAddParams);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFinalizeRoutingV2Grad_20012, MoeFinalizeRoutingV2GradSplitHTilingData);
REGISTER_TILING_DATA_CLASS(MoeFinalizeRoutingV2Grad_20022, MoeFinalizeRoutingV2GradSplitHTilingData);

struct MoeFinalizeRoutingV2GradCompileInfo {
    int64_t aivNum = 0;
    int64_t ubSize = 0;
};

constexpr int64_t BUFFER_NUM_WITH_BIAS = 4;    // gradY + expandedX + gradExpandedX + bias
constexpr int64_t BUFFER_NUM_WITHOUT_BIAS = 3; // gradY + expandedX + gradExpandedX
constexpr int64_t ULONG_BIT_LEN = 64;

class MoeFinalizeRoutingV2GradTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeFinalizeRoutingV2GradTiling(gert::TilingContext *context)
        : TilingBaseClass(context),
          nodeName_(context->GetNodeName()) {};

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        nodeName_ = context->GetNodeName();
    }

protected:
    // 重载TilingBase基类方法
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    // 内部计算方法
    virtual void CalcBaseInfo();
    void SetBinaryAddParams(MoeFinalizeRoutingV2GradBinaryAddTilingData &params, int64_t factor);
    virtual ge::graphStatus CalcTilingKey() = 0;
    ge::graphStatus GetRequiredTensorInfo();
    ge::graphStatus GetOptionalTensorInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckRequiredInput();
    ge::graphStatus CheckOptionalInputShape();
    virtual ge::graphStatus CheckOptionalInputDtype();
    ge::graphStatus CheckBiasExpertIdxDtype();
    ge::graphStatus CheckExpandedXAndRowIdxDtype();
    ge::graphStatus CheckScalesDtypeForDavid();
    ge::graphStatus CheckOutput();
    ge::graphStatus CheckGradExpandedXShape();
    ge::graphStatus CheckGradExpandedXDtype();
    ge::graphStatus CheckGradScalesShapeAndDtype();
    ge::graphStatus CheckParams();
    ge::graphStatus CalcExpandedXDimInfo();
    ge::graphStatus CheckShapeWithScales();
    ge::graphStatus CheckShapeWithoutScales();
    std::string DimNumToString(size_t dimNum) const
    {
        return std::to_string(dimNum) + "D";
    }
    std::string DtypeToString(ge::DataType dtype) const
    {
        return Ops::Base::ToString(dtype);
    }
    std::string ShapeToString(const gert::Shape &shape) const
    {
        return Ops::Base::ToString(shape);
    }

protected:
    // platform info
    std::string nodeName_ = "";
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t blockSize_ = 32U;
    uint64_t vlFp32_ = 0;

    // shape
    gert::Shape gradYShape_;
    gert::Shape expandedRowIdxShape_;
    gert::Shape expandedXShape_;
    gert::Shape scalesShape_;
    gert::Shape expertIdxShape_;
    gert::Shape biasShape_;
    gert::Shape gradExpandedXShape_;
    gert::Shape gradScalesShape_;

    // dtype
    ge::DataType gradYType_;
    ge::DataType expandedRowIdxType_;
    ge::DataType expandedXType_;
    ge::DataType scalesType_;
    ge::DataType expertIdxType_;
    ge::DataType biasType_;
    ge::DataType gradExpandedXType_;
    ge::DataType gradScalesType_;
    int64_t gradYTypeByteSize_ = 0;

    // attr
    int64_t activeNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t expertCapacity_ = 0;

    // expanded_x
    int64_t expandedXDimNum_ = 0;

    // scales
    bool isScalesExist_ = false;

    // bias
    bool isBiasExist_ = false;

    // tiling data
    int64_t initOutNeedCoreNum_ = 0;
    int64_t initOutEachCoreBatchNum_ = 0;
    int64_t initOutModCoreNum_ = 0;
    int64_t computeNeedCoreNum_ = 0;
    int64_t computeEachCoreBatchNum_ = 0;
    int64_t computeModCoreNum_ = 0;
    int64_t dropPadMode_ = 0;
    int64_t topK_ = 0;
    int64_t hidden_ = 0;
    int64_t expandedXDim0_ = 0;
    int64_t hiddenPrePart_ = 0;
    int64_t hiddenInnerLoops_ = 0;
    int64_t hiddenLastPart_ = 0;
    uint64_t tilingKey_ = 0;
    MoeFinalizeRoutingV2GradTilingData tilingData_;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_FINALIZE_ROUTING_V2_GRAD_TILING_H
