/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_V2_GRAD_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_V2_GRAD_H
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MoeV2GradComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, elementCopyLoops);
TILING_DATA_FIELD_DEF(int64_t, elementPerCopyCols);
TILING_DATA_FIELD_DEF(int64_t, elementLastCopyCols);
TILING_DATA_FIELD_DEF(int64_t, binaryAddBufferNum);
TILING_DATA_FIELD_DEF(int64_t, tmpBufferNum);
TILING_DATA_FIELD_DEF(int64_t, exponentOfBinary);
TILING_DATA_FIELD_DEF(int64_t, copyBufferSize);
TILING_DATA_FIELD_DEF(int64_t, tokensFormer);
TILING_DATA_FIELD_DEF(int64_t, perCoreTokensLoop);
TILING_DATA_FIELD_DEF(int64_t, perCoreTailTokensFormer);
TILING_DATA_FIELD_DEF(int64_t, lastCoreTokensLoop);
TILING_DATA_FIELD_DEF(int64_t, lastCoreTailTokensFormer);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeV2GradComputeTilingDataOp, MoeV2GradComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingV2GradTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, e);
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
TILING_DATA_FIELD_DEF_STRUCT(MoeV2GradComputeTilingData, MoeV2GradComputeParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad, MoeInitRoutingV2GradTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingV2GradRegbaseSplitHTilingData)
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, kUbFactor);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, hBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, hUbFactor);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_200001, MoeInitRoutingV2GradRegbaseSplitHTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_200002, MoeInitRoutingV2GradRegbaseSplitHTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_200003, MoeInitRoutingV2GradRegbaseSplitHTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingV2GradRegbaseFullLoadTilingData)
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, nUbFactor);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, hUbFactor);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_300001, MoeInitRoutingV2GradRegbaseFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_300002, MoeInitRoutingV2GradRegbaseFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_300003, MoeInitRoutingV2GradRegbaseFullLoadTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingV2GradRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, nBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, nUbFactor);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, kUbFactor);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, hUbFactor);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_400001, MoeInitRoutingV2GradRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_400002, MoeInitRoutingV2GradRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV2Grad_400003, MoeInitRoutingV2GradRegbaseTilingData)

class MoeInitRoutingV2GradTilingBaseClass : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeInitRoutingV2GradTilingBaseClass(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

private:
    ge::graphStatus CheckParamsValidity(const gert::Shape &xShape, const gert::Shape &rowIdxShape,
                                        const gert::Shape &gradXShape) const;
    ge::graphStatus CheckDtypeValidity();
    ge::graphStatus CheckShapeAllPositive(const gert::Shape &shape, std::string name);
    ge::graphStatus CheckShapeValidity(const gert::Shape &xShape, const gert::Shape &rowIdxShape,
                                       const gert::Shape &gradXShape);

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;
    int64_t ClipMax(int64_t cur, int64_t max)
    {
        return (cur > max) ? max : cur;
    }

    int64_t aivNum = 0;
    int64_t dropPadMode = 0;
    int64_t topK = 1;
    int64_t activeNum = 0;
    int64_t hiddenSize = 0;
    int64_t N = 0;
    int64_t E = 0;
    int64_t C = 0;
    ge::DataType inDtype = ge::DT_FLOAT;
    const char *opName = "";
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

struct MoeInitRoutingV2GradCompileInfo {
    int32_t aivNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_GRAD_H
