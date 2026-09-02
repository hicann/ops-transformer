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
 * \file moe_init_routing_quant_v2_tiling_base.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INNER_MOE_INIT_ROUTING_V2_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INNER_MOE_INIT_ROUTING_V2_H
#include <cmath>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
using namespace Ops::Transformer::OpTiling;
namespace optiling {
BEGIN_TILING_DATA_DEF(InnerMoeV2VBSComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InnerMoeV2VBSComputeTilingDataOp, InnerMoeV2VBSComputeTilingData)

BEGIN_TILING_DATA_DEF(InnerMoeV2VMSMiddleComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InnerMoeV2VMSMiddleComputeTilingDataOp, InnerMoeV2VMSMiddleComputeTilingData)

BEGIN_TILING_DATA_DEF(InnerMoeV2SortOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InnerMoeV2SortOutComputeTilingDataOp, InnerMoeV2SortOutComputeTilingData)

BEGIN_TILING_DATA_DEF(InnerMoeV2GatherOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, activateRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreRows);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreRows);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perLoopCols);
TILING_DATA_FIELD_DEF(int64_t, lastLoopCols);
TILING_DATA_FIELD_DEF(int64_t, colLoops);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InnerMoeV2GatherOutComputeTilingDataOp, InnerMoeV2GatherOutComputeTilingData)

BEGIN_TILING_DATA_DEF(InnerMoeInitRoutingV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, expertCapacity);
TILING_DATA_FIELD_DEF(int64_t, expertNum);
TILING_DATA_FIELD_DEF(int64_t, dropPadMode);
TILING_DATA_FIELD_DEF(int64_t, expertTokensCountOrCumsumFlag);
TILING_DATA_FIELD_DEF(int64_t, expertTokensBeforeCapacityFlag);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2VBSComputeTilingData, vbsComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2VMSMiddleComputeTilingData, vmsMiddleComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2SortOutComputeTilingData, sortOutComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2GatherOutComputeTilingData, srcToDstComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2GatherOutComputeTilingData, srcToDstCapacityComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(InnerMoeV2GatherOutComputeTilingData, gatherOutComputeParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InnerMoeInitRoutingV2, InnerMoeInitRoutingV2TilingData)
struct MoeInitRoutingV2CompileInfo {};

class InnerMoeInitRoutingV2TilingBase : public TilingBaseClass {
public:
    explicit InnerMoeInitRoutingV2TilingBase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~InnerMoeInitRoutingV2TilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    void Reset();

protected:
    ge::graphStatus CheckTokenCount(int64_t num, const char *tag);
    virtual ge::graphStatus CheckOutShape();
    virtual void Tiling4GatherOutCompute() = 0;
    void Tiling4SrcToDstCompute();
    virtual void Tiling4SrcToDstCapacityCompute();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void ShowTilingData();
    void Tiling4VBSMultiCoreCompute(InnerMoeV2VBSComputeTilingData *tilingData);
    void Tiling4VBSOneCoreCompute(InnerMoeV2VBSComputeTilingData *tilingData);
    virtual bool IsFullLoad() = 0;

    int64_t aivNum;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 2040;
    int64_t totalLength = 0;
    int64_t activateNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = 0;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 0;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t inuptXDtypeSize_;
    bool isFullLoad = false;
    bool regBase = false;

    const char *opName = "";
    InnerMoeInitRoutingV2TilingData moeInitRoutingTilingData;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INNER_MOE_INIT_ROUTING_V2_H
