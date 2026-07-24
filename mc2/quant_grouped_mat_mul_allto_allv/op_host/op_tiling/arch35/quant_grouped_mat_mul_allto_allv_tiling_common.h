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
 * \file quant_grouped_mat_mul_allto_allv_tiling_common.h
 * \brief
 */

#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H

#pragma once
#include "securec.h"
#include "tiling/tiling_api.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "../quant_grouped_mat_mul_allto_allv_tiling_base.h"
#include "../../../op_kernel/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"
#include "../../../op_kernel/quant_grouped_mat_mul_allto_allv_tiling_key.h"
#include "register/tilingdata_base.h"

namespace Mc2Tiling {
namespace Mc2GroupedMatmul {

struct QuantGmmAlltoAllvParamsInfo {
    uint64_t A = 0;
    uint64_t H1 = 0;
    uint64_t ep = 0;
    uint64_t BsK = 0;
    uint64_t N1 = 0;
    uint64_t Bs = 0;
    uint64_t H2 = 0;
    uint64_t N2 = 0;
    uint64_t epWorldSize = 0;
    uint64_t aivCoreNum = 0;
    uint64_t aicCoreNum = 0;
    uint64_t gmmWeightDim1 = 0;
    uint64_t gmmWeightDim2 = 0;
    uint64_t mmWeightDim0 = 0;
    uint64_t mmWeightDim1 = 0;
    int64_t gmmXQuantMode = 0;
    int64_t gmmWeightQuantMode = 0;
    int64_t mmXQuantMode = 0;
    int64_t mmWeightQuantMode = 0;
    uint8_t gmmQuantSuit = 0;
    uint8_t mmQuantSuit = 0;
    int64_t commQuantMode = 0;
    int64_t commQuantDtype = 0;
    int64_t attrGmmYDtype = 0;
    int64_t attrMmYDtype = 0;
    int64_t groupSize = 0;
    bool hasSharedMm = 0;
    bool isGmmWeightTrans = 0;
    bool isMmWeightTrans = 0;
    ge::DataType gmmXDtype = ge::DT_UNDEFINED;
    ge::DataType gmmWeightDtype = ge::DT_UNDEFINED;
    ge::DataType gmmXScaleDtype = ge::DT_UNDEFINED;
    ge::DataType gmmWeightScaleDtype = ge::DT_UNDEFINED;
    ge::DataType mmXDtype = ge::DT_UNDEFINED;
    ge::DataType mmWeightDtype = ge::DT_UNDEFINED;
    ge::DataType mmXScaleDtype = ge::DT_UNDEFINED;
    ge::DataType mmWeightScaleDtype = ge::DT_UNDEFINED;
    ge::DataType gmmYDtype = ge::DT_UNDEFINED;
    ge::DataType mmYDtype = ge::DT_UNDEFINED;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    const char *opName = "GMMALLTOALLV";
};

struct TilingInferredInfo {
    uint64_t gmmResultLen = 0UL; // 存储计算GMM的地址大小
    uint64_t commLen = 0UL; // 存储通信结果的临时空间，recvCounts
    uint64_t permuteLen = 0UL; // 重排空间大小, 应该与result一致
    uint32_t biasLen = 0UL; // 暂不支持bias
};

class QuantGroupedMatmulAllToAllvTilingCommon : public QuantGroupedMatmulAllToAllvTilingBase {
public:
    explicit QuantGroupedMatmulAllToAllvTilingCommon(gert::TilingContext *context)
        : QuantGroupedMatmulAllToAllvTilingBase(context) {};
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~QuantGroupedMatmulAllToAllvTilingCommon() override = default;
protected:
    void Reset();
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    virtual bool IsCapable() = 0;
    virtual bool IsContains(const std::vector<uint32_t> &list, uint32_t value);
    ge::graphStatus CheckShapeDimensions(const gert::StorageShape *shape, uint64_t dims, const char *shapeName);
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    virtual ge::graphStatus CheckAndSetInputOutputInfo();
    ge::graphStatus InitTilingContextParameters(); // set默认值，当前不支持功能参数
    ge::graphStatus SetTilingCommonInfo();
    virtual ge::graphStatus SetGmmA2avWorkspaceInfo();
    ge::graphStatus DoQuantGMMTiling(); // 按专家为粒度执行
    ge::graphStatus SetHcclTiling();
    virtual uint32_t GetCommModeIndex() const;
    ge::graphStatus QuantGetAndConvertCommMode(gert::TilingContext *context, uint8_t &commMode) const;
    void PrintQuantGmmA2avTilingData(QuantGmmA2avTilingData &outTilingData);
    uint32_t libApiWorkSpaceSize_{0};
    uint32_t workSpaceSize_{0};
    uint64_t expertNum_{1};
    QuantGmmA2avTilingData localTilingData_;
    TilingInferredInfo inferredInfo_;
    QuantGmmAlltoAllvParamsInfo localParams_;

    virtual ge::graphStatus CheckOpInputSingleParamsTensorNotSupport();
    virtual ge::graphStatus CheckOpInputSingleParamsTensorSupport();
    virtual ge::graphStatus CheckFormat();
    virtual ge::graphStatus CheckOpInputSingleParamsTensorMM();
    virtual ge::graphStatus CheckOpInputSingleParamsTensor();
    virtual ge::graphStatus CheckAndSetLocalParamsGmm();
    virtual ge::graphStatus CheckAndSetLocalParamsMm();
    virtual ge::graphStatus CheckAndSetLocalParamsAttr();
    virtual ge::graphStatus CheckAndSetLocalParams();
    virtual ge::graphStatus CheckParamsRelationGmmTransShape();
    virtual ge::graphStatus CheckParamsRelationGmm();
    virtual ge::graphStatus CheckParamsRelationMmTransShape();
    virtual ge::graphStatus CheckParamsRelationMm();
    virtual ge::graphStatus CheckParamsAttrEpAndSetLocalParams();
    virtual ge::graphStatus CheckAndSetSendRecvCountsAttr();
    virtual ge::graphStatus CheckLocalParams();
    virtual ge::graphStatus CheckParamsRelationAndSetLocalParams();
};

} // namespace Mc2GroupedMatmul
}
#endif
