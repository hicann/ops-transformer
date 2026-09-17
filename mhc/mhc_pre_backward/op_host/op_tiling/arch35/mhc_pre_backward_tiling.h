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
 * \file mhc_pre_backward_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MHC_PRE_GRAD_H__
#define __OP_HOST_MHC_PRE_GRAD_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "err/ops_err.h"
#include "platform/platform_infos_def.h"
#include "../mhc_pre_backward_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MMConfig)
TILING_DATA_FIELD_DEF(uint32_t, baseM);
TILING_DATA_FIELD_DEF(uint32_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, baseK);
TILING_DATA_FIELD_DEF(uint32_t, l1K);
TILING_DATA_FIELD_DEF(uint32_t, depthA1);
TILING_DATA_FIELD_DEF(uint32_t, depthB1);
TILING_DATA_FIELD_DEF(uint32_t, dbL0A);
TILING_DATA_FIELD_DEF(uint32_t, dbL0B);
TILING_DATA_FIELD_DEF(uint32_t, dbL0C);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MMConfigOp, MMConfig);

BEGIN_TILING_DATA_DEF(MhcPreBackwardTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, vecCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint64_t, nD);
TILING_DATA_FIELD_DEF(uint64_t, N);
TILING_DATA_FIELD_DEF(uint64_t, D);
TILING_DATA_FIELD_DEF(uint64_t, fusionSize); // N ^ 2 + 2 * N
TILING_DATA_FIELD_DEF(float, hcEps);
TILING_DATA_FIELD_DEF(uint32_t, implMode);
TILING_DATA_FIELD_DEF_STRUCT(MMConfig, mmConfigC0);
TILING_DATA_FIELD_DEF_STRUCT(MMConfig, mmConfigC1);
TILING_DATA_FIELD_DEF(uint32_t, maxBufferDepth);
TILING_DATA_FIELD_DEF(uint32_t, l1UsedBytes);
TILING_DATA_FIELD_DEF(uint32_t, l1BOffsetElems);
TILING_DATA_FIELD_DEF(uint32_t, l1SingleBufferElems);
TILING_DATA_FIELD_DEF(uint32_t, l0ABUsedBytes);
TILING_DATA_FIELD_DEF(uint32_t, l0ABSingleBufferElems);
TILING_DATA_FIELD_DEF(uint32_t, l0CUsedBytes);
TILING_DATA_FIELD_DEF(uint32_t, l0CSingleBufferElems);
TILING_DATA_FIELD_DEF(uint32_t, c0MBlock);
TILING_DATA_FIELD_DEF(uint32_t, c0NBlock);
TILING_DATA_FIELD_DEF(uint32_t, c1KBlock);
TILING_DATA_FIELD_DEF(uint32_t, c0ToV2Rows);
TILING_DATA_FIELD_DEF(uint32_t, v2ToC1Rows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPreBackward, MhcPreBackwardTilingData);

class MhcPreBackwardTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPreBackwardTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

    ~MhcPreBackwardTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、库API Tiling阶段
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    ge::graphStatus GetInputShape();
    ge::graphStatus GetInputTensors(const gert::Tensor *&gradHInTensor, const gert::Tensor *&gradHPostTensor,
                                    const gert::Tensor *&gradHResTensor);
    ge::graphStatus ValidateInputDims(int64_t gradHInDims, int64_t gradHPostDims, int64_t gradHResDims);
    ge::graphStatus ParseBSDFormat(const gert::Tensor *gradHInTensor, const gert::Tensor *gradHPostTensor,
                                   const gert::Tensor *gradHResTensor);
    ge::graphStatus ParseTNDFormat(const gert::Tensor *gradHInTensor, const gert::Tensor *gradHPostTensor,
                                   const gert::Tensor *gradHResTensor);
    ge::graphStatus ValidateShapeParams();
    void SetMmConfig();
    ge::graphStatus ValidateMmConfig();
    void PrintTilingData();
    ge::graphStatus ParseInputAndAttr();
    void FillTilingData();
    void SetCommonTilingParams();
    ge::graphStatus TilingProcess();

    static uint64_t CalculateWorkspaceSize(uint64_t totalLength, uint64_t fusionSize, uint64_t cubeCoreNum,
                                           uint64_t vecCoreNum, uint64_t elementSize);

private:
    MhcPreBackwardTilingData tilingData_;
    uint32_t blockDim_{0};   // AIC
    uint32_t vecCoreNum_{0}; // AIV
    uint64_t totalLength_{0};
    uint64_t N_{0};
    uint64_t D_{0};
    float hcEps_{0.0f};
    uint32_t implMode_{0U};
    uint64_t fusionSize_{0};
    uint64_t ubSize_{0};
    uint64_t l1Size_{0};
    uint64_t l2Size_{0};
    uint64_t l0ASize_{0};
    uint64_t l0BSize_{0};
    uint64_t l0CSize_{0};
    uint32_t maxBufferDepth_{0};
    uint32_t l1UsedBytes_{0};
    uint32_t l1BOffsetElems_{0};
    uint32_t l1SingleBufferElems_{0};
    uint32_t l0ABUsedBytes_{0};
    uint32_t l0ABSingleBufferElems_{0};
    uint32_t l0CUsedBytes_{0};
    uint32_t l0CSingleBufferElems_{0};
};

} // namespace optiling
#endif // __OP_HOST_MHC_PRE_GRAD_H__
