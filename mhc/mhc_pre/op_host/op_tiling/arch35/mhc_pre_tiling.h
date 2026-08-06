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
 * \file mhc_pre_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MHC_PRE_TILING_H__
#define __OP_HOST_MHC_PRE_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "err/ops_err.h"

namespace optiling {

enum class TilingMode : uint8_t {
    SPLIT_BS = 0,
    SPLIT_ND = 1,
    SPLIT_M_K = 2
};

BEGIN_TILING_DATA_DEF(MhcPreTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, outFlag);
TILING_DATA_FIELD_DEF(uint32_t, hasGamma);
TILING_DATA_FIELD_DEF(uint32_t, hasResi);
TILING_DATA_FIELD_DEF(uint32_t, chunkTSize);
TILING_DATA_FIELD_DEF(uint32_t, v1ChunkDSize);
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint64_t, nD);
TILING_DATA_FIELD_DEF(uint64_t, fusionSize);
TILING_DATA_FIELD_DEF(uint64_t, N);
TILING_DATA_FIELD_DEF(uint64_t, D);
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(float, hcEps);
TILING_DATA_FIELD_DEF(float, scaleMean);
TILING_DATA_FIELD_DEF(uint32_t, mL1Size);
TILING_DATA_FIELD_DEF(uint32_t, kUbSize);
TILING_DATA_FIELD_DEF(uint32_t, kL1Size);
TILING_DATA_FIELD_DEF(uint32_t, fusionAlign);
TILING_DATA_FIELD_DEF(uint32_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(uint32_t, cubeBlockDimK);
TILING_DATA_FIELD_DEF(uint32_t, multCoreSplitKSize);
TILING_DATA_FIELD_DEF(uint32_t, mkWorkspaceMmOffset);
TILING_DATA_FIELD_DEF(uint32_t, mkWorkspaceRmsOffset);
TILING_DATA_FIELD_DEF(uint32_t, mkWorkspaceFinalOffset);
TILING_DATA_FIELD_DEF(uint32_t, mkUseGmStage);
TILING_DATA_FIELD_DEF(uint32_t, stage2UsedAivNum);
TILING_DATA_FIELD_DEF(uint32_t, stage2RowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, implMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPre, MhcPreTilingData);

struct MhcPreCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
};

class MhcPreBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPreBaseTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

    ~MhcPreBaseTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
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
    ge::graphStatus CheckDescAndShape();
    ge::graphStatus CheckShapePositive();
    ge::graphStatus CheckDataType();
    ge::graphStatus CheckDataRange();
    ge::graphStatus CheckOutputShapeConsistency();
    ge::graphStatus CheckBsndOutputShape(uint64_t b, uint64_t s, uint64_t n, uint64_t d);
    ge::graphStatus CheckTndOutputShape(uint64_t t, uint64_t n, uint64_t d);
    ge::graphStatus ParseBsndFormat(const gert::Tensor *xTensor);
    ge::graphStatus ParseTndFormat(const gert::Tensor *xTensor);
    ge::graphStatus ValidateAndSetTilingParams(const gert::Tensor *xTensor);
    ge::graphStatus InitPlatformMemory();
    ge::graphStatus ParseOutputFlags();
    ge::graphStatus ParseEpsAttributes();
    void PrintTilingData();
    ge::graphStatus ParseInputAndAttr();
    void FillTilingData();
    ge::graphStatus TilingProcess();
    ge::graphStatus CheckUbBufferSize();
    ge::graphStatus CalculateBsTiling();
    ge::graphStatus CalculateNdTiling();
    ge::graphStatus CalculateMkTiling();
    bool CanUseMk() const;
    uint32_t SelectMkStage() const;
    size_t GetLastRequiredInputIndex() const;

private:
    MhcPreTilingData tilingData_;
    uint32_t blockDim_ = 32; // AIC
    uint32_t aivBlockDim_ = 64;
    uint64_t totalLength_ = 0;
    uint64_t m_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t matM_ = 0;
    uint64_t matK_ = 0;
    uint64_t matN_ = 0;
    uint64_t nD_ = 0;
    uint64_t N_ = 0;
    uint64_t D_ = 0;
    float normEps_ = 1e-6f;
    float hcEps_ = 1e-6f;
    bool outFlag_ = false;
    uint32_t hasGamma_ = 0;
    uint32_t chunkTSize_ = 0;
    uint32_t v1ChunkDSize_ = 0;
    uint32_t usedCoreNum_ = 0;
    float scaleMean_ = 0.0f;
    uint32_t implMode_ = 0;
    TilingMode tilingMode_ = TilingMode::SPLIT_BS;
    uint32_t hasResi_ = 1;
};

} // namespace optiling
#endif // __OP_HOST_MHC_PRE_TILING_H__
