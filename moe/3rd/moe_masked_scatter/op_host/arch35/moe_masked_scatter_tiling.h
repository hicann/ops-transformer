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
 * \file moe_masked_scatter_tiling.h
 * \brief
 */

#ifndef OPS_OP_TILING_RUNTIME_MOE_MASKED_SCATTER_TILING_H_
#define OPS_OP_TILING_RUNTIME_MOE_MASKED_SCATTER_TILING_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling {

struct MoeMaskedScatterCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

BEGIN_TILING_DATA_DEF(MoeMaskedScatterTilingData)
TILING_DATA_FIELD_DEF(int64_t, normalCoreData);
TILING_DATA_FIELD_DEF(int64_t, tailCoreData);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailUbFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreTailUbFactor);
TILING_DATA_FIELD_DEF(int64_t, updateEleNums);
TILING_DATA_FIELD_DEF(int64_t, bufferSize);
TILING_DATA_FIELD_DEF(int64_t, prefixSumUbSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeMaskedScatter, MoeMaskedScatterTilingData)

ge::graphStatus MoeMaskedScatterTilingForAscendC(gert::TilingContext *context);

class MoeMaskedScatterTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeMaskedScatterTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    ge::graphStatus CheckDataType();
    ge::graphStatus CheckOutputShape();

private:
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t normalCoreData_ = 0;
    int64_t tailCoreData_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubFactor_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t tailUbFactor_ = 0;
    int64_t tailCoreTailUbFactor_ = 0;
    int64_t updateEleNums_ = 0;
    int64_t xEleNums_ = 0;
    int64_t bufferSize_ = 0;
    int64_t prefixSumUbSize_ = 0;
    int64_t dataTypeSize_ = 0;

    ge::DataType dType_ = ge::DT_UNDEFINED;
    gert::Shape inputShape_;
    bool isInt64Mask_ = false;
    MoeMaskedScatterTilingData tilingData_;
};
} // namespace optiling
#endif // OPS_OP_TILING_RUNTIME_MASKED_SCATTER_TILING_H_
