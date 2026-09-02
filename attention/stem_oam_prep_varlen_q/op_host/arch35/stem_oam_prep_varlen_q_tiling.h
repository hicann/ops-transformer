/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "op_host/tiling_base.h"
#include "../../op_kernel/arch35/stem_oam_prep_varlen_q_tiling_data.h"

namespace optiling {

struct StemPrepQCompileInfo {
    int32_t coreNum;
};

namespace stem_oam_prep_varlen_q {

using namespace Ops::Transformer::OpTiling;

class StemOamPrepVarlenQTiling : public TilingBaseClass {
public:
    explicit StemOamPrepVarlenQTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~StemOamPrepVarlenQTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus ValidateAttrs();
    ge::graphStatus ValidateInputShapes();
    ge::graphStatus ValidateDtypes();
    ge::graphStatus ValidateConsistency();
    ge::graphStatus ValidateCuSeqLens(const int64_t *cuSeqLensData, const int64_t *qSeqLensData, uint32_t batchU32);
    ge::graphStatus ValidateOutputShape();
    ge::graphStatus CalcExpectedMaxQb();
    void CalcCoreDistribution();
    ge::graphStatus CalcUBFactor();

    StemPrepQTilingData *tilingData_ = nullptr;
    const char *opName_ = nullptr;
    int64_t stemBlockSize_ = 0;
    int64_t stemStride_ = 0;
    int64_t batch_ = 0;
    int64_t numQHeads_ = 0;
    ge::DataType qDtype_ = ge::DT_UNDEFINED;
    uint32_t totalBlocks_ = 0;
    uint32_t maxQb_ = 0;
    uint32_t totalTokens_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t blockDim_ = 0;
};

} // namespace stem_oam_prep_varlen_q
} // namespace optiling
