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
 * \file masked_causal_conv1d_tiling_arch35.h
 * \brief MaskedCausalConv1d tiling class declaration for Arch35 (ascend950)
 */

#ifndef MASKED_CAUSAL_CONV1D_TILING_ARCH35_H
#define MASKED_CAUSAL_CONV1D_TILING_ARCH35_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "platform/platform_ascendc.h"
#include "../op_kernel/arch35/masked_causal_conv1d_struct.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

class MaskedCausalConv1dTilingArch35 : public TilingBaseClass {
public:
    explicit MaskedCausalConv1dTilingArch35(gert::TilingContext *context)
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
    ge::graphStatus GetInputShapes();
    ge::graphStatus GetInputDtypes();
    ge::graphStatus CheckInputParams();
    ge::graphStatus SearchBestCoreSplit();
    ge::graphStatus CalcUbTiling();
    ge::graphStatus CalcLoopParams(int64_t len, int64_t factor, int64_t &loopNum, int64_t &tailFactor);

    // Hardware
    uint64_t ubSize_ = 0;
    uint64_t coreNum_ = 0;
    uint64_t ubBlockSize_ = 0;

    // Shape
    uint64_t S_ = 0, B_ = 0, H_ = 0;
    ge::DataType xType_;
    uint32_t xDtypeSize_ = 0;
    uint64_t hReg_ = 64; // 128 / xDtypeSize_: min H granularity (bf16/fp16→64, fp32→32)

    // Mask None flag
    int64_t isMaskNone_ = 0;

    // Inter-core split result
    uint64_t hCoreCnt_ = 1, hMainCnt_ = 0, hBlockFactor_ = 0, hBlockTailFactor_ = 0;
    uint64_t bCoreCnt_ = 1, bMainCnt_ = 0, bBlockFactor_ = 0, bBlockTailFactor_ = 0;
    uint64_t sCoreCnt_ = 1, sMainCnt_ = 0, sBlockFactor_ = 0, sBlockTailFactor_ = 0;
    uint64_t realCoreNum_ = 1;

    // UB tile sizes
    int64_t ubFactorH_ = 0;
    int64_t ubFactorB_ = 1;
    int64_t ubFactorS_ = 1;

    // Loop params — main core
    int64_t loopNumH_ = 1, ubTailFactorH_ = 0;
    int64_t loopNumB_ = 1, ubTailFactorB_ = 1;
    int64_t loopNumS_ = 1, ubTailFactorS_ = 1;

    // Loop params — tail core
    int64_t tailBlockLoopNumH_ = 1, tailBlockUbTailFactorH_ = 0;
    int64_t tailBlockLoopNumB_ = 1, tailBlockUbTailFactorB_ = 1;
    int64_t tailBlockLoopNumS_ = 1, tailBlockUbTailFactorS_ = 1;

    uint64_t workspaceSize_ = 0;

    gert::Shape xShape_;
    gert::Shape weightShape_;
    gert::Shape maskShape_;
};

} // namespace optiling

#endif // MASKED_CAUSAL_CONV1D_TILING_ARCH35_H
