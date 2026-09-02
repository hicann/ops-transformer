/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mhc_sinkhorn_backward_tiling.h
 * \brief mhc_sinkhorn_backward_tiling
 */

#ifndef MHC_SINKHORN_BACKWARD_TILING_H_
#define MHC_SINKHORN_BACKWARD_TILING_H_
#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../op_kernel/arch35/mhc_sinkhorn_backward_tiling_data.h"
#include "../op_kernel/arch35/mhc_sinkhorn_backward_tiling_key.h"
#include "../../mhc_sinkhorn_common/op_host/arch35/mhc_sinkhorn_base_tiling.h"

namespace optiling {
using Ops::Transformer::OpTiling::TilingBaseClass;

struct MhcSinkhornBackwardCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

class MhcSinkhornBackwardTiling : public MhcSinkhornBaseTiling {
public:
    explicit MhcSinkhornBackwardTiling(gert::TilingContext *context)
        : MhcSinkhornBaseTiling(context)
    {}

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
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    // 8、Dump Tiling数据
    void DumpTilingInfo() override;

private:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckGradYShape();
    ge::graphStatus CheckNormShape();
    ge::graphStatus CheckSumShape();
    ge::graphStatus CheckGradInputShape();
    ge::graphStatus SplitCores();
    void SetTilingData();

    // Shape Info
    const gert::Shape *gradYShape_ = nullptr;
    const gert::Shape *normShape_ = nullptr;
    const gert::Shape *sumShape_ = nullptr;
    const gert::Shape *gradInputShape_ = nullptr;
    int64_t tSize_ = 0;
    int64_t bSize_ = 0;
    int64_t sSize_ = 0;
    int64_t nSize_ = 0;
    int64_t nAlignSize_ = 0;
    int64_t totalLength_ = 0;
    // Dtype Info
    int64_t inputDTypeSize_ = 0;
    // Attr Info
    int64_t numIters_ = 0;
    bool isTShape_ = false;
    // Platform Info
    int64_t totalCoreNum_ = 0;
    int64_t ubSize_ = 0;
    // SplitCore Info
    int64_t needCoreNum_ = 0; // 使用到的核数
    int64_t coreTSize_ = 0;   // 每个核处理的数量

    int64_t normCoreTLoops_ = 0;        // 普通核循环次数
    int64_t normCorePerLoopTSize_ = 0;  // 普通核每次循环处理的数据量
    int64_t normCoreLastLoopTSize_ = 0; // 普通核最后一次循环处理的数据量

    int64_t tailCoreTSize_ = 0;         // 尾核处理的数据量
    int64_t tailCoreLoops_ = 0;         // 尾核循环次数
    int64_t tailCorePerLoopTSize_ = 0;  // 尾核每次循环处理的数据量
    int64_t tailCoreLastLoopTSize_ = 0; // 尾核最后一次循环处理的数据量

    const char *opName_ = "MhcSinkhornBackward";
    MhcSinkhornBackwardTilingData tilingData_;
};
} // namespace optiling
#endif // MHC_SINKHORN_BACKWARD_TILING_H_
