/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_APACE_TILING_BASE_H
#define ALLTO_ALL_MATMUL_APACE_TILING_BASE_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <ascendc/host_api/tiling/template_argument.h>
#include "securec.h"
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace/tiling/comm_tiling_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/arch35/allto_all_matmul_v2_tiling_key.h"

namespace MC2Tiling {

using namespace Ops::Transformer::OpTiling;

class AlltoAllMatmulV2TilingClass : public TilingBaseClass {
public:
    explicit AlltoAllMatmulV2TilingClass(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~AlltoAllMatmulV2TilingClass() override = default;

protected:
    uint32_t usedCoreNum_ = 0; // 保存 tiling 实际使用的核数
    bool IsCapable() override;

    ge::graphStatus GetPlatformInfo() override;

    static constexpr size_t IDX_INPUT_X1 = 1;
    static constexpr size_t IDX_INPUT_X2 = 2;
    static constexpr size_t IDX_INPUT_BIAS = 3;
    static constexpr size_t IDX_INPUT_X1_SCALE = 4;
    static constexpr size_t IDX_INPUT_X2_SCALE = 5;
    static constexpr size_t IDX_OUTPUT_Y = 0;
    static constexpr size_t IDX_ATTR_GROUP = 0;
    static constexpr size_t IDX_ATTR_WORLD_SIZE = 1;
    static constexpr size_t IDX_ATTR_HCCL_BUFFER_SIZE = 2;
    static constexpr size_t IDX_ATTR_Y_DTYPE = 3;
    static constexpr size_t IDX_ATTR_X1_QUANT_MODE = 4;
    static constexpr size_t IDX_ATTR_X2_QUANT_MODE = 5;
    static constexpr size_t IDX_ATTR_X1_QUANT_DTYPE = 6;
    static constexpr size_t IDX_ATTR_TRANSPOSE_X1 = 7;
    static constexpr size_t IDX_ATTR_TRANSPOSE_X2 = 8;
    static constexpr size_t IDX_ATTR_GROUP_SIZE = 9;
    static constexpr size_t IDX_ATTR_COMM_MODE = 10;
    static constexpr size_t IDX_ATTR_PRECISION_MODE = 11;
    static constexpr uint64_t MX_SCALE_ALIGN = 64;
    static constexpr uint64_t SCALE_LAST_DIM = 2;
    static constexpr uint64_t SCALE_DIM_NUM = 3;
    static constexpr int64_t MAX_INT32_VAL = 2147483647;
    static constexpr uint64_t K_MAX_VAL = 65535;
    static constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
    static constexpr uint64_t GROUP_M_OFFSET = 32;
    static constexpr uint64_t GROUP_N_OFFSET = 16;
    static constexpr uint64_t MX_GROUP_M = 1;
    static constexpr uint64_t MX_GROUP_N = 1;
    static constexpr uint64_t MX_GROUP_K = 32;
    static constexpr uint64_t MIN_WORLD_SIZE = 2;
    static constexpr uint64_t MAX_WORLD_SIZE = 16;
    static constexpr uint64_t MAX_GROUP_NAME_LEN = 128;

    ge::graphStatus CheckTensorAttrs();

    bool CheckInputDescs(const char *opName);

    bool CheckWorldSizeAttr(const char *opName);

    bool CheckGroupAttr(const char *opName);

    bool CheckTensorFormats(const char *opName);

    bool CheckTensorDtypes(const char *opName);

    bool CheckScaleDesc(const char *opName);

    bool CheckBiasDesc(const char *opName);

    bool CheckQuantAndYDtypeAttr(const char *opName);

    bool CheckOtherAttrs(const char *opName);

    bool CheckGroupSizeAttr(const char *opName);

    ge::graphStatus CheckTensorShapes();

    bool CheckMatmulDimsAndBounds(const char *opName);

    bool CheckMatmulDivisibility(const char *opName);

    bool CheckBiasShape(const char *opName);

    bool CheckScaleShapes(const char *opName);

    ge::graphStatus CheckOpInputInfo();

    ge::graphStatus GetShapeAttrsInfo() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus DoLibApiTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

private:
    uint64_t m_{0}, k_{0}, n_{0};
    uint64_t worldSize_{1};
    uint32_t precisionMode_{0};
#if MC2_DFX_ENABLE
    uint64_t libApiWorkSpaceSize_{0};
#endif
    QuantMatmulPlatformInfo quantPlatformInfo_;
};

} // namespace MC2Tiling

#endif
