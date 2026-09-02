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
 * \file matmul_allto_all_tiling_base.h
 * \brief
 */
#ifndef MATMUL_ALLTO_ALL_TILING_BASE_H
#define MATMUL_ALLTO_ALL_TILING_BASE_H

#include "tiling/tiling_api.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "common/matmul_allto_all_util_tiling.h"
#include "common/allto_all_formulaic_tiling.h"

namespace MC2Tiling {

class MatmulAllToAllTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MatmulAllToAllTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~MatmulAllToAllTilingBase() override = default;

protected:
    uint64_t GetTilingKey() const override = 0;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus CheckInput()
    {
        return ge::GRAPH_SUCCESS;
    }
    virtual CutResult GetTilingResult();
    ge::graphStatus TileCommAndCompute();
    void SetUserWorkSpace();
    ge::graphStatus Check2DMatrixMulShapes(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckKcQuantMatrixMulShapes(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckKcQuantScaleShapes(const gert::TilingContext *context, const char *opName);
    platform_ascendc::SocVersion socVersion_;
    NpuArch npuArch_;
    uint32_t libApiWorkSpaceSize_{0};
    const char *opName_{nullptr};
    TilingContextInfo contextInfo;
    TilingInferredInfo inferredInfo;

private:
    // 功能后移，MatmulAlltoAll的GetShapeAttrsInfo在isCapable之前，当前将校验和参数获取放到子类的DoOptiling中
    ge::graphStatus GetShapeAttrsInfo() override;
    // 在框架中没有实际使用
    ge::graphStatus DoLibApiTiling() override;
};
} // namespace MC2Tiling
#endif
