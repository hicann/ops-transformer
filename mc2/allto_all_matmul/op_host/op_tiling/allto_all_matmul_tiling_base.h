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
 * \file allto_all_matmul_tiling_base.h
 * \brief
 */
#ifndef ALLTO_ALL_MATMUL_TILING_BASE_H
#define ALLTO_ALL_MATMUL_TILING_BASE_H

#include "mc2/matmul_allto_all/op_host/op_tiling/common/allto_all_formulaic_tiling.h"
#include "mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"

namespace MC2Tiling {
class AllToAllMatmulTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AllToAllMatmulTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~AllToAllMatmulTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus CheckInput()
    {
        return ge::GRAPH_SUCCESS;
    }
    virtual CutResult GetCutResOfCommAndCompute();
    ge::graphStatus TileCommAndCompute();
    void SetUserWorkSpace();
    ge::graphStatus CheckAlltoAllOutShape(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckAlltoAllOut(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckMatrixMulShapes(const gert::TilingContext *context, const char *opName);
    ge::graphStatus SetAlltoAllMatmulShapeInfo(const gert::TilingContext *context, TilingContextInfo &contextInfo);
    ge::graphStatus CheckKcQuantTensorDataType(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckKcQuantShapeInfo(const gert::TilingContext *context, const char *opName,
                                          const OpAttrIndexSchema &indexSchema);

    NpuArch npuArch_;
    const char *opName_{nullptr};
    uint32_t libApiWorkSpaceSize_{0};
    TilingContextInfo contextInfo_;
    TilingInferredInfo inferredInfo_;

private:
    // 功能后移，基类的GetShapeAttrsInfo在isCapable之前，当前将校验和参数获取放到子类的DoOptiling中
    ge::graphStatus GetShapeAttrsInfo() override;
    // 在框架中没有实际使用
    ge::graphStatus DoLibApiTiling() override;
};
} // namespace MC2Tiling
#endif
