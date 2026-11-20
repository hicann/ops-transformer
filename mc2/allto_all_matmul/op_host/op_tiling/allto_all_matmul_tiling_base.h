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
#include "mc2/matmul_allto_all/op_host/op_tiling/common/allto_all_hccl_utils.h"
#include "mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/mc2_tiling_struct.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"

namespace MC2Tiling {
class AllToAllMatmulTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AllToAllMatmulTilingBase(gert::TilingContext* context)
        : TilingBaseClass(context)
    {}
    ~AllToAllMatmulTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
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
    // CCU 单次通信量约束下允许的 HCCL 任务数上限；AllToAllMatmul MX 会多一次 x1Scale AllToAll，需预留 1 个
    virtual uint64_t GetHcclMaxCommTasks() const
    {
        return HCCL_MAX_TOTAL_TILES;
    }
    // 单行通信字节口径：AllToAll 单个 peer 的 K 轴宽度（per-peer），不是 rank 扩展后的完整宽度
    virtual CommSizeInfo GetHcclCommSizeInfo() const
    {
        return CommSizeInfo{contextInfo_.args_.orgKValue, contextInfo_.args_.inputDtypeSize};
    }
    void SetUserWorkSpace();
    ge::graphStatus CheckAlltoAllOutShape(const gert::TilingContext* context, const char* opName);
    ge::graphStatus CheckAlltoAllOut(const gert::TilingContext* context, const char* opName);
    ge::graphStatus CheckMatrixMulShapes(const gert::TilingContext* context, const char* opName);
    ge::graphStatus SetAlltoAllMatmulShapeInfo(const gert::TilingContext* context, TilingContextInfo& contextInfo);
    ge::graphStatus CheckKcQuantTensorDataType(const gert::TilingContext* context, const char* opName);
    ge::graphStatus CheckKcQuantShapeInfo(const gert::TilingContext* context, const char* opName,
                                          const OpAttrIndexSchema& indexSchema);

    NpuArch npuArch_;
    const char* opName_{nullptr};
    uint32_t libApiWorkSpaceSize_{0};
    uint32_t cvRatio_{1};
    uint8_t hcclServerType_{0};
    TilingContextInfo contextInfo_;
    TilingInferredInfo inferredInfo_;
#if MC2_DFX_ENABLE
    Utils::DfxWorkspaceLayoutInfo workspaceLayout_{};
#endif

private:
    // CCU 门控及总量前置检查；非 CCU 场景返回成功且 useCcuCommLimit=false
    ge::graphStatus PrepareCcuLimitBeforeTiling(bool& useCcuCommLimit, CommSizeInfo& commSize,
                                                uint64_t& maxCommTasks) const;
    // 功能后移，基类的GetShapeAttrsInfo在isCapable之前，当前将校验和参数获取放到子类的DoOptiling中
    ge::graphStatus GetShapeAttrsInfo() override;
    // 在框架中没有实际使用
    ge::graphStatus DoLibApiTiling() override;
};
} // namespace MC2Tiling
#endif
