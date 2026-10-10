/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <vector>
#include "grouped_matmul_quant_tiling.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        printf("nullptr error!"); \
        return ge::GRAPH_FAILED; \
    }

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret) \
    if ((ptr) == nullptr) { \
        const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
        printf("EZ9999 op[%s], %s is nullptr!", name, #ptr); \
        return ret; \
    }

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) printf(err_msg, ##__VA_ARGS__)

#define OP_TILING_CHECK(cond, log_func, expr) \
    do { \
        if (cond) { \
            log_func; \
            fflush(stdout); \
            expr; \
        } \
    } while (0)

bool AddWorkspaceGMM(gert::TilingContext* context, const size_t workspace)
{
    size_t* workspace_size = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
    *workspace_size = workspace;
    return true;
}

bool GroupedMatmulQuantTiling::GetPlatformInfo(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, platformInfo, false);

    _Params.CoreNum = platformInfo->GetCoreNum() / 2;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, _Params.UBSize);
    platformInfo->GetLocalMemSize(fe::LocalMemType::L1, _Params.L1Size);
    platformInfo->GetLocalMemSize(fe::LocalMemType::L0_A, _Params.L0ASize);
    platformInfo->GetLocalMemSize(fe::LocalMemType::L0_B, _Params.L0BSize);
    platformInfo->GetLocalMemSize(fe::LocalMemType::L0_C, _Params.L0CSize);
    return true;
}

bool GroupedMatmulQuantTiling::CheckTensorShape(gert::TilingContext* context, gert::Shape& shape, uint64_t ndim,
                                                std::vector<uint64_t> dims)
{
    OP_TILING_CHECK(
        shape.GetDimNum() != ndim,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                        "GroupedMatmulQuant tensor shape ndim:%lu wrong expected %lu, please check.",
                                        static_cast<uint64_t>(shape.GetDimNum()), ndim),
        return false);
    for (uint32_t i = 0; i < ndim; i++) {
        OP_TILING_CHECK(
            static_cast<uint64_t>(shape.GetDim(i)) != dims[i],
            VECTOR_INNER_ERR_REPORT_TILIING(
                context->GetNodeName(), "GroupedMatmulQuant tensor shape dim[%d]:%lu wrong expected %lu, please check.",
                i, static_cast<uint64_t>(shape.GetDim(i)), dims[i]),
            return false);
    }
    return true;
}

bool GroupedMatmulQuantTiling::CheckInOutShapes(gert::TilingContext* context)
{
    uint32_t idx = 0;
    auto x = context->GetInputShape(idx++);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, x, false);
    auto shape = x->GetStorageShape();
    OP_TILING_CHECK(shape.GetDimNum() != 2,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "GroupedMatmulQuant get x shape ndim is not 2, please check."),
                    return false);
    OP_TILING_CHECK(
        shape.GetDim(1) % _Params.scaleGroupSize != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                        "GroupedMatmulQuant get x shape k-dim not aligned to %lu, please check.",
                                        _Params.scaleGroupSize),
        return false);
    _Params.originM = (uint64_t)shape.GetDim(0);
    _Params.originK = (uint64_t)shape.GetDim(1);
    _Params.fracK = _Params.originK / FRACTAL_FLOAT16;
    _Params.scaleK = _Params.originK / _Params.scaleGroupSize;

    auto quantized_weight = context->GetInputShape(idx++);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, quantized_weight, false);
    shape = quantized_weight->GetStorageShape();
    OP_TILING_CHECK(
        shape.GetDimNum() != 5,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                        "GroupedMatmulQuant get quantized_weight shape ndim is not 5, please check."),
        return false);
    OP_TILING_CHECK((uint64_t)shape.GetDim(1) != _Params.fracK || shape.GetDim(3) != FRACTAL_FLOAT16 ||
                        shape.GetDim(4) != (FRACTAL_FLOAT16 / 8),
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        context->GetNodeName(),
                        "GroupedMatmulQuant get quantized_weight shape not (G, K//16, N//16, 16, 2), please check."),
                    return false);
    _Params.originE = (uint64_t)shape.GetDim(0);
    _Params.fracN = (uint64_t)shape.GetDim(2);
    _Params.originN = _Params.fracN * FRACTAL_FLOAT16;

    auto weight_scale = context->GetInputShape(idx++);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, weight_scale, false);
    shape = weight_scale->GetStorageShape();
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {_Params.originE, _Params.scaleK, _Params.originN}),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "GroupedMatmulQuant get weight_scale shape wrong, please check."),
                    return false);

    auto weight_offset = context->GetInputShape(idx++);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, weight_offset, false);
    shape = weight_offset->GetStorageShape();
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {_Params.originE, _Params.scaleK, _Params.originN}),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "GroupedMatmulQuant get weight_offset shape wrong, please check."),
                    return false);

    auto group_list = context->GetInputShape(idx++);
    if (group_list == nullptr) {
        OP_TILING_CHECK(_Params.originE != 1,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            context->GetNodeName(), "GroupedMatmulQuant get group_list null when G!=1, please check."),
                        return false);
        _Params.noGroup = 1;
    } else {
        _Params.noGroup = 0;
        shape = group_list->GetStorageShape();
        OP_TILING_CHECK(!CheckTensorShape(context, shape, 1, {_Params.originE}),
                        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                        "GroupedMatmulQuant get group_list shape wrong, please check."),
                        return false);
    }

    auto y = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, y, false);
    shape = y->GetStorageShape();
    OP_TILING_CHECK(
        !CheckTensorShape(context, shape, 2, {_Params.originM, _Params.originN}),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GroupedMatmulQuant get y shape wrong, please check."),
        return false);
    return true;
}

bool GroupedMatmulQuantTiling::GetTilingData(gert::TilingContext* context)
{
    _Params.splitK = (_Params.originM / 128) * (_Params.originN / 256) < (_Params.CoreNum * 9) ? 1 : 0;

    _Params.clearBaseN = _Params.UBSize / 4;
    uint64_t total_count = _Params.originM * _Params.originN;
    uint64_t step_count = 2 * _Params.CoreNum * _Params.clearBaseN;

    _Params.clearOutLoop = total_count / step_count;
    uint64_t total_count_tail = total_count % step_count;
    uint64_t total_repeat_tail = (total_count_tail + 128 - 1UL) / 128;
    _Params.clearOutTailN = total_repeat_tail / (_Params.CoreNum * 2) * 128;
    _Params.clearOutTailCoreNum = total_repeat_tail % (_Params.CoreNum * 2);

    _Params.castBaseN = _Params.UBSize / 6 / 2;
    total_count = _Params.originM * _Params.originN;
    step_count = 2 * _Params.CoreNum * _Params.castBaseN;

    _Params.castOutLoop = total_count / step_count;
    total_count_tail = total_count % step_count;
    total_repeat_tail = (total_count_tail + 256 - 1UL) / 256;
    _Params.castOutTailN = total_repeat_tail / (_Params.CoreNum * 2) * 256;
    _Params.castOutTailCoreNum = total_repeat_tail % (_Params.CoreNum * 2);
    return true;
}

bool GroupedMatmulQuantTiling::SetTilingData(gert::TilingContext* context)
{
    tilingData.set_CoreNum(_Params.CoreNum);
    tilingData.set_dataType(static_cast<uint32_t>(_Params.dataType));
    tilingData.set_UBSize(_Params.UBSize);
    tilingData.set_L1Size(_Params.L1Size);
    tilingData.set_L0ASize(_Params.L0ASize);
    tilingData.set_L0BSize(_Params.L0BSize);
    tilingData.set_L0CSize(_Params.L0CSize);

    tilingData.set_noGroup(_Params.noGroup);
    tilingData.set_originE(_Params.originE);
    tilingData.set_originM(_Params.originM);
    tilingData.set_originN(_Params.originN);
    tilingData.set_originK(_Params.originK);
    tilingData.set_scaleK(_Params.scaleK);
    tilingData.set_scaleGroupSize(_Params.scaleGroupSize);
    tilingData.set_fracN(_Params.fracN);
    tilingData.set_fracK(_Params.fracK);
    tilingData.set_splitK(static_cast<uint32_t>(_Params.splitK));

    tilingData.set_clearBaseN(_Params.clearBaseN);
    tilingData.set_clearOutLoop(_Params.clearOutLoop);
    tilingData.set_clearOutTailN(_Params.clearOutTailN);
    tilingData.set_clearOutTailCoreNum(_Params.clearOutTailCoreNum);

    tilingData.set_castBaseN(_Params.castBaseN);
    tilingData.set_castOutLoop(_Params.castOutLoop);
    tilingData.set_castOutTailN(_Params.castOutTailN);
    tilingData.set_castOutTailCoreNum(_Params.castOutTailCoreNum);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return true;
}

bool GroupedMatmulQuantTiling::SetLaunchInfo(gert::TilingContext* context)
{
    context->SetBlockDim(_Params.CoreNum);
    context->SetScheduleMode(1);
    tilingKey = GroupedMatmulQuantTilingKey::NORMAL;
    context->SetTilingKey(static_cast<uint64_t>(tilingKey));

    bool splitK = (_Params.splitK != 0);
    int64_t workspaceSize = SYS_WORKSPACE_910B + _Params.L0ASize * 8 * _Params.CoreNum +
                            (splitK ? (_Params.originM * _Params.originN * sizeof(float)) : 0);
    AddWorkspaceGMM(context, workspaceSize);
    return true;
}

bool GroupedMatmulQuantTiling::GetCheckAttr(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
    const int64_t* scale_group_size = attrs->GetAttrPointer<int64_t>(0);
    OP_TILING_CHECK(*scale_group_size <= 0 || *scale_group_size % 32 != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        context->GetNodeName(),
                        "GroupedMatmulQuant scale_group_size must be positive and aligned to 32, please check."),
                    return false);
    _Params.scaleGroupSize = *scale_group_size;
    return true;
}

ge::graphStatus GroupedMatmulQuantTiling::runTiling(gert::TilingContext* context)
{
    // 910b AscendC platformINFO
    OP_TILING_CHECK(!GetPlatformInfo(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get platforminfo fail."),
                    return ge::GRAPH_FAILED);
    // attr
    OP_TILING_CHECK(!GetCheckAttr(context), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check attr fail."),
                    return ge::GRAPH_FAILED);
    // shape
    OP_TILING_CHECK(!CheckInOutShapes(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check shape fail."),
                    return ge::GRAPH_FAILED);
    auto tempGetInputDesc = context->GetInputDesc(0);
    OP_TILING_CHECK(tempGetInputDesc == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "inputDesc is nullptr."),
                    return ge::GRAPH_FAILED);
    _Params.dataType = tempGetInputDesc->GetDataType();
    // calculate tilingdata
    OP_TILING_CHECK(!GetTilingData(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get tiling data fail."),
                    return ge::GRAPH_FAILED);
    // tilingdata
    OP_TILING_CHECK(!SetTilingData(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tiling data fail."),
                    return ge::GRAPH_FAILED);
    // launchinfo: tilingkey, workspace, blockdim
    OP_TILING_CHECK(!SetLaunchInfo(context),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set launchinfo fail."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingGroupedMatmulQuant(gert::TilingContext* context)
{
    GroupedMatmulQuantTiling tiling_handle;
    return tiling_handle.runTiling(context);
}

static ge::graphStatus TilingPrepareForGroupedMatmulQuant(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulQuant)
    .Tiling(TilingGroupedMatmulQuant)
    .TilingParse<GroupedMatmulQuantCompileInfo>(TilingPrepareForGroupedMatmulQuant);
} // namespace optiling
