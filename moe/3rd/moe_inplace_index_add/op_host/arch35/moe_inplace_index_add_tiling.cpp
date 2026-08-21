/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_inplace_index_add_tiling.cpp
 * \brief
 */
#include "moe_inplace_index_add_tiling.h"
#include <math.h>
#include <string>
#include "moe_inplace_index_add_tiling_arch35.h"

using namespace ge;

namespace optiling {
static const int64_t SOC_VERSION_MULTI_ATOMIC_ADD = 1;
static const int64_t SUPPORT_ATOMIC_ADD = 1;
static const int64_t ASSIGNED_UB_SIZE = 2048;
static const int64_t ONE_BLOCK_SIZE = 32;
static const int64_t ALIGN_SIZE = 64;
static const int64_t INDEX_ATTR_AXIS = 0;
static const int64_t TENSOR_NUM = 2;

static int64_t GetCeilInt(int64_t value1, int64_t value2)
{
    MOE_OP_TILING_CHECK(
        value2 == 0,
        MOE_VECTOR_INNER_ERR_REPORT_TILIING("MoeInplaceIndexAdd", "In the GetCeilInt function, the divisor is 0."),
        return value1);
    return (value1 + value2 - 1) / value2;
}

static void PrintTilingParams(const gert::TilingContext *context, const MoeInplaceIndexAddTilingData *tiling_params)
{
    OP_LOGD(context->GetNodeName(), "block_num=%ld.", tiling_params->block_num);
    OP_LOGD(context->GetNodeName(), "indices_num=%ld.", tiling_params->indices_num);
    OP_LOGD(context->GetNodeName(), "outer_loop=%ld.", tiling_params->outer_loop);
    OP_LOGD(context->GetNodeName(), "full_num_per_block=%ld.", tiling_params->full_num_per_block);
    OP_LOGD(context->GetNodeName(), "tail_num=%ld.", tiling_params->tail_num);
    OP_LOGD(context->GetNodeName(), "axis_and_after_data_num_updates=%ld.",
            tiling_params->axis_and_after_data_num_updates);
    OP_LOGD(context->GetNodeName(), "axis_and_after_data_num_var=%ld.", tiling_params->axis_and_after_data_num_var);
    OP_LOGD(context->GetNodeName(), "update_data_num=%ld.", tiling_params->update_data_num);
    OP_LOGD(context->GetNodeName(), "axis=%ld.", tiling_params->axis);
    OP_LOGD(context->GetNodeName(), "updates_ub_size=%ld.", tiling_params->updates_ub_size);
    OP_LOGD(context->GetNodeName(), "indices_ub_size=%ld.", tiling_params->indices_ub_size);
    OP_LOGD(context->GetNodeName(), "tiling_core_num=%ld.", tiling_params->tiling_core_num);
    OP_LOGD(context->GetNodeName(), "var_shape_num=%ld.", tiling_params->var_shape_num);
    OP_LOGD(context->GetNodeName(), "updates_shape_num=%ld.", tiling_params->updates_shape_num);
}

static void CalRunningCaseGeneral(MoeInplaceIndexAddTilingData *params)
{
    params->block_num = 1;
    params->full_num_per_block = params->outer_loop;
}

static void CalRunningCaseBF16(MoeInplaceIndexAddTilingData *params)
{
    params->block_num = 1;
    params->full_num_per_block = params->indices_num;
}

static bool CalRunningCaseAtomicAdd(const MoeInplaceIndexAddCompileInfo *compile_info,
                                    MoeInplaceIndexAddTilingData *params)
{
    if (params->indices_num < compile_info->core_num) {
        params->block_num = 1;
        params->full_num_per_block = params->indices_num;
        params->tail_num = params->full_num_per_block;
    } else {
        params->full_num_per_block = GetCeilInt(params->indices_num, compile_info->core_num);
        params->block_num = GetCeilInt(params->indices_num, params->full_num_per_block);
        params->tail_num = params->full_num_per_block;
        MOE_OP_TILING_CHECK(params->block_num == 0,
                            MOE_VECTOR_INNER_ERR_REPORT_TILIING("MoeInplaceIndexAdd", "block num cannot be 0."),
                            return false);
        if (params->indices_num % params->block_num != 0) {
            params->tail_num = params->indices_num % params->full_num_per_block;
        }
    }
    return true;
}

static bool InitRunningInfo(MoeInplaceIndexAddTilingData *params, const gert::Shape &indices_shape_val,
                            int32_t axis_val)
{
    if (params == nullptr) {
        return false;
    }
    params->block_num = 0;
    params->indices_num = indices_shape_val.GetShapeSize();
    MOE_OP_TILING_CHECK(params->indices_num == 0,
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING("MoeInplaceIndexAdd", "indices num cannot be 0."),
                        return false);
    params->outer_loop = 1;
    params->full_num_per_block = 0;
    params->tail_num = 0;
    params->axis_and_after_data_num_updates = 1;
    params->axis_and_after_data_num_var = 1;
    params->update_data_num = 1;
    params->axis = axis_val;
    params->updates_ub_size = ASSIGNED_UB_SIZE;
    params->indices_ub_size = ASSIGNED_UB_SIZE;
    params->tiling_core_num = 0;
    params->var_shape_num = 0;
    params->updates_shape_num = 0;
    return true;
}

static bool CalRunningInfo(const MoeInplaceIndexAddCompileInfo *compile_info, MoeInplaceIndexAddTilingData *params,
                           const gert::Shape &var_shape_val, const gert::Shape &updates_shape_val, int64_t axis_val,
                           ge::DataType dtype)
{
    if (compile_info == nullptr || params == nullptr) {
        return false;
    }
    for (int64_t i = 0; i < axis_val; i++) {
        params->outer_loop = params->outer_loop * var_shape_val.GetDim(i);
    }

    if (axis_val == 0) {
        params->outer_loop = params->outer_loop * var_shape_val.GetDim(0);
    }

    const int64_t updates_dims = updates_shape_val.GetDimNum();
    for (int64_t i = axis_val; i < updates_dims; i++) {
        params->axis_and_after_data_num_updates = params->axis_and_after_data_num_updates * updates_shape_val.GetDim(i);
    }

    const int64_t var_dims = var_shape_val.GetDimNum();
    for (int64_t i = axis_val; i < var_dims; i++) {
        params->axis_and_after_data_num_var = params->axis_and_after_data_num_var * var_shape_val.GetDim(i);
    }
    for (int64_t i = axis_val + 1; i < var_dims; i++) {
        params->update_data_num = params->update_data_num * var_shape_val.GetDim(i);
    }

    params->tiling_core_num = compile_info->core_num;
    params->var_shape_num = static_cast<int64_t>(var_shape_val.GetShapeSize());
    params->updates_shape_num = static_cast<int64_t>(updates_shape_val.GetShapeSize());

    int64_t soc_version = compile_info->soc_version;
    int64_t atomic_add = compile_info->atomic_add;
    if (dtype == DT_BF16) {
        CalRunningCaseBF16(params);
    } else if ((soc_version == SOC_VERSION_MULTI_ATOMIC_ADD) && (atomic_add == SUPPORT_ATOMIC_ADD)) {
        MOE_OP_TILING_CHECK(
            !CalRunningCaseAtomicAdd(compile_info, params),
            MOE_VECTOR_INNER_ERR_REPORT_TILIING("MoeInplaceIndexAdd", "CalRunningCaseAtomicAdd failed."), return false);
        params->updates_ub_size =
            (compile_info->ub_size - params->indices_ub_size * static_cast<int64_t>(sizeof(int64_t))) /
            (TENSOR_NUM * static_cast<int64_t>(sizeof(float)));
        params->updates_ub_size = GetCeilInt(params->updates_ub_size, ALIGN_SIZE) * ALIGN_SIZE;
    } else {
        CalRunningCaseGeneral(params);
    }
    return true;
}

ge::graphStatus Tiling4MoeInplaceIndexAdd(gert::TilingContext *context)
{
    auto compile_info = reinterpret_cast<const MoeInplaceIndexAddCompileInfo *>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    if (compile_info->isAscendc) {
        return MoeInplaceIndexAddTilingForAscendC(context);
    }

    auto params = context->GetTilingData<MoeInplaceIndexAddTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, params);

    auto var_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_shape);
    auto &var_shape_val = Ops::Transformer::OpTiling::EnsureNotScalar(var_shape->GetStorageShape());

    auto indices_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
    auto &indices_shape_val = Ops::Transformer::OpTiling::EnsureNotScalar(indices_shape->GetStorageShape());

    auto updates_shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, updates_shape);
    auto &updates_shape_val = Ops::Transformer::OpTiling::EnsureNotScalar(updates_shape->GetStorageShape());

    auto alpha_shape = context->GetInputShape(3);
    if (alpha_shape != nullptr) {
        auto &alpha_shape_val = Ops::Transformer::OpTiling::EnsureNotScalar(alpha_shape->GetStorageShape());
        MOE_OP_TILING_CHECK(
            !(alpha_shape_val.GetDimNum() == 1 && alpha_shape_val.GetDim(0) == 1) || alpha_shape_val.IsScalar(),
            MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "shape of alpha must be (1, )"),
            return ge::GRAPH_FAILED);
    }

    auto attr_ptr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_ptr);
    const int64_t *axis = attr_ptr->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    const int64_t updates_dims = updates_shape_val.GetDimNum();
    int64_t axis_val = static_cast<int64_t>(*axis);

    if (axis_val < 0) {
        axis_val = axis_val + updates_dims;
    }

    auto var_desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_desc);
    ge::DataType dtype = var_desc->GetDataType();

    MOE_OP_TILING_CHECK(!InitRunningInfo(params, indices_shape_val, axis_val),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "InitRunningInfo failed."),
                        return ge::GRAPH_FAILED);
    MOE_OP_TILING_CHECK(!CalRunningInfo(compile_info, params, var_shape_val, updates_shape_val, axis_val, dtype),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "CalRunningInfo failed."),
                        return ge::GRAPH_FAILED);
    PrintTilingParams(context, params);

    size_t *workspaces = context->GetWorkspaceSizes(2);
    MOE_OP_TILING_CHECK(workspaces == nullptr,
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "workspaces is null."),
                        return ge::GRAPH_FAILED);
    if (dtype == ge::DT_BF16) {
        workspaces[0] = static_cast<size_t>(params->var_shape_num * sizeof(float));
        workspaces[1] = static_cast<size_t>(params->updates_shape_num * sizeof(float));
    } else {
        workspaces[0] = static_cast<size_t>(ONE_BLOCK_SIZE);
        workspaces[1] = static_cast<size_t>(ONE_BLOCK_SIZE);
    }

    context->SetBlockDim(params->block_num);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareMoeInplaceIndexAddForAscendC(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareMoeInplaceIndexAddForAscendC entering.");
    // auto compileInfo = GetCompileInfoPtr<MoeInplaceIndexAddCompileInfo>(context);
    auto compileInfo = context->GetCompiledInfo<MoeInplaceIndexAddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    MOE_OP_TILING_CHECK((compileInfo->coreNum <= 0),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get core num."),
                        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    MOE_OP_TILING_CHECK((compileInfo->ubSize <= 0),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get ub size."),
                        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4MoeInplaceIndexAdd(gert::TilingParseContext *context)
{
    // auto compile_info = GetCompileInfoPtr<MoeInplaceIndexAddCompileInfo>(context);
    auto compile_info = context->GetCompiledInfo<MoeInplaceIndexAddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    compile_info->isAscendc = Ops::Transformer::OpTiling::IsRegbaseSocVersion(context);
    if (compile_info->isAscendc) {
        return TilingPrepareMoeInplaceIndexAddForAscendC(context);
    }
    std::unique_ptr<nlohmann::json> parsed_object_cinfo = MoeGetCompileInfoJson(context);
    MOE_OP_TILING_CHECK(
        compile_info == nullptr || parsed_object_cinfo == nullptr,
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "compile_info or json_str nullptr!"),
        return ge::GRAPH_FAILED);
    const nlohmann::json &vars = (*parsed_object_cinfo)["vars"];
    MOE_OP_TILING_CHECK(vars.empty(), MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get vars failed."),
                        return ge::GRAPH_FAILED);
    uint32_t core_num = 0;
    MOE_OP_TILING_CHECK(!MoeGetTilingCoreNum(context, core_num),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get tiling core num failed."),
                        return ge::GRAPH_FAILED);
    compile_info->core_num = static_cast<int64_t>(core_num);
    MOE_OP_TILING_CHECK(
        !optiling::MoeReadCompileItem(vars, "ub_size", compile_info->ub_size),
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GetCompileParams, get ub_size failed."),
        return ge::GRAPH_FAILED);
    MOE_OP_TILING_CHECK(
        !optiling::MoeReadCompileItem(vars, "var_size", compile_info->var_size),
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GetCompileParams, get var_size failed."),
        return ge::GRAPH_FAILED);
    MOE_OP_TILING_CHECK(!optiling::MoeReadCompileItem(vars, "var_data_each_block", compile_info->var_data_each_block),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                            "GetCompileParams, get var_data_each_block failed."),
                        return ge::GRAPH_FAILED);
    MOE_OP_TILING_CHECK(
        !optiling::MoeReadCompileItem(vars, "indices_size", compile_info->indices_size),
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GetCompileParams, get indices_size failed."),
        return ge::GRAPH_FAILED);
    MOE_OP_TILING_CHECK(
        !optiling::MoeReadCompileItem(vars, "indices_data_each_block", compile_info->indices_data_each_block),
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                            "GetCompileParams, get indices_data_each_block failed."),
        return ge::GRAPH_FAILED);
    MOE_OP_TILING_CHECK(
        !optiling::MoeReadCompileItem(vars, "soc_version", compile_info->soc_version),
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GetCompileParams, get soc_version failed."),
        return ge::GRAPH_FAILED);
    MOE_OP_TILING_CHECK(
        !optiling::MoeReadCompileItem(vars, "atomic_add", compile_info->atomic_add),
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GetCompileParams, get atomic_add failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the MoeInplaceIndexAdd op.
IMPL_OP_OPTILING(MoeInplaceIndexAdd)
    .Tiling(Tiling4MoeInplaceIndexAdd)
    .TilingParse<MoeInplaceIndexAddCompileInfo>(TilingPrepare4MoeInplaceIndexAdd);
} // namespace optiling
