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
 * \file grouped_matmul_finalize_routing_weight_quant_tiling.cpp
 * \brief
 */
#include "grouped_matmul_finalize_routing_weight_quant_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_key.h"
#include "op_host/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using namespace GMMFinalizeRoutingArch35Tiling;

namespace {
static constexpr const char *OP_NAME = "GroupedMatmulFinalizeRouting";
} // namespace

namespace optiling {
namespace GroupedMatmulFinalizeRoutingArch35WeightQuantTiling {
using namespace GroupedMatmulFinalizeRoutingArch35TilingConstant;
using namespace GmmConstant;

// Additional constants for MX-A8W4 validation
constexpr uint32_t DIM_NUM_BIAS = 2;
constexpr uint32_t DIM_NUM_LOGIT = 1;
constexpr uint32_t DIM_NUM_ROW_INDEX = 1;
constexpr uint32_t MX_BLOCK_SIZE = 32;
constexpr uint32_t MX_SCALE_BLOCK_SIZE = 64; // For scale/pertoken_scale K dimension
constexpr uint32_t ALIGN_SIZE_KN = 32;       // K/N alignment requirement for MX-A8W4 weight quant
constexpr int64_t MX_INNER_DIM = 2;
constexpr int64_t MIN_KN_SIZE = 32;       // Minimum K or N dimension size for MX-A8W4 weight quant
constexpr uint32_t DIM_NUM_WEIGHT_NZ = 5; // FRACTAL_NZ format: [E, K/32, N/16, 16, 32]
DataSize GetSizeByDataType(ge::DataType dType)
{
    if (dType == ge::DT_FLOAT4_E2M1 || dType == ge::DT_FLOAT4_E1M2 || dType == ge::DT_INT4) {
        return DataSize::B4_DATA_SIZE;
    }
    if (ge::GetSizeByDataType(dType) == static_cast<int>(DataSize::B8_DATA_SIZE)) {
        return DataSize::B8_DATA_SIZE;
    }
    return DataSize::RESERVED;
}

bool GMMFRWeightQuantTiling::IsCapable()
{
    // Currently there are no multiple templates, no template validity verification needed
    return true;
}

bool GMMFRWeightQuantTiling::CheckCoreNum() const
{
    OP_CHECK_IF(coreNum_ <= 0 || aivNum_ <= 0,
                OP_LOGE(context_->GetNodeName(), "Invalid aicNum[%ld] or aivNum[%ld], expect greater than 0", coreNum_,
                        aivNum_),
                return false);

    OP_CHECK_IF(coreNum_ * static_cast<int64_t>(AIC_AIV_CORE_RATIO) != aivNum_,
                OP_LOGE(context_->GetNodeName(),
                        "Invalid cube/vector core ratio. Expected cube:vector = 1:2, "
                        "but got cube=%ld, vector=%ld",
                        coreNum_, aivNum_),
                return false);
    return true;
}

ge::graphStatus GMMFRWeightQuantTiling::GetPlatformInfo()
{
    auto compileInfoPtr = context_->GetCompileInfo<GroupedMatmulFinalizeRoutingCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(OP_NAME, "CompileInfo is null"), return ge::GRAPH_FAILED);
    compileInfoPtr_ = compileInfoPtr;
    coreNum_ = static_cast<int64_t>(compileInfoPtr_->aicNum);
    aivNum_ = static_cast<int64_t>(compileInfoPtr_->aivNum);

    OP_LOGI(context_,
            "Compile info: aicNum(%lu) aivNum(%lu) ubSize(bytes)(%lu) l1Size(bytes)(%lu) l0aSize(bytes)(%lu) "
            "l0bSize(bytes)(%lu) l0cSize(bytes)(%lu).",
            compileInfoPtr_->aicNum, compileInfoPtr_->aivNum, compileInfoPtr_->ubSize, compileInfoPtr_->l1Size,
            compileInfoPtr_->l0ASize, compileInfoPtr_->l0BSize, compileInfoPtr_->l0CSize);

    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(context_->GetNodeName(), "Invalid core number ratio"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    OP_LOGD(context_->GetNodeName(), "Start inferring scenario and checking params.");

    OP_CHECK_IF(!InferScenario(), OP_LOGE(inputParams_.opName, "Failed to infer scenario."), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "Scenario inferred successfully, type=%d.", static_cast<int>(scenarioType_));

    OP_CHECK_IF(!RunCheckFunc(), OP_LOGE(inputParams_.opName, "Failed to check input params."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "All input params validated.");

    RunSetInputFunc();
    OP_LOGD(context_->GetNodeName(),
            "Input params set: mSize=%ld, kSize=%ld, nSize=%ld, "
            "groupNum=%ld, outputBS=%ld, hasBias=%d.",
            inputParams_.mSize, inputParams_.kSize, inputParams_.nSize, inputParams_.groupNum, inputParams_.outputBS,
            inputParams_.hasBias);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Start setting tiling data.");

    tilingData_.groupListType = inputParams_.groupListType;
    tilingData_.hasBias = inputParams_.hasBias;
    tilingData_.coreNum = compileInfoPtr_->aicNum;
    tilingData_.groupNum = inputParams_.groupNum;
    tilingData_.outputBs = inputParams_.outputBS;
    tilingData_.sharedInputOffset = inputParams_.shareInputOffset;
    tilingData_.sharedInputLen = inputParams_.sharedInputLen;
    tilingData_.sharedInputWeight = inputParams_.sharedInputWeight;
    tilingData_.groupSize = MX_BLOCK_SIZE;

    tilingData_.kSize = inputParams_.kSize;
    tilingData_.nSize = inputParams_.nSize;

    // Calculate initSize: initSize = (outputBS - sharedInputLen) * nSize
    // When sharedInputLen == 0, this simplifies to outputBS * nSize
    // When sharedInputLen > 0, we subtract the shared input region that doesn't need initialization
    tilingData_.initSize = (inputParams_.outputBS - inputParams_.sharedInputLen) * inputParams_.nSize;

    OP_LOGD(context_->GetNodeName(),
            "Tiling data set: coreNum=%u, groupNum=%ld, outputBs=%ld, "
            "kSize=%ld, nSize=%ld, initSize=%ld, hasBias=%d, groupListType=%d.",
            tilingData_.coreNum, tilingData_.groupNum, tilingData_.outputBs, tilingData_.kSize, tilingData_.nSize,
            tilingData_.initSize, tilingData_.hasBias, tilingData_.groupListType);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMFRWeightQuantTiling::DoLibApiTiling()
{
    // 当前为低阶api实现，无对应tiling
    return ge::GRAPH_SUCCESS;
}

uint64_t GMMFRWeightQuantTiling::GetTilingKey() const
{
    // 当前支持:
    // 0:ATRANS=0 ; 1:BTRANS=1
    return GET_TPL_TILING_KEY(0, 1, static_cast<bool>(inputParams_.hasBias));
}
// 6、计算Workspace 大小
ge::graphStatus GMMFRWeightQuantTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1); // get second variable
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is nullptr."),
                return ge::GRAPH_FAILED); // check workspaces is not null
    workspaces[0] = 16777216U;            // 16777216U: 16 * 1024 * 1024,default workspace size
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus GMMFRWeightQuantTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "Start posting tiling data, blockDim=%u, tilingSize=%zu.", tilingData_.coreNum,
            sizeof(tilingData_));

    context_->SetBlockDim(tilingData_.coreNum);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr, OP_LOGE(context_->GetNodeName(), "RawTilingData is nullptr."),
                return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));

    OP_LOGD(context_->GetNodeName(), "Tiling data posted successfully.");
    return ge::GRAPH_SUCCESS;
}

void GMMFRWeightQuantTiling::Reset()
{
    tilingData_ = GMMFinalizeRoutingWeightQuantTilingData();
    OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                         context_->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
}

bool GMMFRWeightQuantTiling::InferScenario()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input xDesc is nullptr."), return false);
    auto wDesc = context_->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input wDesc is nullptr."), return false);

    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    // GetPrimaryFormat returns the format value masked with 0xFF
    // FORMAT_FRACTAL_NZ = 29, FORMAT_FRACTAL_NZ_C0_32 = 51

    auto scaleDesc = context_->GetOptionalInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "Input scaleDesc is nullptr."), return false);
    auto scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    auto perTokenScaleDtype = pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : ge::DT_INT8;
    auto xDtype = xDesc->GetDataType();
    auto wDtype = wDesc->GetDataType();
    OP_LOGD(
        context_->GetNodeName(), "Current xDtype: %s, wDtype: %s, scaleDtype: %s, perTokenScaleDtype: %s, wFormat: %s",
        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(wDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(scaleDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(perTokenScaleDtype).c_str(),
        (wFormat == ge::FORMAT_FRACTAL_NZ || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) ? "FRACTAL_NZ" : "ND");
    if (xDtype == ge::DT_FLOAT8_E4M3FN && wDtype == ge::DT_FLOAT4_E2M1 &&
        (wFormat == ge::FORMAT_FRACTAL_NZ || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) &&
        scaleDtype == ge::DT_FLOAT8_E8M0 && perTokenScaleDtype == ge::DT_FLOAT8_E8M0) {
        scenarioType_ = ScenarioType::MX_A8W4_WEIGHT_NZ;
        OP_LOGD(context_->GetNodeName(), "Enable MX-A8W4-WEIGHT-NZ mode.");
        SetMxA8W4NzConditionFunc();
        SetMxA8W4NzInputFunc();
        return true;
    }

    std::string incorrectDtypes = ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                                  ge::TypeUtils::DataTypeToSerialString(wDtype) + ", " +
                                  ge::TypeUtils::DataTypeToSerialString(scaleDtype) + ", " +
                                  ge::TypeUtils::DataTypeToSerialString(perTokenScaleDtype);
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(OP_NAME, "x, w, scale, perTokenScale", incorrectDtypes,
                                           "Only support MX-A8W4-WEIGHT-NZ mode");
    return false;
}

static bool CheckMxA8W4NzInputPtr(gert::TilingContext *context)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context->GetNodeName(), "Input xDesc is nullptr."), return false);
    auto xStorageShape = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context->GetNodeName(), "Input xStorageShape is nullptr."),
                return false);

    auto wDesc = context->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context->GetNodeName(), "Input wDesc is nullptr."), return false);
    auto wStorageShape = context->GetInputShape(W_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context->GetNodeName(), "Input wStorageShape is nullptr."),
                return false);

    auto scaleDesc = context->GetOptionalInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context->GetNodeName(), "Input scaleDesc is nullptr."), return false);
    auto scaleStorageShape = context->GetOptionalInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr, OP_LOGE(context->GetNodeName(), "Input scaleStorageShape is nullptr."),
                return false);

    // pertoken_scale is optional - use GetOptionalInputDesc
    auto pertokenScaleDesc = context->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    OP_CHECK_IF(pertokenScaleDesc == nullptr, OP_LOGE(context->GetNodeName(), "Input pertokenScaleDesc is nullptr."),
                return false);
    auto pertokenScaleStorageShape = context->GetOptionalInputShape(PERTOKEN_SCALE_INDEX);
    OP_CHECK_IF(pertokenScaleStorageShape == nullptr,
                OP_LOGE(context->GetNodeName(), "Input pertokenScaleStorageShape is nullptr."), return false);

    // Add: group_list must not be nullptr (optional input - use GetOptional API)
    auto groupListDesc = context->GetOptionalInputDesc(GROUPLIST_INDEX);
    OP_CHECK_IF(groupListDesc == nullptr, OP_LOGE(context->GetNodeName(), "Input groupListDesc is nullptr."),
                return false);
    auto groupListStorageShape = context->GetOptionalInputShape(GROUPLIST_INDEX);
    OP_CHECK_IF(groupListStorageShape == nullptr,
                OP_LOGE(context->GetNodeName(), "Input groupListStorageShape is nullptr."), return false);

    // Add: row_index must not be nullptr (required for 91095 - use GetInput API)
    auto rowIndexDesc = context->GetOptionalInputDesc(ROW_INDEX_INDEX);
    OP_CHECK_IF(rowIndexDesc == nullptr, OP_LOGE(context->GetNodeName(), "Input rowIndexDesc is nullptr."),
                return false);
    auto rowIndexStorageShape = context->GetOptionalInputShape(ROW_INDEX_INDEX);
    OP_CHECK_IF(rowIndexStorageShape == nullptr,
                OP_LOGE(context->GetNodeName(), "Input rowIndexStorageShape is nullptr."), return false);

    return true;
}

static bool CheckMxA8W4NzAttrPtr(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "Attrs is nullptr"), return false);

    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X);
    OP_CHECK_IF((transposeXPtr != nullptr && (*transposeXPtr)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "transposeX", "true",
                                                      "The value of transposeX must be false or nullptr"),
                return false);

    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_W);
    OP_CHECK_IF(transposeWeightPtr == nullptr,
                OP_LOGE(context->GetNodeName(), "transposeW cannot be nullptr, but now is nullptr"), return false);
    if (unlikely(!(*transposeWeightPtr))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "transposeW", "false", "The value of transposeW must be true");
        return false;
    }

    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    OP_CHECK_IF(groupListTypePtr == nullptr, OP_LOGE(context->GetNodeName(), "groupListType cannot be nullptr"),
                return false);
    if (unlikely(*groupListTypePtr != 0 && *groupListTypePtr != 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupListType", std::to_string(*groupListTypePtr),
                                              "Attr groupListType must be 0 or 1");
        return false;
    }

    const int64_t *outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DTYPE);
    OP_CHECK_IF(outputDtypePtr == nullptr, OP_LOGE(context->GetNodeName(), "dtype cannot be nullptr"), return false);
    if (unlikely(*outputDtypePtr != 0)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "dtype", std::to_string(*outputDtypePtr),
                                              "Attr dtype must be 0(float32)");
        return false;
    }

    // offset must be nullptr in MxA8W4 weight Nz scenario
    auto offsetDesc = context->GetOptionalInputDesc(OFFSET_INDEX);
    if (unlikely(offsetDesc != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "offset", "not nullptr",
                                              "offset must be nullptr in MxA8W4 weight Nz scenario");
        return false;
    }

    return true;
}

// Helper: Validate x shape and extract M, K
static bool ValidateXShape(gert::TilingContext *context, int64_t &mSize, int64_t &kSize)
{
    auto xStorageShape = context->GetInputShape(X_INDEX);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    auto xDimNum = xShape.GetDimNum();
    if (unlikely(xDimNum != DIM_NUM_X)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "x", std::to_string(xDimNum),
                                                 "The shape dim of x must be " + std::to_string(DIM_NUM_X));
        return false;
    }
    mSize = xShape.GetDim(0);
    kSize = xShape.GetDim(1);
    return true;
}

// Helper: Validate w shape and extract E, N, K
static bool ValidateWShape(gert::TilingContext *context, ge::Format wFormat, int64_t &eFromW, int64_t &nSize,
                           int64_t &kFromW)
{
    auto wShape = context->GetInputShape(W_INDEX);
    const gert::Shape &wStorageShape = wShape->GetStorageShape();
    auto wStorageDimNum = wStorageShape.GetDimNum();

    if (wFormat == ge::FORMAT_FRACTAL_NZ || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) {
        if (unlikely(wStorageDimNum != DIM_NUM_WEIGHT_NZ)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "w", std::to_string(wStorageDimNum),
                ("The shape dim of w (FRACTAL_NZ) must be " + std::to_string(DIM_NUM_WEIGHT_NZ)));
            return false;
        }
        const gert::Shape &wOriginShape = wShape->GetOriginShape();
        nSize = wOriginShape.GetDim(1);
        kFromW = wOriginShape.GetDim(2); // w origin shape is [E, N, K], dim 2 is K axis
    } else {
        if (unlikely(wStorageDimNum != DIM_NUM_WEIGHT)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "w", std::to_string(wStorageDimNum),
                                                     ("The shape dim of w must be " + std::to_string(DIM_NUM_WEIGHT)));
            return false;
        }
        nSize = wStorageShape.GetDim(1);
        kFromW = wStorageShape.GetDim(2); // w storage shape is [E, N, K], dim 2 is K axis
    }
    eFromW = wStorageShape.GetDim(0);
    return true;
}

// Helper: Validate group_list shape matches E
static bool ValidateGroupListShape(gert::TilingContext *context, int64_t eFromW)
{
    auto groupListStorageShape = context->GetOptionalInputShape(GROUPLIST_INDEX);
    const gert::Shape &groupListShape = groupListStorageShape->GetOriginShape();
    if (unlikely(eFromW != groupListShape.GetDim(0))) {
        std::string incorrectValues =
            "weight E=" + std::to_string(eFromW) + ", groupList dim[0]=" + std::to_string(groupListShape.GetDim(0));
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "E of weight, groupList", incorrectValues,
                                              "E mismatch: weight E dimension must be equal to groupList dim[0]");
        return false;
    }
    return true;
}

// Helper: Validate row_index shape is [M]
static bool ValidateRowIndexShape(gert::TilingContext *context, int64_t mSize)
{
    auto rowIndexStorageShape = context->GetOptionalInputShape(ROW_INDEX_INDEX);
    const gert::Shape &rowIndexShape = rowIndexStorageShape->GetOriginShape();
    if (unlikely(rowIndexShape.GetDimNum() != DIM_NUM_ROW_INDEX)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "rowIndex", std::to_string(rowIndexShape.GetDimNum()),
            "The shape dim of rowIndex must be " + std::to_string(DIM_NUM_ROW_INDEX));
        return false;
    }
    if (unlikely(rowIndexShape.GetDim(0) != mSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "rowIndex", Ops::Base::ToString(rowIndexShape),
                                              "The axis 0 of rowIndex must be equal to M=" + std::to_string(mSize));
        return false;
    }
    return true;
}

// Helper: Validate logit shape is [M] if present
static bool ValidateLogitShape(gert::TilingContext *context, int64_t mSize)
{
    auto logitDesc = context->GetOptionalInputDesc(LOGIT_INDEX);
    if (logitDesc == nullptr)
        return false;

    auto logitStorageShape = context->GetOptionalInputShape(LOGIT_INDEX);
    const gert::Shape &logitShape = logitStorageShape->GetOriginShape();
    if (unlikely(logitShape.GetDimNum() != DIM_NUM_LOGIT)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "logit", std::to_string(logitShape.GetDimNum()),
                                                 "The shape dim of logit must be " + std::to_string(DIM_NUM_LOGIT));
        return false;
    }
    if (unlikely(logitShape.GetDim(0) != mSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "logit", Ops::Base::ToString(logitShape),
                                              "The axis 0 of logit must be equal to M=" + std::to_string(mSize));
        return false;
    }
    return true;
}

// Helper: Validate bias shape is [E, N] if present and non-empty
static bool ValidateBiasShape(gert::TilingContext *context, int64_t eFromW, int64_t nSize)
{
    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc == nullptr)
        return true;

    auto biasStorageShape = context->GetOptionalInputShape(BIAS_INDEX);
    if (biasStorageShape == nullptr)
        return true;

    const gert::Shape &biasShape = biasStorageShape->GetOriginShape();
    if (biasShape.GetDimNum() == 0)
        return true;

    if (unlikely(biasShape.GetDimNum() != DIM_NUM_BIAS)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "bias", std::to_string(biasShape.GetDimNum()),
                                                 "The shape dim of bias must be " + std::to_string(DIM_NUM_BIAS));
        return false;
    }
    if (unlikely(biasShape.GetDim(0) != eFromW)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "bias", Ops::Base::ToString(biasShape),
                                              "The axis 0 of bias must be equal to E=" + std::to_string(eFromW));
        return false;
    }
    if (unlikely(biasShape.GetDim(1) != nSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "bias", Ops::Base::ToString(biasShape),
                                              "The axis 1 of bias must be equal to N=" + std::to_string(nSize));
        return false;
    }
    return true;
}

// Axis indices for scale shape [E, N, CeilDiv(K, 64), MX_INNER_DIM]
constexpr uint32_t SCALE_AXIS_E = 0;
constexpr uint32_t SCALE_AXIS_N = 1;
constexpr uint32_t SCALE_AXIS_K_CEIL_DIV = 2;
constexpr uint32_t SCALE_AXIS_INNER = 3;

// Axis indices for pertoken_scale shape [M, CeilDiv(K, 64), MX_INNER_DIM]
constexpr uint32_t PTSCALE_AXIS_M = 0;
constexpr uint32_t PTSCALE_AXIS_K_CEIL_DIV = 1;
constexpr uint32_t PTSCALE_AXIS_INNER = 2;

// Helper: Validate scale shape [E, N, CeilDiv(K, 64), 2]
static bool ValidateScaleShape(gert::TilingContext *context, int64_t eFromW, int64_t nSize, int64_t kCeilDiv64)
{
    auto scaleStorageShape = context->GetOptionalInputShape(SCALE_INDEX);
    const gert::Shape &scaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = scaleShape.GetDimNum();
    if (unlikely(scaleDimNum != DIM_NUM_MX_SCALE)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "scale", std::to_string(scaleDimNum),
                                                 "The shape dim of scale must be " + std::to_string(DIM_NUM_MX_SCALE));
        return false;
    }
    if (unlikely(scaleShape.GetDim(SCALE_AXIS_E) != eFromW)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "scale", Ops::Base::ToString(scaleShape),
                                              "The axis 0 of scale must be equal to E=" + std::to_string(eFromW));
        return false;
    }
    if (unlikely(scaleShape.GetDim(SCALE_AXIS_N) != nSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "scale", Ops::Base::ToString(scaleShape),
                                              "The axis 1 of scale must be equal to N=" + std::to_string(nSize));
        return false;
    }
    if (unlikely(scaleShape.GetDim(SCALE_AXIS_K_CEIL_DIV) != kCeilDiv64)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "scale", Ops::Base::ToString(scaleShape),
            "The axis 2 of scale must be equal to CeilDiv(K, 64)=" + std::to_string(kCeilDiv64));
        return false;
    }
    if (unlikely(scaleShape.GetDim(SCALE_AXIS_INNER) != MX_INNER_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "scale", Ops::Base::ToString(scaleShape),
                                              "The axis 3 of scale must be equal to " + std::to_string(MX_INNER_DIM));
        return false;
    }
    return true;
}

// Helper: Validate pertoken_scale shape [M, CeilDiv(K, 64), 2]
static bool ValidatePertokenScaleShape(gert::TilingContext *context, int64_t mSize, int64_t kCeilDiv64)
{
    auto pertokenScaleStorageShape = context->GetOptionalInputShape(PERTOKEN_SCALE_INDEX);
    const gert::Shape &pertokenScaleShape = pertokenScaleStorageShape->GetOriginShape();
    auto pertokenScaleDimNum = pertokenScaleShape.GetDimNum();
    if (unlikely(pertokenScaleDimNum != DIM_NUM_MX_PERTOKENSCALE)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "pertokenScale", std::to_string(pertokenScaleDimNum),
            "The shape dim of pertokenScale must be " + std::to_string(DIM_NUM_MX_PERTOKENSCALE));
        return false;
    }
    if (unlikely(pertokenScaleShape.GetDim(PTSCALE_AXIS_M) != mSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "pertokenScale", Ops::Base::ToString(pertokenScaleShape),
            "The axis 0 of pertokenScale must be equal to M=" + std::to_string(mSize));
        return false;
    }
    if (unlikely(pertokenScaleShape.GetDim(PTSCALE_AXIS_K_CEIL_DIV) != kCeilDiv64)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "pertokenScale", Ops::Base::ToString(pertokenScaleShape),
            "The axis 1 of pertokenScale must be equal to CeilDiv(K, 64)=" + std::to_string(kCeilDiv64));
        return false;
    }
    if (unlikely(pertokenScaleShape.GetDim(PTSCALE_AXIS_INNER) != MX_INNER_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "pertokenScale", Ops::Base::ToString(pertokenScaleShape),
            "The axis 2 of pertokenScale must be equal to " + std::to_string(MX_INNER_DIM));
        return false;
    }
    return true;
}

static bool CheckKNAlignmentAndMinSize(int64_t kSize, int64_t nSize)
{
    if (unlikely(kSize % ALIGN_SIZE_KN != 0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", "K=" + std::to_string(kSize),
                                              "K of x must be aligned to " + std::to_string(ALIGN_SIZE_KN));
        return false;
    }
    if (unlikely(nSize % ALIGN_SIZE_KN != 0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight", "N=" + std::to_string(nSize),
                                              "N of weight must be aligned to " + std::to_string(ALIGN_SIZE_KN));
        return false;
    }
    if (unlikely(kSize < MIN_KN_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", "K=" + std::to_string(kSize),
                                              "K of x must be >= " + std::to_string(MIN_KN_SIZE));
        return false;
    }
    if (unlikely(nSize < MIN_KN_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight", "N=" + std::to_string(nSize),
                                              "N of weight must be >= " + std::to_string(MIN_KN_SIZE));
        return false;
    }
    return true;
}

static bool CheckMxA8W4InputShape(gert::TilingContext *context)
{
    // Get w format first
    auto wDesc = context->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context->GetNodeName(), "Input wDesc is nullptr."), return false);
    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));

    // 1. Validate x shape and get M, K
    int64_t mSize;
    int64_t kSize;
    OP_CHECK_IF(!ValidateXShape(context, mSize, kSize), OP_LOGE(context->GetNodeName(), "ValidateXShape failed."),
                return false);

    // 2. Validate w shape and get E, N, K
    int64_t eFromW;
    int64_t nSize;
    int64_t kFromW;
    OP_CHECK_IF(!ValidateWShape(context, wFormat, eFromW, nSize, kFromW),
                OP_LOGE(context->GetNodeName(), "ValidateWShape failed."), return false);

    // Validate K consistency between x and w
    if (unlikely(kSize != kFromW)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(OP_NAME, "x, weight",
                                               "x K=" + std::to_string(kSize) + ", weight K=" + std::to_string(kFromW),
                                               "K of x must be equal to K of weight");
        return false;
    }

    OP_CHECK_IF(!CheckKNAlignmentAndMinSize(kSize, nSize),
                OP_LOGE(context->GetNodeName(), "CheckKNAlignmentAndMinSize failed."), return false);

    // 3. Validate group_list shape
    OP_CHECK_IF(!ValidateGroupListShape(context, eFromW),
                OP_LOGE(context->GetNodeName(), "ValidateGroupListShape failed."), return false);

    // 4. Validate row_index shape
    OP_CHECK_IF(!ValidateRowIndexShape(context, mSize),
                OP_LOGE(context->GetNodeName(), "ValidateRowIndexShape failed."), return false);

    // 5. Validate logit shape
    OP_CHECK_IF(!ValidateLogitShape(context, mSize), OP_LOGE(context->GetNodeName(), "ValidateLogitShape failed."),
                return false);

    // 6. Validate bias shape
    OP_CHECK_IF(!ValidateBiasShape(context, eFromW, nSize),
                OP_LOGE(context->GetNodeName(), "ValidateBiasShape failed."), return false);

    // Calculate CeilDiv(K, 64) for scale validation
    int64_t kCeilDiv64 = (kSize + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE;

    // 7. Validate scale shape
    OP_CHECK_IF(!ValidateScaleShape(context, eFromW, nSize, kCeilDiv64),
                OP_LOGE(context->GetNodeName(), "ValidateScaleShape failed."), return false);

    // 8. Validate pertoken_scale shape
    OP_CHECK_IF(!ValidatePertokenScaleShape(context, mSize, kCeilDiv64),
                OP_LOGE(context->GetNodeName(), "ValidatePertokenScaleShape failed."), return false);

    return true;
}

static bool CheckMxA8W4AttrWithInput(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "Attrs is nullptr"), return false);

    const int64_t *outputBSPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_OUTPUT_BS);
    int64_t outputBS = outputBSPtr != nullptr ? *outputBSPtr : 0;
    if (unlikely(outputBS < 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "outputBS", std::to_string(outputBS),
                                              "Attr outputBS must be >=0");
        return false;
    }

    auto sharedInputDesc = context->GetOptionalInputDesc(SHARE_INPUT_INDEX);
    if (sharedInputDesc != nullptr) {
        auto sharedInputStorageShape = context->GetOptionalInputShape(SHARE_INPUT_INDEX);
        OP_CHECK_IF(sharedInputStorageShape == nullptr,
                    OP_LOGE(context->GetNodeName(), "Input sharedInputStorageShape cannot be nullptr."), return false);

        const float *shareInputWeightPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_SHARE_INPUT_WEIGHT);
        OP_CHECK_IF(shareInputWeightPtr == nullptr,
                    OP_LOGE(context->GetNodeName(), "Input shareInputWeightPtr cannot be nullptr."), return false);

        const int64_t *shareInputOffsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SHARE_INPUT_OFFSET);
        OP_CHECK_IF(shareInputOffsetPtr == nullptr,
                    OP_LOGE(context->GetNodeName(), "Input shareInputOffsetPtr cannot be nullptr."), return false);

        if (unlikely((*shareInputOffsetPtr) < 0)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "shareInputOffset", std::to_string(*shareInputOffsetPtr),
                                                  "The value of shareInputOffset must be >=0");
            return false;
        }
        if (unlikely((*shareInputOffsetPtr) + sharedInputStorageShape->GetOriginShape().GetDim(0) > outputBS)) {
            std::string incorrectValues = "shareInputOffset=" + std::to_string(*shareInputOffsetPtr) +
                                          ", sharedInputStorageShape.GetDim(0)=" +
                                          std::to_string(sharedInputStorageShape->GetOriginShape().GetDim(0)) +
                                          ", outputBS=" + std::to_string(outputBS);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "shareInputOffset", incorrectValues,
                "Attr shareInputOffset + sharedInputStorageShape.GetDim(0) must be <= outputBS");
            return false;
        }
    }

    return true;
}

bool GMMFRWeightQuantTiling::SetMxA8W4NzConditionFunc()
{
    checkConditionFuncs_.push_back(CheckMxA8W4NzInputPtr);
    checkConditionFuncs_.push_back(CheckMxA8W4NzAttrPtr);
    checkConditionFuncs_.push_back(CheckMxA8W4InputShape);
    checkConditionFuncs_.push_back(CheckMxA8W4AttrWithInput);
    return true;
}

bool GMMFRWeightQuantTiling::RunCheckFunc()
{
    for (size_t conditionIdx = 0; conditionIdx < checkConditionFuncs_.size(); conditionIdx++) {
        OP_CHECK_IF(!checkConditionFuncs_[conditionIdx](context_),
                    OP_LOGE(context_->GetNodeName(), "Failed to check input params in rule[%lu].", conditionIdx),
                    return false);
    }
    return true;
}

static bool SetMxA8W4NzAttrs(gert::TilingContext *context, GMMFRWeightQuantInputParams &inputParams)
{
    OP_LOGD(context->GetNodeName(), "Start extracting attributes.");

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "Attrs is nullptr"), return false);

    inputParams.xTrans = false;
    inputParams.wTrans = true;

    const float *shareInputWeightPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_SHARE_INPUT_WEIGHT);
    inputParams.sharedInputWeight = shareInputWeightPtr == nullptr ? 0.0f : *shareInputWeightPtr;

    const int64_t *shareInputOffsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SHARE_INPUT_OFFSET);
    inputParams.shareInputOffset = shareInputOffsetPtr == nullptr ? 0 : *shareInputOffsetPtr;

    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    inputParams.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : 0;

    const int64_t *outputBSPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_OUTPUT_BS);
    if (outputBSPtr != nullptr) {
        inputParams.outputBS = *outputBSPtr;
        OP_LOGD(context->GetNodeName(), "outputBS from attr = %ld.", inputParams.outputBS);
    } else {
        // Use M / E as default value when outputBs is not provided
        OP_CHECK_IF(inputParams.groupNum == 0,
                    OP_LOGE(context->GetNodeName(), "groupNum(E) is 0, cannot compute default outputBs as M/E."),
                    return false);
        inputParams.outputBS = inputParams.mSize / inputParams.groupNum;
        OP_LOGI(context->GetNodeName(), "outputBs not provided, using default M/E = %ld/%ld = %ld", inputParams.mSize,
                inputParams.groupNum, inputParams.outputBS);
    }

    const int64_t *outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DTYPE);
    inputParams.outputDtype = outputDtypePtr != nullptr ? *outputDtypePtr : 0;

    OP_LOGD(context->GetNodeName(),
            "Attributes extracted: sharedInputWeight=%f, "
            "shareInputOffset=%ld, groupListType=%ld, outputDtype=%ld.",
            inputParams.sharedInputWeight, inputParams.shareInputOffset, inputParams.groupListType,
            inputParams.outputDtype);
    return true;
}

// Helper: Set w format in input params
static bool SetWFormat(gert::TilingContext *context, GMMFRWeightQuantInputParams &inputParams)
{
    OP_LOGD(context->GetNodeName(), "Start extracting w format.");

    auto wDesc = context->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context->GetNodeName(), "Input wDesc is nullptr."), return false);
    inputParams.wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));

    OP_LOGD(context->GetNodeName(), "wFormat = %s.", Ops::Base::ToString(inputParams.wFormat).c_str());
    return true;
}

// Helper: Extract M and K from x shape
static bool ExtractXDimensions(gert::TilingContext *context, GMMFRWeightQuantInputParams &inputParams)
{
    OP_LOGD(context->GetNodeName(), "Start extracting M, K from x shape.");

    auto xStorageShape = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context->GetNodeName(), "Input xStorageShape is nullptr."),
                return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    uint64_t xDimNum = static_cast<uint64_t>(xShape.GetDimNum());
    inputParams.mSize = xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX);
    inputParams.kSize = xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX);

    OP_LOGD(context->GetNodeName(), "Extracted mSize = %ld, kSize = %ld.", inputParams.mSize, inputParams.kSize);
    return true;
}

// Helper: Extract E, N from w shape
static bool ExtractWDimensions(gert::TilingContext *context, GMMFRWeightQuantInputParams &inputParams)
{
    OP_LOGD(context->GetNodeName(), "Start extracting E, N from w shape.");

    auto wStorageShape = context->GetInputShape(W_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context->GetNodeName(), "Input wStorageShape is nullptr."),
                return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();

    if (inputParams.wFormat == ge::FORMAT_FRACTAL_NZ || inputParams.wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) {
        // For FRACTAL_NZ, get N from origin shape [E, N, K]
        inputParams.nSize = wShape.GetDim(1);
    } else {
        uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
        inputParams.nSize = wShape.GetDim(wDimNum - LAST_SECOND_DIM_INDEX);
    }
    inputParams.groupNum = wShape.GetDim(0);

    OP_LOGD(context->GetNodeName(), "Extracted groupNum(E) = %ld, nSize = %ld.", inputParams.groupNum,
            inputParams.nSize);
    return true;
}

// Helper: Set bias info
static void SetBiasInfo(gert::TilingContext *context, GMMFRWeightQuantInputParams &inputParams)
{
    OP_LOGD(context->GetNodeName(), "Start checking bias info.");

    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc == nullptr) {
        inputParams.hasBias = false;
        OP_LOGD(context->GetNodeName(), "No bias input.");
        return;
    }
    auto biasStorageShape = context->GetOptionalInputShape(BIAS_INDEX);
    if (biasStorageShape == nullptr) {
        inputParams.hasBias = false;
        OP_LOGD(context->GetNodeName(), "No bias shape.");
        return;
    }
    const gert::Shape &biasShape = biasStorageShape->GetOriginShape();
    inputParams.hasBias = (biasShape.GetDimNum() > 0);
    OP_LOGD(context->GetNodeName(), "hasBias = %s.", inputParams.hasBias ? "true" : "false");
}

// Helper: Set shared input info
static bool SetSharedInputInfo(gert::TilingContext *context, GMMFRWeightQuantInputParams &inputParams)
{
    OP_LOGD(context->GetNodeName(), "Start checking shared input info.");

    auto sharedInputDesc = context->GetOptionalInputDesc(SHARE_INPUT_INDEX);
    if (sharedInputDesc == nullptr) {
        inputParams.sharedInputLen = 0;
        inputParams.residualScale = 0.0f;
        OP_LOGD(context->GetNodeName(), "No shared input.");
        return true;
    }
    auto sharedInputStorageShape = context->GetOptionalInputShape(SHARE_INPUT_INDEX);
    OP_CHECK_IF(sharedInputStorageShape == nullptr,
                OP_LOGE(context->GetNodeName(), "Input sharedInputStorageShape is nullptr."), return false);
    inputParams.sharedInputLen = sharedInputStorageShape->GetOriginShape().GetDim(0);
    OP_LOGD(context->GetNodeName(), "sharedInputLen = %ld.", inputParams.sharedInputLen);
    return true;
}

static bool SetMxA8W4NzInput(gert::TilingContext *context, GMMFRWeightQuantInputParams &inputParams)
{
    OP_LOGD(context->GetNodeName(), "Start extracting all inputs.");

    OP_CHECK_IF(!SetWFormat(context, inputParams), OP_LOGE(context->GetNodeName(), "SetWFormat failed."), return false);
    OP_CHECK_IF(!ExtractXDimensions(context, inputParams),
                OP_LOGE(context->GetNodeName(), "ExtractXDimensions failed."), return false);
    OP_CHECK_IF(!ExtractWDimensions(context, inputParams),
                OP_LOGE(context->GetNodeName(), "ExtractWDimensions failed."), return false);
    SetBiasInfo(context, inputParams);
    OP_CHECK_IF(!SetSharedInputInfo(context, inputParams),
                OP_LOGE(context->GetNodeName(), "SetSharedInputInfo failed."), return false);

    OP_LOGD(context->GetNodeName(), "All inputs extracted successfully.");
    return true;
}

void GMMFRWeightQuantTiling::RunSetInputFunc()
{
    for (size_t setFuncIdx = 0; setFuncIdx < SetInputFuncs_.size(); setFuncIdx++) {
        SetInputFuncs_[setFuncIdx](context_, inputParams_);
    }
}

bool GMMFRWeightQuantTiling::SetMxA8W4NzInputFunc()
{
    OP_LOGD(context_->GetNodeName(), "Setting up input functions.");
    SetInputFuncs_.push_back(SetMxA8W4NzInput); // First: extract M and E from shapes
    SetInputFuncs_.push_back(SetMxA8W4NzAttrs); // Second: use M/E for default outputBs
    return true;
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulFinalizeRouting, GMMFRWeightQuantTiling,
                             GMMFR_WEIGHT_QUANT_TILING_VEC_ANTIQUANT);
} // namespace GroupedMatmulFinalizeRoutingArch35WeightQuantTiling
} // namespace optiling
