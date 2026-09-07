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
 * \file moe_update_expert_tiling.cpp
 * \brief
 */

#include <string>
#include <numeric>
#include <climits>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/moe_update_expert_tiling.h"
#include "../../op_kernel/moe_update_expert_tiling_key.h"

using namespace ge;
using namespace AscendC;
using namespace MoeUpdateExpertNamespace;
using namespace Mc2Tiling;

namespace optiling {
constexpr uint32_t EXPERT_IDS_INDEX = 0U;
constexpr uint32_t EPLB_TABLE_INDEX = 1U;
constexpr uint32_t EXPERT_SCALES_INDEX = 2U;
constexpr uint32_t PRUNING_THRESHOLD_INDEX = 3U;
constexpr uint32_t ACTIVE_MASK_INDEX = 4U;
constexpr uint32_t OUTPUT_BALANCED_EXPERT_IDS = 0U;
constexpr uint32_t OUTPUT_ACTIVE_MASK_IDS = 1U;

constexpr uint32_t ATTR_LOCAL_RANK_ID_INDEX = 0U;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_BALANCE_MODE_INDEX = 2U;

constexpr uint32_t NUM_ONE = 1U;
constexpr uint32_t NUM_TWO = 2U;
constexpr int32_t K_MAX = 16;
constexpr int32_t BS_MAX = 512;
constexpr int32_t MAX_MOE_EXPERT_NUM = 1024;
constexpr int64_t MAX_WORLD_SIZE = 768LL;
constexpr int64_t MIN_WORLD_SIZE = 2LL;

constexpr uint32_t TAILOR_NONE = 0x0U;
constexpr uint32_t TAILOR_EXPERT_SCALES = 0x04U;
constexpr uint32_t TAILOR_PRUNING_THRESHOLD = 0x08U;
constexpr uint32_t TAILOR_PRUNING_THRESHOLD_EXPERT_SCALES = 0x0CU;
constexpr uint32_t TAILOR_ACTIVE_MASK = 0x10U;
constexpr uint32_t TAILOR_ACTIVE_MASK_EXPERT_SCALES = 0x14U;
constexpr uint32_t TAILOR_ACTIVE_MASK_PRUNING_THRESHOLD = 0x18U;
constexpr uint32_t TAILOR_TAILOR_ACTIVE_MASK_PRUNING_THRESHOLD_EXPERT_SCALES = 0x1CU;
constexpr uint64_t KEY_SCALES_FP32 = 0ULL;
constexpr uint64_t KEY_SCALES_FP16 = 10ULL;
constexpr uint64_t KEY_SCALES_BF16 = 20ULL;

const char *MOE_UPDATE_EXPERT_DEBUG = "MoeUpdateExpert Tiling";

class MoeUpdateExpertTiling {
public:
    MoeUpdateExpertTilingData *tilingData;

    ge::graphStatus Init(gert::TilingContext *context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext *context);

protected:
    ge::graphStatus CheckAttrs(const gert::TilingContext *context) const;
    ge::graphStatus CheckInputDataType(const gert::TilingContext *context) const;
    ge::graphStatus CheckOptionalInputDataType(const gert::TilingContext *context);
    ge::graphStatus CheckOutputDataType(const gert::TilingContext *context) const;
    ge::graphStatus CheckDataType(const gert::TilingContext *context);
    ge::graphStatus CheckFormat(const gert::TilingContext *context) const;
    ge::graphStatus CheckInputShape(const gert::TilingContext *context) const;
    ge::graphStatus CheckExpertScalesShape(const gert::TilingContext *context);
    ge::graphStatus CheckPruningThresholdShape(const gert::TilingContext *context);
    ge::graphStatus CheckActiveMaskShape(const gert::TilingContext *context);
    ge::graphStatus CheckOptionalInputShape(const gert::TilingContext *context);
    ge::graphStatus CheckOutputShape(const gert::TilingContext *context) const;
    ge::graphStatus CheckShape(const gert::TilingContext *context);
    uint64_t GetTilingKey() const;

private:
    uint32_t libApiWorkSpaceSize_{0U};
    uint32_t tailorCfg_{0U};
    uint64_t keyScales_{0ULL};
};

ge::graphStatus MoeUpdateExpertTiling::CheckAttrs(const gert::TilingContext *context) const
{
    int64_t localRankId = -1LL;
    int64_t worldSize = -1LL;

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "attrs"),
                    return ge::GRAPH_FAILED);

    auto balanceModePtr = attrs->GetAttrPointer<int64_t>(ATTR_BALANCE_MODE_INDEX);
    if (balanceModePtr == nullptr) {
        tilingData->balanceMode = 0;
    } else {
        OP_TILING_CHECK(((*balanceModePtr < 0) || (*balanceModePtr > 1)),
                        OP_LOGE_WITH_INVALID_ATTR(MOE_UPDATE_EXPERT_DEBUG, "balanceMode",
                                                  std::to_string(*balanceModePtr).c_str(), "[0, 1]"),
                        return ge::GRAPH_FAILED);
        tilingData->balanceMode = *balanceModePtr;
    }

    if (tilingData->balanceMode == 0) {
        auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
        OP_TILING_CHECK(worldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "worldSize"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (*worldSizePtr < MIN_WORLD_SIZE) || (*worldSizePtr > MAX_WORLD_SIZE),
            OP_LOGE_WITH_INVALID_ATTR(
                MOE_UPDATE_EXPERT_DEBUG, "worldSize", std::to_string(*worldSizePtr).c_str(),
                (std::string("[") + std::to_string(MIN_WORLD_SIZE) + ", " + std::to_string(MAX_WORLD_SIZE) + "]")
                    .c_str()),
            return ge::GRAPH_FAILED);
        worldSize = *worldSizePtr;

        auto localRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_ID_INDEX);
        OP_TILING_CHECK(localRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "localRankId"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (*localRankIdPtr < 0) || (*localRankIdPtr >= worldSize),
            OP_LOGE_WITH_INVALID_ATTR(MOE_UPDATE_EXPERT_DEBUG, "localRankId", std::to_string(*localRankIdPtr).c_str(),
                                      (std::string("[0, ") + std::to_string(worldSize) + ")").c_str()),
            return ge::GRAPH_FAILED);
        localRankId = *localRankIdPtr;
    }

    tilingData->localRankId = localRankId;
    tilingData->worldSize = worldSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckInputDataType(const gert::TilingContext *context) const
{
    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "expertIds"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((expertIdsDesc->GetDataType() != ge::DT_INT32) && (expertIdsDesc->GetDataType() != ge::DT_INT64)),
        OP_LOGE_FOR_INVALID_DTYPE(MOE_UPDATE_EXPERT_DEBUG, "expertIds",
                                  Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str(), "int32 or int64"),
        return ge::GRAPH_FAILED);

    auto eplbTableDesc = context->GetInputDesc(EPLB_TABLE_INDEX);
    OP_TILING_CHECK(eplbTableDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "eplbTable"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(eplbTableDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(MOE_UPDATE_EXPERT_DEBUG, "eplbTable",
                                              Ops::Base::ToString(eplbTableDesc->GetDataType()).c_str(), "int32"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckOptionalInputDataType(const gert::TilingContext *context)
{
    auto expertScalesDesc = context->GetOptionalInputDesc(EXPERT_SCALES_INDEX);
    if (expertScalesDesc != nullptr) {
        auto dtScales = expertScalesDesc->GetDataType();
        if (dtScales == ge::DT_FLOAT) {
            keyScales_ = KEY_SCALES_FP32;
        } else if (dtScales == ge::DT_FLOAT16) {
            keyScales_ = KEY_SCALES_FP16;
        } else if (dtScales == ge::DT_BF16) {
            keyScales_ = KEY_SCALES_BF16;
        } else {
            OP_LOGE_FOR_INVALID_DTYPE(MOE_UPDATE_EXPERT_DEBUG, "expertScales", Ops::Base::ToString(dtScales).c_str(),
                                      "fp16/bf16/float");
            return ge::GRAPH_FAILED;
        }
    }

    auto pruningThresholdDesc = context->GetOptionalInputDesc(PRUNING_THRESHOLD_INDEX);
    if (pruningThresholdDesc != nullptr) {
        OP_TILING_CHECK(
            (pruningThresholdDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE(MOE_UPDATE_EXPERT_DEBUG, "pruningThreshold",
                                      Ops::Base::ToString(pruningThresholdDesc->GetDataType()).c_str(), "float"),
            return ge::GRAPH_FAILED);
    }

    auto activeMaskDesc = context->GetOptionalInputDesc(ACTIVE_MASK_INDEX);
    if (activeMaskDesc != nullptr) {
        OP_TILING_CHECK((activeMaskDesc->GetDataType() != ge::DT_BOOL),
                        OP_LOGE_FOR_INVALID_DTYPE(MOE_UPDATE_EXPERT_DEBUG, "activeMask",
                                                  Ops::Base::ToString(activeMaskDesc->GetDataType()).c_str(), "bool"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckOutputDataType(const gert::TilingContext *context) const
{
    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "expertIds"),
                    return ge::GRAPH_FAILED);

    auto balancedExpertIdsDesc = context->GetOutputDesc(OUTPUT_BALANCED_EXPERT_IDS);
    OP_TILING_CHECK(balancedExpertIdsDesc == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "balancedExpertIds"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        balancedExpertIdsDesc->GetDataType() != expertIdsDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(MOE_UPDATE_EXPERT_DEBUG, "balancedExpertIds",
                                              Ops::Base::ToString(balancedExpertIdsDesc->GetDataType()).c_str(),
                                              "The dtype of balancedExpertIds must be the same as that of expertIds"),
        return ge::GRAPH_FAILED);

    auto balancedActiveMaskDesc = context->GetOutputDesc(OUTPUT_ACTIVE_MASK_IDS);
    OP_TILING_CHECK(balancedActiveMaskDesc == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "balancedActiveMask"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        balancedActiveMaskDesc->GetDataType() != ge::DT_BOOL,
        OP_LOGE_FOR_INVALID_DTYPE(MOE_UPDATE_EXPERT_DEBUG, "balancedActiveMask",
                                  Ops::Base::ToString(balancedActiveMaskDesc->GetDataType()).c_str(), "bool"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckDataType(const gert::TilingContext *context)
{
    OP_TILING_CHECK(CheckInputDataType(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check input data type failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOptionalInputDataType(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check optional input data type failed!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDataType(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check input data type failed!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckFormat(const gert::TilingContext *context) const
{
    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "expertIds"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            MOE_UPDATE_EXPERT_DEBUG, "expertIds",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdsDesc->GetStorageFormat())))
                .c_str(),
            "FRACTAL_NZ"),
        return ge::GRAPH_FAILED);

    auto eplbTableDesc = context->GetInputDesc(EPLB_TABLE_INDEX);
    OP_TILING_CHECK(eplbTableDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "eplbTable"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(eplbTableDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            MOE_UPDATE_EXPERT_DEBUG, "eplbTable",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(eplbTableDesc->GetStorageFormat())))
                .c_str(),
            "FRACTAL_NZ"),
        return ge::GRAPH_FAILED);

    auto expertScalesDesc = context->GetOptionalInputDesc(EXPERT_SCALES_INDEX);
    if (expertScalesDesc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())) ==
                ge::FORMAT_FRACTAL_NZ,
            OP_LOGE_FOR_INVALID_FORMAT(
                MOE_UPDATE_EXPERT_DEBUG, "expertScales",
                Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())))
                    .c_str(),
                "FRACTAL_NZ"),
            return ge::GRAPH_FAILED);
    }

    auto pruningThresholdDesc = context->GetOptionalInputDesc(PRUNING_THRESHOLD_INDEX);
    if (pruningThresholdDesc != nullptr) {
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(pruningThresholdDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(MOE_UPDATE_EXPERT_DEBUG, "pruningThreshold",
                                                   Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(
                                                                           pruningThresholdDesc->GetStorageFormat())))
                                                       .c_str(),
                                                   "FRACTAL_NZ"),
                        return ge::GRAPH_FAILED);
    }

    auto activeMaskDesc = context->GetOptionalInputDesc(ACTIVE_MASK_INDEX);
    if (activeMaskDesc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(activeMaskDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE_FOR_INVALID_FORMAT(
                MOE_UPDATE_EXPERT_DEBUG, "activeMask",
                Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(activeMaskDesc->GetStorageFormat())))
                    .c_str(),
                "FRACTAL_NZ"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckInputShape(const gert::TilingContext *context) const
{
    int64_t worldSize = static_cast<int64_t>(MAX_WORLD_SIZE);
    if (tilingData->worldSize != -1LL) {
        worldSize = tilingData->worldSize;
    }

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "expertIds"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (expertIdsStorageShape->GetStorageShape().GetDimNum() != NUM_TWO),
        OP_LOGE_FOR_INVALID_SHAPEDIM(MOE_UPDATE_EXPERT_DEBUG, "expert_ids",
                                     std::to_string(expertIdsStorageShape->GetStorageShape().GetDimNum()).c_str(), "2"),
        return ge::GRAPH_FAILED);
    tilingData->bs = expertIdsStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(((tilingData->bs <= 0) || (tilingData->bs > BS_MAX)),
                    OP_LOGE_FOR_INVALID_VALUE(MOE_UPDATE_EXPERT_DEBUG, "expert_ids",
                                              (std::string("dim0=") + std::to_string(tilingData->bs)).c_str(),
                                              (std::string("(0, ") + std::to_string(BS_MAX) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    tilingData->k = expertIdsStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(((tilingData->k <= 0) || (tilingData->k > K_MAX)),
                    OP_LOGE_FOR_INVALID_VALUE(MOE_UPDATE_EXPERT_DEBUG, "expert_ids",
                                              (std::string("dim1=") + std::to_string(tilingData->k)).c_str(),
                                              (std::string("(0, ") + std::to_string(K_MAX) + "]").c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *eplbTableStorageShape = context->GetInputShape(EPLB_TABLE_INDEX);
    OP_TILING_CHECK(eplbTableStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "eplbTable"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (eplbTableStorageShape->GetStorageShape().GetDimNum() != NUM_TWO),
        OP_LOGE_FOR_INVALID_SHAPEDIM(MOE_UPDATE_EXPERT_DEBUG, "eplb_table",
                                     std::to_string(eplbTableStorageShape->GetStorageShape().GetDimNum()).c_str(), "2"),
        return ge::GRAPH_FAILED);
    tilingData->moeExpertNum = eplbTableStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(((tilingData->moeExpertNum <= 0) || (tilingData->moeExpertNum > MAX_MOE_EXPERT_NUM) ||
                     (tilingData->moeExpertNum < tilingData->k)),
                    OP_LOGE_FOR_INVALID_VALUE(MOE_UPDATE_EXPERT_DEBUG, "eplb_table",
                                              (std::string("dim0=") + std::to_string(tilingData->moeExpertNum)).c_str(),
                                              (std::string("(0, ") + std::to_string(MAX_MOE_EXPERT_NUM) +
                                               "] and >= " + std::to_string(tilingData->k))
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    tilingData->f = eplbTableStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(((tilingData->f <= 1) || (static_cast<int64_t>(tilingData->f) > worldSize + 1)),
                    OP_LOGE_FOR_INVALID_VALUE(MOE_UPDATE_EXPERT_DEBUG, "eplb_table",
                                              (std::string("dim1=") + std::to_string(tilingData->f)).c_str(),
                                              (std::string("(1, ") + std::to_string(worldSize + 1) + "]").c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckExpertScalesShape(const gert::TilingContext *context)
{
    const gert::StorageShape *expertScalesShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);

    if (expertScalesShape != nullptr) {
        OP_TILING_CHECK(
            (expertScalesShape->GetStorageShape().GetDimNum() != NUM_TWO),
            OP_LOGE_FOR_INVALID_SHAPEDIM(MOE_UPDATE_EXPERT_DEBUG, "expert_scales",
                                         std::to_string(expertScalesShape->GetStorageShape().GetDimNum()).c_str(), "2"),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(((expertScalesShape->GetStorageShape().GetDim(0) != tilingData->bs) ||
                         (expertScalesShape->GetStorageShape().GetDim(1) != tilingData->k)),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            MOE_UPDATE_EXPERT_DEBUG, "expert_scales",
                            (std::string("[") + std::to_string(expertScalesShape->GetStorageShape().GetDim(0)) + ", " +
                             std::to_string(expertScalesShape->GetStorageShape().GetDim(1)) + "]")
                                .c_str(),
                            "The shape of expert_scales must be [bs, k]"),
                        return ge::GRAPH_FAILED);
        tailorCfg_ += (1U << EXPERT_SCALES_INDEX);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckPruningThresholdShape(const gert::TilingContext *context)
{
    const gert::StorageShape *pruningThresholdShape = context->GetOptionalInputShape(PRUNING_THRESHOLD_INDEX);

    if (pruningThresholdShape != nullptr) {
        if (pruningThresholdShape->GetStorageShape().GetDimNum() == NUM_ONE) {
            OP_TILING_CHECK(
                (pruningThresholdShape->GetStorageShape().GetDim(0) != tilingData->k),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    MOE_UPDATE_EXPERT_DEBUG, "pruning_threshold",
                    (std::string("dim0=") + std::to_string(pruningThresholdShape->GetStorageShape().GetDim(0))).c_str(),
                    "Dim0 of pruning_threshold must be equal to k"),
                return ge::GRAPH_FAILED);
        } else if (pruningThresholdShape->GetStorageShape().GetDimNum() == NUM_TWO) {
            OP_TILING_CHECK(((pruningThresholdShape->GetStorageShape().GetDim(0) != 1) ||
                             (pruningThresholdShape->GetStorageShape().GetDim(1) != tilingData->k)),
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                MOE_UPDATE_EXPERT_DEBUG, "pruning_threshold",
                                (std::string("[") + std::to_string(pruningThresholdShape->GetStorageShape().GetDim(0)) +
                                 ", " + std::to_string(pruningThresholdShape->GetStorageShape().GetDim(1)) + "]")
                                    .c_str(),
                                "The shape of pruning_threshold must be [1, k]"),
                            return ge::GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIM(MOE_UPDATE_EXPERT_DEBUG, "pruning_threshold",
                                         std::to_string(pruningThresholdShape->GetStorageShape().GetDimNum()).c_str(),
                                         "1 or 2");
            return ge::GRAPH_FAILED;
        }

        tailorCfg_ += (1U << PRUNING_THRESHOLD_INDEX);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckActiveMaskShape(const gert::TilingContext *context)
{
    const gert::StorageShape *activeMaskShape = context->GetOptionalInputShape(ACTIVE_MASK_INDEX);

    if (activeMaskShape != nullptr) {
        OP_TILING_CHECK(
            (activeMaskShape->GetStorageShape().GetDimNum() != NUM_ONE),
            OP_LOGE_FOR_INVALID_SHAPEDIM(MOE_UPDATE_EXPERT_DEBUG, "active_mask",
                                         std::to_string(activeMaskShape->GetStorageShape().GetDimNum()).c_str(), "1"),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (activeMaskShape->GetStorageShape().GetDim(0) != tilingData->bs),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                MOE_UPDATE_EXPERT_DEBUG, "active_mask",
                (std::string("dim0=") + std::to_string(activeMaskShape->GetStorageShape().GetDim(0))).c_str(),
                "Dim0 of active_mask must be equal to bs"),
            return ge::GRAPH_FAILED);
        tailorCfg_ += (1U << ACTIVE_MASK_INDEX);
        tilingData->isActiveMask = 1;
    } else {
        tilingData->isActiveMask = 0;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckOptionalInputShape(const gert::TilingContext *context)
{
    OP_TILING_CHECK(CheckExpertScalesShape(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "CheckExpertScalesShape failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckPruningThresholdShape(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "CheckPruningThresholdShape failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckActiveMaskShape(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "CheckActiveMaskShape failed!"), return ge::GRAPH_FAILED);

    // active_mask, expert_scales, pruning_threshold,
    OP_TILING_CHECK(
        tailorCfg_ == TAILOR_EXPERT_SCALES,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MOE_UPDATE_EXPERT_DEBUG, "tailorCfg", std::to_string(tailorCfg_).c_str(),
                                              "expert_scales has been set, pruning_threshold must be set"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        tailorCfg_ == TAILOR_PRUNING_THRESHOLD,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MOE_UPDATE_EXPERT_DEBUG, "tailorCfg", std::to_string(tailorCfg_).c_str(),
                                              "pruning_threshold has been set, expert_scales must be set"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tailorCfg_ == TAILOR_ACTIVE_MASK,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        MOE_UPDATE_EXPERT_DEBUG, "tailorCfg", std::to_string(tailorCfg_).c_str(),
                        "active_mask has been set, pruning_threshold and expert_scales must be set"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tailorCfg_ == TAILOR_ACTIVE_MASK_EXPERT_SCALES,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        MOE_UPDATE_EXPERT_DEBUG, "tailorCfg", std::to_string(tailorCfg_).c_str(),
                        "active_mask and expert_scales have been set, pruning_threshold must be set"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tailorCfg_ == TAILOR_ACTIVE_MASK_PRUNING_THRESHOLD,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        MOE_UPDATE_EXPERT_DEBUG, "tailorCfg", std::to_string(tailorCfg_).c_str(),
                        "active_mask and pruning_threshold have been set, expert_scales must be set"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckOutputShape(const gert::TilingContext *context) const
{
    const gert::StorageShape *balancedExpertIdsStorageShape = context->GetOutputShape(OUTPUT_BALANCED_EXPERT_IDS);
    OP_TILING_CHECK(balancedExpertIdsStorageShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "balancedExpertIds"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((balancedExpertIdsStorageShape->GetStorageShape().GetDimNum() != NUM_TWO),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        MOE_UPDATE_EXPERT_DEBUG, "balanced_expert_ids",
                        std::to_string(balancedExpertIdsStorageShape->GetStorageShape().GetDimNum()).c_str(), "2"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((balancedExpertIdsStorageShape->GetStorageShape().GetDim(0) != tilingData->bs) ||
                     (balancedExpertIdsStorageShape->GetStorageShape().GetDim(1) != tilingData->k)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        MOE_UPDATE_EXPERT_DEBUG, "balanced_expert_ids",
                        (std::string("[") + std::to_string(balancedExpertIdsStorageShape->GetStorageShape().GetDim(0)) +
                         ", " + std::to_string(balancedExpertIdsStorageShape->GetStorageShape().GetDim(1)) + "]")
                            .c_str(),
                        "The shape of balanced_expert_ids must be [bs, k]"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *balancedActiveMaskShape = context->GetOutputShape(OUTPUT_ACTIVE_MASK_IDS);
    OP_TILING_CHECK(balancedActiveMaskShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(MOE_UPDATE_EXPERT_DEBUG, "balancedActiveMask"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((balancedActiveMaskShape->GetStorageShape().GetDimNum() != NUM_TWO),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        MOE_UPDATE_EXPERT_DEBUG, "balanced_active_mask",
                        std::to_string(balancedActiveMaskShape->GetStorageShape().GetDimNum()).c_str(), "2"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((balancedActiveMaskShape->GetStorageShape().GetDim(0) != tilingData->bs) ||
                     (balancedActiveMaskShape->GetStorageShape().GetDim(1) != tilingData->k)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        MOE_UPDATE_EXPERT_DEBUG, "balanced_active_mask",
                        (std::string("[") + std::to_string(balancedActiveMaskShape->GetStorageShape().GetDim(0)) +
                         ", " + std::to_string(balancedActiveMaskShape->GetStorageShape().GetDim(1)) + "]")
                            .c_str(),
                        "The shape of balanced_active_mask must be [bs, k]"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::CheckShape(const gert::TilingContext *context)
{
    OP_TILING_CHECK(CheckInputShape(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check input shape failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOptionalInputShape(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check optional input shape failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputShape(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check output shape failed!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeUpdateExpertTiling::Init(gert::TilingContext *context)
{
    tilingData = context->GetTilingData<MoeUpdateExpertTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "tilingData is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAttrs(context) != ge::GRAPH_SUCCESS, OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check attrs failed!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDataType(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check data type failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckFormat(context) != ge::GRAPH_SUCCESS, OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check format failed!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "Check shape failed!"),
                    return ge::GRAPH_FAILED);

    OP_LOGD(MOE_UPDATE_EXPERT_DEBUG, "end tiling init, bs=%d, k=%d, moeExpertNum=%d, f=%d", tilingData->bs,
            tilingData->k, tilingData->moeExpertNum, tilingData->f);
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeUpdateExpertTiling::GetTilingKey() const
{
    uint32_t tilingKeyBalanceMode = (tailorCfg_ == TAILOR_NONE) ? RANK_ID_BALANCING_MODE : TOKEN_ID_BALANCING_MODE;
    uint32_t tilingKeyDataType = TILINGKEY_FLOAT;
    if (keyScales_ == KEY_SCALES_FP16) {
        tilingKeyDataType = TILINGKEY_HALF;
    } else if (keyScales_ == KEY_SCALES_BF16) {
        tilingKeyDataType = TILINGKEY_BFLOAT16;
    }

    uint64_t tilingKey = GET_TPL_TILING_KEY(tilingKeyBalanceMode, tilingKeyDataType);
    return tilingKey;
}

ge::graphStatus MoeUpdateExpertTiling::RunFusionKernelTiling(gert::TilingContext *context)
{
    OP_LOGD(MOE_UPDATE_EXPERT_DEBUG, "begin RunFusionKernelTiling.");

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->aivCoreNum = aivNum;

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "get workspace failed"),
                    return ge::GRAPH_FAILED);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaces[0] = libApiWorkSpaceSize_;

    uint64_t tilingKey = GetTilingKey();
    context->SetTilingKey(tilingKey);

    OP_LOGD(MOE_UPDATE_EXPERT_DEBUG, "end RunFusionKernelTiling, tilingKey=%lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeUpdateExpertTilingFunc(gert::TilingContext *context)
{
    MoeUpdateExpertTiling tiling;
    OP_TILING_CHECK(tiling.Init(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(MOE_UPDATE_EXPERT_DEBUG, "MoeUpdateExpert tiling init failed."), return ge::GRAPH_FAILED);
    return tiling.RunFusionKernelTiling(context);
}

struct MoeUpdateExpertCompileInfo {};
static ge::graphStatus TilingPrepareForMoeUpdateExpert(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<MoeUpdateExpertCompileInfo>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeUpdateExpert)
    .Tiling(MoeUpdateExpertTilingFunc)
    .TilingParse<MoeUpdateExpertCompileInfo>(TilingPrepareForMoeUpdateExpert); // 向框架注册入口函数
} // namespace optiling
