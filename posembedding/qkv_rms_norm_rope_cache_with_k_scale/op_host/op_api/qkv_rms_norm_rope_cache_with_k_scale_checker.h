/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CHECKER_H
#define OP_API_INC_QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CHECKER_H

#include "op_common/log/log.h"
#include "opdev/op_errno.h"
#include "opdev/shape_utils.h"

#include <cstdint>

namespace QkvRmsNormRopeCacheWithKScaleCheck {

constexpr float DEFAULT_EPSILON = 1e-6f;

enum class QQuantMode : uint8_t {
    PER_TOKEN_PER_HEAD = 0,
    NO_QUANT = 1,
    MX_QUANT = 2,
    INVALID = 3,
};

enum class KQuantMode : uint8_t {
    PER_TOKEN_PER_HEAD = 0,
    MX_QUANT = 1,
    INVALID = 2,
};

struct QkvRmsNormRopeCacheWithKScaleParams {
    // Required inputs
    const aclTensor *qkv{nullptr};
    const aclTensor *qGamma{nullptr};
    const aclTensor *kGamma{nullptr};
    const aclTensor *cosSin{nullptr};
    const aclTensor *slotMapping{nullptr};
    aclTensor *kCache{nullptr};
    aclTensor *vCache{nullptr};
    aclTensor *kScaleCache{nullptr};
    // Scene-specific optional inputs
    const aclTensor *queryStartLoc{nullptr};
    const aclTensor *seqLens{nullptr};
    const aclTensor *rotationOptional{nullptr};
    const aclTensor *vScaleOptional{nullptr};
    const aclTensor *mropePositionOptional{nullptr};
    // Attributes
    const aclIntArray *headNums{nullptr};
    const char *layoutQkv{nullptr};
    const char *layoutQOut{nullptr};
    float epsilon{DEFAULT_EPSILON};
    const aclIntArray *mropeSectionOptional{nullptr};
    QQuantMode qQuantMode{QQuantMode::INVALID};
    KQuantMode kQuantMode{KQuantMode::INVALID};
    // Outputs
    aclTensor *qOut{nullptr};
    aclTensor *qScale{nullptr};
    // Workspace/executor
    uint64_t *workspaceSize{nullptr};
    aclOpExecutor **executor{nullptr};
};

class QkvRmsNormRopeCacheWithKScaleChecker {
public:
    static constexpr const char *ACLNN_NAME = "aclnnQkvRmsNormRopeCacheWithKScale";

    // Semantic validation belongs to InferShape/graph tiling. ACLNN only checks
    // pointers required to construct the task and public optional-output presence.
    aclnnStatus CheckParams(const QkvRmsNormRopeCacheWithKScaleParams &params) const
    {
        const PointerNullRule requiredPointers[] = {
            {"qkv", params.qkv, "qkv can not be nullptr"},
            {"qGamma", params.qGamma, "qGamma can not be nullptr"},
            {"kGamma", params.kGamma, "kGamma can not be nullptr"},
            {"cosSin", params.cosSin, "cosSin can not be nullptr"},
            {"slotMapping", params.slotMapping, "slotMapping can not be nullptr"},
            {"kCacheRef", params.kCache, "kCacheRef can not be nullptr"},
            {"vCacheRef", params.vCache, "vCacheRef can not be nullptr"},
            {"kScaleCacheRef", params.kScaleCache, "kScaleCacheRef can not be nullptr"},
            {"vScaleOptional", params.vScaleOptional, "vScaleOptional can not be nullptr"},
            {"headNums", params.headNums, "headNums can not be nullptr"},
            {"qOut", params.qOut, "qOut can not be nullptr"},
            {"workspaceSize", params.workspaceSize, "workspaceSize can not be nullptr"},
            {"executor", params.executor, "executor can not be nullptr"},
        };
        for (const auto &rule : requiredPointers) {
            if (rule.ptr == nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, rule.paramName, "nullptr", rule.reason);
                return ACLNN_ERR_PARAM_NULLPTR;
            }
        }

        const bool isRopeScene = params.qQuantMode == QQuantMode::PER_TOKEN_PER_HEAD &&
                                 params.kQuantMode == KQuantMode::PER_TOKEN_PER_HEAD &&
                                 params.mropePositionOptional == nullptr;
        if (isRopeScene && params.queryStartLoc == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "queryStartLoc", "nullptr",
                                                  "queryStartLoc can not be nullptr in RoPE scene");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (isRopeScene && params.seqLens == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "seqLens", "nullptr",
                                                  "seqLens can not be nullptr in RoPE scene");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if ((isRopeScene || params.qQuantMode == QQuantMode::NO_QUANT) && params.rotationOptional == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "rotationOptional", "nullptr",
                                                  "rotationOptional can not be nullptr in RoPE/M-RoPE scene");
            return ACLNN_ERR_PARAM_NULLPTR;
        }

        if (params.qQuantMode == QQuantMode::NO_QUANT && params.qScale != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qScaleOptional", "not nullptr",
                                                  "qScaleOptional must be nullptr when qQuantMode is NoQuant");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((params.qQuantMode == QQuantMode::PER_TOKEN_PER_HEAD || params.qQuantMode == QQuantMode::MX_QUANT) &&
            params.qScale == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qScaleOptional", "nullptr",
                                                  "qScaleOptional is required when Q output is quantized");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        return ACLNN_SUCCESS;
    }

private:
    struct PointerNullRule {
        const char *paramName;
        const void *ptr;
        const char *reason;
    };
};

} // namespace QkvRmsNormRopeCacheWithKScaleCheck

#endif // OP_API_INC_QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CHECKER_H
