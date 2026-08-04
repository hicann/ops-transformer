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
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_errno.h"
#include "opdev/shape_utils.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace QkvRmsNormRopeCacheWithKScaleCheck {

constexpr float DEFAULT_EPSILON = 1e-6f;

enum class QQuantMode : uint8_t {
    PER_TOKEN_PER_HEAD = 0,
    NO_QUANT = 1,
    INVALID = 2,
};

inline bool CheckType(op::DataType dtype, const std::vector<op::DataType> &supportList)
{
    for (const auto &supported : supportList) {
        if (dtype == supported) {
            return true;
        }
    }
    return false;
}

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
    // Outputs
    aclTensor *qOut{nullptr};
    aclTensor *qScale{nullptr};
    // Workspace/executor
    uint64_t *workspaceSize{nullptr};
    aclOpExecutor **executor{nullptr};
};

class QkvRmsNormRopeCacheWithKScaleChecker {
    enum class SceneType {
        ROPE,
        MROPE
    };

public:
    // ACLNN operator name for log messages
    static constexpr const char *ACLNN_NAME = "aclnnQkvRmsNormRopeCacheWithKScale";

    // Constants
    static constexpr uint64_t HEAD_NUMS_SIZE = 3;
    static constexpr int64_t SUPPORTED_HEAD_DIM = 128;
    static constexpr int64_t MAX_Q_HEAD_NUM = 64;
    static constexpr int64_t Q_TO_K_HEAD_RATIO = 8;
    static constexpr uint64_t RANK_1D = 1;
    static constexpr uint64_t RANK_2D = 2;
    static constexpr uint64_t RANK_3D = 3;
    static constexpr uint64_t RANK_4D = 4;
    static constexpr uint64_t HEAD_NUMS_NQ_INDEX = 0;
    static constexpr uint64_t HEAD_NUMS_NK_INDEX = 1;
    static constexpr uint64_t HEAD_NUMS_NV_INDEX = 2;
    static constexpr uint64_t QKV_NTD_HEAD_INDEX = 0;
    static constexpr uint64_t QKV_NTD_TOKEN_INDEX = 1;
    static constexpr uint64_t QKV_TND_TOKEN_INDEX = 0;
    static constexpr uint64_t QKV_TND_HEAD_INDEX = 1;
    static constexpr uint64_t QKV_HEAD_DIM_INDEX = 2;
    static constexpr uint64_t COS_SIN_HEAD_DIM_INDEX = 1;
    static constexpr uint64_t VECTOR_LENGTH_INDEX = 0;
    static constexpr uint64_t CACHE_BLOCK_NUM_INDEX = 0;
    static constexpr uint64_t CACHE_HEAD_INDEX = 1;
    static constexpr uint64_t CACHE_BLOCK_SIZE_INDEX = 2;
    static constexpr uint64_t CACHE_HEAD_DIM_INDEX = 3;
    static constexpr int64_t MIN_QUERY_START_LOC_SIZE = 2;
    static constexpr int64_t SEQ_LENS_BATCH_DIFF = 1;
    static constexpr int64_t K_SCALE_CACHE_SCALE_DIM = 1;
    static constexpr uint64_t MROPE_SECTION_SIZE = 3;
    static constexpr const char *QKV_LAYOUT_NTD = "NTD";
    static constexpr const char *QKV_LAYOUT_TND = "TND";
    static constexpr const char *DEFAULT_QKV_LAYOUT = QKV_LAYOUT_TND;
    static constexpr const char *DEFAULT_Q_OUT_LAYOUT = QKV_LAYOUT_NTD;
    // Dtype support lists
    inline static const std::vector<op::DataType> QKV_TYPE_LIST = {op::DataType::DT_BF16};
    inline static const std::vector<op::DataType> GAMMA_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::vector<op::DataType> COS_SIN_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::vector<op::DataType> SLOT_MAPPING_TYPE_LIST = {op::DataType::DT_INT32};
    inline static const std::vector<op::DataType> CACHE_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
    inline static const std::vector<op::DataType> MROPE_K_CACHE_TYPE_LIST = {op::DataType::DT_INT8};
    inline static const std::vector<op::DataType> K_SCALE_CACHE_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::vector<op::DataType> QUERY_START_LOC_TYPE_LIST = {op::DataType::DT_INT32};
    inline static const std::vector<op::DataType> SEQ_LENS_TYPE_LIST = {op::DataType::DT_INT32};
    inline static const std::vector<op::DataType> ROTATION_TYPE_LIST = {op::DataType::DT_BF16};
    inline static const std::vector<op::DataType> V_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT};
    inline static const std::vector<op::DataType> MROPE_POSITION_TYPE_LIST = {op::DataType::DT_INT32};
    inline static const std::vector<op::DataType> Q_OUT_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
    inline static const std::vector<op::DataType> MROPE_Q_OUT_TYPE_LIST = {op::DataType::DT_BF16};
    inline static const std::vector<op::DataType> Q_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT};

    // Main entry point
    aclnnStatus CheckParams(const QkvRmsNormRopeCacheWithKScaleParams &p)
    {
        params_ = p;
        aclnnStatus ret = CheckNotNull();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = DetermineScene();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckSceneNotNull();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = scene_ == SceneType::MROPE ? CheckMropeSceneParams() : CheckRopeSceneParams();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckEmptyTensor();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckDtypeValid();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckShape();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckFormat();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        return ACLNN_SUCCESS;
    }

private:
    QkvRmsNormRopeCacheWithKScaleParams params_;

    SceneType scene_{SceneType::ROPE};

    // Classify the call from the presence of the two complete position sources.
    aclnnStatus DetermineScene()
    {
        const bool hasQueryStartLoc = params_.queryStartLoc != nullptr;
        const bool hasSeqLens = params_.seqLens != nullptr;
        const bool hasMropePosition = params_.mropePositionOptional != nullptr;
        const bool hasMropeSection =
            params_.mropeSectionOptional != nullptr && params_.mropeSectionOptional->Size() != 0;
        if (hasMropePosition != hasMropeSection) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "mropePositionOptional/mropeSectionOptional",
                                                  "presence mismatch", "both parameters must be provided together");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (hasMropePosition && (hasQueryStartLoc || hasSeqLens)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "queryStartLocOptional/seqLensOptional", "not nullptr",
                                                  "both parameters must be nullptr in the M-RoPE scene");
            return ACLNN_ERR_PARAM_INVALID;
        }
        scene_ = hasMropePosition ? SceneType::MROPE : SceneType::ROPE;
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckRopeSceneParams() const
    {
        if (params_.qQuantMode != QQuantMode::PER_TOKEN_PER_HEAD) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qQuantMode", "invalid mode",
                                                  "the RoPE scene requires PerTokenPerHead");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return CheckDtypeSupported("kCacheRef", params_.kCache, CACHE_TYPE_LIST);
    }

    aclnnStatus CheckMropeSceneParams() const
    {
        if (params_.qQuantMode != QQuantMode::NO_QUANT) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qQuantMode", "invalid mode",
                                                  "the M-RoPE scene requires NoQuant");
            return ACLNN_ERR_PARAM_INVALID;
        }
        aclnnStatus kCacheRet = CheckDtypeSupported("kCacheRef", params_.kCache, MROPE_K_CACHE_TYPE_LIST);
        if (kCacheRet != ACLNN_SUCCESS) {
            return kCacheRet;
        }
        if (params_.qScale != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "qScaleOptional", "not nullptr",
                                                  "qScaleOptional must be nullptr in the M-RoPE scene");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckMropePositionShape(int64_t tokenCount) const
    {
        return CheckExactShape("mropePositionOptional", params_.mropePositionOptional,
                               {tokenCount, static_cast<int64_t>(MROPE_SECTION_SIZE)},
                               "mropePositionOptional must have shape [T, 3] matching qkv token count");
    }

    bool IsMropeScene() const { return scene_ == SceneType::MROPE; }

    struct PointerNullRule {
        const char *paramName;
        const void *ptr;
        const char *reason;
    };

    struct TensorDtypeRule {
        const char *paramName;
        const aclTensor *tensor;
        const std::vector<op::DataType> *supportList;
    };

    struct TensorRule {
        const char *paramName;
        const aclTensor *tensor;
    };

    // Helper: shape to string
    static std::string ShapeToString(const aclTensor *tensor)
    {
        std::string result;
        const auto &shape = tensor->GetViewShape();
        for (uint64_t i = 0; i < shape.GetDimNum(); ++i) {
            if (i != 0) {
                result += ", ";
            }
            result += std::to_string(shape.GetDim(i));
        }
        return result;
    }

    // Helper: check if tensor has expected shape
    static bool IsShape(const aclTensor *tensor, const std::vector<int64_t> &expectedShape)
    {
        const auto &shape = tensor->GetViewShape();
        if (shape.GetDimNum() != expectedShape.size()) {
            return false;
        }
        for (uint64_t i = 0; i < static_cast<uint64_t>(expectedShape.size()); ++i) {
            if (shape.GetDim(i) != expectedShape[i]) {
                return false;
            }
        }
        return true;
    }

    static std::string DataTypeListToString(const std::vector<op::DataType> &supportList)
    {
        std::string result;
        for (uint64_t i = 0; i < static_cast<uint64_t>(supportList.size()); ++i) {
            if (i != 0) {
                result += ", ";
            }
            result += op::ToString(supportList[i]).GetString();
        }
        return result;
    }

    static aclnnStatus CheckDtypeSupported(const std::string &paramName, const aclTensor *tensor,
                                           const std::vector<op::DataType> &supportList)
    {
        if (!CheckType(tensor->GetDataType(), supportList)) {
            const std::string incorrectDtype = op::ToString(tensor->GetDataType()).GetString();
            const std::string reason =
                "the dtype of " + paramName + " must be one of " + DataTypeListToString(supportList);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ACLNN_NAME, paramName.c_str(), incorrectDtype.c_str(),
                                                  reason.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckFormatValid(const TensorRule &rule)
    {
        if (unlikely(op::IsPrivateFormat(rule.tensor->GetStorageFormat()))) {
            const std::string incorrectFormat = op::ToString(rule.tensor->GetStorageFormat()).GetString();
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, rule.paramName, incorrectFormat.c_str(), "ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckTensorNotEmpty(const TensorRule &rule)
    {
        if (rule.tensor->GetViewShape().GetShapeSize() == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, rule.paramName, "empty tensor",
                                                  "tensor can not be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static aclnnStatus CheckExactShape(const char *paramName, const aclTensor *tensor,
                                       const std::vector<int64_t> &expectedShape, const char *reason)
    {
        if (!IsShape(tensor, expectedShape)) {
            const std::string incorrectShape = ShapeToString(tensor);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, paramName, incorrectShape.c_str(), reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    // 1. Common null pointer check
    aclnnStatus CheckNotNull()
    {
        const PointerNullRule commonRules[] = {
            {"qkv", params_.qkv, "qkv can not be nullptr"},
            {"qGamma", params_.qGamma, "qGamma can not be nullptr"},
            {"kGamma", params_.kGamma, "kGamma can not be nullptr"},
            {"cosSin", params_.cosSin, "cosSin can not be nullptr"},
            {"slotMapping", params_.slotMapping, "slotMapping can not be nullptr"},
            {"kCacheRef", params_.kCache, "kCacheRef can not be nullptr"},
            {"vCacheRef", params_.vCache, "vCacheRef can not be nullptr"},
            {"kScaleCacheRef", params_.kScaleCache, "kScaleCacheRef can not be nullptr"},
            {"rotationOptional", params_.rotationOptional, "rotationOptional can not be nullptr"},
            {"vScaleOptional", params_.vScaleOptional, "vScaleOptional can not be nullptr"},
            {"headNums", params_.headNums, "headNums can not be nullptr"},
            {"qOut", params_.qOut, "qOut can not be nullptr"},
            {"workspaceSize", params_.workspaceSize, "workspaceSize can not be nullptr"},
            {"executor", params_.executor, "executor can not be nullptr"},
        };
        for (const auto &rule : commonRules) {
            if (rule.ptr == nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, rule.paramName, "nullptr", rule.reason);
                return ACLNN_ERR_PARAM_NULLPTR;
            }
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckSceneNotNull() const
    {
        if (IsMropeScene()) {
            return ACLNN_SUCCESS;
        }
        const PointerNullRule ropeRules[] = {
            {"queryStartLoc", params_.queryStartLoc, "queryStartLoc can not be nullptr"},
            {"seqLens", params_.seqLens, "seqLens can not be nullptr"},
            {"qScale", params_.qScale, "qScale can not be nullptr"},
        };
        for (const auto &rule : ropeRules) {
            if (rule.ptr == nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, rule.paramName, "nullptr", rule.reason);
                return ACLNN_ERR_PARAM_NULLPTR;
            }
        }
        return ACLNN_SUCCESS;
    }

    // 2. Dtype check
    aclnnStatus CheckDtypeValid()
    {
        const bool isMrope = IsMropeScene();
        const TensorDtypeRule commonRules[] = {
            {"qkv", params_.qkv, &QKV_TYPE_LIST},
            {"qGamma", params_.qGamma, &GAMMA_TYPE_LIST},
            {"kGamma", params_.kGamma, &GAMMA_TYPE_LIST},
            {"cosSin", params_.cosSin, &COS_SIN_TYPE_LIST},
            {"slotMapping", params_.slotMapping, &SLOT_MAPPING_TYPE_LIST},
            {"vCacheRef", params_.vCache, &CACHE_TYPE_LIST},
            {"kScaleCacheRef", params_.kScaleCache, &K_SCALE_CACHE_TYPE_LIST},
            {"rotationOptional", params_.rotationOptional, &ROTATION_TYPE_LIST},
            {"vScaleOptional", params_.vScaleOptional, &V_SCALE_TYPE_LIST},
            {"qOut", params_.qOut, isMrope ? &MROPE_Q_OUT_TYPE_LIST : &Q_OUT_TYPE_LIST},
        };
        for (const auto &rule : commonRules) {
            aclnnStatus ret = CheckDtypeSupported(rule.paramName, rule.tensor, *rule.supportList);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
        if (isMrope) {
            return CheckDtypeSupported("mropePositionOptional", params_.mropePositionOptional,
                                       MROPE_POSITION_TYPE_LIST);
        }
        const TensorDtypeRule ropeRules[] = {
            {"queryStartLoc", params_.queryStartLoc, &QUERY_START_LOC_TYPE_LIST},
            {"seqLens", params_.seqLens, &SEQ_LENS_TYPE_LIST},
            {"qScale", params_.qScale, &Q_SCALE_TYPE_LIST},
        };
        for (const auto &rule : ropeRules) {
            aclnnStatus ret = CheckDtypeSupported(rule.paramName, rule.tensor, *rule.supportList);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
        return ACLNN_SUCCESS;
    }

    // 3. Shape check
    aclnnStatus ParseHeadNums(int64_t &nq, int64_t &nk, int64_t &nv)
    {
        if (params_.headNums->Size() != static_cast<int64_t>(HEAD_NUMS_SIZE)) {
            const std::string incorrectSize = std::to_string(params_.headNums->Size());
            const std::string correctSize = std::to_string(HEAD_NUMS_SIZE);
            OP_LOGE_FOR_INVALID_LISTSIZE(ACLNN_NAME, "headNums", incorrectSize.c_str(), correctSize.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t *headNumsData = params_.headNums->GetData();
        if (headNumsData == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "headNums", "nullptr",
                                                  "headNums data can not be nullptr");
            return ACLNN_ERR_PARAM_INVALID;
        }

        nq = headNumsData[HEAD_NUMS_NQ_INDEX];
        nk = headNumsData[HEAD_NUMS_NK_INDEX];
        nv = headNumsData[HEAD_NUMS_NV_INDEX];
        if (nq <= 0 || nk <= 0 || nv <= 0) {
            const std::string incorrectValue =
                std::to_string(nq) + ", " + std::to_string(nk) + ", " + std::to_string(nv);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "headNums", incorrectValue.c_str(),
                                                  "headNums values must be positive");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (nq > MAX_Q_HEAD_NUM) {
            const std::string incorrectValue = std::to_string(nq);
            OP_LOGE_FOR_INVALID_VALUE(ACLNN_NAME, "Nq", incorrectValue.c_str(), "less than or equal to 64");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (nq % Q_TO_K_HEAD_RATIO != 0 || nk != nq / Q_TO_K_HEAD_RATIO) {
            const std::string incorrectValue = std::to_string(nq) + ", " + std::to_string(nk);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "Nq, Nk", incorrectValue.c_str(),
                                                  "nq must be equal to 8 * nk");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (nv != nk) {
            const std::string incorrectValue = std::to_string(nv) + ", " + std::to_string(nk);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "Nv, Nk", incorrectValue.c_str(),
                                                  "nv must be equal to nk");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static const char *GetLayoutQkvOrDefault(const char *layout)
    {
        return layout == nullptr || layout[0] == '\0' ? DEFAULT_QKV_LAYOUT : layout;
    }

    static const char *GetLayoutQOutOrDefault(const char *layout)
    {
        return layout == nullptr || layout[0] == '\0' ? DEFAULT_Q_OUT_LAYOUT : layout;
    }

    aclnnStatus CheckLayoutQkv(bool &isTnd) const
    {
        const char *layoutQkv = GetLayoutQkvOrDefault(params_.layoutQkv);
        const bool isNtd = std::strcmp(layoutQkv, QKV_LAYOUT_NTD) == 0;
        isTnd = std::strcmp(layoutQkv, QKV_LAYOUT_TND) == 0;
        if (!isNtd && !isTnd) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "layout_qkv", layoutQkv, "layout_qkv must be NTD or TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckLayoutQOut(bool &isTnd) const
    {
        const char *layoutQkv = GetLayoutQkvOrDefault(params_.layoutQkv);
        const char *layoutQOut = GetLayoutQOutOrDefault(params_.layoutQOut);
        const bool isNtd = std::strcmp(layoutQOut, QKV_LAYOUT_NTD) == 0;
        isTnd = std::strcmp(layoutQOut, QKV_LAYOUT_TND) == 0;
        if (!isNtd && !isTnd) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "layout_q_out", layoutQOut,
                                                  "layout_q_out must be NTD or TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const bool qkvIsNtd = std::strcmp(layoutQkv, QKV_LAYOUT_NTD) == 0;
        if (qkvIsNtd && isTnd) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "layout_q_out", layoutQOut,
                                                  "layout_qkv=NTD with layout_q_out=TND is not supported");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckQkvShape(int64_t nq, int64_t nk, int64_t nv, bool isTnd, int64_t &t, int64_t &d) const
    {
        const auto &qkvShape = params_.qkv->GetViewShape();
        uint64_t qkvDimNum = qkvShape.GetDimNum();
        if (unlikely(qkvDimNum != RANK_3D)) {
            const std::string incorrectDim = std::to_string(qkvDimNum) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "qkv", incorrectDim.c_str(),
                                                     "qkv must be a 3D tensor [Nq+Nk+Nv, T, D] or [T, Nq+Nk+Nv, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t hqkv = qkvShape.GetDim(isTnd ? QKV_TND_HEAD_INDEX : QKV_NTD_HEAD_INDEX);
        t = qkvShape.GetDim(isTnd ? QKV_TND_TOKEN_INDEX : QKV_NTD_TOKEN_INDEX);
        d = qkvShape.GetDim(QKV_HEAD_DIM_INDEX);
        if (unlikely(hqkv != nq + nk + nv)) {
            const std::string incorrectValues = std::to_string(hqkv) + ", " + std::to_string(nq + nk + nv);
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(ACLNN_NAME, "qkv, headNums", incorrectValues.c_str(),
                                                   "qkv logical N dimension must equal Nq+Nk+Nv from headNums");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(d != SUPPORTED_HEAD_DIM)) {
            const std::string incorrectValue = std::to_string(d);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "D", incorrectValue.c_str(),
                                                  "head dimension must be 128");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    static int64_t GetMropeAxisCapacity(int64_t halfHeadDim, uint64_t axis)
    {
        const int64_t axisOffset = static_cast<int64_t>(axis);
        if (halfHeadDim <= axisOffset) {
            return 0;
        }
        // Axis capacity counts lanes of the form lane = 3 * r + axis below D/2.
        return (halfHeadDim - 1 - axisOffset) / static_cast<int64_t>(MROPE_SECTION_SIZE) + 1;
    }

    aclnnStatus CheckMropeSection(int64_t headDim) const
    {
        const uint64_t sectionSize = params_.mropeSectionOptional->Size();
        if (sectionSize != MROPE_SECTION_SIZE) {
            const std::string incorrectSize = std::to_string(sectionSize);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "mropeSectionOptional", incorrectSize.c_str(),
                                                  "mropeSectionOptional must contain exactly 3 values [t, h, w]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const int64_t halfHeadDim = headDim / 2;
        int64_t sectionSum = 0;
        for (uint64_t i = 0; i < MROPE_SECTION_SIZE; ++i) {
            const int64_t value = (*params_.mropeSectionOptional)[i];
            if (value < 0) {
                const std::string incorrectValue = std::to_string(value);
                const std::string reason = "mropeSectionOptional[" + std::to_string(i) + "] must be non-negative";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "mropeSectionOptional", incorrectValue.c_str(),
                                                      reason.c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (i != 0U) {
                const int64_t axisCapacity = GetMropeAxisCapacity(halfHeadDim, i);
                if (value > axisCapacity) {
                    const std::string incorrectValue = std::to_string(value);
                    const std::string reason = "mropeSectionOptional[" + std::to_string(i) + "] must be in [0, " +
                                               std::to_string(axisCapacity) +
                                               "] according to lane = 3 * r + axis within D/2";
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "mropeSectionOptional",
                                                          incorrectValue.c_str(), reason.c_str());
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            if (value > halfHeadDim - sectionSum) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "mropeSectionOptional", "sum exceeds D/2",
                                                      "the sum of mropeSectionOptional must not exceed D/2");
                return ACLNN_ERR_PARAM_INVALID;
            }
            sectionSum += value;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckNonCacheShapes(int64_t t, int64_t d, int64_t nv) const
    {
        if (!IsShape(params_.qGamma, {d}) || !IsShape(params_.kGamma, {d})) {
            const std::string incorrectShapes = ShapeToString(params_.qGamma) + ", " + ShapeToString(params_.kGamma);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "qGamma, kGamma", incorrectShapes.c_str(),
                                                   "qGamma and kGamma must have shape [D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const auto &cosSinShape = params_.cosSin->GetViewShape();
        if (unlikely(cosSinShape.GetDimNum() != RANK_2D || cosSinShape.GetDim(COS_SIN_HEAD_DIM_INDEX) != d)) {
            const std::string incorrectShape = ShapeToString(params_.cosSin);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "cosSin", incorrectShape.c_str(),
                                                  "cosSin must have shape [MaxSeqLen, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        aclnnStatus ret = CheckExactShape("slotMapping", params_.slotMapping, {t}, "slotMapping must have shape [T]");
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        if (!IsMropeScene()) {
            const auto &queryStartLocShape = params_.queryStartLoc->GetViewShape();
            if (unlikely(queryStartLocShape.GetDimNum() != RANK_1D ||
                         queryStartLocShape.GetDim(VECTOR_LENGTH_INDEX) < MIN_QUERY_START_LOC_SIZE)) {
                const std::string incorrectShape = ShapeToString(params_.queryStartLoc);
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "queryStartLoc", incorrectShape.c_str(),
                                                      "queryStartLoc must be a 1D tensor with length >= 2");
                return ACLNN_ERR_PARAM_INVALID;
            }
            const auto &seqLensShape = params_.seqLens->GetViewShape();
            if (unlikely(seqLensShape.GetDimNum() != RANK_1D ||
                         seqLensShape.GetDim(VECTOR_LENGTH_INDEX) !=
                             queryStartLocShape.GetDim(VECTOR_LENGTH_INDEX) - SEQ_LENS_BATCH_DIFF)) {
                const std::string incorrectShape = ShapeToString(params_.seqLens);
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "seqLens", incorrectShape.c_str(),
                                                      "seqLens must have shape [queryStartLoc.shape[0] - 1]");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (!IsShape(params_.rotationOptional, {d, d})) {
            const std::string incorrectShape = ShapeToString(params_.rotationOptional);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "rotationOptional", incorrectShape.c_str(),
                                                  "rotationOptional must have shape [D, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (IsMropeScene()) {
            return CheckExactShape("vScaleOptional", params_.vScaleOptional, {nv, d},
                                   "vScaleOptional must have shape [Nv, D] in M-RoPE scene");
        }
        return CheckExactShape("vScaleOptional", params_.vScaleOptional, {nv},
                               "vScaleOptional must have shape [Nv] in RoPE scene");
    }

    aclnnStatus CheckCacheShapes(int64_t nk, int64_t nv, int64_t d) const
    {
        const auto &kCacheShape = params_.kCache->GetViewShape();
        if (unlikely(kCacheShape.GetDimNum() != RANK_4D || kCacheShape.GetDim(CACHE_HEAD_INDEX) != nk ||
                     kCacheShape.GetDim(CACHE_HEAD_DIM_INDEX) != d)) {
            const std::string incorrectShape = ShapeToString(params_.kCache);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "kCacheRef", incorrectShape.c_str(),
                                                  "kCacheRef must have shape [BlockNum, Nk, BlockSize, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const auto &vCacheShape = params_.vCache->GetViewShape();
        if (unlikely(vCacheShape.GetDimNum() != RANK_4D || vCacheShape.GetDim(CACHE_HEAD_INDEX) != nv ||
                     vCacheShape.GetDim(CACHE_HEAD_DIM_INDEX) != d)) {
            const std::string incorrectShape = ShapeToString(params_.vCache);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "vCacheRef", incorrectShape.c_str(),
                                                  "vCacheRef must have shape [BlockNum, Nv, BlockSize, D]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const auto &kScaleCacheShape = params_.kScaleCache->GetViewShape();
        if (unlikely(kScaleCacheShape.GetDimNum() != RANK_4D || kScaleCacheShape.GetDim(CACHE_HEAD_INDEX) != nk ||
                     kScaleCacheShape.GetDim(CACHE_HEAD_DIM_INDEX) != K_SCALE_CACHE_SCALE_DIM)) {
            const std::string incorrectShape = ShapeToString(params_.kScaleCache);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "kScaleCacheRef", incorrectShape.c_str(),
                                                  "kScaleCacheRef must have shape [BlockNum, Nk, BlockSize, 1]");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (unlikely(kCacheShape.GetDim(CACHE_BLOCK_NUM_INDEX) != kScaleCacheShape.GetDim(CACHE_BLOCK_NUM_INDEX) ||
                     kCacheShape.GetDim(CACHE_BLOCK_SIZE_INDEX) != kScaleCacheShape.GetDim(CACHE_BLOCK_SIZE_INDEX))) {
            const std::string incorrectShapes = "kCacheRef=[" + ShapeToString(params_.kCache) + "], kScaleCacheRef=[" +
                                                ShapeToString(params_.kScaleCache) + "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "kCacheRef and kScaleCacheRef", incorrectShapes.c_str(),
                                                   "kScaleCacheRef BlockNum/BlockSize must match kCacheRef");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (unlikely(vCacheShape.GetDim(CACHE_BLOCK_NUM_INDEX) != kCacheShape.GetDim(CACHE_BLOCK_NUM_INDEX) ||
                     vCacheShape.GetDim(CACHE_BLOCK_SIZE_INDEX) != kCacheShape.GetDim(CACHE_BLOCK_SIZE_INDEX))) {
            const std::string incorrectShapes =
                "vCacheRef=[" + ShapeToString(params_.vCache) + "], kCacheRef=[" + ShapeToString(params_.kCache) + "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "vCacheRef and kCacheRef", incorrectShapes.c_str(),
                                                   "vCacheRef BlockNum/BlockSize must match kCacheRef");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    // ACLNN outputs are caller-allocated; validate their logical shapes before constructing the L0 task.
    aclnnStatus CheckOutputShapes(int64_t nq, int64_t t, int64_t d, bool isTnd) const
    {
        const std::vector<int64_t> expectedQOutShape =
            isTnd ? std::vector<int64_t>{t, nq, d} : std::vector<int64_t>{nq, t, d};
        aclnnStatus ret = CheckExactShape("qOut", params_.qOut, expectedQOutShape, "qOut must match layout_q_out");
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        if (!IsMropeScene()) {
            const std::vector<int64_t> expectedQScaleShape =
                isTnd ? std::vector<int64_t>{t, nq} : std::vector<int64_t>{nq, t};
            ret = CheckExactShape("qScale", params_.qScale, expectedQScaleShape, "qScale must match layout_q_out");
        }
        return ret;
    }

    aclnnStatus CheckShape()
    {
        int64_t nq = 0;
        int64_t nk = 0;
        int64_t nv = 0;
        aclnnStatus ret = ParseHeadNums(nq, nk, nv);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        bool isTnd = false;
        ret = CheckLayoutQkv(isTnd);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        bool qOutIsTnd = false;
        ret = CheckLayoutQOut(qOutIsTnd);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        if (IsMropeScene() && (!isTnd || !qOutIsTnd)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "layout_qkv/layout_q_out", "not TND/TND",
                                                  "the M-RoPE scene only supports TND to TND");
            return ACLNN_ERR_PARAM_INVALID;
        }

        int64_t t = 0;
        int64_t d = 0;
        ret = CheckQkvShape(nq, nk, nv, isTnd, t, d);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        if (IsMropeScene()) {
            ret = CheckMropePositionShape(t);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
            ret = CheckMropeSection(d);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }

        ret = CheckNonCacheShapes(t, d, nv);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        ret = CheckCacheShapes(nk, nv, d);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        return CheckOutputShapes(nq, t, d, qOutIsTnd);
    }

    // 4. Format check
    aclnnStatus CheckFormat()
    {
        const TensorRule commonRules[] = {
            {"qkv", params_.qkv},
            {"qGamma", params_.qGamma},
            {"kGamma", params_.kGamma},
            {"cosSin", params_.cosSin},
            {"slotMapping", params_.slotMapping},
            {"kCacheRef", params_.kCache},
            {"vCacheRef", params_.vCache},
            {"kScaleCacheRef", params_.kScaleCache},
            {"rotationOptional", params_.rotationOptional},
            {"vScaleOptional", params_.vScaleOptional},
            {"qOut", params_.qOut},
        };
        for (const auto &rule : commonRules) {
            aclnnStatus ret = CheckFormatValid(rule);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
        std::vector<TensorRule> sceneRules;
        if (IsMropeScene()) {
            sceneRules.push_back({"mropePositionOptional", params_.mropePositionOptional});
        } else {
            sceneRules.push_back({"queryStartLoc", params_.queryStartLoc});
            sceneRules.push_back({"seqLens", params_.seqLens});
            sceneRules.push_back({"qScale", params_.qScale});
        }
        for (const auto &rule : sceneRules) {
            aclnnStatus ret = CheckFormatValid(rule);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
        return ACLNN_SUCCESS;
    }

    // 5. Empty tensor check
    aclnnStatus CheckEmptyTensor()
    {
        const TensorRule commonRules[] = {
            {"qkv", params_.qkv},
            {"qGamma", params_.qGamma},
            {"kGamma", params_.kGamma},
            {"cosSin", params_.cosSin},
            {"slotMapping", params_.slotMapping},
            {"kCacheRef", params_.kCache},
            {"vCacheRef", params_.vCache},
            {"kScaleCacheRef", params_.kScaleCache},
            {"rotationOptional", params_.rotationOptional},
            {"vScaleOptional", params_.vScaleOptional},
            {"qOut", params_.qOut},
        };
        for (const auto &rule : commonRules) {
            aclnnStatus ret = CheckTensorNotEmpty(rule);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
        std::vector<TensorRule> sceneRules;
        if (IsMropeScene()) {
            sceneRules.push_back({"mropePositionOptional", params_.mropePositionOptional});
        } else {
            sceneRules.push_back({"queryStartLoc", params_.queryStartLoc});
            sceneRules.push_back({"seqLens", params_.seqLens});
            sceneRules.push_back({"qScale", params_.qScale});
        }
        for (const auto &rule : sceneRules) {
            aclnnStatus ret = CheckTensorNotEmpty(rule);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
        return ACLNN_SUCCESS;
    }
};

} // namespace QkvRmsNormRopeCacheWithKScaleCheck

#endif
