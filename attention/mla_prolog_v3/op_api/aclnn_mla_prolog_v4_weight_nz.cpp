/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <string>
#include <map>
#include <memory>
#include <set>
#include "graph/types.h"
#include "aclnn_mla_prolog_v4_weight_nz.h"
#include "log/log.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMlaPrologV3GetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk,
    const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin, const aclTensor *ropeCos, aclTensor *kvCacheRef, aclTensor *krCacheRef,
    const aclTensor *cacheIndexOptional, const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional, const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional, const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional, const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional, const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv, char *cacheModeOptional, bool queryNormFlag, int64_t weightQuantMode,
    int64_t kvCacheQuantMode, int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode,
    int64_t tileSize, double qcQrScale, double kcScale, bool doRope, const aclTensor *queryOut,
    const aclTensor *queryRopeOut, const aclTensor *dequantScaleQNopeOut, const aclTensor *queryNormOut,
    const aclTensor *dequantScaleQNormOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMlaPrologV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace {

class TensorHolder {
public:
    TensorHolder(const aclTensor *&output, aclDataType dataType, std::string varName)
    {
        inner_ = nullptr;
        name_ = varName;
        if (output == nullptr) {
            std::vector<int64_t> shape = {0};
            int64_t addr = 0xff;
            inner_ = aclCreateTensor(shape.data(), shape.size(), dataType, shape.data(), 0, ACL_FORMAT_ND, shape.data(),
                                     shape.size(), static_cast<void *>(&addr));
            output = inner_;
        }
    }

    ~TensorHolder()
    {
        if (inner_) {
            aclDestroyTensor(inner_);
            inner_ = nullptr;
        }
    }

    bool CheckTensorConditionalNotNull(bool conditional) const
    {
        if (inner_ && conditional) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMlaPrologV4", name_.c_str(), "null",
                                                  "this parameter is required under current configuration");
            return false;
        } else if (!inner_ && !conditional) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMlaPrologV4", name_.c_str(), "not null",
                                                  "this parameter should be empty under current configuration");
            return false;
        }
        return true;
    }

    bool IsTensorNotNull() const
    {
        return inner_ == nullptr;
    }

private:
    const aclTensor *inner_;
    std::string name_;
};

bool CheckWeightQuantModeValidity(int64_t weightQuantMode)
{
    std::set<int64_t> supportedWeightQuantMode;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        supportedWeightQuantMode = {0LL, 1LL, 2LL, 3LL, 4LL, 5LL};
    } else {
        supportedWeightQuantMode = {0LL, 1LL, 2LL};
    }
    if (supportedWeightQuantMode.find(weightQuantMode) == supportedWeightQuantMode.end()) {
        std::string supportedStr;
        size_t total = supportedWeightQuantMode.size();
        size_t idx = 0;
        for (auto mode : supportedWeightQuantMode) {
            if (idx > 0) {
                if (idx == total - 1) {
                    supportedStr += " or ";
                } else {
                    supportedStr += ", ";
                }
            }
            supportedStr += std::to_string(mode);
            idx++;
        }
        OP_LOGE_FOR_INVALID_VALUE("aclnnMlaPrologV4", "weightQuantMode", std::to_string(weightQuantMode), supportedStr);
        return false;
    }
    return true;
}

bool CheckKvCacheQuantModeValidity(int64_t weightQuantMode, int64_t kvCacheQuantMode)
{
    std::map<int64_t, std::set<int64_t>> supportedKvQuantMode;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        supportedKvQuantMode = {
            {0LL, {0LL}},           {1LL, {0LL, 2LL, 3LL}}, {2LL, {0LL, 1LL, 3LL}},
            {3LL, {0LL, 1LL, 3LL}}, {4LL, {0LL, 1LL, 3LL}}, {5LL, {0LL, 1LL, 3LL}},
        };
    } else {
        supportedKvQuantMode = {
            {0LL, {0LL}},
            {1LL, {0LL, 2LL, 3LL}},
            {2LL, {0LL, 1LL, 3LL}},
        };
    }
    auto it = supportedKvQuantMode.find(weightQuantMode);
    if (it == supportedKvQuantMode.end()) {
        return true; // weightQuantMode itself is invalid, already checked by CheckWeightQuantModeValidity
    }
    if (it->second.find(kvCacheQuantMode) == it->second.end()) {
        std::string supportedStr;
        const auto &modes = it->second;
        size_t total = modes.size();
        size_t idx = 0;
        for (auto mode : modes) {
            if (idx > 0) {
                if (idx == total - 1) {
                    supportedStr += " or ";
                } else {
                    supportedStr += ", ";
                }
            }
            supportedStr += std::to_string(mode);
            idx++;
        }
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnMlaPrologV4", "kvCacheQuantMode", std::to_string(kvCacheQuantMode),
            "When weightQuantMode==" + std::to_string(weightQuantMode) + ", must be within " + supportedStr);
        return false;
    }
    return true;
}

bool CheckQueryQuantModeValidity(int64_t queryQuantMode)
{
    std::set<int64_t> supportedQueryQuantMode = {0LL, 1LL};
    if (supportedQueryQuantMode.find(queryQuantMode) == supportedQueryQuantMode.end()) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMlaPrologV4", "queryQuantMode", std::to_string(queryQuantMode), "0 or 1");
        return false;
    }
    return true;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMlaPrologV4WeightNzGetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk,
    const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin, const aclTensor *ropeCos, aclTensor *kvCacheRef, aclTensor *krCacheRef,
    const aclTensor *cacheIndexOptional, const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional, const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional, const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional, const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional, const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv, char *cacheModeOptional, int64_t weightQuantMode, int64_t kvCacheQuantMode,
    int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize, double qcQrScale,
    double kcScale, bool doRope, const aclTensor *queryOut, const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional, const aclTensor *queryNormOutOptional,
    const aclTensor *dequantScaleQNormOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const int WEIGHT_QUANT_MODE_NO_QUANT = 0;
    const int WEIGHT_QUANT_MODE_PARTIAL_QUANT = 1;
    const int WEIGHT_QUANT_MODE_FULL_QUANT = 2;
    const int WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT = 3;
    const int WEIGHT_QUANT_MODE_FULL_QUANT_FP8 = 4;
    const int WEIGHT_QUANT_MODE_FULL_QUANT_HIF8 = 5;
    const int KV_CACHE_QUANT_MODE_NO_QUANT = 0;
    const int KV_CACHE_QUANT_MODE_PER_TENSOR = 1;
    const int KV_CACHE_QUANT_MODE_PER_CHANNEL = 2;
    const int KV_CACHE_QUANT_MODE_PER_TILE = 3;
    if (!CheckWeightQuantModeValidity(weightQuantMode)) {
        return ge::GRAPH_FAILED;
    };
    if (!CheckKvCacheQuantModeValidity(weightQuantMode, kvCacheQuantMode)) {
        return ge::GRAPH_FAILED;
    };
    if (!CheckQueryQuantModeValidity(queryQuantMode)) {
        return ge::GRAPH_FAILED;
    };

    // do_rope 作为 attr 透传 inner，tiling 通过 do_rope attr 判断 RoPE 开关；
    // doRope=true 时 ropeSin/ropeCos 不允许为空（null 或空 tensor）；
    // doRope=false 时 ropeSin/ropeCos 必须同时为空，并利用 TensorHolder 将 null 转为空 tensor 传入 inner，
    std::unique_ptr<TensorHolder> ropeSinHolder;
    std::unique_ptr<TensorHolder> ropeCosHolder;
    auto IsRopeInputEmpty = [](const aclTensor *rope) {
        return rope == nullptr || rope->GetViewShape().GetShapeSize() == 0;
    };
    if (doRope) {
        if (IsRopeInputEmpty(ropeSin)) {
            OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV4", "ropeSin");
            return ge::GRAPH_FAILED;
        }
        if (IsRopeInputEmpty(ropeCos)) {
            OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV4", "ropeCos");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (!IsRopeInputEmpty(ropeSin)) {
            OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV4", "ropeSin");
            return ge::GRAPH_FAILED;
        }
        if (!IsRopeInputEmpty(ropeCos)) {
            OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV4", "ropeCos");
            return ge::GRAPH_FAILED;
        }
        ropeSinHolder = std::make_unique<TensorHolder>(ropeSin, aclDataType::ACL_BF16, std::string("ropeSin"));
        ropeCosHolder = std::make_unique<TensorHolder>(ropeCos, aclDataType::ACL_BF16, std::string("ropeCos"));
    }
    auto dequantScaleQNopeHolder =
        TensorHolder(dequantScaleQNopeOutOptional, aclDataType::ACL_FLOAT, std::string("dequantScaleQNopeOut"));
    aclDataType queryNormDataType =
        weightQuantMode == WEIGHT_QUANT_MODE_NO_QUANT ? aclDataType::ACL_BF16 : aclDataType::ACL_INT8;
    aclDataType dequantScaleQNormDataType =
        weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT ? aclDataType::ACL_FLOAT8_E8M0 : aclDataType::ACL_FLOAT;
    if (weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT || weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT_FP8) {
        queryNormDataType = aclDataType::ACL_FLOAT8_E4M3FN;
    } else if (weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT_HIF8) {
        queryNormDataType = aclDataType::ACL_HIFLOAT8;
    }
    auto queryNormHolder = TensorHolder(queryNormOutOptional, queryNormDataType, std::string("queryNormOut"));
    auto dequantScaleQNormHolder =
        TensorHolder(dequantScaleQNormOutOptional, dequantScaleQNormDataType, std::string("dequantScaleQNormOut"));
    if (dequantScaleQNopeOutOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV4", "dequantScaleQNopeOut");
        return ge::GRAPH_FAILED;
    }
    if (queryNormOutOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV4", "queryNormOut");
        return ge::GRAPH_FAILED;
    }
    if (dequantScaleQNormOutOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV4", "dequantScaleQNormOut");
        return ge::GRAPH_FAILED;
    }
    // weightQuantMode == 2,4,5:全量化场景(int8,fp8,hif8)
    // weightQuantMode == 3:mxfp8全量化场景
    // kvCacheQuantMode == 1:KV_PER_TENSOR量化场景
    if (!dequantScaleQNopeHolder.CheckTensorConditionalNotNull((weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT ||
                                                                weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT ||
                                                                weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT_FP8 ||
                                                                weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT_HIF8) &&
                                                               kvCacheQuantMode == KV_CACHE_QUANT_MODE_PER_TENSOR)) {
        return ge::GRAPH_FAILED;
    }
    bool queryNormFlag = queryNormHolder.IsTensorNotNull();
    // weightQuantMode != 0:量化场景
    if (!dequantScaleQNormHolder.CheckTensorConditionalNotNull(weightQuantMode != WEIGHT_QUANT_MODE_NO_QUANT &&
                                                               queryNormFlag)) {
        return ge::GRAPH_FAILED;
    }

    return aclnnInnerMlaPrologV3GetWorkspaceSize(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
        kvCacheRef, krCacheRef, cacheIndexOptional, dequantScaleXOptional, dequantScaleWDqOptional,
        dequantScaleWUqQrOptional, dequantScaleWDkvKrOptional, quantScaleCkvOptional, quantScaleCkrOptional,
        smoothScalesCqOptional, actualSeqLenOptional, kNopeClipAlphaOptional, rmsnormEpsilonCq, rmsnormEpsilonCkv,
        cacheModeOptional, queryNormFlag, weightQuantMode, kvCacheQuantMode, queryQuantMode, ckvkrRepoMode,
        quantScaleRepoMode, tileSize, qcQrScale, kcScale, doRope, queryOut, queryRopeOut, dequantScaleQNopeOutOptional,
        queryNormOutOptional, dequantScaleQNormOutOptional, workspaceSize, executor);
}

aclnnStatus aclnnMlaPrologV4WeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerMlaPrologV3(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
