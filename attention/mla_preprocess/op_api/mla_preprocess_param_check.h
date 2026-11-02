/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_ATTENTION_MLA_PREPROCESS_OP_API_PARAM_CHECK_H
#define OPS_TRANSFORMER_ATTENTION_MLA_PREPROCESS_OP_API_PARAM_CHECK_H

#include <cstdint>
#include <initializer_list>

#include "aclnn/acl_meta.h"
#include "opdev/op_log.h"

namespace MlaPreprocessApi {

constexpr int64_t kTokenNumMax = 256;
constexpr int64_t kHiddenMin = 2048;
constexpr int64_t kHiddenMax = 10240;
constexpr int64_t kHiddenAlign = 256;
constexpr int64_t kHeadNumMax = 128;
constexpr int64_t kQLoraMin = 32;
constexpr int64_t kQLoraMax = 4096;
constexpr int64_t kQLoraAlign = 32;
constexpr int64_t kNopeMin = 16;
constexpr int64_t kNopeMax = 256;
constexpr int64_t kNopeAlign = 16;
constexpr int64_t kSupportedRopeDim = 64;
constexpr int64_t kKvMerged = 576;
constexpr int64_t kKvSplit = 512;
constexpr int64_t kNzHeight0 = 16;
constexpr int64_t kCacheModeMax = 3;

struct TensorView {
    const aclTensor* tensor = nullptr;
    int64_t* dims = nullptr;
    uint64_t rank = 0;
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_UNDEFINED;
    bool ok = false;

    explicit TensorView(const aclTensor* input)
        : tensor(input)
    {
        if (input == nullptr) {
            return;
        }
        ok = aclGetViewShape(input, &dims, &rank) == ACLNN_SUCCESS && dims != nullptr &&
             aclGetDataType(input, &dtype) == ACLNN_SUCCESS && aclGetFormat(input, &format) == ACLNN_SUCCESS;
    }
};

inline aclnnStatus Invalid(const char* message)
{
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s", message);
    return ACLNN_ERR_PARAM_INVALID;
}

inline aclnnStatus NullTensor(const char* name)
{
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s is nullptr.", name);
    return ACLNN_ERR_PARAM_NULLPTR;
}

inline bool IsNd(aclFormat format)
{
    return format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW || format == ACL_FORMAT_NHWC ||
           format == ACL_FORMAT_NCL;
}

inline bool IsNz(aclFormat format)
{
    return format == ACL_FORMAT_FRACTAL_NZ || format == ACL_FORMAT_FRACTAL_NZ_C0_16 ||
           format == ACL_FORMAT_FRACTAL_NZ_C0_32 || format == ACL_FORMAT_FRACTAL_NZ_C0_2 ||
           format == ACL_FORMAT_FRACTAL_NZ_C0_4 || format == ACL_FORMAT_FRACTAL_NZ_C0_8;
}

inline bool IsInputDtype(aclDataType dtype)
{
    return dtype == ACL_FLOAT16 || dtype == ACL_BF16;
}

inline bool SameShape(const TensorView& view, const std::initializer_list<int64_t>& expect)
{
    if (!view.ok || view.rank != expect.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t dim : expect) {
        if (view.dims[index] != dim) {
            return false;
        }
        ++index;
    }
    return true;
}

inline bool DecodeNzMatrix(const TensorView& view, int64_t& rows, int64_t& cols)
{
    if (!view.ok || !IsNz(view.format)) {
        return false;
    }
    if (view.rank == 2 && view.dims[0] > 0 && view.dims[1] > 0) {
        rows = view.dims[0];
        cols = view.dims[1];
        return true;
    }
    if (view.rank != 4) {
        return false;
    }
    const int64_t elemSize = view.dtype == ACL_INT8 ? 1 : 2;
    const int64_t width0 = 32 / elemSize;
    if (view.dims[2] != kNzHeight0 || view.dims[3] != width0 || view.dims[0] <= 0 || view.dims[1] <= 0) {
        return false;
    }
    rows = view.dims[1] * kNzHeight0;
    cols = view.dims[0] * width0;
    return true;
}

inline bool MatchKvCache(const TensorView& view, int64_t cacheMode, aclDataType inputDtype, int64_t& blockNum,
                         int64_t& blockSize)
{
    if (!view.ok || view.rank != 4) {
        return false;
    }
    const int64_t lastDim = cacheMode == 0 ? kKvMerged : kKvSplit;
    const aclDataType expectDtype = cacheMode == 2 ? ACL_INT8 : inputDtype;
    if (view.dtype != expectDtype) {
        return false;
    }
    if (cacheMode == 0 || cacheMode == 1) {
        if (!IsNd(view.format) || view.dims[2] != 1 || view.dims[3] != lastDim || view.dims[0] <= 0 ||
            view.dims[1] <= 0) {
            return false;
        }
        blockNum = view.dims[0];
        blockSize = view.dims[1];
        return true;
    }
    if (!IsNz(view.format)) {
        return false;
    }
    if (view.dims[2] == 1 && view.dims[3] == lastDim && view.dims[0] > 0 && view.dims[1] > 0) {
        blockNum = view.dims[0];
        blockSize = view.dims[1];
        return true;
    }
    const int64_t c1 = view.dims[1];
    const int64_t c0 = view.dims[3];
    if (view.dims[0] <= 0 || view.dims[2] <= 0 || c1 <= 0 || (c0 != 16 && c0 != 32) || c1 * c0 != lastDim) {
        return false;
    }
    blockNum = view.dims[0];
    blockSize = view.dims[2];
    return true;
}

inline bool MatchRopeCache(const TensorView& view, int64_t cacheMode, int64_t blockNum, int64_t blockSize)
{
    if (!view.ok || view.rank != 4) {
        return false;
    }
    if (cacheMode == 1) {
        return IsNd(view.format) && SameShape(view, {blockNum, blockSize, 1, kSupportedRopeDim});
    }
    if (!IsNz(view.format)) {
        return false;
    }
    if (SameShape(view, {blockNum, blockSize, 1, kSupportedRopeDim})) {
        return true;
    }
    const int64_t c1 = view.dims[1];
    const int64_t c0 = view.dims[3];
    return view.dims[0] == blockNum && view.dims[2] == blockSize && c1 > 0 && (c0 == 16 || c0 == 32) &&
           c1 * c0 == kSupportedRopeDim;
}

inline aclnnStatus CheckSameDtypeNd(const TensorView& view, const char* name, aclDataType dtype,
                                    const std::initializer_list<int64_t>& shape)
{
    if (!view.ok) {
        return Invalid("failed to read tensor shape, dtype or format.");
    }
    if (view.dtype != dtype || !IsNd(view.format) || !SameShape(view, shape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s shape, dtype or format is not supported.", name);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

inline aclnnStatus CheckMlaPreprocessApiParams(
    const aclTensor* input, const aclTensor* gamma0, const aclTensor* beta0, const aclTensor* quantScale0,
    const aclTensor* quantOffset0, const aclTensor* wdqkv, const aclTensor* deScale0, const aclTensor* bias0,
    const aclTensor* gamma1, const aclTensor* beta1, const aclTensor* quantScale1, const aclTensor* quantOffset1,
    const aclTensor* wuq, const aclTensor* deScale1, const aclTensor* bias1, const aclTensor* gamma2,
    const aclTensor* cos, const aclTensor* sin, const aclTensor* wuk, const aclTensor* kvCache,
    const aclTensor* kvCacheRope, const aclTensor* slotMapping, const aclTensor* ctkvScale, const aclTensor* qNopeScale,
    int64_t qRopeDim, int64_t kRopeDim, int64_t qRotaryCoeff, int64_t kRotaryCoeff, bool transposeWdq,
    bool transposeWuq, bool transposeWuk, int64_t cacheMode, int64_t quantMode, int64_t wdkvSplitCount,
    const aclTensor* qOut, const aclTensor* kvCacheOut, const aclTensor* qRopeOut, const aclTensor* krCacheOut,
    const aclTensor* qDownOut, bool qDownOutFlag)
{
    if (qRopeDim != 64 || kRopeDim != 64) {
        return Invalid("qRopeDim and kRopeDim only support 64.");
    }
    if (qRotaryCoeff != kRotaryCoeff || qRotaryCoeff != 2) {
        return Invalid("qRotaryCoeff and kRotaryCoeff only support 2.");
    }
    if (!transposeWdq || !transposeWuq || !transposeWuk) {
        return Invalid("transposeWdq, transposeWuq and transposeWuk only support true.");
    }
    if (wdkvSplitCount != 1) {
        return Invalid("wdkvSplitCount only supports 1.");
    }
    if (cacheMode < 0 || cacheMode > kCacheModeMax) {
        return Invalid("cacheMode only supports 0, 1, 2 and 3.");
    }
    if (quantMode != 0) {
        return Invalid("quantMode only supports 0.");
    }

    if (input == nullptr) {
        return NullTensor("input");
    }
    if (gamma0 == nullptr || beta0 == nullptr || quantScale0 == nullptr || quantOffset0 == nullptr ||
        wdqkv == nullptr || deScale0 == nullptr || bias0 == nullptr || gamma1 == nullptr || beta1 == nullptr ||
        quantScale1 == nullptr || quantOffset1 == nullptr || wuq == nullptr || deScale1 == nullptr ||
        bias1 == nullptr || gamma2 == nullptr || wuk == nullptr || kvCache == nullptr || slotMapping == nullptr ||
        qOut == nullptr || kvCacheOut == nullptr) {
        return NullTensor("required tensor");
    }

    const TensorView inputView(input);
    if (!inputView.ok || !IsInputDtype(inputView.dtype) || !IsNd(inputView.format) || inputView.rank != 2) {
        return Invalid("input shape must be [tokenNum, hiddenSize], dtype FLOAT16 or BFLOAT16, format ND.");
    }
    const int64_t tokenNum = inputView.dims[0];
    const int64_t hiddenSize = inputView.dims[1];
    if (tokenNum < 0 || tokenNum > kTokenNumMax) {
        return Invalid("tokenNum must be in range [0, 256].");
    }
    if (hiddenSize < kHiddenMin || hiddenSize > kHiddenMax || (hiddenSize % kHiddenAlign) != 0) {
        return Invalid("hiddenSize must be in range [2048, 10240] and a multiple of 256.");
    }
    const aclDataType dtype = inputView.dtype;

    aclnnStatus status = CheckSameDtypeNd(TensorView(gamma0), "gamma0", dtype, {hiddenSize});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSameDtypeNd(TensorView(beta0), "beta0", dtype, {hiddenSize});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSameDtypeNd(TensorView(quantScale0), "quantScale0", dtype, {1});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSameDtypeNd(TensorView(quantOffset0), "quantOffset0", ACL_INT8, {1});
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    const TensorView deScale0View(deScale0);
    const aclDataType deScaleDtype = dtype == ACL_FLOAT16 ? ACL_INT64 : ACL_FLOAT;
    if (!deScale0View.ok || deScale0View.dtype != deScaleDtype || !IsNd(deScale0View.format) ||
        deScale0View.rank != 1) {
        return Invalid("deScale0 dtype must be INT64 for FLOAT16 input and FLOAT for BFLOAT16 input.");
    }
    const int64_t rows0 = deScale0View.dims[0];
    const int64_t qLoraRank = rows0 - kKvMerged;
    if (qLoraRank < kQLoraMin || qLoraRank > kQLoraMax || (qLoraRank % kQLoraAlign) != 0) {
        return Invalid("qLoraRank must be in range [32, 4096] and aligned to 32.");
    }
    status = CheckSameDtypeNd(TensorView(bias0), "bias0", ACL_INT32, {rows0});
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    const TensorView wdqkvView(wdqkv);
    int64_t wdqkvRows = 0;
    int64_t wdqkvCols = 0;
    const bool wdqkvDtypeOk = wdqkvView.ok && (wdqkvView.dtype == ACL_INT8 || wdqkvView.dtype == dtype);
    if (!wdqkvDtypeOk || !DecodeNzMatrix(wdqkvView, wdqkvRows, wdqkvCols) || wdqkvRows != rows0 ||
        wdqkvCols != hiddenSize) {
        return Invalid("wdqkv dtype must be INT8 or the same as input, format NZ, "
                       "shape [qLoraRank + 576, hiddenSize].");
    }

    status = CheckSameDtypeNd(TensorView(gamma1), "gamma1", dtype, {qLoraRank});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSameDtypeNd(TensorView(beta1), "beta1", dtype, {qLoraRank});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSameDtypeNd(TensorView(quantScale1), "quantScale1", dtype, {1});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSameDtypeNd(TensorView(quantOffset1), "quantOffset1", ACL_INT8, {1});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSameDtypeNd(TensorView(gamma2), "gamma2", dtype, {kKvSplit});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSameDtypeNd(TensorView(slotMapping), "slotMapping", ACL_INT32, {tokenNum});
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    const TensorView wukView(wuk);
    if (!wukView.ok || wukView.dtype != dtype || (!IsNd(wukView.format) && !IsNz(wukView.format)) ||
        wukView.rank != 3 || wukView.dims[2] != kKvSplit) {
        return Invalid("wuk shape must be [headNum, nopeDim, 512], dtype the same as input, format ND or NZ.");
    }
    const int64_t headNum = wukView.dims[0];
    const int64_t nopeDim = wukView.dims[1];
    if (headNum <= 0 || headNum > kHeadNumMax) {
        return Invalid("headNum must be in range [1, 128].");
    }
    if (nopeDim < kNopeMin || nopeDim > kNopeMax || (nopeDim % kNopeAlign) != 0) {
        return Invalid("nopeDim must be in range [16, 256] and aligned to 16.");
    }
    const int64_t wuqRows = headNum * (nopeDim + kSupportedRopeDim);
    const TensorView wuqView(wuq);
    int64_t wuqDecodedRows = 0;
    int64_t wuqDecodedCols = 0;
    const bool wuqDtypeOk = wuqView.ok && (wuqView.dtype == ACL_INT8 || wuqView.dtype == dtype);
    if (!wuqDtypeOk || !DecodeNzMatrix(wuqView, wuqDecodedRows, wuqDecodedCols) || wuqDecodedRows != wuqRows ||
        wuqDecodedCols != qLoraRank) {
        return Invalid("wuq dtype must be INT8 or the same as input, format NZ, "
                       "shape [headNum * (nopeDim + 64), qLoraRank].");
    }
    status = CheckSameDtypeNd(TensorView(deScale1), "deScale1", deScaleDtype, {wuqRows});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSameDtypeNd(TensorView(bias1), "bias1", ACL_INT32, {wuqRows});
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    if (cos != nullptr || sin != nullptr) {
        status = CheckSameDtypeNd(TensorView(cos), "cos", dtype, {tokenNum, kSupportedRopeDim});
        if (status != ACLNN_SUCCESS) {
            return status;
        }
        status = CheckSameDtypeNd(TensorView(sin), "sin", dtype, {tokenNum, kSupportedRopeDim});
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }

    const int64_t qOutLast = cacheMode == 0 ? kKvMerged : kKvSplit;
    const aclDataType qOutDtype = cacheMode == 2 ? ACL_INT8 : dtype;
    status = CheckSameDtypeNd(TensorView(qOut), "qOut", qOutDtype, {tokenNum, headNum, qOutLast});
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    const TensorView kvCacheView(kvCache);
    int64_t blockNum = 0;
    int64_t blockSize = 0;
    if (!MatchKvCache(kvCacheView, cacheMode, dtype, blockNum, blockSize)) {
        return Invalid("kvCache shape, dtype or format does not match cacheMode.");
    }
    const TensorView kvCacheOutView(kvCacheOut);
    if (!kvCacheOutView.ok || kvCacheOutView.dtype != kvCacheView.dtype ||
        kvCacheOutView.format != kvCacheView.format || kvCacheOutView.rank != kvCacheView.rank) {
        return Invalid("kvCacheOut shape, dtype or format does not match kvCache.");
    }
    for (uint64_t dim = 0; dim < kvCacheView.rank; ++dim) {
        if (kvCacheOutView.dims[dim] != kvCacheView.dims[dim]) {
            return Invalid("kvCacheOut shape, dtype or format does not match kvCache.");
        }
    }

    if (cacheMode != 0) {
        if (qRopeOut == nullptr || krCacheOut == nullptr || kvCacheRope == nullptr) {
            return NullTensor("qRopeOut, krCacheOut or kvCacheRope");
        }
        status = CheckSameDtypeNd(TensorView(qRopeOut), "qRopeOut", dtype, {tokenNum, headNum, kSupportedRopeDim});
        if (status != ACLNN_SUCCESS) {
            return status;
        }
        const TensorView kvCacheRopeView(kvCacheRope);
        if (kvCacheRopeView.dtype != dtype || !MatchRopeCache(kvCacheRopeView, cacheMode, blockNum, blockSize)) {
            return Invalid("kvCacheRope shape, dtype or format does not match cacheMode.");
        }
        const TensorView krCacheOutView(krCacheOut);
        if (!krCacheOutView.ok || krCacheOutView.dtype != kvCacheRopeView.dtype ||
            krCacheOutView.format != kvCacheRopeView.format || krCacheOutView.rank != kvCacheRopeView.rank) {
            return Invalid("krCacheOut shape, dtype or format does not match kvCacheRope.");
        }
        for (uint64_t dim = 0; dim < kvCacheRopeView.rank; ++dim) {
            if (krCacheOutView.dims[dim] != kvCacheRopeView.dims[dim]) {
                return Invalid("krCacheOut shape, dtype or format does not match kvCacheRope.");
            }
        }
    }

    if (cacheMode == 2) {
        if (ctkvScale == nullptr || qNopeScale == nullptr) {
            return NullTensor("ctkvScale or qNopeScale");
        }
        status = CheckSameDtypeNd(TensorView(ctkvScale), "ctkvScale", dtype, {1});
        if (status != ACLNN_SUCCESS) {
            return status;
        }
        status = CheckSameDtypeNd(TensorView(qNopeScale), "qNopeScale", dtype, {1});
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    } else {
        if (ctkvScale != nullptr) {
            status = CheckSameDtypeNd(TensorView(ctkvScale), "ctkvScale", dtype, {1});
            if (status != ACLNN_SUCCESS) {
                return status;
            }
        }
        if (qNopeScale != nullptr) {
            status = CheckSameDtypeNd(TensorView(qNopeScale), "qNopeScale", dtype, {1});
            if (status != ACLNN_SUCCESS) {
                return status;
            }
        }
    }

    if (qDownOutFlag && qDownOut == nullptr) {
        return NullTensor("qDownOut");
    }
    if (qDownOut != nullptr) {
        status = CheckSameDtypeNd(TensorView(qDownOut), "qDownOut", dtype, {tokenNum, qLoraRank});
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    return ACLNN_SUCCESS;
}

} // namespace MlaPreprocessApi

#endif
