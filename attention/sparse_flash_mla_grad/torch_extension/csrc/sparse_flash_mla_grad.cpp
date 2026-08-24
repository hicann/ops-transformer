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
 * \file sparse_flash_mla_grad.cpp
 * \brief
 */

#include <cstring>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t SMLAG_METADATA_SIZE = 1024;

bool IsAscend950()
{
    const char *socName = aclrtGetSocName();
    return socName != nullptr && std::strstr(socName, "Ascend950") != nullptr;
}

at::Tensor SparseFlashMlaGradMetadata(
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensOriKv, const c10::optional<at::Tensor> &cuSeqlensCmpKv,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedOriKv,
    const c10::optional<at::Tensor> &sequsedCmpKv, const c10::optional<at::Tensor> &cmpResidualKv,
    const c10::optional<at::Tensor> &oriTopkLength, const c10::optional<at::Tensor> &cmpTopkLength, int64_t batchSize,
    int64_t maxSeqlenQ, int64_t maxSeqlenOriKv, int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk,
    int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
    c10::string_view layoutQ, c10::string_view layoutKv, bool hasOriKv, bool hasCmpKv)
{
    if (!IsAscend950()) {
        return at::Tensor();
    }

    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (cuSeqlensOriKv.has_value()) {
        outputDevice = cuSeqlensOriKv.value().device();
    } else if (cuSeqlensCmpKv.has_value()) {
        outputDevice = cuSeqlensCmpKv.value().device();
    } else if (sequsedQ.has_value()) {
        outputDevice = sequsedQ.value().device();
    } else if (sequsedOriKv.has_value()) {
        outputDevice = sequsedOriKv.value().device();
    } else if (sequsedCmpKv.has_value()) {
        outputDevice = sequsedCmpKv.value().device();
    } else if (cmpResidualKv.has_value()) {
        outputDevice = cmpResidualKv.value().device();
    } else if (oriTopkLength.has_value()) {
        outputDevice = oriTopkLength.value().device();
    } else if (cmpTopkLength.has_value()) {
        outputDevice = cmpTopkLength.value().device();
    }

    at::Tensor output = torch::empty({SMLAG_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = get_valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensOriKvVal = get_valid_tensor(cuSeqlensOriKv, outputDevice);
    auto cuSeqlensCmpKvVal = get_valid_tensor(cuSeqlensCmpKv, outputDevice);
    auto sequsedQVal = get_valid_tensor(sequsedQ, outputDevice);
    auto sequsedOriKvVal = get_valid_tensor(sequsedOriKv, outputDevice);
    auto sequsedCmpKvVal = get_valid_tensor(sequsedCmpKv, outputDevice);
    auto cmpResidualKvVal = get_valid_tensor(cmpResidualKv, outputDevice);
    auto oriTopkLengthVal = get_valid_tensor(oriTopkLength, outputDevice);
    auto cmpTopkLengthVal = get_valid_tensor(cmpTopkLength, outputDevice);

    // convert str
    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    ACLNN_CMD(aclnnSparseFlashMlaGradMetadata, cuSeqlensQVal, cuSeqlensOriKvVal, cuSeqlensCmpKvVal, sequsedQVal,
              sequsedOriKvVal, sequsedCmpKvVal, cmpResidualKvVal, oriTopkLengthVal, cmpTopkLengthVal, numHeadsQ,
              numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, oriTopk, cmpTopk, cmpRatio,
              oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr, hasOriKv, hasCmpKv, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> SparseFlashMlaGrad(
    const at::Tensor &q, const at::Tensor &dout, const at::Tensor &attnOut, const at::Tensor &softmaxLse,
    const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices, const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv, const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, c10::optional<double> softmaxScale, c10::optional<int64_t> cmpRatio,
    c10::optional<int64_t> oriMaskMode, c10::optional<int64_t> cmpMaskMode, c10::optional<int64_t> oriWinLeft,
    c10::optional<int64_t> oriWinRight, c10::optional<c10::string_view> layoutQ,
    c10::optional<c10::string_view> layoutKv)
{
    const at::Tensor &oriKvConst = oriKv.value_or(at::Tensor());
    const at::Tensor &cmpKvConst = cmpKv.value_or(at::Tensor());
    const at::Tensor &oriSparseIndicesConst = oriSparseIndices.value_or(at::Tensor());
    const at::Tensor &cmpSparseIndicesConst = cmpSparseIndices.value_or(at::Tensor());
    const at::Tensor &cuSeqlensQConst = cuSeqlensQ.value_or(at::Tensor());
    const at::Tensor &cuSeqlensOriKvConst = cuSeqlensOriKv.value_or(at::Tensor());
    const at::Tensor &cuSeqlensCmpKvConst = cuSeqlensCmpKv.value_or(at::Tensor());
    const at::Tensor &sequsedQConst = sequsedQ.value_or(at::Tensor());
    const at::Tensor &sequsedOriKvConst = sequsedOriKv.value_or(at::Tensor());
    const at::Tensor &sequsedCmpKvConst = sequsedCmpKv.value_or(at::Tensor());
    const at::Tensor &cmpResidualKvConst = cmpResidualKv.value_or(at::Tensor());
    const at::Tensor &oriTopkLengthConst = oriTopkLength.value_or(at::Tensor());
    const at::Tensor &cmpTopkLengthConst = cmpTopkLength.value_or(at::Tensor());
    const at::Tensor &sinksConst = sinks.value_or(at::Tensor());
    at::Tensor metadataConst = metadata.value_or(at::Tensor());
    if (!IsAscend950() && metadataConst.defined()) {
        metadataConst = at::Tensor();
    }

    c10::string_view layoutQStrView = layoutQ.value_or("BSND");
    char *layoutQPtr = const_cast<char *>(layoutQStrView.data());

    c10::string_view layoutKvStrView = layoutKv.value_or("BSND");
    char *layoutKvPtr = const_cast<char *>(layoutKvStrView.data());

    at::Tensor dq = at::empty_like(q);
    at::Tensor dOriKv;
    at::Tensor dCmpKv;
    if (oriKvConst.defined()) {
        dOriKv = at::empty_like(oriKvConst);
    } else {
        dOriKv = at::empty({0}, q.options());
    }
    if (cmpKvConst.defined()) {
        dCmpKv = at::empty_like(cmpKvConst);
    } else {
        dCmpKv = at::empty({0}, q.options());
    }
    at::Tensor dSinks;
    if (sinksConst.defined()) {
        dSinks = at::empty_like(sinksConst);
    } else {
        dSinks = at::empty({0}, q.options().dtype(at::kFloat));
    }

    auto headDim = q.size(q.dim() - 1);

    double defaultSoftmax = 1.0 / std::sqrt(headDim);
    const double softmaxScaleConst = softmaxScale.value_or(defaultSoftmax);
    const int64_t cmpRatioConst = cmpRatio.value_or(1);
    const int64_t oriMaskModeConst = oriMaskMode.value_or(0);
    const int64_t cmpMaskModeConst = cmpMaskMode.value_or(0);
    const int64_t oriWinLeftConst = oriWinLeft.value_or(-1);
    const int64_t oriWinRightConst = oriWinRight.value_or(-1);

    at::Tensor oriSoftmaxL1Norm;
    if (oriSparseIndicesConst.defined()) {
        oriSoftmaxL1Norm = at::empty(oriSparseIndicesConst.sizes(), oriSparseIndicesConst.options().dtype(at::kFloat));
    } else {
        oriSoftmaxL1Norm = at::empty({0}, q.options().dtype(at::kFloat));
    }

    at::Tensor cmpSoftmaxL1Norm;
    if (cmpSparseIndicesConst.defined()) {
        cmpSoftmaxL1Norm = at::empty(cmpSparseIndicesConst.sizes(), cmpSparseIndicesConst.options().dtype(at::kFloat));
    } else {
        cmpSoftmaxL1Norm = at::empty({0}, q.options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnSparseFlashMlaGrad, q, dout, attnOut, softmaxLse, oriKvConst, cmpKvConst, oriSparseIndicesConst,
              cmpSparseIndicesConst, cuSeqlensQConst, cuSeqlensOriKvConst, cuSeqlensCmpKvConst, sequsedQConst,
              sequsedOriKvConst, sequsedCmpKvConst, cmpResidualKvConst, oriTopkLengthConst, cmpTopkLengthConst,
              sinksConst, metadataConst, softmaxScaleConst, cmpRatioConst, oriMaskModeConst, cmpMaskModeConst,
              oriWinLeftConst, oriWinRightConst, layoutQPtr, layoutKvPtr, dq, dOriKv, dCmpKv, dSinks, oriSoftmaxL1Norm,
              cmpSoftmaxL1Norm);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        dq, dOriKv, dCmpKv, dSinks, oriSoftmaxL1Norm, cmpSoftmaxL1Norm);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_flash_mla_grad_metadata", &SparseFlashMlaGradMetadata, "sparse_flash_mla_grad_metadata");
    m.def("sparse_flash_mla_grad", &SparseFlashMlaGrad, "sparse_flash_mla_grad");
}
} // namespace op_api
