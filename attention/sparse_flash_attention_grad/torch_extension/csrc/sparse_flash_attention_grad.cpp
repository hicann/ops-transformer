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
 * \file sparse_flash_attention_grad.cpp
 * \brief PTA wrapper of aclnnSparseFlashAttentionGradV2 (SFAG, MLA OSS Sink).
 */

#include <cctype>
#include <string>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t SFAG_EMPTY_NUMEL = 0;
constexpr int64_t SFAG_MIN_SPARSE_BLOCK_SIZE = 1;
constexpr int64_t SFAG_MAX_SPARSE_BLOCK_SIZE = 64;

bool SfagIsAscend950()
{
    const char *socName = aclrtGetSocName();
    return socName != nullptr && std::strstr(socName, "Ascend950") != nullptr;
}

bool SfagLayoutValid(const std::string &layout)
{
    std::string layoutUpper = layout;
    for (auto &c : layoutUpper) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return layoutUpper == "BSND" || layoutUpper == "TND";
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> SparseFlashAttentionGrad(
    const at::Tensor &query, const at::Tensor &key, const c10::optional<at::Tensor> &value,
    const at::Tensor &sparseIndices, const at::Tensor &dOut, const at::Tensor &out, const at::Tensor &softmaxMax,
    const at::Tensor &softmaxSum, const c10::optional<at::Tensor> &sinks, double scaleValue, int64_t sparseBlockSize,
    const c10::optional<at::Tensor> &queryRope, const c10::optional<at::Tensor> &keyRope,
    const c10::optional<at::Tensor> &actualSeqLenQ, const c10::optional<at::Tensor> &actualSeqLenKv, std::string layout,
    int64_t sparseMode, int64_t winLeft, int64_t winRight, int64_t attentionMode)
{
    // attention_mode 为接口占位参数，暂不参与计算（仅收参，不转发到 aclnn/OpDef/tiling）
    (void)attentionMode;
    // ---- 入参校验 ----
    TORCH_CHECK((query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf),
                "query should be bfloat16 or float16, current dtype is: ", query.scalar_type());
    // value 为可选输入：KV merge 场景（value 不传）时跳过 value 的 dtype 校验。
    TORCH_CHECK((key.scalar_type() == query.scalar_type() &&
                 (!value.has_value() || value->scalar_type() == query.scalar_type()) &&
                 dOut.scalar_type() == query.scalar_type() && out.scalar_type() == query.scalar_type()),
                "query/key/value/d_out/out should have the same dtype, current query dtype: ", query.scalar_type());
    TORCH_CHECK((sparseIndices.scalar_type() == at::kInt),
                "sparse_indices should be int32, current dtype is: ", sparseIndices.scalar_type());
    TORCH_CHECK((softmaxMax.scalar_type() == at::kFloat && softmaxSum.scalar_type() == at::kFloat),
                "softmax_max/softmax_sum should be float32, current dtype is: ", softmaxMax.scalar_type());
    TORCH_CHECK(SfagLayoutValid(layout), "layout should be BSND or TND, current is: ", layout);
    TORCH_CHECK((sparseBlockSize == SFAG_MIN_SPARSE_BLOCK_SIZE || sparseBlockSize == 8 || sparseBlockSize == 16 ||
                 sparseBlockSize == 32 || sparseBlockSize == SFAG_MAX_SPARSE_BLOCK_SIZE),
                "sparse_block_size should be in {1,8,16,32,64}, current is: ", sparseBlockSize);

    const at::Tensor &sinksConst = sinks.value_or(at::Tensor());
    if (sinksConst.defined()) {
        TORCH_CHECK((sinksConst.scalar_type() == at::kFloat),
                    "sinks should be float32, current dtype is: ", sinksConst.scalar_type());
    }
    if (queryRope.has_value()) {
        TORCH_CHECK((queryRope->scalar_type() == query.scalar_type()),
                    "query_rope should have the same dtype as query, current dtype is: ", queryRope->scalar_type());
    }
    if (keyRope.has_value()) {
        TORCH_CHECK((keyRope->scalar_type() == query.scalar_type()),
                    "key_rope should have the same dtype as query, current dtype is: ", keyRope->scalar_type());
    }

    const at::Tensor &queryRopeConst = queryRope.value_or(at::Tensor());
    const at::Tensor &keyRopeConst = keyRope.value_or(at::Tensor());
    const at::Tensor &actualSeqLenQConst = actualSeqLenQ.value_or(at::Tensor());
    const at::Tensor &actualSeqLenKvConst = actualSeqLenKv.value_or(at::Tensor());
    // KV merge（value 传 None 时按 value=key 处理）：PTA 层 value 为空 tensor（undefined）时
    // dValue 输出空 tensor [0]。arch35/arch22 host tiling 均支持：
    // value==nullptr 时 tmpData.kvMerge=true；arch35 要求 dValueOut 为空（shape [0]），
    // arch22 直接返回空 dv。
    const at::Tensor &valueConst = value.value_or(at::Tensor());

    // ---- 申请输出张量（DeviceGuard 必须包住输出申请） ----
    at::Tensor dq{nullptr};
    at::Tensor dk{nullptr};
    at::Tensor dv{nullptr};
    at::Tensor dqRope{nullptr};
    at::Tensor dkRope{nullptr};
    at::Tensor dSinks{nullptr};
    {
        auto localDevice = c10::Device(query.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);

        dq = at::empty_like(query);
        dk = at::empty_like(key);
        dv = valueConst.defined() ? at::empty_like(valueConst) : at::empty({SFAG_EMPTY_NUMEL}, query.options());
        dqRope =
            queryRopeConst.defined() ? at::empty_like(queryRopeConst) : at::empty({SFAG_EMPTY_NUMEL}, query.options());
        dkRope = keyRopeConst.defined() ? at::empty_like(keyRopeConst) : at::empty({SFAG_EMPTY_NUMEL}, query.options());
        dSinks = sinksConst.defined() ? at::empty_like(sinksConst) :
                                        at::empty({SFAG_EMPTY_NUMEL}, query.options().dtype(at::kFloat));
    }

    char *layoutPtr = const_cast<char *>(layout.c_str());
    // aclnn 层接口仍保留 bool deterministic 位置参数，但 host tiling(arch35/arch22) 实际用的是
    // context_->GetDeterministic()，即 torch.use_deterministic_algorithms(True/False) 下推的
    // ACL_OPT_DETERMINISTIC，本参数在 host 层不生效（模板切换靠全局 flag）。
    // 必须传左值 bool 变量：ACLNN_CMD 宏 ConvertTypes 要求参数可绑定非 const 左值引用，字面量 true 编不过。
    bool deterministic = true;
    ACLNN_CMD(aclnnSparseFlashAttentionGradV2, query, key, valueConst, sparseIndices, dOut, out, softmaxMax, softmaxSum,
              sinksConst, actualSeqLenQConst, actualSeqLenKvConst, queryRopeConst, keyRopeConst, scaleValue,
              sparseBlockSize, layoutPtr, sparseMode, winLeft, winRight, deterministic, dq, dk, dv, dqRope, dkRope,
              dSinks);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(dq, dk, dv, dqRope,
                                                                                              dkRope, dSinks);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_flash_attention_grad", &SparseFlashAttentionGrad, "sparse_flash_attention_grad");
}
} // namespace op_api
