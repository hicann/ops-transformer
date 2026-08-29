/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <string>
#include "aclnn_common.h"

namespace op_api {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
const int64_t QUANT_MODE_MXFP4 = 5;
const int64_t FLOAT4_E2M1 = 296;
const int64_t FLOAT8_E8M0 = 293;

inline TensorWrapper make_wrapper(const at::Tensor &tensor, aclDataType tensorAcltype)
{
    return {tensor, tensorAcltype};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> quant_flash_attn_grad(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &dout,
    const at::Tensor &attn_out, const at::Tensor &q_descale, const at::Tensor &k_descale, const at::Tensor &v_descale,
    const at::Tensor &do_descale, const at::Tensor &p_scale, const at::Tensor &ds_scale, const at::Tensor &softmax_lse,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &attn_mask,
    const c10::optional<at::Tensor> &metadata, int64_t quant_mode, double softmax_scale, int64_t mask_mode,
    int64_t win_left, int64_t win_right, int64_t max_seqlen_q, int64_t max_seqlen_kv, std::string layout_q,
    std::string layout_k)
{
    int64_t tSize = 0, nSize = 0, dSize = 0, sSize = 0, bSize = 0, skSize = 0;
    at::SmallVector<int64_t, 4> dqSize;
    at::SmallVector<int64_t, 4> dkSize;
    if (layout_q == "TND") {
        tSize = query.size(0);
        nSize = query.size(1);
        dSize = query.size(2);
        dqSize = {tSize, nSize, dSize};
    } else if (layout_q == "BSND") {
        bSize = query.size(0);
        sSize = query.size(1);
        skSize = key.size(1);
        nSize = query.size(2);
        dSize = query.size(3);
        dqSize = {bSize, sSize, nSize, dSize};
        dkSize = {bSize, skSize, nSize, dSize};
    } else {
        bSize = query.size(0);
        nSize = query.size(1);
        sSize = query.size(2);
        skSize = key.size(2);
        dSize = query.size(3);
        dqSize = {bSize, nSize, sSize, dSize};
        dkSize = {bSize, nSize, skSize, dSize};
    }
    at::SmallVector<int64_t, 1> dsinkSize = {nSize};

    at::Tensor dq = at::empty(dqSize, query.options().dtype(at::kBFloat16));
    at::Tensor dk = at::empty(dkSize, query.options().dtype(at::kBFloat16));
    at::Tensor dv = at::empty(dkSize, query.options().dtype(at::kBFloat16));
    at::Tensor dsink = at::empty(dsinkSize, query.options().dtype(at::kFloat));
    TORCH_CHECK(quant_mode == 0, "quant mode must be 0 now is: ", quant_mode);
    TensorWrapper q_wrapper = make_wrapper(query, ACL_HIFLOAT8);
    TensorWrapper k_wrapper = make_wrapper(key, ACL_HIFLOAT8);
    TensorWrapper v_wrapper = make_wrapper(value, ACL_HIFLOAT8);
    TensorWrapper dout_wrapper = make_wrapper(dout, ACL_HIFLOAT8);
    ACLNN_CMD(aclnnInnerQuantFlashAttnGrad, q_wrapper, k_wrapper, v_wrapper, dout_wrapper, attn_out, q_descale,
              k_descale, v_descale, do_descale, p_scale, ds_scale, softmax_lse, cu_seqlens_q, cu_seqlens_kv, seqused_q,
              seqused_kv, sinks, attn_mask, metadata, quant_mode, softmax_scale, mask_mode, win_left, win_right,
              max_seqlen_q, max_seqlen_kv, layout_q, layout_k, dq, dk, dv, dsink);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(dq, dk, dv, dsink);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_flash_attn_grad", &quant_flash_attn_grad, "quant_flash_attn_grad");
}

} // namespace op_api
