/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int64_t DIM_THREE = 3;
const int64_t DIM_FOUR = 4;
const int64_t DIM_FIVE = 5;
const int64_t FP32_BYTES = 4;
const int64_t QBSA_FP8_QUANT_MODE = 1;
const int64_t QBSA_MXFP8_FULL_QUANT_MODE = 2;
const int64_t QBSA_MXFP8_SCALE_GROUP_SIZE = 64;
const int64_t QBSA_MXFP8_SCALE_LAST_DIM = 2;

// 工具函数，推导输出 attention_out / softmax_lse 的 shape 与 dtype
//   - attention_out: BF16，TND 保持原布局，NTD 转为 TND
//   - softmax_lse:   FLOAT32，FP8 为 (N1,T)，MXFP8 为 (T,N1)
std::tuple<at::Tensor, at::Tensor> construct_bsa_output_tensors(const at::Tensor &query, const at::Tensor &value,
                                                                c10::string_view layout_q, int64_t quant_mode,
                                                                bool return_softmax_lse)
{
    const std::string layout_q_str(layout_q);
    TORCH_CHECK(query.dim() == DIM_THREE, "query should be 3D for layout_q TND/NTD, but got ", query.dim(), "D.");
    TORCH_CHECK(layout_q_str == "TND" || layout_q_str == "NTD", "layout_q should be TND or NTD, but got ",
                layout_q_str);
    TORCH_CHECK(quant_mode != QBSA_MXFP8_FULL_QUANT_MODE || layout_q_str == "TND",
                "quant_mode=2 MXFP8 full-quant only supports layout_q TND, but got ", layout_q_str, ".");
    for (auto i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }

    const bool is_ntd = layout_q_str == "NTD";
    int64_t t_size = is_ntd ? query.size(1) : query.size(0);
    int64_t n1_size = is_ntd ? query.size(0) : query.size(1);
    int64_t d_size = query.size(2);
    at::SmallVector<int64_t, SIZE> atten_out_size = {t_size, n1_size, d_size};
    at::SmallVector<int64_t, SIZE> softmax_lse_size;
    if (quant_mode == QBSA_MXFP8_FULL_QUANT_MODE) {
        softmax_lse_size = {t_size, n1_size};
    } else {
        softmax_lse_size = {n1_size, t_size};
    }
    if (!return_softmax_lse) {
        softmax_lse_size = {};
    }

    at::Tensor attention_out = at::empty(atten_out_size, query.options().dtype(at::kBFloat16));
    at::Tensor softmax_lse = at::empty(softmax_lse_size, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}

// step2, 为NPU设备实现前向接口（函数形参顺序 = schema 顺序）
std::tuple<at::Tensor, at::Tensor> npu_quant_block_sparse_attn_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &q_descale,
    const at::Tensor &k_descale, const at::Tensor &v_descale, const c10::optional<at::Tensor> &p_scale,
    const at::Tensor &sparse_indices, const at::Tensor &sparse_seq_len, const c10::optional<at::Tensor> &atten_mask,
    double softmax_scale, int64_t sparse_q_block_size, int64_t sparse_kv_block_size,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &metadata, c10::string_view layout_kv,
    c10::string_view layout_q, c10::string_view layout_sparse_indices, c10::string_view layout_out, int64_t quant_mode,
    int64_t mask_mode, bool return_softmax_lse)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.dim() == DIM_FOUR && value.dim() == DIM_FOUR,
                "key/value should be 4D PA_BNBD [blockNum, Nkv, blockSize, headDim], but got dims ", key.dim(), "/",
                value.dim());
    int64_t paBlockStride = key.stride(0);
    TORCH_CHECK(paBlockStride > 0, "key.stride(0) (paBlockStride) should be greater than 0, but got ", paBlockStride);
    TORCH_CHECK(value.stride(0) == paBlockStride,
                "value.stride(0) should equal key.stride(0), but got value.stride(0)=", value.stride(0),
                " and key.stride(0)=", paBlockStride);
    TORCH_CHECK(key.stride(1) == key.size(2) * key.size(3) && key.stride(2) == key.size(3) && key.stride(3) == 1,
                "key should use segmented PA_BNBD stride [paBlockStride, blockSize * headDim, headDim, 1], but got ",
                key.strides());
    TORCH_CHECK(
        value.stride(1) == value.size(2) * value.size(3) && value.stride(2) == value.size(3) && value.stride(3) == 1,
        "value should use segmented PA_BNBD stride [paBlockStride, blockSize * headDim, headDim, 1], but got ",
        value.strides());
    if (quant_mode == QBSA_MXFP8_FULL_QUANT_MODE) {
        const int64_t keyScaleDSize = (key.size(3) + QBSA_MXFP8_SCALE_GROUP_SIZE - 1) / QBSA_MXFP8_SCALE_GROUP_SIZE;
        const int64_t valueScaleBlockSize =
            (value.size(2) + QBSA_MXFP8_SCALE_GROUP_SIZE - 1) / QBSA_MXFP8_SCALE_GROUP_SIZE;
        TORCH_CHECK(k_descale.dim() == DIM_FIVE,
                    "k_descale should be 5D [blockNum, Nkv, blockSize, D / 64, 2] in quant_mode=2 MXFP8 "
                    "full-quant scenario, but got dim=",
                    k_descale.dim());
        TORCH_CHECK(k_descale.size(0) == key.size(0) && k_descale.size(1) == key.size(1) &&
                        k_descale.size(2) == key.size(2) && k_descale.size(3) == keyScaleDSize &&
                        k_descale.size(4) == QBSA_MXFP8_SCALE_LAST_DIM,
                    "k_descale should be [blockNum, Nkv, blockSize, D / 64, 2] and match key PA_BNBD "
                    "shape, but got k_descale sizes ",
                    k_descale.sizes(), " and key sizes ", key.sizes());
        TORCH_CHECK(k_descale.stride(4) == 1 && k_descale.stride(3) == k_descale.size(4) &&
                        k_descale.stride(2) == k_descale.size(3) * k_descale.size(4) &&
                        k_descale.stride(1) == k_descale.size(2) * k_descale.size(3) * k_descale.size(4) &&
                        k_descale.stride(0) == k_descale.size(1) * k_descale.stride(1),
                    "k_descale should use contiguous [blockNum, Nkv, blockSize, D / 64, 2] stride in "
                    "quant_mode=2 MXFP8 full-quant scenario, but got ",
                    k_descale.strides());
        TORCH_CHECK(v_descale.dim() == DIM_FIVE,
                    "v_descale should be 5D [blockNum, Nkv, blockSize / 64, DV, 2] in quant_mode=2 MXFP8 "
                    "scenario, but got dim=",
                    v_descale.dim());
        TORCH_CHECK(v_descale.size(0) == value.size(0) && v_descale.size(1) == value.size(1) &&
                        v_descale.size(2) == valueScaleBlockSize && v_descale.size(3) == value.size(3) &&
                        v_descale.size(4) == QBSA_MXFP8_SCALE_LAST_DIM,
                    "v_descale should be [blockNum, Nkv, blockSize / 64, DV, 2] and match value PA_BNBD "
                    "shape, but got v_descale sizes ",
                    v_descale.sizes(), " and value sizes ", value.sizes());
        TORCH_CHECK(v_descale.stride(4) == 1 && v_descale.stride(3) == v_descale.size(4) &&
                        v_descale.stride(2) == v_descale.size(3) * v_descale.size(4) &&
                        v_descale.stride(1) == v_descale.size(2) * v_descale.size(3) * v_descale.size(4) &&
                        v_descale.stride(0) == v_descale.size(1) * v_descale.stride(1),
                    "v_descale should use contiguous [blockNum, Nkv, blockSize / 64, DV, 2] stride in "
                    "quant_mode=2 MXFP8 full-quant scenario, but got ",
                    v_descale.strides());
    } else if (quant_mode == QBSA_FP8_QUANT_MODE) {
        TORCH_CHECK(k_descale.dim() == DIM_FOUR,
                    "k_descale should be 4D [blockNum, Nkv, blockSize, 1] in quant_mode=1 FP8 "
                    "full-quant scenario, but got dim=",
                    k_descale.dim());
        TORCH_CHECK(k_descale.size(0) == key.size(0) && k_descale.size(1) == key.size(1) &&
                        k_descale.size(2) == key.size(2) && k_descale.size(3) == 1,
                    "k_descale should be [blockNum, Nkv, blockSize, 1] and match key PA_BNBD shape, "
                    "but got k_descale sizes ",
                    k_descale.sizes(), " and key sizes ", key.sizes());
        const int64_t expectedPaBlockStride = key.size(1) * key.size(2) * key.size(3) +
                                              value.size(1) * value.size(2) * value.size(3) +
                                              k_descale.size(1) * k_descale.size(2) * k_descale.size(3) * FP32_BYTES;
        TORCH_CHECK(paBlockStride == expectedPaBlockStride,
                    "key.stride(0) should equal K/V/k_descale concatenated physical block size, but got ",
                    paBlockStride, " and expected ", expectedPaBlockStride);
        TORCH_CHECK(k_descale.stride(0) * FP32_BYTES == paBlockStride, "k_descale.stride(0) * ", FP32_BYTES,
                    " should equal key.stride(0), but got k_descale.stride(0)=", k_descale.stride(0),
                    " and key.stride(0)=", paBlockStride);
        TORCH_CHECK(k_descale.stride(1) == k_descale.size(2) && k_descale.stride(2) == 1 && k_descale.stride(3) == 1,
                    "k_descale should use segmented PA_BNBD stride [paBlockStride / 4, blockSize, 1, 1], "
                    "but got ",
                    k_descale.strides());
    } else {
        TORCH_CHECK(false, "quant_mode must be 1 (FP8) or 2 (MXFP8 full-quant), but got ", quant_mode);
    }

    // construct the output tensors
    std::tuple<at::Tensor, at::Tensor> outputs =
        construct_bsa_output_tensors(query, value, layout_q, quant_mode, return_softmax_lse);
    at::Tensor attention_out = std::get<0>(outputs);
    at::Tensor softmax_lse = std::get<1>(outputs);

    // convert str
    std::string layout_kv_str = std::string(layout_kv);
    std::string layout_q_str = std::string(layout_q);
    std::string layout_sparse_indices_str = std::string(layout_sparse_indices);
    std::string layout_out_str = std::string(layout_out);
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_sparse_indices_ptr = const_cast<char *>(layout_sparse_indices_str.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out_str.c_str());

    // aclnn 框架不接受 nullptr aclTensor*：将未提供的 optional p_scale 归一化为 numel=0 空 tensor。
    // dtype 须匹配 CheckScaleDtype：FP8->FLOAT, MXFP8->FLOAT8_E8M0
    at::Tensor p_scale_placeholder;
    const at::Tensor *p_scale_arg;
    if (p_scale.has_value() && p_scale.value().defined()) {
        p_scale_arg = &p_scale.value();
    } else {
        const at::ScalarType empty_dtype =
            (quant_mode == QBSA_MXFP8_FULL_QUANT_MODE) ? at::kFloat8_e8m0fnu : at::kFloat;
        p_scale_placeholder = at::empty({0}, query.options().dtype(empty_dtype));
        p_scale_arg = &p_scale_placeholder;
    }

    // EXEC_NPU_CMD_V1 实参顺序 = 算子 IR 声明顺序（输入 -> 属性 -> 输出），与 schema 形参顺序不同
    EXEC_NPU_CMD_V1(aclnnQuantBlockSparseAttn, query, key, value, q_descale, k_descale, v_descale, *p_scale_arg,
                    cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sparse_indices, sparse_seq_len, block_table,
                    atten_mask, metadata, softmax_scale, sparse_q_block_size, sparse_kv_block_size, layout_kv_ptr,
                    layout_q_ptr, layout_sparse_indices_ptr, layout_out_ptr, quant_mode, mask_mode, return_softmax_lse,
                    attention_out, softmax_lse);

    return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_quant_block_sparse_attn_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &q_descale,
    const at::Tensor &k_descale, const at::Tensor &v_descale, const c10::optional<at::Tensor> &p_scale,
    const at::Tensor &sparse_indices, const at::Tensor &sparse_seq_len, const c10::optional<at::Tensor> &atten_mask,
    double softmax_scale, int64_t sparse_q_block_size, int64_t sparse_kv_block_size,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &metadata, c10::string_view layout_kv,
    c10::string_view layout_q, c10::string_view layout_sparse_indices, c10::string_view layout_out, int64_t quant_mode,
    int64_t mask_mode, bool return_softmax_lse)
{
    return construct_bsa_output_tensors(query, value, layout_q, quant_mode, return_softmax_lse);
}
} // namespace custom

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_quant_block_sparse_attn", &custom::npu_quant_block_sparse_attn_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_quant_block_sparse_attn", &custom::npu_quant_block_sparse_attn_meta);
}
