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
 * \file mla_prolog.cpp
 * \brief PyTorch extension wrapper for aclnnMlaPrologV4WeightNz.
 */

#include <algorithm>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {

constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t MODE_1 = 1;
constexpr int64_t MODE_2 = 2;
constexpr int64_t MODE_3 = 3;
constexpr int64_t MODE_4 = 4;
constexpr int64_t MODE_5 = 5;
constexpr int64_t FP8_E4M3_BLOCK_SIZE = 32;
// shape 格式字段约束
constexpr int64_t HE_SUPPORTED[] = {1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192};
constexpr int64_t HCQ_SUPPORTED[] = {1536, 2048};
constexpr int64_t D_SUPPORTED[] = {128, 192};
constexpr int64_t HCKV = 512;
constexpr int64_t DR = 64;
constexpr int64_t NKV = 1;
constexpr int64_t BLOCK_SIZE_MIN = 16;
constexpr int64_t BLOCK_SIZE_MAX = 1024;
constexpr int64_t DTILE_NON_PTPG = 512;
constexpr int64_t DTILE_PTPG = 656;
// N（Head-Num 多头数）支持 [1, 128] 之间的任意整型值
constexpr int64_t HEAD_NUM_MIN = 1;
constexpr int64_t HEAD_NUM_MAX = 128;

inline bool HasDefinedTensor(const c10::optional<at::Tensor> &opt_tensor)
{
    return opt_tensor.has_value() && opt_tensor.value().defined() && opt_tensor.value().numel() > 0;
}

// do_rope 由 ropeSin/ropeCos 是否成对传入决定：二者同时为空（或 None）视为关闭，
// 二者同时非空视为开启，一空一非空为非法输入，须拦截报错。
bool ResolveDoRope(const c10::optional<at::Tensor> &rope_sin, const c10::optional<at::Tensor> &rope_cos)
{
    const bool has_rope_sin = HasDefinedTensor(rope_sin);
    const bool has_rope_cos = HasDefinedTensor(rope_cos);
    TORCH_CHECK(has_rope_sin == has_rope_cos,
                "rope_sin and rope_cos must both be provided or both be empty/None, but the actual "
                "presence states are sin=",
                has_rope_sin, ", cos=", has_rope_cos);
    return has_rope_sin;
}

// do_rope=false 时 Dr 由 weightDkvKr 的 (Hckv+Dr) 与 weightUk 的 Hckv 推导；
// do_rope=true 时 Dr 取 ropeSin 最后一维。
int64_t ResolveRopeDim(const at::Tensor &token_x, const c10::optional<at::Tensor> &rope_sin,
                       const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr)
{
    const bool has_rope_sin = HasDefinedTensor(rope_sin);
    if (!has_rope_sin) {
        const int64_t hckv_plus_dr =
            (weight_dkv_kr.dim() == 4) ? weight_dkv_kr.size(0) * weight_dkv_kr.size(3) : weight_dkv_kr.size(1);
        return hckv_plus_dr - weight_uk.size(2);
    }
    const at::Tensor &rope_sin_tensor = rope_sin.value();
    const int64_t token_x_dim = token_x.dim();
    const int64_t rope_sin_dim = rope_sin_tensor.dim();
    if (token_x_dim == DIM_3) {
        TORCH_CHECK(rope_sin_dim == DIM_3,
                    "when token_x dim num is 3, rope_sin dim num should be 3, but the actual value is ", rope_sin_dim);
        return rope_sin_tensor.size(2);
    }
    TORCH_CHECK(rope_sin_dim == DIM_2,
                "when token_x dim num is 2, rope_sin dim num should be 2, but the actual value is ", rope_sin_dim);
    return rope_sin_tensor.size(1);
}

// 返回权重张量的逻辑行列数：2 维（ND）直接取 (s0, s1)；4 维（FRACTAL_NZ）取 (s0*16, s1*16)。
std::pair<int64_t, int64_t> ResolveLogicalDims(const at::Tensor &tensor)
{
    if (tensor.dim() == 4) {
        return {tensor.size(0) * 16, tensor.size(1) * 16};
    }
    return {tensor.size(0), tensor.size(1)};
}

bool IsInSupportedSet(int64_t value, const int64_t *supported, int64_t size)
{
    return std::find(supported, supported + size, value) != supported + size;
}

// 算子对 weight_dq/weight_uq_qr/weight_dkv_kr 严格要求 FRACTAL_NZ 格式，此处提前校验。
void CheckFractalNzFormat(const at::Tensor &tensor, const std::string &name)
{
    if (!torch_npu::utils::is_npu(tensor)) {
        return;
    }
    const auto format = static_cast<aclFormat>(at_npu::native::get_npu_format(tensor));
    TORCH_CHECK(format == ACL_FORMAT_FRACTAL_NZ, name,
                " must be in FRACTAL_NZ format "
                "(use torch_npu.npu_format_cast(t, 29)), but the actual format is ",
                static_cast<int64_t>(format));
}

constexpr int64_t HE_SUPPORTED_SIZE = sizeof(HE_SUPPORTED) / sizeof(HE_SUPPORTED[0]);
constexpr int64_t HCQ_SUPPORTED_SIZE = sizeof(HCQ_SUPPORTED) / sizeof(HCQ_SUPPORTED[0]);
constexpr int64_t D_SUPPORTED_SIZE = sizeof(D_SUPPORTED) / sizeof(D_SUPPORTED[0]);

// 校验 kv_cache 最后一维（Dtile）：pertoken-pergroup 场景（kv_cache_quant_mode=3）为 656，其余为 512。
void _CheckDtile(const at::Tensor &kv_cache, int64_t kv_cache_quant_mode)
{
    const int64_t dtile = kv_cache.size(kv_cache.dim() - 1);
    const int64_t expected_dtile = (kv_cache_quant_mode == MODE_3) ? DTILE_PTPG : DTILE_NON_PTPG;
    TORCH_CHECK(dtile == expected_dtile, "kv_cache last dim (Dtile) must be ", expected_dtile,
                " (kv_cache_quant_mode=", kv_cache_quant_mode, "), but the actual value is ", dtile);
}

// 校验 cache 张量（kv_cache/kr_cache）的 shape 格式字段约束。
void CheckCacheShapes(const at::Tensor &kv_cache, const at::Tensor &kr_cache, const std::string &cache_mode,
                      int64_t kv_cache_quant_mode, int64_t rope_dim, const c10::optional<at::Tensor> &cache_index)
{
    const bool is_tnd = cache_mode == "TND";
    if (is_tnd) {
        TORCH_CHECK(kv_cache.dim() == DIM_3 && kr_cache.dim() == DIM_3,
                    "when cache_mode is TND, kv_cache/kr_cache must be 3D");
        TORCH_CHECK(kv_cache.size(kv_cache.dim() - 2) == NKV, "kv head num Nkv (kv_cache dim -2) must be ", NKV,
                    ", but the actual value is ", kv_cache.size(kv_cache.dim() - 2));
        TORCH_CHECK(kr_cache.size(kr_cache.dim() - 2) == NKV, "kv head num Nkv (kr_cache dim -2) must be ", NKV,
                    ", but the actual value is ", kr_cache.size(kr_cache.dim() - 2));
        TORCH_CHECK(kr_cache.size(kr_cache.dim() - 1) == rope_dim, "kr_cache last dim must equal Dr=", rope_dim,
                    ", but the actual value is ", kr_cache.size(kr_cache.dim() - 1));
        _CheckDtile(kv_cache, kv_cache_quant_mode);
        return;
    }

    TORCH_CHECK(kv_cache.dim() == 4 && kr_cache.dim() == 4, "when cache_mode is ", cache_mode,
                ", kv_cache/kr_cache must be 4D");
    TORCH_CHECK(kv_cache.size(kv_cache.dim() - 2) == NKV, "kv head num Nkv (kv_cache dim -2) must be ", NKV,
                ", but the actual value is ", kv_cache.size(kv_cache.dim() - 2));
    TORCH_CHECK(kr_cache.size(kr_cache.dim() - 2) == NKV, "kv head num Nkv (kr_cache dim -2) must be ", NKV,
                ", but the actual value is ", kr_cache.size(kr_cache.dim() - 2));
    TORCH_CHECK(kr_cache.size(kr_cache.dim() - 1) == rope_dim, "kr_cache last dim must equal Dr=", rope_dim,
                ", but the actual value is ", kr_cache.size(kr_cache.dim() - 1));
    _CheckDtile(kv_cache, kv_cache_quant_mode);

    const bool is_pa =
        cache_mode == "PA_BSND" || cache_mode == "PA_NZ" || cache_mode == "PA_BLK_BSND" || cache_mode == "PA_BLK_NZ";
    if (is_pa) {
        TORCH_CHECK(HasDefinedTensor(cache_index), "when cache_mode is ", cache_mode,
                    ", cache_index must be provided "
                    "(PagedAttention cache modes require a non-empty cache_index)");
        const int64_t block_size = kv_cache.size(1);
        TORCH_CHECK(block_size >= BLOCK_SIZE_MIN && block_size <= BLOCK_SIZE_MAX && block_size % 16 == 0,
                    "BlockSize (kv_cache dim 1) must be in [", BLOCK_SIZE_MIN, ", ", BLOCK_SIZE_MAX,
                    "] and a multiple of 16, but the actual value is ", block_size);
        TORCH_CHECK(kr_cache.size(1) == block_size, "kr_cache dim 1 must equal BlockSize (", block_size,
                    "), but the actual value is ", kr_cache.size(1));
    }
}

// 校验接口的 shape 格式字段约束（与 torch_npu.npu_mla_prolog_v3 对齐）。
// N（Head-Num 多头数）在新接口上放开取值，支持 [1, 128] 之间的任意整型值。
void CheckShapeConstraints(const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr,
                           const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
                           const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv,
                           const at::Tensor &kv_cache, const at::Tensor &kr_cache,
                           const c10::optional<at::Tensor> &rope_sin, const c10::optional<at::Tensor> &rope_cos,
                           const c10::optional<at::Tensor> &cache_index, const std::string &cache_mode,
                           int64_t kv_cache_quant_mode)
{
    const int64_t token_x_dim = token_x.dim();
    TORCH_CHECK(token_x_dim == DIM_2 || token_x_dim == DIM_3,
                "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim);
    const int64_t he = token_x.size(token_x_dim - 1);
    TORCH_CHECK(IsInSupportedSet(he, HE_SUPPORTED, HE_SUPPORTED_SIZE),
                "head size He (token_x last dim) must be one of 1024/2048/3072/4096/5120/6144/7168/7680/8192, "
                "but the actual value is ",
                he);
    // B 约束：3 维时 B = size(0) 需在 [0, 65536] 内
    if (token_x_dim == DIM_3) {
        const int64_t b = token_x.size(0);
        TORCH_CHECK(b >= 0 && b <= 65536, "batch B (token_x.size(0)) must be in [0, 65536], but the actual value is ",
                    b);
    }

    TORCH_CHECK(weight_uk.dim() == DIM_3, "weight_uk dim num should be 3, but the actual value is ", weight_uk.dim());
    const int64_t head_num = weight_uk.size(0);
    const int64_t qk_dim = weight_uk.size(1);
    const int64_t kv_lora_rank = weight_uk.size(2);
    TORCH_CHECK(head_num >= HEAD_NUM_MIN && head_num <= HEAD_NUM_MAX,
                "head num N (weight_uk.size(0)) must be an integer in [", HEAD_NUM_MIN, ", ", HEAD_NUM_MAX,
                "], but the actual value is ", head_num);
    TORCH_CHECK(IsInSupportedSet(qk_dim, D_SUPPORTED, D_SUPPORTED_SIZE),
                "qk dim D (weight_uk.size(1)) must be one of 128/192, but the actual value is ", qk_dim);
    TORCH_CHECK(kv_lora_rank == HCKV, "kv lora rank Hckv (weight_uk.size(2)) must be ", HCKV,
                ", but the actual value is ", kv_lora_rank);

    const auto dq_dims = ResolveLogicalDims(weight_dq);
    CheckFractalNzFormat(weight_dq, "weight_dq");
    TORCH_CHECK(dq_dims.first == he, "weight_dq.size(0) must equal He (", he, "), but the actual value is ",
                dq_dims.first);
    const int64_t hcq = dq_dims.second;
    TORCH_CHECK(IsInSupportedSet(hcq, HCQ_SUPPORTED, HCQ_SUPPORTED_SIZE),
                "q lora rank Hcq (weight_dq.size(1)) must be one of 1536/2048, but the actual value is ", hcq);

    const int64_t rope_dim = ResolveRopeDim(token_x, rope_sin, weight_uk, weight_dkv_kr);
    TORCH_CHECK(rope_dim == DR, "qk rope dim Dr must be ", DR, ", but the actual value is ", rope_dim);
    if (HasDefinedTensor(rope_cos)) {
        TORCH_CHECK(rope_cos.value().size(rope_cos.value().dim() - 1) == rope_dim,
                    "rope_sin and rope_cos last dim must be equal, but rope_cos last dim is ",
                    rope_cos.value().size(rope_cos.value().dim() - 1));
    }

    const auto uq_dims = ResolveLogicalDims(weight_uq_qr);
    CheckFractalNzFormat(weight_uq_qr, "weight_uq_qr");
    TORCH_CHECK(uq_dims.first == hcq, "weight_uq_qr.size(0) must equal Hcq (", hcq, "), but the actual value is ",
                uq_dims.first);
    const int64_t expected_uq_cols = head_num * (qk_dim + rope_dim);
    TORCH_CHECK(uq_dims.second == expected_uq_cols, "weight_uq_qr.size(1) must equal N*(D+Dr) = ", expected_uq_cols,
                ", but the actual value is ", uq_dims.second);

    const auto dkv_dims = ResolveLogicalDims(weight_dkv_kr);
    CheckFractalNzFormat(weight_dkv_kr, "weight_dkv_kr");
    TORCH_CHECK(dkv_dims.first == he, "weight_dkv_kr.size(0) must equal He (", he, "), but the actual value is ",
                dkv_dims.first);
    const int64_t expected_dkv_cols = kv_lora_rank + rope_dim;
    TORCH_CHECK(dkv_dims.second == expected_dkv_cols, "weight_dkv_kr.size(1) must equal Hckv+Dr = ", expected_dkv_cols,
                ", but the actual value is ", dkv_dims.second);

    TORCH_CHECK(rmsnorm_gamma_cq.dim() == 1 && rmsnorm_gamma_cq.size(0) == hcq,
                "rmsnorm_gamma_cq must be 1D with shape [", hcq, "], but got ", rmsnorm_gamma_cq.sizes());
    TORCH_CHECK(rmsnorm_gamma_ckv.dim() == 1 && rmsnorm_gamma_ckv.size(0) == kv_lora_rank,
                "rmsnorm_gamma_ckv must be 1D with shape [", kv_lora_rank, "], but got ", rmsnorm_gamma_ckv.sizes());

    CheckCacheShapes(kv_cache, kr_cache, cache_mode, kv_cache_quant_mode, rope_dim, cache_index);
}

bool IsFullQuantKvScene(int64_t weight_quant_mode, int64_t kv_cache_quant_mode)
{
    return (weight_quant_mode == MODE_2 || weight_quant_mode == MODE_3 || weight_quant_mode == MODE_4 ||
            weight_quant_mode == MODE_5) &&
           kv_cache_quant_mode == MODE_1;
}

at::Tensor MakeQueryTensor(const at::Tensor &token_x, const at::Tensor &weight_uk, int64_t weight_quant_mode,
                           int64_t kv_cache_quant_mode, bool is_hifloat8)
{
    const bool full_quant_kv = IsFullQuantKvScene(weight_quant_mode, kv_cache_quant_mode);
    at::ScalarType query_dtype = full_quant_kv ? (is_hifloat8 ? at::kByte : token_x.scalar_type()) : at::kBFloat16;
    if (token_x.dim() == DIM_3) {
        return at::empty({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)},
                         token_x.options().dtype(query_dtype));
    }
    return at::empty({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(query_dtype));
}

at::Tensor MakeQueryRopeTensor(const at::Tensor &token_x, const at::Tensor &weight_uk, int64_t rope_dim)
{
    if (token_x.dim() == DIM_3) {
        return at::empty({token_x.size(0), token_x.size(1), weight_uk.size(0), rope_dim},
                         token_x.options().dtype(at::kBFloat16));
    }
    return at::empty({token_x.size(0), weight_uk.size(0), rope_dim}, token_x.options().dtype(at::kBFloat16));
}

at::Tensor MakeDequantScaleQNopeTensor(const at::Tensor &token_x, const at::Tensor &weight_uk)
{
    if (token_x.dim() == DIM_3) {
        return at::empty({token_x.size(0) * token_x.size(1), weight_uk.size(0), 1},
                         token_x.options().dtype(at::kFloat));
    }
    return at::empty({token_x.size(0), weight_uk.size(0), 1}, token_x.options().dtype(at::kFloat));
}

at::Tensor MakeQueryNormTensor(const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr,
                               bool is_hifloat8)
{
    at::ScalarType query_norm_dtype = is_hifloat8 ? at::kByte : weight_uq_qr.scalar_type();
    if (token_x.dim() == DIM_3) {
        return at::empty({token_x.size(0), token_x.size(1), weight_dq.size(1)},
                         token_x.options().dtype(query_norm_dtype));
    }
    return at::empty({token_x.size(0), weight_dq.size(1)}, token_x.options().dtype(query_norm_dtype));
}

at::Tensor MakeDequantScaleQNormTensor(const at::Tensor &token_x, const at::Tensor &weight_dq,
                                       int64_t weight_quant_mode, const c10::optional<at::Tensor> &dequant_scale_x)
{
    at::ScalarType dtype = at::kFloat;
    int64_t norm_dim = 1;
    if (weight_quant_mode == MODE_3) {
        dtype = dequant_scale_x.value().scalar_type();
        norm_dim = weight_dq.size(1) / FP8_E4M3_BLOCK_SIZE;
    }
    if (token_x.dim() == DIM_3) {
        return at::empty({token_x.size(0) * token_x.size(1), norm_dim}, token_x.options().dtype(dtype));
    }
    return at::empty({token_x.size(0), norm_dim}, token_x.options().dtype(dtype));
}

at::Tensor MakeEmptyScalarTensor(const at::Tensor &ref, at::ScalarType dtype)
{
    return at::empty({0}, ref.options().dtype(dtype));
}

void CheckMxfp8Inputs(int64_t weight_quant_mode, const c10::optional<at::Tensor> &dequant_scale_x,
                      const c10::optional<at::Tensor> &dequant_scale_w_dq,
                      const c10::optional<at::Tensor> &dequant_scale_w_uq_qr,
                      const c10::optional<at::Tensor> &dequant_scale_w_dkv_kr)
{
    if (weight_quant_mode != MODE_3) {
        return;
    }
    TORCH_CHECK(HasDefinedTensor(dequant_scale_x) && HasDefinedTensor(dequant_scale_w_dq) &&
                    HasDefinedTensor(dequant_scale_w_uq_qr) && HasDefinedTensor(dequant_scale_w_dkv_kr),
                "when weight_quant_mode is 3, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, "
                "dequant_scale_w_dkv_kr must all be provided");
    TORCH_CHECK(dequant_scale_x.value().scalar_type() == at::kFloat8_e8m0fnu &&
                    dequant_scale_w_dq.value().scalar_type() == at::kFloat8_e8m0fnu &&
                    dequant_scale_w_uq_qr.value().scalar_type() == at::kFloat8_e8m0fnu &&
                    dequant_scale_w_dkv_kr.value().scalar_type() == at::kFloat8_e8m0fnu,
                "when weight_quant_mode is 3, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, "
                "dequant_scale_w_dkv_kr dtype must be torch.float8_e8m0fnu");
}

bool IsHifloat8Scene(int64_t weight_quant_mode, const at::Tensor &token_x, const at::Tensor &weight_dq,
                     const at::Tensor &weight_uq_qr, const at::Tensor &weight_dkv_kr,
                     const c10::optional<int64_t> &token_x_dtype, const c10::optional<int64_t> &weight_dq_dtype,
                     const c10::optional<int64_t> &weight_uq_qr_dtype,
                     const c10::optional<int64_t> &weight_dkv_kr_dtype, const c10::optional<int64_t> &kv_cache_dtype,
                     int64_t kv_cache_quant_mode)
{
    if (weight_quant_mode != MODE_5) {
        return false;
    }
    const bool has_byte_input = token_x.scalar_type() == at::kByte || weight_dq.scalar_type() == at::kByte ||
                                weight_uq_qr.scalar_type() == at::kByte || weight_dkv_kr.scalar_type() == at::kByte;
    if (!has_byte_input) {
        return false;
    }
    TORCH_CHECK(token_x_dtype.has_value() && weight_dq_dtype.has_value() && weight_uq_qr_dtype.has_value() &&
                    weight_dkv_kr_dtype.has_value(),
                "when weight_quant_mode is 5 and input dtype is hifloat8, token_x_dtype, weight_dq_dtype, "
                "weight_uq_qr_dtype, weight_dkv_kr_dtype cannot be null");
    TORCH_CHECK(GetAclDataType(token_x_dtype.value()) == ACL_HIFLOAT8 &&
                    GetAclDataType(weight_dq_dtype.value()) == ACL_HIFLOAT8 &&
                    GetAclDataType(weight_uq_qr_dtype.value()) == ACL_HIFLOAT8 &&
                    GetAclDataType(weight_dkv_kr_dtype.value()) == ACL_HIFLOAT8,
                "when weight_quant_mode is 5 and input dtype is hifloat8, token_x_dtype, weight_dq_dtype, "
                "weight_uq_qr_dtype, weight_dkv_kr_dtype value must be torch_npu.hifloat8");
    if (kv_cache_quant_mode == MODE_1 || kv_cache_quant_mode == MODE_3) {
        TORCH_CHECK(kv_cache_dtype.has_value(),
                    "when weight_quant_mode is 5 and kv_cache_quant_mode is 1 or 3 and input dtype is hifloat8, "
                    "kv_cache_dtype cannot be null");
        TORCH_CHECK(GetAclDataType(kv_cache_dtype.value()) == ACL_HIFLOAT8,
                    "when weight_quant_mode is 5 and input dtype is hifloat8, kv_cache_dtype value must be "
                    "torch_npu.hifloat8");
    }
    return true;
}

inline aclDataType ResolveTensorAclDtype(const at::Tensor &tensor, bool force_hifloat8)
{
    if (force_hifloat8) {
        return ACL_HIFLOAT8;
    }
    return tensor.defined() ? ConvertToAclDataType(tensor.scalar_type()) : ACL_DT_UNDEFINED;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mla_prolog(
    const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr, const at::Tensor &weight_uk,
    const at::Tensor &weight_dkv_kr, const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv,
    at::Tensor &kv_cache, at::Tensor &kr_cache, const c10::optional<at::Tensor> &rope_sin,
    const c10::optional<at::Tensor> &rope_cos, const c10::optional<at::Tensor> &cache_index,
    const c10::optional<at::Tensor> &dequant_scale_x, const c10::optional<at::Tensor> &dequant_scale_w_dq,
    const c10::optional<at::Tensor> &dequant_scale_w_uq_qr, const c10::optional<at::Tensor> &dequant_scale_w_dkv_kr,
    const c10::optional<at::Tensor> &quant_scale_ckv, const c10::optional<at::Tensor> &quant_scale_ckr,
    const c10::optional<at::Tensor> &smooth_scales_cq, const c10::optional<at::Tensor> &actual_seq_len,
    const c10::optional<at::Tensor> &k_nope_clip_alpha, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv,
    const std::string &cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
    int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size,
    double qc_qr_scale, double kc_scale, const c10::optional<int64_t> &token_x_dtype,
    const c10::optional<int64_t> &weight_dq_dtype, const c10::optional<int64_t> &weight_uq_qr_dtype,
    const c10::optional<int64_t> &weight_dkv_kr_dtype, const c10::optional<int64_t> &kv_cache_dtype)
{
    const c10::OptionalDeviceGuard device_guard(token_x.device());

    const int64_t token_x_dim = token_x.dim();
    TORCH_CHECK(token_x_dim == DIM_2 || token_x_dim == DIM_3,
                "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim);
    TORCH_CHECK(weight_uk.dim() == DIM_3, "weight_uk dim num should be 3, but the actual value is ", weight_uk.dim());
    CheckShapeConstraints(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
                          rmsnorm_gamma_ckv, kv_cache, kr_cache, rope_sin, rope_cos, cache_index, cache_mode,
                          kv_cache_quant_mode);

    const bool is_hifloat8 =
        IsHifloat8Scene(weight_quant_mode, token_x, weight_dq, weight_uq_qr, weight_dkv_kr, token_x_dtype,
                        weight_dq_dtype, weight_uq_qr_dtype, weight_dkv_kr_dtype, kv_cache_dtype, kv_cache_quant_mode);
    CheckMxfp8Inputs(weight_quant_mode, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
                     dequant_scale_w_dkv_kr);

    const bool do_rope = ResolveDoRope(rope_sin, rope_cos);
    const int64_t rope_dim = ResolveRopeDim(token_x, rope_sin, weight_uk, weight_dkv_kr);

    at::Tensor query{nullptr};
    at::Tensor query_rope{nullptr};
    at::Tensor dequant_scale_q_nope{nullptr};
    at::Tensor query_norm{nullptr};
    at::Tensor dequant_scale_q_norm{nullptr};

    query = MakeQueryTensor(token_x, weight_uk, weight_quant_mode, kv_cache_quant_mode, is_hifloat8);
    if (IsFullQuantKvScene(weight_quant_mode, kv_cache_quant_mode)) {
        dequant_scale_q_nope = MakeDequantScaleQNopeTensor(token_x, weight_uk);
    }
    query_rope = MakeQueryRopeTensor(token_x, weight_uk, rope_dim);
    if (query_norm_flag) {
        query_norm = MakeQueryNormTensor(token_x, weight_dq, weight_uq_qr, is_hifloat8);
        if (weight_quant_mode != 0) {
            dequant_scale_q_norm = MakeDequantScaleQNormTensor(token_x, weight_dq, weight_quant_mode, dequant_scale_x);
        }
    }

    char *cache_mode_ptr = const_cast<char *>(cache_mode.data());

    // 仅当 hifloat8 场景且 kv_cache 以 HIFLOAT8 存储（kv_cache_quant_mode 1/3）时强制 HIFLOAT8，
    // 其余 tensor 按自身 dtype 透传（非 hifloat8 场景下与直接传裸 tensor 完全等价）。
    const bool force_kv_cache_hifloat8 =
        is_hifloat8 && (kv_cache_quant_mode == MODE_1 || kv_cache_quant_mode == MODE_3);
    // query 仅在全量化 KV（kv_cache_quant_mode=1）场景才以 HIFLOAT8 输出，否则为 bf16。
    const bool force_query_hifloat8 = is_hifloat8 && kv_cache_quant_mode == MODE_1;
    TensorWrapper token_x_wrapper{token_x, ResolveTensorAclDtype(token_x, is_hifloat8)};
    TensorWrapper weight_dq_wrapper{weight_dq, ResolveTensorAclDtype(weight_dq, is_hifloat8)};
    TensorWrapper weight_uq_qr_wrapper{weight_uq_qr, ResolveTensorAclDtype(weight_uq_qr, is_hifloat8)};
    TensorWrapper weight_dkv_kr_wrapper{weight_dkv_kr, ResolveTensorAclDtype(weight_dkv_kr, is_hifloat8)};
    TensorWrapper kv_cache_wrapper{kv_cache, ResolveTensorAclDtype(kv_cache, force_kv_cache_hifloat8)};
    TensorWrapper query_wrapper{query, ResolveTensorAclDtype(query, force_query_hifloat8)};
    TensorWrapper query_norm_wrapper{query_norm, ResolveTensorAclDtype(query_norm, is_hifloat8)};
    // do_rope=false 时 ropeSin/ropeCos 必须为空（或 None），统一以空 optional 传给 aclnn（转换为 nullptr）。
    const c10::optional<at::Tensor> acl_rope_sin = do_rope ? rope_sin : c10::nullopt;
    const c10::optional<at::Tensor> acl_rope_cos = do_rope ? rope_cos : c10::nullopt;
    ACLNN_CMD(aclnnMlaPrologV4WeightNz, token_x_wrapper, weight_dq_wrapper, weight_uq_qr_wrapper, weight_uk,
              weight_dkv_kr_wrapper, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, acl_rope_sin, acl_rope_cos, kv_cache_wrapper,
              kr_cache, cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr,
              quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len, k_nope_clip_alpha, rmsnorm_epsilon_cq,
              rmsnorm_epsilon_ckv, cache_mode_ptr, weight_quant_mode, kv_cache_quant_mode, query_quant_mode,
              ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, do_rope, query_wrapper,
              query_rope, dequant_scale_q_nope, query_norm_wrapper, dequant_scale_q_norm);

    if (!query_norm.defined()) {
        query_norm = MakeEmptyScalarTensor(weight_uq_qr, weight_uq_qr.scalar_type());
    }
    if (!dequant_scale_q_nope.defined()) {
        dequant_scale_q_nope = MakeEmptyScalarTensor(token_x, at::kFloat);
    }
    if (!dequant_scale_q_norm.defined()) {
        if (weight_quant_mode == MODE_3) {
            dequant_scale_q_norm =
                MakeEmptyScalarTensor(dequant_scale_x.value(), dequant_scale_x.value().scalar_type());
        } else {
            dequant_scale_q_norm = MakeEmptyScalarTensor(token_x, at::kFloat);
        }
    }

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mla_prolog", &mla_prolog, "mla_prolog");
}

} // namespace op_api
