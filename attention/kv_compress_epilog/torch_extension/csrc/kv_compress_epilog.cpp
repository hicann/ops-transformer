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
 * \file kv_compress_epilog.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

namespace {
constexpr int64_t MIN_TAIL_DIM = 64;
constexpr int64_t TAIL_DIM_ALIGN = 64;
constexpr int64_t QUANT_MODE_MIN = 0;
constexpr int64_t QUANT_MODE_MAX = 4;        // 0:group(bf16) 1:group(e8m0) 2:rope hifloat8 + nope FLOAT4_E2M1
                                             // 3:MXFP8 全行量化 4:MXFP4 packed 全行量化
constexpr int64_t MXFP_TAIL_DIM_ALIGN = 128; // mode3/4 的 d 对齐约束（v1.12）
} // namespace

void KvCompressEpilog(at::Tensor& cache, const at::Tensor& x, const at::Tensor& slotMapping, int64_t quantGroupSize,
                      int64_t quantMode, bool roundScale, double xScale)
{
    TORCH_CHECK(cache.numel() > 0, "Tensor cache is empty.");
    TORCH_CHECK(x.dim() >= 2, "x should be at least 2-dim, but got ", x.dim());
    TORCH_CHECK(slotMapping.dim() == x.dim() - 1, "slot_mapping dim should equal x dim - 1, but got slot_mapping dim ",
                slotMapping.dim(), " and x dim ", x.dim());
    TORCH_CHECK(quantMode >= QUANT_MODE_MIN && quantMode <= QUANT_MODE_MAX,
                "quant_mode should be one of 0/1/2/3/4 "
                "(fp8_bf16/fp8_e8m0/hifloat8_fp4/nope_rope_fp8_bf16/nope_rope_fp4_bf16), but got ",
                quantMode);
    int64_t tailDim = x.size(x.dim() - 1);
    if (quantMode == 3 || quantMode == 4) {
        // mode3/4：全行量化（无 rope/nope 拆分），d > 0 且 d % quant_group_size == 0；
        // cache 支持平铺 2D / 分页 4D；mode3 cache 须为 float8_e4m3fn/float8_e5m2，mode4 须为 uint8。
        TORCH_CHECK(tailDim > 0, "The last dim (d) of x should be positive when quant_mode is 3/4, but got ", tailDim);
        TORCH_CHECK(tailDim % quantGroupSize == 0,
                    "The last dim (d) of x should be divisible by quant_group_size when quant_mode is 3/4, but got d ",
                    tailDim, " and group ", quantGroupSize);
        TORCH_CHECK(tailDim % MXFP_TAIL_DIM_ALIGN == 0,
                    "The last dim (d) of x should be a multiple of 128 when quant_mode is 3/4, but got ", tailDim);
        TORCH_CHECK((quantMode == 3 && quantGroupSize == 32) ||
                        (quantMode == 4 && (quantGroupSize == 16 || quantGroupSize == 32)),
                    "quant_group_size should be 32 when quant_mode==3, and 16 or 32 when quant_mode==4, but got mode ",
                    quantMode, " and group ", quantGroupSize);
        TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x dtype should be bfloat16 when quant_mode is 3/4, but got ",
                    x.scalar_type());
        TORCH_CHECK(slotMapping.scalar_type() == at::kInt || slotMapping.scalar_type() == at::kLong,
                    "slot_mapping dtype should be int32 or int64 when quant_mode is 3/4, but got ",
                    slotMapping.scalar_type());
        TORCH_CHECK(cache.dim() == 2 || cache.dim() == 4,
                    "cache should be 2D (flat) or 4D (paged) when quant_mode is 3/4, but got ", cache.dim(), "-dim");
        if (quantMode == 3) {
            TORCH_CHECK(cache.scalar_type() == at::kFloat8_e4m3fn || cache.scalar_type() == at::kFloat8_e5m2,
                        "cache dtype should be float8_e4m3fn or float8_e5m2 when quant_mode==3, but got ",
                        cache.scalar_type());
        } else {
            TORCH_CHECK(cache.scalar_type() == at::kByte, "cache dtype should be uint8 when quant_mode==4, but got ",
                        cache.scalar_type());
        }
    } else {
        TORCH_CHECK(tailDim > MIN_TAIL_DIM && tailDim % TAIL_DIM_ALIGN == 0,
                    "The last dim (d) of x should be greater than 64 and 64-aligned, but got ", tailDim);
    }

    auto localDevice = c10::Device(cache.device());
    const c10::OptionalDeviceGuard deviceGuard(localDevice);

    float xScaleF = static_cast<float>(xScale);
    StorageShapeTensor xWrapped{x};
    StorageShapeTensor slotMappingWrapped{slotMapping};
    ACLNN_CMD(aclnnKvCompressEpilog, cache, xWrapped, slotMappingWrapped, quantGroupSize, quantMode, roundScale,
              xScaleF);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kv_compress_epilog", &KvCompressEpilog, "kv_compress_epilog");
}
} // namespace op_api
