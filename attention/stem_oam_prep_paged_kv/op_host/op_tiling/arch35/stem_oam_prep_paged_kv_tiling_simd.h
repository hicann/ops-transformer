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
 * \file stem_oam_prep_paged_kv_tiling.h
 * \brief StemOamPrepPagedKv simd TilingData definition and Tiling class
 */
#ifndef STEM_OAM_PREP_PAGED_KV_TILING_SIMD_H
#define STEM_OAM_PREP_PAGED_KV_TILING_SIMD_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"
#include "attention/stem_oam_prep_paged_kv/op_kernel/arch35/stem_oam_prep_paged_kv_tiling_data.h"
#include "attention/stem_oam_prep_paged_kv/op_kernel/arch35/stem_oam_prep_paged_kv_tiling_key.h"
#include "err/ops_err.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

constexpr uint64_t MAX_KV_PADDED = 262144;

constexpr int64_t DIM_QK = 128;
constexpr int64_t KV_BLOCK_SIZE_64 = 64;
constexpr int64_t KV_BLOCK_SIZE_128 = 128;
constexpr int64_t STEM_BLOCK_SIZE_ALIGN = 32;
constexpr int64_t STEM_BLOCK_SIZE_MAX = 256;
constexpr int64_t STEM_STRIDE_ALIGN = 16;
constexpr int64_t STEM_STRIDE_MAX = 64;
constexpr int64_t KV_LAYOUT_BBND = 0;
constexpr int64_t KV_LAYOUT_BNBD = 1;
constexpr int64_t KV_CACHE_DIM_NUM = 4;
constexpr int64_t V_SCALE_DIM_NUM = 1;
constexpr int64_t STRIDE_DIM_NUM = 4;
constexpr int64_t HKVMAX = 8;
constexpr int64_t BATCHMAX = 16;
constexpr int64_t KVBLOCKSIZEONE = 64;
constexpr int64_t KVBLOCKSIZETWO = 128;

constexpr size_t INPUT_KCACHE_INDEX = 0;
constexpr size_t INPUT_VCACHE_INDEX = 1;
constexpr size_t INPUT_KV_INDICES_INDEX = 2;
constexpr size_t INPUT_KV_SEQ_LENS_INDEX = 3;
constexpr size_t INPUT_K_SCALE_CACHE_INDEX = 4;
constexpr size_t INPUT_V_SCALE_INDEX = 5;
constexpr size_t INPUT_COUNT = 6;

constexpr size_t ATTR_LAMBDA_MAG_INDEX = 0;
constexpr size_t ATTR_KV_LAYOUT_INDEX = 1;
constexpr size_t ATTR_STEM_BLOCK_SIZE_INDEX = 2;
constexpr size_t ATTR_STEM_STRIDE_INDEX = 3;

constexpr size_t OUTPUT_KFLAT_INDEX = 0;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;

struct StemOamPrepPagedKvCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

class StemOamPrepPagedKvTilingSimd : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit StemOamPrepPagedKvTilingSimd(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    ge::graphStatus CheckParams();
    ge::graphStatus ContinuousStridesCompute(gert::Shape &shape, gert::Stride &stride, size_t idx);
    ge::graphStatus GetTensorInfo(gert::Shape &shape, gert::Stride &inputStride, size_t idx);

private:
    uint64_t totalCoreNum_ = 0;

    int64_t batchSize_ = 0;
    int64_t numKvHeads_ = 0;
    int64_t kvLayout_ = KV_LAYOUT_BNBD;
    int64_t kvBlockSize_ = KV_BLOCK_SIZE_64;
    int64_t stemBlocks_ = STEM_BLOCK_SIZE_MAX;
    uint32_t stemStride_ = STEM_STRIDE_ALIGN;
    float lambdaMag_ = 0.3f;
    int64_t maxKvBlocks_ = 0;
    int64_t dimQk_ = DIM_QK;
    int64_t maxKb_ = 0;
    int64_t kflatDim_ = 0;
    int64_t rVal_ = 0;
    uint32_t meanMinSize_ = 0;
    uint32_t meanMaxSize_ = 0;

    gert::Stride kCacheStride_;
    gert::Stride vCacheStride_;
    gert::Stride kScaleCacheStride_;
    gert::Shape kcacheShape_;
    gert::Shape vcacheShape_;
    gert::Shape kScaleCacheShape_;

    StemOamPrepPagedKvTilingData *tilingData_ = nullptr;
};

} // namespace optiling

#endif // STEM_OAM_PREP_PAGED_KV_TILING_SIMD_H
