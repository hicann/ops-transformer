/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_rms_norm_rope_cache_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_KV_RMS_NORM_ROPE_CACHE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_KV_RMS_NORM_ROPE_CACHE_H_

#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
using namespace Ops::Base;
// DS
BEGIN_TILING_DATA_DEF(KvRmsNormRopeCacheTilingData)
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(int64_t, rowsPerBlock);
TILING_DATA_FIELD_DEF(int64_t, cacheLength);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, numHead);
TILING_DATA_FIELD_DEF(int64_t, seqLength);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, reciprocal);
TILING_DATA_FIELD_DEF(int8_t, isOutputKv);
TILING_DATA_FIELD_DEF(int8_t, isKQuant);
TILING_DATA_FIELD_DEF(int8_t, isVQuant);
TILING_DATA_FIELD_DEF(int64_t, methodMode);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(KvRmsNormRopeCacheDefaultTilingData)
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(int64_t, rowsPerBlock);
TILING_DATA_FIELD_DEF(int64_t, cacheLength);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, numHead);
TILING_DATA_FIELD_DEF(int64_t, seqLength);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, reciprocal);
TILING_DATA_FIELD_DEF(int8_t, isOutputKv);
TILING_DATA_FIELD_DEF(int8_t, isKQuant);
TILING_DATA_FIELD_DEF(int8_t, isVQuant);
TILING_DATA_FIELD_DEF(int64_t, methodMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache, KvRmsNormRopeCacheDefaultTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_1000, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_1001, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_1010, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_1011, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_2000, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_2001, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_3000, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_3001, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_3010, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_4000, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_4001, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_5000, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_5001, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_4010, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_4011, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_5010, KvRmsNormRopeCacheTilingData)
REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_5011, KvRmsNormRopeCacheTilingData)
// D全载
BEGIN_TILING_DATA_DEF(KvRmsNormRopeCacheRegbaseFullLoadTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, numHead);
TILING_DATA_FIELD_DEF(int64_t, seqLength);
TILING_DATA_FIELD_DEF(int64_t, cacheLength);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, cosSinNeedBrc);
TILING_DATA_FIELD_DEF(int64_t, kScaleType);
TILING_DATA_FIELD_DEF(int64_t, kOffsetType);
TILING_DATA_FIELD_DEF(int64_t, vScaleType);
TILING_DATA_FIELD_DEF(int64_t, vOffsetType);
TILING_DATA_FIELD_DEF(int64_t, isOutputKv);
TILING_DATA_FIELD_DEF(int64_t, cacheMode);
TILING_DATA_FIELD_DEF(int64_t, dk);
TILING_DATA_FIELD_DEF(int64_t, dkAlign);
TILING_DATA_FIELD_DEF(int64_t, dkB8Align);
TILING_DATA_FIELD_DEF(int64_t, halfDk);
TILING_DATA_FIELD_DEF(int64_t, halfDkAlign);
TILING_DATA_FIELD_DEF(int64_t, dv);
TILING_DATA_FIELD_DEF(int64_t, dvAlign);
TILING_DATA_FIELD_DEF(int64_t, dvB8Align);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, inUbSize);
TILING_DATA_FIELD_DEF(int64_t, outUbSize);
TILING_DATA_FIELD_DEF(int64_t, rmsNormWspSize);
TILING_DATA_FIELD_DEF(int64_t, cacheRowLimit); // index 取值上界：Norm 为 Scache，PA 系为 BlockNum * BlockSize
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, reciprocal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_10000, KvRmsNormRopeCacheRegbaseFullLoadTilingData)

// Recompute
BEGIN_TILING_DATA_DEF(KvRmsNormRopeCacheRegbaseRecomputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, numHead);
TILING_DATA_FIELD_DEF(int64_t, seqLength);
TILING_DATA_FIELD_DEF(int64_t, cacheLength);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, cosSinNeedBrc);
TILING_DATA_FIELD_DEF(int64_t, kScaleType);
TILING_DATA_FIELD_DEF(int64_t, kOffsetType);
TILING_DATA_FIELD_DEF(int64_t, vScaleType);
TILING_DATA_FIELD_DEF(int64_t, vOffsetType);
TILING_DATA_FIELD_DEF(int64_t, isOutputKv);
TILING_DATA_FIELD_DEF(int64_t, cacheMode);
TILING_DATA_FIELD_DEF(int64_t, dk);
TILING_DATA_FIELD_DEF(int64_t, dv); // 计算rmsnorm的D
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactorDvTail);
TILING_DATA_FIELD_DEF(int64_t, ubFactorDvLoopCountCeil);
TILING_DATA_FIELD_DEF(int64_t, ubFactorDkTail);
TILING_DATA_FIELD_DEF(int64_t, ubFactorDkLoopCountCeil);
TILING_DATA_FIELD_DEF(int64_t, basicBlockLoop); // 二分累加：循环次数，折叠点左半部分的block数量
TILING_DATA_FIELD_DEF(int64_t, mainFoldCount); // 二分累加：折叠的块数，折叠点右半部分的block数量-1
TILING_DATA_FIELD_DEF(int64_t, cacheRowLimit); // index 取值上界：Norm 为 Scache，PA 系为 BlockNum * BlockSize
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, reciprocal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvRmsNormRopeCache_20000, KvRmsNormRopeCacheRegbaseRecomputeTilingData)

constexpr int32_t TEMPLATE_DS_PRIORITY = 1000;
constexpr int32_t TEMPLATE_D_FULL_LOAD_PRIORITY = 2000;
constexpr int32_t TEMPLATE_D_RECOMPUTE_PRIORITY = 3000;

struct KvRmsNormRopeCacheCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

enum CacheMode {
    Norm = 0,
    PA = 1,
    PA_NZ = 2,
    PA_BLK_BNSD = 3,
    PA_BLK_NZ = 4
};

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t ONE_BUFFER = 1;
constexpr int64_t DIM_NUM_ONE = 1;
constexpr int64_t KV_INDEX = 0;
constexpr int64_t GAMMA_INDEX = 1;
constexpr int64_t COS_INDEX = 2;
constexpr int64_t SIN_INDEX = 3;
constexpr int64_t INDEX_INDEX = 4;
constexpr int64_t K_CACHE_INDEX = 5;
constexpr int64_t V_CACHE_INDEX = 6;
constexpr int64_t K_ROPE_SCALE_IDX = 7;
constexpr int64_t C_KV_SCALE_IDX = 8;
constexpr int64_t K_ROPE_OFFSET_IDX = 9;
constexpr int64_t C_KV_OFFSET_IDX = 10;
constexpr int64_t V_IDX = 11;
constexpr int64_t CACHE_MODE_IDX = 1;
constexpr int64_t IS_OUTPUT_KV_IDX = 2;
constexpr int64_t SHAPE_IDX_B = 0;
constexpr int64_t SHAPE_IDX_N = 1;
constexpr int64_t SHAPE_IDX_S = 2;
constexpr int64_t SHAPE_IDX_D = 3;
constexpr int64_t SHAPE_IDX_BLOCK_NUM = 0;
constexpr int64_t SHAPE_IDX_BLOCK_SIZE = 1;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t FP32_BLOCK_ALIGN_NUM = 8;
constexpr int64_t FP16_BLOCK_ALIGN_NUM = 16;
constexpr int64_t INT8_BLOCK_ALIGN_NUM = 32;
;
constexpr int64_t DIM_SIZE = 4;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;
constexpr int64_t BATCHES_FOR_EACH_CORE = 4;

constexpr int64_t NON_QUANT_MODE = 0;
constexpr int64_t QUANT_MODE = 1;

static constexpr int64_t UB_RESERVED_BYTE = 1024;
static constexpr int64_t FULL_LOAD_BASE_TILING_KEY = 10000;
static constexpr int64_t NON_FULL_LOAD_BASE_TILING_KEY = 20000;

class KvRmsNormRopeCacheTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit KvRmsNormRopeCacheTilingBase(gert::TilingContext *tillingContext)
        : TilingBaseClass(tillingContext)
    {}
    ~KvRmsNormRopeCacheTilingBase() override {}

    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t vlFp32_ = 0;
    int64_t vlFp16_ = 0;
    bool isRegbase_{false};

    int64_t kv_{DIM_NUM_ONE};
    int64_t dv_{DIM_NUM_ONE};
    int64_t dk_{DIM_NUM_ONE};
    int64_t vlen_{DIM_NUM_ONE};
    int64_t cacheLength_ = 0;
    int64_t blockSize_ = 0;
    int64_t ubBlockSize_ = 0;
    float epsilon_ = 0.0;
    float reciprocal_ = 0.0;
    bool isOutputKv_ = true;
    int64_t cosSinNeedBrc_ = 0;
    bool isPagedAttention_ = false;
    bool isMTP_ = false;
    CacheMode currentCacheMode_ = CacheMode::Norm;
    int64_t quantMode_ = 0;
    int64_t methodMode_ = 0;

    ge::DataType kvDtype_{ge::DataType::DT_FLOAT};
    int64_t kvDtypeSize_{0};
    int64_t cacheRowLimit_ = 0;

    // index 取值上界，FullLoad 与 Recompute 共用，避免两个模板对同一份 index 输入的防御口径分叉
    bool GetCacheRowLimit(int64_t &cacheRowLimit);

protected:
    bool IsCapable() override
    {
        return false;
    }
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

protected:
    std::tuple<int64_t, int64_t, int64_t, int64_t> GetShapeTuple(const gert::TilingContext *context,
                                                                 const int64_t index = 0);
    std::tuple<int64_t, int64_t, int64_t, int64_t> GetOptionalShapeTuple(const gert::TilingContext *context,
                                                                         const int64_t index = 0);
    bool IsB1SD(const gert::TilingContext *context);
    void GetMethodeMode(const gert::TilingContext *context);
    bool CheckKvValid(const gert::TilingContext *context, int64_t batchSize, int64_t numHead, int64_t seqLen,
                      int64_t headSize);
    bool CheckVValid(const gert::TilingContext *context, int64_t batchSize, int64_t numHead, int64_t seqLen,
                     int64_t headSize);
    bool CheckCosSinValid(const gert::TilingContext *context, int64_t batchSize, int64_t numHead, int64_t seqLen,
                          int64_t headSize);
    bool CheckGammaValid(const gert::TilingContext *context, int64_t headSize);
    bool CheckKCacheValid(const gert::TilingContext *context, int64_t batchSize, int64_t numHead, int64_t cacheLen,
                          int64_t headSize);
    bool CheckVCacheValid(const gert::TilingContext *context, int64_t batchSize, int64_t numHead, int64_t cacheLen,
                          int64_t headSize);
    bool CheckCacheValid(const gert::TilingContext *context, int64_t batchSize, int64_t numHead, int64_t cacheLen,
                         int64_t headSize, size_t cacheIndex, const char *cacheName);
    bool CheckKCacheValidPA(const gert::TilingContext *context, int64_t numHead, int64_t headSize);
    bool CheckVCacheValidPA(const gert::TilingContext *context, int64_t numHead, int64_t headSize);
    bool CheckIndexValid(const gert::TilingContext *context, int64_t batchSize, int64_t seqLen, int64_t pageSize,
                         CacheMode mode);
    int64_t GetQuantMode(const gert::TilingContext *context);
};

class KvRmsNormRopeCacheTilingDs : virtual public KvRmsNormRopeCacheTilingBase {
public:
    explicit KvRmsNormRopeCacheTilingDs(gert::TilingContext *tillingContext)
        : KvRmsNormRopeCacheTilingBase(tillingContext)
    {}
    ~KvRmsNormRopeCacheTilingDs() override {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

protected:
    void DoOpTilingPaBlkNz();
    bool CheckScaleValid(const gert::TilingContext *context);
    bool CheckOffsetValid(const gert::TilingContext *context);

private:
    KvRmsNormRopeCacheTilingData tilingData_;
};

class KvRmsNormRopeCacheRegbaseFullLoadTiling : virtual public KvRmsNormRopeCacheTilingBase {
public:
    explicit KvRmsNormRopeCacheRegbaseFullLoadTiling(gert::TilingContext *tillingContext)
        : KvRmsNormRopeCacheTilingBase(tillingContext)
    {}
    ~KvRmsNormRopeCacheRegbaseFullLoadTiling() override {}

    uint64_t usedCoreNum_ = 0;
    int64_t kScaleType_ = 0;
    int64_t kOffsetType_ = 0;
    int64_t vScaleType_ = 0;
    int64_t vOffsetType_ = 0;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

protected:
    bool CheckScaleOffsetShape(const gert::StorageShape *inShape, int64_t lastDim, int64_t &brcFlag);
    bool CheckInputDtype();
    bool CheckCacheIsQuant(ge::DataType &cacheDtype);
    ge::graphStatus CheckInputShapeIsEmpty();

private:
    KvRmsNormRopeCacheRegbaseFullLoadTilingData tilingData_;
};

class KvRmsNormRopeCacheRegbaseRecomputeTiling : virtual public KvRmsNormRopeCacheTilingBase {
public:
    explicit KvRmsNormRopeCacheRegbaseRecomputeTiling(gert::TilingContext *tillingContext)
        : KvRmsNormRopeCacheTilingBase(tillingContext)
    {}
    ~KvRmsNormRopeCacheRegbaseRecomputeTiling() override {}

    uint64_t usedCoreNum_ = 0;
    int64_t kScaleType_ = 0;
    int64_t kOffsetType_ = 0;
    int64_t vScaleType_ = 0;
    int64_t vOffsetType_ = 0;

    int64_t GetCacheBlockElemNum(const ge::DataType cacheDtype) const;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

protected:
    bool CheckScaleOffsetShape(const gert::StorageShape *inShape, int64_t lastDim, int64_t &brcFlag);
    bool CheckInputDtype();
    bool CheckCacheIsQuant(ge::DataType &cacheDtype);
    ge::graphStatus CheckInputShapeIsEmpty();
    int64_t FindNearestPower2(const int64_t value);

private:
    KvRmsNormRopeCacheRegbaseRecomputeTilingData tilingData_;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_KV_RMS_NORM_ROPE_CACHE_H_
