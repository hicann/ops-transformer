/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_tiling.h
 * \brief
 */

#ifndef MLA_PROLOG_TILING_H
#define MLA_PROLOG_TILING_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <sstream>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../op_kernel/mla_prolog_template_tiling_key.h"
#include "../op_kernel/mla_prolog_tiling_data.h"

#ifdef ASCENDC_OP_TEST
#define MLA_EXTERN_C extern "C"
#else
#define MLA_EXTERN_C
#endif

namespace optiling {

// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_DQ_INPUT_INDEX = 1;
constexpr uint32_t WEIGHT_UQ_QR_INPUT_INDEX = 2;
constexpr uint32_t WEIGHT_UK_INPUT_INDEX = 3;
constexpr uint32_t WEIGHT_DKV_KR_INPUT_INDEX = 4;
constexpr uint32_t RMSNORM_GAMMA_CQ_INPUT_INDEX = 5;
constexpr uint32_t RMS_NORM_GAMMA_CKV_INPUT_INDEX = 6;
constexpr uint32_t ROPE_SIN_INPUT_INDEX = 7;
constexpr uint32_t ROPE_COS_INPUT_INDEX = 8;
constexpr uint32_t CACHE_INDEX_INPUT_INDEX = 9;
constexpr uint32_t KV_CACHE_INPUT_INDEX = 10;
constexpr uint32_t KR_CACHE_INPUT_INDEX = 11;

constexpr uint32_t KV_CACHE_INPUT_INDEX_V3 = 9;
constexpr uint32_t KR_CACHE_INPUT_INDEX_V3 = 10;
constexpr uint32_t CACHE_INDEX_INPUT_INDEX_V3 = 11;

// INPUT(OPTION)
constexpr uint32_t DEQUANT_SCALE_X_INDEX = 12;
constexpr uint32_t DEQUANT_SCALE_W_DQ_INDEX = 13;
constexpr uint32_t DEQUANT_SCALE_W_UQ_QR_INDEX = 14;
constexpr uint32_t DEQUANT_SCALE_W_DKV_KR_INDEX = 15;
constexpr uint32_t QUANT_SCALE_CKV_INDEX = 16;
constexpr uint32_t QUANT_SCALE_CKR_INDEX = 17;
constexpr uint32_t SMOOTH_SCALES_CQ_INDEX = 18;

// OUTPUT
constexpr uint32_t QUERY_OUTPUT_INDEX = 0;
constexpr uint32_t QUERY_ROPE_OUTPUT_INDEX = 1;
constexpr uint32_t KV_CACHE_OUT_OUTPUT_INDEX = 2;
constexpr uint32_t KR_CACHE_OUT_OUTPUT_INDEX = 3;
constexpr uint32_t DEQUANT_SCALE_Q_NOPE_OUTPUT_INDEX = 4;

// ATTR
constexpr uint32_t RMS_NORM_EPSILON_CQ_ATTR_INDEX = 0;
constexpr uint32_t RMS_NORM_EPSILON_CKV_ATTR_INDEX = 1;
constexpr uint32_t CACHE_MODE_ATTR_INDEX = 2;
constexpr uint32_t QUERY_NORM_FLAG_ATTR_INDEX = 3;
constexpr uint32_t WEIGHT_QUANT_MODE_ATTR_INDEX = 4;
constexpr uint32_t KV_CACHE_QUANT_MODE_ATTR_INDEX = 5;
constexpr uint32_t QUERY_QUANT_MODE_ATTR_INDEX = 6;
constexpr uint32_t CKVKR_REPO_MODE_ATTR_INDEX = 7;
constexpr uint32_t QUANT_SCALE_REPO_MODE_ATTR_INDEX = 8;
constexpr uint32_t TILE_SIZE_ATTR_INDEX = 9;
constexpr uint32_t K_NOPE_CLIP_ALPHA_ATTR_INDEX = 10;
constexpr uint32_t QC_QR_SCALE_ATTR_INDEX = 11;
constexpr uint32_t KC_SCALE_ATTR_INDEX = 12;

constexpr uint32_t MLA_PROLOG_DIM_INDEX_0 = 0;
constexpr uint32_t MLA_PROLOG_DIM_INDEX_1 = 1;
constexpr uint32_t MLA_PROLOG_DIM_INDEX_2 = 2;
constexpr uint32_t MLA_PROLOG_DIM_INDEX_3 = 3;

constexpr uint32_t MLA_PROLOG_DIM_NUM_0 = 0;
constexpr uint32_t MLA_PROLOG_DIM_NUM_1 = 1;
constexpr uint32_t MLA_PROLOG_DIM_NUM_2 = 2;
constexpr uint32_t MLA_PROLOG_DIM_NUM_3 = 3;
constexpr uint32_t MLA_PROLOG_DIM_NUM_4 = 4;

constexpr char CACHE_MODE_PA_BSND[] {"PA_BSND"};
constexpr char CACHE_MODE_PA_NZ[] {"PA_NZ"};
constexpr char V1_OP_NAME[] {"MlaProlog"};

struct MlaPrologBaseShapeInfo {
    uint32_t bSize = 0;     // B
    uint32_t s1Size = 0;      // S1
    uint32_t tSize = 0;     // T
    uint32_t s2Size = 0;      // S2
    uint32_t heSize = 0;     // He
    uint32_t hcqSize = 0;    // Hcq
    uint32_t hckvSize = 0;   // Hckv
    uint32_t headSizeQc = 0;    // N * D
    uint32_t headSizeQr = 0;    // N * Dr
    uint32_t headSizeUqQr = 0;  // N * (D + Dr)
    uint32_t nSize = 0;   // N
    uint32_t nkvSize = 0; // Nkv
    uint32_t dSize = 0; // D
    uint32_t blockNum = 0;
    uint32_t blockSize = 0;
    uint32_t drSize = 0;   // Dr
};

enum class CACHE_MODE:uint8_t {
    BNSD = 0,
    PA_BSND = 1,
    PA_NZ = 2,
};

enum class EMPTY_TENSOR_MODE:uint8_t {
    NON_EMPTY = 0,
    EMPTY_CACHE = 1,
    EMPTY_QUERY = 2
};

enum class QUANT_MODE:int8_t {
    NO_QUANT = -1,
    PARTIAL_QUANT_KV_NO_QUANT = 0,
    PARTIAL_QUANT_KV_QUANT = 1,
    FULL_QUANT_KV_NO_QUANT = 2,
    FULL_QUANT_KV_QUANT = 3
};

struct MlaPrologScenarioInfo {
    bool isV1Flag_;
    bool batchSeqFusedFlag_;
    QUANT_MODE quantMode_;
    CACHE_MODE cacheMode_;
    EMPTY_TENSOR_MODE emptyTensorMode_;
};

struct MlaPrologCompileInfo {
    int64_t core_num;
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
    const gert::Tensor *tensor;
};

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t NUM_BYTES_INT8 = 1;
constexpr uint32_t NUM_BYTES_INT32 = 4;
constexpr uint32_t NUM_BYTES_FP32 = 4;

constexpr uint32_t GROUP_COMPUTE_CUBE_NUM_PER_GROUP = 8U;
constexpr uint32_t HIGH_THROUGHPUT__D_SIZE = 128U;
constexpr uint32_t GROUP_COMPUTE_T_SIZE = 1U;
constexpr uint32_t GROUP_COMPUTE_NKV_SIZE = 8U;
constexpr uint32_t GROUP_COMPUTE_MIN_AIC_NUM = 16U;
constexpr uint32_t GROUP_COMPUTE_MIN_AIV_NUM = 32U;
constexpr uint32_t GROUP_COMPUTE_N_SIZE = 8U;

constexpr uint64_t MLA_PROLOG_TILINGKEY_BASE_OFFSET = uint64_t(10000000000000000UL);          // 10^16
constexpr uint64_t MLA_PROLOG_TYPE_OFFSET = uint64_t(10UL);
constexpr uint64_t MLA_PROLOG_QUANT_TYPE_OFFSET = uint64_t(100UL);
constexpr uint64_t MLA_PROLOG_ENABLE_DEQUANT_OPT_OFFSET = uint64_t(1000UL);
constexpr uint64_t MLA_PROLOG_ENABLE_GROUP_COMPUTE_OPT_OFFSET = uint64_t(10000UL);
constexpr uint64_t MLA_PROLOG_EMPTY_TENSOR_MODE_OFFSET = uint64_t(100000UL);

struct MlaPrologContext {
    const char *opName;
    const char *opType;
    fe::PlatFormInfos *platformInfo;
    RequiredParaInfo tokenX;
    RequiredParaInfo weightDq;
    RequiredParaInfo weightUqQr;
    RequiredParaInfo weightUk;
    RequiredParaInfo weightDkvKr;
    RequiredParaInfo rmsnormGammaCq;
    RequiredParaInfo rmsnormGammaCkv;
    RequiredParaInfo ropeSin;
    RequiredParaInfo ropeCos;
    RequiredParaInfo cacheIndex;
    RequiredParaInfo kvCache;
    RequiredParaInfo krCache;
    OptionalParaInfo dequantScaleX;
    OptionalParaInfo dequantScaleWDq;
    OptionalParaInfo dequantScaleWUqQr;
    OptionalParaInfo dequantScaleWDkvKr;
    OptionalParaInfo quantScaleCkv;
    OptionalParaInfo quantScaleCkr;
    OptionalParaInfo smoothScalesCq;
    RequiredParaInfo query;
    RequiredParaInfo queryRope;
    RequiredParaInfo kvCacheOut;
    RequiredParaInfo krCacheOut;
    OptionalParaInfo dequantScaleQNope;

    const float *rmsNormEspilonCq;
    const float *rmsNormEspilonCkv;
    const char *cacheMode;
    const float *qcQrScale;
    const float *kcScale;

    size_t *workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

class MlaPrologTiling {
public:
    MlaPrologTiling() = default;
    ~MlaPrologTiling() = default;

    ge::graphStatus RunBigKernelTiling(MlaPrologContext &context, MlaPrologTilingData* tilingData);
    ge::graphStatus MlaPrologSetTilingData(gert::TilingContext &context, MlaPrologTilingData* tilingData);
    static ge::graphStatus ConvertContext(gert::TilingContext &context, MlaPrologContext &mlaPrologContext);

private:
    static void ConvertRequiredParams(gert::TilingContext &context, MlaPrologContext &mlaPrologContext);
    static void ConvertOptionalParams(gert::TilingContext &context, MlaPrologContext &mlaPrologContext);
    ge::graphStatus GetNpuInfo();
    ge::graphStatus SetScenarioInfo();
    QUANT_MODE GetQuantizationMode() const;
    ge::graphStatus SetShapeInfo();
    ge::graphStatus ProcessBaseInputs();
    ge::graphStatus FillTiling();
    ge::graphStatus FillMatmul1Tiling();
    ge::graphStatus FillMatmul2Tiling();
    ge::graphStatus FillMatmul3Tiling();
    ge::graphStatus FillMatmul4Tiling();
    uint32_t CalcSingleCoreN(uint32_t n, uint32_t coreNum, uint32_t alignNum=16) const;
    bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType);
    ge::graphStatus CalcWorkSpace();
    ge::graphStatus GenTilingKey() const;

    MlaPrologBaseShapeInfo baseShapeInfo_;
    MlaPrologScenarioInfo scenarioInfo_;

    uint32_t stepBatchSize_ = 0;
    uint32_t stepNumHeadDequant_ = 0;

    uint32_t mm1BlockNum_ = 0;
    uint32_t mm2BlockNum_ = 0;
    uint32_t mm3BlockNum_ = 0;
    uint32_t mm4BlockNum_ = 0;
    uint32_t vectorBlockNum_ = 0;

    uint32_t singlecoreHeadSizeCq_ = 0;
    uint32_t singlecoreHeadSizeQcQr_ = 0;
    uint32_t singlecoreHeadSizeCkvKr_ = 0;
    uint32_t singlecoreNumHeadSize_ = 0;

    float reciprocalCq_ = 0.00001f;
    float epsilonCq_ = 1.0;
    float reciprocalCkv_ = 0.00001f;
    float epsilonCkv_ = 1.0;
    float qcQrScale_ = 1.0;
    float kcScale_ = 1.0;

    ge::DataType mmDateType_ = ge::DT_BF16;
    bool enableDequantOpt_ = false;
    bool enableGroupComputeOpt_ = false;    // 低延时场景算例分组标记

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;
    size_t workspaceSize_ = 0;

    MlaPrologContext *context_ = nullptr;
    MlaPrologBaseParams *baseParams_ = nullptr;
};

ge::graphStatus TilingPrepareForMlaProlog(gert::TilingParseContext *context);
MLA_EXTERN_C ge::graphStatus TilingMlaProlog(gert::TilingContext *context);
} // optiling

#endif // MLA_PROLOG_TILING_H
