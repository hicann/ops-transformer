/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback_mla_prolog_v3.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;

graphStatus GetMlaPrologV3InputTensor(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param)
{
    param.tokenX = ctx->GetRequiredInputTensor(TOKEN_X_INDEX_V3);
    OP_CHECK_IF(param.tokenX == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "tokenX"), return GRAPH_FAILED);

    param.weightDq = ctx->GetRequiredInputTensor(WEIGHT_DQ_INDEX_V3);
    OP_CHECK_IF(param.weightDq == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "weightDq"), return GRAPH_FAILED);

    param.weightUqQr = ctx->GetRequiredInputTensor(WEIGHT_UQ_QR_INDEX_V3);
    OP_CHECK_IF(param.weightUqQr == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "weightUqQr"),
                return GRAPH_FAILED);

    param.weightUk = ctx->GetRequiredInputTensor(WEIGHT_UK_INDEX_V3);
    OP_CHECK_IF(param.weightUk == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "weightUk"), return GRAPH_FAILED);

    param.weightDkvKr = ctx->GetRequiredInputTensor(WEIGHT_DKV_KR_INDEX_V3);
    OP_CHECK_IF(param.weightDkvKr == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "weightDkvKr"),
                return GRAPH_FAILED);

    param.rmsnormGammaCq = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CQ_INDEX_V3);
    OP_CHECK_IF(param.rmsnormGammaCq == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "rmsnormGammaCq"),
                return GRAPH_FAILED);

    param.rmsnormGammaCkv = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CKV_INDEX_V3);
    OP_CHECK_IF(param.rmsnormGammaCkv == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "rmsnormGammaCkv"),
                return GRAPH_FAILED);

    param.ropeSin = ctx->GetRequiredInputTensor(ROPE_SIN_INDEX_V3);
    OP_CHECK_IF(param.ropeSin == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "ropeSin"), return GRAPH_FAILED);

    param.ropeCos = ctx->GetRequiredInputTensor(ROPE_COS_INDEX_V3);
    OP_CHECK_IF(param.ropeCos == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "ropeCos"), return GRAPH_FAILED);

    param.kvCache = ctx->GetRequiredInputTensor(KV_CACHE_INDEX_V3);
    OP_CHECK_IF(param.kvCache == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "kvCache"), return GRAPH_FAILED);

    param.krCache = ctx->GetRequiredInputTensor(KR_CACHE_INDEX_V3);
    OP_CHECK_IF(param.krCache == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "krCache"), return GRAPH_FAILED);

    param.cacheIndex = ctx->GetOptionalInputTensor(CACHE_INDEX_V3);
    param.dequantScaleX = ctx->GetOptionalInputTensor(DEQUANT_SCALE_X_INDEX_V3);
    param.dequantScaleWDq = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DQ_INDEX_V3);
    param.dequantScaleWUqQr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_UQ_QR_INDEX_V3);
    param.dequantScaleWDkvKr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DKV_KR_INDEX_V3);
    param.quantScaleCkv = ctx->GetOptionalInputTensor(QUANT_SCALE_CKV_INDEX_V3);
    param.quantScaleCkr = ctx->GetOptionalInputTensor(QUANT_SCALE_CKR_INDEX_V3);
    param.smoothScalesCq = ctx->GetOptionalInputTensor(SMOOTH_SCALES_CQ_INDEX_V3);
    param.actualAeqLen = ctx->GetOptionalInputTensor(ACTUAL_SEQ_LEN_INDEX_V3);
    param.kNopeClipAlpha = ctx->GetOptionalInputTensor(K_NOPE_CLIP_ALPHA_INDEX_V3);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV3OutputTensor(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param)
{
    param.query = ctx->GetOutputTensor(QUERY_INDEX);
    OP_CHECK_IF(param.query == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "query"), return GRAPH_FAILED);

    param.queryRope = ctx->GetOutputTensor(QUERY_ROPE_INDEX);
    OP_CHECK_IF(param.queryRope == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "queryRope"),
                return GRAPH_FAILED);

    param.kvCacheOut = ctx->GetOutputTensor(KV_CACHE_OUT_INDEX);
    OP_CHECK_IF(param.kvCacheOut == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "kvCacheOut"),
                return GRAPH_FAILED);

    param.krCacheOut = ctx->GetOutputTensor(KR_CACHE_OUT_INDEX);
    OP_CHECK_IF(param.krCacheOut == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "krCacheOut"),
                return GRAPH_FAILED);

    param.dequantScaleQNope = ctx->GetOutputTensor(DEQUANT_SCALE_Q_NOPE_INDEX);
    OP_CHECK_IF(param.dequantScaleQNope == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "dequantScaleQNope"),
                return GRAPH_FAILED);

    param.queryNorm = ctx->GetOutputTensor(QUERY_NORM_INDEX);
    OP_CHECK_IF(param.queryNorm == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "queryNorm"),
                return GRAPH_FAILED);

    param.dequantScaleQNorm = ctx->GetOutputTensor(DEQUANT_SCALE_Q_NORM_INDEX);
    OP_CHECK_IF(param.dequantScaleQNorm == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "dequantScaleQNorm"),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV3Attr(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param)
{
    auto attrs = ctx->GetAttrs();
    const double *getRmsnormEpsilonCq = attrs->GetAttrPointer<double>(ATTR_RMSNORM_EPSILON_CQ_INDEX);
    const double *getRmsnormEpsilonCkv = attrs->GetAttrPointer<double>(ATTR_RMSNORM_EPSILON_CKV_INDEX);
    const char *getCacheMode = attrs->GetAttrPointer<char>(ATTR_CACHE_MODE_INDEX);
    const bool *getQueryNormFlag = attrs->GetAttrPointer<bool>(ATTR_QUERY_NORM_FLAG_INDEX);
    const int64_t *getWeightQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_WEIGHT_QUANT_MODE_INDEX);
    const int64_t *getKvCacheQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_KV_CACHE_QUANT_MODE_INDEX);
    const int64_t *getQueryQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    const int64_t *getCkvkrRepoMode = attrs->GetAttrPointer<int64_t>(ATTR_CKVKR_REPO_MODE_INDEX);
    const int64_t *getQuantScaleRepoMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_SCALE_REPO_MODE_INDEX);
    const int64_t *getTileSize = attrs->GetAttrPointer<int64_t>(ATTR_TILE_SIZE_INDEX);
    const double *getQcQrScale = attrs->GetAttrPointer<double>(ATTR_QC_QR_SCALE_INDEX);
    const double *getKcScale = attrs->GetAttrPointer<double>(ATTR_KC_SCALE_INDEX);
    param.rmsnormEpsilonCq = *getRmsnormEpsilonCq;
    param.rmsnormEpsilonCkv = *getRmsnormEpsilonCkv;
    param.cacheMode = getCacheMode;
    param.queryNormFlag = *getQueryNormFlag;
    param.weightQuantMode = *getWeightQuantMode;
    param.kvCacheQuantMode = *getKvCacheQuantMode;
    param.queryQuantMode = *getQueryQuantMode;
    param.ckvkrRepoMode = *getCkvkrRepoMode;
    param.quantScaleRepoMode = *getQuantScaleRepoMode;
    param.tileSize = *getTileSize;
    param.qcQrScale = getQcQrScale ? *getQcQrScale : 1.0;
    param.kcScale = getKcScale ? *getKcScale : 1.0;
    // 入图场景下旧图不携带 do_rope，attr 数组可能不足 13 个，先判越界再取值，缺省视为开启
    param.doRope = true;
    if (attrs->GetAttrNum() > ATTR_DO_ROPE_INDEX) {
        const bool *getDoRope = attrs->GetAttrPointer<bool>(ATTR_DO_ROPE_INDEX);
        if (getDoRope != nullptr) {
            param.doRope = *getDoRope;
        }
    }
    return GRAPH_SUCCESS;
}

graphStatus MlaV3HostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV3", "host_api_ctx"),
                return GRAPH_FAILED);

    MlaPrologV3FallBackParam param{};
    GetMlaPrologV3Attr(host_api_ctx, param);

    auto apiRet = GetMlaPrologV3InputTensor(host_api_ctx, param);
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    apiRet = GetMlaPrologV3OutputTensor(host_api_ctx, param);
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    auto attrs = host_api_ctx->GetAttrs();
    if (attrs->GetAttrNum() > ATTR_DO_ROPE_INDEX) {
        // attrNum > 12,说明是入图接口，走aclnnV4
        // doRope=false 时图上是空 tensor 占位，V4 要求 ropeSin/ropeCos 同时为 nullptr
        const gert::Tensor *ropeSin = param.doRope ? param.ropeSin : nullptr;
        const gert::Tensor *ropeCos = param.doRope ? param.ropeCos : nullptr;
        apiRet = EXEC_OPAPI_CMD(
            aclnnMlaPrologV4WeightNz, param.tokenX, param.weightDq, param.weightUqQr, param.weightUk, param.weightDkvKr,
            param.rmsnormGammaCq, param.rmsnormGammaCkv, ropeSin, ropeCos, param.kvCache, param.krCache,
            param.cacheIndex, param.dequantScaleX, param.dequantScaleWDq, param.dequantScaleWUqQr,
            param.dequantScaleWDkvKr, param.quantScaleCkv, param.quantScaleCkr, param.smoothScalesCq,
            param.actualAeqLen, param.kNopeClipAlpha, param.rmsnormEpsilonCq, param.rmsnormEpsilonCkv, param.cacheMode,
            param.weightQuantMode, param.kvCacheQuantMode, param.queryQuantMode, param.ckvkrRepoMode,
            param.quantScaleRepoMode, param.tileSize, param.qcQrScale, param.kcScale, param.doRope, param.query,
            param.queryRope, param.dequantScaleQNope, param.queryNorm, param.dequantScaleQNorm);
    } else {
        apiRet = EXEC_OPAPI_CMD(
            aclnnMlaPrologV3WeightNz, param.tokenX, param.weightDq, param.weightUqQr, param.weightUk, param.weightDkvKr,
            param.rmsnormGammaCq, param.rmsnormGammaCkv, param.ropeSin, param.ropeCos, param.kvCache, param.krCache,
            param.cacheIndex, // cacheIndex当前为可选参数
            param.dequantScaleX, param.dequantScaleWDq, param.dequantScaleWUqQr, param.dequantScaleWDkvKr,
            param.quantScaleCkv, param.quantScaleCkr, param.smoothScalesCq, param.actualAeqLen, param.kNopeClipAlpha,
            param.rmsnormEpsilonCq, param.rmsnormEpsilonCkv, param.cacheMode, param.weightQuantMode,
            param.kvCacheQuantMode, param.queryQuantMode, param.ckvkrRepoMode, param.quantScaleRepoMode, param.tileSize,
            param.qcQrScale, param.kcScale, param.query, param.queryRope, param.dequantScaleQNope, param.queryNorm,
            param.dequantScaleQNorm);
    }

    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPL_OP(MlaPrologV3).OpExecuteFunc(MlaV3HostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
