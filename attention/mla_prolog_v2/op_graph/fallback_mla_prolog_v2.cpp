/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback_mla_prolog_v2.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;

graphStatus GetMlaPrologV2InputTensor(const OpExecuteContext* ctx, MlaPrologV2FallBackParam& param)
{
    auto ret = GetMlaPrologInputTensor(ctx, param);
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "GetInputTensor failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV2OutputTensor(const OpExecuteContext* ctx, MlaPrologV2FallBackParam& param)
{
    auto ret = GetMlaPrologOutputTensor(ctx, param);
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "GetOutputTensor failed"), return GRAPH_FAILED);

    param.dequantScaleQNope = ctx->GetOutputTensor(DEQUANT_SCALE_Q_NOPE_INDEX);
    OP_CHECK_IF(param.dequantScaleQNope == nullptr, OP_LOGE("aclnnfallback", "dequantScaleQNope is null"),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV2Attr(const OpExecuteContext* ctx, MlaPrologV2FallBackParam& param)
{
    auto ret = GetMlaPrologAttr(ctx, param);
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "GetAttr failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

graphStatus MlaV2HostExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    MlaPrologV2FallBackParam param{};
    auto apiRet = GetMlaPrologInputTensor(host_api_ctx, param);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get input tesnor failed"),
                return GRAPH_FAILED);

    apiRet = GetMlaPrologV2OutputTensor(host_api_ctx, param);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get output tesnor failed"),
                return GRAPH_FAILED);

    apiRet = GetMlaPrologAttr(host_api_ctx, param);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get attr failed"), return GRAPH_FAILED);

    apiRet = EXEC_OPAPI_CMD(
        aclnnMlaPrologV2WeightNz, param.tokenX, param.weightDq, param.weightUqQr, param.weightUk, param.weightDkvKr,
        param.rmsnormGammaCq, param.rmsnormGammaCkv, param.ropeSin, param.ropeCos, param.cacheIndex, param.kvCache,
        param.krCache, param.dequantScaleX, param.dequantScaleWDq, param.dequantScaleWUqQr, param.dequantScaleWDkvKr,
        param.quantScaleCkv, param.quantScaleCkr, param.smoothScalesCq, param.rmsnormEpsilonCq, param.rmsnormEpsilonCkv,
        param.cacheMode, param.query, param.queryRope, param.kvCacheOut, param.krCacheOut, param.dequantScaleQNope);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "ret failed:%u", apiRet), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MlaPrologV2).OpExecuteFunc(MlaV2HostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
