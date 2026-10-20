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
 * \file aclnn_flash_mla.cpp
 * \brief
 */

#include "aclnn/aclnn_base.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// aclnnInner* 实现由构建工具自动生成（build/autogen/inner/aclnnInner_flash_mla.cpp）
// 并链接进 libcust_opapi.so，此处仅作前置声明（合并自原 aclnn_flash_mla_inner.h）。
extern aclnnStatus aclnnInnerFlashMlaGetWorkspaceSize(
    const aclTensor *q, const aclTensor *kCache, const aclTensor *blockTableOptional,
    const aclTensor *cacheSeqlensOptional, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
    const aclTensor *attnMaskOptional, const aclTensor *metadataOptional, int64_t headDimV, double softmaxScale,
    int64_t maskMode, int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ, const char *layoutKv,
    const char *layoutOut, int64_t returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLse,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerFlashMla(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      const aclrtStream stream);

namespace {

aclnnStatus FlashMlaProcessSoftmaxLse(int64_t returnSoftmaxLse, const aclTensor *softmaxLse,
                                      const aclTensor *&tempTensor, const aclTensor *&placeHolder)
{
    if (returnSoftmaxLse == false) {
        std::vector<int64_t> shape = {0};
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), nullptr);
        if (tempTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create placeholder tensor for softmaxLse failed.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        placeHolder = tempTensor;
    } else {
        if (softmaxLse == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                    "When returnSoftmaxLse is enabled, softmaxLse must be provided, but got nullptr.");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        placeHolder = softmaxLse;
    }
    return ACLNN_SUCCESS;
}

} // namespace

/**
 * @brief aclnnFlashMla的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train_infer
 *
 * @param q                   [IN]  query tensor。数据类型FLOAT16/BFLOAT16，ND格式。
 *                                  layout由layoutQ决定，支持TND。
 *                                  TND: shape=(T_q, N_q, D)，T_q为总token数，
 *                                  D=576=nope(512)+rope(64)，rope已合并在q中。
 * @param kCache              [IN]  key-cache tensor。数据类型与q一致，layout由layoutKv决定。
 *                                  PA_NZ: shape=(NumBlocks, N_kv, BlockSize/16, D, 16)，
 *                                  分页注意力PA_NZ格式，D=576=nope(512)+rope(64)，rope已合并在k_cache中。
 *                                  MLA场景固定N_kv=1（单latent KV头）。
 * @param blockTableOptional  [IN]  可选。分页KV缓存的块映射表，数据类型INT32。
 *                                  shape=(B, MaxBlocksPerSeq)，与layoutKv=PA_Nz配合使用。
 * @param cacheSeqlensOptional [IN] 可选。各batch的KV cache实际长度，数据类型INT32。
 *                                  shape=(B,)，表征每个batch的KV序列长度（seqused_kv语义）。
 * @param cuSeqlensQOptional  [IN]  可选。query的累积序列长度（前缀和），数据类型INT32。
 *                                  shape=(B+1,)，cu_seqlens_q[i+1]-cu_seqlens_q[i]为第i个sample的q序列长度。
 *                                  TND layout变长场景时有效。与seqused_q互斥。
 * @param sequsedQOptional    [IN]  可选。各batch中query的实际序列长度，数据类型INT32。
 *                                  shape=(B,)，padded batch模式下有效。与cuSeqlensQOptional互斥。
 * @param attnMaskOptional    [IN]  可选。attnmask的参数，数据类型INT8。
 * @param metadataOptional    [IN]  必传。预计算的tiling切分方案，数据类型INT32，由配套metadata算子生成，不能传nullptr。
 *                                  tiling侧直接使用该元数据中的预计算切分方案。
 * @param headDimV            [IN]  ATTR可选。k_cache中nope（value）段的宽度。数据类型INT，默认值512。
 *                                  q/k_cache最后维=headDimV+64（rope宽）。
 * @param softmaxScale        [IN]  ATTR可选。softmax缩放系数，对应老接口的scaleValue。
 *                                  数据类型DOUBLE。默认值0.0表示使用1/sqrt(D)。
 * @param maskMode            [IN]  ATTR可选。掩码模式，对应老接口的sparseMode。数据类型INT。
 *                                  0: 不使用掩码；1: 因果掩码（下三角）。
 * @param maxSeqlenQ          [IN]  ATTR可选。query的最大序列长度。数据类型INT，默认值-1。
 * @param maxSeqlenKV         [IN]  ATTR可选。kv的最大序列长度。数据类型INT，默认值-1。
 * @param layoutQ             [IN]  ATTR可选。query的数据布局，支持"TND"（大小写不敏感）。
 * @param layoutKv            [IN]  ATTR可选。key/value的数据布局，支持"PA_NZ"。
 * @param layoutOut           [IN]  ATTR可选。输出的数据布局，支持"NTD"（大小写不敏感）。
 * @param returnSoftmaxLse    [IN]  ATTR可选。是否输出softmax_lse。INT类型，1表示输出，0表示不输出。
 *                                  训练正向传播时置1，推理时置0。
 * @param attnOut        [OUT] 必选输出。attention计算结果，数据类型与q一致，layout由layoutOut决定。
 * @param softmaxLseOptional  [OUT] 可选输出。softmax的log-sum-exp值，FLOAT32类型。
 *                                  returnSoftmaxLse=1时有效。
 * @param workspaceSize       [OUT] workspace大小（字节数）。
 * @param executor            [OUT] op执行器句柄，供第二段接口使用。
 * @return aclnnStatus 执行状态。ACLNN_SUCCESS表示成功。
 */
// 第一段接口：计算workspace大小
aclnnStatus aclnnFlashMlaGetWorkspaceSize(const aclTensor *q, const aclTensor *kCache,
                                          const aclTensor *blockTableOptional, const aclTensor *cacheSeqlensOptional,
                                          const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
                                          const aclTensor *attnMaskOptional, const aclTensor *metadataOptional,
                                          int64_t headDimV, double softmaxScale, int64_t maskMode, int64_t maxSeqlenQ,
                                          int64_t maxSeqlenKV, const char *layoutQ, const char *layoutKv,
                                          const char *layoutOut, int64_t returnSoftmaxLse, const aclTensor *attnOut,
                                          const aclTensor *softmaxLseOptional, uint64_t *workspaceSize,
                                          aclOpExecutor **executor)
{
    OP_LOGD("start aclnnFlashMlaGetWorkspaceSize");

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;

    aclnnStatus ret = FlashMlaProcessSoftmaxLse(returnSoftmaxLse, softmaxLseOptional, tempTensor, placeHolder);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = aclnnInnerFlashMlaGetWorkspaceSize(
        q, kCache, blockTableOptional, cacheSeqlensOptional, cuSeqlensQOptional, sequsedQOptional, attnMaskOptional,
        metadataOptional, headDimV, softmaxScale, maskMode, maxSeqlenQ, maxSeqlenKV, layoutQ, layoutKv, layoutOut,
        returnSoftmaxLse, attnOut, placeHolder, workspaceSize, executor);

    // 销毁占位符
    if (returnSoftmaxLse == 0) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

/**
 * @brief aclnnFlashMla的第二段接口，用于执行计算。
 * @param workspace       [IN] 由第一段接口计算得到的workspace设备内存指针。
 * @param workspaceSize   [IN] workspace大小（字节数）。
 * @param executor        [IN] 第一段接口输出的op执行器句柄。
 * @param stream          [IN] 用于执行计算的acl stream。
 * @return aclnnStatus 执行状态。
 */
// 第二段接口：执行计算
aclnnStatus aclnnFlashMla(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerFlashMla(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
