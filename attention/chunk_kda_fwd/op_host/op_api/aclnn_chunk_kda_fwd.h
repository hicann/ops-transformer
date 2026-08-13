/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_CHUNK_KDA_FWD_H
#define OP_API_INC_ACLNN_CHUNK_KDA_FWD_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnChunkKdaFwd的第一段接口,根据具体的计算流程,计算workspace大小
 * @domain aclnn_ops_infer
 *
 * @param [in] q: Query输入。BSND为(B, S, H, K)，BNSD为(B, H, S, K)，TND为(T, H, K)，NTD为(H, T, K)，支持FLOAT16、BFLOAT16
 * @param [in] k: Key输入。shape和数据类型必须与q相同，支持FLOAT16、BFLOAT16
 * @param [in] v: Value输入。BSND为(B, S, HV, V)，BNSD为(B, HV, S, V)，TND为(T, HV, V)，NTD为(HV, T, V)，数据类型必须与q相同
 * @param [in] g: raw gate或已激活的自然对数gate。BSND为(B, S, HV, K)，BNSD为(B, HV, S, K)，TND为(T, HV, K)，NTD为(HV, T, K)，支持FLOAT、BFLOAT16
 * @param [in] beta: Delta系数。BSND为(B, S, HV)，BNSD为(B, HV, S)，TND为(T, HV)，NTD为(HV, T)，支持FLOAT、BFLOAT16
 * @param [in] aLogOptional: gate衰减参数，shape为(HV)，useGateInKernel=true时必须传入，支持FLOAT，可传nullptr
 * @param [in] dtBiasOptional: gate偏置，shape为(HV×K)，支持FLOAT，可传nullptr
 * @param [in] initialStateOptional: 初始状态。stateVFirst=false时shape为(N, HV, K, V)，否则为(N, HV, V, K)，支持FLOAT，可传nullptr
 * @param [in] cuSeqlensOptional: 变长序列累计长度，shape为(N+1)，首元素为0，末元素为T或S，元素单调不减，支持INT64，可传nullptr
 * @param [in] chunkIndicesOptional: chunk索引，shape为(2×NC)，按(seq_id, chunk_id)保存，必须采用sequence-major canonical顺序，
 *            传入时必须同时传入cuSeqlensOptional，支持INT64，可传nullptr
 * @param [in] layout: 输入布局，支持"BSND"、"BNSD"、"TND"、"NTD"，默认值为"BSND"
 * @param [in] scale: Query缩放系数，通常取K^(-0.5)
 * @param [in] chunkSize: chunk大小，支持64、128，默认值为64
 * @param [in] safeGate: 是否使用有界gate，默认值为false
 * @param [in] lowerBound: 有界gate下界，safeGate=true时取值范围为[-5, 0)，默认值为-5.0
 * @param [in] useGateInKernel: 是否在kernel内由raw gate计算激活，默认值为false
 * @param [in] stateVFirst: 是否将状态张量末两维排列为(V, K)，默认值为false
 * @param [in] attnOut: Attention输出tensor。rank-4输入固定为(B, S, HV, V)，rank-3输入固定为(T, HV, V)，支持FLOAT16、BFLOAT16
 * @param [in] finalStateOut: 最终状态tensor。stateVFirst=false时shape为(N, HV, K, V)，否则为(N, HV, V, K)，支持FLOAT，不导出时传nullptr
 * @param [in] gkOut: chunk-local log2累计gate。rank-4输入为(B, HV, S, K)，rank-3输入为(HV, T, K)，支持FLOAT，不导出时传nullptr
 * @param [in] aqkOut: chunk内Query-Key系数矩阵。rank-4输入为(B, HV, S, C)，rank-3输入为(HV, T, C)，支持FLOAT16、BFLOAT16
 * @param [in] akkOut: chunk内下三角求逆结果。rank-4输入为(B, HV, S, C)，rank-3输入为(HV, T, C)，支持FLOAT16、BFLOAT16
 * @param [in] wOut: 供反向使用的W中间量。rank-4输入为(B, HV, S, K)，rank-3输入为(HV, T, K)，支持FLOAT16、BFLOAT16，不导出时传nullptr
 * @param [in] uOut: 供反向使用的U中间量。rank-4输入为(B, HV, S, V)，rank-3输入为(HV, T, V)，支持FLOAT16、BFLOAT16，不导出时传nullptr
 * @param [in] qgOut: 供反向使用的gate缩放Query。rank-4输入为(B, HV, S, K)，rank-3输入为(HV, T, K)，支持FLOAT16、BFLOAT16，不导出时传nullptr
 * @param [in] kgOut: 供反向使用的gate缩放Key。rank-4输入为(B, HV, S, K)，rank-3输入为(HV, T, K)，支持FLOAT16、BFLOAT16，不导出时传nullptr
 * @param [in] vNewOut: 供反向使用的Value中间量。rank-4输入为(B, HV, S, V)，rank-3输入为(HV, T, V)，支持FLOAT16、BFLOAT16，不导出时传nullptr
 * @param [in] hOut: 公开chunk状态。rank-4输入为(B, NC, HV, K, V)，rank-3输入为(NC, HV, K, V)，stateVFirst=true时交换末两维，
 *            支持FLOAT16、BFLOAT16，不导出时传nullptr
 * @param [out] workspaceSize: 返回计算所需workspace大小
 * @param [out] executor: 返回op执行器
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnChunkKdaFwd的第二段接口,用于执行计算
 *
 * @param [in] workspace: 工作空间地址
 * @param [in] workspaceSize: 工作空间大小，由第一段接口aclnnChunkKdaFwdGetWorkspaceSize获取
 * @param [in] executor: op执行器，包含了算子计算流程
 * @param [in] stream: acl stream流
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnChunkKdaFwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
