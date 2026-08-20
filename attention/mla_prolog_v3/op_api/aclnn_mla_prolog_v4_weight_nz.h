/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_MLA_PROLOG_V4_WEIGHT_NZ_H
#define ACLNN_MLA_PROLOG_V4_WEIGHT_NZ_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算MlaPrologV4WeightNz所需的workspace大小并创建算子执行器。
 *
 * @param tokenX [IN] 公式中用于计算Query和Key的输入tensor。
 * @param weightDq [IN] 公式中用于计算Query的下采样权重矩阵。
 * @param weightUqQr [IN] 公式中用于计算Query的上采样权重矩阵和位置编码权重矩阵。
 * @param weightUk [IN] 公式中用于计算Key的上采样权重。
 * @param weightDkvKr [IN] 公式中用于计算Key的下采样权重矩阵和位置编码权重矩阵。
 * @param rmsnormGammaCq [IN] 计算c^Q的RmsNorm公式中的γ参数。
 * @param rmsnormGammaCkv [IN] 计算c^KV的RmsNorm公式中的γ参数。
 * @param ropeSin [IN]
 * 用于计算旋转位置编码的正弦参数矩阵。doRope=true时必选，doRope=false时必须与ropeCos同时为空（nullptr）。
 * @param ropeCos [IN]
 * 用于计算旋转位置编码的余弦参数矩阵。doRope=true时必选，doRope=false时必须与ropeSin同时为空（nullptr）。
 * @param kvCacheRef [IN/OUT] 用于cache索引的aclTensor，计算结果原地更新。
 * @param krCacheRef [IN/OUT] 用于key位置编码的cache，计算结果原地更新。
 * @param cacheIndexOptional [IN] 用于存储kvCache和krCache的索引。
 * @param dequantScaleXOptional [IN] tokenX的反量化参数。
 * @param dequantScaleWDqOptional [IN] weightDq的反量化参数。
 * @param dequantScaleWUqQrOptional [IN] 用于MatmulQcQr矩阵乘后反量化操作的perchannel参数。
 * @param dequantScaleWDkvKrOptional [IN] weightDkvKr的反量化参数。
 * @param quantScaleCkvOptional [IN] 用于对kvCache输出数据做量化操作的参数。
 * @param quantScaleCkrOptional [IN] 用于对krCache输出数据做量化操作的参数。
 * @param smoothScalesCqOptional [IN] 用于对RmsNormCq输出做动态量化操作的参数。
 * @param actualSeqLenOptional [IN] 表示每个batch中的序列长度，以前缀和的形式储存。
 * @param kNopeClipAlphaOptional [IN] 表示对kvCache做clip操作时的缩放因子。
 * @param rmsnormEpsilonCq [IN] 计算c^Q的RmsNorm公式中的ε参数。
 * @param rmsnormEpsilonCkv [IN] 计算c^KV的RmsNorm公式中的ε参数。
 * @param cacheModeOptional [IN]
 * 表示kvCache的模式，可选值为"PA_BSND"、"PA_NZ"、"PA_BLK_BSND"、"PA_BLK_NZ"、"BSND"、"TND"。
 * @param weightQuantMode [IN] 表示weightDq、weightUqQr、weightUk、weightDkvKr的量化模式，0表示非量化。
 * @param kvCacheQuantMode [IN]
 * 表示kvCache的量化模式，0表示非量化，1表示pertensor量化，2表示perchannel量化，3表示pertoken-pergroup量化。
 * @param queryQuantMode [IN] 表示query的量化模式，0表示非量化，1表示per-token-head量化。
 * @param ckvkrRepoMode [IN] 表示kvCache和krCache的存储模式，0表示分别存储，1表示合并存储。
 * @param quantScaleRepoMode [IN]
 * 表示量化scale的存储模式，0表示量化scale和数据分别存储，1表示合并存储作为kvCacheRef输出。
 * @param tileSize [IN] 表示pertoken-pergroup量化时每个tile的大小，仅在kvCacheQuantMode为3时有效，默认值为128。
 * @param qcQrScale [IN] 表示Query的尺度矫正系数，用户不特意指定时需要传入1.0。
 * @param kcScale [IN] 表示Key的尺度矫正系数，用户不特意指定时需要传入1.0。
 * @param doRope [IN]
 * 表示是否对queryRopeOut与krCache执行旋转位置编码（RoPE）。默认true：执行RoPE，此时ropeSin/ropeCos必须为非空有效Tensor；false：跳过RoPE并写直通结果，此时ropeSin/ropeCos必须同时为空（nullptr）。
 * @param queryOut [OUT] 公式中Query的输出tensor。
 * @param queryRopeOut [OUT] 公式中Query位置编码的输出tensor。doRope=false时输出直通结果，shape不受影响。
 * @param dequantScaleQNopeOutOptional [OUT] 公式中Query输出的反量化参数。
 * @param queryNormOutOptional [OUT] 公式中tokenX做rmsNorm后的输出tensor。
 * @param dequantScaleQNormOutOptional [OUT] queryNormOutOptional的反量化参数。
 * @param workspaceSize [OUT] 返回需在Device侧申请的workspace大小，单位为字节。
 * @param executor [OUT] 返回包含算子执行流程的执行器。
 * @return aclnnStatus 成功时返回ACLNN_SUCCESS，否则返回对应错误码。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMlaPrologV4WeightNzGetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk,
    const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin, const aclTensor *ropeCos, aclTensor *kvCacheRef, aclTensor *krCacheRef,
    const aclTensor *cacheIndexOptional, const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional, const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional, const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional, const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional, const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv, char *cacheModeOptional, int64_t weightQuantMode, int64_t kvCacheQuantMode,
    int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize, double qcQrScale,
    double kcScale, bool doRope, const aclTensor *queryOut, const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional, const aclTensor *queryNormOutOptional,
    const aclTensor *dequantScaleQNormOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief 使用aclnnMlaPrologV4WeightNzGetWorkspaceSize创建的执行器异步执行MlaPrologV4WeightNz。
 *
 * @param workspace [IN] Device侧workspace地址。workspaceSize为0时可以传入nullptr。
 * @param workspaceSize [IN] 第一阶段接口返回的Device侧workspace大小，单位为字节。
 * @param executor [IN] aclnnMlaPrologV4WeightNzGetWorkspaceSize返回的算子执行器。
 * @param stream [IN] 执行算子使用的ACL运行时Stream。
 * @return aclnnStatus 下发成功时返回ACLNN_SUCCESS，否则返回对应错误码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMlaPrologV4WeightNz(void *workspace, uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_MLA_PROLOG_V4_WEIGHT_NZ_H
