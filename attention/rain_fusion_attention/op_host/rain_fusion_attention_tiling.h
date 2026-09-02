/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RAIN_FUSION_ATTENTION_TILING_H
#define RAIN_FUSION_ATTENTION_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

// RainFusionAttention Tiling数据定义
BEGIN_TILING_DATA_DEF(RainFusionAttentionTilingData)
// 基础参数
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, maskType);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, totalQBlocks);   // T: 所有batch中Q方向切块的总数
TILING_DATA_FIELD_DEF(uint32_t, firstQBlockNum); // T: 所有batch中Q方向切块的总数

// 稀疏分块参数 (blockShape)
TILING_DATA_FIELD_DEF(uint64_t, blockShapeX); // block的x维度(Q方向)
TILING_DATA_FIELD_DEF(uint64_t, blockShapeY); // block的y维度(KV方向)

// selectIdx相关参数
TILING_DATA_FIELD_DEF(uint32_t, maxKvBlockNum); // 最大KV块数量（selectIdx的最后一维）

// query Layout: 0=TND, 1=BNSD
TILING_DATA_FIELD_DEF(uint32_t, queryLayout);

// KV Cache Layout: 0=TND, 1=BNSD
TILING_DATA_FIELD_DEF(uint32_t, kvCacheLayout);

// BNSD格式的最大序列长度（用于计算stride）
// 当actualSeqLengths为nullptr时，maxQSeqlen也用作统一的qseqlen值
TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen); // BNSD格式Q的第三维（S维度），或统一的qseqlen值
// 当actualSeqLengthsKv为nullptr时，maxKvSeqlen也用作统一的kvseqlen值
TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen);        // BNSD格式KV的第三维（S维度），或统一的kvseqlen值
TILING_DATA_FIELD_DEF(uint32_t, useUniformQSeqlen);  // 是否使用统一的qseqlen值（1=是，0=否）
TILING_DATA_FIELD_DEF(uint32_t, useUniformKvSeqlen); // 是否使用统一的kvseqlen值（1=是，0=否）

// TilingKey for kernel dispatch (生成在tiling层)
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);

// Workspace大小
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize);
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
TILING_DATA_FIELD_DEF(uint64_t, updateSize);
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RainFusionAttention, RainFusionAttentionTilingData)

// RainFusionAttention编译信息
struct RainFusionAttentionCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;

    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion;
};

// 输入参数信息
struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// KVCache Layout枚举
enum RFAKvCacheLayout : uint32_t {
    TND = 0, // [T, N, D] format
    BNSD = 1 // [B, N, S, D] format
};

// Q Input Layout枚举
enum RFAQInputLayout : uint32_t {
    TND_Q = 0, // [T, N, D] format
    BNSD_Q = 1 // [B, N, S, D] format
};

// Tiling类
class RFATiling {
public:
    RFATiling() = default;
    ~RFATiling() = default;

    ge::graphStatus GetRFATiling(gert::TilingContext *rfaContext, RainFusionAttentionTilingData &tilingData);
    ge::graphStatus RFASetTilingData(gert::TilingContext *context, RainFusionAttentionTilingData &tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *rfaContext);
    ge::graphStatus ProcessInput(gert::TilingContext *rfaContext);
    ge::graphStatus CheckAttr(gert::TilingContext *rfaContext);
    ge::graphStatus CheckKvCacheLayout(gert::TilingContext *rfaContext);
    ge::graphStatus CalculateTaskSplit(gert::TilingContext *rfaContext);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *rfaContext);
    ge::graphStatus FillTilingData(gert::TilingContext *rfaContext);
    uint64_t GenerateTilingKey(gert::TilingContext *rfaContext);

    ge::graphStatus ParseQInputLayout(gert::TilingContext *rfaContext);
    ge::graphStatus ParseKvInputLayout(gert::TilingContext *rfaContext);
    ge::graphStatus ValidateTNDFormat(gert::TilingContext *rfaContext);
    ge::graphStatus ValidateBNSDFormat(gert::TilingContext *rfaContext);

    ge::graphStatus ProcessQueryShape(gert::TilingContext *rfaContext);
    ge::graphStatus ProcessSelectIdx(gert::TilingContext *rfaContext);
    ge::graphStatus ProcessActualSeqLengths(gert::TilingContext *rfaContext);
    ge::graphStatus ProcessBlockShape(gert::TilingContext *rfaContext);
    ge::graphStatus ValidateConfiguration(gert::TilingContext *rfaContext);
    ge::graphStatus ValidateTNDSeqlenSum(gert::TilingContext *rfaContext);

    ge::graphStatus ProcessQSeqLengths(gert::TilingContext *rfaContext, const gert::Tensor *actualSeqLengths);
    ge::graphStatus ProcessKvSeqLengths(gert::TilingContext *rfaContext, const gert::Tensor *actualSeqLengthsKv);
    ge::graphStatus ValidateBNSDQSeqlen(gert::TilingContext *rfaContext);
    ge::graphStatus ValidateBNSDKvSeqlen(gert::TilingContext *rfaContext);
    ge::graphStatus GetKvSeqlenFromShape(gert::TilingContext *rfaContext, uint32_t &kvSeqlen);

    bool CheckShouldUseUniformKvSeqlen(const gert::Tensor *actualSeqLengthsKv);
    ge::graphStatus SetupUniformKvSeqlen(gert::TilingContext *rfaContext);
    ge::graphStatus ProcessKvSeqLengthsBNSD(gert::TilingContext *rfaContext, const gert::Tensor *actualSeqLengthsKv);
    ge::graphStatus ProcessKvSeqLengthsTND(gert::TilingContext *rfaContext, const gert::Tensor *actualSeqLengthsKv);
    ge::graphStatus ProcessKvSeqLengthsWithArray(gert::TilingContext *rfaContext,
                                                 const gert::Tensor *actualSeqLengthsKv);

    void CalculateBatchTaskSplit(uint32_t batchIdx, int64_t qSeqlen, uint32_t groupSize, uint32_t &curTaskNum,
                                 uint32_t &curQBlockNum);

private:
    uint32_t batch_ = 0;
    uint32_t qSeqlen_ = 0;
    uint32_t kvSeqlen_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t embeddingSize_ = 0;
    uint32_t blockSize_ = 128;
    int64_t blockShapeX_ = 0; // block的x维度
    int64_t blockShapeY_ = 0; // block的y维度
    float scaleValue_ = 0.0f;
    uint32_t maskType_ = 0;
    uint32_t innerPrecise_ = 1; // 0=float32 softmax, 1=fp16 softmax

    uint32_t totalQBlocks_ = 0;
    uint32_t maxKvBlockNum_ = 0;
    uint32_t firstQBlockNum_ = 0;
    uint32_t firstBatchTaskNum_ = 0;
    uint32_t totalTaskNum_ = 0;
    uint32_t maxNumBlocksPerBatch_ = 0;
    const int64_t *qSeqLenList = nullptr;
    const int64_t *kvSeqLenList = nullptr;
    const int64_t *blockShapeList = nullptr;
    bool useUniformQSeqlen_ = false;  // 是否使用统一的qseqlen值（使用maxQSeqlen_）
    bool useUniformKvSeqlen_ = false; // 是否使用统一的kvseqlen值（使用maxKvSeqlen_）

    uint64_t mm1OutSize_ = 0;
    uint64_t smOnlineOutSize_ = 0;
    uint64_t mm2OutSize_ = 0;
    uint64_t updateSize_ = 0;

    RFAKvCacheLayout kvCacheLayout_ = RFAKvCacheLayout::TND;
    RFAQInputLayout qInputLayout_ = RFAQInputLayout::TND_Q;

    uint32_t blockDim_ = 20;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t workSpaceSize_ = 0;
    uint64_t libapiSize_ = 0;

    uint32_t maxQSeqlen_ = 0;   // BNSD格式Q的第三维（S维度）
    uint32_t maxKvSeqlen_ = 0;  // BNSD格式KV的第三维（S维度）
    int64_t totalTokensT_ = 0;  // TND格式Q的第一维（T维度，总token数）
    int64_t totalTokensKv_ = 0; // TND格式KV的第一维（T维度，总token数）

    ge::DataType dataType_ = ge::DT_FLOAT16;

    RainFusionAttentionTilingData *tilingData_ = nullptr;
};

} // namespace optiling

#endif // RAIN_FUSION_ATTENTION_TILING_H
