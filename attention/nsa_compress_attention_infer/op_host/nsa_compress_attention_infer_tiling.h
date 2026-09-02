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
 * \file nsa_compress_attention_infer_tiling.h
 * \brief
 */

#ifndef NSA_COMPRESS_ATTENTION_INFER_TILING_H
#define NSA_COMPRESS_ATTENTION_INFER_TILING_H
#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(NsaCompressAttentionInferBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, headSizeQk)
TILING_DATA_FIELD_DEF(uint32_t, headSizeVo)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, groupSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, selectSize)
TILING_DATA_FIELD_DEF(uint32_t, selectNum)
TILING_DATA_FIELD_DEF(uint32_t, compSizeL)
TILING_DATA_FIELD_DEF(uint32_t, compStrideD)
TILING_DATA_FIELD_DEF(uint32_t, ubSize)
TILING_DATA_FIELD_DEF(uint32_t, topKInWorkSpaceSize);

TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
// 新增
TILING_DATA_FIELD_DEF(uint32_t, workSpaceElemNum);
TILING_DATA_FIELD_DEF(uint32_t, mm1ResWorkSpaceSize);
TILING_DATA_FIELD_DEF(uint32_t, mm2InWorkSpaceSize);
TILING_DATA_FIELD_DEF(uint32_t, scoreInWorkSpaceSize);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(NsaCompressAttentionInferBaseParamsOp, NsaCompressAttentionInferBaseParams)

BEGIN_TILING_DATA_DEF(NsaCompressAttentionInferSplitBNParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNumUsed)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadSplitSize)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadSplitNum)
TILING_DATA_FIELD_DEF(uint32_t, qSeqLenSplitSize)
TILING_DATA_FIELD_DEF(uint32_t, processPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, processNum)

TILING_DATA_FIELD_DEF(uint32_t, bnTotalSize)
TILING_DATA_FIELD_DEF(uint32_t, headCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, tailCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, bnPerHeadCore)
TILING_DATA_FIELD_DEF(uint32_t, bnPerTailCore)
TILING_DATA_FIELD_DEF(uint32_t, rowLenPerHeadCore)
TILING_DATA_FIELD_DEF(uint32_t, rowLenPerTailCore)
TILING_DATA_FIELD_DEF(uint32_t, baseRowLenHeadCore)
TILING_DATA_FIELD_DEF(uint32_t, baseRowLenTailCore)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(NsaCompressAttentionInferSplitBNParamsOp, NsaCompressAttentionInferSplitBNParams)

BEGIN_TILING_DATA_DEF(NsaCompressAttentionInferTilingData)
TILING_DATA_FIELD_DEF_STRUCT(NsaCompressAttentionInferBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaCompressAttentionInferSplitBNParams, splitBNParams);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaCompressAttentionInfer, NsaCompressAttentionInferTilingData)

struct NsaCompressAttentionInferCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;

    uint32_t dataNumSingleUb = 1; // UB空间可处理的最大数据量
    uint32_t blockNum = 1;        // 32B对齐使用 //
    uint32_t cacheLineLen = 1;    // 512B对齐使用

    uint32_t coreNum = 1;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t l0CSize = 0;
    uint64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion;
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct NsaCompressAttentionInferContext {
    const char *opName;
    fe::PlatFormInfos *platformInfo;
    RequiredParaInfo query;
    RequiredParaInfo key;
    RequiredParaInfo value;
    OptionalParaInfo attenMask;
    OptionalParaInfo blockTable;
    OptionalParaInfo actualQSeqLengths;
    OptionalParaInfo actualCmpKvSeqLengths;
    OptionalParaInfo actualSelKvSeqLengths;
    OptionalParaInfo topkMask;

    RequiredParaInfo attenOut;
    RequiredParaInfo selectOut;

    const int64_t *numHeads;
    const int64_t *kvHeadNums;
    const int64_t *selectSize;
    const int64_t *selectNum;
    const int64_t *compSizeL;
    const int64_t *compStrideD;
    const float *scaleValue;

    const char *layOut;
    const int64_t *blockSize;
    const int64_t *sparseMode;

    size_t *workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

enum NSICALayout : uint32_t {
    TND = 0,
};

class NCAITiling {
public:
    NCAITiling() = default;
    ~NCAITiling() = default;

    ge::graphStatus GetNCAITiling(NsaCompressAttentionInferContext &ncaiContext,
                                  NsaCompressAttentionInferTilingData &tilingData, bool isWorkspace = false);
    ge::graphStatus NSICASetTilingData(gert::TilingContext &context, NsaCompressAttentionInferTilingData &tilingData);
    static ge::graphStatus ConvertContext(gert::TilingContext &context, NsaCompressAttentionInferContext &ncaiContext);

private:
    ge::graphStatus GetNpuInfo();
    ge::graphStatus Split();
    ge::graphStatus SplitBN();
    ge::graphStatus ProcessQkv();
    ge::graphStatus CheckQkvShape();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckTopK();
    ge::graphStatus ProcessEpilogue();
    ge::graphStatus ProcessInput();
    ge::graphStatus CheckQkv();
    ge::graphStatus ParamsPostCheck();
    ge::graphStatus SoftmaxTiling();
    ge::graphStatus TopKTiling();
    ge::graphStatus FillTilingData();
    ge::graphStatus CalcWorkSpace();
    ge::graphStatus CalcTilingKey();
    ge::graphStatus GetMaxMinSeqlen();
    ge::graphStatus CheckSelKvSeqlen();
    ge::graphStatus GetMaxQSeqlen();
    ge::graphStatus CheckAttenMask();

private:
    uint32_t batchSize_ = 0;
    uint32_t qSeqSize_ = 0;
    uint32_t sMax_ = 0;
    uint32_t sMin_ = 0;
    uint32_t qHeadNum_ = 0;
    uint32_t kvHeadNum_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t headSizeQk_ = 0;
    uint32_t headSizeVo_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlockNumPerBatch_ = 0;
    float scaleValue_ = 0;
    uint32_t attenMaskFlag_ = 0;
    uint32_t selectSize_ = 0;
    uint32_t selectNum_ = 0;
    uint32_t compSizeL_ = 0;
    uint32_t compStrideD_ = 0;
    uint32_t kvHeadSplitSize_ = 0;
    uint32_t kvHeadSplitNum_ = 0;
    uint32_t qSeqLenSplitSize_ = 1;
    uint32_t qSeqLenCumSum_ = 0;
    uint32_t processPerBatch_ = 0;
    uint32_t processNum_ = 0;
    bool pagedAttentionFlag_ = true;

    bool isWorkspace_ = false;
    uint32_t blockDim_ = 0;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t libapiSize_ = 0;
    uint32_t workSpaceSize_ = 0;
    uint64_t ubSize_ = 0;

    uint32_t bnTotalSize_ = 0;
    uint32_t coreNumUsed_ = 0;
    uint32_t headCoreNum_ = 0;
    uint32_t tailCoreNum_ = 0;
    uint32_t bnPerHeadCore_ = 0;
    uint32_t bnPerTailCore_ = 0;

    uint32_t sInnerLoopTimes_ = 0;

    uint32_t workSpaceElemNum_ = 0;
    uint32_t mm1ResWorkSpaceSize_ = 0;
    uint32_t mm2InWorkSpaceSize_ = 0;
    uint32_t scoreInWorkSpaceSize_ = 0;
    uint32_t topKInWorkSpaceSize_ = 0;

    uint32_t rowLenPerHeadCore_ = 0;
    uint32_t rowLenPerTailCore_ = 0;
    uint32_t baseRowLenHeadCore_ = 0;
    uint32_t baseRowLenTailCore_ = 0;

    NsaCompressAttentionInferContext *ncaiContext_ = nullptr;
    NsaCompressAttentionInferTilingData *tilingData_ = nullptr;
};

} // namespace optiling
#endif // NSA_COMPRESS_ATTENTION_INFER_TILING_H
