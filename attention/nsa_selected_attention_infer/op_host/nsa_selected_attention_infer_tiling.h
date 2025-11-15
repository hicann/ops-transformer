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
 * \file nsa_select_flash_attention_infer_tiling.h
 * \brief
 */
#ifndef NSA_SELECTED_ATTENTION_INFER_TILING_H_
#define NSA_SELECTED_ATTENTION_INFER_TILING_H_

#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"

constexpr uint32_t DEAL_BN2_NUM = 2;
const uint32_t MAX_CORE_NUM = 50;
const uint32_t MAX_SIZE_BATCH = 3072;
#ifdef ASCENDC_OP_TEST
#define Nsa_EXTERN_C extern "C"
#else
#define Nsa_EXTERN_C
#endif
namespace optiling {
BEGIN_TILING_DATA_DEF(NsaSelectAttentionInferBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, headSize)
TILING_DATA_FIELD_DEF(uint32_t, headSizeV)
TILING_DATA_FIELD_DEF(uint32_t, headSizeRope)
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockCount)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, qHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDims)
TILING_DATA_FIELD_DEF(uint32_t, msdIterNum)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, isMtpFlag)
TILING_DATA_FIELD_DEF(uint32_t, actualLenQDims)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(NsaSelectAttentionInferBaseParamsOp, NsaSelectAttentionInferBaseParams)

BEGIN_TILING_DATA_DEF(NsaSelectAttentionInferCoreParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSidxEnd); // 50:MAX_CORE_NUM  coreSidxEnd数组首地址要保证8字节对齐
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaSelectAttentionInferCoreParamsOp, NsaSelectAttentionInferCoreParams);

BEGIN_TILING_DATA_DEF(NsaSelectAttentionInferSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, groupSplitSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaSelectAttentionInferSingleCoreParamsOp, NsaSelectAttentionInferSingleCoreParams)

BEGIN_TILING_DATA_DEF(NsaSelectAttentionInferSingleCoreTensorSize)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaSelectAttentionInferSingleCoreTensorSizeOp, NsaSelectAttentionInferSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(NsaSelectAttentionInferSplitKVParams)
TILING_DATA_FIELD_DEF(uint32_t, s2)
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(NsaSelectAttentionInferSplitKVParamsOp, NsaSelectAttentionInferSplitKVParams)

BEGIN_TILING_DATA_DEF(NsaSelectAttentionInferTilingData)
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectAttentionInferBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectAttentionInferSplitKVParams, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectAttentionInferCoreParams, nsaSelectAttentionInferCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectAttentionInferSingleCoreParams, nsaSelectAttentionInferSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectAttentionInferSingleCoreTensorSize, nsaSelectAttentionInferSingleCoreTensorSize);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(NsaSelectAttentionInferTilingDataOp, NsaSelectAttentionInferTilingData)

BEGIN_TILING_DATA_DEF(NsaSelectAttentionInferTilingDataV2)
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectAttentionInferTilingData, tilingBase);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionInfer, NsaSelectAttentionInferTilingDataV2)

struct NsaSelectAttentionCompileInfo {
    int64_t core_num;
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct NsaSelectAttentionInferContext {
    const char *opName;
    fe::PlatFormInfos *platformInfo;
    RequiredParaInfo query;
    RequiredParaInfo key;
    RequiredParaInfo value;
    RequiredParaInfo topkIndices;
    OptionalParaInfo attenMask;
    OptionalParaInfo blockTable;
    OptionalParaInfo actualQSeqLengths;
    OptionalParaInfo actualKVSeqLengths;

    RequiredParaInfo attenOut;
    const char *layOut;
    const int64_t *numHeads;
    const int64_t *numKVHeads;
    const int64_t *kvHeadNums;
    const int64_t *selectedBlockSize;
    const int64_t *selectedBlockCount;
    const int64_t *blockSize;
    const float *scaleValue;
    const int64_t *sparseMode;

    size_t *workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

enum class TilingInOutMode : uint32_t {
    IO_INVALID = 0,
    INT8_INT8 = 1,
    FP16_INT8 = 2,
    INT8_FP16 = 3,
    FP16_FP16 = 4,
    BF16_BF16 = 5,
    FP32_FP32 = 6,
    FP16_FP16_SPLITKV = 7,
    BF16_INT8 = 8,
};

enum class NsaPerfMode : uint32_t {
    NORMAL = 0,
    BMM_ALL_BY_VEC, // 1
    C1_V1, // 2
    CUBE_VIEW_MM, // 3
    CUBE_VIEW_MM_FULL_LOAD, // 4
    CUBE_VIEW_MM_MLA  // 5
};

enum NsaSocVersion : uint32_t {
    A2 = 0
};

enum NsaLayout : uint32_t {
    BSND = 0,
    BSH = 1,
    BNSD = 2,
    TND = 3,
};

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t NUM_BYTES_BOOL = 1;
constexpr uint32_t NUM_BYTES_INT8 = 1;
constexpr uint32_t MAX_MATMUL_BASE = 512;
constexpr uint32_t MATMUL_BASE_N = 256;
constexpr uint32_t MAX_MATMUL_BASE_M = 128;
constexpr uint32_t MAX_SPLIT_SIZE = 8192;
constexpr uint32_t L0B_SIZE = 64U * 1024U;
constexpr uint32_t L0C_SIZE = 128U * 1024U;
constexpr uint32_t DIM_BNSD = 4;
constexpr uint32_t DIM_BNSD_OR_BNSD = 4;
constexpr uint32_t DIM_BSH = 3;
constexpr uint32_t DIM_TND = 3;
constexpr uint64_t s1BasicBlock = 1;

class NsaSelectTiling {
public:
    NsaSelectTiling() = default;
    ~NsaSelectTiling() = default;

    ge::graphStatus DoTiling(gert::TilingContext &context);
    ge::graphStatus RunBigKernelTiling(NsaSelectAttentionInferContext &context, NsaSelectAttentionInferTilingDataV2 &tilingData,
                                    bool isWorkspace = false);
    ge::graphStatus NsaSelectAttentionInferSetTilingData(gert::TilingContext &context,
                                                    NsaSelectAttentionInferTilingDataV2 &tilingData) const;
    static ge::graphStatus ConvertContext(gert::TilingContext &context, NsaSelectAttentionInferContext &nsaContext);

private:
    ge::graphStatus GetNpuInfo();
    ge::graphStatus PreProcess();
    ge::graphStatus ProcessBaseInputs();
    ge::graphStatus ProcessOptionalTensors();
    ge::graphStatus ProcessActualSeqLen();
    ge::graphStatus ProcessBlockTable();

    void SetupPerfMode();
    void SetCoreNum();

    ge::graphStatus InitInOutMode();
    ge::graphStatus CheckBaseInputsNull() const;
    ge::graphStatus InputAttrsPreProcess() const;
    ge::graphStatus QKVPreProcess();
    ge::graphStatus ProcessPageAttentionFlag();
    ge::graphStatus KvShapePostProcess();
    ge::graphStatus CheckQKOutShape();
    ge::graphStatus TNDCheckQKOutShape() const;
    ge::graphStatus ZeroTensorProcess();

    ge::graphStatus CheckUbSpace();
    ge::graphStatus CheckPABlockSize() const;

    ge::graphStatus CheckInputFormatAndLimits() const;
    ge::graphStatus CheckInputParameterFormat();
    ge::graphStatus CheckInputAntiquantFormat();
    bool CalcUbBmm();

    ge::graphStatus Split();
    ge::graphStatus CalcInnerSize(uint32_t seqSize);
    ge::graphStatus SplitBN();

    std::vector<int64_t> InitSparseValidArray(const int64_t *actualLens) const;
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                    std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx) const;
    void InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                    const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue) const;
    void SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                        uint32_t *sparseStartIdx, int64_t splitFactorSize);

    ge::graphStatus SplitBN_V0();

    bool CheckWorkSpace();

    ge::graphStatus CalcWorkSpace();
    ge::graphStatus CalcBlockDim();
    ge::graphStatus GenTilingKey() const;

    ge::graphStatus FillTiling();
    void FillTilingBaseParams() const;
    void FillTilingSplitKV() const;
    void FillTilingCoreParams();
    void FillTilingSingleCoreParams() const;
    void FillTilingSingleCoreTensorSize() const;
    void FillTilingTranspose();
    void FillTilingOutputParams();
    bool FillTilingBmm(); // may fail

private:
    bool passToOldTiling_ = false;
    int64_t numHeads_ = 0;
    float scaleValue_ = 0;
    int64_t numKvHeads_ = 0;
    int64_t selectedBlockCount_ = 0;
    int64_t selectedBlockSize_ = 0;
    int64_t blockSize_ = 0;
    uint32_t innerPrecise_ = 0;
    uint32_t nNumOfQInOneGroup_ = 1;
    uint32_t msdIterNum_ = 1;

    uint32_t headDim_ = 0;
    uint32_t headDimV_ = 0;
    uint32_t headDimRope_ = 0;
    uint32_t seqSize_ = 0;
    uint32_t batchSize_ = 0;
    NsaLayout inputLayout_ = NsaLayout::BSND;
    NsaLayout inputKvLayout_ = NsaLayout::BSND;
    uint32_t sMax_ = 0;
    uint32_t qSeqSize_ = 1;
    uint32_t kvSplitPart_ = 1;
    int64_t maxactLen_ = 1;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    NsaSocVersion socVersion_ = NsaSocVersion::A2;
    size_t libapiSize_ = 0;

    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;

    bool attenMaskFlag_ = false;
    uint32_t attenMaskSize_ = 0;

    uint32_t maxBlockNumPerBatch_ = 0;
    size_t kvPageResUbSize_ = 0;

    std::vector<int64_t> kvListSeqLens_;

    bool actualSeqLenFlag_ = false;
    bool kvPaddingSizeFlag_ = false;
    bool isMtpFlag_ = false;
    uint32_t actualLenQDims_ = 0;
    uint32_t totalQseq_ = 0;
    uint32_t actualLenDims_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameActualseq_ = true;
    bool isSameSeqAllKVTensor_ = true;
    bool hasZeroActualseq_ = false;
    bool hasZeroSeqKVTensor_ = false;

    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t headDimAlign_ = 0;

    bool splitKVFlag_ = false;
    uint32_t kvSplit_ = 0;

    NsaPerfMode perfMode_ = NsaPerfMode::NORMAL;
    TilingInOutMode inOutMode_ = TilingInOutMode::FP16_FP16;
    size_t workspaceSize_ = 0;

    uint32_t usedCoreNum_ = 0;

    uint32_t startIdxEachCore_[MAX_CORE_NUM] = {};
    NsaSelectAttentionInferContext *context_ = nullptr;
    NsaSelectAttentionInferTilingData *tilingData_ = nullptr;
    bool isWorkspace_ = false;

    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;
    uint32_t groupSplitSize_ = 0;
    uint32_t gOuter_ = 1;
    uint32_t sparseMode_ = 3;

    uint32_t batchSizeQ_ = 1;
    uint32_t dimOne = 1;
    uint32_t dimTwo = 2;
    uint32_t dimThree = 3;
    uint32_t doubleAiv = 2;

    uint32_t ONE = 1;
};

ge::graphStatus TilingPrepareForNsaSelectAttentionInfer(gert::TilingParseContext *context);
ge::graphStatus TilingNsaSelectAttentionInferAdapter(gert::TilingContext *context, NsaSelectAttentionInferContext &nsaContext,
                                                NsaSelectAttentionInferTilingDataV2 &nsaTilingData);

Nsa_EXTERN_C ge::graphStatus TilingNsaSelectAttentionInfer(gert::TilingContext *context);

} // namespace optiling
#endif // NSA_SELECTED_ATTENTION_INFER_TILING_H_
 