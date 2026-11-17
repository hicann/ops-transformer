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
 * \file nsa_compress_with_cache_tiling_general.cpp
 * \brief
 */

#include "log/log.h"
#include "nsa_compress_with_cache_tiling.h"
#include "nsa_compress_with_cache_tiling_common.h"
#include <vector>

using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace Nsa {

static const size_t ACT_SEQ_LEN_INPUT_INDEX = 4UL;
static const size_t BLOCK_TABLE_INPUT_INDEX = 5UL;
static const size_t COMPRESS_ALIGN = 16UL;
static const size_t ACT_SEQ_TYPE_COUNT = 1UL;

static const size_t INPUT_INDEX_INPUT = 0UL;
static const size_t INPUT_INDEX_WEIGHT = 1UL;
static const size_t INPUT_INDEX_OUTPUT_CACHE = 3UL;

static const uint64_t DIM_NUM_0 = 0;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;

static const size_t MAX_BATCH_SIZE = 2048UL;
static const size_t DIM_ALIGN = 16UL;
static const size_t UB_BLOCK_BYTES = 32UL;
static const size_t MAX_CORE_NUM = 50UL;
static const size_t FP32_NUM_PER_UB_BLOCK = 8UL;
static const size_t TOKEN_NUM_PER_TILE = 8UL;
static const size_t MAX_COMPRESS_BLOCK_SIZE = 64UL;
static const size_t MAX_ODD_HEAD_SIZE = 50UL;

static const size_t WEIGHT_INIT_FLAG_DOUBLE_BROADCAST = 0UL;
static const size_t WEIGHT_INIT_FLAG_TILE_BROADCAST = 1UL;
static const size_t WEIGHT_INIT_FLAG_FULL_BROADCAST = 2UL;

static const size_t SAL_BIT_1 = 1UL;
static const size_t SAL_BIT_2 = 2UL;
static const size_t SAL_BIT_4 = 4UL;
static const size_t SAL_BIT_8 = 8UL;
static const size_t SAL_BIT_16 = 16UL;

static const size_t NUM_2 = 2UL;

static void ReportTilingFailure(const std::string& reason)
{
    constexpr const char* opName = "NsaCompressWithCache";
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
}

class NsaCompressWithCacheTiling : public TilingBaseClass {
public:
    explicit NsaCompressWithCacheTiling(gert::TilingContext* context)
        : TilingBaseClass(context) {};
    ~NsaCompressWithCacheTiling() override = default;

protected:
    ge::graphStatus CheckContext()
    {
        auto attrs = context_->GetAttrs();
        OP_CHECK_IF(attrs == nullptr, ReportTilingFailure("Tiling context attributes are null."),
                    return ge::GRAPH_FAILED);
        // 除了第一个属性都是必选属性
        size_t idx = 1;
        auto compressBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto compressStridePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto actSeqLenTypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto pageBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
        size_t* workspaces = context_->GetWorkspaceSizes(1);

        OP_CHECK_IF(compressBlockSizePtr == nullptr, ReportTilingFailure("compress_block_size attribute is null."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(compressStridePtr == nullptr, ReportTilingFailure("compress_stride attribute is null."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(actSeqLenTypePtr == nullptr, ReportTilingFailure("act_seq_len_type attribute is null."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(pageBlockSizePtr == nullptr, ReportTilingFailure("page_block_size attribute is null."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(workspaces == nullptr, ReportTilingFailure("Workspace sizes are null."), return ge::GRAPH_FAILED);

        // 检查input、weight和outputCache
        auto inputShape = context_->GetInputShape(INPUT_INDEX_INPUT);
        auto weightShape = context_->GetInputShape(INPUT_INDEX_WEIGHT);
        auto outputCacheShape = context_->GetInputShape(INPUT_INDEX_OUTPUT_CACHE);

        OP_CHECK_IF(inputShape == nullptr, ReportTilingFailure("Input shape is null."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(weightShape == nullptr, ReportTilingFailure("Weight shape is null."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(outputCacheShape == nullptr, ReportTilingFailure("Output cache shape is null."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->GetRawTilingData() == nullptr, ReportTilingFailure("Raw tiling data is null."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->GetRawTilingData()->GetData() == nullptr,
                    ReportTilingFailure("Raw tiling data buffer is null."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(),
                    ReportTilingFailure(
                        "Raw tiling data capacity " + std::to_string(context_->GetRawTilingData()->GetCapacity()) +
                        " is smaller than the required size " + std::to_string(tilingData.GetDataSize()) + "."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    bool AnalyzeAttrs()
    {
        auto attrs = context_->GetAttrs();

        size_t idx = 0;
        inputLayout = attrs->GetAttrPointer<char>(idx++);
        auto compressBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto compressStridePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto actSeqLenTypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto pageBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);

        OP_CHECK_IF(*compressStridePtr <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "compress_stride", std::to_string(*compressStridePtr),
                                                          "Must be a positive integer multiple of 16."),
                    return false);

        compressBlockSize = *compressBlockSizePtr;
        compressStride = *compressStridePtr;
        actSeqLenType = *actSeqLenTypePtr;
        pageBlockSize = *pageBlockSizePtr;

        OP_CHECK_IF(compressBlockSize % COMPRESS_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName, "compress_block_size", std::to_string(compressBlockSize), "Must be a multiple of 16."),
                    return false);
        OP_CHECK_IF(compressStride % COMPRESS_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "compress_stride", std::to_string(compressStride),
                                                          "Must be a multiple of 16."),
                    return false);
        OP_CHECK_IF(compressBlockSize < compressStride,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName, "compress_block_size, compress_stride",
                        std::to_string(compressBlockSize) + ", " + std::to_string(compressStride),
                        "compress_block_size must be greater than or equal to compress_stride."),
                    return false);
        OP_CHECK_IF(actSeqLenType != ACT_SEQ_TYPE_COUNT,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "act_seq_len_type", std::to_string(actSeqLenType),
                                                          "Only 1 is supported."),
                    return false);

        OP_LOGD(context_, "attrs: compressBlockSize[%u] compressStride[%u] actSeqLenType[%u] pageBlockSize[%u].",
                compressBlockSize, compressStride, actSeqLenType, pageBlockSize);
        return true;
    }

    bool AnalyzeDtype()
    {
        // 目前默认转换成float32计算
        inputDtype = context_->GetInputDesc(INPUT_INDEX_INPUT)->GetDataType();
        ge::DataType weightDtype = context_->GetInputDesc(INPUT_INDEX_WEIGHT)->GetDataType();
        OP_CHECK_IF(inputDtype != weightDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, "weight", ge::TypeUtils::DataTypeToSerialString(weightDtype),
                        "Must match input dtype " + ge::TypeUtils::DataTypeToSerialString(inputDtype) + "."),
                    return false);
        mInputDtypeBytes = ge::GetSizeByDataType(inputDtype);
        switch (inputDtype) {
            case ge::DT_FLOAT16:
                mCalcTypeBytes = ge::GetSizeByDataType(ge::DT_FLOAT);
                break;
            case ge::DT_BF16:
                mCalcTypeBytes = ge::GetSizeByDataType(ge::DT_FLOAT);
                break;
            default:
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "input",
                                                      ge::TypeUtils::DataTypeToSerialString(inputDtype),
                                                      "Only float16 and bfloat16 are supported.");
                return false;
        }
        return true;
    }

    bool AnalyzeInputShape()
    {
        auto inputShape = context_->GetInputShape(INPUT_INDEX_INPUT);
        auto weightShape = context_->GetInputShape(INPUT_INDEX_WEIGHT);
        auto outputCacheShape = context_->GetInputShape(INPUT_INDEX_OUTPUT_CACHE);
        headNum = inputShape->GetStorageShape().GetDim(DIM_NUM_2);
        headDim = inputShape->GetStorageShape().GetDim(DIM_NUM_3);
        outTokenNum = outputCacheShape->GetStorageShape().GetDim(DIM_NUM_0);
        OP_CHECK_IF(headNum != weightShape->GetStorageShape().GetDim(DIM_NUM_1),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName, "weight.shape[1]", std::to_string(weightShape->GetStorageShape().GetDim(DIM_NUM_1)),
                        "Must match input head count " + std::to_string(headNum) + "."),
                    return false);

        OP_CHECK_IF(compressBlockSize != weightShape->GetStorageShape().GetDim(DIM_NUM_0),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName, "weight.shape[0]", std::to_string(weightShape->GetStorageShape().GetDim(DIM_NUM_0)),
                        "Must match compress_block_size " + std::to_string(compressBlockSize) + "."),
                    return false);

        OP_CHECK_IF(
            headNum != outputCacheShape->GetStorageShape().GetDim(DIM_NUM_1),
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName, "output_cache.shape[1]", std::to_string(outputCacheShape->GetStorageShape().GetDim(DIM_NUM_1)),
                "Must match input head count " + std::to_string(headNum) + "."),
            return false);
        OP_CHECK_IF(
            headDim != outputCacheShape->GetStorageShape().GetDim(DIM_NUM_2),
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName, "output_cache.shape[2]", std::to_string(outputCacheShape->GetStorageShape().GetDim(DIM_NUM_2)),
                "Must match input head dimension " + std::to_string(headDim) + "."),
            return false);

        OP_CHECK_IF(headDim % DIM_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "input.shape[3]", std::to_string(headDim),
                                                          "Head dimension must be a multiple of 16."),
                    return false);

        OP_CHECK_IF(compressBlockSize > MAX_COMPRESS_BLOCK_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "compress_block_size",
                                                          std::to_string(compressBlockSize), "Must not exceed 64."),
                    return false);
        OP_CHECK_IF(headNum == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "input.shape[2]", std::to_string(headNum),
                                                          "Head count must be greater than 0."),
                    return false);
        OP_CHECK_IF(headNum > MAX_COMPRESS_BLOCK_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "input.shape[2]", std::to_string(headNum),
                                                          "Head count must not exceed 64."),
                    return false);
        OP_CHECK_IF(headNum > MAX_ODD_HEAD_SIZE && headNum % 2 == 1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "input.shape[2]", std::to_string(headNum),
                                                          "An odd head count must not exceed 50."),
                    return false);

        return true;
    }

    void AnalyzeActSeqLenData()
    {
        auto actSeqLenTensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
        if (actSeqLenTensor == nullptr) {
            OP_LOGW(context_, "[%s]actSeqLenTensor is null pointer", opName);
            return;
        }
        auto& actSeqLenShape = actSeqLenTensor->GetShape().GetStorageShape();
        if (actSeqLenShape.GetDimNum() != 1) {
            OP_LOGW(context_, "[%s]actSeqLenShape is invalid %lu %ld", opName, actSeqLenShape.GetDimNum(),
                    actSeqLenShape.GetDim(0));
            return;
        }
        /* Get Data from tensor. */
        const int64_t* value = actSeqLenTensor->GetData<int64_t>();
        if (value == nullptr) {
            OP_LOGW(context_, "[%s]actSeqLenTensor data is null pointer", opName);
            return;
        }
        compressBatchSize = 0;
        for (int64_t i = 0; i < actSeqLenShape.GetDim(DIM_NUM_0); ++i) {
            if (value[i] >= compressBlockSize && (value[i] - compressBlockSize) % compressStride == 0) {
                compressBatchList[compressBatchSize] = i;
                compressBatchSize++;
            }
        }
    }

    uint32_t GetFullBroadcastByteSize(uint32_t buf_num)
    {
        uint32_t kvQueSize = TOKEN_NUM_PER_TILE * headDim * headNum;
        uint32_t weightQueSize = compressBlockSize * headDim;
        uint32_t inputQueSize = kvQueSize > weightQueSize ? kvQueSize : weightQueSize;
        return (inputQueSize * mInputDtypeBytes + kvQueSize * (mCalcTypeBytes + mCalcTypeBytes) +
                (compressBlockSize / TOKEN_NUM_PER_TILE) * headDim * headNum * mCalcTypeBytes +
                headNum * headDim * mInputDtypeBytes) *
                   buf_num +
               compressBlockSize * headNum * mCalcTypeBytes;
    }

    uint32_t GetDoubleBroadcastByteSize(uint32_t headNumPerCore, uint32_t buf_num)
    {
        if (headNumPerCore == 0) {
            return ubSize + 1;
        } else {
            uint32_t kvQueSize = TOKEN_NUM_PER_TILE * headNumPerCore * headDim;
            uint32_t weightQueSize = compressBlockSize * headDim;
            uint32_t inputQueSize = kvQueSize > weightQueSize ? kvQueSize : weightQueSize;
            return (mInputDtypeBytes * inputQueSize + kvQueSize * (mCalcTypeBytes + mCalcTypeBytes) +
                    (compressBlockSize / TOKEN_NUM_PER_TILE) * headNumPerCore * headDim * mCalcTypeBytes +
                    compressBlockSize * headNum * (FP32_NUM_PER_UB_BLOCK / headNumPerCore) * mCalcTypeBytes +
                    TOKEN_NUM_PER_TILE * UB_BLOCK_BYTES + headNumPerCore * headDim * mInputDtypeBytes) *
                       buf_num +
                   compressBlockSize * headNum * mCalcTypeBytes;
        }
    }

    bool AnalyzeOptionalInput()
    {
        actSeqLenExistFlag = 0;
        auto actSeqLenShape = context_->GetOptionalInputShape(ACT_SEQ_LEN_INPUT_INDEX);
        auto actSeqLenInput = context_->GetOptionalInputDesc(ACT_SEQ_LEN_INPUT_INDEX);
        if (actSeqLenShape != nullptr && actSeqLenShape->GetStorageShape().GetDimNum() != 0) {
            actSeqLenExistFlag = 1;
            auto actSeqLenDtype = actSeqLenInput->GetDataType();
            OP_CHECK_IF(actSeqLenDtype != ge::DT_INT64,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "act_seq_len",
                                                              ge::TypeUtils::DataTypeToSerialString(actSeqLenDtype),
                                                              "Only int64 is supported."),
                        return false);
            batchSize = actSeqLenShape->GetStorageShape().GetDim(0);
            OP_CHECK_IF(batchSize > MAX_BATCH_SIZE,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "act_seq_len.shape[0]", std::to_string(batchSize),
                                                              "Batch size must not exceed 2048."),
                        return false);
            AnalyzeActSeqLenData();
        }
        blockTableExistFlag = 0;
        auto blockTableShape = context_->GetOptionalInputShape(BLOCK_TABLE_INPUT_INDEX);
        auto blockTableInput = context_->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX);
        if (blockTableShape != nullptr && blockTableShape->GetStorageShape().GetDimNum() != 0) {
            blockTableExistFlag = 1;
            auto blockTableDtype = blockTableInput->GetDataType();
            OP_CHECK_IF(blockTableDtype != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "block_table",
                                                              ge::TypeUtils::DataTypeToSerialString(blockTableDtype),
                                                              "Only int32 is supported."),
                        return false);
            pageNumPerBatch = blockTableShape->GetStorageShape().GetDim(1);
        }
        inputLayoutExitFlag = 0;
        if (inputLayout != nullptr) {
            inputLayoutExitFlag = 1;
            std::string inputLayoutStr = std::string(inputLayout);
            for (auto& c : inputLayoutStr) {
                c = toupper(c);
            }
            OP_CHECK_IF(inputLayoutStr != "BSH" && inputLayoutStr != "SBH" && inputLayoutStr != "BSND" &&
                            inputLayoutStr != "BNSD" && inputLayoutStr != "TND",
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            opName, "layout", inputLayout,
                            "Only BSH, SBH, BSND, BNSD and TND are supported (case-insensitive)."),
                        return false);
        }
        OP_LOGD(context_, "actSeqLenExistFlag: %d, blockTableExistFlag: %d.", actSeqLenExistFlag, blockTableExistFlag);
        return true;
    }

    uint32_t getCeilPower2(uint32_t num)
    {
        num--;
        num |= num >> SAL_BIT_1;
        num |= num >> SAL_BIT_2;
        num |= num >> SAL_BIT_4;
        num |= num >> SAL_BIT_8;
        num |= num >> SAL_BIT_16;
        return ++num;
    }

    ge::graphStatus getHeadsNumPerCore()
    {
        headsNumPerCore = 0;
        bufferNum = 1;
        if (GetFullBroadcastByteSize(1) < ubSize) {
            headsNumPerCore = headNum;
            if (GetFullBroadcastByteSize(NUM_2) < ubSize) {
                bufferNum = NUM_2;
            }
            weightInitFlag = WEIGHT_INIT_FLAG_FULL_BROADCAST;
        } else {
            weightInitFlag = WEIGHT_INIT_FLAG_DOUBLE_BROADCAST;
            if (headNum % SAL_BIT_8 == 0 && GetDoubleBroadcastByteSize(SAL_BIT_8, 1) < ubSize) {
                weightInitFlag = WEIGHT_INIT_FLAG_TILE_BROADCAST;
                headsNumPerCore = FP32_NUM_PER_UB_BLOCK;
                if (GetDoubleBroadcastByteSize(SAL_BIT_8, NUM_2) < ubSize) {
                    bufferNum = NUM_2;
                }
            } else if (headNum % SAL_BIT_4 == 0 && GetDoubleBroadcastByteSize(SAL_BIT_4, 1) < ubSize) {
                headsNumPerCore = SAL_BIT_4;
                if (GetDoubleBroadcastByteSize(SAL_BIT_4, NUM_2) < ubSize) {
                    bufferNum = NUM_2;
                }
            } else if (headNum % SAL_BIT_2 == 0 && GetDoubleBroadcastByteSize(SAL_BIT_2, 1) < ubSize) {
                headsNumPerCore = SAL_BIT_2;
                if (GetDoubleBroadcastByteSize(SAL_BIT_2, NUM_2) < ubSize) {
                    bufferNum = NUM_2;
                }
            } else if (headNum % SAL_BIT_1 == 0 && GetDoubleBroadcastByteSize(SAL_BIT_1, 1) < ubSize) {
                headsNumPerCore = SAL_BIT_1;
                if (GetDoubleBroadcastByteSize(SAL_BIT_1, NUM_2) < ubSize) {
                    bufferNum = NUM_2;
                }
            } else {
                headsNumPerCore = 0;
            }
            OP_CHECK_IF(headsNumPerCore == 0,
                        ReportTilingFailure("No head fits into the available UB for the input shape."),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    void setTilingData()
    {
        tilingData.set_kvCacheSize(static_cast<uint64_t>(batchSize) * pageNumPerBatch * pageBlockSize * headNum *
                                   headDim);
        tilingData.set_kvCacheSizePerCore(TOKEN_NUM_PER_TILE * headsNumPerCore * headDim);
        tilingData.set_weightSize(static_cast<uint64_t>(compressBlockSize) * headNum);
        tilingData.set_weightSizePerCore(TOKEN_NUM_PER_TILE * headsNumPerCore * headDim);
        tilingData.set_compressKvCacheSize(static_cast<uint64_t>(outTokenNum) * headNum * headDim);
        tilingData.set_compressKvCacheSizePerCore(headsNumPerCore * headDim);
        tilingData.set_blockTableSize(batchSize * pageNumPerBatch);
        tilingData.set_headNum(headNum);
        tilingData.set_headDim(headDim);
        tilingData.set_batchSize(batchSize);
        tilingData.set_compressBlockSize(compressBlockSize);
        tilingData.set_compressStride(compressStride);
        tilingData.set_pageBlockSize(pageBlockSize);
        tilingData.set_pageNumPerBatch(pageNumPerBatch);
        tilingData.set_headsNumPerCore(headsNumPerCore);
        tilingData.set_coresNumPerCompress(coresNumPerCompress);
        tilingData.set_isEmpty(isEmpty);
        tilingData.set_alignReduceSize(alignReduceSize);
        tilingData.set_tokenNumPerTile(TOKEN_NUM_PER_TILE);
        tilingData.set_tokenRepeatCount(tokenRepeatCount);
        tilingData.set_bufferNum(bufferNum);
    }

    bool IsCapable() override
    {
        // do some check
        return true;
    };
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override
    {
        auto platformInfoPtr = context_->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compileInfoPtr = reinterpret_cast<const NsaCompressWithCacheCompileInfo*>(context_->GetCompileInfo());
            OP_CHECK_IF(compileInfoPtr == nullptr, ReportTilingFailure("Both platform info and compile info are null."),
                        return ge::GRAPH_FAILED);
            aivNum = compileInfoPtr->aivNum;
            ubSize = compileInfoPtr->ubSize;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            aivNum = ascendcPlatform.GetCoreNumAiv();
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        }
        OP_LOGI(context_, "get platform from compileInfo.aivNum(%u) ubSize(%lu).", aivNum, ubSize);
        return ge::GRAPH_SUCCESS;
    };
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override
    {
        opName = context_->GetNodeName();
        OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());
        OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(opName, "invalid context."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeInputShape() || !AnalyzeDtype() || !AnalyzeOptionalInput(),
                    OP_LOGE_WITHOUT_REPORT(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override
    {
        isEmpty = false;
        if (compressBatchSize == 0) {
            isEmpty = true;
            bufferNum = 1;
            tilingData.set_isEmpty(isEmpty);
            blockDim = 1;
            weightInitFlag = WEIGHT_INIT_FLAG_TILE_BROADCAST;
            return ge::GRAPH_SUCCESS;
        }
        getHeadsNumPerCore();
        coresNumPerCompress = (headsNumPerCore == 0) ? 0 : (headNum / headsNumPerCore);
        uint32_t totalBlockDim = coresNumPerCompress * compressBatchSize;
        uint32_t headNumPerRealCore = totalBlockDim / aivNum;
        uint32_t tailHeadNum = totalBlockDim - headNumPerRealCore * aivNum;
        uint32_t repeatNumList[MAX_CORE_NUM];
        uint32_t coreStartHeadIdx[MAX_CORE_NUM];
        uint32_t coreStartBatchIdx[MAX_CORE_NUM];
        tokenRepeatCount = compressBlockSize / TOKEN_NUM_PER_TILE;
        coreStartHeadIdx[0] = 0;
        for (uint32_t i = 0; i < aivNum; i++) {
            repeatNumList[i] = headNumPerRealCore;
        }
        for (uint32_t i = 0; i < aivNum; i++) {
            if (i < tailHeadNum) {
                repeatNumList[i] = repeatNumList[i] + 1;
            }
            coreStartHeadIdx[i + 1] = coreStartHeadIdx[i] + repeatNumList[i];
        }
        for (uint32_t i = 0; i <= aivNum; i++) {
            coreStartHeadIdx[i] *= headsNumPerCore;
        }

        if (totalBlockDim < aivNum)
            blockDim = totalBlockDim;
        else
            blockDim = aivNum;
        for (uint32_t i = 0; i < blockDim; i++) {
            if (coreStartHeadIdx[i] / headNum < MAX_BATCH_SIZE)
                coreStartBatchIdx[i] = compressBatchList[coreStartHeadIdx[i] / headNum];
        }
        if (compressBatchSize < MAX_BATCH_SIZE)
            coreStartBatchIdx[blockDim] = compressBatchList[compressBatchSize - 1];
        alignReduceSize = getCeilPower2(tokenRepeatCount) / NUM_2;

        setTilingData();
        tilingData.set_coreStartBatchIdxList(coreStartBatchIdx);
        tilingData.set_coreStartHeadIdxList(coreStartHeadIdx);
        return ge::GRAPH_SUCCESS;
    };
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override
    {
        return weightInitFlag;
    };
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint32_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0 + sysWorkSpaceSize;
        return ge::GRAPH_SUCCESS;
    };
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        context_->SetBlockDim(blockDim);
        tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
        return ge::GRAPH_SUCCESS;
    };

    ge::DataType inputDtype;
    uint32_t mInputDtypeBytes;
    uint32_t mCalcTypeBytes;

    const char* opName = "NsaCompressWithCache";
    uint32_t aivNum;
    uint64_t ubSize;

    uint32_t weightInitFlag;

    const char* inputLayout;
    uint32_t blockDim;
    uint32_t compressBlockSize;
    uint32_t compressStride;
    uint32_t actSeqLenType;
    uint32_t pageBlockSize;
    uint32_t batchSize;
    uint32_t pageNumPerBatch;
    uint32_t compressBatchSize;
    uint32_t headNum;
    uint32_t headDim;
    uint32_t totalInSeqLen;
    uint32_t totalOutSeqLen;
    uint32_t outTokenNum;
    uint32_t isEmpty;
    uint32_t bufferNum;
    uint32_t alignReduceSize;
    uint32_t headsNumPerCore;
    uint32_t coresNumPerCompress;
    uint32_t tokenRepeatCount;
    uint32_t compressBatchList[MAX_BATCH_SIZE];
    uint8_t actSeqLenExistFlag = 0;
    uint8_t blockTableExistFlag = 0;
    uint8_t inputLayoutExitFlag = 0;

    NsaCompressWithCacheTilingData tilingData;
};
// NOTE manually initialize tiling data in hostapi scenario in highest priority template
REGISTER_OPS_TILING_TEMPLATE(NsaCompressWithCache, NsaCompressWithCacheTiling, 0);
} // namespace Nsa
} // namespace optiling
