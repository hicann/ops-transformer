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
#include "err/ops_err.h"

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

class NsaCompressWithCacheTiling : public TilingBaseClass {
public:
    explicit NsaCompressWithCacheTiling(gert::TilingContext* context)
        : TilingBaseClass(context) {};
    ~NsaCompressWithCacheTiling() override = default;

protected:
    ge::graphStatus CheckContext()
    {
        auto attrs = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
        // 除了第一个属性都是必选属性
        size_t idx = 1;
        auto compressBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto compressStridePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto actSeqLenTypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto pageBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
        size_t* workspaces = context_->GetWorkspaceSizes(1);

        OP_CHECK_NULL_WITH_CONTEXT(context_, compressBlockSizePtr);
        OP_CHECK_NULL_WITH_CONTEXT(context_, compressStridePtr);
        OP_CHECK_NULL_WITH_CONTEXT(context_, actSeqLenTypePtr);
        OP_CHECK_NULL_WITH_CONTEXT(context_, pageBlockSizePtr);
        OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

        // 检查input、weight和outputCache
        auto inputShape = context_->GetInputShape(INPUT_INDEX_INPUT);
        auto weightShape = context_->GetInputShape(INPUT_INDEX_WEIGHT);
        auto outputCacheShape = context_->GetInputShape(INPUT_INDEX_OUTPUT_CACHE);

        OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
        OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputCacheShape);
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
        OP_CHECK_IF(context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(),
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache",
                                                "context tiling data capacity %zu < actual tiling data size %zu.",
                                                context_->GetRawTilingData()->GetCapacity(), tilingData.GetDataSize()),
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

        compressBlockSize = *compressBlockSizePtr;
        compressStride = *compressStridePtr;
        actSeqLenType = *actSeqLenTypePtr;
        pageBlockSize = *pageBlockSizePtr;

        OP_CHECK_IF(compressBlockSize % COMPRESS_ALIGN != 0,
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "compressBlockSize should be divisible by 16."),
                    return false);
        OP_CHECK_IF(compressStride % COMPRESS_ALIGN != 0,
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "compressStride should be divisible by 16."),
                    return false);
        OP_CHECK_IF(compressBlockSize < compressStride,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        "NsaCompressWithCache",
                        "got compressBlockSize < compressStride, expect compressBlockSize >= compressStride."),
                    return false);
        OP_CHECK_IF(actSeqLenType != ACT_SEQ_TYPE_COUNT,
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "attrs: actSeqLenType only support 1 now"),
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
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "Dtype of input must equal to weight"),
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
                OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "not support input dtype: %s for now",
                                            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
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
                    OPS_REPORT_VECTOR_INNER_ERR(
                        "NsaCompressWithCache",
                        "2nd dim of weight should equal to headNum, but got weight.shape[1]=%ld and headNum=%u",
                        weightShape->GetStorageShape().GetDim(DIM_NUM_1), headNum),
                    return false);

        OP_CHECK_IF(compressBlockSize != weightShape->GetStorageShape().GetDim(DIM_NUM_0),
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache",
                                                "1st dim of weight should equal to compressBlockSize, but got "
                                                "weight.shape[0]=%ld and compressBlockSize=%u",
                                                weightShape->GetStorageShape().GetDim(DIM_NUM_0), compressBlockSize),
                    return false);

        OP_CHECK_IF(
            headNum != outputCacheShape->GetStorageShape().GetDim(DIM_NUM_1),
            OPS_REPORT_VECTOR_INNER_ERR(
                "NsaCompressWithCache",
                "2nd dim of outputCache should equal to headNum, but got outputCache.shape[1]=%ld and headNum=%u",
                outputCacheShape->GetStorageShape().GetDim(DIM_NUM_1), headNum),
            return false);
        OP_CHECK_IF(
            headDim != outputCacheShape->GetStorageShape().GetDim(DIM_NUM_2),
            OPS_REPORT_VECTOR_INNER_ERR(
                "NsaCompressWithCache",
                "3rd dim of outputCache should equal to headDim, but got outputCache.shape[2]=%ld and headDim=%u",
                outputCacheShape->GetStorageShape().GetDim(DIM_NUM_1), headDim),
            return false);

        OP_CHECK_IF(headDim % DIM_ALIGN != 0,
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache",
                                                "headDim should be divisible by 16 , but got headDim=%u", headDim),
                    return false);

        OP_CHECK_IF(
            compressBlockSize > MAX_COMPRESS_BLOCK_SIZE,
            OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "do not support compressBlockSize larger than 64."),
            return false);
        OP_CHECK_IF(headNum == 0, OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "headNum can not be 0!"),
                    return false);
        OP_CHECK_IF(headNum > MAX_COMPRESS_BLOCK_SIZE,
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "do not support headNum larger than 64."),
                    return false);
        OP_CHECK_IF(headNum > MAX_ODD_HEAD_SIZE && headNum % 2 == 1,
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "do not support odd headNum larger than 50."),
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
            OP_CHECK_IF(
                actSeqLenDtype != ge::DT_INT64,
                OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "invalid actSeqLen dtype[%s], only support int64.",
                                            ge::TypeUtils::DataTypeToSerialString(actSeqLenDtype).c_str()),
                return false);
            batchSize = actSeqLenShape->GetStorageShape().GetDim(0);
            OP_CHECK_IF(
                batchSize > MAX_BATCH_SIZE,
                OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "do not support batchSize larger than 2048."),
                return false);
            AnalyzeActSeqLenData();
        }
        blockTableExistFlag = 0;
        auto blockTableShape = context_->GetOptionalInputShape(BLOCK_TABLE_INPUT_INDEX);
        auto blockTableInput = context_->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX);
        if (blockTableShape != nullptr && blockTableShape->GetStorageShape().GetDimNum() != 0) {
            blockTableExistFlag = 1;
            auto blockTableDtype = blockTableInput->GetDataType();
            OP_CHECK_IF(
                blockTableDtype != ge::DT_INT32,
                OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "invalid blockTable dtype[%s], only support int32.",
                                            ge::TypeUtils::DataTypeToSerialString(blockTableDtype).c_str()),
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
                        OPS_REPORT_VECTOR_INNER_ERR(
                            "NsaCompressWithCache",
                            "The inputLayout should be BSH/SBH/BSND/BNSD/TND(case-insensitive), but got %s.",
                            inputLayoutStr.c_str()),
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
            OP_CHECK_IF(headsNumPerCore == 0, OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "input too large."),
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
            OP_CHECK_IF(compileInfoPtr == nullptr,
                        OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "compileInfoPtr is null."),
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
        OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "invalid context."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeInputShape() || !AnalyzeDtype() || !AnalyzeOptionalInput(),
                    OPS_REPORT_VECTOR_INNER_ERR("NsaCompressWithCache", "fail to analyze context info."),
                    return ge::GRAPH_FAILED);
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
