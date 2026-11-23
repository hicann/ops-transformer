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
 * \file engram_fetch_tiling.cpp
 * \brief host侧tiling实现
 */

#include <string>
#include <climits>
#include <cstdint>
#include <algorithm>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../../../op_kernel/engram_fetch_tiling_data.h"
#include "../../../op_kernel/engram_fetch_tiling_key.h"
#include "../../../../engram_fetch_grad/op_kernel/engram_fetch_grad_tiling_data.h"

using namespace AscendC;
using namespace ge;

namespace Mc2Tiling {
constexpr uint32_t COMM_CONTEXT_INDEX = 0U;
constexpr uint32_t INDICES_INDEX = 1U;
constexpr uint32_t LOCAL_STORAGE_ADDR_INDEX = 2U;
constexpr uint32_t FETCHED_INDEX = 0U;
constexpr uint32_t PERM_OUT_INDEX = 1U;
constexpr uint32_t SEND_COUNTS_OUT_INDEX = 2U;
constexpr uint32_t RECV_COUNTS_OUT_INDEX = 3U;
constexpr uint32_t RECV_LOCAL_ENTRY_OUT_INDEX = 4U;
constexpr uint32_t NUM_RECV_OUT_INDEX = 5U;
constexpr uint32_t FETCHED_SF_INDEX = 6U;

constexpr uint32_t ATTR_HIDDEN_SIZE_INDEX = 0U;
constexpr uint32_t ATTR_NUM_ENTRIES_PER_RANK_INDEX = 1U;
constexpr uint32_t ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX = 2U;
constexpr uint32_t ATTR_COMM_BUFFER_SIZE_INDEX = 3U;
constexpr uint32_t ATTR_WITH_GRAD_INDEX = 4U;
constexpr uint32_t ATTR_HYBRID_MODE_INDEX = 5U;
constexpr int64_t RAIL_MODE_ALIGNED = 1;
constexpr uint32_t SF_TABLE_INDEX = 3U;

constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

constexpr int64_t UB_ALIGN = 32;
constexpr int64_t TILE_BYTES_INFER_MAX = 24 * 1024;
constexpr int64_t FLAG_SCRATCH_SIZE = 32;
constexpr int64_t WORKSPACE_ALIGN_2MB = 2 * 1024 * 1024;
constexpr int64_t SIMT_DCACHE_SIZE = 64 * 1024LL; // SortLib SIMT dcache 预留

static int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

static int64_t AlignTo(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

static const std::vector<ge::DataType> OUTPUT_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                                            ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

static bool IsContains(const std::vector<ge::DataType>& list, ge::DataType value)
{
    return std::find(list.begin(), list.end(), value) != list.end();
}

/**
 * @brief 打印tiling数据
 */
static void PrintEngramFetchTilingData(const EngramFetchTilingData* tilingData, const char* nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);

    OP_LOGD(nodeName, "========== EngramFetchTilingData ==========");
    OP_LOGD(nodeName, "numTokens is %ld", tilingData->numTokens);
    OP_LOGD(nodeName, "numEntriesPerRank is %d", tilingData->numEntriesPerRank);
    OP_LOGD(nodeName, "hiddenDim is %ld", tilingData->hiddenDim);
    OP_LOGD(nodeName, "hiddenBytes is %ld", tilingData->hiddenBytes);
    OP_LOGD(nodeName, "aivNum is %u", tilingData->aivNum);
    OP_LOGD(nodeName, "rankSize is %u", tilingData->rankSize);
    OP_LOGD(nodeName, "numMaxTokensPerRank is %ld", tilingData->numMaxTokensPerRank);
    OP_LOGD(nodeName, "totalRecv is %ld", tilingData->totalRecv);
    OP_LOGD(nodeName, "commBufferSize is %ld", tilingData->commBufferSize);
    OP_LOGD(nodeName, "sortNumTileData is %u", tilingData->sortNumTileData);
    OP_LOGD(nodeName, "sortTileCount is %u", tilingData->sortTileCount);
    OP_LOGD(nodeName, "sortTmpUbSize is %u", tilingData->sortTmpUbSize);
}

/**
 * @brief 校验tensor指针非空
 */
static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext* context)
{
    auto commContextDesc = context->GetInputDesc(COMM_CONTEXT_INDEX);
    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, commContextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, fetchedDesc);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor数据类型
 */
static ge::graphStatus CheckTensorDataType(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    auto commContextDesc = context->GetInputDesc(COMM_CONTEXT_INDEX);
    OP_TILING_CHECK(commContextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "commContext",
                                                          Ops::Base::ToString(commContextDesc->GetDataType()).c_str(),
                                                          "The dtype of commContext must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    OP_TILING_CHECK(indicesDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "indices",
                                                          Ops::Base::ToString(indicesDesc->GetDataType()).c_str(),
                                                          "The dtype of indices must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    ge::DataType fetchedDtype = fetchedDesc->GetDataType();
    OP_TILING_CHECK(
        !IsContains(OUTPUT_DTYPE_LIST, fetchedDtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            nodeName, "fetched", Ops::Base::ToString(fetchedDtype).c_str(),
            "The dtype of fetched must be DT_BF16, DT_FLOAT16, DT_FLOAT, DT_FLOAT8_E4M3FN or DT_FLOAT8_E5M2."),
        return ge::GRAPH_FAILED);

    auto localStorageAddrDesc = context->GetInputDesc(LOCAL_STORAGE_ADDR_INDEX);
    const gert::StorageShape* localStorageAddrShape = context->GetInputShape(LOCAL_STORAGE_ADDR_INDEX);
    if (localStorageAddrDesc != nullptr && localStorageAddrShape != nullptr) {
        OP_TILING_CHECK(
            localStorageAddrDesc->GetDataType() != ge::DT_INT64,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "localStorageAddr",
                                                  Ops::Base::ToString(localStorageAddrDesc->GetDataType()).c_str(),
                                                  "The dtype of localStorageAddr must be DT_INT64."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor维度
 */
static ge::graphStatus CheckTensorDim(const gert::TilingContext* context, int64_t& numTokens)
{
    const char* nodeName = context->GetNodeName();
    // input dim check
    const gert::StorageShape* commContextShape = context->GetInputShape(COMM_CONTEXT_INDEX);
    OP_TILING_CHECK(commContextShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        commContextShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "commContext", (std::to_string(commContextShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of commContext must be 1D."),
        return ge::GRAPH_FAILED);
    int64_t commContextDim0 = commContextShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "commContext dim0 = %ld", commContextDim0);
    OP_TILING_CHECK(commContextDim0 <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "commContext",
                                              (std::string("dim0=") + std::to_string(commContextDim0)).c_str(), "> 0"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* indicesShape = context->GetInputShape(INDICES_INDEX);
    OP_TILING_CHECK(indicesShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "indices"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        indicesShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "indices", (std::to_string(indicesShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of indices must be 1D."),
        return ge::GRAPH_FAILED);
    numTokens = indicesShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        numTokens < 0 || numTokens > UINT32_MAX,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "indices", (std::string("dim0=") + std::to_string(numTokens)).c_str(),
                                  "[0, UINT32_MAX]"),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "indices dim0 (numTokens) = %ld", numTokens);

    // output dim check
    const gert::StorageShape* fetchedShape = context->GetOutputShape(FETCHED_INDEX);
    OP_TILING_CHECK(fetchedShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "fetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        fetchedShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "fetched", (std::to_string(fetchedShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of fetched must be 2D."),
        return ge::GRAPH_FAILED);
    const int64_t fetchedDim0 = fetchedShape->GetStorageShape().GetDim(0);
    const int64_t fetchedDim1 = fetchedShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "fetched dim0 = %ld", fetchedDim0);
    OP_LOGD(nodeName, "fetched dim1 = %ld", fetchedDim1);
    OP_TILING_CHECK(fetchedDim0 != numTokens,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "fetched", (std::string("dim0=") + std::to_string(fetchedDim0)).c_str(),
                        (std::string("dim0 must equal numTokens=") + std::to_string(numTokens)).c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* localStorageAddrShape = context->GetInputShape(LOCAL_STORAGE_ADDR_INDEX);
    if (localStorageAddrShape != nullptr) {
        OP_TILING_CHECK(localStorageAddrShape->GetStorageShape().GetDimNum() != DIM_ONE,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "localStorageAddr",
                            (std::to_string(localStorageAddrShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                            "The shape dim of localStorageAddr must be 1D."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            localStorageAddrShape->GetStorageShape().GetDim(0) != 1,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName, "localStorageAddr",
                (std::string("dim0=") + std::to_string(localStorageAddrShape->GetStorageShape().GetDim(0))).c_str(),
                "dim0 must be 1."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor格式
 */
static ge::graphStatus CheckTensorFormat(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    auto commContextDesc = context->GetInputDesc(COMM_CONTEXT_INDEX);
    ge::Format commContextFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(commContextDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        commContextFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "commContext", Ops::Base::ToString(commContextFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    ge::Format indicesFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(indicesDesc->GetStorageFormat()));
    OP_TILING_CHECK(indicesFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "indices", Ops::Base::ToString(indicesFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    ge::Format fetchedFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(fetchedDesc->GetStorageFormat()));
    OP_TILING_CHECK(fetchedFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "fetched", Ops::Base::ToString(fetchedFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验所有tensor（指针、数据类型、维度、格式）
 */
static ge::graphStatus TilingCheckEngramFetch(const gert::TilingContext* context, int64_t& numTokens)
{
    const char* nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckTensorPtrNullptr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckTensorDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors and output tensors",
                                                 "The dtype of an input tensor or output tensor is invalid"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckTensorDim(context, numTokens) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors and output tensors",
                                                 "The shape of an input tensor or output tensor is invalid"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckTensorFormat(context) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors and output tensors",
                                                 "The format of an input tensor or output tensor is invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验属性指针非空
 */
static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE_INDEX);
    OP_TILING_CHECK(hiddenSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "hidden_size"),
                    return ge::GRAPH_FAILED);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    OP_TILING_CHECK(numEntriesPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "num_entries_per_rank"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验属性参数值
 */
static ge::graphStatus CheckAttrParams(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    const gert::StorageShape* fetchedShape = context->GetOutputShape(FETCHED_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, fetchedDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, fetchedShape);

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE_INDEX);
    OP_TILING_CHECK(*hiddenSizePtr <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size", std::to_string(*hiddenSizePtr).c_str(), "> 0"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "hidden_size is %ld", *hiddenSizePtr);

    // fetched dim1 must equal hidden_size attr
    int64_t hiddenDim = *hiddenSizePtr;
    const int64_t fetchedDim1 = fetchedShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(fetchedDim1 != hiddenDim,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "fetched", (std::string("dim1=") + std::to_string(fetchedDim1)).c_str(),
                        (std::string("dim1 must equal hidden_size attr(") + std::to_string(hiddenDim) + ").").c_str()),
                    return ge::GRAPH_FAILED);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    OP_TILING_CHECK(*numEntriesPerRankPtr < 0 || *numEntriesPerRankPtr > INT32_MAX,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_entries_per_rank",
                                              std::to_string(*numEntriesPerRankPtr).c_str(), "[0, INT32_MAX]"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "num_entries_per_rank is %ld", *numEntriesPerRankPtr);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验训练场景的额外属性和输出shape
 */
static ge::graphStatus CheckTrainingParams(const gert::TilingContext* context, int64_t numTokens)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX);
    OP_TILING_CHECK(numMaxTokensPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "num_max_tokens_per_rank"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*numMaxTokensPerRankPtr <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_max_tokens_per_rank",
                                              std::to_string(*numMaxTokensPerRankPtr).c_str(), "> 0"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *numMaxTokensPerRankPtr < numTokens,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_max_tokens_per_rank", std::to_string(*numMaxTokensPerRankPtr).c_str(),
                                  (std::string(">= numTokens(") + std::to_string(numTokens) + ")").c_str()),
        return ge::GRAPH_FAILED);

    auto commBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_BUFFER_SIZE_INDEX);
    OP_TILING_CHECK(commBufferSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "comm_buffer_size"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *commBufferSizePtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "comm_buffer_size", std::to_string(*commBufferSizePtr).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* permOutShape = context->GetOutputShape(PERM_OUT_INDEX);
    OP_TILING_CHECK(permOutShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "permOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        permOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "permOut", (std::to_string(permOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of permOut must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(permOutShape->GetStorageShape().GetDim(0) != numTokens,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "permOut",
                        (std::string("dim0=") + std::to_string(permOutShape->GetStorageShape().GetDim(0))).c_str(),
                        (std::string("dim0 must equal numTokens=") + std::to_string(numTokens)).c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* sendCountsOutShape = context->GetOutputShape(SEND_COUNTS_OUT_INDEX);
    OP_TILING_CHECK(sendCountsOutShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCountsOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "sendCountsOut",
                        (std::to_string(sendCountsOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of sendCountsOut must be 1D."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sendCountsOutShape->GetStorageShape().GetDim(0) <= 0,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "sendCountsOut",
            (std::string("dim0=") + std::to_string(sendCountsOutShape->GetStorageShape().GetDim(0))).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* recvCountsOutShape = context->GetOutputShape(RECV_COUNTS_OUT_INDEX);
    OP_TILING_CHECK(recvCountsOutShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvCountsOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "recvCountsOut",
                        (std::to_string(recvCountsOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of recvCountsOut must be 1D."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvCountsOutShape->GetStorageShape().GetDim(0) <= 0,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "recvCountsOut",
            (std::string("dim0=") + std::to_string(recvCountsOutShape->GetStorageShape().GetDim(0))).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* recvLocalEntryOutShape = context->GetOutputShape(RECV_LOCAL_ENTRY_OUT_INDEX);
    OP_TILING_CHECK(recvLocalEntryOutShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvLocalEntryOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvLocalEntryOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "recvLocalEntryOut",
                        (std::to_string(recvLocalEntryOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of recvLocalEntryOut must be 1D."),
                    return ge::GRAPH_FAILED);
    int64_t recvLocalEntryDim0 = recvLocalEntryOutShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        recvLocalEntryDim0 < 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "recvLocalEntryOut",
                                  (std::string("dim0=") + std::to_string(recvLocalEntryDim0)).c_str(), ">= 0"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* numRecvOutShape = context->GetOutputShape(NUM_RECV_OUT_INDEX);
    OP_TILING_CHECK(numRecvOutShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numRecvOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        numRecvOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "numRecvOut", (std::to_string(numRecvOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of numRecvOut must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numRecvOutShape->GetStorageShape().GetDim(0) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "numRecvOut",
                        (std::string("dim0=") + std::to_string(numRecvOutShape->GetStorageShape().GetDim(0))).c_str(),
                        "dim0 must be 1."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验所有属性
 */
static ge::graphStatus CheckAttrs(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAttrPtrNullptr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "attr params check nullptr failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check attr params failed."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool IsEmptySfPlaceholder(const gert::CompileTimeTensorDesc* desc, const gert::StorageShape* shape)
{
    if (desc == nullptr || shape == nullptr) {
        return true;
    }
    const auto& storageShape = shape->GetStorageShape();
    return storageShape.GetDimNum() == DIM_ONE && storageShape.GetDim(0) == 0;
}

static bool IsSfValueDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT8_E8M0;
}

static ge::graphStatus CheckFetchedSf(const char* nodeName, const gert::CompileTimeTensorDesc* desc,
                                      const gert::StorageShape* shape, int64_t numTokens, int64_t& dim1)
{
    const auto& storageShape = shape->GetStorageShape();
    OP_TILING_CHECK(storageShape.GetDimNum() != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "fetchedSf", (std::to_string(storageShape.GetDimNum()) + "D").c_str(),
                        "The shape dim of fetchedSf must be 2D when SF is provided."),
                    return ge::GRAPH_FAILED);
    int64_t fetchedSfDim0 = storageShape.GetDim(0);
    dim1 = storageShape.GetDim(1);
    OP_TILING_CHECK(dim1 <= 0 || !IsSfValueDtype(desc->GetDataType()),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        nodeName, "fetchedSf",
                        "dim1 must be positive and the dtype must be DT_FLOAT or DT_FLOAT8_E8M0 when fetchedSf is 2D"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(fetchedSfDim0 != numTokens,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "fetchedSf", (std::string("dim0=") + std::to_string(fetchedSfDim0)).c_str(),
                        (std::string("dim0 must equal numTokens=") + std::to_string(numTokens)).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTrainingSfTable(const char* nodeName, const gert::CompileTimeTensorDesc* desc,
                                            const gert::StorageShape* shape)
{
    OP_TILING_CHECK(
        desc == nullptr || shape == nullptr || shape->GetStorageShape().GetDimNum() != DIM_ONE ||
            shape->GetStorageShape().GetDim(0) != 1 || desc->GetDataType() != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            nodeName, "sfTable",
            "In training mode, sfTable must be a 1-element DT_INT64 device-address tensor when fetchedSf is provided"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInferenceSfTable(const char* nodeName, const gert::CompileTimeTensorDesc* desc,
                                             const gert::StorageShape* shape, int64_t fetchedSfDim1,
                                             ge::DataType fetchedSfDtype, int64_t numEntriesPerRank)
{
    OP_TILING_CHECK(desc == nullptr || shape == nullptr || shape->GetStorageShape().GetDimNum() != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "sfTable",
                        (std::to_string(shape == nullptr ? 0 : shape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of inference sfTable must be 2D when fetchedSf is provided."),
                    return ge::GRAPH_FAILED);

    int64_t sfTableDim0 = shape->GetStorageShape().GetDim(0);
    int64_t sfTableDim1 = shape->GetStorageShape().GetDim(1);
    // Inference gathers SF rows by global vocab index. The exact full-vocab row count is
    // validated by the Python wrapper, which knows world_size; tiling can only validate
    // that the supplied table is a positive whole-shard multiple.
    OP_TILING_CHECK(numEntriesPerRank <= 0 || sfTableDim0 <= 0 || sfTableDim0 % numEntriesPerRank != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "sfTable", (std::string("dim0=") + std::to_string(sfTableDim0)).c_str(),
                        (std::string("dim0 must be a positive multiple of num_entries_per_rank=") +
                         std::to_string(numEntriesPerRank))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sfTableDim1 != fetchedSfDim1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "sfTable", (std::string("dim1=") + std::to_string(sfTableDim1)).c_str(),
                        (std::string("dim1 must equal fetchedSf dim1=") + std::to_string(fetchedSfDim1)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsSfValueDtype(desc->GetDataType()) || desc->GetDataType() != fetchedSfDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "sfTable", Ops::Base::ToString(desc->GetDataType()).c_str(),
                        "The dtype of inference sfTable must match fetchedSf and be DT_FLOAT or DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSfTensors(const gert::TilingContext* context, bool isTraining, int64_t numTokens,
                                      int64_t numEntriesPerRank)
{
    const char* nodeName = context->GetNodeName();
    auto sfTableDesc = context->GetInputDesc(SF_TABLE_INDEX);
    auto sfTableShape = context->GetInputShape(SF_TABLE_INDEX);
    auto fetchedSfDesc = context->GetOutputDesc(FETCHED_SF_INDEX);
    auto fetchedSfShape = context->GetOutputShape(FETCHED_SF_INDEX);

    const bool hasSfTable = !IsEmptySfPlaceholder(sfTableDesc, sfTableShape);
    const bool hasFetchedSf = !IsEmptySfPlaceholder(fetchedSfDesc, fetchedSfShape);
    OP_TILING_CHECK(hasSfTable != hasFetchedSf,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "sfTable and fetchedSf",
                                                             "sfTable and fetchedSf must be provided together"),
                    return ge::GRAPH_FAILED);
    if (!hasFetchedSf) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t fetchedSfDim1 = 0;
    OP_TILING_CHECK(
        CheckFetchedSf(nodeName, fetchedSfDesc, fetchedSfShape, numTokens, fetchedSfDim1) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check fetchedSf failed."), return ge::GRAPH_FAILED);
    if (isTraining) {
        OP_TILING_CHECK(CheckTrainingSfTable(nodeName, sfTableDesc, sfTableShape) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "check training sfTable failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(
        CheckInferenceSfTable(nodeName, sfTableDesc, sfTableShape, fetchedSfDim1, fetchedSfDesc->GetDataType(),
                              numEntriesPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "sfTable", "The shape or dtype of sfTable is invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置平台信息
 */
static ge::graphStatus SetPlatformInfo(gert::TilingContext* context, EngramFetchTilingData& tilingData, bool isTraining)
{
    const char* nodeName = context->GetNodeName();

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.aivNum = aivNum;
    OP_LOGD(nodeName, "aicNum=%u, aivNum=%u, numBlocks=%u", aicNum, aivNum, numBlocks);

    if (isTraining) {
        // 训练模式 kernel 内借用 SortLib（SIMT）做收端全局排序，须预留 SIMT dcache
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        int64_t availUbSize = static_cast<int64_t>(ubSize) - SIMT_DCACHE_SIZE;
        OP_TILING_CHECK(availUbSize <= 0,
                        OP_LOGE(nodeName, "availUbSize(%lld) <= 0, ubSize=%llu, SIMT_DCACHE_SIZE=%lld", availUbSize,
                                ubSize, SIMT_DCACHE_SIZE),
                        return ge::GRAPH_FAILED);
        auto ret = context->SetLocalMemorySize(static_cast<size_t>(availUbSize));
        OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "SetLocalMemorySize failed, availUbSize=%lld", availUbSize),
                        return ge::GRAPH_FAILED);
        OP_LOGD(nodeName, "SetLocalMemorySize: ubSize=%llu, availUbSize=%lld", ubSize, availUbSize);
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tiling数据
 */
static ge::graphStatus SetTilingData(const gert::TilingContext* context, EngramFetchTilingData& tilingData,
                                     int64_t numTokens, bool isTraining)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    tilingData.numTokens = numTokens;
    tilingData.rankSize = 0;
    tilingData.numMaxTokensPerRank = 0;
    tilingData.totalRecv = 0;
    tilingData.commBufferSize = 0;
    tilingData.numSfPacks = 0;
    tilingData.sfElemSize = 0;
    tilingData.sortNumTileData = 0;
    tilingData.sortTileCount = 0;
    tilingData.sortTmpUbSize = 0;

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE_INDEX);
    tilingData.hiddenDim = *hiddenSizePtr;

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    tilingData.numEntriesPerRank = static_cast<int32_t>(*numEntriesPerRankPtr);

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    ge::DataType fetchedDtype = fetchedDesc->GetDataType();
    int64_t hiddenDim = tilingData.hiddenDim;
    int64_t bytesPerElem = ge::GetSizeByDataType(fetchedDtype);
    OP_TILING_CHECK(hiddenDim > INT64_MAX / bytesPerElem,
                    OP_LOGE(nodeName, "hiddenBytes overflow: hiddenDim=%ld * bytesPerElem=%ld exceeds INT64_MAX",
                            hiddenDim, bytesPerElem),
                    return ge::GRAPH_FAILED);
    tilingData.hiddenBytes = hiddenDim * bytesPerElem;
    OP_TILING_CHECK(tilingData.hiddenBytes > UINT32_MAX,
                    OP_LOGE(nodeName, "hiddenBytes=%ld exceeds UINT32_MAX", tilingData.hiddenBytes),
                    return ge::GRAPH_FAILED);

    auto fetchedSfDesc = context->GetOutputDesc(FETCHED_SF_INDEX);
    if (fetchedSfDesc != nullptr) {
        const gert::StorageShape* fetchedSfShape = context->GetOutputShape(FETCHED_SF_INDEX);
        if (fetchedSfShape != nullptr && fetchedSfShape->GetStorageShape().GetDimNum() == DIM_TWO) {
            tilingData.numSfPacks = fetchedSfShape->GetStorageShape().GetDim(1);
        }
        ge::DataType sfDtype = fetchedSfDesc->GetDataType();
        tilingData.sfElemSize = ge::GetSizeByDataType(sfDtype);
    }
    if (tilingData.numSfPacks != 0) {
        OP_TILING_CHECK(tilingData.numSfPacks < 0 || tilingData.sfElemSize <= 0 ||
                            tilingData.numSfPacks > INT64_MAX / tilingData.sfElemSize,
                        OP_LOGE(nodeName, "invalid sf row size: numSfPacks=%ld, sfElemSize=%ld", tilingData.numSfPacks,
                                tilingData.sfElemSize),
                        return ge::GRAPH_FAILED);
        if (!isTraining) {
            const int64_t sfBytes = tilingData.numSfPacks * tilingData.sfElemSize;
            int64_t relayTileBytes = AlignTo(tilingData.hiddenBytes, UB_ALIGN);
            if (relayTileBytes > TILE_BYTES_INFER_MAX) {
                relayTileBytes = TILE_BYTES_INFER_MAX;
            }
            OP_TILING_CHECK(
                sfBytes > relayTileBytes,
                OP_LOGE(nodeName,
                        "inference sf row size %ld exceeds relay tile bytes %ld; numSfPacks=%ld, "
                        "sfElemSize=%ld, hiddenBytes=%ld",
                        sfBytes, relayTileBytes, tilingData.numSfPacks, tilingData.sfElemSize, tilingData.hiddenBytes),
                return ge::GRAPH_FAILED);
        } else {
            // 训练分支: kernel侧行宽按uint32参与对齐与容量守卫, 对齐后之和超UINT32_MAX将在kernel内回绕
            const int64_t sfBytes = tilingData.numSfPacks * tilingData.sfElemSize;
            OP_TILING_CHECK(AlignTo(tilingData.hiddenBytes, UB_ALIGN) + AlignTo(sfBytes, UB_ALIGN) >
                                static_cast<int64_t>(UINT32_MAX),
                            OP_LOGE(nodeName,
                                    "training row size hiddenBytes=%ld + sfBytes=%ld exceeds UINT32_MAX after align; "
                                    "numSfPacks=%ld, sfElemSize=%ld",
                                    tilingData.hiddenBytes, sfBytes, tilingData.numSfPacks, tilingData.sfElemSize),
                            return ge::GRAPH_FAILED);
        }
    }

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (isTraining) {
        OP_TILING_CHECK(static_cast<int64_t>(ubSize) <= SIMT_DCACHE_SIZE,
                        OP_LOGE(nodeName, "ubSize(%llu) <= SIMT_DCACHE_SIZE(%lld)", ubSize, SIMT_DCACHE_SIZE),
                        return ge::GRAPH_FAILED);
        ubSize -= SIMT_DCACHE_SIZE;
    }
    tilingData.ubSize = ubSize;

    if (isTraining) {
        auto permOutShape = context->GetOutputShape(PERM_OUT_INDEX);
        auto sendCountsOutShape = context->GetOutputShape(SEND_COUNTS_OUT_INDEX);
        auto recvLocalEntryOutShape = context->GetOutputShape(RECV_LOCAL_ENTRY_OUT_INDEX);
        if (permOutShape != nullptr && sendCountsOutShape != nullptr) {
            tilingData.rankSize = static_cast<uint32_t>(sendCountsOutShape->GetStorageShape().GetDim(0));
        }
        if (recvLocalEntryOutShape != nullptr) {
            tilingData.totalRecv = recvLocalEntryOutShape->GetStorageShape().GetDim(0);
        }
        auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX);
        if (numMaxTokensPerRankPtr != nullptr) {
            tilingData.numMaxTokensPerRank = *numMaxTokensPerRankPtr;
        }
        auto commBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_BUFFER_SIZE_INDEX);
        if (commBufferSizePtr != nullptr) {
            tilingData.commBufferSize = *commBufferSizePtr;
        }

        // 与 kernel InitWinLayout 的 fixedSize 公式同源：跨kernel保留区 2*nr*32 + indices 计数区
        // 4*nr*32 + token 计数区 2*nr*32*groupSize（groupSize >= 1），下限取 groupSize=1 时的
        // nr*32*8。commBufferSize 低于该下限时 kernel 侧 remaining=0、槽容量派生为 0，直接拒绝。
        uint64_t winFixedSizeLower = static_cast<uint64_t>(tilingData.rankSize) * 32ULL * 8ULL;
        OP_TILING_CHECK(tilingData.commBufferSize < static_cast<int64_t>(winFixedSizeLower),
                        OP_LOGE(nodeName,
                                "commBufferSize(%ld) is smaller than window fixedSize lower bound(%llu), "
                                "kernel would derive zero slot capacity",
                                tilingData.commBufferSize, static_cast<unsigned long long>(winFixedSizeLower)),
                        return ge::GRAPH_FAILED);

        // SortLib 参数：请求方对输入 indices 全局排序（key = entry id，规模 numTokens）
        constexpr uint64_t kSortLibFixedBytes = 5120U;
        // sortlib 内部 histUb/histCumsumUb 队列按 256 bin × uint16 = 512B 使用
        // 而 outValueQueue_ 仅按 numTileData 字节分配，故 numTileData 下界为 512
        constexpr uint64_t kMinTileElems = 512U;
        constexpr uint64_t kBytesPerElem = 21U;
        uint64_t budget = tilingData.ubSize;
        OP_TILING_CHECK(budget < kSortLibFixedBytes + kMinTileElems * kBytesPerElem,
                        OP_LOGE(nodeName, "SortLib UB budget too small: ubSize=%llu, need>=%llu",
                                static_cast<unsigned long long>(budget),
                                static_cast<unsigned long long>(kSortLibFixedBytes + kMinTileElems * kBytesPerElem)),
                        return ge::GRAPH_FAILED);
        int64_t numTokensI64 = tilingData.numTokens;
        int64_t aivNumI64 = static_cast<int64_t>(tilingData.aivNum);
        uint32_t numTile = static_cast<uint32_t>((numTokensI64 + aivNumI64 - 1) / aivNumI64);
        uint32_t ubCap = static_cast<uint32_t>((budget - kSortLibFixedBytes) / kBytesPerElem);
        if (numTile < kMinTileElems) {
            numTile = kMinTileElems;
        }
        if (numTile > ubCap) {
            numTile = ubCap;
        }
        tilingData.sortNumTileData = numTile;
        tilingData.sortTileCount =
            static_cast<uint32_t>((numTokensI64 + static_cast<int64_t>(numTile) - 1) / static_cast<int64_t>(numTile));
        constexpr uint32_t kSortTmpHeadBytes = 512U;
        constexpr uint32_t kSortTmpTailBytes = 256U;
        constexpr uint32_t kSortTmpBytesPerGroup = 7U;
        constexpr uint32_t kSortTmpGroupTiles = 32U;
        uint32_t roundTile = (numTile + kSortTmpGroupTiles - 1U) / kSortTmpGroupTiles * kSortTmpGroupTiles;
        tilingData.sortTmpUbSize = kSortTmpHeadBytes + kSortTmpBytesPerGroup * roundTile + kSortTmpTailBytes;
        OP_LOGD(nodeName, "SortLib params: numTile=%u tileCount=%u tmpUb=%u budget=%llu", tilingData.sortNumTileData,
                tilingData.sortTileCount, tilingData.sortTmpUbSize, static_cast<unsigned long long>(budget));
    }

    OP_LOGD(nodeName,
            "SetTilingData: numTokens=%ld, hiddenDim=%ld, numEntriesPerRank=%d, hiddenBytes=%ld, ubSize=%lu, "
            "rankSize=%u, numMaxTokensPerRank=%ld, totalRecv=%ld, commBufferSize=%ld, isTraining=%d",
            tilingData.numTokens, tilingData.hiddenDim, tilingData.numEntriesPerRank, tilingData.hiddenBytes,
            tilingData.ubSize, tilingData.rankSize, tilingData.numMaxTokensPerRank, tilingData.totalRecv,
            tilingData.commBufferSize, isTraining);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tiling key
 */
static void SetTilingKey(gert::TilingContext* context, bool isTraining, int64_t hybridMode)
{
    const char* nodeName = context->GetNodeName();
    bool hasSf = false;
    auto sfTableDesc = context->GetInputDesc(SF_TABLE_INDEX);
    auto sfTableShape = context->GetInputShape(SF_TABLE_INDEX);
    auto fetchedSfDesc = context->GetOutputDesc(FETCHED_SF_INDEX);
    auto fetchedSfShape = context->GetOutputShape(FETCHED_SF_INDEX);
    // hasSf 训练/推理统一判定: sf_table 与 fetched_sf 成对提供时为真。
    // 无 SF 时两者均为空占位 (1-D), fetched_sf 非 2D 使 hasSf 恒为假;
    // 训练的 sf_table 为本卡 SF 表地址 (1元 int64 张量), 推理为全量 SF 表。
    hasSf = sfTableDesc != nullptr && sfTableShape != nullptr && fetchedSfDesc != nullptr &&
            fetchedSfShape != nullptr && fetchedSfShape->GetStorageShape().GetDimNum() == DIM_TWO &&
            fetchedSfShape->GetStorageShape().GetDim(1) > 0;
    const uint64_t tilingKey =
        isTraining ? GET_TPL_TILING_KEY(ENGRAM_FETCH_TRAIN_MODE, hasSf) :
                     (hybridMode == RAIL_MODE_ALIGNED ? GET_TPL_TILING_KEY(ENGRAM_FETCH_RAIL_MODE, hasSf) :
                                                        GET_TPL_TILING_KEY(ENGRAM_FETCH_DEFAULT_MODE, hasSf));
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu] in engram_fetch (isTraining=%d, hybridMode=%ld, hasSf=%d).", tilingKey,
            isTraining, hybridMode, static_cast<int32_t>(hasSf));
}

/**
 * @brief 设置workspace大小
 */
static ge::graphStatus SetWorkSpace(gert::TilingContext* context, const EngramFetchTilingData& tilingData,
                                    bool isTraining)
{
    const char* nodeName = context->GetNodeName();
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    if (isTraining) {
        int64_t numRanks = static_cast<int64_t>(tilingData.rankSize);
        int64_t numTokens = tilingData.numTokens;
        int64_t aivNum = static_cast<int64_t>(tilingData.aivNum);
        OP_TILING_CHECK(aivNum <= 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "aivNum", std::to_string(aivNum).c_str(),
                                                              "The value of aivNum must be positive"),
                        return ge::GRAPH_FAILED);

        int64_t wsSdispls = AlignTo(numRanks * static_cast<int64_t>(sizeof(int64_t)), UB_ALIGN);
        int64_t wsRdispls = AlignTo(numRanks * static_cast<int64_t>(sizeof(int64_t)), UB_ALIGN);
        int64_t wsSortedIndices = AlignTo(numTokens * static_cast<int64_t>(sizeof(int32_t)), UB_ALIGN);
        int64_t wsCounterScratch = aivNum * UB_ALIGN;
        int64_t wsPartialCounts = aivNum * numRanks * static_cast<int64_t>(sizeof(int32_t));
        int64_t wsIndicesReadyFlag = AlignTo(numRanks * static_cast<int64_t>(sizeof(int32_t)), UB_ALIGN);
        // 数值安全防护：sf 分量非负、乘法不溢出，numSfPacks=0 为无 sf 合法场景
        OP_TILING_CHECK(tilingData.numSfPacks < 0 || tilingData.sfElemSize < 0 ||
                            (tilingData.sfElemSize > 0 && tilingData.numSfPacks > INT64_MAX / tilingData.sfElemSize),
                        OP_LOGE(nodeName, "invalid sf dims: numSfPacks=%ld, sfElemSize=%ld", tilingData.numSfPacks,
                                tilingData.sfElemSize),
                        return ge::GRAPH_FAILED);
        int64_t sfBytes = static_cast<int64_t>(tilingData.numSfPacks) * static_cast<int64_t>(tilingData.sfElemSize);
        int64_t rowStride = tilingData.hiddenBytes + sfBytes;
        OP_TILING_CHECK(tilingData.totalRecv < 0 || rowStride <= 0 || tilingData.totalRecv > INT64_MAX / rowStride,
                        OP_LOGE(nodeName, "token staging size overflow: totalRecv=%ld, rowStride=%ld",
                                tilingData.totalRecv, rowStride),
                        return ge::GRAPH_FAILED);
        int64_t wsTokenStaging = tilingData.totalRecv * rowStride;
        OP_LOGD(nodeName, "token staging bytes = %ld, totalRecv=%ld, rowStride=%ld", wsTokenStaging,
                tilingData.totalRecv, rowStride);

        // SortLib 六段 workspace，与 kernel 侧 InitWorkspaceLayout 逐段一致（按 numTokens 规模）；
        // 前段尺寸不保证 32B 对齐，与 kernel 一致在排序区前插入对齐 padding
        int64_t wsBeforeSort = wsSdispls + wsRdispls + wsSortedIndices + wsCounterScratch + wsPartialCounts +
                               wsIndicesReadyFlag + wsTokenStaging;
        int64_t wsSortPad = AlignTo(wsBeforeSort, UB_ALIGN) - wsBeforeSort;
        int64_t numTokensI64 = tilingData.numTokens;
        OP_TILING_CHECK(numTokensI64 > INT64_MAX / static_cast<int64_t>(sizeof(int32_t)),
                        OP_LOGE(nodeName, "sort array overflow: numTokens=%lld", numTokensI64),
                        return ge::GRAPH_FAILED);
        constexpr int64_t kRadixRounds = 4;
        int64_t slSeg0 = AlignTo(256LL * kRadixRounds * 4LL, UB_ALIGN);
        int64_t slSeg1 = AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * 256LL * kRadixRounds * 4LL, UB_ALIGN);
        int64_t slSeg2 = AlignTo(numTokensI64 * 4LL, UB_ALIGN);
        int64_t slSeg3 = 2LL * AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * 256LL * 2LL, UB_ALIGN);
        int64_t slSeg4 = AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * tilingData.sortNumTileData, UB_ALIGN);
        int64_t slSeg5 = AlignTo(numTokensI64 * 4LL, UB_ALIGN);
        int64_t wsSortTotal = wsSortPad + slSeg0 + slSeg1 + slSeg2 + slSeg3 + slSeg4 + slSeg5;

        int64_t wsTotal = wsBeforeSort + wsSortTotal;
        wsTotal = ((wsTotal + WORKSPACE_ALIGN_2MB - 1) / WORKSPACE_ALIGN_2MB) * WORKSPACE_ALIGN_2MB;
        wsTotal += SYSTEM_NEED_WORKSPACE;
        workSpaces[0] = static_cast<size_t>(wsTotal);
    } else {
        workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    }
    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief host侧tiling实现 — EngramFetch 分轨模型
 */

// 分轨段：由老 tiling 入口 EngramFetchTilingFunc 转调（同文件下方）。

constexpr uint32_t IN_COMM_CONTEXT = 0U;
constexpr uint32_t IN_INDICES = 1U;
constexpr uint32_t IN_LOCAL_STORAGE_ADDR = 2U;
constexpr uint32_t IN_SF_TABLE = 3U;
constexpr uint32_t OUT_FETCHED = 0U;
constexpr uint32_t OUT_PERM = 1U;
constexpr uint32_t OUT_SEND_COUNTS = 2U;
constexpr uint32_t OUT_RECV_COUNTS = 3U;
constexpr uint32_t OUT_RECV_LOCAL_ENTRY = 4U;
constexpr uint32_t OUT_NUM_RECV = 5U;
constexpr uint32_t OUT_FETCHED_SF = 6U;

constexpr uint32_t ATTR_HIDDEN_SIZE = 0U;
constexpr uint32_t ATTR_NUM_ENTRIES_PER_RANK = 1U;
constexpr uint32_t ATTR_NUM_MAX_TOKENS_PER_RANK = 2U;
constexpr uint32_t ATTR_COMM_BUFFER_SIZE = 3U;
constexpr uint32_t ATTR_WITH_GRAD = 4U;
// 拓扑感知 attr（host HCCL 感知自动透传；索引与 def 声明顺序一致——append-only 尾部）
constexpr uint32_t ATTR_RANKS_PER_SUPERNODE = 6U; // 每超节点卡数（非用户输入；hybrid_mode 占 5）

constexpr int64_t BUFFER_ALIGNMENT = 2 * 1024 * 1024;
constexpr uint32_t SEND_COUNTS_ALIGN_FACTOR = 8U; // send_counts_out 每项 32B 槽距（world = dim0/8）
constexpr uint32_t MIN_AIV_NUM = 4U;              // 分轨硬要求：S/F/O 三组各 ≥1

static const std::vector<ge::DataType> FETCH_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                                           ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

struct RailTopology {
    uint32_t ranksPerSupernode{0}; // 每超节点卡数（感知 attr）
    uint32_t supernodeCnt{0};      // 超节点数 = world/ranksPerSupernode（推导，含自己）
    uint32_t railCnt{0};           // 轨道对端数 = world/ranksPerSupernode − 1（推导）
    uint64_t winHead{0}; // (W+1)×32；W = ranksPerSupernode + world/ranksPerSupernode（写者槽 + epoch 槽）
    uint64_t areas{0};   // W = rail 格 world/ranksPerSupernode + owner 格 ranksPerSupernode（每族同构）
};

/// @brief 集群规模运行期推导：由感知数与 world 推导超节点数/轨道对端数/写者槽 W
/// @return RailTopology（与 kernel 侧 MakeRailTopo 同公式，零字面量）
static RailTopology DeriveRailTopology(uint32_t ranksPerSupernode, uint32_t world)
{
    RailTopology topo;
    topo.ranksPerSupernode = ranksPerSupernode;
    topo.supernodeCnt = (ranksPerSupernode > 0U) ? world / ranksPerSupernode : 0U;
    topo.railCnt = (topo.supernodeCnt > 0U) ? topo.supernodeCnt - 1U : 0U;
    topo.areas = static_cast<uint64_t>(topo.supernodeCnt) + ranksPerSupernode;
    topo.winHead = (topo.areas + 1ULL) * 32ULL;
    return topo;
}

/// @brief 分轨段输入/输出 dtype/format 校验（commContext 平铺 rail int32、计数输出 int32 等）
static ge::graphStatus CheckRailTensorDataType(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    auto commDesc = context->GetInputDesc(IN_COMM_CONTEXT);
    OP_TILING_CHECK(commDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "commContext",
                                                          Ops::Base::ToString(commDesc->GetDataType()).c_str(),
                                                          "The dtype must be DT_INT32."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(commDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(
                        nodeName, "commContext",
                        Ops::Base::ToString(static_cast<ge::Format>(commDesc->GetStorageFormat())).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    // 分轨硬校验：平铺 rail ctx（头部标量 + node 表两张，CTX_RAIL_ELEMS int32）——分发入口已查，此处兜底
    const gert::StorageShape* commShape = context->GetInputShape(IN_COMM_CONTEXT);
    OP_TILING_CHECK(commShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"), return ge::GRAPH_FAILED);
    int64_t ctxElems = commShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(ctxElems != Mc2Kernel::CTX_RAIL_ELEMS,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "commContext", (std::string("dim0=") + std::to_string(ctxElems)).c_str(),
                        (std::string("== ") + std::to_string(Mc2Kernel::CTX_RAIL_ELEMS) + " (flat rail ctx)").c_str()),
                    return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(IN_INDICES);
    OP_TILING_CHECK(indicesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "indices"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(indicesDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "indices",
                                                          Ops::Base::ToString(indicesDesc->GetDataType()).c_str(),
                                                          "The dtype must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    auto storageDesc = context->GetInputDesc(IN_LOCAL_STORAGE_ADDR);
    OP_TILING_CHECK(storageDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "localStorageAddr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(storageDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "localStorageAddr", Ops::Base::ToString(storageDesc->GetDataType()).c_str(),
                        "The dtype must be DT_INT64 (1-element device-address tensor)."),
                    return ge::GRAPH_FAILED);

    auto fetchedDesc = context->GetOutputDesc(OUT_FETCHED);
    OP_TILING_CHECK(fetchedDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "fetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(FETCH_DTYPE_LIST, fetchedDesc->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "fetched", Ops::Base::ToString(fetchedDesc->GetDataType()).c_str(),
                        "The dtype of fetched must be BF16/FP16/FLOAT/FLOAT8_E4M3FN/FLOAT8_E5M2."),
                    return ge::GRAPH_FAILED);

    // 四件计数输出 + num_recv 全部 int32 ND
    const char* int32Outputs[] = {"perm", "sendCountsOut", "recvCountsOut", "recvLocalEntryOut", "numRecvOut"};
    const uint32_t int32Idx[] = {OUT_PERM, OUT_SEND_COUNTS, OUT_RECV_COUNTS, OUT_RECV_LOCAL_ENTRY, OUT_NUM_RECV};
    for (size_t i = 0; i < sizeof(int32Idx) / sizeof(int32Idx[0]); i++) {
        auto desc = context->GetOutputDesc(int32Idx[i]);
        OP_TILING_CHECK(desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, int32Outputs[i]),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, int32Outputs[i],
                                                              Ops::Base::ToString(desc->GetDataType()).c_str(),
                                                              "The dtype must be DT_INT32."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/// @brief 分轨段 shape 校验并提取关键规模（numTokens/worldSize/totalRecv/hiddenDim/numSfPacks）
static ge::graphStatus CheckRailTensorDim(const gert::TilingContext* context, int64_t& numTokens, uint32_t& worldSize,
                                          int64_t& totalRecv, int64_t& hiddenDim, int64_t& numSfPacks)
{
    const char* nodeName = context->GetNodeName();
    numSfPacks = 0;

    const gert::StorageShape* indicesShape = context->GetInputShape(IN_INDICES);
    OP_TILING_CHECK(indicesShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "indices"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        indicesShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "indices", (std::to_string(indicesShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of indices must be 1D."),
        return ge::GRAPH_FAILED);
    numTokens = indicesShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numTokens < 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "indices",
                                              (std::string("dim0=") + std::to_string(numTokens)).c_str(), ">= 0"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* fetchedShape = context->GetOutputShape(OUT_FETCHED);
    OP_TILING_CHECK(fetchedShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "fetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(fetchedShape->GetStorageShape().GetDimNum() != DIM_TWO ||
                        fetchedShape->GetStorageShape().GetDim(0) != numTokens,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "fetched",
                        (std::to_string(fetchedShape->GetStorageShape().GetDim(0)) + "x" +
                         std::to_string(fetchedShape->GetStorageShape().GetDim(1)))
                            .c_str(),
                        (std::string("dim0 must equal indices dim0=") + std::to_string(numTokens)).c_str()),
                    return ge::GRAPH_FAILED);
    hiddenDim = fetchedShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(hiddenDim <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "fetched",
                                              (std::string("dim1=") + std::to_string(hiddenDim)).c_str(), "> 0"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* permShape = context->GetOutputShape(OUT_PERM);
    OP_TILING_CHECK(permShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "permOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        permShape->GetStorageShape().GetDimNum() != DIM_ONE || permShape->GetStorageShape().GetDim(0) != numTokens,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "permOut",
            (std::string("dim0=") + std::to_string(permShape->GetStorageShape().GetDim(0))).c_str(),
            (std::string("dim0 must equal indices dim0=") + std::to_string(numTokens)).c_str()),
        return ge::GRAPH_FAILED);

    // 分轨世界规则：world = send_counts_out.dim0/8（32B 槽距；反向读同一张张量同口径）
    const gert::StorageShape* sendCountsShape = context->GetOutputShape(OUT_SEND_COUNTS);
    OP_TILING_CHECK(sendCountsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCountsOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sendCountsShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "sendCountsOut", (std::to_string(sendCountsShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of sendCountsOut must be 1D."),
        return ge::GRAPH_FAILED);
    int64_t sendCountsDim0 = sendCountsShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(sendCountsDim0 <= 0 || sendCountsDim0 % SEND_COUNTS_ALIGN_FACTOR != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "sendCountsOut.dim0", std::to_string(sendCountsDim0).c_str(),
                                              "must be positive multiple of 8"),
                    return ge::GRAPH_FAILED);
    int64_t world = sendCountsDim0 / SEND_COUNTS_ALIGN_FACTOR;
    OP_TILING_CHECK(world <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "world", std::to_string(world).c_str(),
                                              "must be positive (send_counts_out.dim0/8)"),
                    return ge::GRAPH_FAILED);
    worldSize = static_cast<uint32_t>(world);

    const gert::StorageShape* recvCountsShape = context->GetOutputShape(OUT_RECV_COUNTS);
    OP_TILING_CHECK(recvCountsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvCountsOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsShape->GetStorageShape().GetDimNum() != DIM_ONE ||
                        recvCountsShape->GetStorageShape().GetDim(0) != world,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "recvCountsOut",
                        (std::string("dim0=") + std::to_string(recvCountsShape->GetStorageShape().GetDim(0))).c_str(),
                        (std::string("dim0 must equal world=") + std::to_string(world)).c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* recvLocalEntryShape = context->GetOutputShape(OUT_RECV_LOCAL_ENTRY);
    OP_TILING_CHECK(recvLocalEntryShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvLocalEntryOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvLocalEntryShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "recvLocalEntryOut",
                        (std::to_string(recvLocalEntryShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of recvLocalEntryOut must be 1D."),
                    return ge::GRAPH_FAILED);
    totalRecv = recvLocalEntryShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(totalRecv < 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "recvLocalEntryOut",
                                              (std::string("dim0=") + std::to_string(totalRecv)).c_str(), ">= 0"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* numRecvShape = context->GetOutputShape(OUT_NUM_RECV);
    OP_TILING_CHECK(numRecvShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numRecvOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        numRecvShape->GetStorageShape().GetDimNum() != DIM_ONE || numRecvShape->GetStorageShape().GetDim(0) != 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "numRecvOut",
            (std::string("dim0=") + std::to_string(numRecvShape->GetStorageShape().GetDim(0))).c_str(),
            "dim0 must be 1."),
        return ge::GRAPH_FAILED);

    // SF 成对校验：fetchedSf 2D (T, sfPacks) 与 sfTable 1 元 int64 地址张量成对出现
    auto fetchedSfDesc = context->GetOutputDesc(OUT_FETCHED_SF);
    auto fetchedSfShape = context->GetOutputShape(OUT_FETCHED_SF);
    auto sfTableDesc = context->GetInputDesc(IN_SF_TABLE);
    auto sfTableShape = context->GetInputShape(IN_SF_TABLE);
    const bool hasFetchedSf = !IsEmptySfPlaceholder(fetchedSfDesc, fetchedSfShape);
    const bool hasSfTable = !IsEmptySfPlaceholder(sfTableDesc, sfTableShape);
    OP_TILING_CHECK(
        hasFetchedSf != hasSfTable,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "sfTable",
                                              (std::string("sfTable=") + std::to_string(static_cast<int>(hasSfTable)) +
                                               ", fetchedSf=" + std::to_string(static_cast<int>(hasFetchedSf)))
                                                  .c_str(),
                                              "sfTable and fetchedSf must be provided together"),
        return ge::GRAPH_FAILED);
    if (hasFetchedSf) {
        const auto& sfShape = fetchedSfShape->GetStorageShape();
        OP_TILING_CHECK(
            sfShape.GetDimNum() != DIM_TWO || sfShape.GetDim(0) != numTokens || sfShape.GetDim(1) <= 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName, "fetchedSf",
                (std::to_string(sfShape.GetDim(0)) + "x" + std::to_string(sfShape.GetDim(1))).c_str(),
                (std::string("must be (numTokens=") + std::to_string(numTokens) + ", positive sfPacks)").c_str()),
            return ge::GRAPH_FAILED);
        numSfPacks = sfShape.GetDim(1);
        OP_TILING_CHECK(sfTableDesc->GetDataType() != ge::DT_INT64 ||
                            sfTableShape->GetStorageShape().GetDimNum() != DIM_ONE ||
                            sfTableShape->GetStorageShape().GetDim(0) != 1,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "sfTable",
                            (std::string(Ops::Base::ToString(sfTableDesc->GetDataType())) + " " +
                             std::to_string(sfTableShape->GetStorageShape().GetDim(0)) + "-element")
                                .c_str(),
                            "must be a 1-element DT_INT64 device-address tensor when fetchedSf is provided"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/// @brief 分轨段 attr 校验并推导拓扑
/// @note with_grad 必须为 1；ranks_per_supernode 须 (0, 1024] 且整除 world；超节点数须 <= 2048
static ge::graphStatus CheckRailAttrs(const gert::TilingContext* context, RailTopology& topoAttr)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK);
    OP_TILING_CHECK(numEntriesPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "num_entries_per_rank"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*numEntriesPerRankPtr < 0 || *numEntriesPerRankPtr > INT32_MAX,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_entries_per_rank",
                                              std::to_string(*numEntriesPerRankPtr).c_str(), "[0, INT32_MAX]"),
                    return ge::GRAPH_FAILED);

    auto commBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_BUFFER_SIZE);
    OP_TILING_CHECK(commBufferSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "comm_buffer_size"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *commBufferSizePtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "comm_buffer_size", std::to_string(*commBufferSizePtr).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    auto withGradPtr = attrs->GetAttrPointer<int64_t>(ATTR_WITH_GRAD);
    OP_TILING_CHECK(withGradPtr == nullptr || *withGradPtr == 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "with_grad", "0 (or missing)", "1 (training path)"),
                    return ge::GRAPH_FAILED);

    // 感知 attr（host HCCL topo 感知自动填入，非用户输入）：ranksPerSupernode=每超节点卡数（直连域）。
    // 分轨模型唯一拓扑事实——超节点数=world÷ranksPerSupernode、轨道对端数=world÷ranksPerSupernode−1 全推导。
    auto ranksPerSupernodePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANKS_PER_SUPERNODE);
    OP_TILING_CHECK(ranksPerSupernodePtr == nullptr || *ranksPerSupernodePtr <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "ranks_per_supernode",
                        (ranksPerSupernodePtr == nullptr) ? "missing" : std::to_string(*ranksPerSupernodePtr).c_str(),
                        "rail-aligned tiling requires sensed attr ranks_per_supernode > 0; pass "
                        "ElasticBuffer._get_ranks_per_supernode_attr() after rank_size (stale cached graph or "
                        "direct aclnn call without sensing?)"),
                    return ge::GRAPH_FAILED);
    int64_t ranksPerSupernode = *ranksPerSupernodePtr;
    OP_TILING_CHECK(
        ranksPerSupernode > static_cast<int64_t>(Mc2Kernel::MAX_QP_SIZE),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "ranks_per_supernode", std::to_string(ranksPerSupernode).c_str(),
                                  (std::string("(0, ") + std::to_string(Mc2Kernel::MAX_QP_SIZE) +
                                   "] (rail ctx node table supernode-segment limit)")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    uint32_t worldFromShape = 0U;
    {
        const gert::StorageShape* sendCountsShape = context->GetOutputShape(OUT_SEND_COUNTS);
        OP_TILING_CHECK(sendCountsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCountsOut"),
                        return ge::GRAPH_FAILED);
        worldFromShape = static_cast<uint32_t>(sendCountsShape->GetStorageShape().GetDim(0) / SEND_COUNTS_ALIGN_FACTOR);
    }
    OP_TILING_CHECK(static_cast<int64_t>(worldFromShape) % ranksPerSupernode != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "ranks_per_supernode", std::to_string(ranksPerSupernode).c_str(),
                        (std::string("world=") + std::to_string(worldFromShape) +
                         " is not a multiple of ranks_per_supernode (sensed topology inconsistent with "
                         "send_counts_out.dim0/8)")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    // rail ctx node 表轨道段容量上限：超节点数 world/ranksPerSupernode ≤ MAX_SN_DOMAIN
    OP_TILING_CHECK(
        static_cast<int64_t>(worldFromShape) / ranksPerSupernode > static_cast<int64_t>(Mc2Kernel::MAX_SN_DOMAIN),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "supernode_count",
                                  std::to_string(static_cast<int64_t>(worldFromShape) / ranksPerSupernode).c_str(),
                                  (std::string("(0, ") + std::to_string(Mc2Kernel::MAX_SN_DOMAIN) +
                                   "] (rail ctx node table rail-segment limit)")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    // 判据：world ≥ ranksPerSupernode（world == ranksPerSupernode 单超节点全直连走老模板；
    // world < ranksPerSupernode 是 attr/shape 不一致）
    OP_TILING_CHECK(
        static_cast<int64_t>(worldFromShape) < ranksPerSupernode,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "world", std::to_string(worldFromShape).c_str(),
                                  (std::string(">= ranks_per_supernode (") + std::to_string(ranksPerSupernode) +
                                   "; rail-aligned handles cross-sn, smaller world means attrs/shape mismatch)")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    topoAttr = DeriveRailTopology(static_cast<uint32_t>(ranksPerSupernode), worldFromShape);
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────── 平台信息 ───────────────────────

/// @brief 平台信息落表：AIV 核数、blockDim、UB 预算（分轨要求 AIV >= 4，三组各 >= 1）
static ge::graphStatus SetRailPlatformInfo(gert::TilingContext* context, EngramFetchTilingData& tilingData)
{
    const char* nodeName = context->GetNodeName();

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    // 分轨硬要求：三组并发（S/F/O 常驻消费），至少 4 个 v 核
    OP_TILING_CHECK(aivNum < MIN_AIV_NUM,
                    OP_LOGE(nodeName, "rail requires at least %u AIV cores, got aivNum=%u", MIN_AIV_NUM, aivNum),
                    return ge::GRAPH_FAILED);
    uint32_t numBlocks = ascendcPlatform.CalcTschNumBlocks(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.aivNum = aivNum;

    // 训练模式 kernel 内借用 SortLib（SIMT）做输入排序，须预留 SIMT dcache
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t availUbSize = static_cast<int64_t>(ubSize) - SIMT_DCACHE_SIZE;
    OP_TILING_CHECK(availUbSize <= 0,
                    OP_LOGE(nodeName, "availUbSize(%lld) <= 0, ubSize=%llu, SIMT_DCACHE_SIZE=%lld", availUbSize, ubSize,
                            SIMT_DCACHE_SIZE),
                    return ge::GRAPH_FAILED);
    auto ret = context->SetLocalMemorySize(static_cast<size_t>(availUbSize));
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "SetLocalMemorySize failed, availUbSize=%lld", availUbSize),
                    return ge::GRAPH_FAILED);
    tilingData.ubSize = static_cast<uint64_t>(availUbSize);
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────── 分轨 tiling data ───────────────────────

/// @brief 分轨 tilingData 落表：拓扑、行距、N 反解、批行数、SortLib 参数、UB 预算校验
/// @note 行距：req 族恒 32B，resp 族按 hiddenBytes+sf 对齐到 32B；
///       N 反解把 commBufferSize 分满，批行数按 65504B 预算（blockLen 为 uint16）
static ge::graphStatus SetRailTilingData(gert::TilingContext* context, EngramFetchTilingData& tilingData,
                                         const RailTopology& topoArg, int64_t numTokens, uint32_t worldSize,
                                         int64_t totalRecv, int64_t hiddenDim, int64_t numSfPacks)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    tilingData.numTokens = numTokens;
    tilingData.rankSize = worldSize; // 分轨语义：world
    tilingData.totalRecv = totalRecv;
    tilingData.hiddenDim = hiddenDim;
    tilingData.numMaxTokensPerRank = 0; // 分轨不读（仅老路径容量口径）
    tilingData.numSfPacks = numSfPacks;

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK);
    tilingData.numEntriesPerRank = static_cast<int32_t>(*numEntriesPerRankPtr);
    // int32 空间校验：entry id 空间按 world 口径
    OP_TILING_CHECK(tilingData.numEntriesPerRank < 0 ||
                        (worldSize > 0U && static_cast<int64_t>(worldSize) * tilingData.numEntriesPerRank > INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_entries_per_rank",
                                              std::to_string(tilingData.numEntriesPerRank).c_str(),
                                              (std::string("[0, INT32_MAX/world (world=") + std::to_string(worldSize) +
                                               ", entry id space is world-scaled)")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    auto fetchedDesc = context->GetOutputDesc(OUT_FETCHED);
    ge::DataType fetchedDtype = fetchedDesc->GetDataType();
    int64_t bytesPerElem = ge::GetSizeByDataType(fetchedDtype);
    OP_TILING_CHECK(tilingData.hiddenDim > INT64_MAX / bytesPerElem,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "hidden_dim", std::to_string(tilingData.hiddenDim).c_str(),
                        (std::string("hiddenDim * bytesPerElem (") + std::to_string(bytesPerElem) +
                         ") overflows INT64_MAX (hiddenBytes overflow)")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    tilingData.hiddenBytes = tilingData.hiddenDim * bytesPerElem;

    if (numSfPacks > 0) {
        auto fetchedSfDesc = context->GetOutputDesc(OUT_FETCHED_SF);
        tilingData.sfElemSize = ge::GetSizeByDataType(fetchedSfDesc->GetDataType());
        OP_TILING_CHECK(tilingData.sfElemSize <= 0 || numSfPacks > INT64_MAX / tilingData.sfElemSize,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "fetched_sf",
                            (std::string("numSfPacks=") + std::to_string(numSfPacks) +
                             ", sfElemSize=" + std::to_string(tilingData.sfElemSize))
                                .c_str(),
                            "invalid sf row size (numSfPacks * sfElemSize overflows INT64_MAX or sfElemSize <= 0)"),
                        return ge::GRAPH_FAILED);
    } else {
        tilingData.sfElemSize = 0;
    }

    auto commBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_BUFFER_SIZE);
    tilingData.commBufferSize = *commBufferSizePtr;

    RailTopology topo = topoArg;
    tilingData.ranksPerSupernode = topo.ranksPerSupernode;

    // sfOff 对齐保证 S 拼块 UB 侧恒 32B 对齐
    const int64_t sfBytes = numSfPacks * tilingData.sfElemSize;
    const int64_t sfOff = AlignTo(tilingData.hiddenBytes, Mc2Kernel::UB_ALIGN);
    const int64_t rowStrideReq = Mc2Kernel::UB_ALIGN; // P_req = 32
    const int64_t rowStrideResp = AlignTo(Mc2Kernel::ROW_HEAD_BYTES + sfOff + sfBytes, Mc2Kernel::UB_ALIGN);
    OP_TILING_CHECK(rowStrideResp > UINT32_MAX,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "hidden_dim",
                                                          (std::string("P_resp=") + std::to_string(rowStrideResp) +
                                                           " (hiddenBytes=" + std::to_string(tilingData.hiddenBytes) +
                                                           ", sfBytes=" + std::to_string(sfBytes) + ")")
                                                              .c_str(),
                                                          "resp row stride overflows UINT32_MAX"),
                    return ge::GRAPH_FAILED);

    const int64_t headBudget =
        static_cast<int64_t>(topo.winHead) + 2LL * static_cast<int64_t>(topo.areas) * Mc2Kernel::RECV_AREA_HEAD;
    OP_TILING_CHECK(
        tilingData.commBufferSize <= headBudget,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "comm_buffer_size", std::to_string(tilingData.commBufferSize).c_str(),
                                  (std::string("> winHead + 2*areas*96 (winHead=") + std::to_string(topo.winHead) +
                                   ", areas=" + std::to_string(topo.areas) + "; WIN budget too small)")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    const int64_t perDestBytes = static_cast<int64_t>(topo.areas) * (rowStrideReq + rowStrideResp);
    int64_t nFit = (tilingData.commBufferSize - headBudget) / perDestBytes;
    // 批行数(族) = min(65504÷P(族), N÷2)：blockLen 为 uint16，满批 65536B 会截断为 0、
    constexpr int64_t BATCH_BLOCK_BYTES = 65504LL;
    // 硬拒绝超批块预算的行距（单行 > 65504 无法按批块搬运）
    OP_TILING_CHECK(rowStrideResp > BATCH_BLOCK_BYTES,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "hidden_dim",
                        (std::string("P_resp=") + std::to_string(rowStrideResp) +
                         " (hiddenBytes=" + std::to_string(tilingData.hiddenBytes) +
                         ", sfBytes=" + std::to_string(sfBytes) + ", budget=" + std::to_string(BATCH_BLOCK_BYTES) + ")")
                            .c_str(),
                        "resp row stride exceeds the batch block budget (blockLen uint16, single row not movable)"),
                    return ge::GRAPH_FAILED);
    int64_t batchRespCap = BATCH_BLOCK_BYTES / rowStrideResp;
    if (batchRespCap < 1LL) {
        batchRespCap = 1LL;
    }
    OP_TILING_CHECK(nFit < 2LL * batchRespCap,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "comm_buffer_size", std::to_string(tilingData.commBufferSize).c_str(),
                        (std::string("N=") + std::to_string(nFit) + " < 2*batchRowsResp=" +
                         std::to_string(batchRespCap) + " (P_req=" + std::to_string(rowStrideReq) +
                         ", P_resp=" + std::to_string(rowStrideResp) + "; buffer too small or row too large)")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    uint32_t numRows = static_cast<uint32_t>(nFit);
    uint32_t batchRowsResp = static_cast<uint32_t>((batchRespCap < numRows / 2LL) ? batchRespCap : numRows / 2);
    int64_t batchReqCap = BATCH_BLOCK_BYTES / rowStrideReq;
    uint32_t batchRowsReq = static_cast<uint32_t>((batchReqCap < numRows / 2LL) ? batchReqCap : numRows / 2);
    if (batchRowsReq < 1U) {
        batchRowsReq = 1U;
    }
    if (batchRowsResp < 1U) {
        batchRowsResp = 1U;
    }
    tilingData.numRows = numRows;
    tilingData.batchRowsReq = batchRowsReq;
    tilingData.batchRowsResp = batchRowsResp;

    int64_t cap = totalRecv;
    OP_TILING_CHECK(cap < 0 || cap > INT64_MAX / 8,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "recv_local_entry", std::to_string(cap).c_str(),
                                                          "collect cap overflow (cap < 0 or cap > INT64_MAX/8)"),
                    return ge::GRAPH_FAILED);

    uint32_t rowBlockBytes = static_cast<uint32_t>(batchRowsResp * rowStrideResp);
    uint32_t statusBytes = static_cast<uint32_t>(
        AlignTo(2 * Mc2Kernel::STATE_OFFSET + static_cast<int64_t>(tilingData.aivNum) * 8, Mc2Kernel::UB_ALIGN));
    uint32_t coreArrayBytes = static_cast<uint32_t>(
        AlignTo(AlignTo(static_cast<int64_t>(tilingData.aivNum) * sizeof(int32_t), Mc2Kernel::UB_ALIGN) +
                    static_cast<int64_t>(tilingData.aivNum) * sizeof(int32_t),
                Mc2Kernel::UB_ALIGN));
    uint32_t tempBytes = statusBytes;
    if (tempBytes < Mc2Kernel::ENTRY_BATCH_CAP * sizeof(int32_t)) {
        tempBytes = Mc2Kernel::ENTRY_BATCH_CAP * sizeof(int32_t);
    }
    if (tempBytes < coreArrayBytes) {
        tempBytes = coreArrayBytes;
    }
    uint32_t indicesBytes = Mc2Kernel::IDX_BUF_BYTES;
    // F 组流状态：每槽滚动 staged（ranksPerSupernode 个 int32）；S 组流状态：5 × 名下目的数 int32
    uint32_t flowBytes = static_cast<uint32_t>(
        AlignTo(static_cast<int64_t>(topo.ranksPerSupernode) * sizeof(int32_t), Mc2Kernel::UB_ALIGN));
    {
        uint32_t sDests = topo.ranksPerSupernode + topo.supernodeCnt;
        uint32_t sCnt = (tilingData.sendCoreCnt > 0U) ? tilingData.sendCoreCnt : 1U;
        uint32_t sFlowBytes = static_cast<uint32_t>(
            AlignTo(static_cast<int64_t>(5) * ((sDests + sCnt - 1U) / sCnt) * sizeof(int32_t), Mc2Kernel::UB_ALIGN));
        // 常驻 UB 预算账本（WinLocalCopy ping/pong 与 SortLib 分时复用，由 ubSize 总量兜底）
        const uint64_t creditSpinBytes = Mc2Kernel::UB_ALIGN;  // 信用读自旋格
        const uint64_t cursorBytes = 2U * Mc2Kernel::UB_ALIGN; // 游标
        uint64_t permanentUb = Mc2Kernel::HCOMM_INIT_SIZE + statusBytes + tempBytes + indicesBytes + creditSpinBytes +
                               Mc2Kernel::RECV_AREA_HEAD + static_cast<uint64_t>(rowBlockBytes) +
                               static_cast<uint64_t>(flowBytes) + static_cast<uint64_t>(sFlowBytes) + cursorBytes;
        OP_TILING_CHECK(
            permanentUb >= tilingData.ubSize,
            OP_LOGE(nodeName, "permanent UB (%llu) exceeds availUbSize (%llu)", permanentUb, tilingData.ubSize),
            return ge::GRAPH_FAILED);
    }

    constexpr uint64_t kSortLibFixedBytes = 5120U;
    constexpr uint64_t kMinTileElems = 512U;
    constexpr uint64_t kBytesPerElem = 21U;
    uint64_t budget = tilingData.ubSize;
    OP_TILING_CHECK(budget < kSortLibFixedBytes + kMinTileElems * kBytesPerElem,
                    OP_LOGE(nodeName, "SortLib UB budget too small: ubSize=%llu, need>=%llu", tilingData.ubSize,
                            kSortLibFixedBytes + kMinTileElems * kBytesPerElem),
                    return ge::GRAPH_FAILED);
    int64_t numTokensI64 = tilingData.numTokens;
    int64_t aivNumI64 = static_cast<int64_t>(tilingData.aivNum);
    uint32_t numTile = static_cast<uint32_t>((numTokensI64 + aivNumI64 - 1) / aivNumI64);
    uint32_t ubCap = static_cast<uint32_t>((budget - kSortLibFixedBytes) / kBytesPerElem);
    if (numTile < kMinTileElems) {
        numTile = kMinTileElems;
    }
    if (numTile > ubCap) {
        numTile = ubCap;
    }
    tilingData.sortNumTileData = numTile;
    tilingData.sortTileCount =
        static_cast<uint32_t>((numTokensI64 + static_cast<int64_t>(numTile) - 1) / static_cast<int64_t>(numTile));
    constexpr uint32_t kSortTmpHeadBytes = 512U;
    constexpr uint32_t kSortTmpTailBytes = 256U;
    constexpr uint32_t kSortTmpBytesPerGroup = 7U;
    constexpr uint32_t kSortTmpGroupTiles = 32U;
    uint32_t roundTile = (numTile + kSortTmpGroupTiles - 1U) / kSortTmpGroupTiles * kSortTmpGroupTiles;
    tilingData.sortTmpUbSize = kSortTmpHeadBytes + kSortTmpBytesPerGroup * roundTile + kSortTmpTailBytes;

    return ge::GRAPH_SUCCESS;
}

/// @brief 分轨 workspace 尺寸计算（与 kernel CalcWorkSpaceOffset 逐段一致）
static ge::graphStatus SetRailWorkSpace(gert::TilingContext* context, const EngramFetchTilingData& tilingData)
{
    const char* nodeName = context->GetNodeName();
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    int64_t world = static_cast<int64_t>(tilingData.rankSize);
    int64_t numTokens = tilingData.numTokens;
    uint32_t aivNum = (tilingData.aivNum > 0U) ? tilingData.aivNum : 1U;
    int64_t cap = tilingData.totalRecv; // cap = recv_local_entry.dim0
    RailTopology topo = DeriveRailTopology(tilingData.ranksPerSupernode, tilingData.rankSize);
    const int64_t sfBytes = tilingData.numSfPacks * tilingData.sfElemSize;
    const int64_t sfOff = AlignTo(tilingData.hiddenBytes, Mc2Kernel::UB_ALIGN);
    const int64_t rowStrideReq = Mc2Kernel::UB_ALIGN;
    const int64_t rowStrideResp = AlignTo(Mc2Kernel::ROW_HEAD_BYTES + sfOff + sfBytes, Mc2Kernel::UB_ALIGN);
    const int64_t ringStride =
        std::max(tilingData.batchRowsReq * rowStrideReq, tilingData.batchRowsResp * rowStrideResp);
    const int64_t slots = static_cast<int64_t>(topo.ranksPerSupernode); // F 暂存环仅 owner 槽

    OP_TILING_CHECK(numTokens > 0 && numTokens > INT64_MAX / 4,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "num_tokens", std::to_string(numTokens).c_str(),
                                                          "workspace overflow (numTokens > INT64_MAX/4)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cap > 0 && cap > INT64_MAX / 8,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "recv_local_entry", std::to_string(cap).c_str(),
                                                          "workspace overflow (cap > INT64_MAX/8)"),
                    return ge::GRAPH_FAILED);

    // ① sortedIndices：SortLib 产物（按 entry 升序）
    int64_t wsSortedIndices = AlignTo(numTokens * 4, Mc2Kernel::UB_ALIGN);
    // ② srcSorted：recv_local_entry 每行位的来源 src 伴随（B2 S 逐行取行头 dst）
    int64_t wsSrcSorted = AlignTo(cap * 4, Mc2Kernel::UB_ALIGN);
    // ③ 桶表：头部(taskCount+每核段数) + 核条带(段(dst,start,cnt)×3 int32) + 任务表((kind,dst,start,cnt)×4)
    //    条带容量 = max(T, world)/核 + 2（B1 按 owner 切段、B2 按 world 切片出非零源条带）
    int64_t stripCap = (numTokens > world ? numTokens : world) / aivNum + 2;
    int64_t wsBucketSeg = AlignTo((1 + aivNum) * 4, Mc2Kernel::UB_ALIGN) +
                          static_cast<int64_t>(aivNum) * stripCap * 3 * sizeof(int32_t) +
                          (numTokens + aivNum) * 4 * sizeof(int32_t);
    wsBucketSeg = AlignTo(wsBucketSeg, Mc2Kernel::UB_ALIGN);
    // ④ F 暂存环（两族分时复用）：fCoreCnt × ranksPerSupernode 槽 × ringStride——每 (F核,owner槽)
    //    私有环格（SPSC：F 备行/属主 S 核发走；背压 = staged−drained < 批行数(族)）
    int64_t wsRelay = static_cast<int64_t>(tilingData.fwdCoreCnt) * slots * ringStride;
    // ⑤ relay 三表：staged/drained/eof 各 align32(fCoreCnt×ranksPerSupernode×4)
    int64_t relayCtlBytes =
        AlignTo(static_cast<int64_t>(tilingData.fwdCoreCnt) * slots * sizeof(int32_t), Mc2Kernel::UB_ALIGN);
    int64_t wsRelayCtl = 3 * relayCtlBytes;
    // ⑥ S 拼块 GM 暂存：行块经此转 WriteNbi（每 S 核一格；两子阶段分时复用）
    int64_t wsSStage = static_cast<int64_t>(tilingData.sendCoreCnt) * ringStride;
    // ⑦ 收集数组：aivNum × cap × (src,entry) 对——每核独立游标段（B1 记账；C1 稳定散射输入）
    int64_t wsCollect = AlignTo(static_cast<int64_t>(aivNum) * cap * 8, Mc2Kernel::UB_ALIGN);
    // ⑧ srcArrive：world×4B（B1 每源到达 / C1 fill / B2 每源响应计数——分时复用）
    int64_t wsSrcArrive = AlignTo(world * sizeof(int32_t), Mc2Kernel::UB_ALIGN);
    // ⑨ 双前缀表：recvPrefix(rdispls)/sendPrefix(sdispls)，各含 [world] 总量位
    int64_t wsRecvPrefix = AlignTo((world + 1) * sizeof(int32_t), Mc2Kernel::UB_ALIGN);
    int64_t wsSendPrefix = AlignTo((world + 1) * sizeof(int32_t), Mc2Kernel::UB_ALIGN);
    // ⑩ counterScratch：信用读落地格（每核独占 32B）
    int64_t wsCounterScratch = static_cast<int64_t>(aivNum) * Mc2Kernel::UB_ALIGN;
    // ⑪ flagScratch：DFX 汇总槽 [0..15] + S 核 ③ 失败快照 [16 + c*4 ..+3]（与 kernel
    //    CalcWorkSpaceOffset 同宽）
    int64_t wsFlagScratch = (16 + static_cast<int64_t>(tilingData.sendCoreCnt) * 4) * 4;
    // ⑫ 前缀两遍 scratch + DFX 进度码
    int64_t coreArrayAligned = AlignTo(static_cast<int64_t>(aivNum) * sizeof(int32_t), Mc2Kernel::UB_ALIGN);
    int64_t wsSegCount = coreArrayAligned;
    int64_t wsCoreStart = coreArrayAligned;
    int64_t wsDfx = coreArrayAligned;

    // ⑬ SortLib 六段 workspace（老公式，numTokens 规模；与 kernel InitWorkspaceLayout 逐段一致）
    constexpr int64_t kRadixRounds = 4;
    int64_t slSeg0 = AlignTo(256LL * kRadixRounds * 4LL, Mc2Kernel::UB_ALIGN);
    int64_t slSeg1 =
        AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * 256LL * kRadixRounds * 4LL, Mc2Kernel::UB_ALIGN);
    int64_t slSeg2 = AlignTo(numTokens * 4LL, Mc2Kernel::UB_ALIGN);
    int64_t slSeg3 = 2LL * AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * 256LL * 2LL, Mc2Kernel::UB_ALIGN);
    int64_t slSeg4 =
        AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * tilingData.sortNumTileData, Mc2Kernel::UB_ALIGN);
    int64_t slSeg5 = AlignTo(numTokens * 4LL, Mc2Kernel::UB_ALIGN);
    int64_t wsSortTotal = slSeg0 + slSeg1 + slSeg2 + slSeg3 + slSeg4 + slSeg5;

    int64_t wsTotal = wsSortedIndices + wsSrcSorted + wsBucketSeg + wsRelay + wsRelayCtl + wsSStage + wsCollect +
                      wsSrcArrive + wsRecvPrefix + wsSendPrefix + wsCounterScratch + wsFlagScratch + wsSegCount +
                      wsCoreStart + wsDfx + wsSortTotal;
    wsTotal = AlignTo(wsTotal, BUFFER_ALIGNMENT);
    wsTotal += SYSTEM_NEED_WORKSPACE;

    workSpaces[0] = static_cast<size_t>(wsTotal);
    return ge::GRAPH_SUCCESS;
}

/// @brief EngramFetch 分轨模型 tiling 主入口
ge::graphStatus EngramFetchRailAlignedTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("engram_fetch_railaligned_tiling", "failed to get tiling context."),
                    return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT("engram_fetch_railaligned_tiling", "nodeName"),
                    return ge::GRAPH_FAILED);

    EngramFetchTilingData* tilingData = context->GetTilingData<EngramFetchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    int64_t numTokens = 0;
    uint32_t worldSize = 0;
    int64_t totalRecv = 0;
    int64_t hiddenDim = 0;
    int64_t numSfPacks = 0;
    RailTopology topoAttr;
    OP_TILING_CHECK(CheckRailTensorDataType(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail check input/output dtype/format failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckRailTensorDim(context, numTokens, worldSize, totalRecv, hiddenDim, numSfPacks) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "rail check input/output shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRailAttrs(context, topoAttr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail check attrs failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetRailPlatformInfo(context, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail set platform info failed."), return ge::GRAPH_FAILED);

    // 三组核数（S:F:O ≈ 1:2:2 查表；tiling 落表、kernel 校验兜底）
    {
        // S/F/O 三组核配比查表：aiv 按档取整定配比（O=余量），低档按 1:2:2 比例兜底
        constexpr uint32_t AIV_TIER_HI = 72U;
        constexpr uint32_t AIV_TIER_MID = 64U;
        constexpr uint32_t AIV_TIER_LOW = 56U;
        constexpr uint32_t TIER_HI_SEND = 14U;
        constexpr uint32_t TIER_HI_FWD = 29U;
        constexpr uint32_t TIER_MID_SEND = 12U;
        constexpr uint32_t TIER_MID_FWD = 26U;
        constexpr uint32_t TIER_LOW_SEND = 11U;
        constexpr uint32_t TIER_LOW_FWD = 22U;
        uint32_t aiv = tilingData->aivNum;
        uint32_t sendCores = 0;
        uint32_t fwdCores = 0;
        uint32_t ownerCores = 0;
        if (aiv >= AIV_TIER_HI) {
            sendCores = TIER_HI_SEND;
            fwdCores = TIER_HI_FWD;
            ownerCores = aiv - TIER_HI_SEND - TIER_HI_FWD;
        } else if (aiv >= AIV_TIER_MID) {
            sendCores = TIER_MID_SEND;
            fwdCores = TIER_MID_FWD;
            ownerCores = aiv - TIER_MID_SEND - TIER_MID_FWD;
        } else if (aiv >= AIV_TIER_LOW) {
            sendCores = TIER_LOW_SEND;
            fwdCores = TIER_LOW_FWD;
            ownerCores = aiv - TIER_LOW_SEND - TIER_LOW_FWD;
        } else {
            sendCores = (aiv / 5U >= 1U) ? aiv / 5U : 1U;
            fwdCores = (aiv >= 5U) ? 2U * aiv / 5U : 1U;
            ownerCores = (aiv > sendCores + fwdCores) ? aiv - sendCores - fwdCores : 1U;
            if (sendCores + fwdCores + ownerCores != aiv) {
                fwdCores = aiv - sendCores - 1U;
                ownerCores = 1U;
            }
        }
        OP_TILING_CHECK(
            sendCores < 1U || fwdCores < 1U || ownerCores < 1U ||
                static_cast<uint64_t>(sendCores) + fwdCores + ownerCores != aiv,
            OP_LOGE(nodeName, "invalid core partition: S=%u, F=%u, O=%u, aiv=%u", sendCores, fwdCores, ownerCores, aiv),
            return ge::GRAPH_FAILED);
        tilingData->sendCoreCnt = sendCores;
        tilingData->fwdCoreCnt = fwdCores;
        tilingData->ownerCoreCnt = ownerCores;
    }

    OP_TILING_CHECK(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail SetScheduleMode(1) failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetRailTilingData(context, *tilingData, topoAttr, numTokens, worldSize, totalRecv, hiddenDim,
                                      numSfPacks) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail set tiling data failed."), return ge::GRAPH_FAILED);

    // tilingKey：train×rail×hasSf 组合——框架按 TPL 列表实例化分轨 kernel 模板参数
    // （mode=3 训练 rail；mode=2 为推理 rail（PR #15511），tiling 按 isTraining 分流）
    {
        bool hasSf = false;
        auto fetchedSfDesc = context->GetOutputDesc(OUT_FETCHED_SF);
        auto fetchedSfShape = context->GetOutputShape(OUT_FETCHED_SF);
        hasSf = fetchedSfDesc != nullptr && fetchedSfShape != nullptr &&
                fetchedSfShape->GetStorageShape().GetDimNum() == DIM_TWO &&
                fetchedSfShape->GetStorageShape().GetDim(1) > 0;
        const uint64_t tilingKey = GET_TPL_TILING_KEY(ENGRAM_FETCH_TRAIN_RAIL_MODE, hasSf);
        context->SetTilingKey(tilingKey);
    }

    OP_TILING_CHECK(SetRailWorkSpace(context, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail set workspace failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief engram_fetch算子的tiling函数
 */
static ge::graphStatus EngramFetchTilingFunc(gert::TilingContext* context)
{
    const char* nodeName = (context != nullptr) ? context->GetNodeName() : "engram_fetch";
    OP_TILING_CHECK(context == nullptr, OP_LOGE(nodeName, "failed to get tiling context."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "nodeName"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Enter EngramFetch tiling check func.");

    EngramFetchTilingData* tilingData = context->GetTilingData<EngramFetchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    bool isTraining = false;
    int64_t hybridMode = 0;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto withGradPtr = attrs->GetAttrPointer<int64_t>(ATTR_WITH_GRAD_INDEX);
        if (withGradPtr != nullptr && *withGradPtr != 0) {
            isTraining = true;
        }
        auto hybridModePtr = attrs->GetAttrPointer<int64_t>(ATTR_HYBRID_MODE_INDEX);
        if (hybridModePtr != nullptr) {
            hybridMode = *hybridModePtr;
        }
    }
    OP_TILING_CHECK(hybridMode != 0 && hybridMode != RAIL_MODE_ALIGNED,
                    OP_LOGE(nodeName, "unsupported hybrid_mode: %ld", hybridMode), return ge::GRAPH_FAILED);

    if (isTraining) {
        constexpr int64_t DISPATCH_DEFAULT_RANKS_PER_SUPERNODE = 128;
        auto ranksPerSupernodePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANKS_PER_SUPERNODE);
        const bool attrSensed = ranksPerSupernodePtr != nullptr && *ranksPerSupernodePtr > 0 &&
                                *ranksPerSupernodePtr <= static_cast<int64_t>(Mc2Kernel::MAX_QP_SIZE);
        const int64_t ranksPerSupernode = attrSensed ? *ranksPerSupernodePtr : DISPATCH_DEFAULT_RANKS_PER_SUPERNODE;
        const gert::StorageShape* railCtxShape = context->GetInputShape(COMM_CONTEXT_INDEX);
        OP_TILING_CHECK(railCtxShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"),
                        return ge::GRAPH_FAILED);
        int64_t railCtxElems = railCtxShape->GetStorageShape().GetDim(0);
        const gert::StorageShape* railSendShape = context->GetOutputShape(SEND_COUNTS_OUT_INDEX);
        OP_TILING_CHECK(railSendShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCountsOut"),
                        return ge::GRAPH_FAILED);
        int64_t railSendDim0 = railSendShape->GetStorageShape().GetDim(0);
        int64_t railWorld = (railSendDim0 > 0 && railSendDim0 % SEND_COUNTS_ALIGN_FACTOR == 0) ?
                                railSendDim0 / SEND_COUNTS_ALIGN_FACTOR :
                                -1;
        if (railWorld > ranksPerSupernode && railCtxElems == Mc2Kernel::CTX_RAIL_ELEMS) {
            // 轨道分支：world > ranksPerSupernode（跨超节点轨道腿存在）且建链产物为平铺 rail ctx
            OP_LOGI(nodeName, "Enter rail-aligned tiling func, world=%lld, ranksPerSupernode=%lld.", railWorld,
                    ranksPerSupernode);
            return EngramFetchRailAlignedTilingFunc(context);
        }
        OP_TILING_CHECK(railWorld > ranksPerSupernode && railCtxElems != Mc2Kernel::CTX_RAIL_ELEMS &&
                            railCtxElems != 0 && (attrSensed || railCtxElems != Mc2Kernel::CTX_LEGACY_ELEMS),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "comm_context", std::to_string(railCtxElems).c_str(),
                            (std::string("world=") + std::to_string(railWorld) +
                             " (>ranks_per_supernode=" + std::to_string(ranksPerSupernode) +
                             ") requires flat rail ctx " + std::to_string(Mc2Kernel::CTX_RAIL_ELEMS) +
                             " int32; rebuild ElasticBuffer under rail-aligned link setup (allow_hybrid_mode=True)")
                                .c_str()),
                        return ge::GRAPH_FAILED);
    }

    // 1. tensor check (ptr + dtype + shape + format)
    int64_t numTokens = 0;
    OP_TILING_CHECK(TilingCheckEngramFetch(context, numTokens) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check input/output failed."), return ge::GRAPH_FAILED);

    // 2. attr check
    OP_TILING_CHECK(CheckAttrs(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check attrs failed."),
                    return ge::GRAPH_FAILED);

    // 2.1 SF tensor contract check
    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    OP_TILING_CHECK(numEntriesPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "num_entries_per_rank"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckSfTensors(context, isTraining, numTokens, *numEntriesPerRankPtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check sf tensors failed."), return ge::GRAPH_FAILED);

    // 2.2 training params check
    if (isTraining) {
        OP_TILING_CHECK(CheckTrainingParams(context, numTokens) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "check training params failed."), return ge::GRAPH_FAILED);
    }

    // 3. platform info
    OP_TILING_CHECK(SetPlatformInfo(context, *tilingData, isTraining) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set platform info failed."), return ge::GRAPH_FAILED);

    if (isTraining) {
        auto schedRet = context->SetScheduleMode(1);
        OP_TILING_CHECK(schedRet != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "SetScheduleMode(1) failed"),
                        return ge::GRAPH_FAILED);
    }

    // 4. set tiling data
    OP_TILING_CHECK(SetTilingData(context, *tilingData, numTokens, isTraining) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set tiling data failed."), return ge::GRAPH_FAILED);

    // 5. set tiling key
    SetTilingKey(context, isTraining, hybridMode);

    // 6. set workspace
    OP_TILING_CHECK(SetWorkSpace(context, *tilingData, isTraining) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set workspace failed."), return ge::GRAPH_FAILED);

    // 7. print info
    PrintEngramFetchTilingData(tilingData, nodeName);
    OP_LOGI(nodeName, "EngramFetch tiling end. (isTraining=%d)", isTraining);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetch).Tiling(EngramFetchTilingFunc);
} // namespace Mc2Tiling
