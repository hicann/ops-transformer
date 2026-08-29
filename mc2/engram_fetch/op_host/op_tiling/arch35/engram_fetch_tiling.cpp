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

constexpr uint32_t ATTR_HIDDEN_SIZE_INDEX = 0U;
constexpr uint32_t ATTR_NUM_ENTRIES_PER_RANK_INDEX = 1U;
constexpr uint32_t ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX = 2U;
constexpr uint32_t ATTR_COMM_BUFFER_SIZE_INDEX = 3U;
constexpr uint32_t ATTR_WITH_GRAD_INDEX = 4U;

constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

constexpr int32_t HIDDEN_SIZE_ALIGN = 128;
constexpr int64_t UB_ALIGN = 32;
constexpr int64_t FLAG_SCRATCH_SIZE = 32;
constexpr int64_t WORKSPACE_ALIGN_2MB = 2 * 1024 * 1024;

static int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

static int64_t AlignTo(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

static const std::vector<ge::DataType> OUTPUT_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

static bool IsContains(const std::vector<ge::DataType> &list, ge::DataType value)
{
    return std::find(list.begin(), list.end(), value) != list.end();
}

/**
 * @brief 打印tiling数据
 */
static void PrintEngramFetchTilingData(const EngramFetchTilingData *tilingData, const char *nodeName)
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
}

/**
 * @brief 校验tensor指针非空
 */
static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context)
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
static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

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
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "fetched", Ops::Base::ToString(fetchedDtype).c_str(),
                                              "The dtype of fetched must be DT_BF16, DT_FLOAT16 or DT_FLOAT."),
        return ge::GRAPH_FAILED);

    auto localStorageAddrDesc = context->GetInputDesc(LOCAL_STORAGE_ADDR_INDEX);
    const gert::StorageShape *localStorageAddrShape = context->GetInputShape(LOCAL_STORAGE_ADDR_INDEX);
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
static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, int64_t &numTokens)
{
    const char *nodeName = context->GetNodeName();
    // input dim check
    const gert::StorageShape *commContextShape = context->GetInputShape(COMM_CONTEXT_INDEX);
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

    const gert::StorageShape *indicesShape = context->GetInputShape(INDICES_INDEX);
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
    OP_LOGD(nodeName, "indices dim0 (numTokens) = %ld", numTokens);

    // output dim check
    const gert::StorageShape *fetchedShape = context->GetOutputShape(FETCHED_INDEX);
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

    const gert::StorageShape *localStorageAddrShape = context->GetInputShape(LOCAL_STORAGE_ADDR_INDEX);
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
static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

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
static ge::graphStatus TilingCheckEngramFetch(const gert::TilingContext *context, int64_t &numTokens)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckTensorPtrNullptr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDataType(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDim(context, numTokens) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorFormat(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "params format is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验属性指针非空
 */
static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
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
static ge::graphStatus CheckAttrParams(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    const gert::StorageShape *fetchedShape = context->GetOutputShape(FETCHED_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, fetchedDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, fetchedShape);

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE_INDEX);
    OP_TILING_CHECK(*hiddenSizePtr <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size", std::to_string(*hiddenSizePtr).c_str(), "> 0"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *hiddenSizePtr % HIDDEN_SIZE_ALIGN != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size", std::to_string(*hiddenSizePtr).c_str(),
                                  (std::string("must be ") + std::to_string(HIDDEN_SIZE_ALIGN) + "-aligned").c_str()),
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
static ge::graphStatus CheckTrainingParams(const gert::TilingContext *context, int64_t numTokens)
{
    const char *nodeName = context->GetNodeName();
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

    const gert::StorageShape *permOutShape = context->GetOutputShape(PERM_OUT_INDEX);
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

    const gert::StorageShape *sendCountsOutShape = context->GetOutputShape(SEND_COUNTS_OUT_INDEX);
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

    const gert::StorageShape *recvCountsOutShape = context->GetOutputShape(RECV_COUNTS_OUT_INDEX);
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

    const gert::StorageShape *recvLocalEntryOutShape = context->GetOutputShape(RECV_LOCAL_ENTRY_OUT_INDEX);
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

    const gert::StorageShape *numRecvOutShape = context->GetOutputShape(NUM_RECV_OUT_INDEX);
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
static ge::graphStatus CheckAttrs(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAttrPtrNullptr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "attr params check nullptr failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check attr params failed."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置平台信息
 */
static ge::graphStatus SetPlatformInfo(gert::TilingContext *context, EngramFetchTilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.aivNum = aivNum;
    OP_LOGD(nodeName, "aicNum=%u, aivNum=%u, numBlocks=%u", aicNum, aivNum, numBlocks);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tiling数据
 */
static ge::graphStatus SetTilingData(gert::TilingContext *context, EngramFetchTilingData &tilingData, int64_t numTokens,
                                     bool isTraining)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    tilingData.numTokens = numTokens;
    tilingData.rankSize = 0;
    tilingData.numMaxTokensPerRank = 0;
    tilingData.totalRecv = 0;
    tilingData.commBufferSize = 0;

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

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
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
static void SetTilingKey(gert::TilingContext *context, bool isTraining)
{
    const char *nodeName = context->GetNodeName();
    const uint64_t tilingKey =
        isTraining ? GET_TPL_TILING_KEY(ENGRAM_FETCH_TRAIN_MODE) : GET_TPL_TILING_KEY(ENGRAM_FETCH_DEFAULT_MODE);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu] in engram_fetch (isTraining=%d).", tilingKey, isTraining);
}

/**
 * @brief 设置workspace大小
 */
static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const EngramFetchTilingData &tilingData,
                                    bool isTraining)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    if (isTraining) {
        int64_t numRanks = static_cast<int64_t>(tilingData.rankSize);
        int64_t numTokens = tilingData.numTokens;
        int64_t hiddenBytes = tilingData.hiddenBytes;
        int64_t totalRecv = tilingData.totalRecv;
        int64_t aivNum = static_cast<int64_t>(tilingData.aivNum);
        OP_TILING_CHECK(aivNum <= 0, OP_LOGE(nodeName, "aivNum is %ld, must be positive.", aivNum),
                        return ge::GRAPH_FAILED);

        int64_t wsSdispls = AlignTo(numRanks * static_cast<int64_t>(sizeof(int64_t)), UB_ALIGN);
        int64_t wsRdispls = AlignTo(numRanks * static_cast<int64_t>(sizeof(int64_t)), UB_ALIGN);
        int64_t wsSortedIndices = AlignTo(numTokens * static_cast<int64_t>(sizeof(int32_t)), UB_ALIGN);
        int64_t slotSize = AlignTo(tilingData.numMaxTokensPerRank * static_cast<int64_t>(sizeof(int32_t)), UB_ALIGN);
        if (slotSize == 0) {
            slotSize = UB_ALIGN;
        }
        int64_t rankCores = (aivNum < numRanks) ? aivNum : numRanks;
        int64_t numOwnerRanksMax = (numRanks + rankCores - 1) / rankCores;
        if (numOwnerRanksMax == 0) {
            numOwnerRanksMax = 1;
        }
        int64_t perCoreTempSize = numOwnerRanksMax * slotSize;
        int64_t wsSortedIndicesTemp = aivNum * perCoreTempSize;
        int64_t wsPermOutTemp = aivNum * perCoreTempSize;
        int64_t wsLocalData = totalRecv * hiddenBytes;
        int64_t wsRecvData = numTokens * hiddenBytes;
        int64_t wsCounterScratch = aivNum * UB_ALIGN;
        int64_t wsPartialCounts = aivNum * numRanks * static_cast<int64_t>(sizeof(int32_t));
        int64_t wsFlagScratch = aivNum * UB_ALIGN;
        int64_t wsIndicesReadyFlag = numRanks * static_cast<int64_t>(sizeof(int32_t));

        int64_t wsTotal = wsSdispls + wsRdispls + wsSortedIndices + wsSortedIndicesTemp + wsPermOutTemp + wsLocalData +
                          wsRecvData + wsCounterScratch + wsPartialCounts + wsFlagScratch + wsIndicesReadyFlag;
        wsTotal = ((wsTotal + WORKSPACE_ALIGN_2MB - 1) / WORKSPACE_ALIGN_2MB) * WORKSPACE_ALIGN_2MB;
        wsTotal += SYSTEM_NEED_WORKSPACE;
        workSpaces[0] = static_cast<size_t>(wsTotal);
    } else {
        workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief engram_fetch算子的tiling函数
 */
static ge::graphStatus EngramFetchTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("engram_fetch_tiling", "failed to get tiling context."),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "nodeName"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Enter EngramFetch tiling check func.");

    EngramFetchTilingData *tilingData = context->GetTilingData<EngramFetchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    bool isTraining = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto withGradPtr = attrs->GetAttrPointer<int64_t>(ATTR_WITH_GRAD_INDEX);
        if (withGradPtr != nullptr && *withGradPtr != 0) {
            isTraining = true;
        }
    }

    // 1. tensor check (ptr + dtype + shape + format)
    int64_t numTokens = 0;
    OP_TILING_CHECK(TilingCheckEngramFetch(context, numTokens) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check input/output failed."), return ge::GRAPH_FAILED);

    // 2. attr check
    OP_TILING_CHECK(CheckAttrs(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check attrs failed."),
                    return ge::GRAPH_FAILED);

    // 2.1 training params check
    if (isTraining) {
        OP_TILING_CHECK(CheckTrainingParams(context, numTokens) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "check training params failed."), return ge::GRAPH_FAILED);
    }

    // 3. platform info
    OP_TILING_CHECK(SetPlatformInfo(context, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set platform info failed."), return ge::GRAPH_FAILED);

    // 4. set tiling data
    OP_TILING_CHECK(SetTilingData(context, *tilingData, numTokens, isTraining) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set tiling data failed."), return ge::GRAPH_FAILED);

    // 5. set tiling key
    SetTilingKey(context, isTraining);

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
