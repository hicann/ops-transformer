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
 * \file engram_fetch_grad_tiling.cpp
 * \brief host侧tiling实现 — EngramFetchGrad
 *        校验 7 input + 3 output（dtype/shape/format）
 *        hiddenBytes 按 fp32 计算（反向 FP32 聚合）
 *        rankSize 从 sendCounts dim0 获取
 *        totalRecv 从 recvLocalEntry dim0 获取
 *        workspace: gradSorted + recvGrad + sdispls + rdispls
 *                 + counterScratch + flagScratch + 16MB
 */

#include <string>
#include <climits>
#include <cstdint>
#include <algorithm>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../../../op_kernel/engram_fetch_grad_tiling_data.h"
#include "../../../op_kernel/engram_fetch_grad_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace Mc2Tiling {
constexpr uint32_t IN_COMM_CONTEXT = 0U;
constexpr uint32_t IN_GRAD_FETCHED = 1U;
constexpr uint32_t IN_PERM = 2U;
constexpr uint32_t IN_SEND_COUNTS = 3U;
constexpr uint32_t IN_RECV_COUNTS = 4U;
constexpr uint32_t IN_RECV_LOCAL_ENTRY = 5U;
constexpr uint32_t IN_NUM_RECV = 6U;
constexpr uint32_t OUT_GRAD_UNIQUE = 0U;
constexpr uint32_t OUT_UNIQUE_LOCAL_ENTRY = 1U;
constexpr uint32_t OUT_NUM_UNIQUE = 2U;

constexpr uint32_t ATTR_NUM_ENTRIES_PER_RANK = 0U;
constexpr uint32_t ATTR_COMM_BUFFER_SIZE = 1U;

constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr int64_t SIMT_DCACHE_SIZE = 64 * 1024LL;

constexpr int64_t BUFFER_ALIGNMENT = 2 * 1024 * 1024;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2U;
constexpr uint32_t HALF_DTYPE_BYTES = 2U;
constexpr uint32_t DUAL_SEG_NUM = 2U;
constexpr uint32_t SCHEDULE_MODE_BATCH = 1U;

static const std::vector<ge::DataType> GRAD_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

static bool IsContains(const std::vector<ge::DataType>& list, ge::DataType value)
{
    return std::find(list.begin(), list.end(), value) != list.end();
}

static int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

static int64_t AlignTo(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

static void PrintEngramFetchGradTilingData(const EngramFetchGradTilingData* tilingData, const char* nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);

    OP_LOGD(nodeName, "========== EngramFetchGradTilingData ==========");
    OP_LOGD(nodeName, "numTokens is %lld", tilingData->numTokens);
    OP_LOGD(nodeName, "numEntriesPerRank is %d", tilingData->numEntriesPerRank);
    OP_LOGD(nodeName, "hiddenDim is %lld", tilingData->hiddenDim);
    OP_LOGD(nodeName, "hiddenBytes is %lld", tilingData->hiddenBytes);
    OP_LOGD(nodeName, "aivNum is %u", tilingData->aivNum);
    OP_LOGD(nodeName, "rankSize is %u", tilingData->rankSize);
    OP_LOGD(nodeName, "totalRecv is %lld", tilingData->totalRecv);
    OP_LOGD(nodeName, "commBufferSize is %lld", tilingData->commBufferSize);
    OP_LOGD(nodeName, "inputDtype is %d", tilingData->inputDtype);
    OP_LOGD(nodeName, "outputDtype is %d", tilingData->outputDtype);
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext* context)
{
    auto commContextDesc = context->GetInputDesc(IN_COMM_CONTEXT);
    auto gradDesc = context->GetInputDesc(IN_GRAD_FETCHED);
    auto permDesc = context->GetInputDesc(IN_PERM);
    auto sendCountsDesc = context->GetInputDesc(IN_SEND_COUNTS);
    auto recvCountsDesc = context->GetInputDesc(IN_RECV_COUNTS);
    auto recvLocalEntryDesc = context->GetInputDesc(IN_RECV_LOCAL_ENTRY);
    auto numRecvDesc = context->GetInputDesc(IN_NUM_RECV);
    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    auto uniqueLocalEntryDesc = context->GetOutputDesc(OUT_UNIQUE_LOCAL_ENTRY);
    auto numUniqueDesc = context->GetOutputDesc(OUT_NUM_UNIQUE);

    OP_CHECK_NULL_WITH_CONTEXT(context, commContextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, permDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, sendCountsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvCountsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvLocalEntryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradUniqueDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, uniqueLocalEntryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, numUniqueDesc);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDataType(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    auto commContextDesc = context->GetInputDesc(IN_COMM_CONTEXT);
    OP_TILING_CHECK(commContextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "commContext",
                                                          Ops::Base::ToString(commContextDesc->GetDataType()).c_str(),
                                                          "The dtype of commContext must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    auto gradDesc = context->GetInputDesc(IN_GRAD_FETCHED);
    OP_TILING_CHECK(!IsContains(GRAD_DTYPE_LIST, gradDesc->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "gradFetched", Ops::Base::ToString(gradDesc->GetDataType()).c_str(),
                        "The dtype of gradFetched must be DT_BF16, DT_FLOAT16 or DT_FLOAT."),
                    return ge::GRAPH_FAILED);

    auto permDesc = context->GetInputDesc(IN_PERM);
    OP_TILING_CHECK(
        permDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "perm", Ops::Base::ToString(permDesc->GetDataType()).c_str(),
                                              "The dtype of perm must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto sendCountsDesc = context->GetInputDesc(IN_SEND_COUNTS);
    OP_TILING_CHECK(sendCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "sendCounts",
                                                          Ops::Base::ToString(sendCountsDesc->GetDataType()).c_str(),
                                                          "The dtype of sendCounts must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    auto recvCountsDesc = context->GetInputDesc(IN_RECV_COUNTS);
    OP_TILING_CHECK(recvCountsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "recvCounts",
                                                          Ops::Base::ToString(recvCountsDesc->GetDataType()).c_str(),
                                                          "The dtype of recvCounts must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    auto recvLocalEntryDesc = context->GetInputDesc(IN_RECV_LOCAL_ENTRY);
    OP_TILING_CHECK(recvLocalEntryDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "recvLocalEntry", Ops::Base::ToString(recvLocalEntryDesc->GetDataType()).c_str(),
                        "The dtype of recvLocalEntry must be DT_INT32."),
                    return ge::GRAPH_FAILED);
    auto numRecvDesc = context->GetInputDesc(IN_NUM_RECV);
    OP_TILING_CHECK(numRecvDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "numRecv",
                                                          Ops::Base::ToString(numRecvDesc->GetDataType()).c_str(),
                                                          "The dtype of numRecv must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    OP_TILING_CHECK(!IsContains(GRAD_DTYPE_LIST, gradUniqueDesc->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "gradUniqueOut", Ops::Base::ToString(gradUniqueDesc->GetDataType()).c_str(),
                        "The dtype of gradUniqueOut must be DT_BF16, DT_FLOAT16 or DT_FLOAT."),
                    return ge::GRAPH_FAILED);

    auto uniqueLocalEntryDesc = context->GetOutputDesc(OUT_UNIQUE_LOCAL_ENTRY);
    OP_TILING_CHECK(
        uniqueLocalEntryDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "uniqueLocalEntryOut",
                                              Ops::Base::ToString(uniqueLocalEntryDesc->GetDataType()).c_str(),
                                              "The dtype of uniqueLocalEntryOut must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto numUniqueDesc = context->GetOutputDesc(OUT_NUM_UNIQUE);
    OP_TILING_CHECK(numUniqueDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "numUniqueOut",
                                                          Ops::Base::ToString(numUniqueDesc->GetDataType()).c_str(),
                                                          "The dtype of numUniqueOut must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDim(const gert::TilingContext* context, int64_t& numTokens, uint32_t& rankSize,
                                      int64_t& totalRecv, int64_t& hiddenDim)
{
    const char* nodeName = context->GetNodeName();

    const gert::StorageShape* commContextShape = context->GetInputShape(IN_COMM_CONTEXT);
    OP_TILING_CHECK(commContextShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        commContextShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "commContext", (std::to_string(commContextShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of commContext must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        commContextShape->GetStorageShape().GetDim(0) <= 0,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "commContext",
            (std::string("dim0=") + std::to_string(commContextShape->GetStorageShape().GetDim(0))).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    // gradFetched: 2D (T, H)
    const gert::StorageShape* gradShape = context->GetInputShape(IN_GRAD_FETCHED);
    OP_TILING_CHECK(gradShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "gradFetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        gradShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "gradFetched", (std::to_string(gradShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of gradFetched must be 2D."),
        return ge::GRAPH_FAILED);
    numTokens = gradShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numTokens < 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "gradFetched",
                                              (std::string("dim0=") + std::to_string(numTokens)).c_str(), ">= 0"),
                    return ge::GRAPH_FAILED);
    hiddenDim = gradShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(hiddenDim <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "gradFetched",
                                              (std::string("dim1=") + std::to_string(hiddenDim)).c_str(), "> 0"),
                    return ge::GRAPH_FAILED);

    // perm: 1D (T,), dim0 == numTokens
    const gert::StorageShape* permShape = context->GetInputShape(IN_PERM);
    OP_TILING_CHECK(permShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "perm"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(permShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "perm", (std::to_string(permShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of perm must be 1D."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        permShape->GetStorageShape().GetDim(0) != numTokens,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "perm", (std::string("dim0=") + std::to_string(permShape->GetStorageShape().GetDim(0))).c_str(),
            (std::string("dim0 must equal gradFetched dim0=") + std::to_string(numTokens)).c_str()),
        return ge::GRAPH_FAILED);

    // sendCounts: 1D (W,)
    const gert::StorageShape* sendCountsShape = context->GetInputShape(IN_SEND_COUNTS);
    OP_TILING_CHECK(sendCountsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCounts"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sendCountsShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "sendCounts", (std::to_string(sendCountsShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of sendCounts must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sendCountsShape->GetStorageShape().GetDim(0) <= 0,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "sendCounts",
            (std::string("dim0=") + std::to_string(sendCountsShape->GetStorageShape().GetDim(0))).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    constexpr int64_t SEND_COUNTS_ALIGN_FACTOR = 8;
    int64_t sendCountsDim0 = sendCountsShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(sendCountsDim0 % SEND_COUNTS_ALIGN_FACTOR != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "sendCounts.dim0", std::to_string(sendCountsDim0).c_str(),
                                              "must be multiple of SEND_COUNTS_ALIGN_FACTOR(8)"),
                    return ge::GRAPH_FAILED);
    rankSize = static_cast<uint32_t>(sendCountsDim0 / SEND_COUNTS_ALIGN_FACTOR);
    // 上界校验：与 Mc2Kernel::MAX_QP_SIZE（tiling_data.h）同源，同时保证 Kernel 侧 displs 批量写
    // blockLen(numRanks*32) 不超过 uint16 上限 65535、UB 常驻区可控
    OP_TILING_CHECK(rankSize > Mc2Kernel::MAX_QP_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "rankSize", std::to_string(rankSize).c_str(),
                                              (std::string("<= ") + std::to_string(Mc2Kernel::MAX_QP_SIZE)).c_str()),
                    return ge::GRAPH_FAILED);

    // recvCounts: 1D (W,)
    const gert::StorageShape* recvCountsShape = context->GetInputShape(IN_RECV_COUNTS);
    OP_TILING_CHECK(recvCountsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvCounts"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvCountsShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "recvCounts", (std::to_string(recvCountsShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of recvCounts must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvCountsShape->GetStorageShape().GetDim(0) <= 0,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "recvCounts",
            (std::string("dim0=") + std::to_string(recvCountsShape->GetStorageShape().GetDim(0))).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    // recvLocalEntry: 1D (R,)
    const gert::StorageShape* recvLocalEntryShape = context->GetInputShape(IN_RECV_LOCAL_ENTRY);
    OP_TILING_CHECK(recvLocalEntryShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvLocalEntry"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvLocalEntryShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "recvLocalEntry",
                        (std::to_string(recvLocalEntryShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of recvLocalEntry must be 1D."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvLocalEntryShape->GetStorageShape().GetDim(0) < 0,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "recvLocalEntry",
            (std::string("dim0=") + std::to_string(recvLocalEntryShape->GetStorageShape().GetDim(0))).c_str(), ">= 0"),
        return ge::GRAPH_FAILED);
    totalRecv = recvLocalEntryShape->GetStorageShape().GetDim(0);

    // numRecv: 1D, dim0 == 1
    const gert::StorageShape* numRecvShape = context->GetInputShape(IN_NUM_RECV);
    OP_TILING_CHECK(numRecvShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numRecv"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        numRecvShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "numRecv", (std::to_string(numRecvShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of numRecv must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numRecvShape->GetStorageShape().GetDim(0) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "numRecv",
                        (std::string("dim0=") + std::to_string(numRecvShape->GetStorageShape().GetDim(0))).c_str(),
                        "dim0 must be 1."),
                    return ge::GRAPH_FAILED);

    // gradUniqueOut: 2D
    const gert::StorageShape* gradUniqueShape = context->GetOutputShape(OUT_GRAD_UNIQUE);
    OP_TILING_CHECK(gradUniqueShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "gradUniqueOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        gradUniqueShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "gradUniqueOut", (std::to_string(gradUniqueShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of gradUniqueOut must be 2D."),
        return ge::GRAPH_FAILED);

    // uniqueLocalEntryOut: 1D
    const gert::StorageShape* uniqueLocalEntryShape = context->GetOutputShape(OUT_UNIQUE_LOCAL_ENTRY);
    OP_TILING_CHECK(uniqueLocalEntryShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "uniqueLocalEntryOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(uniqueLocalEntryShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "uniqueLocalEntryOut",
                        (std::to_string(uniqueLocalEntryShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of uniqueLocalEntryOut must be 1D."),
                    return ge::GRAPH_FAILED);

    // numUniqueOut: 1D, dim0 == 1
    const gert::StorageShape* numUniqueShape = context->GetOutputShape(OUT_NUM_UNIQUE);
    OP_TILING_CHECK(numUniqueShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numUniqueOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        numUniqueShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "numUniqueOut", (std::to_string(numUniqueShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of numUniqueOut must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numUniqueShape->GetStorageShape().GetDim(0) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "numUniqueOut",
                        (std::string("dim0=") + std::to_string(numUniqueShape->GetStorageShape().GetDim(0))).c_str(),
                        "dim0 must be 1."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorFormat(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    auto commContextDesc = context->GetInputDesc(IN_COMM_CONTEXT);
    ge::Format commContextFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(commContextDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        commContextFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "commContext", Ops::Base::ToString(commContextFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto gradFetchedDesc = context->GetInputDesc(IN_GRAD_FETCHED);
    ge::Format gradFetchedFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(gradFetchedDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        gradFetchedFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "gradFetched", Ops::Base::ToString(gradFetchedFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto permDesc = context->GetInputDesc(IN_PERM);
    ge::Format permFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(permDesc->GetStorageFormat()));
    OP_TILING_CHECK(permFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "perm", Ops::Base::ToString(permFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto sendCountsDesc = context->GetInputDesc(IN_SEND_COUNTS);
    ge::Format sendCountsFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(sendCountsDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        sendCountsFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "sendCounts", Ops::Base::ToString(sendCountsFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto recvCountsDesc = context->GetInputDesc(IN_RECV_COUNTS);
    ge::Format recvCountsFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(recvCountsDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        recvCountsFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "recvCounts", Ops::Base::ToString(recvCountsFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto recvLocalEntryDesc = context->GetInputDesc(IN_RECV_LOCAL_ENTRY);
    ge::Format recvLocalEntryFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(recvLocalEntryDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        recvLocalEntryFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "recvLocalEntry", Ops::Base::ToString(recvLocalEntryFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);
    auto numRecvDesc = context->GetInputDesc(IN_NUM_RECV);
    ge::Format numRecvFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(numRecvDesc->GetStorageFormat()));
    OP_TILING_CHECK(numRecvFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "numRecv", Ops::Base::ToString(numRecvFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    ge::Format gradUniqueFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(gradUniqueDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        gradUniqueFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "gradUniqueOut", Ops::Base::ToString(gradUniqueFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto uniqueLocalEntryDesc = context->GetOutputDesc(OUT_UNIQUE_LOCAL_ENTRY);
    ge::Format uniqueLocalEntryFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(uniqueLocalEntryDesc->GetStorageFormat()));
    OP_TILING_CHECK(uniqueLocalEntryFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "uniqueLocalEntryOut",
                                               Ops::Base::ToString(uniqueLocalEntryFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto numUniqueDesc = context->GetOutputDesc(OUT_NUM_UNIQUE);
    ge::Format numUniqueFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(numUniqueDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        numUniqueFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "numUniqueOut", Ops::Base::ToString(numUniqueFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckEngramFetchGrad(const gert::TilingContext* context, int64_t& numTokens,
                                                  uint32_t& rankSize, int64_t& totalRecv, int64_t& hiddenDim)
{
    const char* nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckTensorPtrNullptr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDataType(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckTensorDim(context, numTokens, rankSize, totalRecv, hiddenDim) != ge::GRAPH_SUCCESS,
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

static ge::graphStatus CheckAttrs(const gert::TilingContext* context)
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

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetPlatformInfo(gert::TilingContext* context, EngramFetchGradTilingData& tilingData)
{
    const char* nodeName = context->GetNodeName();

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    // 平台信息异常时拦在 tiling 源头: a2a 死锁硬约束要求至少 2 个 v 核
    // (1 send + 1 recv 并发, credit 背压依赖对端 receiver 在场)
    OP_TILING_CHECK(aivNum < 2U,
                    OP_LOGE(nodeName, "EngramFetchGrad requires at least 2 AIV cores, got aivNum=%u", aivNum),
                    return ge::GRAPH_FAILED);
    uint32_t numBlocks = ascendcPlatform.CalcTschNumBlocks(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.aivNum = aivNum;
    OP_LOGD(nodeName, "aicNum=%u, aivNum=%u, numBlocks=%u", aicNum, aivNum, numBlocks);

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
    OP_LOGD(nodeName, "SetLocalMemorySize: ubSize=%llu, availUbSize=%lld, SIMT_DCACHE_SIZE=%lld", ubSize, availUbSize,
            SIMT_DCACHE_SIZE);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(const gert::TilingContext* context, EngramFetchGradTilingData& tilingData,
                                     int64_t numTokens, uint32_t rankSize, int64_t totalRecv, int64_t hiddenDim)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    // 布局常量已收敛至 tiling_data.h（Mc2Kernel 单一权威定义），此处直接引用、不再镜像；
    // GRAD_SUB_BATCH_CAP 为 Host 侧预算上界（Kernel 旧 tiling 兼容回落值 8 与此相互独立）
    constexpr uint32_t GRAD_SUB_BATCH_CAP = 64U;

    tilingData.numTokens = numTokens;
    tilingData.rankSize = rankSize;
    tilingData.totalRecv = totalRecv;
    tilingData.hiddenDim = hiddenDim;

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK);
    tilingData.numEntriesPerRank = static_cast<int32_t>(*numEntriesPerRankPtr);
    // entry id 空间为 int32（rankId*numEntriesPerRank + 局部序号），乘积不得越界（SEC-2.1）
    OP_TILING_CHECK(tilingData.numEntriesPerRank < 0 ||
                        (rankSize > 0 && static_cast<int64_t>(rankSize) * tilingData.numEntriesPerRank > INT32_MAX),
                    OP_LOGE(nodeName, "numEntriesPerRank(%d) * rankSize(%u) exceeds INT32_MAX",
                            tilingData.numEntriesPerRank, rankSize),
                    return ge::GRAPH_FAILED);

    // hiddenBytes 按输入 dtype（gradFetched）算，作为 a2a 交换 stride
    auto gradFetchedDesc = context->GetInputDesc(IN_GRAD_FETCHED);
    ge::DataType inputDtype = gradFetchedDesc->GetDataType();
    int64_t bytesPerElem = ge::GetSizeByDataType(inputDtype);
    OP_TILING_CHECK(tilingData.hiddenDim > INT64_MAX / bytesPerElem,
                    OP_LOGE(nodeName, "hiddenBytes overflow: hiddenDim=%lld * bytesPerElem=%lld exceeds INT64_MAX",
                            tilingData.hiddenDim, bytesPerElem),
                    return ge::GRAPH_FAILED);
    tilingData.hiddenBytes = tilingData.hiddenDim * bytesPerElem;
    bool fullRowFits = tilingData.hiddenBytes <= static_cast<int64_t>(Mc2Kernel::GRAD_BUF_BYTES);
    tilingData.inputDtype = static_cast<int32_t>(inputDtype);

    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    tilingData.outputDtype = static_cast<int32_t>(gradUniqueDesc->GetDataType());
    if (tilingData.outputDtype != static_cast<int32_t>(ge::DT_FLOAT)) {
        int64_t flushCastNeed =
            Mc2Kernel::FLUSH_CAST_HEAD_BYTES +
            DOUBLE_BUFFER_NUM * AlignTo(static_cast<int64_t>(hiddenDim) * HALF_DTYPE_BYTES, Mc2Kernel::UB_ALIGN);
        if (flushCastNeed > Mc2Kernel::ENTRY_BUF_BYTES) {
            fullRowFits = false;
        }
    }

    auto commBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_BUFFER_SIZE);
    tilingData.commBufferSize = *commBufferSizePtr;

    uint32_t statusBytes =
        static_cast<uint32_t>(AlignTo(static_cast<int64_t>(rankSize) * Mc2Kernel::STATE_OFFSET, Mc2Kernel::UB_ALIGN));
    uint32_t tempBytes = statusBytes;
    if (tempBytes < (Mc2Kernel::ENTRY_BATCH_CAP * sizeof(int32_t))) {
        tempBytes = (Mc2Kernel::ENTRY_BATCH_CAP * sizeof(int32_t));
    }
    // displs 批量构造区：sdispl+rdispl 两段各 rankSize*32B
    uint32_t displsBatchBytes = DUAL_SEG_NUM * rankSize * Mc2Kernel::STATE_OFFSET;
    if (tempBytes < displsBatchBytes) {
        tempBytes = displsBatchBytes;
    }
    uint32_t coreArrayBytes = static_cast<uint32_t>(
        AlignTo(static_cast<int64_t>(tilingData.aivNum) * sizeof(int32_t) * DUAL_SEG_NUM, Mc2Kernel::UB_ALIGN));
    if (tempBytes < coreArrayBytes) {
        tempBytes = coreArrayBytes;
    }
    uint32_t indicesBytes = (Mc2Kernel::IDX_BUF_BYTES > statusBytes) ? Mc2Kernel::IDX_BUF_BYTES : statusBytes;
    // countsBuf 暂存区：recvCounts + sendCounts（偏移 rankSize*32B + 长度 rankSize*32B）
    uint32_t stagingBytes = DUAL_SEG_NUM * rankSize * Mc2Kernel::STATE_OFFSET;
    if (indicesBytes < stagingBytes) {
        indicesBytes = stagingBytes;
    }
    uint64_t permanentUb = Mc2Kernel::HCOMM_INIT_SIZE + statusBytes + tempBytes + indicesBytes;
    OP_TILING_CHECK(permanentUb >= static_cast<uint64_t>(tilingData.ubSize),
                    OP_LOGE(nodeName, "permanent UB (%llu) exceeds availUbSize (%llu)", permanentUb, tilingData.ubSize),
                    return ge::GRAPH_FAILED);
    // UB 侧按 32B 对齐行 stride 排布 gradRaw/cast 行，容量计算必须与 kernel 侧一致（含非对齐 hiddenBytes）
    uint32_t inRowStride = static_cast<uint32_t>(
        AlignTo(static_cast<int64_t>(tilingData.hiddenBytes), static_cast<int64_t>(Mc2Kernel::UB_ALIGN)));
    uint32_t maxByPong = Mc2Kernel::GRAD_PING_BYTES / inRowStride;
    if (maxByPong < 1U) {
        maxByPong = 1U;
    }
    uint32_t accumNeed = Mc2Kernel::ACCUM_BUF_COPIES *
                         static_cast<uint32_t>(AlignTo(static_cast<int64_t>(tilingData.hiddenDim) * sizeof(float),
                                                       static_cast<int64_t>(Mc2Kernel::UB_ALIGN)));
    bool needCast = (tilingData.inputDtype != static_cast<int32_t>(ge::DT_FLOAT));
    uint32_t availableForPool = static_cast<uint32_t>(tilingData.ubSize) - static_cast<uint32_t>(permanentUb);
    uint32_t uniqueEntryBytes = Mc2Kernel::FLUSH_CAST_HEAD_BYTES;
    if (tilingData.outputDtype != static_cast<int32_t>(ge::DT_FLOAT)) {
        uniqueEntryBytes +=
            DOUBLE_BUFFER_NUM * static_cast<uint32_t>(AlignTo(hiddenDim * HALF_DTYPE_BYTES, Mc2Kernel::UB_ALIGN));
    }
    // cast 缓冲行 stride 同样 32B 对齐（fp32 行）
    uint32_t fp32RowStride = static_cast<uint32_t>(
        AlignTo(static_cast<int64_t>(tilingData.hiddenDim) * sizeof(float), static_cast<int64_t>(Mc2Kernel::UB_ALIGN)));
    uint32_t minUniqueNeed =
        Mc2Kernel::GRAD_BUF_BYTES + uniqueEntryBytes + accumNeed + (needCast ? DOUBLE_BUFFER_NUM * fp32RowStride : 0U);
    if (minUniqueNeed > availableForPool) {
        fullRowFits = false;
    }
    if (fullRowFits) {
        tilingData.chunkElems = 0U;
        uint32_t availableForCast = availableForPool - Mc2Kernel::GRAD_BUF_BYTES - uniqueEntryBytes - accumNeed;
        uint32_t maxByCast = needCast ? (availableForCast / (fp32RowStride * Mc2Kernel::ACCUM_BUF_COPIES)) : maxByPong;
        uint32_t gradSubBatch = maxByPong;
        if (maxByCast < gradSubBatch) {
            gradSubBatch = maxByCast;
        }
        if (gradSubBatch < 1U) {
            gradSubBatch = 1U;
        }
        if (gradSubBatch > GRAD_SUB_BATCH_CAP) {
            gradSubBatch = GRAD_SUB_BATCH_CAP;
        }
        tilingData.gradSubBatch = gradSubBatch;
    } else {
        tilingData.chunkElems = Mc2Kernel::HIDDEN_CHUNK_ELEMS;
        uint32_t chunkInStride =
            static_cast<uint32_t>(AlignTo(static_cast<int64_t>(Mc2Kernel::HIDDEN_CHUNK_ELEMS) * bytesPerElem,
                                          static_cast<int64_t>(Mc2Kernel::UB_ALIGN)));
        uint32_t chunkFp32Stride =
            static_cast<uint32_t>(AlignTo(static_cast<int64_t>(Mc2Kernel::HIDDEN_CHUNK_ELEMS) * sizeof(float),
                                          static_cast<int64_t>(Mc2Kernel::UB_ALIGN)));
        // flush cast staging 按 fp32 stride 上界预留(保守; 与 kernel 侧 LoadTilingData 同口径)
        uint32_t chunkEntryBytes = Mc2Kernel::FLUSH_CAST_HEAD_BYTES + DOUBLE_BUFFER_NUM * chunkFp32Stride;
        uint32_t chunkMinNeed =
            Mc2Kernel::GRAD_BUF_BYTES + chunkEntryBytes + chunkFp32Stride + DOUBLE_BUFFER_NUM * chunkFp32Stride;
        OP_TILING_CHECK(chunkMinNeed > availableForPool,
                        OP_LOGE(nodeName, "chunk-mode UB pool overflow: need %u, availableForPool %u", chunkMinNeed,
                                availableForPool),
                        return ge::GRAPH_FAILED);
        uint32_t chunkRows = Mc2Kernel::GRAD_BUF_BYTES / chunkInStride;
        if (chunkRows < 1U) {
            chunkRows = 1U;
        }
        tilingData.gradSubBatch = chunkRows;
    }

    uint32_t sortCoreCnt = tilingData.aivNum / 2U;
    if (sortCoreCnt < 1U) {
        sortCoreCnt = 1U;
    }
    tilingData.sortCoreCnt = sortCoreCnt;

    uint64_t sortTileCalibReserve =
        512U +
        ((static_cast<uint64_t>(tilingData.rankSize) * Mc2Kernel::STATE_OFFSET + Mc2Kernel::UB_ALIGN - 1U) /
         Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN) +
        2U * Mc2Kernel::A2A_COPY_HALF_BYTES;
    uint64_t budget = static_cast<uint64_t>(tilingData.ubSize) - sortTileCalibReserve;
    constexpr uint64_t kSortLibFixedBytes = 5120U;
    constexpr uint64_t kMinTileElems = 64U;
    constexpr uint64_t kBytesPerElem = 21U;
    constexpr uint32_t kSortTmpHeadBytes = 512U;
    constexpr uint32_t kSortTmpTailBytes = 256U;
    constexpr uint32_t kSortTmpBytesPerGroup = 7U;
    constexpr uint32_t kSortTmpGroupTiles = 32U;
    if (budget < kSortLibFixedBytes + kMinTileElems * kBytesPerElem) {
        OP_LOGE(nodeName, "SortLib UB budget too small: ubSize=%llu, calibReserve=%llu, need>=%llu",
                static_cast<uint64_t>(tilingData.ubSize), sortTileCalibReserve,
                static_cast<uint64_t>(kSortLibFixedBytes + kMinTileElems * kBytesPerElem));
        return ge::GRAPH_FAILED;
    }

    uint32_t tilesPerSortCore = 2U;
    uint32_t numTile =
        static_cast<uint32_t>((totalRecv + sortCoreCnt * tilesPerSortCore - 1) / (sortCoreCnt * tilesPerSortCore));
    uint32_t ubCap = static_cast<uint32_t>((budget - kSortLibFixedBytes) / kBytesPerElem);
    if (numTile < kMinTileElems) {
        numTile = kMinTileElems;
    }
    if (numTile > ubCap) {
        numTile = ubCap;
    }
    tilingData.sortNumTileData = numTile;
    tilingData.sortTileCount = static_cast<uint32_t>((totalRecv + numTile - 1) / numTile);
    uint32_t roundTile = (numTile + kSortTmpGroupTiles - 1U) / kSortTmpGroupTiles * kSortTmpGroupTiles;
    tilingData.sortTmpUbSize = kSortTmpHeadBytes + kSortTmpBytesPerGroup * roundTile + kSortTmpTailBytes;
    OP_LOGD(nodeName, "SortLib params: numTile=%u tileCount=%u tmpUb=%u budget=%llu", tilingData.sortNumTileData,
            tilingData.sortTileCount, tilingData.sortTmpUbSize, static_cast<uint64_t>(budget));
    OP_LOGD(nodeName, "gradSubBatch=%u chunkElems=%u (hiddenBytes=%lld, hiddenDim=%lld, maxByPong=%u)",
            tilingData.gradSubBatch, tilingData.chunkElems, tilingData.hiddenBytes, tilingData.hiddenDim, maxByPong);
    OP_LOGD(nodeName, "permanentUb=%llu (status=%u, temp=%u, indices=%u)", permanentUb, statusBytes, tempBytes,
            indicesBytes);

    OP_LOGD(nodeName,
            "SetTilingData: numTokens=%lld, hiddenDim=%lld, numEntriesPerRank=%d, hiddenBytes=%lld, "
            "ubSize=%llu, rankSize=%u, totalRecv=%lld, commBufferSize=%lld",
            tilingData.numTokens, tilingData.hiddenDim, tilingData.numEntriesPerRank, tilingData.hiddenBytes,
            tilingData.ubSize, tilingData.rankSize, tilingData.totalRecv, tilingData.commBufferSize);
    return ge::GRAPH_SUCCESS;
}

static void SetTilingKey(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    const uint64_t tilingKey = GET_TPL_TILING_KEY(ENGRAM_FETCH_GRAD_DEFAULT_MODE);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%llu] in engram_fetch_grad.", tilingKey);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext* context, const EngramFetchGradTilingData& tilingData)
{
    const char* nodeName = context->GetNodeName();
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    int64_t numRanks = static_cast<int64_t>(tilingData.rankSize);
    int64_t numTokens = tilingData.numTokens;
    int64_t hiddenBytes = tilingData.hiddenBytes;
    int64_t totalRecv = tilingData.totalRecv;

    OP_TILING_CHECK(numTokens > 0 && hiddenBytes > INT64_MAX / numTokens,
                    OP_LOGE(nodeName, "workspace overflow: numTokens=%lld, hiddenBytes=%lld", numTokens, hiddenBytes),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(totalRecv > 0 && hiddenBytes > INT64_MAX / totalRecv,
                    OP_LOGE(nodeName, "workspace overflow: totalRecv=%lld, hiddenBytes=%lld", totalRecv, hiddenBytes),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        totalRecv > 0 && tilingData.hiddenDim > INT64_MAX / totalRecv,
        OP_LOGE(nodeName, "workspace overflow: totalRecv=%lld, hiddenDim=%lld", totalRecv, tilingData.hiddenDim),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        totalRecv > 0 && tilingData.hiddenDim > (INT64_MAX / static_cast<int64_t>(sizeof(float))) / totalRecv,
        OP_LOGE(nodeName, "workspace overflow: totalRecv=%lld, hiddenDim=%lld", totalRecv, tilingData.hiddenDim),
        return ge::GRAPH_FAILED);

    int64_t wsGradSorted = numTokens * hiddenBytes;
    int64_t wsRecvGrad = totalRecv * hiddenBytes;
    int64_t wsSdispls = numRanks * Mc2Kernel::UB_ALIGN;
    int64_t wsRdispls = numRanks * Mc2Kernel::UB_ALIGN;
    int64_t wsCounterScratch = static_cast<int64_t>(tilingData.aivNum) * Mc2Kernel::UB_ALIGN;
    constexpr int64_t kFlagScratchBytes = 32;
    int64_t wsFlagScratch = kFlagScratchBytes;
    int64_t coreArrayAligned = AlignTo(static_cast<int64_t>(tilingData.aivNum) * sizeof(int32_t), Mc2Kernel::UB_ALIGN);
    int64_t wsSegCount = coreArrayAligned;
    int64_t wsCoreStart = coreArrayAligned;

    OP_TILING_CHECK(totalRecv > INT64_MAX / static_cast<int64_t>(sizeof(int32_t)),
                    OP_LOGE(nodeName, "sort companion overflow: totalRecv=%lld", totalRecv), return ge::GRAPH_FAILED);
    int64_t wsSortCompanion = (totalRecv * static_cast<int64_t>(sizeof(int32_t)) + Mc2Kernel::UB_ALIGN - 1) /
                              Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    constexpr int64_t kRadixRounds = 4;
    constexpr int64_t kRadixBinNum = 256;
    constexpr int64_t kRadixCountBytes = 4;
    constexpr int64_t kRadixHistElemBytes = 2;
    constexpr int64_t kRadixHistArrayNum = 2;
    int64_t slSeg0 = AlignTo(kRadixBinNum * kRadixRounds * kRadixCountBytes, Mc2Kernel::UB_ALIGN);
    int64_t slSeg1 =
        AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * kRadixBinNum * kRadixRounds * kRadixCountBytes,
                Mc2Kernel::UB_ALIGN);
    int64_t slSeg2 = AlignTo(totalRecv * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);
    int64_t slSeg3 = kRadixHistArrayNum *
                     AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * kRadixBinNum * kRadixHistElemBytes,
                             Mc2Kernel::UB_ALIGN);
    int64_t slSeg4 =
        AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * tilingData.sortNumTileData, Mc2Kernel::UB_ALIGN);
    int64_t slSeg5 = AlignTo(totalRecv * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);

    constexpr int64_t SOFT_BARRIER_MAX_GENS = 16;
    int64_t wsSortBarrier = SOFT_BARRIER_MAX_GENS *
                            ((static_cast<int64_t>(tilingData.aivNum) * static_cast<int64_t>(Mc2Kernel::STATE_OFFSET) +
                              Mc2Kernel::UB_ALIGN - 1) /
                             Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN);
    int64_t wsSortTotal = wsSortCompanion + slSeg0 + slSeg1 + slSeg2 + slSeg3 + slSeg4 + slSeg5 + wsSortBarrier;

    int64_t wsTotal = wsGradSorted + wsRecvGrad + wsSdispls + wsRdispls + wsCounterScratch + wsFlagScratch +
                      wsSegCount + wsCoreStart + wsSortTotal;
    wsTotal = AlignTo(wsTotal, BUFFER_ALIGNMENT);
    wsTotal += SYSTEM_NEED_WORKSPACE;

    workSpaces[0] = static_cast<size_t>(wsTotal);
    OP_LOGD(nodeName,
            "backward workspace: gradSorted=%lld, recvGrad=%lld, sdispls=%lld, rdispls=%lld, "
            "counterScratch=%lld, flagScratch=%lld, total=%zu",
            wsGradSorted, wsRecvGrad, wsSdispls, wsRdispls, wsCounterScratch, wsFlagScratch, workSpaces[0]);
    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief host侧tiling实现 — EngramFetchGrad 分轨模型（rail-aligned，204800 适配）
 *        - 由老 tiling 入口在 world > ranksPerSupernode 且平铺 rail ctx 时转调（老函数体零改动）
 *        - 感知只取一数（ranksPerSupernode，attr 透传），超节点数全推导（kernel 与 ctx 头部双源校验）
 *        - 校验：world%B==0；ranksPerSupernode ≤ 1024、超节点数 ≤ 2048（ctx node 表容量）；
 *          numEntriesPerRank×world ≤ INT32_MAX；N ≥ 2×批行数（信用窗口有效）
 *        - workspace 新布局；UB 常驻 O(1)（与 world 无关）
 */

// 分轨段：由老 tiling 入口 EngramFetchGradTilingFunc 转调（同文件下方）。

// 感知 attr（host HCCL 感知自动透传；索引与 def 声明顺序一致）
constexpr uint32_t ATTR_RANKS_PER_SUPERNODE = 2U; // 每超节点卡数（非用户输入）

constexpr uint32_t SEND_COUNTS_ALIGN_FACTOR = 8U;
constexpr uint32_t GRAD_SUB_BATCH_CAP = 64U;
constexpr uint32_t MIN_AIV_NUM = 4U; // 分轨硬要求：S/F/O 三组各 ≥1

// ═══ 集群规模运行期推导（与 kernel 侧 DeriveWinLayout 同公式，零字面量） ═══
// 分轨模型（host 侧权威；与 kernel RailTopo 同公式，见 route.h MakeRailTopo）
struct RailTopology {
    uint32_t ranksPerSupernode{0}; // 每超节点卡数（感知 attr）
    uint32_t supernodeCnt{0};      // 超节点数 = world/ranksPerSupernode（推导，含自己）
    uint32_t railCnt{0};           // 轨道对端数 = world/ranksPerSupernode − 1（推导）
    uint64_t winHead{0}; // (W+1)×32；W = ranksPerSupernode + world/ranksPerSupernode（写者槽 + epoch 槽）
    uint64_t areas{0};   // W（rail: world/ranksPerSupernode 格 + owner: ranksPerSupernode 格）
};

/// @brief 集群规模运行期推导：由感知数与 world 推导超节点数/轨道对端数/写者槽 W
/// @return RailTopology（与 kernel 侧同公式，零字面量）
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

// ─────────────────────── 校验 ───────────────────────

/// @brief 分轨段输入/输出 dtype/format 校验
static ge::graphStatus CheckRailTensorDataType(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto checkInt32 = [&](uint32_t idx, const char* name) {
        auto desc = context->GetInputDesc(idx);
        OP_TILING_CHECK(desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, name), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            desc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, name, Ops::Base::ToString(desc->GetDataType()).c_str(),
                                                  "The dtype must be DT_INT32."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat())) != ge::FORMAT_ND,
            OP_LOGE_FOR_INVALID_FORMAT(
                nodeName, name, Ops::Base::ToString(static_cast<ge::Format>(desc->GetStorageFormat())).c_str(), "ND"),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    // commContext/perm/sendCounts/recvCounts/recvLocalEntry/numRecv 全部 int32 ND
    const char* int32Inputs[] = {"commContext", "perm", "sendCounts", "recvCounts", "recvLocalEntry", "numRecv"};
    const uint32_t int32Idx[] = {IN_COMM_CONTEXT,     IN_PERM,    IN_SEND_COUNTS, IN_RECV_COUNTS,
                                 IN_RECV_LOCAL_ENTRY, IN_NUM_RECV};
    for (size_t i = 0; i < sizeof(int32Idx) / sizeof(int32Idx[0]); i++) {
        OP_TILING_CHECK(checkInt32(int32Idx[i], int32Inputs[i]) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "check int32 input %s failed.", int32Inputs[i]), return ge::GRAPH_FAILED);
    }

    auto gradDesc = context->GetInputDesc(IN_GRAD_FETCHED);
    OP_TILING_CHECK(gradDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "gradFetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(GRAD_DTYPE_LIST, gradDesc->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "gradFetched", Ops::Base::ToString(gradDesc->GetDataType()).c_str(),
                        "The dtype of gradFetched must be DT_BF16, DT_FLOAT16 or DT_FLOAT."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(gradDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(
                        nodeName, "gradFetched",
                        Ops::Base::ToString(static_cast<ge::Format>(gradDesc->GetStorageFormat())).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    OP_TILING_CHECK(gradUniqueDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "gradUniqueOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(GRAD_DTYPE_LIST, gradUniqueDesc->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "gradUniqueOut", Ops::Base::ToString(gradUniqueDesc->GetDataType()).c_str(),
                        "The dtype of gradUniqueOut must be DT_BF16, DT_FLOAT16 or DT_FLOAT."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(gradUniqueDesc->GetStorageFormat())) != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(
                        nodeName, "gradUniqueOut",
                        Ops::Base::ToString(static_cast<ge::Format>(gradUniqueDesc->GetStorageFormat())).c_str(), "ND"),
                    return ge::GRAPH_FAILED);

    auto uniqueLocalEntryDesc = context->GetOutputDesc(OUT_UNIQUE_LOCAL_ENTRY);
    OP_TILING_CHECK(uniqueLocalEntryDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "uniqueLocalEntryOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        uniqueLocalEntryDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "uniqueLocalEntryOut",
                                              Ops::Base::ToString(uniqueLocalEntryDesc->GetDataType()).c_str(),
                                              "The dtype of uniqueLocalEntryOut must be DT_INT32."),
        return ge::GRAPH_FAILED);
    auto numUniqueDesc = context->GetOutputDesc(OUT_NUM_UNIQUE);
    OP_TILING_CHECK(numUniqueDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numUniqueOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numUniqueDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "numUniqueOut",
                                                          Ops::Base::ToString(numUniqueDesc->GetDataType()).c_str(),
                                                          "The dtype of numUniqueOut must be DT_INT32."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/// @brief 分轨段 shape 校验并提取关键规模（numTokens/worldSize/totalRecv/hiddenDim）
static ge::graphStatus CheckRailTensorDim(const gert::TilingContext* context, int64_t& numTokens, uint32_t& worldSize,
                                          int64_t& totalRecv, int64_t& hiddenDim, int64_t& gradUniqueRows)
{
    const char* nodeName = context->GetNodeName();

    const gert::StorageShape* commContextShape = context->GetInputShape(IN_COMM_CONTEXT);
    OP_TILING_CHECK(commContextShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        commContextShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "commContext", (std::to_string(commContextShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of commContext must be 1D."),
        return ge::GRAPH_FAILED);
    // 分轨硬校验：平铺 rail ctx（头部标量 + node 表两张，CTX_RAIL_ELEMS int32）——分发入口已查，此处兜底
    int64_t ctxElems = commContextShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(ctxElems != Mc2Kernel::CTX_RAIL_ELEMS,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "commContext", (std::string("dim0=") + std::to_string(ctxElems)).c_str(),
                        (std::string("== ") + std::to_string(Mc2Kernel::CTX_RAIL_ELEMS) + " (flat rail ctx)").c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* gradShape = context->GetInputShape(IN_GRAD_FETCHED);
    OP_TILING_CHECK(gradShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "gradFetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        gradShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "gradFetched", (std::to_string(gradShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of gradFetched must be 2D."),
        return ge::GRAPH_FAILED);
    numTokens = gradShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numTokens < 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "gradFetched",
                                              (std::string("dim0=") + std::to_string(numTokens)).c_str(), ">= 0"),
                    return ge::GRAPH_FAILED);
    hiddenDim = gradShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(hiddenDim <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "gradFetched",
                                              (std::string("dim1=") + std::to_string(hiddenDim)).c_str(), "> 0"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* permShape = context->GetInputShape(IN_PERM);
    OP_TILING_CHECK(permShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "perm"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        permShape->GetStorageShape().GetDimNum() != DIM_ONE || permShape->GetStorageShape().GetDim(0) != numTokens,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "perm", (std::string("dim0=") + std::to_string(permShape->GetStorageShape().GetDim(0))).c_str(),
            (std::string("dim0 must equal gradFetched dim0=") + std::to_string(numTokens)).c_str()),
        return ge::GRAPH_FAILED);

    // 分轨世界规则：world = sendCounts.dim0/8；分发判据 world > ranksPerSupernode 由
    // CheckRailAttrs 与感知 attr 双源校验（缩比床 world 可 ≤1024）
    const gert::StorageShape* sendCountsShape = context->GetInputShape(IN_SEND_COUNTS);
    OP_TILING_CHECK(sendCountsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCounts"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sendCountsShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "sendCounts", (std::to_string(sendCountsShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of sendCounts must be 1D."),
        return ge::GRAPH_FAILED);
    int64_t sendCountsDim0 = sendCountsShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(sendCountsDim0 <= 0 || sendCountsDim0 % SEND_COUNTS_ALIGN_FACTOR != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "sendCounts.dim0", std::to_string(sendCountsDim0).c_str(),
                                              "must be positive multiple of 8"),
                    return ge::GRAPH_FAILED);
    int64_t world = sendCountsDim0 / SEND_COUNTS_ALIGN_FACTOR;
    OP_TILING_CHECK(world <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "world", std::to_string(world).c_str(),
                                              "must be positive (send_counts.dim0/8)"),
                    return ge::GRAPH_FAILED);
    worldSize = static_cast<uint32_t>(world);

    const gert::StorageShape* recvCountsShape = context->GetInputShape(IN_RECV_COUNTS);
    OP_TILING_CHECK(recvCountsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvCounts"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsShape->GetStorageShape().GetDimNum() != DIM_ONE ||
                        recvCountsShape->GetStorageShape().GetDim(0) != world,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "recvCounts",
                        (std::string("dim0=") + std::to_string(recvCountsShape->GetStorageShape().GetDim(0))).c_str(),
                        (std::string("dim0 must equal world=") + std::to_string(world)).c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* recvLocalEntryShape = context->GetInputShape(IN_RECV_LOCAL_ENTRY);
    OP_TILING_CHECK(recvLocalEntryShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvLocalEntry"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvLocalEntryShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "recvLocalEntry",
                        (std::to_string(recvLocalEntryShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of recvLocalEntry must be 1D."),
                    return ge::GRAPH_FAILED);
    totalRecv = recvLocalEntryShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(totalRecv < 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "recvLocalEntry",
                                              (std::string("dim0=") + std::to_string(totalRecv)).c_str(), ">= 0"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* numRecvShape = context->GetInputShape(IN_NUM_RECV);
    OP_TILING_CHECK(numRecvShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numRecv"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        numRecvShape->GetStorageShape().GetDimNum() != DIM_ONE || numRecvShape->GetStorageShape().GetDim(0) != 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "numRecv",
            (std::string("dim0=") + std::to_string(numRecvShape->GetStorageShape().GetDim(0))).c_str(),
            "dim0 must be 1."),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* gradUniqueShape = context->GetOutputShape(OUT_GRAD_UNIQUE);
    OP_TILING_CHECK(gradUniqueShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "gradUniqueOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        gradUniqueShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "gradUniqueOut", (std::to_string(gradUniqueShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of gradUniqueOut must be 2D."),
        return ge::GRAPH_FAILED);
    gradUniqueRows = gradUniqueShape->GetStorageShape().GetDim(0);

    const gert::StorageShape* uniqueLocalEntryShape = context->GetOutputShape(OUT_UNIQUE_LOCAL_ENTRY);
    OP_TILING_CHECK(uniqueLocalEntryShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "uniqueLocalEntryOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(uniqueLocalEntryShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "uniqueLocalEntryOut",
                        (std::to_string(uniqueLocalEntryShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of uniqueLocalEntryOut must be 1D."),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape* numUniqueShape = context->GetOutputShape(OUT_NUM_UNIQUE);
    OP_TILING_CHECK(numUniqueShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numUniqueOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        numUniqueShape->GetStorageShape().GetDimNum() != DIM_ONE || numUniqueShape->GetStorageShape().GetDim(0) != 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "numUniqueOut",
            (std::string("dim0=") + std::to_string(numUniqueShape->GetStorageShape().GetDim(0))).c_str(),
            "dim0 must be 1."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/// @brief 分轨段 attr 校验并推导拓扑
/// @note ranks_per_supernode 须 (0, 1024] 且整除 world；超节点数须 <= 2048
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
        const gert::StorageShape* sendCountsShape = context->GetInputShape(IN_SEND_COUNTS);
        OP_TILING_CHECK(sendCountsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCounts"),
                        return ge::GRAPH_FAILED);
        worldFromShape = static_cast<uint32_t>(sendCountsShape->GetStorageShape().GetDim(0) / SEND_COUNTS_ALIGN_FACTOR);
    }
    OP_TILING_CHECK(static_cast<int64_t>(worldFromShape) % ranksPerSupernode != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "ranks_per_supernode", std::to_string(ranksPerSupernode).c_str(),
                        (std::string("world=") + std::to_string(worldFromShape) +
                         " is not a multiple of ranks_per_supernode (sensed topology inconsistent with "
                         "send_counts.dim0/8)")
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
    topoAttr = DeriveRailTopology(static_cast<uint32_t>(ranksPerSupernode), worldFromShape);
    return ge::GRAPH_SUCCESS;
}

// ─────────────────────── 平台信息 ───────────────────────

/// @brief 平台信息落表：AIV 核数、blockDim、UB 预算（分轨要求 AIV >= 4，三组各 >= 1）
static ge::graphStatus SetRailPlatformInfo(gert::TilingContext* context, EngramFetchGradTilingData& tilingData)
{
    const char* nodeName = context->GetNodeName();

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    // 分轨硬要求：三组并发（S/F/O 常驻消费），至少 MIN_AIV_NUM 个 v 核
    OP_TILING_CHECK(aivNum < MIN_AIV_NUM,
                    OP_LOGE(nodeName, "rail requires at least %u AIV cores, got aivNum=%u", MIN_AIV_NUM, aivNum),
                    return ge::GRAPH_FAILED);
    uint32_t numBlocks = ascendcPlatform.CalcTschNumBlocks(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.aivNum = aivNum;

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

/// @brief 分轨 tilingData 落表：拓扑、行距、批行数、N 反解、SortLib 参数、UB 预算校验
/// @note N 反解把 commBufferSize 分满；批行数按 65504B 预算（blockLen 为 uint16）
static ge::graphStatus SetRailTilingData(gert::TilingContext* context, EngramFetchGradTilingData& tilingData,
                                         const RailTopology& topoArg, int64_t numTokens, uint32_t worldSize,
                                         int64_t totalRecv, int64_t hiddenDim, int64_t gradUniqueRows)
{
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    tilingData.numTokens = numTokens;
    tilingData.rankSize = worldSize; // 分轨语义：world
    tilingData.totalRecv = totalRecv;
    tilingData.hiddenDim = hiddenDim;

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

    auto gradFetchedDesc = context->GetInputDesc(IN_GRAD_FETCHED);
    ge::DataType inputDtype = gradFetchedDesc->GetDataType();
    int64_t bytesPerElem = ge::GetSizeByDataType(inputDtype);
    OP_TILING_CHECK(tilingData.hiddenDim > INT64_MAX / bytesPerElem,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "hidden_dim", std::to_string(tilingData.hiddenDim).c_str(),
                        (std::string("hiddenDim * bytesPerElem (") + std::to_string(bytesPerElem) +
                         ") overflows INT64_MAX (hiddenBytes overflow)")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    tilingData.hiddenBytes = tilingData.hiddenDim * bytesPerElem;
    tilingData.inputDtype = static_cast<int32_t>(inputDtype);

    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    tilingData.outputDtype = static_cast<int32_t>(gradUniqueDesc->GetDataType());

    auto commBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_BUFFER_SIZE);
    tilingData.commBufferSize = *commBufferSizePtr;

    // ═══ 拓扑落表（感知 ranksPerSupernode → kernel 唯一权威；超节点数/轨道数 tiling 与 kernel 同公式推导） ═══
    RailTopology topo = topoArg;
    tilingData.ranksPerSupernode = topo.ranksPerSupernode;

    // ═══ 发送批行数（拼行块 ≤ 65504B，不进 WIN 结构公式） ═══
    // blockLen 为 uint16，预算 65504：P 整除 65536 的满批 65536B 会截断为 0、拷 0 字节，
    // sStage 陈旧内容被当行发出；行距在 int64 域校验后硬拒绝超预算值（host 期干净报错）
    const int64_t rowStrideI64 = AlignTo(tilingData.hiddenBytes + Mc2Kernel::ROW_HEAD_BYTES, Mc2Kernel::UB_ALIGN); // P
    constexpr uint32_t BATCH_BLOCK_BYTES = 65504U;
    OP_TILING_CHECK(rowStrideI64 > static_cast<int64_t>(BATCH_BLOCK_BYTES),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "hidden_dim",
                        (std::string("P=") + std::to_string(rowStrideI64) +
                         " (hiddenBytes=" + std::to_string(tilingData.hiddenBytes) + ")")
                            .c_str(),
                        "grad row stride exceeds the batch block budget (blockLen uint16, single row not movable)"),
                    return ge::GRAPH_FAILED);
    const uint32_t rowStride = static_cast<uint32_t>(rowStrideI64);
    uint32_t batchRows = BATCH_BLOCK_BYTES / rowStride;
    if (batchRows < 1U) {
        batchRows = 1U;
    }

    // ═══ N 反解：把默认 buffer 分完——向下取整装到装不下为止（非 2 的幂，
    //     环形回绕用条件减法）；尾差 < W×P 留 DFX；校验 N ≥ 2×批行数（信用窗口有效） ═══
    int64_t nBudget =
        (tilingData.commBufferSize - static_cast<int64_t>(topo.winHead)) / static_cast<int64_t>(topo.areas) -
        Mc2Kernel::RECV_AREA_HEAD;
    OP_TILING_CHECK(nBudget <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "comm_buffer_size", std::to_string(tilingData.commBufferSize).c_str(),
                        (std::string("> winHead + areas*96 + 2*rows (winHead=") + std::to_string(topo.winHead) +
                         ", areas=" + std::to_string(topo.areas) + "; WIN budget too small)")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    int64_t nFit = nBudget / rowStride;
    OP_TILING_CHECK(nFit < 2LL * static_cast<int64_t>(batchRows),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "comm_buffer_size", std::to_string(tilingData.commBufferSize).c_str(),
                        (std::string("N=") + std::to_string(nFit) + " < 2*batchRows=" + std::to_string(batchRows) +
                         " (P=" + std::to_string(rowStride) + "; buffer too small or row too large)")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    uint32_t numRows = static_cast<uint32_t>(nFit);
    tilingData.numRows = numRows;
    tilingData.batchRows = batchRows;

    // ═══ 收集数组容量：cap = recv_local_entry.dim0（前向记账直接提供，零新增 attr） ═══
    int64_t cap = totalRecv;
    OP_TILING_CHECK(cap < 0 || cap > INT64_MAX / tilingData.hiddenBytes,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "recv_local_entry", std::to_string(cap).c_str(),
                        (std::string("collect cap overflow (cap > INT64_MAX/hiddenBytes, hiddenBytes=") +
                         std::to_string(tilingData.hiddenBytes) + ")")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        gradUniqueRows < cap,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "grad_unique_out", std::to_string(gradUniqueRows).c_str(),
                                  (std::string(">= cap (") + std::to_string(cap) +
                                   " = recv_local_entry.dim0; rail output rows must cover expected reverse inflow)")
                                      .c_str()),
        return ge::GRAPH_FAILED);

    // ═══ UB 常驻 O(1)（与 world 无关——替换老的 O(rankSize) 三件套） ═══
    uint32_t rowBlockBytes = batchRows * rowStride;
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
    uint32_t indicesBytes = Mc2Kernel::IDX_BUF_BYTES; // unsort perm 分片 + sendCounts walk 分块
    // 常驻 UB 预算账本：信用读自旋格 + 收件区三计数头 + 拼行块 + 游标/桶指针
    const uint64_t creditSpinBytes = Mc2Kernel::UB_ALIGN;  // 信用读自旋格
    const uint64_t cursorBytes = 2U * Mc2Kernel::UB_ALIGN; // 游标/桶指针
    uint64_t permanentUb = Mc2Kernel::HCOMM_INIT_SIZE + statusBytes + tempBytes + indicesBytes + creditSpinBytes +
                           Mc2Kernel::RECV_AREA_HEAD + static_cast<uint64_t>(rowBlockBytes) + cursorBytes;
    OP_TILING_CHECK(permanentUb >= tilingData.ubSize,
                    OP_LOGE(nodeName, "permanent UB (%llu) exceeds availUbSize (%llu)", permanentUb, tilingData.ubSize),
                    return ge::GRAPH_FAILED);

    // ═══ 阶段 C（unique/sort）缓冲预算——逻辑与老实现同口径（与 world 无关） ═══
    bool fullRowFits = tilingData.hiddenBytes <= static_cast<int64_t>(Mc2Kernel::GRAD_BUF_BYTES);
    uint32_t inRowStride = static_cast<uint32_t>(AlignTo(tilingData.hiddenBytes, Mc2Kernel::UB_ALIGN));
    uint32_t maxByPong = Mc2Kernel::GRAD_PING_BYTES / inRowStride;
    if (maxByPong < 1U) {
        maxByPong = 1U;
    }
    uint32_t fp32RowStride =
        static_cast<uint32_t>(AlignTo(static_cast<int64_t>(tilingData.hiddenDim) * sizeof(float), Mc2Kernel::UB_ALIGN));
    uint32_t accumNeed =
        Mc2Kernel::ACCUM_BUF_COPIES *
        static_cast<uint32_t>(AlignTo(static_cast<int64_t>(tilingData.hiddenDim) * sizeof(float), Mc2Kernel::UB_ALIGN));
    bool needCast = (tilingData.inputDtype != static_cast<int32_t>(ge::DT_FLOAT));
    uint32_t uniqueEntryBytes = Mc2Kernel::FLUSH_CAST_HEAD_BYTES;
    if (tilingData.outputDtype != static_cast<int32_t>(ge::DT_FLOAT)) {
        uniqueEntryBytes +=
            DOUBLE_BUFFER_NUM * static_cast<uint32_t>(AlignTo(hiddenDim * HALF_DTYPE_BYTES, Mc2Kernel::UB_ALIGN));
        if (uniqueEntryBytes > Mc2Kernel::ENTRY_BUF_BYTES) {
            fullRowFits = false;
        }
    }
    uint64_t availableForPool = tilingData.ubSize - permanentUb;
    uint32_t minUniqueNeed =
        Mc2Kernel::GRAD_BUF_BYTES + uniqueEntryBytes + accumNeed + (needCast ? DOUBLE_BUFFER_NUM * fp32RowStride : 0U);
    if (minUniqueNeed > availableForPool) {
        fullRowFits = false;
    }
    if (fullRowFits) {
        tilingData.chunkElems = 0U;
        uint64_t availableForCast = availableForPool - Mc2Kernel::GRAD_BUF_BYTES - uniqueEntryBytes - accumNeed;
        uint32_t maxByCast =
            needCast ? static_cast<uint32_t>(availableForCast / (fp32RowStride * Mc2Kernel::ACCUM_BUF_COPIES)) :
                       maxByPong;
        uint32_t gradSubBatch = maxByPong;
        if (maxByCast < gradSubBatch) {
            gradSubBatch = maxByCast;
        }
        if (gradSubBatch < 1U) {
            gradSubBatch = 1U;
        }
        if (gradSubBatch > GRAD_SUB_BATCH_CAP) {
            gradSubBatch = GRAD_SUB_BATCH_CAP;
        }
        tilingData.gradSubBatch = gradSubBatch;
    } else {
        tilingData.chunkElems = Mc2Kernel::HIDDEN_CHUNK_ELEMS;
        uint32_t chunkInStride = static_cast<uint32_t>(
            AlignTo(static_cast<int64_t>(Mc2Kernel::HIDDEN_CHUNK_ELEMS) * bytesPerElem, Mc2Kernel::UB_ALIGN));
        uint32_t chunkFp32Stride = static_cast<uint32_t>(
            AlignTo(static_cast<int64_t>(Mc2Kernel::HIDDEN_CHUNK_ELEMS) * sizeof(float), Mc2Kernel::UB_ALIGN));
        uint32_t chunkEntryBytes = Mc2Kernel::FLUSH_CAST_HEAD_BYTES + DOUBLE_BUFFER_NUM * chunkFp32Stride;
        uint32_t chunkMinNeed =
            Mc2Kernel::GRAD_BUF_BYTES + chunkEntryBytes + chunkFp32Stride + DOUBLE_BUFFER_NUM * chunkFp32Stride;
        OP_TILING_CHECK(chunkMinNeed > availableForPool,
                        OP_LOGE(nodeName, "chunk-mode UB pool overflow: need %u, availableForPool %llu", chunkMinNeed,
                                availableForPool),
                        return ge::GRAPH_FAILED);
        uint32_t chunkRows = Mc2Kernel::GRAD_BUF_BYTES / chunkInStride;
        if (chunkRows < 1U) {
            chunkRows = 1U;
        }
        tilingData.gradSubBatch = chunkRows;
    }

    // 三组核数 S:F:O ≈ 1:2:2
    constexpr uint32_t AIV_TIER_HI = 72U;
    constexpr uint32_t AIV_TIER_MID = 64U;
    constexpr uint32_t AIV_TIER_LOW = 56U;
    constexpr uint32_t TIER_HI_SEND = 14U;
    constexpr uint32_t TIER_HI_FWD = 29U;
    constexpr uint32_t TIER_MID_SEND = 12U;
    constexpr uint32_t TIER_MID_FWD = 26U;
    constexpr uint32_t TIER_LOW_SEND = 11U;
    constexpr uint32_t TIER_LOW_FWD = 22U;
    uint32_t aiv = tilingData.aivNum;
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
    tilingData.sendCoreCnt = sendCores;
    tilingData.fwdCoreCnt = fwdCores;
    tilingData.ownerCoreCnt = ownerCores;

    // ═══ SortLib 参数（与老同公式，total → cap=totalRecv 口径） ═══
    uint32_t sortCoreCnt = aiv / 2U;
    if (sortCoreCnt < 1U) {
        sortCoreCnt = 1U;
    }
    tilingData.sortCoreCnt = sortCoreCnt;

    uint64_t sortTileCalibReserve = 512U + statusBytes + rowBlockBytes + 2U * Mc2Kernel::A2A_COPY_HALF_BYTES;
    uint64_t budget = tilingData.ubSize - sortTileCalibReserve;
    constexpr uint64_t kSortLibFixedBytes = 5120U;
    constexpr uint64_t kMinTileElems = 64U;
    constexpr uint64_t kBytesPerElem = 21U;
    constexpr uint32_t kSortTmpHeadBytes = 512U;
    constexpr uint32_t kSortTmpTailBytes = 256U;
    constexpr uint32_t kSortTmpBytesPerGroup = 7U;
    constexpr uint32_t kSortTmpGroupTiles = 32U;
    OP_TILING_CHECK(
        budget < kSortLibFixedBytes + kMinTileElems * kBytesPerElem,
        OP_LOGE(nodeName, "SortLib UB budget too small: ubSize=%llu, calibReserve=%llu, need>=%llu", tilingData.ubSize,
                sortTileCalibReserve, kSortLibFixedBytes + kMinTileElems * kBytesPerElem),
        return ge::GRAPH_FAILED);

    uint32_t tilesPerSortCore = 2U;
    uint32_t numTile =
        static_cast<uint32_t>((cap + sortCoreCnt * tilesPerSortCore - 1) / (sortCoreCnt * tilesPerSortCore));
    uint32_t ubCap = static_cast<uint32_t>((budget - kSortLibFixedBytes) / kBytesPerElem);
    if (numTile < kMinTileElems) {
        numTile = kMinTileElems;
    }
    if (numTile > ubCap) {
        numTile = ubCap;
    }
    tilingData.sortNumTileData = numTile;
    tilingData.sortTileCount = static_cast<uint32_t>((cap + numTile - 1) / numTile);
    uint32_t roundTile = (numTile + kSortTmpGroupTiles - 1U) / kSortTmpGroupTiles * kSortTmpGroupTiles;
    tilingData.sortTmpUbSize = kSortTmpHeadBytes + kSortTmpBytesPerGroup * roundTile + kSortTmpTailBytes;

    return ge::GRAPH_SUCCESS;
}

// ─────────────────────── 分轨 workspace（11 段布局） ───────────────────────

/// @brief 分轨 workspace 尺寸计算（与 kernel CalcWorkSpaceOffset 逐段一致，11 段布局）
static ge::graphStatus SetRailWorkSpace(gert::TilingContext* context, const EngramFetchGradTilingData& tilingData)
{
    const char* nodeName = context->GetNodeName();
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    int64_t world = static_cast<int64_t>(tilingData.rankSize);
    int64_t numTokens = tilingData.numTokens;
    int64_t hiddenBytes = tilingData.hiddenBytes;
    int64_t cap = tilingData.totalRecv; // cap = recv_local_entry.dim0（零新增 attr）
    RailTopology topo = DeriveRailTopology(tilingData.ranksPerSupernode, tilingData.rankSize);
    const int64_t rowStrideI64 = AlignTo(hiddenBytes + Mc2Kernel::ROW_HEAD_BYTES, Mc2Kernel::UB_ALIGN);
    OP_TILING_CHECK(rowStrideI64 > static_cast<int64_t>(UINT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "hidden_dim",
                                                          (std::string("P=") + std::to_string(rowStrideI64) +
                                                           " (hiddenBytes=" + std::to_string(hiddenBytes) + ")")
                                                              .c_str(),
                                                          "grad row stride overflows UINT32_MAX"),
                    return ge::GRAPH_FAILED);
    const uint32_t rowStride = static_cast<uint32_t>(rowStrideI64);
    uint32_t aivNum = tilingData.aivNum;

    OP_TILING_CHECK(numTokens > 0 && hiddenBytes > INT64_MAX / numTokens,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "num_tokens",
                        (std::string("numTokens=") + std::to_string(numTokens) +
                         ", hiddenBytes=" + std::to_string(hiddenBytes) + "; product overflows INT64_MAX)")
                            .c_str(),
                        "workspace overflow (wsGradSorted = numTokens * hiddenBytes)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cap > 0 && hiddenBytes > INT64_MAX / cap,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "recv_local_entry",
                        (std::string("cap=") + std::to_string(cap) + ", hiddenBytes=" + std::to_string(hiddenBytes) +
                         "; product overflows INT64_MAX)")
                            .c_str(),
                        "workspace overflow"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cap > 0 && tilingData.hiddenDim > INT64_MAX / cap,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "recv_local_entry",
                        (std::string("cap=") + std::to_string(cap) +
                         ", hiddenDim=" + std::to_string(tilingData.hiddenDim) + "; product overflows INT64_MAX)")
                            .c_str(),
                        "workspace overflow"),
                    return ge::GRAPH_FAILED);

    // ① gradSorted：UnsortGrad 产出（按 dst 升序段）
    int64_t wsGradSorted = numTokens * hiddenBytes;
    // ② ownerSorted：每行 dst 伴随数组
    int64_t wsOwnerSorted = AlignTo(numTokens * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);
    // ③ 桶表：头部(taskCount+每核段数) + 核条带(段(dst,start,cnt)×3 int32) + 任务表((kind,dst,start,cnt)×4)
    //    条带容量 = 行/核+2；任务容量 = numTokens+核数（每核段数上界之和）
    int64_t stripCap = (numTokens + aivNum - 1) / aivNum + 2;
    int64_t wsBucketSeg = AlignTo((1 + aivNum) * 4, Mc2Kernel::UB_ALIGN) +
                          static_cast<int64_t>(aivNum) * stripCap * 3 * static_cast<int64_t>(sizeof(int32_t)) +
                          (numTokens + aivNum) * 4 * static_cast<int64_t>(sizeof(int32_t));
    wsBucketSeg = AlignTo(wsBucketSeg, Mc2Kernel::UB_ALIGN);
    // ④ F 暂存环：fCoreCnt × ranksPerSupernode 槽 × batchRows 行 × P——每 (F核, owner 槽) 私有环格
    //    （SPSC：F 备行/属主 S 核发走；背压 = staged−drained < batchRows）
    int64_t wsRelay = static_cast<int64_t>(tilingData.fwdCoreCnt) * static_cast<int64_t>(topo.ranksPerSupernode) *
                      tilingData.batchRows * rowStride;
    // ⑤ relay 三表：staged（F 已备行数，F 单写）/ drained（属主 S 核游标，S 单写）/
    //    eof（F 相位完结标记）——各 align32(fCoreCnt×ranksPerSupernode×4)，与 kernel 逐段一致
    int64_t relayCtlBytes =
        AlignTo(static_cast<int64_t>(tilingData.fwdCoreCnt) * static_cast<int64_t>(topo.ranksPerSupernode) *
                    static_cast<int64_t>(sizeof(int32_t)),
                Mc2Kernel::UB_ALIGN);
    int64_t wsRelayCtl = 3 * relayCtlBytes;
    // ⑥ S 组拼块 GM 暂存：行头与 gradSorted 数据拼块后经此转 WriteNbi（每 S 核一格）
    int64_t wsSStage = static_cast<int64_t>(tilingData.sendCoreCnt) * tilingData.batchRows * rowStride;
    // ⑦ collectEntry：O 组收集条目号
    int64_t wsCollectEntry = AlignTo(cap * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);
    // ⑧ collectRows：O 组收集行
    int64_t wsCollectRows = cap * hiddenBytes;
    // ⑨ counterScratch：信用读落地格（每核独占 32B）
    int64_t wsCounterScratch = static_cast<int64_t>(aivNum) * Mc2Kernel::UB_ALIGN;
    // ⑩ srcArriveCnt：每源到达计数（O 组 (src,计数) 反查）
    int64_t wsSrcArrive = AlignTo(world * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);
    // ⑪ recvPrefix：recv_counts 前缀（O 组反查段首）
    int64_t wsRecvPrefix = AlignTo(world * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);
    // ⑫ sendPrefix：send_counts 前缀（MarkOwnerSorted 二分定位）
    int64_t wsSendPrefix = AlignTo((world + 1) * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);
    // ⑬ flagScratch：barrier flag(=1) 32B + collectedCount 32B
    constexpr int64_t kFlagScratchBytes = 64;
    int64_t wsFlagScratch = kFlagScratchBytes;

    // ⑫ 排序/去重区段（老公式，totalRecv → cap；segCount/coreStart/sortCompanion/sort 六段/sortBarrier）
    int64_t coreArrayAligned = AlignTo(static_cast<int64_t>(aivNum) * sizeof(int32_t), Mc2Kernel::UB_ALIGN);
    int64_t wsSegCount = coreArrayAligned;
    int64_t wsCoreStart = coreArrayAligned;
    OP_TILING_CHECK(
        cap > INT64_MAX / static_cast<int64_t>(sizeof(int32_t)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "recv_local_entry", std::to_string(cap).c_str(),
                                              "sort companion overflow (cap > INT32_MAX; wsSortCompanion = cap * 4)"),
        return ge::GRAPH_FAILED);
    int64_t wsSortCompanion = AlignTo(cap * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);

    constexpr int64_t kRadixRounds = 4;
    constexpr int64_t kRadixBinNum = 256;
    constexpr int64_t kRadixCountBytes = 4;
    constexpr int64_t kRadixHistElemBytes = 2;
    constexpr int64_t kRadixHistArrayNum = 2;
    int64_t slSeg0 = AlignTo(kRadixBinNum * kRadixRounds * kRadixCountBytes, Mc2Kernel::UB_ALIGN);
    int64_t slSeg1 =
        AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * kRadixBinNum * kRadixRounds * kRadixCountBytes,
                Mc2Kernel::UB_ALIGN);
    int64_t slSeg2 = AlignTo(cap * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);
    int64_t slSeg3 = kRadixHistArrayNum *
                     AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * kRadixBinNum * kRadixHistElemBytes,
                             Mc2Kernel::UB_ALIGN);
    int64_t slSeg4 =
        AlignTo(static_cast<int64_t>(tilingData.sortTileCount) * tilingData.sortNumTileData, Mc2Kernel::UB_ALIGN);
    int64_t slSeg5 = AlignTo(cap * static_cast<int64_t>(sizeof(int32_t)), Mc2Kernel::UB_ALIGN);

    constexpr int64_t SOFT_BARRIER_MAX_GENS = 16;
    int64_t wsSortBarrier =
        SOFT_BARRIER_MAX_GENS *
        ((static_cast<int64_t>(aivNum) * static_cast<int64_t>(Mc2Kernel::STATE_OFFSET) + Mc2Kernel::UB_ALIGN - 1) /
         Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN);
    int64_t wsSortTotal = wsSortCompanion + slSeg0 + slSeg1 + slSeg2 + slSeg3 + slSeg4 + slSeg5 + wsSortBarrier;

    int64_t wsTotal = wsGradSorted + wsOwnerSorted + wsBucketSeg + wsRelay + wsRelayCtl + wsSStage + wsCollectEntry +
                      wsCollectRows + wsCounterScratch + wsSrcArrive + wsRecvPrefix + wsSendPrefix + wsFlagScratch +
                      wsSegCount + wsCoreStart + wsSortTotal;
    wsTotal = AlignTo(wsTotal, BUFFER_ALIGNMENT);
    wsTotal += SYSTEM_NEED_WORKSPACE;

    workSpaces[0] = static_cast<size_t>(wsTotal);
    return ge::GRAPH_SUCCESS;
}

// ═══ 分轨 tiling 主入口（老 tiling 入口 EngramFetchGradTilingFunc 转调）═══
/// @brief EngramFetchGrad 分轨模型 tiling 主入口
ge::graphStatus EngramFetchGradRailAlignedTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr,
                    OP_LOGE("engram_fetch_grad_railaligned_tiling", "failed to get tiling context."),
                    return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    // nodeName 为空时不可作 entityName 传入宏（宏内 std::string 构造空指针为 UB）——
    // 以入口字面量兜底上报，与上行 context 判空分支口径一致
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT("engram_fetch_grad_railaligned_tiling", "nodeName"),
                    return ge::GRAPH_FAILED);

    EngramFetchGradTilingData* tilingData = context->GetTilingData<EngramFetchGradTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    int64_t numTokens = 0;
    uint32_t worldSize = 0;
    int64_t totalRecv = 0;
    int64_t hiddenDim = 0;
    int64_t gradUniqueRows = 0;
    RailTopology topoAttr;
    OP_TILING_CHECK(CheckRailTensorDataType(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail check input/output dtype/format failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckRailTensorDim(context, numTokens, worldSize, totalRecv, hiddenDim, gradUniqueRows) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "rail check input/output shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRailAttrs(context, topoAttr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail check attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetRailPlatformInfo(context, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail set platform info failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context->SetScheduleMode(SCHEDULE_MODE_BATCH) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail SetScheduleMode(1) failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetRailTilingData(context, *tilingData, topoAttr, numTokens, worldSize, totalRecv, hiddenDim,
                                      gradUniqueRows) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail set tiling data failed."), return ge::GRAPH_FAILED);

    // 框架按分轨 tilingKey 实例分轨 kernel 模板
    context->SetTilingKey(GET_TPL_TILING_KEY(ENGRAM_FETCH_GRAD_RAIL_MODE));

    OP_TILING_CHECK(SetRailWorkSpace(context, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "rail set workspace failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EngramFetchGradTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("engram_fetch_grad_tiling", "failed to get tiling context."),
                    return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT("engram_fetch_grad_tiling", "nodeName"),
                    return ge::GRAPH_FAILED);

    // ═══ 分轨模型分发：world > ranksPerSupernode（attr 缺省 128）且平铺 rail ctx → 分轨模板；
    //     ctx 长度仅为建链产物一致性防御断言；world == ranksPerSupernode 单超节点走老模板。老函数体零改动。═══
    {
        auto railAttrs = context->GetAttrs();
        OP_TILING_CHECK(railAttrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
        // 128 = 新老模板分发兼容边界（非感知缺省）：attr 未感知/非法时按 128 判界
        constexpr int64_t DISPATCH_DEFAULT_RANKS_PER_SUPERNODE = 128;
        auto ranksPerSupernodePtr = railAttrs->GetAttrPointer<int64_t>(ATTR_RANKS_PER_SUPERNODE);
        const bool attrSensed = ranksPerSupernodePtr != nullptr && *ranksPerSupernodePtr > 0 &&
                                *ranksPerSupernodePtr <= static_cast<int64_t>(Mc2Kernel::MAX_QP_SIZE);
        const int64_t ranksPerSupernode = attrSensed ? *ranksPerSupernodePtr : DISPATCH_DEFAULT_RANKS_PER_SUPERNODE;
        const gert::StorageShape* railCtxShape = context->GetInputShape(IN_COMM_CONTEXT);
        OP_TILING_CHECK(railCtxShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"),
                        return ge::GRAPH_FAILED);
        int64_t railCtxElems = railCtxShape->GetStorageShape().GetDim(0);
        const gert::StorageShape* railSendShape = context->GetInputShape(IN_SEND_COUNTS);
        OP_TILING_CHECK(railSendShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCounts"),
                        return ge::GRAPH_FAILED);
        int64_t railSendDim0 = railSendShape->GetStorageShape().GetDim(0);
        int64_t railWorld = (railSendDim0 > 0 && railSendDim0 % SEND_COUNTS_ALIGN_FACTOR == 0) ?
                                railSendDim0 / SEND_COUNTS_ALIGN_FACTOR :
                                -1;
        if (railWorld > ranksPerSupernode && railCtxElems == Mc2Kernel::CTX_RAIL_ELEMS) {
            // 轨道分支：world > ranksPerSupernode（跨超节点轨道腿存在）且建链产物为平铺 rail ctx
            OP_LOGI(nodeName, "Enter rail-aligned tiling func, world=%lld, ranksPerSupernode=%lld.", railWorld,
                    ranksPerSupernode);
            return EngramFetchGradRailAlignedTilingFunc(context);
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

    OP_LOGI(nodeName, "Enter EngramFetchGrad tiling func.");

    EngramFetchGradTilingData* tilingData = context->GetTilingData<EngramFetchGradTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    int64_t numTokens = 0;
    uint32_t rankSize = 0;
    int64_t totalRecv = 0;
    int64_t hiddenDim = 0;
    OP_TILING_CHECK(TilingCheckEngramFetchGrad(context, numTokens, rankSize, totalRecv, hiddenDim) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check input/output failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrs(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "check attrs failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetPlatformInfo(context, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set platform info failed."), return ge::GRAPH_FAILED);

    auto schedRet = context->SetScheduleMode(SCHEDULE_MODE_BATCH);
    OP_TILING_CHECK(schedRet != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "SetScheduleMode(1) failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetTilingData(context, *tilingData, numTokens, rankSize, totalRecv, hiddenDim) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set tiling data failed."), return ge::GRAPH_FAILED);

    SetTilingKey(context);

    OP_TILING_CHECK(SetWorkSpace(context, *tilingData) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "set workspace failed."),
                    return ge::GRAPH_FAILED);

    PrintEngramFetchGradTilingData(tilingData, nodeName);
    OP_LOGI(nodeName, "EngramFetchGrad tiling end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetchGrad).Tiling(EngramFetchGradTilingFunc);
} // namespace Mc2Tiling
