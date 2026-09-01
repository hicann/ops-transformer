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

constexpr int32_t HIDDEN_SIZE_ALIGN = 128;
constexpr int64_t BUFFER_ALIGNMENT = 2 * 1024 * 1024;

static const std::vector<ge::DataType> GRAD_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

static bool IsContains(const std::vector<ge::DataType> &list, ge::DataType value)
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

static void PrintEngramFetchGradTilingData(const EngramFetchGradTilingData *tilingData, const char *nodeName)
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

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context)
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

static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

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

static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, int64_t &numTokens, uint32_t &rankSize,
                                      int64_t &totalRecv, int64_t &hiddenDim)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *commContextShape = context->GetInputShape(IN_COMM_CONTEXT);
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
    const gert::StorageShape *gradShape = context->GetInputShape(IN_GRAD_FETCHED);
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
    OP_TILING_CHECK(
        hiddenDim % HIDDEN_SIZE_ALIGN != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "gradFetched", (std::string("dim1=") + std::to_string(hiddenDim)).c_str(),
                                  (std::string("must be ") + std::to_string(HIDDEN_SIZE_ALIGN) + "-aligned").c_str()),
        return ge::GRAPH_FAILED);

    // perm: 1D (T,), dim0 == numTokens
    const gert::StorageShape *permShape = context->GetInputShape(IN_PERM);
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
    const gert::StorageShape *sendCountsShape = context->GetInputShape(IN_SEND_COUNTS);
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
    const gert::StorageShape *recvCountsShape = context->GetInputShape(IN_RECV_COUNTS);
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
    const gert::StorageShape *recvLocalEntryShape = context->GetInputShape(IN_RECV_LOCAL_ENTRY);
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
    const gert::StorageShape *numRecvShape = context->GetInputShape(IN_NUM_RECV);
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
    const gert::StorageShape *gradUniqueShape = context->GetOutputShape(OUT_GRAD_UNIQUE);
    OP_TILING_CHECK(gradUniqueShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "gradUniqueOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        gradUniqueShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "gradUniqueOut", (std::to_string(gradUniqueShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of gradUniqueOut must be 2D."),
        return ge::GRAPH_FAILED);

    // uniqueLocalEntryOut: 1D
    const gert::StorageShape *uniqueLocalEntryShape = context->GetOutputShape(OUT_UNIQUE_LOCAL_ENTRY);
    OP_TILING_CHECK(uniqueLocalEntryShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "uniqueLocalEntryOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(uniqueLocalEntryShape->GetStorageShape().GetDimNum() != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "uniqueLocalEntryOut",
                        (std::to_string(uniqueLocalEntryShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of uniqueLocalEntryOut must be 1D."),
                    return ge::GRAPH_FAILED);

    // numUniqueOut: 1D, dim0 == 1
    const gert::StorageShape *numUniqueShape = context->GetOutputShape(OUT_NUM_UNIQUE);
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

static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

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

static ge::graphStatus TilingCheckEngramFetchGrad(const gert::TilingContext *context, int64_t &numTokens,
                                                  uint32_t &rankSize, int64_t &totalRecv, int64_t &hiddenDim)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckTensorPtrNullptr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDataType(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDim(context, numTokens, rankSize, totalRecv, hiddenDim) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorFormat(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "params format is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
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

static ge::graphStatus SetPlatformInfo(gert::TilingContext *context, EngramFetchGradTilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
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

static ge::graphStatus SetTilingData(gert::TilingContext *context, EngramFetchGradTilingData &tilingData,
                                     int64_t numTokens, uint32_t rankSize, int64_t totalRecv, int64_t hiddenDim)
{
    const char *nodeName = context->GetNodeName();
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
    // 单行 grad 必须可放入整缓冲 gradBuf_（64KB），同时消除 uint32 截断风险（SEC-2.3/topk-7/SEC-1.1）
    OP_TILING_CHECK(tilingData.hiddenBytes > static_cast<int64_t>(Mc2Kernel::GRAD_BUF_BYTES),
                    OP_LOGE(nodeName, "hiddenBytes=%lld exceeds grad buffer capacity %u", tilingData.hiddenBytes,
                            Mc2Kernel::GRAD_BUF_BYTES),
                    return ge::GRAPH_FAILED);
    tilingData.inputDtype = static_cast<int32_t>(inputDtype);

    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    tilingData.outputDtype = static_cast<int32_t>(gradUniqueDesc->GetDataType());
    // 半精度输出时 FlushAccum 借用 entryBuf_ 尾部（20KB 偏移后）作 flush cast 双缓冲，
    // 需 20KB + 2*Align32(hiddenDim*2) 不越界（SEC-4.2③/TIL-3①；Kernel 侧另有 RUNTIME_ABORT 兜底）
    if (tilingData.outputDtype != static_cast<int32_t>(ge::DT_FLOAT)) {
        int64_t flushCastNeed =
            Mc2Kernel::FLUSH_CAST_HEAD_BYTES + 2 * AlignTo(static_cast<int64_t>(hiddenDim) * 2, Mc2Kernel::UB_ALIGN);
        OP_TILING_CHECK(flushCastNeed > Mc2Kernel::ENTRY_BUF_BYTES,
                        OP_LOGE(nodeName,
                                "hiddenDim=%lld too large for half-precision output: flush cast staging "
                                "needs %lld bytes, entryBuf capacity %lld",
                                hiddenDim, flushCastNeed, Mc2Kernel::ENTRY_BUF_BYTES),
                        return ge::GRAPH_FAILED);
    }

    auto commBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_BUFFER_SIZE);
    tilingData.commBufferSize = *commBufferSizePtr;

    // 动态计算 gradSubBatch：常驻区与 Kernel 侧 InitBuffer 布局一一对应（精确推导而非固定 9KB），
    // accum 预留双缓冲 2 行与 Kernel AccumBufBytes 对齐（TIL-2/SIMT-UB-01 修复）
    uint32_t statusBytes =
        static_cast<uint32_t>(AlignTo(static_cast<int64_t>(rankSize) * Mc2Kernel::STATE_OFFSET, Mc2Kernel::UB_ALIGN));
    uint32_t tempBytes = statusBytes;
    if (tempBytes < (Mc2Kernel::ENTRY_BATCH_CAP * sizeof(int32_t))) {
        tempBytes = (Mc2Kernel::ENTRY_BATCH_CAP * sizeof(int32_t));
    }
    // displs 批量构造区：sdispl+rdispl 两段各 rankSize*32B
    uint32_t displsBatchBytes = 2U * rankSize * Mc2Kernel::STATE_OFFSET;
    if (tempBytes < displsBatchBytes) {
        tempBytes = displsBatchBytes;
    }
    uint32_t coreArrayBytes = static_cast<uint32_t>(
        AlignTo(static_cast<int64_t>(tilingData.aivNum) * sizeof(int32_t) * 2U, Mc2Kernel::UB_ALIGN));
    if (tempBytes < coreArrayBytes) {
        tempBytes = coreArrayBytes;
    }
    uint32_t indicesBytes = (Mc2Kernel::IDX_BUF_BYTES > statusBytes) ? Mc2Kernel::IDX_BUF_BYTES : statusBytes;
    // countsBuf 暂存区：recvCounts + sendCounts（偏移 rankSize*32B + 长度 rankSize*32B）
    uint32_t stagingBytes = 2U * rankSize * Mc2Kernel::STATE_OFFSET;
    if (indicesBytes < stagingBytes) {
        indicesBytes = stagingBytes;
    }
    uint64_t permanentUb = Mc2Kernel::HCOMM_INIT_SIZE + statusBytes + tempBytes + indicesBytes;
    OP_TILING_CHECK(permanentUb >= static_cast<uint64_t>(tilingData.ubSize),
                    OP_LOGE(nodeName, "permanent UB (%llu) exceeds availUbSize (%llu)", permanentUb, tilingData.ubSize),
                    return ge::GRAPH_FAILED);
    uint32_t maxByPong = Mc2Kernel::GRAD_PING_BYTES / static_cast<uint32_t>(tilingData.hiddenBytes);
    if (maxByPong < 1U) {
        maxByPong = 1U;
    }
    uint32_t accumNeed = Mc2Kernel::ACCUM_BUF_COPIES *
                         static_cast<uint32_t>(AlignTo(static_cast<int64_t>(tilingData.hiddenDim) * sizeof(float),
                                                       static_cast<int64_t>(Mc2Kernel::UB_ALIGN)));
    bool needCast = (tilingData.inputDtype != static_cast<int32_t>(ge::DT_FLOAT));
    uint32_t availableForPool = static_cast<uint32_t>(tilingData.ubSize) - static_cast<uint32_t>(permanentUb);
    uint32_t availableForCast = (availableForPool > Mc2Kernel::COMM_BUF_BYTES + accumNeed) ?
                                    (availableForPool - Mc2Kernel::COMM_BUF_BYTES - accumNeed) :
                                    0U;
    uint32_t maxByCast = needCast ? (availableForCast / (static_cast<uint32_t>(tilingData.hiddenDim) * sizeof(float) *
                                                         Mc2Kernel::ACCUM_BUF_COPIES)) :
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
    OP_LOGD(nodeName, "gradSubBatch=%u (hiddenBytes=%lld, hiddenDim=%lld, maxByPong=%u, maxByCast=%u, available=%u)",
            gradSubBatch, tilingData.hiddenBytes, tilingData.hiddenDim, maxByPong, maxByCast, availableForCast);
    OP_LOGD(nodeName, "permanentUb=%llu (status=%u, temp=%u, indices=%u)", permanentUb, statusBytes, tempBytes,
            indicesBytes);

    OP_LOGD(nodeName,
            "SetTilingData: numTokens=%lld, hiddenDim=%lld, numEntriesPerRank=%d, hiddenBytes=%lld, "
            "ubSize=%llu, rankSize=%u, totalRecv=%lld, commBufferSize=%lld",
            tilingData.numTokens, tilingData.hiddenDim, tilingData.numEntriesPerRank, tilingData.hiddenBytes,
            tilingData.ubSize, tilingData.rankSize, tilingData.totalRecv, tilingData.commBufferSize);
    return ge::GRAPH_SUCCESS;
}

static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    const uint64_t tilingKey = GET_TPL_TILING_KEY(ENGRAM_FETCH_GRAD_DEFAULT_MODE);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%llu] in engram_fetch_grad.", tilingKey);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const EngramFetchGradTilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
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
    int64_t wsFlagScratch = 32;
    int64_t coreArrayAligned = AlignTo(static_cast<int64_t>(tilingData.aivNum) * sizeof(int32_t), Mc2Kernel::UB_ALIGN);
    int64_t wsSegCount = coreArrayAligned;
    int64_t wsCoreStart = coreArrayAligned;

    int64_t maxSortCount = totalRecv;
    OP_TILING_CHECK(maxSortCount > INT64_MAX / static_cast<int64_t>(sizeof(int32_t)),
                    OP_LOGE(nodeName, "sort temp overflow: totalRecv=%lld", totalRecv), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxSortCount > INT64_MAX - 4095,
                    OP_LOGE(nodeName, "sortTileCount CeilDiv overflow: totalRecv=%lld", totalRecv),
                    return ge::GRAPH_FAILED);
    int64_t sortTempSize =
        (maxSortCount * sizeof(int32_t) + Mc2Kernel::UB_ALIGN - 1) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    int64_t sortTileCount = (maxSortCount + 4096 - 1) / 4096;
    int64_t sortTileCountByCore = static_cast<int64_t>(tilingData.aivNum);
    if (sortTileCountByCore > sortTileCount) {
        sortTileCount = sortTileCountByCore;
    }
    // 镜像 Kernel 侧 engram_fetch_grad_sort.h 常量（CG-4.2）：HISTOGRAM_BINS/SIMT_SLOT_ALIGN/
    // MAX_SINGLE_CORE_ELEMENTS。当前以注释耦合，Tiling UT 中应增加与 Kernel 常量相等的断言
    constexpr int64_t SORT_HISTOGRAM_BINS = 256; // 镜像 sort.h HISTOGRAM_BINS
    constexpr int64_t SIMT_SLOT_ALIGN = 32;      // 镜像 sort.h SIMT_SLOT_ALIGN
    constexpr int64_t SIMT_TILE_ELEMENTS = 8192; // 镜像 sort.h MAX_SINGLE_CORE_ELEMENTS

    int64_t wsSortTempVal = sortTempSize;
    int64_t wsSortTempIdx = sortTempSize;
    int64_t wsSortCompanion = sortTempSize;
    int64_t wsSortHist = sortTileCount * SORT_HISTOGRAM_BINS * sizeof(int32_t);
    int64_t wsSortPrefix = SORT_HISTOGRAM_BINS * sizeof(int32_t);
    int64_t wsSortCoreSums = static_cast<int64_t>(tilingData.aivNum) * SORT_HISTOGRAM_BINS * sizeof(int32_t);
    int64_t wsSortOffsets = sortTileCount * SORT_HISTOGRAM_BINS * sizeof(int32_t);
    // SIMT gather staging: mirrors EngramFetchGradSort::SimtStagingPerCore — per-core GM
    // slot area x2 (values + indices), each tileElements*4B + 256*28B group pads + 32B
    // guard, 32B aligned. tileElements must match the device's MAX_SINGLE_CORE_ELEMENTS:
    // the single-core path spills a full array (2*count int32) to core 0's staging, so the
    // per-core area can no longer be shrunk by the dynTile reduction.
    int64_t simtTileElements = SIMT_TILE_ELEMENTS;
    int64_t simtStagingOneArea = (simtTileElements * static_cast<int64_t>(sizeof(int32_t)) +
                                  SORT_HISTOGRAM_BINS * (SIMT_SLOT_ALIGN - static_cast<int64_t>(sizeof(int32_t))) +
                                  SIMT_SLOT_ALIGN + Mc2Kernel::UB_ALIGN - 1) /
                                 Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    int64_t wsSortSimtStaging = static_cast<int64_t>(tilingData.aivNum) * simtStagingOneArea * 2;
    int64_t wsSortTotal = wsSortTempVal + wsSortTempIdx + wsSortCompanion + wsSortHist + wsSortPrefix + wsSortCoreSums +
                          wsSortOffsets + wsSortSimtStaging;

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

static ge::graphStatus EngramFetchGradTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("engram_fetch_grad_tiling", "failed to get tiling context."),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "nodeName"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Enter EngramFetchGrad tiling func.");

    EngramFetchGradTilingData *tilingData = context->GetTilingData<EngramFetchGradTilingData>();
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
