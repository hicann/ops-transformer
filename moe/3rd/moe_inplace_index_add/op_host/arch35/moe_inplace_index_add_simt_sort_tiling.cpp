/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file moe_inplace_index_add_simt_sort_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "op_host/tiling_templates_registry.h"
#include "moe_inplace_index_add_tiling.h"
#include "moe_inplace_index_add_simt_sort_tiling.h"

namespace optiling {
constexpr int64_t ALIGN_SIZE = 32;
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t FP32_BYTES = 4;
constexpr uint32_t DOUBLE = 2;
constexpr uint32_t FIVE = 5;
constexpr uint64_t SIMD_RESERVED_SIZE = 8UL * 1024UL;
constexpr uint64_t HASH_BUCKET_SIZE = 128 * sizeof(float);

#ifdef DAVID_FPGA
constexpr int64_t MAX_THREAD_NUM = 256;
#else
constexpr int64_t MAX_THREAD_NUM = 2048;
#endif
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16L * 1024L * 1024L;

static const std::set<ge::DataType> VAR_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT64, ge::DT_INT32,
                                                 ge::DT_INT16, ge::DT_INT8,    ge::DT_UINT8, ge::DT_BOOL};

bool MoeInplaceIndexAddSimtSortTiling::IsCapable()
{
    if (isSimtSort_ == 1) {
        return true;
    }
    return false;
}

uint64_t MoeInplaceIndexAddSimtSortTiling::computeIndicesUbFactor()
{
    int64_t mid = 0;
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(indicesAxis_ + 1);
    int64_t sortTmpSize = 0;
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));

    while (end - start > 1) {
        mid = (end + start) / static_cast<int64_t>(DOUBLE);
        int64_t totalIndexSize = 0;
        if (indicesCastMode_ == 0) {
            // 所需空间：indicesLocal + indicesSortedLocal + updatesOriginIdxLocal + uniqueIdCountLocal
            totalIndexSize =
                Ops::Base::CeilAlign(mid * static_cast<int64_t>(indicesTypeSize_), ubBlock) * DOUBLE +
                Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(uint32_t)), ubBlock) + ubBlock * DOUBLE +
                Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(int32_t)), ubBlock) + ubBlock * DOUBLE;
            sortTmpSize = static_cast<int64_t>(GetSortTmpSize(indicesDtype_, static_cast<uint32_t>(mid), false));
        } else {
            // 所需空间：indicesLocal + CastindicesLocal + CastindicesSortedLocal + updatesOriginIdxLocal +
            // uniqueIdCountLocal
            totalIndexSize =
                Ops::Base::CeilAlign(mid * static_cast<int64_t>(indicesTypeSize_), ubBlock) * DOUBLE +
                Ops::Base::CeilAlign(mid * static_cast<int64_t>(indicesCastDtypeSize_), ubBlock) * DOUBLE +
                Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(uint32_t)), ubBlock) + ubBlock * DOUBLE +
                Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(int32_t)), ubBlock) + ubBlock * DOUBLE;
            sortTmpSize = static_cast<int64_t>(GetSortTmpSize(indicesCastDtype_, static_cast<uint32_t>(mid), false));
        }

        sortTmpSize = Ops::Base::CeilAlign(sortTmpSize, ubBlock);
        int64_t updateSize1Pre = Ops::Base::CeilAlign(static_cast<int64_t>(mid * afterAxis_ * varTypeSize_), ubBlock);
        int64_t tmpTotalSize = totalIndexSize + sortTmpSize + updateSize1Pre + HASH_BUCKET_SIZE;
        if (tmpTotalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }
    return start;
}

void MoeInplaceIndexAddSimtSortTiling::TilingForSimtSort()
{
    // 先计算pre中只搬1份updates时，最多搬多少个indices
    indicesUbFactor_ = computeIndicesUbFactor();
    int64_t totalLoop = Ops::Base::CeilDiv(indicesAxis_, indicesUbFactor_);
    // 按照UB总循环次数分核
    indicesLoopSize_ = Ops::Base::CeilDiv(totalLoop, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, indicesLoopSize_);
    tailBlockIndicesLoopSize_ = totalLoop - indicesLoopSize_ * (usedCoreNum_ - 1);
    indiceAxisTailNum_ = indicesAxis_ - indicesUbFactor_ * indicesLoopSize_ * (usedCoreNum_ - 1) -
                         indicesUbFactor_ * (tailBlockIndicesLoopSize_ - 1);
    while (usedCoreNum_ < totalCoreNum_ / DOUBLE && indicesUbFactor_ > 1) {
        indicesUbFactor_ = indicesUbFactor_ / DOUBLE;
        totalLoop = Ops::Base::CeilDiv(indicesAxis_, indicesUbFactor_);
        indicesLoopSize_ = Ops::Base::CeilDiv(totalLoop, totalCoreNum_);
        usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, indicesLoopSize_);
        tailBlockIndicesLoopSize_ = totalLoop - indicesLoopSize_ * (usedCoreNum_ - 1);
        indiceAxisTailNum_ = indicesAxis_ - indicesUbFactor_ * indicesLoopSize_ * (usedCoreNum_ - 1) -
                             indicesUbFactor_ * (tailBlockIndicesLoopSize_ - 1);
    }
    eachCoreIndexCount_ = indicesUbFactor_ * indicesLoopSize_;

    // indices数量固定情况下，增加搬运pre中updates份数，计算每次最多能搬多少份updates
    int64_t totalIndexSize = 0;
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    if (indicesCastMode_ == 0) {
        totalIndexSize = Ops::Base::CeilAlign(indicesUbFactor_ * indicesTypeSize_, ubBlock) * DOUBLE +
                         Ops::Base::CeilAlign(indicesUbFactor_ * static_cast<int64_t>(sizeof(uint32_t)), ubBlock) +
                         ubBlock * DOUBLE +
                         Ops::Base::CeilAlign(indicesUbFactor_ * static_cast<int64_t>(sizeof(int32_t)), ubBlock) +
                         ubBlock * DOUBLE;
        sortShareBufSize_ =
            static_cast<int64_t>(GetSortTmpSize(indicesDtype_, static_cast<uint32_t>(indicesUbFactor_), false));
    } else {
        totalIndexSize = Ops::Base::CeilAlign(indicesUbFactor_ * indicesTypeSize_, ubBlock) * DOUBLE +
                         Ops::Base::CeilAlign(indicesUbFactor_ * indicesCastDtypeSize_, ubBlock) * DOUBLE +
                         Ops::Base::CeilAlign(indicesUbFactor_ * static_cast<int64_t>(sizeof(uint32_t)), ubBlock) +
                         ubBlock * DOUBLE +
                         Ops::Base::CeilAlign(indicesUbFactor_ * static_cast<int64_t>(sizeof(int32_t)), ubBlock) +
                         ubBlock * DOUBLE;
        sortShareBufSize_ =
            static_cast<int64_t>(GetSortTmpSize(indicesCastDtype_, static_cast<uint32_t>(indicesUbFactor_), false));
    }
    normalUpdatesPreNum_ = preAxis_;
    for (int64_t preNum = 2; preNum <= preAxis_; preNum++) { // 前面已计算preNum = 1确保UB空间能容纳，因此preNum从2开始
        int64_t updateSizeNPre =
            Ops::Base::CeilAlign(static_cast<int64_t>(indicesUbFactor_ * afterAxis_ * varTypeSize_), ubBlock) * preNum;
        int64_t tmpTotalSize = totalIndexSize + sortShareBufSize_ + updateSizeNPre + HASH_BUCKET_SIZE;
        if (tmpTotalSize > ubSize_) {
            normalUpdatesPreNum_ = preNum - 1;
            break;
        }
    }
    updatesPreLoop_ = Ops::Base::CeilDiv(preAxis_, normalUpdatesPreNum_);
    tailUpdatesPreNum_ = preAxis_ - normalUpdatesPreNum_ * (updatesPreLoop_ - 1);
}

ge::graphStatus MoeInplaceIndexAddSimtSortTiling::DoOpTiling()
{
    GetCastTypeForSort();
    ubSize_ = ubSize_ - SIMD_RESERVED_SIZE;
    TilingForSimtSort();

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddSimtSortTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInplaceIndexAddSimtSortTiling::GetTilingKey() const
{
    uint64_t factor = 10;
    uint64_t tilingKey = 500000;
    uint64_t clcNumType = std::max(updatesAxis_, varAxis_) > INT32_MAX ? 1 : 0;

    return tilingKey + clcNumType * factor + withAlpha_;
}

ge::graphStatus MoeInplaceIndexAddSimtSortTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddSimtSortTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingKey_ = GetTilingKey();
    context_->SetTilingKey(tilingKey_);
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_);
    context_->SetBlockDim(usedCoreNum_);
    MOE_OP_TILING_CHECK((res != ge::GRAPH_SUCCESS),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                            "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
                        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeInplaceIndexAddSimtSortTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "preAxis: " << tilingData_.get_preAxis();
    info << ", varInAxis: " << tilingData_.get_varInAxis();
    info << ", afterAxis: " << tilingData_.get_afterAxis();
    info << ", updatesInAxis: " << tilingData_.get_updatesInAxis();
    info << ", indicesStride: " << tilingData_.get_indicesStride();
    info << ", tilingKey: " << tilingKey_;
    info << ", usedCoreNum: " << usedCoreNum_;
    OP_LOGD(context_->GetNodeName(), "%s", info.str().c_str());
}

void MoeInplaceIndexAddSimtSortTiling::SetTilingData()
{
    tilingData_.set_preAxis(preAxis_);
    tilingData_.set_varInAxis(varInAxis_);
    tilingData_.set_updatesInAxis(updatesInAxis_);
    tilingData_.set_afterAxis(afterAxis_);
    tilingData_.set_indicesUbFactor(indicesUbFactor_);
    tilingData_.set_indicesLoopSize(indicesLoopSize_);
    tilingData_.set_indiceAxisTailNum(indiceAxisTailNum_);
    tilingData_.set_tailBlockIndicesLoopSize(tailBlockIndicesLoopSize_);
    tilingData_.set_eachCoreIndexCount(eachCoreIndexCount_);
    tilingData_.set_sortShareBufSize(sortShareBufSize_);
    tilingData_.set_normalUpdatesPreNum(normalUpdatesPreNum_);
    tilingData_.set_tailUpdatesPreNum(tailUpdatesPreNum_);
    tilingData_.set_updatesPreLoop(updatesPreLoop_);
    tilingData_.set_indicesStride(indicesStride_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_indicesCastMode(indicesCastMode_);
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeInplaceIndexAdd, MoeInplaceIndexAddSimtSortTiling, 3);
} // namespace optiling
