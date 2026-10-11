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
 * \file moe_inplace_index_add_simt_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "op_host/tiling_templates_registry.h"
#include "moe_inplace_index_add_tiling.h"
#include "moe_inplace_index_add_simt_tiling.h"

namespace optiling {
constexpr int64_t SIMT_DB_BUFFER = 1;

constexpr int64_t GM_ALIGN = 512;
constexpr int64_t USE_UB_MAX_SIZE = 64L * 1024L;

#ifdef DAVID_FPGA
constexpr int64_t MAX_THREAD_NUM = 256;
#else
constexpr int64_t MAX_THREAD_NUM = 1024;
#endif
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16L * 1024L * 1024L;

bool MoeInplaceIndexAddSimtTiling::IsCapable()
{
    if (isSimtNoSort_ == 1) {
        return true;
    }
    return false;
}

ge::graphStatus MoeInplaceIndexAddSimtTiling::DoOpTiling()
{
    usedCoreNum_ = Ops::Base::CeilDiv(updatesAxis_, MAX_THREAD_NUM);
    usedCoreNum_ = std::min(usedCoreNum_, totalCoreNum_);
    ubSize_ = std::min(ubSize_, USE_UB_MAX_SIZE);
    GetCastTypeSize();
    if (dtype_ == ge::DT_INT16 || dtype_ == ge::DT_INT8 || dtype_ == ge::DT_UINT8 || dtype_ == ge::DT_BOOL) {
        int64_t ubLength = ubSize_ / SIMT_DB_BUFFER / (castTypeSize_ + varTypeSize_);
        int64_t oneBlockNum = Ops::Base::GetUbBlockSize(context_) / varTypeSize_;
        ubFactor_ = Ops::Base::FloorAlign(ubLength, oneBlockNum);
    }

    tilingData_.set_preAxis(preAxis_);
    tilingData_.set_varInAxis(varInAxis_);
    tilingData_.set_updatesInAxis(updatesInAxis_);
    tilingData_.set_afterAxis(afterAxis_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_colUbFactor(colUbFactor_);
    tilingData_.set_indicesUbFactor(indicesUbFactor_);
    tilingData_.set_indicesStride(indicesStride_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddSimtTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInplaceIndexAddSimtTiling::GetTilingKey() const
{
    uint64_t tilingKey = 100000;
    uint64_t factorStart = 100;
    uint64_t factor = 10;

    // 后两位 dtype
    tilingKey += static_cast<uint64_t>(dtype_);
    // 百位 SIMT计算类型
    uint64_t hundredDigit = std::max(updatesAxis_, varAxis_) > INT32_MAX ? 1 : 0;
    tilingKey += factorStart * hundredDigit;
    // 千位 alpha是否为none
    tilingKey += factorStart * factor * withAlpha_;
    return tilingKey;
}

ge::graphStatus MoeInplaceIndexAddSimtTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    if (castTypeSize_ != 0) {
        int64_t varWsSize = Ops::Base::CeilAlign(varAxis_ * castTypeSize_, GM_ALIGN);
        workspaceSize_ += varWsSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddSimtTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    tilingKey_ = GetTilingKey();
    context_->SetTilingKey(tilingKey_);
    workspaces[0] = workspaceSize_;
    auto res = context_->SetLocalMemorySize(ubSize_);
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    MOE_OP_TILING_CHECK((res != ge::GRAPH_SUCCESS),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                            "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
                        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeInplaceIndexAddSimtTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "preAxis: " << tilingData_.get_preAxis();
    info << ", varInAxis: " << tilingData_.get_varInAxis();
    info << ", updatesInAxis: " << tilingData_.get_updatesInAxis();
    info << ", afterAxis: " << tilingData_.get_afterAxis();
    info << ", ubFactor: " << tilingData_.get_ubFactor();
    info << ", colUbFactor: " << tilingData_.get_colUbFactor();
    info << ", indicesUbFactor: " << tilingData_.get_indicesUbFactor();
    info << ", indicesStride: " << tilingData_.get_indicesStride();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", tilingKey: " << tilingKey_;
    OP_LOGD(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(MoeInplaceIndexAdd, MoeInplaceIndexAddSimtTiling, 1);
} // namespace optiling
