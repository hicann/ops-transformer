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
 * \file moe_distribute_dispatch_setup_tiling_arch22.cpp
 * \brief
 */

#include "moe_distribute_dispatch_setup_tiling_arch22.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"

namespace {
constexpr uint32_t OP_TYPE_BATCH_WRITE = 18U;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 40U;
} // namespace

namespace optiling {

void MoeDistributeDispatchSetupTilingA3::SetHcommCfg()
{
    OP_LOGD(nodeName_, "MoeDistributeDispatchSetup groupEp = %s", groupEp_.c_str());
    uint32_t opType = OP_TYPE_BATCH_WRITE;
    std::string algConfigStr = "BatchWrite=level0:fullmesh";
    uint32_t reduceType = 0U;
    uint32_t aivNum = tilingData_->moeDistributeDispatchSetupInfo.aivNum;
    uint32_t sdmaUsedStreamPerCore = tilingData_->moeDistributeDispatchSetupInfo.sdmaUsedStreamPerCore;

    OP_LOGD(nodeName_, "QueueNum = %u, aivNum = %u, sdmaUsedStreamPerCore = %u", (LOCAL_STREAM_MAX_NUM / aivNum),
            aivNum, sdmaUsedStreamPerCore);
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp_, opType, algConfigStr, reduceType);
    mc2CcTilingConfig.SetQueueNum(sdmaUsedStreamPerCore);
    mc2CcTilingConfig.SetCommBlockNum(aivNum);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling);
}

bool MoeDistributeDispatchSetupTilingA3::IsCapable()
{
    OP_LOGD(nodeName_, "Do MoeDistributeDispatchSetupTilingA3 tiling.");
    return false;
}

ge::graphStatus MoeDistributeDispatchSetupTilingA3::DoOpTiling()
{
    return MoeDistributeDispatchSetupTilingFuncImpl();
}
} // namespace optiling
