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
 * \file quant_matmul_all_reduce_tiling_310_general.cc
 * \brief
 */
#include "quant_matmul_all_reduce_tiling_310_general.h"
#include "../../../op_kernel/matmul_all_reduce_tiling_key.h"
#include "mc2_log.h"
#include "common/utils/op_mc2.h"
using namespace Mc2Log;
using namespace Mc2Tiling;

namespace optiling {
bool QuantMatmulAllReduceTiling310General::IsCapable()
{
    if (args_.aType == matmul_tiling::DataType::DT_INT8 && args_.bType == matmul_tiling::DataType::DT_INT8) {
        OP_LOGI(opName_, "start with 310p quant tiling.");
        return true;
    }
    OP_LOGI(opName_, "skip 310p quant tiling as dtype not support");
    return false;
}

ge::graphStatus QuantMatmulAllReduceTiling310General::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckA8W8());
    MC2_CHECK_LOG_RET(opName_, CheckCommModeA2());
    DoRCSTiling();
    DoSplitMTiling();
    MC2_CHECK_LOG_RET(opName_, DoQuantTiling());
    DoAllReduceTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantMatmulAllReduceTiling310General::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(ASCEND_310P), static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_QUANT_MATMUL),
        MATMUL_ALLREDUCE_EMPTY_INPUT_F, MATMUL_ALLREDUCE_INT8_COMM_F, static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(SET_NOT_USE_PARAM), SET_NOT_USE_FM_MM_TPL_TILING, SET_NOT_USE_QUANT_MM_TPL_TILING,
        SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING);

    OP_LOGI(opName_, "QuantMatmulAllReduceTiling310General get tilingKey %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus QuantMatmulAllReduceTiling310General::GetWorkspaceSize()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::GetWorkspaceSize());
    myWorkSpaceSize_ = myWorkSpaceSize_ + (workspaceSize_ - libApiWorkSpaceSize_);
    OP_LOGI(opName_, " set max workspace size %lu to context", myWorkSpaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTiling310General::PostTiling()
{
    size_t tilingDataSize = sizeof(QuantMatmulAllReduceTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu and context capacity size: %zu ", tilingDataSize,
            context_->GetRawTilingData()->GetCapacity());
    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&quantMatmulAllReduceTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    if (MutableRCSTilingData().rankID == 0) {
        PrintRCSTilingData(context_->GetNodeName(), MutableRCSTilingData());
        PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTileTilingData());
        PrintMc2MsgData(context_->GetNodeName(), MutableMc2MsgData());
        if (MutableRCSTilingData().tailM > 0) {
            OP_LOGD(opName_, "have tail");
            PrintTCubeTilingData(context_->GetNodeName(), MutableTCubeTailTilingData());
        }
    }

    context_->SetBlockDim(args_.aicCoreNum);

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    ret = context_->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(opName_, ret);
    return ge::GRAPH_SUCCESS;
}

Mc2Tiling::Mc2Msg &QuantMatmulAllReduceTiling310General::MutableMc2MsgData()
{
    return quantMatmulAllReduceTilingData_.msg;
}

Mc2Tiling::RCSTiling &QuantMatmulAllReduceTiling310General::MutableRCSTilingData()
{
    return quantMatmulAllReduceTilingData_.param;
}

AscendC::tiling::TCubeTiling &QuantMatmulAllReduceTiling310General::MutableTCubeTileTilingData()
{
    return quantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
}

AscendC::tiling::TCubeTiling &QuantMatmulAllReduceTiling310General::MutableTCubeTailTilingData()
{
    return quantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus QuantMatmulAllReduceTiling310General::DoQuantTiling()
{
    args_.mValue = tileMValue_;
    QuantTilingTransferHelper mmTile(*this, quantMatmulAllReduceTilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        return mmTile.DoTiling();
    } else {
        MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
        if (MutableRCSTilingData().tailCnt == 0) {
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        QuantTilingTransferHelper mmTail(*this, quantMatmulAllReduceTilingData_.tailmatmulTiling);
        return mmTail.DoTiling();
    }
}

CutResult QuantMatmulAllReduceTiling310General::GetTilingResult()
{
    CutResult mCutAllreduceResult;
    SocVersion_2201 inputSocVer = SocVersion_2201::SOC910_B;
    SetMCutSocVersion(inputSocVer);
    const gert::StorageShape *commQuantScaleShapeOne = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleShapeTwo = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQuantScaleShapeOne != nullptr) && (commQuantScaleShapeTwo != nullptr)) { // low-bit comm
        OP_LOGD(opName_, "TileCnt enter comm quant.");
        MMPlusQuantAllReduce quantAllReduceTilingHcclObj(args_, args_.rankDim, KernelType::ALL_REDUCE, npuArch_);
        quantAllReduceTilingHcclObj.GetTiling();
        mCutAllreduceResult = quantAllReduceTilingHcclObj.tilingM_.cutRes;
    } else {
        MMPlusAllReduce allReduceTilingHcclObj(args_, args_.rankDim, KernelType::ALL_REDUCE, npuArch_, isPerBlock_);
        allReduceTilingHcclObj.GetTiling();
        mCutAllreduceResult = allReduceTilingHcclObj.tilingM_.cutRes;
    }
    return mCutAllreduceResult;
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce, QuantMatmulAllReduceTiling310General,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND310P), 1);
} // namespace optiling
