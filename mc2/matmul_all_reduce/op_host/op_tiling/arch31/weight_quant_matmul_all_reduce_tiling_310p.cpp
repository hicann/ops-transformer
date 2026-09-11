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
 * \file weight_quant_matmul_all_reduce_tiling_310p.cc
 * \brief
 */
#include "weight_quant_matmul_all_reduce_tiling_310p.h"
#include "../../../op_kernel/matmul_all_reduce_tiling_key.h"
#include "mc2_log.h"
#include "common/utils/op_mc2.h"
using namespace Mc2Log;
using namespace Mc2Tiling;

namespace optiling {
bool WeightQuantMatmulAllReduceTiling310P::IsCapable()
{
    auto weightTensor = context_->GetInputDesc(static_cast<size_t>(ParamValue::WEIGHT));
    OP_TILING_CHECK(weightTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "weight"),
                    return false);
    auto format = weightTensor->GetStorageFormat();
    if (npuArch_ != Ops::Base::DAV_2002 || format == ge::Format::FORMAT_ND) {
        OP_LOGI(opName_, "skip weight quant tiling when is not 310p or is not weight nz[%d].", format);
        return false;
    }
    OP_LOGI(opName_, "sformat is %d.", format);
    if ((args_.aType == matmul_tiling::DataType::DT_FLOAT16 || args_.aType == matmul_tiling::DataType::DT_BFLOAT16) &&
        (args_.bType == matmul_tiling::DataType::DT_INT4 || args_.bType == matmul_tiling::DataType::DT_INT8)) {
        OP_LOGI(opName_, "start with 310p weight quant tiling.");
        return true;
    }
    OP_LOGI(opName_, "skip 310p weight quant tiling as dtype not support");
    return false;
}

void WeightQuantMatmulAllReduceTiling310P::UpdateCommOffset()
{
    auto &&args = MutableMc2MsgData();
    auto debugMode = mc2tiling::Mc2TilingUtils::GetDebugMode();
    // 只通信不计算模式下，如果K < N，sendOff的offset和sendCnt需要根据K计算
    auto columnNum = args_.orgNValue;
    if (debugMode == MC2_DEBUG_ONLY_AICPU && args_.orgKValue < args_.orgNValue) {
        columnNum = args_.orgKValue;
    }
    // AllReduce
    args.sendOff = tileMValue_ * args_.orgNValue * args_.outputDtypeSize;
    args.recvOff = tileMValue_ * columnNum * args_.outputDtypeSize;
    args.sendCnt = tileMValue_ * args_.orgNValue;
    args.recvCnt = tileMValue_ * columnNum;

    // 通信公式化Tiling计算中，可能有多个尾块
    args.tailSendOff = tailMValue_ * args_.orgNValue * args_.outputDtypeSize;
    args.tailRecvOff = tailMValue_ * columnNum * args_.outputDtypeSize;
    args.tailSendCnt = tailMValue_ * args_.orgNValue;
    args.tailRecvCnt = tailMValue_ * columnNum;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckA16W8());
    MC2_CHECK_LOG_RET(opName_, CheckCommModeA2());
    DoRCSTiling();
    DoSplitMTiling();
    weightQuantMatmulAllReduceTilingData_.tilematmulTiling.cubeBlockDimM = 1;
    weightQuantMatmulAllReduceTilingData_.tilematmulTiling.cubeBlockDimN = 1;
    if (isKZero_) {
        MutableTCubeTileTilingData().M = args_.orgMValue;
        MutableTCubeTileTilingData().isBias = args_.isBias;
        MutableTCubeTileTilingData().usedCoreNum = 1;
        DoAllReduceTiling();
        return ge::GRAPH_SUCCESS;
    }
    MC2_CHECK_LOG_RET(opName_, DoWeightQuantTiling());
    DoAllReduceTiling();
    UpdateCommOffset();
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantMatmulAllReduceTiling310P::GetTilingKey() const
{
    if (isKZero_) {
        const uint64_t emptyTensorKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_310P), static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL),
            static_cast<uint64_t>(isKZero_), MATMUL_ALLREDUCE_INT8_COMM_F, static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM), SET_NOT_USE_FM_MM_TPL_TILING, SET_NOT_USE_QUANT_MM_TPL_TILING,
            static_cast<uint64_t>(SET_NOT_USE_PARAM), static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM), static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM), static_cast<uint64_t>(FORMAT_B_NZ));

        OP_LOGI(opName_, "WeightQuantMatmulAllReduceTiling310P get tilingKey %lu. isKZero_ %lu.", emptyTensorKey,
                isKZero_);
        return emptyTensorKey;
    }

    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(ASCEND_310P), static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL),
        MATMUL_ALLREDUCE_EMPTY_INPUT_F, MATMUL_ALLREDUCE_INT8_COMM_F, static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(SET_NOT_USE_PARAM), SET_NOT_USE_FM_MM_TPL_TILING, SET_NOT_USE_QUANT_MM_TPL_TILING,
        static_cast<uint64_t>(SET_NOT_USE_PARAM), weightQuantMatmul310TPLParam_.hasAntiquantOffset,
        weightQuantMatmul310TPLParam_.antiQuantType, weightQuantMatmul310TPLParam_.transA,
        static_cast<uint64_t>(SET_NOT_USE_PARAM), static_cast<uint64_t>(FORMAT_B_NZ));

    OP_LOGI(opName_,
            "WeightQuantMatmulAllReduceTiling310P get tilingKey %lu. AntiQuantType %lu, hasAntiquantOffset %lu, transA "
            "%lu.",
            tilingKey, weightQuantMatmul310TPLParam_.antiQuantType, weightQuantMatmul310TPLParam_.hasAntiquantOffset,
            weightQuantMatmul310TPLParam_.transA);
    return tilingKey;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::GetWorkspaceSize()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::GetWorkspaceSize());
    myWorkSpaceSize_ = myWorkSpaceSize_ + (workspaceSize_ - libApiWorkSpaceSize_);
    OP_LOGI(opName_, " set max workspace size %lu to context", myWorkSpaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::PostTiling()
{
    size_t tilingDataSize = sizeof(WeightQuantMatmulAllReduceNzTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu and context capacity size: %zu ", tilingDataSize,
            context_->GetRawTilingData()->GetCapacity());

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&weightQuantMatmulAllReduceTilingData_), tilingDataSize);
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

    int32_t tile_core_num = weightQuantMatmulAllReduceTilingData_.tilematmulTiling.cubeBlockDimM *
                            weightQuantMatmulAllReduceTilingData_.tilematmulTiling.cubeBlockDimN;
    int32_t tail_core_num = weightQuantMatmulAllReduceTilingData_.tailmatmulTiling.cubeBlockDimM *
                            weightQuantMatmulAllReduceTilingData_.tailmatmulTiling.cubeBlockDimN;
    int32_t max_core_num = tile_core_num > tail_core_num ? tile_core_num : tail_core_num;
    OP_LOGI(opName_, " PostTiling tile_core_num:%d tail_core_num:%d max_core_num:%d", tile_core_num, tail_core_num,
            max_core_num);
    context_->SetBlockDim(max_core_num);

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    ret = context_->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(opName_, ret);

    return ge::GRAPH_SUCCESS;
}

Mc2Tiling::Mc2Msg &WeightQuantMatmulAllReduceTiling310P::MutableMc2MsgData()
{
    return weightQuantMatmulAllReduceTilingData_.msg;
}

Mc2Tiling::RCSTiling &WeightQuantMatmulAllReduceTiling310P::MutableRCSTilingData()
{
    return weightQuantMatmulAllReduceTilingData_.param;
}

AscendC::tiling::TCubeTiling &WeightQuantMatmulAllReduceTiling310P::MutableTCubeTileTilingData()
{
    return weightQuantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
}

AscendC::tiling::TCubeTiling &WeightQuantMatmulAllReduceTiling310P::MutableTCubeTailTilingData()
{
    return weightQuantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling310P::DoWeightQuantTiling()
{
    args_.mValue = tileMValue_;
    OP_LOGI(opName_, "DoWeightQuantTiling tileMValue_:%lu", tileMValue_);
    WeightQuantTilingTransferHelper mmTile(*this, weightQuantMatmulAllReduceTilingData_.tilematmulTiling);
    OP_LOGI(opName_, "DoWeightQuantTiling enableSplitK:%d", args_.enableSplitK);
    if (args_.enableSplitK) {
        auto res = mmTile.DoTiling();
        weightQuantMatmul310TPLParam_ = mmTile.GetWeightQuantMatmul310TPLParam();
        return res;
    } else {
        OP_LOGI(opName_, "DoWeightQuantTiling tailMValue_:%lu", tailMValue_);
        MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
        weightQuantMatmul310TPLParam_ = mmTile.GetWeightQuantMatmul310TPLParam();
        tileTilingKey_ = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_310P), static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL),
            MATMUL_ALLREDUCE_EMPTY_INPUT_F, MATMUL_ALLREDUCE_INT8_COMM_F, static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM), SET_NOT_USE_FM_MM_TPL_TILING, SET_NOT_USE_QUANT_MM_TPL_TILING,
            static_cast<uint64_t>(SET_NOT_USE_PARAM), weightQuantMatmul310TPLParam_.hasAntiquantOffset,
            weightQuantMatmul310TPLParam_.antiQuantType, weightQuantMatmul310TPLParam_.transA,
            static_cast<uint64_t>(SET_NOT_USE_PARAM), static_cast<uint64_t>(FORMAT_B_NZ));
        OP_LOGI(opName_, " tilematmulTiling tilingKey %lu. AntiQuantType %lu, hasAntiquantOffset %lu, transA %lu.",
                tileTilingKey_, weightQuantMatmul310TPLParam_.antiQuantType,
                weightQuantMatmul310TPLParam_.hasAntiquantOffset, weightQuantMatmul310TPLParam_.transA);
        if (MutableRCSTilingData().tailCnt == 0) {
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        WeightQuantTilingTransferHelper mmTail(*this, weightQuantMatmulAllReduceTilingData_.tailmatmulTiling);
        auto res = mmTail.DoTiling();
        weightQuantMatmul310TPLParam_ = mmTile.GetWeightQuantMatmul310TPLParam();
        return res;
    }
}

CutResult WeightQuantMatmulAllReduceTiling310P::GetTilingResult()
{
    CutResult mCutAllreduceOut;
    SocVersion socVersionInput = SocVersion::SOC910_B;
    SetMCutSocVersion(socVersionInput);
    const gert::StorageShape *commQuantScaleFirst = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleSecond = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQuantScaleFirst != nullptr) && (commQuantScaleSecond != nullptr)) { // low-bit comm
        OP_LOGD(opName_, "TileCnt enter comm quant.");
        MMPlusQuantAllReduce quantAllReduceTilingObj(args_, args_.rankDim, KernelType::ALL_REDUCE, socVersionInput);
        quantAllReduceTilingObj.GetTiling();
        mCutAllreduceOut = quantAllReduceTilingObj.tilingM_.cutRes;
    } else {
        MMPlusAllReduce allReduceTilingObj(args_, args_.rankDim, KernelType::ALL_REDUCE, socVersionInput, isPerBlock_);
        allReduceTilingObj.GetTiling();
        mCutAllreduceOut = allReduceTilingObj.tilingM_.cutRes;
    }
    return mCutAllreduceOut;
}

// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce, WeightQuantMatmulAllReduceTiling310P,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND310P), 0);
} // namespace optiling
