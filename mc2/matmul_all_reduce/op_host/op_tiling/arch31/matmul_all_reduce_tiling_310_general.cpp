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
 * \file matmul_all_reduce_tiling_310_general.cc
 * \brief
 */
#include "matmul_all_reduce_tiling_310_general.h"
#include "common/utils/op_mc2.h"
#include "../../../op_kernel/matmul_all_reduce_tiling_key.h"
namespace optiling {
bool MatmulAllReduceTiling310General::IsCapable()
{
    OP_LOGI(opName_, "start with MatmulAllReduceTiling310General tiling.");
    return true;
}

ge::graphStatus MatmulAllReduceTiling310General::DoOpTiling()
{
    OP_LOGI(opName_, "In 310p, isA16W8_[%d], isA16W4_[%d]", isA16W8_ ? 1 : 0, isA16W4_ ? 1 : 0);
    if (isA16W8_ || isA16W4_) {
        // ND场景的伪量化校验
        MC2_CHECK_LOG_RET(opName_, CheckA16W8());
    } else {
        // 非量化校验
        OP_LOGI(opName_, "In 310p, check not quant scenario.");
        MC2_CHECK_LOG_RET(opName_, CheckA16W16());
    }
    MC2_CHECK_LOG_RET(opName_, CheckCommModeA2());
    DoRCSTiling();
    DoSplitMTiling();
    if (isKZero_) {
        tilingData_.matmulTiling.M = args_.orgMValue;
        tilingData_.matmulTiling.isBias = args_.isBias;
        tilingData_.matmulTiling.usedCoreNum = 1;
        DoAllReduceTiling();
        return ge::GRAPH_SUCCESS;
    }

    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    matmul_tiling::MultiCoreMatmulTiling mmTile(ascendcPlatform);
    enableL2Cache_ = true;
    if (args_.enableSplitK) {
        DoMatmulTiling310(mmTile, MutableTCubeTileTilingData(), tilingData_.tileL2cacheTiling);
    } else {
        args_.mValue = tileMValue_;
        DoMatmulTiling310(mmTile, MutableTCubeTileTilingData(), tilingData_.tileL2cacheTiling);
        if (MutableRCSTilingData().tailCnt > 0) {
            args_.mValue = tailMValue_;
            matmul_tiling::MultiCoreMatmulTiling mmTail(ascendcPlatform);
            DoMatmulTiling310(mmTail, MutableTCubeTailTilingData(), tilingData_.tailL2cacheTiling);
        }
        args_.mValue = tileMValue_;
    }
    int32_t nAligSize = 16;
    OP_LOGI(context_->GetNodeName(), "DoOpTiling isWeightQuant = %d", isWeightQuant_);
    if (args_.orgNValue % nAligSize != 0 && isWeightQuant_) {
        enableL2Cache_ = false;
    }
    DoAllReduceTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t MatmulAllReduceTiling310General::GetTilingKey() const
{
    if (isKZero_) {
        const uint64_t emptyTensorKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_310P), static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_FP_MM),
            static_cast<uint64_t>(isKZero_), MATMUL_ALLREDUCE_INT8_COMM_F, static_cast<uint64_t>(SET_NOT_USE_PARAM),
            static_cast<uint64_t>(SET_NOT_USE_PARAM), SET_NOT_USE_FM_MM_TPL_TILING, SET_NOT_USE_QUANT_MM_TPL_TILING,
            SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING);
        OP_LOGI(opName_, "MatmulAllReduceTiling310General get tilingKey %lu, isKZero_ %lu", emptyTensorKey, isKZero_);
        return emptyTensorKey;
    }

    uint64_t matmulAllReduceType =
        isWeightQuant_ ? MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL : MATMUL_ALLREDUCE_MM_TYPE_FP_MM;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(ASCEND_310P), static_cast<uint64_t>(matmulAllReduceType), MATMUL_ALLREDUCE_EMPTY_INPUT_F,
        MATMUL_ALLREDUCE_INT8_COMM_F, static_cast<uint64_t>(enableL2Cache_), static_cast<uint64_t>(SET_NOT_USE_PARAM),
        SET_NOT_USE_FM_MM_TPL_TILING, SET_NOT_USE_QUANT_MM_TPL_TILING, static_cast<uint64_t>(SET_NOT_USE_PARAM),
        static_cast<uint64_t>(hasAntiQuantOffset_), static_cast<uint64_t>(antiQuantT_), 1UL,
        static_cast<uint64_t>(isTransB_), static_cast<uint64_t>(FORMAT_B_ND));

    OP_LOGI(
        opName_,
        "MatmulAllReduceTiling310General get tilingKey %lu. antiQuantT_ %lu, hasAntiQuantOffset_ %lu, isTransB_ %lu,     \
        enableL2Cache_ %lu, isWeightQuant_ %lu, isKZero_ %lu",
        tilingKey, antiQuantT_, hasAntiQuantOffset_, isTransB_, enableL2Cache_, isWeightQuant_, isKZero_);
    return tilingKey;
}

void MatmulAllReduceTiling310General::DoMatmulTiling310(matmul_tiling::MultiCoreMatmulTiling &mm1,
                                                        AscendC::tiling::TCubeTiling &cubeTiling,
                                                        Mc2Tiling::Mc2L2cacheTilePara &l2cacheTiling)
{
    DoMatmulTiling(mm1, cubeTiling);
    SetTransLength(mm1, cubeTiling);
    DoL2CacheTiling(l2cacheTiling);
    DoWeightAntiQuantTiling();
}

void MatmulAllReduceTiling310General::DoWeightAntiQuantTiling()
{
    const auto weight = context_->GetInputDesc(static_cast<size_t>(ops::MC2InputIdx::K_X2));
    OP_TILING_CHECK(weight == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "weight"), return);
    if (weight->GetDataType() != ge::DT_INT8) {
        OP_LOGD(context_->GetNodeName(), "No anti quant for weight data type %u.", weight->GetDataType());
        return;
    }

    const auto scale = context_->GetOptionalInputShape(static_cast<size_t>(ops::MC2InputIdx::K_SCALE));
    if (scale == nullptr) {
        OP_LOGD(context_->GetNodeName(), "No anti quant scale.");
        return;
    }
    antiQuantT_ = GetAntiQuantType();
    OP_LOGD(context_->GetNodeName(), "anti quant method %d.", static_cast<int32_t>(antiQuantT_));

    hasAntiQuantOffset_ = HasAntiQuantOffset();
    OP_LOGD(context_->GetNodeName(), "Offset flag %u.", hasAntiQuantOffset_);

    isTransB_ = args_.isBTrans;
    OP_LOGD(context_->GetNodeName(), "Weight trans flag %u.", isTransB_);

    isWeightQuant_ = true;
}

void MatmulAllReduceTiling310General::GetL2CacheParm(uint64_t &l2CacheSize, uint64_t &singleMatrixSize,
                                                     uint32_t &tileSize, uint32_t &tileLimit, bool useNewPara)
{
    (void)useNewPara;
    constexpr uint64_t L2CACHE_SIZE_310 = 16;
    l2CacheSize = L2CACHE_SIZE_310;
    constexpr uint64_t SINGLE_MATRIX_SIZE_310 = 12;
    singleMatrixSize = SINGLE_MATRIX_SIZE_310;
    auto weightTensor = context_->GetInputDesc(static_cast<size_t>(ParamValue::WEIGHT));
    OP_TILING_CHECK(weightTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "weight"), return);
    if (weightTensor->GetStorageFormat() == ge::Format::FORMAT_FRACTAL_NZ) {
        constexpr uint32_t TILE_SIZE_310_NZ = 5;
        tileSize = TILE_SIZE_310_NZ;
    } else {
        constexpr uint32_t TILE_SIZE_310 = 8;
        tileSize = TILE_SIZE_310;
    }
    OP_LOGI(context_->GetNodeName(), "GetL2CacheParm tileSize = %u", tileSize);
    constexpr uint32_t TILE_LIMIT_310 = 1;
    tileLimit = TILE_LIMIT_310;
}

void MatmulAllReduceTiling310General::SetTransLength(matmul_tiling::MultiCoreMatmulTiling &mm1,
                                                     AscendC::tiling::TCubeTiling &cubeTiling)
{
    (void)mm1;
    // mdy after api supoort vec nd2nz shareub size cal.
    uint32_t ubTransLen = 128 * 1024;
    cubeTiling.transLength = ubTransLen;
    cubeTiling.shareUbSize = 0;
}

CutResult MatmulAllReduceTiling310General::GetTilingResult()
{
    CutResult mCutAllreduce;
    SocVersion_2201 socVersion_2201 = SocVersion_2201::SOC910_B;
    SetMCutSocVersion(socVersion_2201);
    const gert::StorageShape *commQuantScaleShape1 = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleShape2 = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQuantScaleShape1 != nullptr) && (commQuantScaleShape2 != nullptr)) { // low-bit comm
        OP_LOGD(opName_, "TileCnt enter comm quant.");
        MMPlusQuantAllReduce quantAllReduceTilingHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, npuArch_);
        quantAllReduceTilingHccl.GetTiling();
        mCutAllreduce = quantAllReduceTilingHccl.tilingM_.cutRes;
    } else {
        MMPlusAllReduce allReduceTilingHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, npuArch_, isPerBlock_);
        allReduceTilingHccl.GetTiling();
        mCutAllreduce = allReduceTilingHccl.tilingM_.cutRes;
    }
    return mCutAllreduce;
}

// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce, MatmulAllReduceTiling310General,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND310P), 3);
} // namespace optiling
