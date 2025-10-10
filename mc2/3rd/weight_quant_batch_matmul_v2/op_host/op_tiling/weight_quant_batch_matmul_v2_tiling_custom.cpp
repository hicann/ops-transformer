/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_tiling_custom.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_custom.h"

#include "weight_quant_batch_matmul_v2_compute_matmul_tiling.h"
#include "weight_quant_batch_matmul_v2_white_list.h"
#include "tiling_base/tiling_key.h"

using Ops::Transformer::OpTiling::RecursiveSum;

namespace optiling {

constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr int32_t MAX_REPEAT_TIMES = 255;
constexpr uint32_t CUSTOM_NZ_TRANS_BASE_N = 64;
constexpr uint32_t CUSTOM_NZ_NO_TRANS_BASE_N = 32;
constexpr uint32_t CUSTOM_NZ_TRANS_BF16_BASE_K = 256;
constexpr uint32_t CUSTOM_NZ_NO_TRANS_BF16_BASE_N = 544;
constexpr uint32_t CUSTOM_NZ_TRANS_FP16_BASE_K = 384;
constexpr uint32_t CUSTOM_NZ_NO_TRANS_FP16_BASE_K = 864;
constexpr int32_t TILING_COMPENSATION_FACTOR = 2;
constexpr uint32_t CUSTOM_NZ_GROUP_BASE_N = 48U;

void WeightQuantBatchMatmulV2TilingCustom::Reset()
{
    cubeBaseN_ = static_cast<uint64_t>(BLOCK_CUBE);
}

/*
The function is limite of custom
1. not support antiquant scale dtype is uint64/int64
*/
bool WeightQuantBatchMatmulV2TilingCustom::IsCapable()
{
    OP_LOGI(opName_, "Begin check custom");
    OP_TILING_CHECK(
        ((matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64) ||
         (matmulInfoPtr_->antiQuantScaleDtype == ge::DT_INT64)),
        OP_LOGI(opName_, "Custom do not support antiquant scale dtype is uint64 and int64"), return false);
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ && matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP) {
        OP_TILING_CHECK(
            matmulInfoPtr_->groupSize != 64 && matmulInfoPtr_->groupSize != 128,
            OP_LOGI(
                opName_, "Custom Nz only support group_size = 64 or 128 for per-group scene, but is [%lu]",
                matmulInfoPtr_->groupSize),
            return false);
        OP_TILING_CHECK(
            matmulInfoPtr_->kSize % matmulInfoPtr_->groupSize != 0,
            OP_LOGI(
                opName_,
                "Custom Nz only support kSize align to group_size for per-group scene, "
                "but kSize is [%lu], group_size is [%lu]",
                matmulInfoPtr_->kSize, matmulInfoPtr_->groupSize),
            return false);
        OP_TILING_CHECK(
            matmulInfoPtr_->kSize % 64 != 0 && matmulInfoPtr_->nSize % 64 != 0,
            OP_LOGI(
                opName_,
                "Custom Nz only support kSize and nSize align to 64 for per-group scene, "
                "but kSize is [%lu], nSize is [%lu]",
                matmulInfoPtr_->kSize, matmulInfoPtr_->nSize),
            return false);
        OP_TILING_CHECK(
            matmulInfoPtr_->transB, OP_LOGI(opName_, "Custom Nz cannot support weight transpose for per-group scene"),
            return false);
        OP_TILING_CHECK(
            matmulInfoPtr_->kSize > MAX_SHAPE_DIM || matmulInfoPtr_->nSize > MAX_SHAPE_DIM,
            OP_LOGI(opName_, "Custom Nz only support and n < 65536 and k < 65536"), return false);
    }
    OP_LOGI(opName_, "Check custom succ");
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::DoOpTiling()
{
    OP_TILING_CHECK(
        InstantiateTilingData() == ge::GRAPH_FAILED,
        OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);
    // Set shape dim and pad of tiling date
    SetShapeSize();
    OP_TILING_CHECK(
        !GetMatMulTiling(),
        OP_LOGE(
            opName_, "failed to get mm tiling for mnk[%lu, %lu, %lu]", matmulInfoPtr_->mSize, matmulInfoPtr_->nSize,
            matmulInfoPtr_->kSize),
        return ge::GRAPH_FAILED);

    uint64_t defaultVecSingleN = 0;
    uint64_t defaultVecSingleK = 0;
    if (matmulInfoPtr_->groupSize > 0) {
        ComputeGroupDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    } else {
        ComputeDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    }

    uint64_t vecSingleN = std::min(defaultVecSingleN, tilingData_->get_nAlign());
    uint64_t vecSingleK = std::min(defaultVecSingleK, tilingData_->get_kAlign());

    tilingData_->set_vecSingleN(static_cast<uint32_t>(vecSingleN));
    tilingData_->set_vecSingleK(static_cast<uint32_t>(vecSingleK));

    uint64_t totalCubeSingleN = cubeBaseN_ * tilingData_->get_cubeBlockDimN();
    totalCubeSingleN = std::min(totalCubeSingleN, tilingData_->get_nAlign());
    tilingData_->set_vecSingleNLoop(ops::CeilDiv(totalCubeSingleN, vecSingleN));
    tilingData_->set_vecSingleNTailLoop(
        ops::CeilDiv(CalcTailSize(matmulInfoPtr_->nSize, cubeBaseN_ * tilingData_->get_cubeBlockDimN()), vecSingleN));
    tilingData_->set_vecSingleKLoop(ops::CeilDiv(matmulInfoPtr_->kSize, vecSingleK));

    tilingData_->set_vecBlockDimK(1);
    uint64_t taskNum = tilingData_->get_vecSingleNLoop() * tilingData_->get_vecSingleKLoop();
    uint64_t singleCoreVecLoop = ops::CeilDiv(taskNum, static_cast<uint64_t>(compileInfoPtr_->aivNum));
    tilingData_->set_vecBlockDimN(ops::CeilDiv(taskNum, singleCoreVecLoop));
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2TilingCustom::SetShapeSize()
{
    tilingData_->set_groupSize(matmulInfoPtr_->groupSize);
    uint64_t weightBlockAlignSize = GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->transB) {
        tilingData_->set_kAlign(ops::CeilAlign(matmulInfoPtr_->kSize, weightBlockAlignSize));
        if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
            tilingData_->set_nAlign(ops::CeilAlign(matmulInfoPtr_->nSize, static_cast<uint64_t>(BLOCK_CUBE)));
        } else {
            tilingData_->set_nAlign(matmulInfoPtr_->nSize);
        }
        tilingData_->set_kPadSize(static_cast<uint8_t>(tilingData_->get_kAlign() - matmulInfoPtr_->kSize));
    } else {
        if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
            tilingData_->set_kAlign(ops::CeilAlign(matmulInfoPtr_->kSize, static_cast<uint64_t>(BLOCK_CUBE)));
        } else {
            tilingData_->set_kAlign(matmulInfoPtr_->kSize);
        }
        tilingData_->set_nAlign(ops::CeilAlign(matmulInfoPtr_->nSize, weightBlockAlignSize));
        tilingData_->set_nPadSize(static_cast<uint8_t>(tilingData_->get_nAlign() - matmulInfoPtr_->nSize));
    }
    tilingData_->set_mSize(matmulInfoPtr_->mSize);
    tilingData_->set_nSize(matmulInfoPtr_->nSize);
    tilingData_->set_kSize(matmulInfoPtr_->kSize);
    // weightquantbmmv2 not support batch dims
    tilingData_->set_haveBatchA(0);
    tilingData_->set_haveBatchB(0);
    tilingData_->set_shapeBatch(1);
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        OP_TILING_CHECK(
            isOutTilingData_, OP_LOGE(opName_, "The out incoming tilingData is nullptr"),
            return ge::GRAPH_FAILED);
        tilingDataManager_ = std::unique_ptr<WeightQuantBatchMatmulV2TilingData>(
            new (std::nothrow) WeightQuantBatchMatmulV2TilingData());
        tilingData_ = tilingDataManager_.get();
    }
    OP_TILING_CHECK(
        tilingData_ == nullptr, OP_LOGE(opName_, "failed to instantiate tilingData"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingData_->GetDataSize(),
        OP_LOGE(
            opName_, "tiling data capacity %zu < actual tiling data size %zu",
            context_->GetRawTilingData()->GetCapacity(), tilingData_->GetDataSize()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2TilingCustom::GetMatMulTiling()
{
    if (!GetTilingFromCache() && !InvokeCacheTiling()) {
        auto mmInputDtype = GetMatmulTilingDtype(matmulInfoPtr_->aDtype);
        auto mmOutputDtype = GetMatmulTilingDtype(matmulInfoPtr_->cDtype);
        matmul_tiling::MultiCoreMatmulTiling mmTiling;
        matmul_tiling::CubeFormat bCubeFormat = (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) ?
                                                    matmul_tiling::CubeFormat::NZ :
                                                    matmul_tiling::CubeFormat::ND;
        mmTiling.SetAType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmInputDtype, matmulInfoPtr_->transA);
        mmTiling.SetBType(matmul_tiling::TPosition::GM, bCubeFormat, mmInputDtype, matmulInfoPtr_->transB);
        mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmOutputDtype);
        mmTiling.SetBias(matmulInfoPtr_->hasBias);
        if (matmulInfoPtr_->hasBias) {
            auto mmBiasDtype = GetMatmulTilingDtype(matmulInfoPtr_->biasDtype);
            mmTiling.SetBiasType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, mmBiasDtype);
        }
        mmTiling.SetDim(compileInfoPtr_->aicNum);
        // 转置场景内轴256对齐
        uint64_t kAlignSize = !matmulInfoPtr_->transB ?
                                  tilingData_->get_kAlign() :
                                  ops::CeilAlign(tilingData_->get_kSize(), static_cast<uint64_t>(256));
        if (kAlignSize >= MAX_SHAPE_DIM) {
            kAlignSize = tilingData_->get_kSize();
        }
        mmTiling.SetOrgShape(matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize, kAlignSize);
        mmTiling.SetShape(matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize);
        mmTiling.SetSingleRange(-1, -1, -1, -1, -1, matmulInfoPtr_->kSize);
        mmTiling.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        OP_TILING_CHECK(
            mmTiling.GetTiling(tilingData_->matmulTiling) == -1,
            OP_LOGE(matmulInfoPtr_->opName, "failed to get matmul tiling"), return false);

        auto mDim =
            ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_singleCoreM()));
        auto nDim =
            ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_singleCoreN()));
        OP_TILING_CHECK(
            mDim * nDim != static_cast<uint64_t>(tilingData_->matmulTiling.get_usedCoreNum()),
            OP_LOGE(
                matmulInfoPtr_->opName, "mDim(%lu) * nDim(%lu) != usedCoreNum(%d)", mDim, nDim,
                tilingData_->matmulTiling.get_usedCoreNum()),
            return false);
        tilingData_->set_cubeBlockDimN(static_cast<uint8_t>(nDim));
        tilingData_->set_cubeBlockDimM(static_cast<uint8_t>(mDim));
    }
    AdjustMatmulTiling();

    uint64_t singleCoreN = ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->get_cubeBlockDimN()));
    tilingData_->matmulTiling.set_singleCoreN(tilingData_->matmulTiling.get_baseN());
    cubeBaseN_ = static_cast<uint64_t>(tilingData_->matmulTiling.get_baseN());
    auto nDim = ops::CeilDiv(matmulInfoPtr_->nSize, ops::CeilAlign(singleCoreN, cubeBaseN_));
    tilingData_->set_cubeBlockDimN(static_cast<uint8_t>(nDim));
    return true;
}

void WeightQuantBatchMatmulV2TilingCustom::AdjustMatmulTiling() const
{
    if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ || matmulInfoPtr_->transB) {
        return;
    }
    int32_t baseN = tilingData_->matmulTiling.get_baseN();
    int32_t minCubeBaseN = ONE_BLK_SIZE;
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        minCubeBaseN = minCubeBaseN << 1;
    }
    if ((baseN * tilingData_->get_cubeBlockDimN()) % minCubeBaseN != 0) {
        tilingData_->matmulTiling.set_baseN(std::max(ops::FloorAlign(baseN, minCubeBaseN), minCubeBaseN));
        int32_t baseK = tilingData_->matmulTiling.get_baseK();
        if (tilingData_->matmulTiling.get_baseN() > baseN) {
            // baseN小于32，被向上对齐了，K要相应缩小并且向下对齐到16
            tilingData_->matmulTiling.set_baseK(
                std::max(
                    ops::FloorAlign(
                        tilingData_->matmulTiling.get_baseK() / TILING_COMPENSATION_FACTOR,
                        static_cast<int32_t>(BLOCK_CUBE)),
                    static_cast<int32_t>(BLOCK_CUBE)));
        }
        if (baseK == tilingData_->matmulTiling.get_baseK()) {
            // kl0没有缩小，就要缩小kL1; 如果stepKb为1时无法调整stepKb，改成调整stepN
            if (tilingData_->matmulTiling.get_stepKb() == 1) {
                tilingData_->matmulTiling.set_stepN(
                    std::max(tilingData_->matmulTiling.get_stepN() / TILING_COMPENSATION_FACTOR, 1));
            } else {
                tilingData_->matmulTiling.set_stepKb(
                    std::max(tilingData_->matmulTiling.get_stepKb() / TILING_COMPENSATION_FACTOR, 1));
            }
            tilingData_->matmulTiling.set_depthB1(
                std::max(tilingData_->matmulTiling.get_depthB1() / TILING_COMPENSATION_FACTOR, 1));
            if (tilingData_->matmulTiling.get_stepKb() > tilingData_->matmulTiling.get_stepKa() &&
                tilingData_->matmulTiling.get_stepKb() % tilingData_->matmulTiling.get_stepKa() != 0 &&
                tilingData_->matmulTiling.get_stepKb() * baseK < static_cast<int32_t>(tilingData_->get_kSize())) {
                tilingData_->matmulTiling.set_stepKb(
                    ops::FloorAlign(tilingData_->matmulTiling.get_stepKb(), tilingData_->matmulTiling.get_stepKa()));
            }
            if (tilingData_->matmulTiling.get_stepKa() > tilingData_->matmulTiling.get_stepKb() &&
                tilingData_->matmulTiling.get_stepKa() % tilingData_->matmulTiling.get_stepKb() != 0 &&
                tilingData_->matmulTiling.get_stepKa() * baseK < static_cast<int32_t>(tilingData_->get_kSize())) {
                tilingData_->matmulTiling.set_stepKa(
                    ops::FloorAlign(tilingData_->matmulTiling.get_stepKa(), tilingData_->matmulTiling.get_stepKb()));
            }
        } else {
            // kl0缩小了，相应的L1上k一定没全载，stepM和stepN只能为1
            tilingData_->matmulTiling.set_depthB1(
                tilingData_->matmulTiling.get_depthB1() / tilingData_->matmulTiling.get_stepN());
            tilingData_->matmulTiling.set_depthA1(
                tilingData_->matmulTiling.get_depthA1() / tilingData_->matmulTiling.get_stepM());
            tilingData_->matmulTiling.set_stepM(1);
            tilingData_->matmulTiling.set_stepN(1);
        }
        AdjustL1Size();
    }
}

void WeightQuantBatchMatmulV2TilingCustom::AdjustL1Size() const
{
    // 如果调整完之后l1size还是大于l1空间，则缩小stepM和depthA1
    uint64_t a1Length = static_cast<uint64_t>(GetShapeSizeWithDataType(
        tilingData_->matmulTiling.get_baseM() * tilingData_->matmulTiling.get_baseK(), matmulInfoPtr_->aDtype));
    uint64_t b1Length = static_cast<uint64_t>(GetShapeSizeWithDataType(
        tilingData_->matmulTiling.get_baseN() * tilingData_->matmulTiling.get_baseK(), matmulInfoPtr_->aDtype));
    uint64_t aL1Size = a1Length * tilingData_->matmulTiling.get_depthA1();
    uint64_t bL1Size = b1Length * tilingData_->matmulTiling.get_depthB1();
    uint64_t biasL1Size =
        matmulInfoPtr_->hasBias ?
            GetShapeSizeWithDataType(tilingData_->matmulTiling.get_baseN(), matmulInfoPtr_->biasDtype) :
            0;
    uint64_t l1Size = aL1Size + bL1Size + biasL1Size;
    if (l1Size > aicoreParams_.l1Size) {
        tilingData_->matmulTiling.set_stepM(tilingData_->matmulTiling.get_stepM() / TILING_COMPENSATION_FACTOR);
        tilingData_->matmulTiling.set_depthA1(tilingData_->matmulTiling.get_depthA1() / TILING_COMPENSATION_FACTOR);
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeDefaultBlock(uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN)
{
    uint64_t defaultInnerAxis = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
    uint64_t defaultOutterAxis = 32;

    // 非group场景，一次求解即可
    if (matmulInfoPtr_->transB) {
        defaultVecSingleN = defaultOutterAxis;
        // 保证mte2的带宽，根据weight的数据类型，默认载入量取512和1024
        defaultVecSingleK = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
    } else {
        // weight不转置场景，n轴取值为cube一轮计算的n轴
        uint64_t weightInnerAxisAlignSize = ONE_BLK_SIZE / sizeof(matmulInfoPtr_->bDtype);

        if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
            // int4场景, 内轴shape按照2倍的ONE_BLK_SIZE对齐
            weightInnerAxisAlignSize = ONE_BLK_SIZE * 2;
        }
        defaultVecSingleN = std::min(
            defaultInnerAxis, ops::CeilAlign(cubeBaseN_ * tilingData_->get_cubeBlockDimN(), weightInnerAxisAlignSize));
        defaultVecSingleK = defaultOutterAxis;
    }
    ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeGroupDefaultBlock(
    uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN)
{
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        defaultVecSingleK = CUSTOM_NZ_TRANS_BF16_BASE_K;
        if (!matmulInfoPtr_->transB && tilingData_->get_nSize() > INT16_MAX) {
            defaultVecSingleK = matmulInfoPtr_->groupSize;
        }
        if (matmulInfoPtr_->aDtype == ge::DT_BF16 && matmulInfoPtr_->transB) {
            defaultVecSingleN = CUSTOM_NZ_NO_TRANS_BASE_N;
        } else {
            defaultVecSingleN = CUSTOM_NZ_TRANS_BASE_N;
        }
        return;
    }
    uint64_t defaultInnerAxis = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
    uint64_t defaultOutterAxis = 32;
    if (matmulInfoPtr_->transB) {
        uint32_t repeatStrideMax = 255;
        uint32_t repeatAxisMax = repeatStrideMax * (ONE_BLK_SIZE / sizeof(matmulInfoPtr_->aDtype));
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            repeatAxisMax = repeatStrideMax * (ONE_BLK_SIZE / sizeof(float));
        }
        tilingData_->set_repeatAxisMax(repeatAxisMax);
        if (tilingData_->get_kAlign() <= repeatAxisMax ||
            (tilingData_->get_kAlign() > repeatAxisMax && matmulInfoPtr_->groupSize <= repeatAxisMax &&
             tilingData_->get_kAlign() % matmulInfoPtr_->groupSize == 0)) {
            // k轴不会导致repeatStride超过限制，或者kAlign满足groupSize对齐的限制。考虑k全载，避免复杂尾块处理
            defaultVecSingleK = tilingData_->get_kAlign();
            ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        }
        if (defaultVecSingleN == 0) {
            OP_LOGD(
                opName_, "the K axis cannot full load, current defaultVecSingleK: [%lu], groupSize: [%lu].",
                defaultVecSingleK, matmulInfoPtr_->groupSize);
            // k无法全载的情况下，需重新设置k轴载入量, 同时保证mte2的带宽，根据weight的数据类型，默认载入量取512和1024
            defaultVecSingleK = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
            if (defaultVecSingleK >= matmulInfoPtr_->groupSize) {
                defaultVecSingleK = defaultVecSingleK / matmulInfoPtr_->groupSize * matmulInfoPtr_->groupSize;
            }
            ReviseGroupDefaultBlockWithTrans(defaultVecSingleK, defaultVecSingleN);
        }
    } else {
        // weight不转置场景，n轴取值为cube一轮计算的n轴
        uint64_t weightInnerAxisAlignSize = ONE_BLK_SIZE / sizeof(matmulInfoPtr_->bDtype);

        if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
            // int4场景, 内轴shape按照32Byte的2倍对齐
            weightInnerAxisAlignSize = ONE_BLK_SIZE * 2;
        }
        defaultVecSingleN = std::min(
            defaultInnerAxis, ops::CeilAlign(cubeBaseN_ * tilingData_->get_cubeBlockDimN(), weightInnerAxisAlignSize));
        defaultVecSingleK = ops::CeilDiv(defaultOutterAxis, matmulInfoPtr_->groupSize) * matmulInfoPtr_->groupSize;
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        ReviseGroupDefaultBlockWithoutTrans(defaultVecSingleK, defaultVecSingleN);
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ReviseGroupDefaultBlockWithTrans(
    uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN)
{
    for (; defaultVecSingleK > matmulInfoPtr_->groupSize; defaultVecSingleK -= matmulInfoPtr_->groupSize) {
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        if (defaultVecSingleN > 0) {
            // n轴大于0,表明该解为合法解，提前退出
            return;
        }
    }

    for (; defaultVecSingleK >= MIN_GROUP_SIZE; defaultVecSingleK -= MIN_GROUP_SIZE) {
        if (matmulInfoPtr_->groupSize % defaultVecSingleK != 0) {
            // 合适的k轴必须满足groupSize_因子的关系
            continue;
        }
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        if (defaultVecSingleN > 0) {
            // 求得一个合法解，提前退出
            return;
        }
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ReviseGroupDefaultBlockWithoutTrans(
    uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN)
{
    while (defaultVecSingleN > 0) {
        // 若groupSize比MAX_REPEAT_TIMES大，则k对齐到groupSize后必定不满足小于MAX_REPEAT_TIMES的要求，
        // 因此排除这种情况下对k的修正
        if (matmulInfoPtr_->groupSize < MAX_REPEAT_TIMES && defaultVecSingleK >= matmulInfoPtr_->groupSize) {
            // 不转置场景下，k在向groupSize取整后应保证小于MAX_REPEAT_TIMES
            defaultVecSingleK =
                std::min(MAX_REPEAT_TIMES / matmulInfoPtr_->groupSize, defaultVecSingleK / matmulInfoPtr_->groupSize) *
                matmulInfoPtr_->groupSize;
            return;
        }
        for (uint32_t targetK = matmulInfoPtr_->groupSize; targetK >= MIN_GROUP_SIZE; targetK -= MIN_GROUP_SIZE) {
            // 合法的k值在不转置场景下应满足小于MAX_REPEAT_TIMES的限制
            if (targetK > MAX_REPEAT_TIMES) {
                continue;
            }

            // 合法的k值需要满足为groupSize的因子
            if (matmulInfoPtr_->groupSize % targetK != 0) {
                continue;
            }
            if (targetK <= defaultVecSingleK) {
                defaultVecSingleK = targetK;
                return;
            }
        }

        // 无法搜索到满足条件的k值，尝试缩小n重新搜索
        defaultVecSingleN = defaultVecSingleN >> 1;
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeVectorDefaultBlock(
    uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN)
{
    /*
        整体vec处理的基本块推导应该满足如下公式：antiquantBufferSize + weightBufferSize < ubSize
        group场景，固定k轴求n，基本公式化简为：n = ubSize * gs / (antiquantCoefficient * k + weightCoefficient * k *
        gs) 非group场景，固定k轴求n，基本公式进一步化简为：n = ub / (antiquantCoefficient + weightCoefficient * k)
        int4场景，weightCoefficient涉及除2操作，因此先放大weightCoefficient 2倍再除2。避免浮点数的系数影响
    */
    if (matmulInfoPtr_->groupSize > 0 || matmulInfoPtr_->bDtype == ge::DT_INT4) {
        ComputeInt4VectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    } else {
        ComputeInt8VectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeInt4VectorDefaultBlock(
    uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN)
{
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            defaultVecSingleN = 64;  // 固定切分，n取64
            defaultVecSingleK = 256; // 固定切分，k取256
        } else {
            if (matmulInfoPtr_->transB) {
                defaultVecSingleN = 64;  // 固定切分，n取64
                defaultVecSingleK = 384; // 固定切分，k取384
            } else {
                defaultVecSingleN = 64;  // 固定切分，n取64
                defaultVecSingleK = 256; // 固定切分，k取256
            }
        }
        return;
    }
    uint64_t start = 0;
    uint64_t length = matmulInfoPtr_->kSize + 1;
    if (matmulInfoPtr_->transB) {
        length = matmulInfoPtr_->nSize + 1;
    }

    // 固定内轴的情况下，二分求解最大的外轴是多少
    while (length > 0) {
        uint64_t mid = start + (length >> 1);
        uint64_t antiquantBuffer;
        uint64_t weightBuffer;
        if (matmulInfoPtr_->transB) {
            antiquantBuffer = ComputeAntiquantBuffer(defaultVecSingleK, mid);
            weightBuffer = ComputeWeightBuffer(defaultVecSingleK, mid);
        } else {
            antiquantBuffer = ComputeAntiquantBuffer(mid, defaultVecSingleN);
            weightBuffer = ComputeWeightBuffer(mid, defaultVecSingleN);
        }

        if (aicoreParams_.ubSize < antiquantBuffer + weightBuffer) {
            length = length >> 1;
        } else {
            start = mid + 1;
            length = length - (length >> 1) - 1;
        }
    }

    // start是不满足条件的最小值，因此最终结果需要-1
    if (matmulInfoPtr_->transB) {
        defaultVecSingleN = start - 1;
    } else {
        defaultVecSingleK = start - 1;
    }
}

uint64_t WeightQuantBatchMatmulV2TilingCustom::ComputeAntiquantBuffer(
    uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN)
{
    uint64_t aDtypeBlockSize = GetBlockAlignSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t antiquantSize = ops::CeilAlign(defaultVecSingleN, aDtypeBlockSize);
    if (matmulInfoPtr_->groupSize > 0) {
        if (matmulInfoPtr_->transB) {
            if (defaultVecSingleK >= matmulInfoPtr_->kSize) {
                // 全载场景，antiquant的n*gourpCount合并成一根轴计算
                antiquantSize = ops::CeilAlign(
                    ops::CeilDiv(defaultVecSingleK, matmulInfoPtr_->groupSize) * defaultVecSingleN, aDtypeBlockSize);
            } else {
                // 非全载场景，antiquant的shape只能当作(n, gourpCount)计算，同时考虑内轴对齐
                antiquantSize =
                    defaultVecSingleN *
                    ops::CeilAlign(ops::CeilDiv(defaultVecSingleK, matmulInfoPtr_->groupSize), aDtypeBlockSize);
            }
        } else {
            // 不转置场景，antiquant的shape只能当作(gourpCount，n)计算，同时考虑内轴对齐
            antiquantSize = ops::CeilDiv(defaultVecSingleK, matmulInfoPtr_->groupSize) *
                            ops::CeilAlign(defaultVecSingleN, aDtypeBlockSize);
        }
    }

    // scale和offset两个入参，需要占用2份空间
    uint64_t antiquantParamsCount = 2;
    uint64_t antiquantInQueSize = antiquantParamsCount * antiquantSize * sizeof(matmulInfoPtr_->aDtype);
    if (matmulInfoPtr_->transB) {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            return antiquantInQueSize + antiquantSize * sizeof(float) +
                   antiquantParamsCount * antiquantSize * ONE_BLK_SIZE;
        } else {
            return antiquantInQueSize + antiquantParamsCount * antiquantSize * ONE_BLK_SIZE;
        }
    } else {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            return antiquantInQueSize + antiquantSize * sizeof(float) +
                   antiquantParamsCount * antiquantSize * sizeof(float);
        } else {
            return antiquantInQueSize + antiquantParamsCount * antiquantSize * sizeof(matmulInfoPtr_->aDtype);
        }
    }
}

uint64_t WeightQuantBatchMatmulV2TilingCustom::ComputeWeightBuffer(
    uint64_t defaultVecSingleK, uint64_t defaultVecSingleN)
{
    uint64_t originWeightAlignAxis = ONE_BLK_SIZE / sizeof(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        // int4场景，内轴的长度是32Byte的2倍
        originWeightAlignAxis = ONE_BLK_SIZE * 2;
    }

    uint64_t weightShape;
    if (matmulInfoPtr_->transB) {
        weightShape = defaultVecSingleN * ops::CeilAlign(defaultVecSingleK, originWeightAlignAxis);
    } else {
        weightShape = defaultVecSingleK * ops::CeilAlign(defaultVecSingleN, originWeightAlignAxis);
    }
    uint64_t originWeightSize = weightShape * sizeof(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        originWeightSize = originWeightSize >> 1;
    }
    uint64_t weight16Size = weightShape * sizeof(matmulInfoPtr_->aDtype);
    uint64_t weight32Size = weightShape * sizeof(float);
    // 输出的buffer共有2份，方便开db
    uint64_t weightOutSize = 2 * weight16Size;

    if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
        return originWeightSize + weight16Size + weight32Size + weightOutSize;
    } else {
        return originWeightSize + weight16Size + weightOutSize;
    }
}

void WeightQuantBatchMatmulV2TilingCustom::ComputeInt8VectorDefaultBlock(
    uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN) const
{
    if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ) {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            if (matmulInfoPtr_->transB) {
                // 需要11nk + 76n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=32
                defaultVecSingleN = 32;
                defaultVecSingleK = BASIC_BLOCK;
            } else {
                // 需要11nk + 12n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=32
                defaultVecSingleK = 32;
                defaultVecSingleN = BASIC_BLOCK;
            }
        } else {
            if (matmulInfoPtr_->transB) {
                // 需要7nk + 68n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=40
                defaultVecSingleN = 40;
                defaultVecSingleK = BASIC_BLOCK;
            } else {
                // 需要7nk + 4n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=48
                defaultVecSingleK = 48;
                defaultVecSingleN = BASIC_BLOCK;
            }
        }
    } else {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            if (matmulInfoPtr_->transB) {
                defaultVecSingleN = CUSTOM_NZ_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_TRANS_BF16_BASE_K;
            } else {
                defaultVecSingleN = CUSTOM_NZ_NO_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_NO_TRANS_BF16_BASE_N;
            }
        } else {
            if (matmulInfoPtr_->transB) {
                defaultVecSingleN = CUSTOM_NZ_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_TRANS_FP16_BASE_K;
            } else {
                defaultVecSingleN = CUSTOM_NZ_NO_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_NO_TRANS_FP16_BASE_K;
            }
        }
    }
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::DoLibApiTiling()
{
    uint64_t cubeBlockDimN = static_cast<uint64_t>(tilingData_->get_cubeBlockDimN());
    uint64_t cubeEachCoreN = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, cubeBlockDimN), cubeBaseN_);
    tilingData_->set_cubeSingleNLoop(ops::CeilDiv(cubeEachCoreN, cubeBaseN_));
    tilingData_->set_cubeSingleNTailLoop(
        ops::CeilDiv(matmulInfoPtr_->nSize - cubeEachCoreN * (cubeBlockDimN - 1), cubeBaseN_));
    tilingData_->set_cubeTailM(
        CalcTailSize(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_singleCoreM())));
    tilingData_->set_cubeTailN(
        CalcTailSize(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_baseN())));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::GetWorkspaceSize()
{
    // weight的缓存最多使用3份空间，实际划分少于3时以实际划分为准
    uint64_t weightCacheCount = std::min(static_cast<uint32_t>(3), tilingData_->get_cubeSingleNLoop());
    uint64_t weightCacheNSize = tilingData_->matmulTiling.get_singleCoreN() * tilingData_->get_cubeBlockDimN();
    if (!matmulInfoPtr_->transB && matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        weightCacheNSize = ops::CeilAlign(weightCacheNSize, static_cast<uint64_t>(ONE_BLK_SIZE));
    }
    uint64_t weightCacheSize = tilingData_->get_kAlign() * weightCacheNSize;
    if (matmulInfoPtr_->transB) {
        // 内轴需256对齐以提高nd2nz效率
        weightCacheSize = ops::CeilAlign(tilingData_->get_kSize(), static_cast<uint64_t>(256)) *
                          tilingData_->matmulTiling.get_singleCoreN() * tilingData_->get_cubeBlockDimN();
    }
    // 向256对齐，可以保证workspace起始地址保证512B对齐，提升mte3性能
    uint64_t weightCacheAlignSize = ops::CeilDiv(weightCacheSize, static_cast<uint64_t>(256)) * 256;
    workspaceSize_ = weightCacheAlignSize * weightCacheCount * ge::GetSizeByDataType(matmulInfoPtr_->aDtype) +
                     compileInfoPtr_->workspaceNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingCustom::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_->GetDataSize());

    OP_TILING_CHECK(
        tilingData_->GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", tilingData_->GetDataSize()),
        return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());
    uint32_t usedAicNum = tilingData_->get_cubeBlockDimM() * tilingData_->get_cubeBlockDimN();
    uint32_t usedAivNum = tilingData_->get_vecBlockDimK() * tilingData_->get_vecBlockDimN();
    context_->SetBlockDim(
        std::max(usedAicNum, CalcTschBlockDim(usedAivNum, compileInfoPtr_->aicNum, compileInfoPtr_->aivNum)));

    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = workspaceSize_;

    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantBatchMatmulV2TilingCustom::GetTilingKey() const
{
    KernelTemplateType templateType = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ ?
                                          KernelTemplateType::WEIGHT_NZ :
                                          KernelTemplateType::CUSTOM_ANTIQUANT;
    return RecursiveSum(
        matmulInfoPtr_->transA, matmulInfoPtr_->transB, matmulInfoPtr_->antiQuantType,
        matmulInfoPtr_->hasAntiQuantOffset, matmulInfoPtr_->quantType, templateType);
}

bool WeightQuantBatchMatmulV2TilingCustom::GetTilingFromCache()
{
    return false;
}

bool WeightQuantBatchMatmulV2TilingCustom::CheckCacheTiling()
{
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        int32_t kAL1Loop = ops::CeilDiv(
            tilingData_->matmulTiling.get_singleCoreK(),
            tilingData_->matmulTiling.get_baseK() * tilingData_->matmulTiling.get_stepKa());
        int32_t kBL1Loop = ops::CeilDiv(
            tilingData_->matmulTiling.get_singleCoreK(),
            tilingData_->matmulTiling.get_baseK() * tilingData_->matmulTiling.get_stepKb());
        if (kAL1Loop == 0 || kBL1Loop == 0) {
            return false;
        }
        if (kAL1Loop % kBL1Loop != 0 && kBL1Loop % kAL1Loop != 0) {
            return false;
        }
    }
    // 拦截分核数小于0.5倍总核数的解
    OP_TILING_CHECK(
        tilingData_->get_cubeBlockDimM() * tilingData_->get_cubeBlockDimN() < 0.5 * compileInfoPtr_->aicNum,
        OP_LOGI(opName_, "Current cache tiling result is aborted for insufficient core use"), return false);

    OP_LOGD(opName_, "get and convert cache tiling success");
    return true;
}

bool WeightQuantBatchMatmulV2TilingCustom::InvokeCacheTiling()
{
    MatmulMultiCoreResult multiCoreResult;
    bool result = ComputeMatmulTiling::GetTiling(
        tilingData_->matmulTiling, multiCoreResult,
        {matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, matmulInfoPtr_->aDtype,
         matmulInfoPtr_->bDtype, matmulInfoPtr_->cDtype, matmulInfoPtr_->biasDtype, matmulInfoPtr_->transA,
         matmulInfoPtr_->transB, matmulInfoPtr_->hasBias, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
         matmulInfoPtr_->quantType, true},
        aicoreParams_, context_);

    OP_LOGI_IF_RETURN(
        !result, false, opName_, "cannot get tiling from cachetiling, mnk[%lu, %lu, %lu]", matmulInfoPtr_->mSize,
        matmulInfoPtr_->kSize, matmulInfoPtr_->nSize);

    tilingData_->set_cubeBlockDimM(static_cast<uint8_t>(multiCoreResult.mDim));
    tilingData_->set_cubeBlockDimN(static_cast<uint8_t>(multiCoreResult.nDim));
    tilingData_->set_blockBatch(static_cast<uint8_t>(multiCoreResult.batchDim));

    return CheckCacheTiling();
}
} // namespace optiling
