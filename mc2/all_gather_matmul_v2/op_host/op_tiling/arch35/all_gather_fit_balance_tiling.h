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
 * \file all_gather_fit_balance_tiling.h
 * \brief
 */
#ifndef __ALL_GATHER_FIT_BALANCE_TILING_H__
#define __ALL_GATHER_FIT_BALANCE_TILING_H__

#pragma once
#include <string>
#include <type_traits>

#include "common/include/op_host/op_const_def_transformer.h"
#include "all_gather_hccl_utils.h"
#include "mc2_log.h"
#include "op_host/op_tiling/mc2_fit_based_balance_tiling.h"

constexpr uint64_t FP4_EXPANSION_RATIO = 2;
constexpr double FP4_COMPUTE_PER_CYCLE_RATIO = 4;
// A6(960DT)公共频率: cube核1.8G。master现状9201在SetCyclePerMicroSec借用A5档位1.65K,
// 此处覆盖为A6实测频率; 待频率收编进MatmulPerformance960dt构造后撤销本覆盖
constexpr double CYCLE_PER_MICRO_SEC_960DT = 1.8 * ONE_KBYTE;
constexpr static double MM_EXPANSION_TIME = 40;
constexpr static double COMM_EXPANSION_TIME = 40;

template <typename MatmulPerf, typename CommPerf>
class AllGatherMMFitBalanceTilingBase : public Mc2FitBasedBalanceTilingBase<MatmulPerf, CommPerf> {
public:
    using Base = Mc2FitBasedBalanceTilingBase<MatmulPerf, CommPerf>;
    using MatmulPerfType = MatmulPerf;
    double frontMMTime_ = 0;

    explicit AllGatherMMFitBalanceTilingBase(const mc2tiling::TilingArgs& args, KernelType kernelType,
                                             TopoType topoType = TopoType::STANDARD_CARD)
        : Base(args, kernelType, topoType)
    {
        this->commPerf_.SetCommShapeLen(this->mmInfo_.kValue);
        // fp4(DT_FLOAT4_E2M1)/hif4(DT_HIFLOAT4)每元素占0.5字节，uint64_t无法表达0.5
        // 此处按"1字节打包2个fp4元素"建模：通信侧用SetCommDTypeSize(1)+ExpansionFraction(2)表达0.5字节/元素
        // (复用HCCL现成子字节机制，CommTime/InverseCommTime/GetLinearThresholdLen均经GetRealDtypeSizes()取0.5)；
        // matmul侧inMatrixADtypeSize置1(L2占用按2x高估，避免0导致memUsage/kLength退化为0)。
        if (args.geAType == ge::DataType::DT_FLOAT4_E2M1 || args.geAType == Ops::Transformer::DT_HIFLOAT4) {
            this->commPerf_.SetCommDTypeSize(1);
            this->commPerf_.SetCommDtypeSizeExpansionFraction(FP4_EXPANSION_RATIO);
            this->matmulPerf_.mmShapeInfo_.inMatrixADtypeSize = 1;
            this->matmulPerf_.mmShapeInfo_.inMatrixBDtypeSize = 1;
            this->mmInfo_.inMatrixADtypeSize = 1;
            this->mmInfo_.inMatrixBDtypeSize = 1;
            // 4bit算力档位: 覆盖 computesPerCycle = 4*COMPUTES_PER_CYCLE(=16384)。
            // fp4经SetCalcType已归类QUANT(8192); hif4经E4M3FN代理同样按QUANT/8192计,
            // 与fp4同为4bit算力档, 此处一并对齐。GetMachineParameters已在基类构造时跑过, 此处覆盖。
            // hif4实测算力待定标(A6)
            this->matmulPerf_.calcType_ = MatmulCalcType::QUANT;
            this->matmulPerf_.mmShapeInfo_.computesPerCycle =
                FP4_COMPUTE_PER_CYCLE_RATIO * MatmulPerformance::COMPUTES_PER_CYCLE;
            this->mmInfo_.computesPerCycle = this->matmulPerf_.mmShapeInfo_.computesPerCycle;
        } else {
            this->commPerf_.SetCommDTypeSize(this->mmInfo_.inMatrixADtypeSize);
        }
        this->tilingM_.SetMinLenByMax(this->matmulPerf_.GetBaseM());
        this->tilingM_.SetAlignLength(this->matmulPerf_.GetBaseM());
    }

    void EstimateMMCommTime() override
    {
        this->matmulPerf_.FindCubeUtil(this->rankTileNum_);
        this->matmulPerf_.GetMatmulGradient();

        // Find total matmul time and comm time
        double totalMatmulTime = this->matmulPerf_.MatmulTime(this->mmInfo_.mValue, this->rankDim_);
        double totalTpTime = this->commPerf_.CommTime(this->mmInfo_.mValue);
        double frontUtil = this->matmulPerf_.FindCubeUtilByL2Usage(this->mmInfo_.mValue, 1);
        if (std::abs(frontUtil) > 1e-10) {
            frontMMTime_ = this->matmulPerf_.MatmulTime(this->mmInfo_.mValue, 1) / frontUtil;
        } else {
            frontMMTime_ = this->matmulPerf_.MatmulTime(this->mmInfo_.mValue, 1);
        }
        this->tilingM_.cutRes.shortTileAtBack = true;
        this->ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));

        uint64_t sizeOfComm = this->mmInfo_.mValue * this->mmInfo_.kValue * (this->rankDim_ - 1) / ONE_MBYTE *
                              this->commPerf_.GetCommDTypeSize();
        OPS_LOG_D("AllGatherMatmul",
                  "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, sizeOfComm %lu, "
                  "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
                  "rankDim_ %lu, rankTile %lu",
                  this->mmInfo_.mValue, this->mmInfo_.nValue, this->mmInfo_.kValue, this->matmulPerf_.cubeUtil_,
                  sizeOfComm, totalMatmulTime, totalTpTime, this->tilingM_.GetMinLen(), this->tilingM_.GetAlignLength(),
                  this->commPerf_.commTimeFactor_, this->rankDim_, this->rankTileNum_);
    }

    void SetLongTileLen() override
    {
        // Long tile should overlap the cost time of local tile
        this->tilingM_.cutRes.longTileLen = this->commPerf_.InverseCommTime(frontMMTime_);
    }

    void SetShortTileLen() override
    {
        // Encourage split more if the comm and calc is balanced and the cost of cutLen is sufficiently high
        bool isCalcCommBalance = this->ratioCalcComm_ < CALC_COMM_RATIO;
        uint64_t cutLen = this->tilingM_.GetAlignLength() / TWO;
        double mmCost = this->matmulPerf_.MatmulTime(cutLen, this->rankDim_);
        double commCost = this->commPerf_.CommTime(cutLen);
        if (isCalcCommBalance && (mmCost > MM_EXPANSION_TIME) && (commCost > COMM_EXPANSION_TIME)) {
            this->tilingM_.SetAlignLength(cutLen);
            this->tilingM_.SetMinLenByMin(cutLen);
        }

        this->tilingM_.cutRes.shortTileLen = this->tilingM_.GetMinLen();
        this->tilingM_.cutRes.numShortTile = 1U;
    }

    void AdjustLongShortTileLen() override
    {
        this->tilingM_.FitTileLengthDiscrete(false, true, true);
        // When the long and short tiles are equal, the long and short pieces become one.
        if (this->tilingM_.cutRes.shortTileLen == this->tilingM_.cutRes.longTileLen) {
            this->tilingM_.cutRes.shortTileLen = 0U;
            this->tilingM_.cutRes.numShortTile = 0U;
            this->tilingM_.cutRes.numLongTile++;
        }
    }
};

// A5(DAV_3510)默认组合, 保持既有符号名不变
using AllGatherMMFitBalanceTiling = AllGatherMMFitBalanceTilingBase<MatmulPerformanceArch35, HCCLPerformanceArch35>;
// A6(DAV_9201)组合: 960dt性能模型(cube 1.8G/带宽7.5T按比例缩放)
using AllGatherMMFitBalanceTilingA6 = AllGatherMMFitBalanceTilingBase<MatmulPerformance960dt, HCCLPerformance960dt>;

template <typename FitBalanceTiling>
CutResult RunAllGatherFitBalanceTiling(const mc2tiling::TilingArgs& args, bool isCcuMode,
                                       const optiling::CommSizeInfo& commSize, const std::string& opName)
{
    FitBalanceTiling tileFormulate(args, KernelType::ALL_GATHER, TopoType::STANDARD_CARD);
    if constexpr (std::is_same_v<typename FitBalanceTiling::MatmulPerfType, MatmulPerformance960dt>) {
        // A6公共频率1.8G(cube核): master现状9201在SetCyclePerMicroSec借用A5档位1.65K, 此处覆盖为A6实测频率
        // 待频率收编进MatmulPerformance960dt构造后撤销本覆盖
        tileFormulate.matmulPerf_.mmShapeInfo_.cyclePerMicroSec = CYCLE_PER_MICRO_SEC_960DT;
        // 加载960dt标定表: 命中5_1_0_2/5_1_0_4→实测曲线+factor=1.0; 缺项(8P+)→保留catch-all原配置
        tileFormulate.commPerf_.GetComm960dtParameters();
    }
    CutResult result = tileFormulate.GetTiling();

    if (isCcuMode) {
        optiling::AdjustCutResultForCCU(result, args.mValue, commSize, opName);
    } else {
        optiling::AdjustCutResultForAICPU(result, args.mValue, tileFormulate.matmulPerf_.GetBaseM(), opName);
    }

    return result;
}

// 按npuArch选择A5/A6性能模型组合
inline CutResult AllGatherFitBalanceTilingByArch(const mc2tiling::TilingArgs& args, NpuArch npuArch, bool isCcuMode,
                                                 const optiling::CommSizeInfo& commSize, const std::string& opName)
{
    if (npuArch == Ops::Base::DAV_9201) {
        return RunAllGatherFitBalanceTiling<AllGatherMMFitBalanceTilingA6>(args, isCcuMode, commSize, opName);
    }
    return RunAllGatherFitBalanceTiling<AllGatherMMFitBalanceTiling>(args, isCcuMode, commSize, opName);
}

#endif // __ALL_GATHER_FIT_BALANCE_TILING_H__
