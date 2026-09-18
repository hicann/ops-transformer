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
#include "op_host/op_tiling/mc2_fit_based_balance_tiling.h"

constexpr uint64_t FP4_EXPANSION_RATIO = 2;
constexpr double FP4_COMPUTE_PER_CYCLE_RATIO = 4;

class AllGatherMMFitBalanceTiling : public Mc2FitBasedBalanceTiling {
public:
    double frontMMTime_ = 0;

    explicit AllGatherMMFitBalanceTiling(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                         TopoType topoType = TopoType::STANDARD_CARD)
        : Mc2FitBasedBalanceTiling(args, kernelType, topoType)
    {
        commPerf_.SetCommShapeLen(mmInfo_.kValue);
        // mxfp4(DT_FLOAT4_E2M1)每元素占0.5字节，uint64_t无法表达0.5
        // 此处按"1字节打包2个fp4元素"建模：通信侧用SetCommDTypeSize(1)+ExpansionFraction(2)表达0.5字节/元素
        // (复用HCCL现成子字节机制，CommTime/InverseCommTime/GetLinearThresholdLen均经GetRealDtypeSizes()取0.5)；
        // matmul侧inMatrixADtypeSize置1(L2占用按2x高估，避免0导致memUsage/kLength退化为0)。
        if (args.geAType == ge::DataType::DT_FLOAT4_E2M1) {
            commPerf_.SetCommDTypeSize(1);
            commPerf_.SetCommDtypeSizeExpansionFraction(FP4_EXPANSION_RATIO);
            matmulPerf_.mmShapeInfo_.inMatrixADtypeSize = 1;
            matmulPerf_.mmShapeInfo_.inMatrixBDtypeSize = 1;
            mmInfo_.inMatrixADtypeSize = 1;
            mmInfo_.inMatrixBDtypeSize = 1;
            // fp4属量化dtype, 但公共层 SetCalcType 不识别fp4(ConvertGeTypeToMmType无fp4映射→DT_MAX),
            // 致 calcType_=FP16、CUBE_CALC_PER_CYCLE_MAP 查表落空→computesPerCycle 退回默认4096
            // (fp8 拿到 8192), totalMatmulTime 被高估4x。此处把fp4归类为QUANT并覆盖 computesPerCycle
            // =4*COMPUTES_PER_CYCLE(=16384);。GetMachineParameters 已在基类构造时按FP16跑过, 此处覆盖。
            matmulPerf_.calcType_ = MatmulCalcType::QUANT;
            matmulPerf_.mmShapeInfo_.computesPerCycle =
                FP4_COMPUTE_PER_CYCLE_RATIO * MatmulPerformance::COMPUTES_PER_CYCLE;
            mmInfo_.computesPerCycle = matmulPerf_.mmShapeInfo_.computesPerCycle;
        } else {
            commPerf_.SetCommDTypeSize(mmInfo_.inMatrixADtypeSize);
        }
        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
    }

    void EstimateMMCommTime() override;
    void SetShortTileLen() override;
    void SetLongTileLen() override;
    void AdjustLongShortTileLen() override;
};

#endif // __ALL_GATHER_FIT_BALANCE_TILING_H__
