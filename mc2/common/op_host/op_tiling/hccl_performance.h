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
 * \file hccl_performance.h
 * \brief
 */
#ifndef __HCCL_PERFORMANCE_H__
#define __HCCL_PERFORMANCE_H__

#pragma once
#include "matmul_formulaic_tiling.h"
#include "formulaic_tiling_datatype.h"
#include "op_host/util/op_const_def.h"
constexpr uint64_t HCCL_MIN_TILE_LEN = 64 * ONE_KBYTE;
constexpr uint64_t HCCL_MIN_TILE_LEN_COARSE = 2 * ONE_MBYTE;
constexpr auto DEFAULT_KEY_FOR_FITTING_MAP = "0_0";
constexpr double FULL_MESH_TIME_FACTOR = 2.0;
constexpr uint64_t DOUBLE_RING_RANKTILE = 2;
constexpr uint64_t RING_TOTAL_STEP_PAR1 = 1;
constexpr uint64_t RING_TOTAL_STEP_PAR2 = 2;
constexpr uint64_t RANK_FOUR = 4;
constexpr uint64_t FITTING_RANK = 8;
constexpr double LOCAL_REDUCE_FACTOR = 0.4; // 确定性Local ReduceScatter额外耗时参数

class HCCLPerformanceModel {
public:
    HCCLFittingParameters commEstimatePar_;
    HCCLInfo commTypeInfo_;
    double commTimeFactor_ = 1.0; // allreduce time is 2.0x allgather or reducescatter
    uint64_t lookUpTileNum_ = 1;  // 数据拟合查表参数
    std::string keyToFittingMap_ = DEFAULT_KEY_FOR_FITTING_MAP;
    SocVersion_2201 socVersion_2201 = SocVersion_2201::SOC910_B;
    NpuArch npuArch_ = Ops::Base::DAV_2201; // 目标平台arch:调用点经框架GetCurNpuArch取后传入,3510/2002/2201判定

    // 拟合查表序列号:arch映射原枚举序值(3510→4,2002→1),保持拟合表键兼容;2201系为档位显式序值
    uint64_t GetFittingSeriesCode() const
    {
        if (npuArch_ == Ops::Base::DAV_3510) {
            return NPUARCH_3510_FITTING_CODE;
        } else if (npuArch_ == Ops::Base::DAV_2002) {
            return NPUARCH_2002_FITTING_CODE;
        } else if (npuArch_ == Ops::Base::DAV_2201) { // A2/A3系(910B/910_93/B4):档位显式序值
            return static_cast<uint32_t>(socVersion_2201);
        }
        return static_cast<uint32_t>(socVersion_2201); // 未识别arch兜底,语义同原catch-all
    }

    // arch粗分入口:3510→A5;2002→310P;2201→系列细分(910B/910_93/B4)
    void SetCommParametersBaseArchType()
    {
        if (npuArch_ == Ops::Base::DAV_3510) { // __NPU_ARCH__ == 3510
            InitParametersForFullMesh();
        } else if (npuArch_ == Ops::Base::DAV_2002) { // 310系:310P只支持MatmulAllreduce算子
            SetCommMethodVersion310P();
        } else if (npuArch_ == Ops::Base::DAV_2201) { // A2/A3系:910B/910_93/B4
            SetCommParametersBySocType();
        } else { // 未识别arch兜底,语义同原catch-all
            InitParametersForFullMesh();
        }
    }

    // 系列细分:仅DAV_2201系(910B/910_93/B4)到达,内部不再含arch判定
    void SetCommParametersBySocType()
    {
        if (socVersion_2201 == SocVersion_2201::SOC910_93) {
            InitSOC91093();
        } else { // SOC910_B/SOC910_B4等
            InitParametersForFullMesh();
        }
    }
    // Constructor
    explicit HCCLPerformanceModel(uint32_t inputRankDim, KernelType inputKernelType,
                                  NpuArch npuArch = Ops::Base::DAV_2201,
                                  SocVersion_2201 inputSeries = SocVersion_2201::SOC910_B)
        : socVersion_2201(inputSeries),
          npuArch_(npuArch)
    {
        commTypeInfo_.kernelType = inputKernelType; // 区分哪个MC2算子
        commTypeInfo_.rankDim = std::max(static_cast<uint64_t>(inputRankDim), MIN_COMM_RANKDIM); // 并行维度最小为2
        commTypeInfo_.commDtypeSizeExpansionFraction = 1;
        SetCommParametersBaseArchType();
        SetMaxStepSize();
        keyToFittingMap_ = GetCommMethodString();
        GetCommEstimateParameters();
    };
    std::string GetCommMethodString() const
    {
        return std::to_string(GetFittingSeriesCode()) + "_" +
               std::to_string(static_cast<int>(commTypeInfo_.commMethod));
    }

    void SetCommMethodVersion310P();
    void SetMaxStepSize();
    void SetCommShapeLen(uint64_t len)
    {
        commTypeInfo_.commMatrixLen = len;
    }
    void SetCommDTypeSize(uint64_t dTypeSize)
    {
        commTypeInfo_.commDtypeSize = dTypeSize;
    }
    inline void SetCommDtypeSizeExpansionFraction(uint64_t number)
    {
        commTypeInfo_.commDtypeSizeExpansionFraction = number;
    }
    double GetRealDtypeSizes() const
    {
        return static_cast<double>(commTypeInfo_.commDtypeSize) / commTypeInfo_.commDtypeSizeExpansionFraction;
    }
    uint64_t GetCommDTypeSize()
    {
        return commTypeInfo_.commDtypeSize;
    }
    void SetCommTimeFactor(double factor)
    {
        commTimeFactor_ = factor;
    }
    void ChangeCommTimeFactorByDivision(double factor);
    void SetFullMeshCommTimeFactor();
    void SetRingCommTimeFactor();
    void InitParametersForFullMesh();
    void InitSOC91093();
    void SetLocalReduceFactor();

    uint64_t GetFullMeshRankTileNum();
    uint64_t GetRankTileNum();
    void GetCommEstimateParameters();
    uint64_t GetMaxStepSize();
    uint64_t GetLinearThresholdLen();
    uint64_t GetLinearThresholdLenCoarse();

    // 性能拟合函数
    double CommTime(uint64_t mSize) const;
    virtual uint64_t InverseCommTime(double targetTime) const;
    void FindStepSize();
};

#endif // __HCCL_PERFORMANCE_H__
