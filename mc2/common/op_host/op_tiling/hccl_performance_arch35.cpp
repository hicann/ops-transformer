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
 * \file hccl_performance_arch35.cpp
 * \brief
 */
#include "hccl_performance_arch35.h"
const static string DEFAULT_KEY_FOR_ARCH35 = "0_0_0_0";

const static map<string, HCCLFittingParameters> ARCH35_FITTING_PARAMETER_MAP = {
    {DEFAULT_KEY_FOR_ARCH35,
     HCCLFittingParameters{0.032, 1.371, 2.59568, 5.8887, 0, 2.86702, 5.608, 5.6997, 9.448, 0, 0, 0}},
    {"4_1_0_4", HCCLFittingParameters{0, 0, 5.841842, 8.590666, 0, 0, // A5 allGather standard_card 4p
                                      0, 0, 0, 0, 0, 0}},
    {"4_4_0_4", HCCLFittingParameters{0, 0, 5.8087, 5.6810, 0, 0, // A5 reduceScatter(alltoAll) standard_card 4p
                                      0, 0, 0, 0, 0, 0}},
    {"4_5_0_4", HCCLFittingParameters{0, 0, 13.3265, 36.1964, 0, 0, // A5 allreduce(two shot) standard_card 4p
                                      0, 0, 0, 0, 0, 0}},
    {"4_4_1_8", HCCLFittingParameters{0, 0, 2.642023, 7.700559, 0, 0, // A5 RS(A2A) POD/Server 8p
                                      0, 0, 0, 0, 0, 0}},
    {"4_3_0_4", HCCLFittingParameters{0, 0, 5.8416, 5.8148, 0, 0, // A5 alltoall standard_card 4p
                                      0, 0, 0, 0, 0, 0}},
    {"4_3_1_8", HCCLFittingParameters{0, 0, 2.5386, 7.6265, 0, 0, // A5 alltoall pod 8p
                                      0, 0, 0, 0, 0, 0}},
    {"4_5_1_8", HCCLFittingParameters{0, 0, 6.1588, 34.8712, 0, 0, // A5 allreduce(two shot) pod 8p
                                      0, 0, 0, 0, 0, 0}},
    {"4_0_1_8", HCCLFittingParameters{0, 0, 9.2419, 22, 0, 0, 0, 0, 0, 0, 0, 0}}, // A5 allreduce 8p
};

string HCCLPerformanceArch35::GetCommInfoString() const
{
    return to_string(GetFittingSeriesCode()) + "_" + to_string(static_cast<int>(commTypeInfo_.kernelType)) + "_" +
           to_string(static_cast<int>(commTypeInfo_.topoType)) + "_" +
           to_string(static_cast<int>(commTypeInfo_.rankDim));
};

void HCCLPerformanceArch35::GetCommArch35Parameters()
{
    keyToFittingMap_ = GetCommInfoString();
    if (ARCH35_FITTING_PARAMETER_MAP.find(keyToFittingMap_) != ARCH35_FITTING_PARAMETER_MAP.end()) {
        commEstimatePar_ = ARCH35_FITTING_PARAMETER_MAP.at(keyToFittingMap_);
    } else {
        commEstimatePar_ = ARCH35_FITTING_PARAMETER_MAP.at(DEFAULT_KEY_FOR_ARCH35);
        ChangeCommTimeFactorByDivision(static_cast<double>(FITTING_RANK) / static_cast<double>(commTypeInfo_.rankDim));
    }
}

void HCCLPerformanceArch35::InitArch35Parameters()
{
    if ((commTypeInfo_.kernelType == KernelType::ALL_GATHER) ||
        (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL)) {
        lookUpTileNum_ = commTypeInfo_.rankDim;
        commTimeFactor_ = 1;
    }
}

uint64_t HCCLPerformanceArch35::InverseCommTime(double targetTime) const
{
    double tmpSize = commEstimatePar_.sizeToTimeBoundary1;
    if (targetTime > commEstimatePar_.timeToSizeBoundary2) {
        tmpSize = (targetTime - commEstimatePar_.sizeToTimeLinearOffset) / commEstimatePar_.sizeToTimeLinearGradient;
    } else if (targetTime > commEstimatePar_.timeToSizeBoundary1) {
        if (commTypeInfo_.kernelType == KernelType::ALL_REDUCE && commEstimatePar_.timeToSizeParabolicPar2 != 0.0f) {
            tmpSize =
                (targetTime - commEstimatePar_.timeToSizeParabolicPar3) / commEstimatePar_.timeToSizeParabolicPar2;
        } else {
            tmpSize = (-sqrt(commEstimatePar_.timeToSizeParabolicPar1 *
                             (targetTime + commEstimatePar_.timeToSizeParabolicPar2)) +
                       commEstimatePar_.timeToSizeParabolicPar3);
            if (commEstimatePar_.timeToSizeParabolicPar3 < 0) {
                tmpSize = sqrt(commEstimatePar_.timeToSizeParabolicPar1 *
                               (targetTime + commEstimatePar_.timeToSizeParabolicPar2)) +
                          commEstimatePar_.timeToSizeParabolicPar3;
            }
        }
    }
    return static_cast<uint64_t>(tmpSize * ONE_MBYTE /
                                 (commTypeInfo_.commMatrixLen * lookUpTileNum_ * GetRealDtypeSizes()));
}
namespace {
constexpr uint64_t NPUARCH_9201_FITTING_CODE = 5;
// 键为平台_通信原语_拓扑_卡数，表项用于 AICPU FullMesh、非确定性 ReduceScatter。
// 参数由 FP16 实测拟合；BF16 按相同的 2 字节通信量复用；其余 rankDim 走 GetComm960dtParameters 的缺项保留原配置。
// X 为各 rank 接收量之和（MiB），T 为单次通信耗时（μs）。
const std::map<std::string, HCCLFittingParameters> HCCL_960DT_FITTING_PARAMETER_MAP = {
    {"5_1_0_2", HCCLFittingParameters{0.0078125, 32, 5.49045011445352, 11.4003457546524, 0.00050722240215114,
                                      5.55673747054181, 8.75975462002455, 8.80316666247156, 187.094749417165,
                                      1971.52175408456, 15210.0728450306, -5477.614402455}},
    {"5_1_0_4", HCCLFittingParameters{0.0625, 32, 1.5471712160617, 28.2841478772818, 0.0057304930492253,
                                      1.92633328083789, 10.2829369220371, 10.4033551368279, 77.7936267912563,
                                      174.505054174211, 151.603652808899, -168.077446765101}},
    {"5_2_0_2", HCCLFittingParameters{0.00390625, 32, 5.48806912105473, 12.2764507360356, 0.00355437046224819,
                                      5.4284198946809, 10.5455506266561, 10.5667554461051, 187.894662609787,
                                      281.343773987894, 2062.09592456222, -763.626069980244}},
    {"5_2_0_4", HCCLFittingParameters{0.00048828125, 64, 1.45473124307579, 23.0171326875506, 0.00319889827026339,
                                      1.38838392604621, 14.1606736624445, 14.161351585046, 116.119932244401,
                                      312.607627849842, 136.485717942312, -217.009702833079}},
};
} // namespace

std::string HCCLPerformance960dt::GetCommInfoString() const
{
    return std::to_string(NPUARCH_9201_FITTING_CODE) + "_" +
           std::to_string(static_cast<int>(commTypeInfo_.kernelType)) + "_" +
           std::to_string(static_cast<int>(commTypeInfo_.topoType)) + "_" + std::to_string(commTypeInfo_.rankDim);
}

bool HCCLPerformance960dt::GetComm960dtParameters()
{
    const auto key = GetCommInfoString();
    const auto row = HCCL_960DT_FITTING_PARAMETER_MAP.find(key);
    if (row == HCCL_960DT_FITTING_PARAMETER_MAP.end()) {
        return false;
    }
    keyToFittingMap_ = key;
    commEstimatePar_ = row->second;
    SetCommTimeFactor(1.0);
    return true;
}
