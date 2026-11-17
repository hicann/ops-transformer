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
 * \file hccl_performance_arch35.h
 * \brief
 */
#ifndef __HCCL_PERFORMANCE_ARCH35_H__
#define __HCCL_PERFORMANCE_ARCH35_H__

#pragma once
#include <map>
#include <string>
#include "hccl_performance.h"
using namespace std;

class HCCLPerformanceArch35 : public HCCLPerformanceModel {
public:
    // A5专用通信拟合模型:仅服务于DAV_3510平台,A5内蕴于类本身,不再接收档位key
    explicit HCCLPerformanceArch35(uint32_t inputRankDim, KernelType inputKernelType,
                                   TopoType topoType = TopoType::STANDARD_CARD)
        : HCCLPerformanceModel(inputRankDim, inputKernelType, Ops::Base::DAV_3510)
    {
        InitArch35Parameters();
        commTypeInfo_.topoType = topoType;
        GetCommArch35Parameters();
    }

    string GetCommInfoString() const;
    void GetCommArch35Parameters();
    void InitArch35Parameters();
    uint64_t InverseCommTime(double targetTime) const override;
};

// A6(960DT)薄标签模型: 序列号5内聚于本类, 标定表按需加载(缺项保留原配置)
class HCCLPerformance960dt : public HCCLPerformanceModel {
public:
    explicit HCCLPerformance960dt(uint32_t inputRankDim, KernelType inputKernelType,
                                  TopoType topoType = TopoType::STANDARD_CARD)
        : HCCLPerformanceModel(inputRankDim, inputKernelType, Ops::Base::DAV_9201)
    {
        commTypeInfo_.topoType = topoType;
    }

    explicit HCCLPerformance960dt(const HCCLPerformanceModel& model)
        : HCCLPerformanceModel(model)
    {}

    std::string GetCommInfoString() const;
    // 调用方确认引擎、通信类型和算法满足标定条件后加载；缺少表项时保留原配置。
    bool GetComm960dtParameters();
};

#endif // __HCCL_PERFORMANCE_ARCH35_H__
