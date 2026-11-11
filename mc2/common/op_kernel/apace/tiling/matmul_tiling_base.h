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
 * \file matmul_tiling_base.h
 * \brief Base tiling class for matmul.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>

#include "matmul_tiling_common.h"
#include "matmul_tiling_data.h"
#include "../utils/apace_constant.h"

class MatmulTilingBase {
public:
    MatmulTilingBase() = default;
    virtual ~MatmulTilingBase() = default;

    virtual void GetTilingData(uint64_t m, uint64_t n, uint64_t k, uint64_t dataTypeSize, bool transA, bool transB,
                               MatmulTilingData& tilingData, bool hasBias = false)
    {
        // Clear the cached state so one tiling object can safely be reused for
        // multiple shapes without leaking the previous decision.
        args_ = {};
        platformInfo_ = {};
        runInfo_ = {};

        if (m == 0UL || n == 0UL || k == 0UL || dataTypeSize == 0UL) {
            tilingData = {};
            return;
        }

        InitCompileInfo();
        InitShapeArgs(m, n, k, dataTypeSize, transA, transB, true, hasBias);
        DoOpTiling(tilingData);
    };

    // Calculate the size of the intermediate space fo GM
    virtual size_t GetWorkSpace()
    {
        return 0;
    };

    void SetOptimizeEnable(bool enable)
    {
        isOpenOptimize_ = enable;
    }

    void SetPlatformInfoPtr(fe::PlatFormInfos* ptr)
    {
        platformInfoPtr_ = ptr;
    }

    // 允许调用方限制计算tiling的核
    void SetCoreLimit(uint32_t aicNum, uint32_t aivNum)
    {
        coreLimitAicNum_ = aicNum;
        coreLimitAivNum_ = aivNum;
        hasCoreLimit_ = true;
    }

protected:
    MatmulPlatformInfo platformInfo_;
    MatmulArgs args_;
    MatmulRunInfo runInfo_;

    bool isOpenOptimize_{true};
    bool hasCoreLimit_{false};
    uint32_t coreLimitAicNum_{0};
    uint32_t coreLimitAivNum_{0};
    fe::PlatFormInfos* platformInfoPtr_{nullptr};

    virtual const char* TilingName() const = 0;
    virtual void DoOpTiling(MatmulTilingData& tilingData) = 0;

private:
    void InitCompileInfo()
    {
        // Query the platform once per tiling request so later decisions can
        // use the real cache sizes and core count of the current device.
        // Prefer the context-injected platform info (UT-friendly); fall back
        // to the global manager singleton for standalone host launchers.
        std::optional<platform_ascendc::PlatformAscendC> ctxPlatform;
        platform_ascendc::PlatformAscendC* ascendcPlatform = nullptr;
        if (platformInfoPtr_ != nullptr) {
            ctxPlatform.emplace(platformInfoPtr_);
            ascendcPlatform = &(*ctxPlatform);
        } else {
            ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        }
        platformInfo_.aicNum = ascendcPlatform->GetCoreNumAic();
        platformInfo_.aivNum = ascendcPlatform->GetCoreNumAiv();
        if (hasCoreLimit_) {
            platformInfo_.aicNum = std::min(coreLimitAicNum_, coreLimitAivNum_);
        }

        if (platformInfo_.aicNum == 0U) {
            platformInfo_.aicNum = 1U;
        }
        if (platformInfo_.aivNum == 0U) {
            platformInfo_.aivNum = 1U;
        }
        platformInfo_.socVersion = ascendcPlatform->GetSocVersion();
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, platformInfo_.btSize);
    }

    void InitShapeArgs(uint64_t m, uint64_t n, uint64_t k, uint64_t dataTypeSize, bool transA, bool transB,
                       bool isHf32 = true, bool hasBias = false)
    {
        args_.m = m;
        args_.n = n;
        args_.k = k;
        args_.isATrans = transA;
        args_.isBTrans = transB;
        args_.dataTypeSize = dataTypeSize;
        args_.isHf32 = dataTypeSize == DATA_SIZE_FP32 ? isHf32 : false;
        args_.hasBias = hasBias;
    }
};
