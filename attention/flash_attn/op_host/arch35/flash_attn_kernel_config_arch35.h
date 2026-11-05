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
 * \file flash_attn_kernel_config_arch35.h
 * \brief FA 分核/切块配置的计算接口
 *
 * 输入 FaKernelParamArch35（算子参数），输出 FaKernelConfigArch35（模板 + 基本块 + 可使用AICORE数）。
 */

#ifndef FLASH_ATTN_KERNEL_CONFIG_ARCH35_H
#define FLASH_ATTN_KERNEL_CONFIG_ARCH35_H

#include <cstdint>
#include <initializer_list>
#include "../flash_attn_kernel_config.h"

namespace optiling {
namespace flash_attn {
namespace fa_tiling_util {

struct FaKernelParamArch35 {
    uint32_t aicNum{0};
    uint32_t cvRatio{0};
    uint32_t batchSize{0};
    uint32_t kvHeads{0};
    uint32_t gSize{0};
    uint32_t qkHeadDim{0};
    uint32_t vHeadDim{0};
    int64_t maskMode{0};
    int64_t maxSeqlenQ{-1};
    int64_t maxSeqlenKv{-1};
    int64_t winLeft{-1};
    int64_t winRight{-1};
};

struct FaKernelConfigArch35 {
    uint32_t templateId{0};  // 0=ND, 1=合轴 DN, 2=不合轴 DN
    uint32_t sOuter{64};     // per-AIV M（× cvRatio = mBase）
    uint32_t sInner{128};    // 基本块 S2
    uint32_t usedCoreNum{1}; // 使用的AICORE数
};

class CalcFaKernelConfigArch35 {
public:
    static void CalcConfig(const FaKernelParamArch35& param, FaKernelConfigArch35& config)
    {
        config.templateId = CalcTemplateId(param);
        CalcBaseSize(param, config);
        config.usedCoreNum =
            GetMaxUsedAicCores(param.aicNum, param.batchSize, param.kvHeads, param.gSize, param.maxSeqlenQ,
                               param.maxSeqlenKv, config.sOuter * param.cvRatio, config.sInner, config.templateId);
    }

private:
    static bool EnableTemplateBn1s1s2SoftmaxDn(const FaKernelParamArch35& param)
    {
        return (param.vHeadDim <= 128 && (param.maskMode == 0 || param.maskMode == 3) && param.maxSeqlenQ >= 256);
    }

    static bool IsAdjustMBaseSizeForTemplateND(const FaKernelParamArch35& param)
    {
        if (param.maxSeqlenQ == -1 || param.maxSeqlenKv == -1) { // 只要存在未传的max_seqlens, 使用默认模板的默认基本块
            return false;
        }
        if (param.vHeadDim >= 256) {
            if (param.gSize * param.maxSeqlenQ < 64) {
                return true;
            }
        } else {
            if (param.maxSeqlenQ <= 64 && param.maxSeqlenKv > 128) {
                if (param.maskMode == 0 || param.maskMode == 3) {
                    return true;
                } else if (param.maskMode == 4) {
                    // winRight不参与判断
                    if ((param.winLeft == -1) || param.winLeft > 128) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    static bool EnableTemplateBn2gs1s2SoftmaxDn(const FaKernelParamArch35& param)
    {
        if (param.vHeadDim >= 256) {
            return false;
        }
        if (param.maskMode != 0) {
            return false;
        }
        // 这个模板仅支持mBaseSize=128 s2BaseSize=128, 如果调整成了mBaseSize=64 s2BaseSize=256就不能走这个模板
        if (IsAdjustMBaseSizeForTemplateND(param)) {
            return false;
        }
        return true;
    }

    static uint32_t CalcTemplateId(const FaKernelParamArch35& param)
    {
        if (param.vHeadDim >= 256) { // VD>=256时, 直接路由到ND合轴模板, 因为其在所有场景性能最优
            return FA_Template_ND;
        }
        if (param.maxSeqlenQ == -1 || param.maxSeqlenKv == -1) { // 只要存在未传的max_seqlens, 使用默认的ND合轴模板
            return FA_Template_ND;
        }
        if (EnableTemplateBn1s1s2SoftmaxDn(param)) {
            return FA_Template_DN_Unmerged;
        }
        if (EnableTemplateBn2gs1s2SoftmaxDn(param)) {
            return FA_Template_DN;
        }
        return FA_Template_ND;
    }

    static void CalcBaseSize(const FaKernelParamArch35& param, FaKernelConfigArch35& config)
    {
        if (config.templateId == FA_Template_ND) {
            config.sOuter = 64;
            config.sInner = 128;
            if (IsAdjustMBaseSizeForTemplateND(param)) {
                config.sOuter = 32;
                config.sInner = 256;
            }
        } else if (config.templateId == FA_Template_DN) {
            config.sOuter = 64;
            config.sInner = 128;
        } else if (config.templateId == FA_Template_DN_Unmerged) {
            // 不合轴 DN: 基本块 M=128(两 AIV 各 64 行) + S2=256, 全档统一
            config.sOuter = 64;
            config.sInner = 256;
        }
    }
};

} // namespace fa_tiling_util
} // namespace flash_attn
} // namespace optiling

#endif // FLASH_ATTN_KERNEL_CONFIG_ARCH35_H
