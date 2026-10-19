/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file memory_copy_arch35.h
 * \brief arch92 mxfp8 fp16 路径的 memcopy 汇总引用与格式转换辅助函数
 */
#ifndef MEMORY_COPY_ARCH35_H
#define MEMORY_COPY_ARCH35_H

#include "../../../common/op_kernel/const_def.h"
#include "../../../common/op_kernel/memcopy/attn/gm_layout.h"
#include "../../../common/op_kernel/memcopy/attn/parser.h"
#include "../../../common/op_kernel/memcopy/attn/offset_calculator_v2.h"
#include "../../../common/op_kernel/memcopy/attn/fa_gm_tensor.h"
#include "../../../common/op_kernel/memcopy/attn/fa_l1_tensor.h"
#include "../../../common/op_kernel/memcopy/attn/fa_ub_tensor.h"
#include "../../../common/op_kernel/memcopy/attn/gm_coord.h"
#include "../../../common/op_kernel/memcopy/attn/attn_copy_gm_to_l1.h"
#include "../../../common/op_kernel/memcopy/attn/copy_ub_to_gm.h"

// 格式转换, 暂时放在这里
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <LayOutTypeEnum LAYOUT, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (kvLayoutType == 0) {
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) {
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) {
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ;
    }
}

template <LayOutTypeEnum LAYOUT, bool useDn = false, bool isPerTokenHead = false>
__aicore__ inline constexpr GmFormat GetQueryScaleGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        if constexpr (isPerTokenHead) {
            return GmFormat::NGT;
        }
        if constexpr (!useDn) {
            return GmFormat::NTGD;
        } else {
            return GmFormat::TNGD;
        }
    } else {
        return GmFormat::TNGD;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKeyScaleGmFormat()
{
    if constexpr (kvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ_K_SCALE;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetValueScaleGmFormat()
{
    if constexpr (kvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND2;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline uint64_t SeqLenFromTensorList(__gm__ uint8_t* keyPtr, uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void*)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) { // v100 没有bsnd
        return keyTensorDesc.GetShape(1);                 // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

#endif
