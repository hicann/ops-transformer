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
 * \file incre_flash_attention_tiling_context.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CONTEXT_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CONTEXT_H_
#include "../op_kernel/incre_flash_attention_tilingdata.h"
#ifdef ASCENDC_OP_TEST
#define IFA_EXTERN_C extern "C"
#else
#define IFA_EXTERN_C
#endif
namespace optiling {
struct RequiredParaInfo {
    void* data = nullptr;
    at::IntArrayRef shape;
    at::ScalarType dType = at::ScalarType::Float;

    int64_t GetShapeSize() const
    {
        if (shape.empty()) {
            return 0;
        }
        int64_t n = 1;
        for (auto d : shape) {
            n *= d;
        }
        return n;
    }
};

struct OptionalTensorParaInfo {
    bool hasValue = false;                        // optional 是否有值
    void* data = nullptr;                         // 无值时为 nullptr
    at::IntArrayRef shape;                        // 无值时通常为空
    at::ScalarType dType = at::ScalarType::Float; // 无值时保留默认类型

    int64_t GetShapeSize() const
    {
        if (!hasValue || shape.empty()) {
            return 0;
        }
        int64_t n = 1;
        for (auto d : shape) {
            n *= d;
        }
        return n;
    }
};

struct IFAContext {
    const char* opName = nullptr;
    RequiredParaInfo query;
    RequiredParaInfo key;
    RequiredParaInfo value;
    OptionalTensorParaInfo pseShift;
    OptionalTensorParaInfo attenMask;
    OptionalTensorParaInfo actualSeqLengthsQ;
    OptionalTensorParaInfo actualSeqLengths;
    OptionalTensorParaInfo deqScale1;
    OptionalTensorParaInfo quantScale1;
    OptionalTensorParaInfo deqScale2;
    OptionalTensorParaInfo quantScale2;
    OptionalTensorParaInfo quantOffset2;
    OptionalTensorParaInfo antiquantScale;
    OptionalTensorParaInfo antiquantOffset;
    OptionalTensorParaInfo blockTable;
    OptionalTensorParaInfo queryPaddingSize;
    OptionalTensorParaInfo kvPaddingSize;
    OptionalTensorParaInfo keyAntiquantScale;
    OptionalTensorParaInfo keyAntiquantOffset;
    OptionalTensorParaInfo valueAntiquantScale;
    OptionalTensorParaInfo valueAntiquantOffset;
    OptionalTensorParaInfo keySharedPrefix;
    OptionalTensorParaInfo valueSharedPrefix;
    OptionalTensorParaInfo actualSharedPrefixLen;
    OptionalTensorParaInfo queryRope;
    OptionalTensorParaInfo keyRope;
    OptionalTensorParaInfo keyRopeAntiquantScale;
    OptionalTensorParaInfo dequantScaleQuery;
    OptionalTensorParaInfo qStartIdx;
    OptionalTensorParaInfo kvStartIdx;

    RequiredParaInfo attenOut;
    RequiredParaInfo lseOut;

    uint32_t numHeads;
    int64_t preToken;
    int64_t nextToken;
    float scaleValue;
    uint32_t kvHeadNums;
    const char* layOut;
    uint32_t blockSize;
    uint32_t innerPrecise;
    int64_t antiquantMode;
    bool softmaxLseFlag;
    int64_t keyAntiquantMode;
    int64_t valueAntiquantMode;
    uint32_t sparseMode;
    int64_t queryQuantMode;
    int64_t pseType;
    int64_t windowSize;

    size_t workSpaceSize = 0;
    std::vector<at::IntArrayRef> kCache;
    std::vector<at::IntArrayRef> vCache;
    uint64_t tilingKey = 0;
    uint32_t numBlocks = 0;
    IncreFlashAttentionTilingDataV2 tilingData;
    uint8_t fdFlag;
    uint8_t layoutVal;
    uint8_t antiquantMode_;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CONTEXT_H_
