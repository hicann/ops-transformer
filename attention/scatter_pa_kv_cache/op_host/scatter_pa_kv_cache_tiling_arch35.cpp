/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file scatter_pa_kv_cache_tiling_arch35.cpp
 * \brief
 */

#include <algorithm>
#include "scatter_pa_kv_cache_tiling.h"
#include "tiling/tiling_api.h"

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info_def.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
constexpr int64_t DIM0 = 0;
constexpr int64_t DIM1 = 1;
constexpr int64_t DIM2 = 2;
constexpr int64_t DIM3 = 3;
constexpr int64_t DIM4 = 4;

constexpr int64_t INDEX_INPUT_KEY = 0;
constexpr int64_t INDEX_INPUT_KEY_CACHE_IN = 1;
constexpr int64_t INDEX_SLOT_MAPPING = 2;
constexpr int64_t INDEX_COMPRESS_LENS = 3;
constexpr int64_t INDEX_COMPRESS_SEQ_OFFSET = 4;
constexpr int64_t INDEX_INPUT_SEQ_LENS = 5;
constexpr int64_t INDEX_INPUT_VALUE = 3;
constexpr int64_t INDEX_INPUT_VALUE_CACHE_IN = 4;
constexpr int64_t DUAL_IN_OUT_MODE_INDEX_OFFSET = 2;

constexpr uint64_t HUNDRED = 100;
constexpr uint64_t TEN = 10;

constexpr int64_t FULLY_LOAD = 1;
constexpr int64_t NOT_FULLY_LOAD = 0;

constexpr int64_t INT32_DTYPE_SIZE = 4;
constexpr int64_t INT64_DTYPE_SIZE = 8;

constexpr int64_t TEMPLATE_NORMAL = 1;
constexpr int64_t TEMPLATE_ROPE = 2;
constexpr int64_t TEMPLATE_ALIBI = 3;
constexpr int64_t TEMPLATE_OMNI = 4;
constexpr int64_t TEMPLATE_NZ = 5;
constexpr int64_t TEMPLATE_NCT = 6;
constexpr int64_t TEMPLATE_NORMAL_NON_CONTIGUOUS = 7;
constexpr int64_t TEMPLATE_NZ_NON_CONTIGUOUS = 8;
constexpr int64_t TEMPLATE_NZ_NON_CONTIGUOUS_FIVE_DIM = 9;
constexpr uint64_t INPUT_CACHE_MODE_INDEX = 0;
constexpr uint64_t INPUT_SCATTER_MODE_INDEX = 1;
constexpr uint64_t INPUT_STRIDES_INDEX = 2;
constexpr uint64_t INPUT_OFFSET_INDEX = 3;

constexpr int64_t SINGLE_IN_OUT = 1;
constexpr int64_t DUAL_IN_OUT = 2;
constexpr int64_t DOUBLE_BUFFER = 2;

constexpr int64_t BLOCK_SIZE = 32;

constexpr int64_t MAX_HANLDE_BYTE_SIZE_PER_LOOP = 16384;

constexpr int64_t RESERVED_UB_SIZE = static_cast<int64_t>(2 * 1024);
constexpr uint64_t ASCENDC_TOOLS_WORKSPACE = static_cast<uint64_t>(16) * 1024 * 1024;

bool ScatterPaKvCacheTiling::IsCapable()
{
    return true;
}

ge::graphStatus ScatterPaKvCacheTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_, "platformInfo is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    if (totalCoreNum_ == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "totalCoreNum", "0",
                                              "totalCoreNum must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == static_cast<uint64_t>(0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize", "0", "ubSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    ubSize_ = static_cast<int64_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckSlotMappingShape(int64_t requiredDimNum)
{
    auto inputSlotMapping = context_->GetRequiredInputTensor(inputSlotMapping_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSlotMapping);
    auto inputSlotMappingShape = inputSlotMapping->GetStorageShape();
    size_t inputSlotMappingDimNum = inputSlotMappingShape.GetDimNum();
    if (inputSlotMappingDimNum != static_cast<size_t>(requiredDimNum)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "slot_mapping", std::to_string(inputSlotMappingDimNum).c_str(),
            (std::string("slot_mapping dim num must be ") + std::to_string(requiredDimNum)).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::GetIndexDtype()
{
    auto inputSlotMappingDesc = context_->GetInputDesc(inputSlotMapping_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSlotMappingDesc);
    ge::DataType inputSlotMappingDtype = inputSlotMappingDesc->GetDataType();
    if (inputSlotMappingDtype == ge::DT_INT32) {
        indexDtypeSize_ = INT32_DTYPE_SIZE;
    } else if (inputSlotMappingDtype == ge::DT_INT64) {
        indexDtypeSize_ = INT64_DTYPE_SIZE;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "slot_mapping",
                                              ge::TypeUtils::DataTypeToSerialString(inputSlotMappingDtype).c_str(),
                                              "slot_mapping dtype must be int32 or int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::GetInputDtype()
{
    auto inputKeyDesc = context_->GetInputDesc(inputKey_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKeyDesc);
    ge::DataType inputKeyDtype = inputKeyDesc->GetDataType();

    auto inputKeyCacheInDesc = context_->GetInputDesc(inputKeyCacheIn_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKeyCacheInDesc);
    ge::DataType inputKeyCacheInDtype = inputKeyCacheInDesc->GetDataType();

    if (inOutMode_ == DUAL_IN_OUT) {
        auto inputValueDesc = context_->GetInputDesc(inputValue_);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
        ge::DataType inputValueDtype = inputValueDesc->GetDataType();
        auto inputValueCacheInDesc = context_->GetInputDesc(inputValueCacheIn_);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueCacheInDesc);
        ge::DataType inputValueCacheInDtype = inputValueCacheInDesc->GetDataType();
        if (templateType_ == TEMPLATE_NZ || templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS ||
            templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS_FIVE_DIM) {
            bool inputKeyDtypeFailCheck = inputKeyDtype != inputKeyCacheInDtype;
            if (inputKeyDtypeFailCheck) {
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "key, key_cache",
                                                       (ge::TypeUtils::DataTypeToSerialString(inputKeyDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(inputKeyCacheInDtype))
                                                           .c_str(),
                                                       "pa_nz mode: key and key_cache dtype must be the same");
                return ge::GRAPH_FAILED;
            }
            inputKeyDtypeFailCheck = inputValueDtype != inputValueCacheInDtype;
            if (inputKeyDtypeFailCheck) {
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "value, value_cache",
                                                       (ge::TypeUtils::DataTypeToSerialString(inputValueDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(inputValueCacheInDtype))
                                                           .c_str(),
                                                       "pa_nz mode: value and value_cache dtype must be the same");
                return ge::GRAPH_FAILED;
            }
        } else {
            bool inputKeyDtypeFailCheck = inputKeyDtype != inputValueDtype || inputKeyDtype != inputKeyCacheInDtype;
            inputKeyDtypeFailCheck = inputKeyDtypeFailCheck || inputValueDtype != inputValueCacheInDtype;
            if (inputKeyDtypeFailCheck) {
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "key, value, key_cache, value_cache",
                    (ge::TypeUtils::DataTypeToSerialString(inputKeyDtype) + ", " +
                     ge::TypeUtils::DataTypeToSerialString(inputValueDtype) + ", " +
                     ge::TypeUtils::DataTypeToSerialString(inputKeyCacheInDtype) + ", " +
                     ge::TypeUtils::DataTypeToSerialString(inputValueCacheInDtype))
                        .c_str(),
                    "key, value, key_cache, value_cache dtype must be the same");
                return ge::GRAPH_FAILED;
            }
        }
        valueDtype_ = inputValueDtype;
        if (inputValueDtype == ge::DT_FLOAT4_E2M1 || inputValueDtype == ge::DT_FLOAT4_E1M2) {
            valueDtypeByteSize_ = DIM1;
        } else {
            valueDtypeByteSize_ = static_cast<int64_t>(GetSizeByDataType(inputValueDtype));
        }
        if (valueDtypeByteSize_ <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "valueDtypeByteSize",
                                                  std::to_string(valueDtypeByteSize_).c_str(),
                                                  "get value input dtype bytes failed");
            return ge::GRAPH_FAILED;
        }
        bool inputValueDtypeCheck = valueDtype_ == ge::DT_FLOAT || valueDtype_ == ge::DT_FLOAT16 ||
                                    valueDtype_ == ge::DT_BF16 || valueDtype_ == ge::DT_INT8 ||
                                    valueDtype_ == ge::DT_UINT8 || valueDtype_ == ge::DT_INT16 ||
                                    valueDtype_ == ge::DT_UINT16 || valueDtype_ == ge::DT_INT32 ||
                                    valueDtype_ == ge::DT_UINT32 || valueDtype_ == ge::DT_HIFLOAT8 ||
                                    valueDtype_ == ge::DT_FLOAT8_E5M2 || valueDtype_ == ge::DT_FLOAT8_E4M3FN ||
                                    valueDtype_ == ge::DT_FLOAT4_E2M1 || valueDtype_ == ge::DT_FLOAT4_E1M2;
        if (!inputValueDtypeCheck) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "value",
                                                  ge::TypeUtils::DataTypeToSerialString(valueDtype_).c_str(),
                                                  "input value dtype not supported");
            return ge::GRAPH_FAILED;
        }
    } else if (inOutMode_ == SINGLE_IN_OUT) {
        if (inputKeyDtype != inputKeyCacheInDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "key, key_cache",
                                                   (ge::TypeUtils::DataTypeToSerialString(inputKeyDtype) + ", " +
                                                    ge::TypeUtils::DataTypeToSerialString(inputKeyCacheInDtype))
                                                       .c_str(),
                                                   "key and key_cache dtype must be the same");
            return ge::GRAPH_FAILED;
        }
    }

    inputDtype_ = inputKeyDtype;
    if (inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2) {
        dtypeByteSize_ = DIM1;
    } else {
        dtypeByteSize_ = static_cast<int64_t>(GetSizeByDataType(inputDtype_));
    }
    if (dtypeByteSize_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dtypeByteSize",
                                              std::to_string(dtypeByteSize_).c_str(), "get input dtype bytes failed");
        return ge::GRAPH_FAILED;
    }
    bool inputDtypeCheck = inputDtype_ == ge::DT_FLOAT || inputDtype_ == ge::DT_FLOAT16 || inputDtype_ == ge::DT_BF16 ||
                           inputDtype_ == ge::DT_INT8 || inputDtype_ == ge::DT_UINT8 || inputDtype_ == ge::DT_INT16 ||
                           inputDtype_ == ge::DT_UINT16 || inputDtype_ == ge::DT_INT32 ||
                           inputDtype_ == ge::DT_UINT32 || inputDtype_ == ge::DT_HIFLOAT8 ||
                           inputDtype_ == ge::DT_FLOAT8_E5M2 || inputDtype_ == ge::DT_FLOAT8_E4M3FN ||
                           inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2;
    if (!inputDtypeCheck) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "key",
                                              ge::TypeUtils::DataTypeToSerialString(inputDtype_).c_str(),
                                              "input dtype not supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

int64_t ScatterPaKvCacheTiling::RoundUp(int64_t x, int64_t dtypeSize)
{
    // 入参保证 UbBlockSize 和 dtypeSize 不为0
    if (dtypeSize == 0) {
        OP_LOGD(this->context_, "dtypeSize is 0.");
        return x;
    }
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(this->context_));
    if (ubBlockSize == 0) {
        OP_LOGD(this->context_, "UbBlockSize is 0.");
        return 0;
    }
    int64_t elemNum = ubBlockSize / dtypeSize;
    return (x + elemNum - 1) / elemNum * elemNum;
}

void ScatterPaKvCacheTiling::GetCommonTilingInfo()
{
    numTokens_ = inputKeyShape_.GetDim(DIM0) * inputKeyShape_.GetDim(DIM2);
    blockFactor_ = Ops::Base::CeilDiv<int64_t>(numTokens_, totalCoreNum_);
    usedCoreNum_ = std::min(Ops::Base::CeilDiv<int64_t>(numTokens_, blockFactor_), totalCoreNum_);
    tailBlockFactor_ = numTokens_ - blockFactor_ * (usedCoreNum_ - 1);
    kHeadSize_ = inputKeyShape_.GetDim(DIM3);
    keyStride0_ = inputKeyShape_.GetDim(DIM1) * inputKeyShape_.GetDim(DIM2) * inputKeyShape_.GetDim(DIM3);
    keyStride1_ = inputKeyShape_.GetDim(DIM2) * inputKeyShape_.GetDim(DIM3);
    keyStride2_ = inputKeyShape_.GetDim(DIM3);
    if (inOutMode_ == SINGLE_IN_OUT) {
        vHeadSize_ = DIM0;
        valueStride0_ = DIM0;
        valueStride1_ = DIM0;
        valueStride2_ = DIM0;
    } else if (inOutMode_ == DUAL_IN_OUT) {
        vHeadSize_ = inputValueShape_.GetDim(DIM3);
        valueStride0_ = inputValueShape_.GetDim(DIM1) * inputValueShape_.GetDim(DIM2) * inputValueShape_.GetDim(DIM3);
        valueStride1_ = inputValueShape_.GetDim(DIM2) * inputValueShape_.GetDim(DIM3);
        valueStride2_ = inputValueShape_.GetDim(DIM3);
    }
    batch_ = inputKeyShape_.GetDim(DIM0);
    seqLen_ = inputKeyShape_.GetDim(DIM1);
    numHead_ = inputKeyShape_.GetDim(DIM2);
    numBlocks_ = inputKeyCacheInShape_.GetDim(DIM0);
    blockSize_ = inputKeyCacheInShape_.GetDim(DIM1);
}

void ScatterPaKvCacheTiling::GetNonContigousStrideInfo()
{
    keyStride0_ = keyStride_[DIM0];
    keyStride1_ = keyStride_[DIM1];
    keyStride2_ = keyStride_[DIM2];
    keyCacheStride0_ = keyCacheStride_[DIM0];
    keyCacheStride1_ = keyCacheStride_[DIM1];
    keyCacheStride2_ = keyCacheStride_[DIM2];
    keyCacheStride3_ = keyCacheStride_[DIM3];
    if (inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2) {
        keyStride0_ /= DIM2;
        keyStride1_ /= DIM2;
        keyStride2_ /= DIM2;
        keyCacheStride0_ /= DIM2;
        keyCacheStride1_ /= DIM2;
        keyCacheStride2_ /= DIM2;
        keyCacheStride3_ /= DIM2;
    }

    if (inOutMode_ == SINGLE_IN_OUT) {
        valueStride0_ = DIM0;
        valueStride1_ = DIM0;
        valueStride2_ = DIM0;
    } else if (inOutMode_ == DUAL_IN_OUT) {
        valueStride0_ = valueStride_[DIM0];
        valueStride1_ = valueStride_[DIM1];
        valueStride2_ = valueStride_[DIM2];
        valueCacheStride0_ = valueCacheStride_[DIM0];
        valueCacheStride1_ = valueCacheStride_[DIM1];
        valueCacheStride2_ = valueCacheStride_[DIM2];
        valueCacheStride3_ = valueCacheStride_[DIM3];
        if (valueDtype_ == ge::DT_FLOAT4_E2M1 || valueDtype_ == ge::DT_FLOAT4_E1M2) {
            valueStride0_ /= DIM2;
            valueStride1_ /= DIM2;
            valueStride2_ /= DIM2;
            valueCacheStride0_ /= DIM2;
            valueCacheStride1_ /= DIM2;
            valueCacheStride2_ /= DIM2;
            valueCacheStride3_ /= DIM2;
        }
    }
}

ge::graphStatus ScatterPaKvCacheTiling::TemplateNormal()
{
    if (CheckDimValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckSlotMappingShape(DIM1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    numTokens_ = inputKeyShape_.GetDim(DIM0);
    numHead_ = inputKeyShape_.GetDim(DIM1);
    kHeadSize_ = inputKeyShape_.GetDim(DIM2);
    if (inOutMode_ == DUAL_IN_OUT) {
        vHeadSize_ = inputValueShape_.GetDim(DIM2);
    }
    numBlocks_ = inputKeyCacheInShape_.GetDim(DIM0);
    blockSize_ = inputKeyCacheInShape_.GetDim(DIM1);
    kHandleNumPerCore_ = inputKeyShape_.GetDim(DIM1) * inputKeyShape_.GetDim(DIM2);
    vHandleNumPerCore_ =
        (inOutMode_ == SINGLE_IN_OUT) ? DIM0 : inputValueShape_.GetDim(DIM1) * inputValueShape_.GetDim(DIM2);
    if (inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2) {
        if (inputKeyShape_.GetDim(DIM2) % DIM2 != 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "key",
                                                  std::to_string(inputKeyShape_.GetDim(DIM2)).c_str(),
                                                  "k_head_size must be an even number when input dtype is fp4");
            return ge::GRAPH_FAILED;
        }
        if (inOutMode_ == DUAL_IN_OUT && inputValueShape_.GetDim(DIM2) % DIM2 != 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "value",
                                                  std::to_string(inputValueShape_.GetDim(DIM2)).c_str(),
                                                  "v_head_size must be an even number when input dtype is fp4");
            return ge::GRAPH_FAILED;
        }
        kHandleNumPerCore_ /= DIM2;
        vHandleNumPerCore_ /= DIM2;
        kHeadSize_ /= DIM2;
        vHeadSize_ /= DIM2;
    }
    blockFactor_ = Ops::Base::CeilDiv<int64_t>(numTokens_, totalCoreNum_);
    usedCoreNum_ = std::min(Ops::Base::CeilDiv<int64_t>(numTokens_, blockFactor_), totalCoreNum_);
    tailBlockFactor_ = numTokens_ - blockFactor_ * (usedCoreNum_ - 1);
    GetNonContigousStrideInfo();
    if (templateType_ == TEMPLATE_NORMAL_NON_CONTIGUOUS) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t maxHandleNumPerLoop = ubSize_ / dtypeByteSize_;
    // check whether tail dim can fully load.
    int64_t ubThreshold = std::max(blockFactor_, tailBlockFactor_) *
                              (RoundUp(kHandleNumPerCore_, dtypeByteSize_) +
                               RoundUp(vHandleNumPerCore_, dtypeByteSize_)) + // inputKey & inputValue
                          RoundUp(std::max(blockFactor_, tailBlockFactor_), dtypeByteSize_) * DIM2 * indexDtypeSize_ /
                              dtypeByteSize_; // slotMapping for key and value
    if (ubThreshold <= maxHandleNumPerLoop) {
        // tail dim can fully load
        isFullyLoad_ = FULLY_LOAD;
        OP_LOGD(context_, "tail dim can fully load.");
        return ge::GRAPH_SUCCESS;
    }
    // tail dim can not fully load
    isFullyLoad_ = NOT_FULLY_LOAD;
    kHandleNumPerLoop_ = (ubSize_ - RESERVED_UB_SIZE) / dtypeByteSize_ / DIM2;
    kLoopNum_ = Ops::Base::CeilDiv<int64_t>(kHandleNumPerCore_, kHandleNumPerLoop_);
    if (kLoopNum_ > 0) {
        kTailHandleNum_ = kHandleNumPerCore_ - (kLoopNum_ - 1) * kHandleNumPerLoop_;
        kLoopNum_--;
    }
    if (inOutMode_ == DUAL_IN_OUT) {
        vHandleNumPerLoop_ = (ubSize_ - RESERVED_UB_SIZE) / dtypeByteSize_ / DIM2;
        vLoopNum_ = Ops::Base::CeilDiv<int64_t>(vHandleNumPerCore_, vHandleNumPerLoop_);
        if (vLoopNum_ > 0) {
            vTailHandleNum_ = vHandleNumPerCore_ - (vLoopNum_ - 1) * vHandleNumPerLoop_;
            vLoopNum_--;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::TemplateNZ()
{
    if (CheckDimValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckSlotMappingShape(DIM1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    numBlocks_ = inputKeyCacheInShape_.GetDim(DIM0);
    blockSize_ = inputKeyCacheInShape_.GetDim(DIM2);
    if (inputKeyCacheInShape_.GetDimNum() != DIM4) {
        blockSize_ = inputKeyCacheInShape_.GetDim(DIM3);
    }
    numTokens_ = inputKeyShape_.GetDim(DIM0);
    kHeadSize_ = inputKeyShape_.GetDim(DIM2);
    numHead_ = inputKeyShape_.GetDim(DIM1);
    if (inOutMode_ == DUAL_IN_OUT) {
        vHeadSize_ = inputValueShape_.GetDim(DIM2);
    }
    kHandleNumPerCore_ = inputKeyShape_.GetDim(DIM1) * inputKeyShape_.GetDim(DIM2);
    vHandleNumPerCore_ =
        (inOutMode_ == SINGLE_IN_OUT) ? DIM0 : inputValueShape_.GetDim(DIM1) * inputValueShape_.GetDim(DIM2);
    if (inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2) {
        if (inputKeyShape_.GetDim(DIM2) % DIM2 != 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "key",
                                                  std::to_string(inputKeyShape_.GetDim(DIM2)).c_str(),
                                                  "k_head_size must be an even number when input dtype is fp4");
            return ge::GRAPH_FAILED;
        }
        kHandleNumPerCore_ /= DIM2;
        kHeadSize_ /= DIM2;
    }
    if (inOutMode_ == DUAL_IN_OUT && (valueDtype_ == ge::DT_FLOAT4_E2M1 || valueDtype_ == ge::DT_FLOAT4_E1M2)) {
        if (inputValueShape_.GetDim(DIM2) % DIM2 != 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "value",
                                                  std::to_string(inputValueShape_.GetDim(DIM2)).c_str(),
                                                  "v_head_size must be an even number when input dtype is fp4");
            return ge::GRAPH_FAILED;
        }
        vHandleNumPerCore_ /= DIM2;
        vHeadSize_ /= DIM2;
    }
    bool isAlign =
        ((kHeadSize_ * dtypeByteSize_) % BLOCK_SIZE == 0 && (vHeadSize_ * valueDtypeByteSize_) % BLOCK_SIZE == 0);
    if (!isAlign) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "kHeadSize, vHeadSize",
                                               (std::to_string(kHeadSize_) + ", " + std::to_string(vHeadSize_)).c_str(),
                                               "kHeadSize and vHeadSize should be aligned to 32");
        return ge::GRAPH_FAILED;
    }
    blockFactor_ = Ops::Base::CeilDiv<int64_t>(numTokens_, totalCoreNum_);
    usedCoreNum_ = std::min(Ops::Base::CeilDiv<int64_t>(numTokens_, blockFactor_), totalCoreNum_);
    tailBlockFactor_ = numTokens_ - blockFactor_ * (usedCoreNum_ - 1);
    GetNonContigousStrideInfo();
    if (templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS || templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS_FIVE_DIM) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t maxHandleNumPerLoop = ubSize_ - RESERVED_UB_SIZE;
    // check whether tail dim can fully load.
    int64_t ubThreshold = kHandleNumPerCore_ * dtypeByteSize_ + vHandleNumPerCore_ * valueDtypeByteSize_;
    if (ubThreshold <= maxHandleNumPerLoop) {
        // tail dim can fully load
        isFullyLoad_ = FULLY_LOAD;
        OP_LOGD(context_, "NZ tail dim can fully load.");
        return ge::GRAPH_SUCCESS;
    }
    // tail dim can not fully load
    isFullyLoad_ = NOT_FULLY_LOAD;
    kHandleNumPerLoop_ = maxHandleNumPerLoop / dtypeByteSize_ / DOUBLE_BUFFER;
    kHandleNumPerLoop_ = Ops::Base::FloorAlign(kHandleNumPerLoop_, BLOCK_SIZE / dtypeByteSize_);
    kLoopNum_ = Ops::Base::CeilDiv<int64_t>(kHandleNumPerCore_, kHandleNumPerLoop_);
    if (kLoopNum_ > 0) {
        kTailHandleNum_ = kHandleNumPerCore_ - (kLoopNum_ - 1) * kHandleNumPerLoop_;
        kLoopNum_--;
    }
    if (inOutMode_ == DUAL_IN_OUT) {
        vHandleNumPerLoop_ = maxHandleNumPerLoop / valueDtypeByteSize_ / DOUBLE_BUFFER;
        vHandleNumPerLoop_ = Ops::Base::FloorAlign(vHandleNumPerLoop_, BLOCK_SIZE / valueDtypeByteSize_);
        vLoopNum_ = Ops::Base::CeilDiv<int64_t>(vHandleNumPerCore_, vHandleNumPerLoop_);
        if (vLoopNum_ > 0) {
            vTailHandleNum_ = vHandleNumPerCore_ - (vLoopNum_ - 1) * vHandleNumPerLoop_;
            vLoopNum_--;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::TemplateRope()
{
    if (inOutMode_ == SINGLE_IN_OUT &&
        (inputDtype_ == ge::DT_HIFLOAT8 || inputDtype_ == ge::DT_FLOAT8_E5M2 || inputDtype_ == ge::DT_FLOAT8_E4M3FN ||
         inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "input", ge::TypeUtils::DataTypeToSerialString(inputDtype_).c_str(),
            "input dtype not support in rope compression when single input and output");
        return ge::GRAPH_FAILED;
    }
    if (CheckDimValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GetCommonTilingInfo();
    // check whether tail dim can fully load.
    int64_t compressSeqOffsetSize = inputKeyShape_.GetDim(DIM0) * inputKeyShape_.GetDim(DIM2);
    int64_t alignKHead = RoundUp(kHeadSize_, dtypeByteSize_);
    int64_t alignVHead = RoundUp(vHeadSize_, dtypeByteSize_);
    int64_t numKHeadSize = seqLen_ * alignKHead;
    int64_t numVHeadSize = seqLen_ * alignVHead;
    int64_t maxHandleNumPerLoop = ubSize_ / dtypeByteSize_;
    int64_t floatFactor = (INT32_DTYPE_SIZE / dtypeByteSize_);
    int64_t inOutModeDim = (inOutMode_ == SINGLE_IN_OUT) ? DIM1 : DIM2;
    int64_t ubThreshold =
        std::max(numKHeadSize, numVHeadSize) * inOutModeDim +      // for inputKeyLocal & inputValueLocal
        blockFactor_ * DIM1 +                                      // slotMapping for key & value
        blockFactor_ * DIM1 +                                      // seqLens
        blockFactor_ * DIM1 +                                      // compressLen
        compressSeqOffsetSize * indexDtypeSize_ / dtypeByteSize_ + // compress_seq_offset size
        std::max(alignKHead, alignVHead) * floatFactor * DIM1 +    // reduce Buf for inputKeyLocal or inputValueLocal
        std::max(alignKHead, alignVHead) * floatFactor * DIM1 +    // divide Buf
        std::max(alignKHead, alignVHead) * floatFactor * DIM1;     // cast Buf
    if (ubThreshold <= maxHandleNumPerLoop) {
        // tail dim can fully load
        isFullyLoad_ = FULLY_LOAD;
        OP_LOGD(context_, "tail dim can fully load.");
        return ge::GRAPH_SUCCESS;
    }
    // can not fully load
    isFullyLoad_ = NOT_FULLY_LOAD;

    kHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
    kLoopNum_ = Ops::Base::CeilDiv<int64_t>(kHeadSize_, kHandleNumPerLoop_);
    if (kLoopNum_ > 0) {
        kTailHandleNum_ = kHeadSize_ - (kLoopNum_ - 1) * kHandleNumPerLoop_;
        kLoopNum_--;
    }

    if (inOutMode_ == DUAL_IN_OUT) {
        vHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
        vLoopNum_ = Ops::Base::CeilDiv<int64_t>(vHeadSize_, vHandleNumPerLoop_);
        if (vLoopNum_ > 0) {
            vTailHandleNum_ = vHeadSize_ - (vLoopNum_ - 1) * vHandleNumPerLoop_;
            vLoopNum_--;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::TemplateOmni()
{
    if (inOutMode_ == SINGLE_IN_OUT &&
        (inputDtype_ == ge::DT_HIFLOAT8 || inputDtype_ == ge::DT_FLOAT8_E5M2 || inputDtype_ == ge::DT_FLOAT8_E4M3FN ||
         inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "input", ge::TypeUtils::DataTypeToSerialString(inputDtype_).c_str(),
            "TemplateOmni input dtype not support in rope compression when single input and output");
        return ge::GRAPH_FAILED;
    }
    if (CheckDimValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GetCommonTilingInfo();
    // check whether tail dim can fully load.
    int64_t numKHeadSize = seqLen_ * RoundUp(kHeadSize_, dtypeByteSize_);
    int64_t numVHeadSize = seqLen_ * RoundUp(vHeadSize_, dtypeByteSize_);
    int64_t maxHandleNum = ubSize_ / dtypeByteSize_;
    int64_t inOutModeDim = (inOutMode_ == SINGLE_IN_OUT) ? DIM1 : DIM2;
    // slotMapping for key & value, seqLens, compressLen, compress_seq_offset size
    int64_t ubThreshold =
        std::max(numKHeadSize, numVHeadSize) * inOutModeDim + blockFactor_ * DIM4 * indexDtypeSize_ / dtypeByteSize_;
    if (ubThreshold <= maxHandleNum) {
        // tail dim can fully load
        isFullyLoad_ = FULLY_LOAD;
        OP_LOGD(context_, "TemplateOmni tail dim can fully load.");
        return ge::GRAPH_SUCCESS;
    }
    // can not fully load
    isFullyLoad_ = NOT_FULLY_LOAD;
    kHandleNumPerLoop_ = inOutMode_ == SINGLE_IN_OUT ? (ubSize_ / dtypeByteSize_) : (ubSize_ / dtypeByteSize_ / DIM2);
    kHandleNumPerLoop_ = Ops::Base::FloorAlign(kHandleNumPerLoop_, BLOCK_SIZE / dtypeByteSize_);
    kLoopNum_ = Ops::Base::CeilDiv<int64_t>(kHeadSize_, kHandleNumPerLoop_);
    if (kLoopNum_ > 0) {
        kTailHandleNum_ = kHeadSize_ - (kLoopNum_ - 1) * kHandleNumPerLoop_;
        kLoopNum_--;
    }

    if (inOutMode_ == DUAL_IN_OUT) {
        vHandleNumPerLoop_ = ubSize_ / dtypeByteSize_ / DIM2;
        vHandleNumPerLoop_ = Ops::Base::FloorAlign(vHandleNumPerLoop_, BLOCK_SIZE / dtypeByteSize_);
        vLoopNum_ = Ops::Base::CeilDiv<int64_t>(vHeadSize_, vHandleNumPerLoop_);
        if (vLoopNum_ > 0) {
            vTailHandleNum_ = vHeadSize_ - (vLoopNum_ - 1) * vHandleNumPerLoop_;
            vLoopNum_--;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::TemplateAlibi()
{
    if (inOutMode_ == SINGLE_IN_OUT &&
        (inputDtype_ == ge::DT_HIFLOAT8 || inputDtype_ == ge::DT_FLOAT8_E5M2 || inputDtype_ == ge::DT_FLOAT8_E4M3FN ||
         inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "input", ge::TypeUtils::DataTypeToSerialString(inputDtype_).c_str(),
            "input dtype not support in alibi compression when single input and output");
        return ge::GRAPH_FAILED;
    }
    if (CheckDimValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GetCommonTilingInfo();
    // check whether tail dim can fully load.
    int64_t maxHandleNumPerLoop = ubSize_ / dtypeByteSize_;
    int64_t inOutModeDim = (inOutMode_ == SINGLE_IN_OUT) ? DIM1 : DIM2;
    int64_t ubThreshold = RoundUp(std::max(kHeadSize_, vHeadSize_), dtypeByteSize_) * seqLen_ *
                              inOutModeDim +    // for inputKeyLocal & inputValueLocal
                          blockFactor_ * DIM1 + // slotMapping for key & value
                          blockFactor_ * DIM1;  // compressLen
    if (ubThreshold <= maxHandleNumPerLoop) {
        // tail dim can fully load
        isFullyLoad_ = FULLY_LOAD;
        OP_LOGD(context_, "tail dim can fully load.");
        return ge::GRAPH_SUCCESS;
    }
    // can not fully load
    isFullyLoad_ = NOT_FULLY_LOAD;

    kHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
    kLoopNum_ = Ops::Base::CeilDiv<int64_t>(kHeadSize_, kHandleNumPerLoop_);
    if (kLoopNum_ > 0) {
        kTailHandleNum_ = kHeadSize_ - (kLoopNum_ - 1) * kHandleNumPerLoop_;
        kLoopNum_--;
    }

    if (inOutMode_ == DUAL_IN_OUT) {
        vHandleNumPerLoop_ = MAX_HANLDE_BYTE_SIZE_PER_LOOP / dtypeByteSize_;
        vLoopNum_ = Ops::Base::CeilDiv<int64_t>(vHeadSize_, vHandleNumPerLoop_);
        if (vLoopNum_ > 0) {
            vTailHandleNum_ = vHeadSize_ - (vLoopNum_ - 1) * vHandleNumPerLoop_;
            vLoopNum_--;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckNormal()
{
    if (inputKeyShape_.GetDim(DIM0) != slotMappingShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim0 and slot_mapping.dim0",
            (std::to_string(inputKeyShape_.GetDim(DIM0)) + ", " + std::to_string(slotMappingShape_.GetDim(DIM0)))
                .c_str(),
            "the dim 0 of key and slot_mapping must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM1) != inputKeyCacheInShape_.GetDim(DIM2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim1 and key_cache.dim2",
            (std::to_string(inputKeyShape_.GetDim(DIM1)) + ", " + std::to_string(inputKeyCacheInShape_.GetDim(DIM2)))
                .c_str(),
            "the dim 1 of key must be equal to the dim 2 of key_cache");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM2) != inputKeyCacheInShape_.GetDim(DIM3)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim2 and key_cache.dim3",
            (std::to_string(inputKeyShape_.GetDim(DIM2)) + ", " + std::to_string(inputKeyCacheInShape_.GetDim(DIM3)))
                .c_str(),
            "the dim2 of key must be equal to the dim3 of key_cache");
        return ge::GRAPH_FAILED;
    }
    if (inOutMode_ != DUAL_IN_OUT) {
        return ge::GRAPH_SUCCESS;
    }
    if (inputKeyShape_.GetDim(DIM0) != inputValueShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim0 and value.dim0",
            (std::to_string(inputKeyShape_.GetDim(DIM0)) + ", " + std::to_string(inputValueShape_.GetDim(DIM0)))
                .c_str(),
            "the dim 0 of key and value must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM1) != inputValueShape_.GetDim(DIM1) ||
        inputKeyShape_.GetDim(DIM1) != inputValueCacheInShape_.GetDim(DIM2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim1, value.dim1 and value_cache.dim2",
            (std::to_string(inputKeyShape_.GetDim(DIM1)) + ", " + std::to_string(inputValueShape_.GetDim(DIM1)) + ", " +
             std::to_string(inputValueCacheInShape_.GetDim(DIM2)))
                .c_str(),
            "the dim 1 of key and value must be equal to the dim 2 of key_cache and value_cache");
        return ge::GRAPH_FAILED;
    }
    if (inputValueShape_.GetDim(DIM2) != inputValueCacheInShape_.GetDim(DIM3)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "value.dim2 and value_cache.dim3",
                                               (std::to_string(inputValueShape_.GetDim(DIM2)) + ", " +
                                                std::to_string(inputValueCacheInShape_.GetDim(DIM3)))
                                                   .c_str(),
                                               "the dim2 of value must be equal to the dim3 of value_cache");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckNz()
{
    int64_t numHead = inputKeyShape_.GetDim(DIM1);
    int64_t kHeadSize = inputKeyShape_.GetDim(DIM2);
    int64_t tokensK = numHead * kHeadSize;
    int64_t lastDimK = inputKeyCacheInShape_.GetDim(inputKeyCacheInShape_.GetDimNum() - DIM1);
    int64_t lastDimKSize = lastDimK * dtypeByteSize_;
    if (inputDtype_ == ge::DT_FLOAT4_E2M1 || inputDtype_ == ge::DT_FLOAT4_E1M2) {
        lastDimKSize /= 2;
    }
    if (lastDimKSize != BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "keyCache.lastDimSize",
                                              std::to_string(lastDimKSize).c_str(),
                                              "the last dim of key cache must be 32 Byte");
        return ge::GRAPH_FAILED;
    }

    if (inputKeyShape_.GetDim(DIM0) != slotMappingShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim0 and slot_mapping.dim0",
            (std::to_string(inputKeyShape_.GetDim(DIM0)) + ", " + std::to_string(slotMappingShape_.GetDim(DIM0)))
                .c_str(),
            "the dim 0 of key and slot_mapping must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyCacheInShape_.GetDimNum() == DIM4) {
        if (Ops::Base::CeilDiv<int64_t>(tokensK, lastDimK) != inputKeyCacheInShape_.GetDim(DIM1)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "ceil(numHead * kHeadSize / lastDimK) and keyCache.dim1",
                (std::to_string(Ops::Base::CeilDiv<int64_t>(tokensK, lastDimK)) + ", " +
                 std::to_string(inputKeyCacheInShape_.GetDim(DIM1)))
                    .c_str(),
                "the dim 1 of key cache must be ceil(numHead * kHeadSize / lastDimK)");
            return ge::GRAPH_FAILED;
        }
    } else {
        if ((kHeadSize / lastDimK) != inputKeyCacheInShape_.GetDim(DIM2)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "kHeadSize / lastDimK and keyCache.dim2",
                (std::to_string(kHeadSize / lastDimK) + ", " + std::to_string(inputKeyCacheInShape_.GetDim(DIM2)))
                    .c_str(),
                "the dim 2 of key cache must be (kHeadSize / lastDimK)");
            return ge::GRAPH_FAILED;
        }
        if (numHead != inputKeyCacheInShape_.GetDim(DIM1)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "numHead and keyCache.dim1",
                (std::to_string(numHead) + ", " + std::to_string(inputKeyCacheInShape_.GetDim(DIM1))).c_str(),
                "the dim 1 of key cache should be same as numHead");
            return ge::GRAPH_FAILED;
        }
    }
    if (inOutMode_ != DUAL_IN_OUT) {
        return ge::GRAPH_SUCCESS;
    }
    if (inputKeyShape_.GetDim(DIM0) != inputValueShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim0 and value.dim0",
            (std::to_string(inputKeyShape_.GetDim(DIM0)) + ", " + std::to_string(inputValueShape_.GetDim(DIM0)))
                .c_str(),
            "the dim 0 of key and value must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM1) != inputValueShape_.GetDim(DIM1)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim1 and value.dim1",
            (std::to_string(inputKeyShape_.GetDim(DIM1)) + ", " + std::to_string(inputValueShape_.GetDim(DIM1)))
                .c_str(),
            "the dim 1 of key and value must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyCacheInShape_.GetDim(inputKeyCacheInShape_.GetDimNum() - DIM2) !=
        inputValueCacheInShape_.GetDim(inputValueCacheInShape_.GetDimNum() - DIM2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "keyCache.blockSize and valueCache.blockSize",
            (std::to_string(inputKeyCacheInShape_.GetDim(inputKeyCacheInShape_.GetDimNum() - DIM2)) + ", " +
             std::to_string(inputValueCacheInShape_.GetDim(inputValueCacheInShape_.GetDimNum() - DIM2)))
                .c_str(),
            "the blockSize of key cache and value cache must be equal");
        return ge::GRAPH_FAILED;
    }

    if (inputKeyCacheInShape_.GetDim(DIM0) != inputValueCacheInShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "keyCache.dim0 and valueCache.dim0",
                                               (std::to_string(inputKeyCacheInShape_.GetDim(DIM0)) + ", " +
                                                std::to_string(inputValueCacheInShape_.GetDim(DIM0)))
                                                   .c_str(),
                                               "the dim0 of keyCache must be equal to the dim0 of valueCache");
        return ge::GRAPH_FAILED;
    }
    int64_t vHeadSize = inputValueShape_.GetDim(DIM2);
    int64_t tokensV = numHead * vHeadSize;
    int64_t lastDimV = inputValueCacheInShape_.GetDim(inputKeyCacheInShape_.GetDimNum() - DIM1);
    int64_t lastDimVSize = lastDimV * valueDtypeByteSize_;
    if (valueDtype_ == ge::DT_FLOAT4_E2M1 || valueDtype_ == ge::DT_FLOAT4_E1M2) {
        lastDimVSize /= 2;
    }
    if (lastDimVSize != BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "valueCache.lastDimSize",
                                              std::to_string(lastDimVSize).c_str(),
                                              "the last dim of value cache must be 32 Byte");
        return ge::GRAPH_FAILED;
    }
    if (inputValueCacheInShape_.GetDimNum() == DIM4) {
        if (Ops::Base::CeilDiv<int64_t>(tokensV, lastDimV) != inputValueCacheInShape_.GetDim(DIM1)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "ceil(numHead * vHeadSize / lastDimV) and valueCache.dim1",
                (std::to_string(Ops::Base::CeilDiv<int64_t>(tokensV, lastDimV)) + ", " +
                 std::to_string(inputValueCacheInShape_.GetDim(DIM1)))
                    .c_str(),
                "the dim 1 of value cache must be ceil(numHead * vHeadSize / lastDimV)");
            return ge::GRAPH_FAILED;
        }
    } else {
        if ((vHeadSize / lastDimV) != inputValueCacheInShape_.GetDim(DIM2)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "vHeadSize / lastDimV and valueCache.dim2",
                (std::to_string(vHeadSize / lastDimV) + ", " + std::to_string(inputValueCacheInShape_.GetDim(DIM2)))
                    .c_str(),
                "the dim 2 of value cache must be (vHeadSize / lastDimV)");
            return ge::GRAPH_FAILED;
        }
        if (numHead != inputValueCacheInShape_.GetDim(DIM1)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "numHead and valueCache.dim1",
                (std::to_string(numHead) + ", " + std::to_string(inputValueCacheInShape_.GetDim(DIM1))).c_str(),
                "the dim 1 of value cache should be same as numHead");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckRope()
{
    if (inputKeyShape_.GetDim(DIM0) != slotMappingShape_.GetDim(DIM0) ||
        inputKeyShape_.GetDim(DIM0) != compressLensShape_.GetDim(DIM0) ||
        inputKeyShape_.GetDim(DIM0) != seqLensShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim0, slot_mapping.dim0, compress_lens.dim0 and seq_lens.dim0",
            (std::to_string(inputKeyShape_.GetDim(DIM0)) + ", " + std::to_string(slotMappingShape_.GetDim(DIM0)) +
             ", " + std::to_string(compressLensShape_.GetDim(DIM0)) + ", " + std::to_string(seqLensShape_.GetDim(DIM0)))
                .c_str(),
            "the dim 0 of key, slot_mapping, compress_lens and seq_lens must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM2) != slotMappingShape_.GetDim(DIM1) ||
        inputKeyShape_.GetDim(DIM2) != compressLensShape_.GetDim(DIM1)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim2, slot_mapping.dim1 and compress_lens.dim1",
            (std::to_string(inputKeyShape_.GetDim(DIM2)) + ", " + std::to_string(slotMappingShape_.GetDim(DIM1)) +
             ", " + std::to_string(compressLensShape_.GetDim(DIM1)))
                .c_str(),
            "the dim2 of the key must be equal to the dim 1 of slot_mapping and the dim 1 of compress_lens");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM3) != inputKeyCacheInShape_.GetDim(DIM3)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim3 and key_cache.dim3",
            (std::to_string(inputKeyShape_.GetDim(DIM3)) + ", " + std::to_string(inputKeyCacheInShape_.GetDim(DIM3)))
                .c_str(),
            "the dim3 of key must be equal to the dim3 of key_cache");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyCacheInShape_.GetDim(DIM2) != DIM1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "key_cache",
                                              std::to_string(inputKeyCacheInShape_.GetDim(DIM2)).c_str(),
                                              "the dim2 of key_cache must be equal to 1");
        return ge::GRAPH_FAILED;
    }
    if (compressLensShape_.GetDim(DIM0) * compressLensShape_.GetDim(DIM1) != compressSeqOffsetShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "compress_lens.dim0 * compress_lens.dim1 and compress_seq_offset.dim0",
            (std::to_string(compressLensShape_.GetDim(DIM0) * compressLensShape_.GetDim(DIM1)) + ", " +
             std::to_string(compressSeqOffsetShape_.GetDim(DIM0)))
                .c_str(),
            "the dim0 of compress_seq_offset must be equal to compress_lens.dim0 multiplied by compress_lens.dim1");
        return ge::GRAPH_FAILED;
    }

    if (inOutMode_ != DUAL_IN_OUT) {
        return ge::GRAPH_SUCCESS;
    }
    if (inputKeyShape_.GetDim(DIM0) != inputValueShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim0 and value.dim0",
            (std::to_string(inputKeyShape_.GetDim(DIM0)) + ", " + std::to_string(inputValueShape_.GetDim(DIM0)))
                .c_str(),
            "the dim 0 of key and value must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM1) != inputValueShape_.GetDim(DIM1)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim1 and value.dim1",
            (std::to_string(inputKeyShape_.GetDim(DIM1)) + ", " + std::to_string(inputValueShape_.GetDim(DIM1)))
                .c_str(),
            "the dim1 of the key and value must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM2) != inputValueShape_.GetDim(DIM2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim2 and value.dim2",
            (std::to_string(inputKeyShape_.GetDim(DIM2)) + ", " + std::to_string(inputValueShape_.GetDim(DIM2)))
                .c_str(),
            "the dim2 of the key must be equal to the dim2 of value");
        return ge::GRAPH_FAILED;
    }
    if (inputValueShape_.GetDim(DIM3) != inputValueCacheInShape_.GetDim(DIM3)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "value.dim3 and value_cache.dim3",
                                               (std::to_string(inputValueShape_.GetDim(DIM3)) + ", " +
                                                std::to_string(inputValueCacheInShape_.GetDim(DIM3)))
                                                   .c_str(),
                                               "the dim3 of value must be equal to the dim3 of value_cache");
        return ge::GRAPH_FAILED;
    }
    if (inputValueCacheInShape_.GetDim(DIM2) != DIM1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "value_cache",
                                              std::to_string(inputValueCacheInShape_.GetDim(DIM2)).c_str(),
                                              "the dim2 of value_cache must be equal to 1");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckAlibi()
{
    if (inputKeyShape_.GetDim(DIM0) != slotMappingShape_.GetDim(DIM0) ||
        inputKeyShape_.GetDim(DIM0) != seqLensShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim0, slot_mapping.dim0 and seq_lens.dim0",
            (std::to_string(inputKeyShape_.GetDim(DIM0)) + ", " + std::to_string(slotMappingShape_.GetDim(DIM0)) +
             ", " + std::to_string(seqLensShape_.GetDim(DIM0)))
                .c_str(),
            "the dim 0 of key, slot_mapping and seq_lens must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyCacheInShape_.GetDim(DIM2) != DIM1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "key_cache",
                                              std::to_string(inputKeyCacheInShape_.GetDim(DIM2)).c_str(),
                                              "the dim2 of key_cache must be equal to 1");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM2) != slotMappingShape_.GetDim(DIM1)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim2 and slot_mapping.dim1",
            (std::to_string(inputKeyShape_.GetDim(DIM2)) + ", " + std::to_string(slotMappingShape_.GetDim(DIM1)))
                .c_str(),
            "the dim2 of key must be equal to the dim1 of slot_mapping");
        return ge::GRAPH_FAILED;
    }

    if (inOutMode_ != DUAL_IN_OUT) {
        return ge::GRAPH_SUCCESS;
    }
    if (inputKeyShape_.GetDim(DIM0) != inputValueShape_.GetDim(DIM0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim0 and value.dim0",
            (std::to_string(inputKeyShape_.GetDim(DIM0)) + ", " + std::to_string(inputValueShape_.GetDim(DIM0)))
                .c_str(),
            "the dim 0 of key and value must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM1) != inputValueShape_.GetDim(DIM1)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim1 and value.dim1",
            (std::to_string(inputKeyShape_.GetDim(DIM1)) + ", " + std::to_string(inputValueShape_.GetDim(DIM1)))
                .c_str(),
            "the dim1 of key and value must be equal");
        return ge::GRAPH_FAILED;
    }
    if (inputValueCacheInShape_.GetDim(DIM2) != DIM1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "value_cache",
                                              std::to_string(inputValueCacheInShape_.GetDim(DIM2)).c_str(),
                                              "the dim2 of value_cache must be equal to 1");
        return ge::GRAPH_FAILED;
    }
    if (inputKeyShape_.GetDim(DIM2) != inputValueShape_.GetDim(DIM2)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "key.dim2 and value.dim2",
            (std::to_string(inputKeyShape_.GetDim(DIM2)) + ", " + std::to_string(inputValueShape_.GetDim(DIM2)))
                .c_str(),
            "the dim2 of key and value must be equal");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::CheckDimValid()
{
    if (templateType_ == TEMPLATE_NORMAL || templateType_ == TEMPLATE_NCT ||
        templateType_ == TEMPLATE_NORMAL_NON_CONTIGUOUS) {
        return CheckNormal();
    }
    if (templateType_ == TEMPLATE_ROPE || templateType_ == TEMPLATE_OMNI) {
        return CheckRope();
    }
    if (templateType_ == TEMPLATE_ALIBI) {
        return CheckAlibi();
    }
    if (templateType_ == TEMPLATE_NZ || templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS ||
        templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS_FIVE_DIM) {
        return CheckNz();
    }
    return ge::GRAPH_FAILED;
}

void ScatterPaKvCacheTiling::SetInputPos()
{
    if (inOutMode_ == SINGLE_IN_OUT) {
        inputKey_ = INDEX_INPUT_KEY;
        inputKeyCacheIn_ = INDEX_INPUT_KEY_CACHE_IN;
        inputSlotMapping_ = INDEX_SLOT_MAPPING;
        inputCompressLens_ = INDEX_COMPRESS_LENS;
        inputCompressSeqOffset_ = INDEX_COMPRESS_SEQ_OFFSET;
        inputSeqLens_ = INDEX_INPUT_SEQ_LENS;
    } else if (inOutMode_ == DUAL_IN_OUT) {
        inputKey_ = INDEX_INPUT_KEY;
        inputKeyCacheIn_ = INDEX_INPUT_KEY_CACHE_IN;
        inputSlotMapping_ = INDEX_SLOT_MAPPING;
        inputValue_ = INDEX_INPUT_VALUE;
        inputValueCacheIn_ = INDEX_INPUT_VALUE_CACHE_IN;
        inputCompressLens_ = INDEX_COMPRESS_LENS + DUAL_IN_OUT_MODE_INDEX_OFFSET;
        inputCompressSeqOffset_ = INDEX_COMPRESS_SEQ_OFFSET + DUAL_IN_OUT_MODE_INDEX_OFFSET;
        inputSeqLens_ = INDEX_INPUT_SEQ_LENS + DUAL_IN_OUT_MODE_INDEX_OFFSET;
    }
}

ge::graphStatus ScatterPaKvCacheTiling::GetTemplateType(int64_t inputKeyDimNum)
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto cacheMode = attrs->GetStr(INPUT_CACHE_MODE_INDEX);
    auto scatterMode = attrs->GetStr(INPUT_SCATTER_MODE_INDEX);
    scatterMode = (scatterMode == nullptr) ? "" : scatterMode;
    if (strcmp(cacheMode, "PA_NZ") == 0) {
        // entering template nz
        templateType_ = TEMPLATE_NZ;
        return ge::GRAPH_SUCCESS;
    } else if (strcmp(cacheMode, "") == 0 || strcmp(cacheMode, "Norm") == 0) {
        if (strcmp(scatterMode, "Rope") == 0) {
            templateType_ = TEMPLATE_ROPE;
            return ge::GRAPH_SUCCESS;
        } else if (strcmp(scatterMode, "Alibi") == 0) {
            templateType_ = TEMPLATE_ALIBI;
            return ge::GRAPH_SUCCESS;
        } else if (strcmp(scatterMode, "Omni") == 0) {
            templateType_ = TEMPLATE_OMNI;
            return ge::GRAPH_SUCCESS;
        } else if (strcmp(scatterMode, "Nct") == 0) {
            templateType_ = TEMPLATE_NCT;
            return ge::GRAPH_SUCCESS;
        } else if (strcmp(scatterMode, "") == 0 || strcmp(scatterMode, "None") == 0) {
            if (inputKeyDimNum == static_cast<size_t>(DIM3)) {
                templateType_ = TEMPLATE_NORMAL;
                return ge::GRAPH_SUCCESS;
            }
            auto compressLens = context_->GetOptionalInputTensor(inputCompressLens_);
            auto compressSeqOffset = context_->GetOptionalInputTensor(inputCompressSeqOffset_);
            auto seqLens = context_->GetOptionalInputTensor(inputSeqLens_);
            if (compressLens != nullptr && compressSeqOffset != nullptr && seqLens != nullptr) {
                // entering template rope
                templateType_ = TEMPLATE_ROPE;
                return ge::GRAPH_SUCCESS;
            } else if (compressLens != nullptr && compressSeqOffset == nullptr && seqLens != nullptr) {
                // entering template alibi
                templateType_ = TEMPLATE_ALIBI;
                return ge::GRAPH_SUCCESS;
            } else {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "compressLens, seqLens", "None",
                    "when dim num of inputKey is 4, compress_lens and seq_lens must not be None");
                return ge::GRAPH_FAILED;
            }
        } else {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "scatterMode", scatterMode,
                                                  "scatterMode only support None, Rope, Alibi, Omni, Nct");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cacheMode", cacheMode,
                                              "cacheMode only support None, Norm or PA_NZ");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::GetContiguousTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx)
{
    auto xStorageShape = context_->GetInputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
    shape = xStorageShape->GetStorageShape();
    stride.SetDimNum(shape.GetDimNum());
    int32_t maxDim = static_cast<int32_t>(shape.GetDimNum()) - 1;
    int64_t xStride = 1;
    for (int32_t j = maxDim; j >= 0; --j) {
        stride.SetStride(j, xStride);
        xStride *= shape.GetDim(j);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::GetTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx)
{
    bool isView = context_->InputIsView(idx);
    if (isView) {
        auto *inputStride = context_->GetInputStride(idx);
        if (inputStride == nullptr || inputStride->GetDimNum() == 0) {
            return GetContiguousTensorInfo(shape, stride, idx);
        } else {
            stride = *inputStride;
            shape = context_->GetInputShape(idx)->GetShape();
        }
    } else {
        return GetContiguousTensorInfo(shape, stride, idx);
    }
    return ge::GRAPH_SUCCESS;
}

bool ScatterPaKvCacheTiling::IsAxesContiguous(const gert::Stride &stride, const gert::Shape &shape, int64_t startAxis,
                                              int64_t endAxis)
{
    if (startAxis < 0 || endAxis < 0 || startAxis >= endAxis) {
        return false;
    }

    int64_t dimNum = static_cast<int64_t>(shape.GetDimNum());
    if (endAxis > dimNum) {
        return false;
    }

    int64_t validStride = 1;
    for (int64_t i = endAxis - 1; i >= startAxis; i--) {
        if (shape.GetDim(i) == 1) {
            continue;
        }
        if (stride.GetStride(i) != validStride) {
            return false;
        }
        validStride *= shape.GetDim(i);
    }
    return true;
}

bool ScatterPaKvCacheTiling::IsLastAxisContiguous(const gert::Stride &stride, const gert::Shape &shape)
{
    int64_t dimNum = static_cast<int64_t>(shape.GetDimNum());
    if (dimNum < 1) {
        return false;
    }

    return stride.GetStride(dimNum - 1) == 1;
}

// 当前非连续场景只支持key, keyCache, value, valueCache尾轴连续的场景
bool ScatterPaKvCacheTiling::IsNonContiguous()
{
    bool keyLastAxisContiguous = IsLastAxisContiguous(keyStride_, inputKeyShape_);
    bool keyCacheLastAxisContiguous = IsLastAxisContiguous(keyCacheStride_, inputKeyCacheInShape_);
    bool valueLastAxisContiguous = true;
    bool valueCacheLastAxisContiguous = true;

    if (inOutMode_ == DUAL_IN_OUT) {
        valueLastAxisContiguous = IsLastAxisContiguous(valueStride_, inputValueShape_);
        valueCacheLastAxisContiguous = IsLastAxisContiguous(valueCacheStride_, inputValueCacheInShape_);
    }

    bool allLastAxisContiguous =
        keyLastAxisContiguous && keyCacheLastAxisContiguous && valueLastAxisContiguous && valueCacheLastAxisContiguous;

    bool anyNonContiguous =
        !IsAxesContiguous(keyStride_, inputKeyShape_, 0, static_cast<int64_t>(inputKeyShape_.GetDimNum())) ||
        !IsAxesContiguous(keyCacheStride_, inputKeyCacheInShape_, 0,
                          static_cast<int64_t>(inputKeyCacheInShape_.GetDimNum()));
    if (inOutMode_ == DUAL_IN_OUT) {
        anyNonContiguous =
            anyNonContiguous ||
            !IsAxesContiguous(valueStride_, inputValueShape_, 0, static_cast<int64_t>(inputValueShape_.GetDimNum())) ||
            !IsAxesContiguous(valueCacheStride_, inputValueCacheInShape_, 0,
                              static_cast<int64_t>(inputValueCacheInShape_.GetDimNum()));
    }

    return allLastAxisContiguous && anyNonContiguous;
}

ge::graphStatus ScatterPaKvCacheTiling::GetShapeAttrsInfo()
{
    SetInputPos();
    if (GetTensorInfo(inputKeyShape_, keyStride_, inputKey_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t inputKeyDimNum = inputKeyShape_.GetDimNum();
    if (GetTensorInfo(inputKeyCacheInShape_, keyCacheStride_, inputKeyCacheIn_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto slotMapping = context_->GetRequiredInputTensor(inputSlotMapping_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, slotMapping);
    slotMappingShape_ = slotMapping->GetStorageShape();

    if (inOutMode_ == DUAL_IN_OUT) {
        if (GetTensorInfo(inputValueShape_, valueStride_, inputValue_) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (GetTensorInfo(inputValueCacheInShape_, valueCacheStride_, inputValueCacheIn_) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        size_t inputValueDimNum = inputValueShape_.GetDimNum();
        if (inputKeyDimNum != inputValueDimNum) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context_->GetNodeName(), "inputKey and inputValue",
                (std::to_string(inputKeyDimNum) + ", " + std::to_string(inputValueDimNum)).c_str(),
                "the dim num of inputKey and inputValue must be the same");
            return ge::GRAPH_FAILED;
        }
    }
    if (inputKeyDimNum != static_cast<size_t>(DIM3) && inputKeyDimNum != static_cast<size_t>(DIM4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "inputKey",
                                                 std::to_string(inputKeyDimNum).c_str(),
                                                 "the dim num of inputKey must be 3 or 4");
        return ge::GRAPH_FAILED;
    }

    if (GetTemplateType(inputKeyDimNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (templateType_ == TEMPLATE_NORMAL && IsNonContiguous()) {
        templateType_ = TEMPLATE_NORMAL_NON_CONTIGUOUS;
    }
    if (templateType_ == TEMPLATE_NZ && IsNonContiguous()) {
        templateType_ = TEMPLATE_NZ_NON_CONTIGUOUS;
        if (inputKeyCacheInShape_.GetDimNum() != DIM4) {
            templateType_ = TEMPLATE_NZ_NON_CONTIGUOUS_FIVE_DIM;
        }
    }
    if (GetInputDtype() != ge::GRAPH_SUCCESS || GetIndexDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (templateType_ == TEMPLATE_NCT) {
        auto attrs = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
        auto strides = attrs->GetListInt(INPUT_STRIDES_INDEX);
        auto offsets = attrs->GetListInt(INPUT_OFFSET_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, strides);
        OP_CHECK_NULL_WITH_CONTEXT(context_, offsets);
        if (strides->GetSize() < DIM2) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "strides size",
                                                  std::to_string(strides->GetSize()).c_str(), "should be at least 2");
            return ge::GRAPH_FAILED;
        }
        if (offsets->GetSize() < DIM2) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "offsets size",
                                                  std::to_string(offsets->GetSize()).c_str(), "should be at least 2");
            return ge::GRAPH_FAILED;
        }
        kStride_ = strides->GetData()[0];
        vStride_ = strides->GetData()[1];
        kOffset_ = offsets->GetData()[0];
        vOffset_ = offsets->GetData()[1];
    }
    if (inputKeyDimNum == static_cast<size_t>(DIM3)) {
        return ge::GRAPH_SUCCESS;
    }
    // else: inputKeyDimNum is 4
    auto compressLens = context_->GetOptionalInputTensor(inputCompressLens_);
    auto compressSeqOffset = context_->GetOptionalInputTensor(inputCompressSeqOffset_);
    auto seqLens = context_->GetOptionalInputTensor(inputSeqLens_);

    auto inputSlotMappingDesc = context_->GetInputDesc(inputSlotMapping_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSlotMappingDesc);
    ge::DataType inputSlotMappingDtype = inputSlotMappingDesc->GetDataType();

    auto inputCompressLensDsc = context_->GetOptionalInputDesc(inputCompressLens_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputCompressLensDsc);
    ge::DataType inputCompressLensDtype = inputCompressLensDsc->GetDataType();
    if (inputSlotMappingDtype != inputCompressLensDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "compressLens and slotMapping",
                                               (ge::TypeUtils::DataTypeToSerialString(inputCompressLensDtype) + ", " +
                                                ge::TypeUtils::DataTypeToSerialString(inputSlotMappingDtype))
                                                   .c_str(),
                                               "the dtype of compressLens and slotMapping must be the same");
        return ge::GRAPH_FAILED;
    }
    auto inputSeqLensDsc = context_->GetOptionalInputDesc(inputSeqLens_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSeqLensDsc);
    ge::DataType inputSeqLensDtype = inputSeqLensDsc->GetDataType();
    if (inputSlotMappingDtype != inputSeqLensDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "seqLens and slotMapping",
                                               (ge::TypeUtils::DataTypeToSerialString(inputSeqLensDtype) + ", " +
                                                ge::TypeUtils::DataTypeToSerialString(inputSlotMappingDtype))
                                                   .c_str(),
                                               "the dtype of seqLens and slotMapping must be the same");
        return ge::GRAPH_FAILED;
    }
    if (templateType_ == TEMPLATE_ROPE || templateType_ == TEMPLATE_OMNI) {
        // entering template rope omni
        auto inputCompressSeqOffsetSDsc = context_->GetOptionalInputDesc(inputCompressSeqOffset_);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputCompressSeqOffsetSDsc);
        ge::DataType inputCompressSeqOffsetSDtype = inputCompressSeqOffsetSDsc->GetDataType();
        if (inputSlotMappingDtype != inputCompressSeqOffsetSDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "compressSeqOffset and slotMapping",
                (ge::TypeUtils::DataTypeToSerialString(inputCompressSeqOffsetSDtype) + ", " +
                 ge::TypeUtils::DataTypeToSerialString(inputSlotMappingDtype))
                    .c_str(),
                "the dtype of compressSeqOffset and slotMapping must be the same");
            return ge::GRAPH_FAILED;
        }
        compressLensShape_ = compressLens->GetStorageShape();
        compressSeqOffsetShape_ = compressSeqOffset->GetStorageShape();
        seqLensShape_ = seqLens->GetStorageShape();
    } else if (templateType_ == TEMPLATE_ALIBI) {
        // entering template alibi
        compressLensShape_ = compressLens->GetStorageShape();
        seqLensShape_ = seqLens->GetStorageShape();
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheTiling::DoOpTiling()
{
    if (templateType_ == TEMPLATE_NORMAL || templateType_ == TEMPLATE_NCT ||
        templateType_ == TEMPLATE_NORMAL_NON_CONTIGUOUS) {
        return TemplateNormal();
    } else if (templateType_ == TEMPLATE_ROPE) {
        return TemplateRope();
    } else if (templateType_ == TEMPLATE_ALIBI) {
        return TemplateAlibi();
    } else if (templateType_ == TEMPLATE_OMNI) {
        return TemplateOmni();
    } else if (templateType_ == TEMPLATE_NZ || templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS ||
               templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS_FIVE_DIM) {
        return TemplateNZ();
    }
    return ge::GRAPH_FAILED;
}

uint64_t ScatterPaKvCacheTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus ScatterPaKvCacheTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

void ScatterPaKvCacheTiling::GenTilingKey()
{
    tilingKey_ = static_cast<uint64_t>(templateType_) * HUNDRED + static_cast<uint64_t>(indexDtypeSize_) * TEN +
                 static_cast<uint64_t>(isFullyLoad_);
    if (templateType_ == TEMPLATE_NZ || templateType_ == TEMPLATE_OMNI || templateType_ == TEMPLATE_NCT ||
        templateType_ == TEMPLATE_NORMAL_NON_CONTIGUOUS || templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS ||
        templateType_ == TEMPLATE_NZ_NON_CONTIGUOUS_FIVE_DIM) {
        tilingKey_ = static_cast<uint64_t>(templateType_) * HUNDRED + static_cast<uint64_t>(isFullyLoad_);
    }
}

ge::graphStatus ScatterPaKvCacheTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(usedCoreNum_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_tailBlockFactor(tailBlockFactor_);
    tilingData_.set_kHandleNumPerCore(kHandleNumPerCore_);
    tilingData_.set_vHandleNumPerCore(vHandleNumPerCore_);
    tilingData_.set_kLoopNum(kLoopNum_);
    tilingData_.set_vLoopNum(vLoopNum_);
    tilingData_.set_kHandleNumPerLoop(kHandleNumPerLoop_);
    tilingData_.set_vHandleNumPerLoop(vHandleNumPerLoop_);
    tilingData_.set_kTailHandleNum(kTailHandleNum_);
    tilingData_.set_vTailHandleNum(vTailHandleNum_);
    tilingData_.set_keyStride0(keyStride0_);
    tilingData_.set_keyStride1(keyStride1_);
    tilingData_.set_keyStride2(keyStride2_);
    tilingData_.set_valueStride0(valueStride0_);
    tilingData_.set_valueStride1(valueStride1_);
    tilingData_.set_valueStride2(valueStride2_);
    tilingData_.set_kHeadSize(kHeadSize_);
    tilingData_.set_vHeadSize(vHeadSize_);
    tilingData_.set_batch(batch_);
    tilingData_.set_seqLen(seqLen_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_numBlocks(numBlocks_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_kStride(kStride_);
    tilingData_.set_vStride(vStride_);
    tilingData_.set_kOffset(kOffset_);
    tilingData_.set_vOffset(vOffset_);
    tilingData_.set_keyCacheStride0(keyCacheStride0_);
    tilingData_.set_keyCacheStride1(keyCacheStride1_);
    tilingData_.set_keyCacheStride2(keyCacheStride2_);
    tilingData_.set_keyCacheStride3(keyCacheStride3_);
    if (inOutMode_ == DUAL_IN_OUT) {
        tilingData_.set_valueCacheStride0(valueCacheStride0_);
        tilingData_.set_valueCacheStride1(valueCacheStride1_);
        tilingData_.set_valueCacheStride2(valueCacheStride2_);
        tilingData_.set_valueCacheStride3(valueCacheStride3_);
    }
    GenTilingKey();
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void ScatterPaKvCacheTiling::DumpTilingInfo()
{
    std::string info;
    info = " totalCoreNum: " + std::to_string(totalCoreNum_);
    info += " usedCoreNum: " + std::to_string(usedCoreNum_);
    info += " blockFactor: " + std::to_string(blockFactor_);
    info += " tailBlockFactor: " + std::to_string(tailBlockFactor_);
    info += " kHandleNumPerCore: " + std::to_string(kHandleNumPerCore_);
    if (inOutMode_ == DUAL_IN_OUT) {
        info += " vHandleNumPerCore: " + std::to_string(vHandleNumPerCore_);
    }
    info += " kLoopNum: " + std::to_string(kLoopNum_);
    if (inOutMode_ == DUAL_IN_OUT) {
        info += " vLoopNum: " + std::to_string(vLoopNum_);
    }
    info += " kHandleNumPerLoop: " + std::to_string(kHandleNumPerLoop_);
    if (inOutMode_ == DUAL_IN_OUT) {
        info += " vHandleNumPerLoop: " + std::to_string(vHandleNumPerLoop_);
    }
    info += " kTailHandleNum: " + std::to_string(kTailHandleNum_);
    if (inOutMode_ == DUAL_IN_OUT) {
        info += " vTailHandleNum: " + std::to_string(vTailHandleNum_);
    }
    info += " keyStride0: " + std::to_string(keyStride0_);
    info += " keyStride1: " + std::to_string(keyStride1_);
    info += " keyStride2: " + std::to_string(keyStride2_);
    if (inOutMode_ == DUAL_IN_OUT) {
        info += " valueStride0: " + std::to_string(valueStride0_);
        info += " valueStride1: " + std::to_string(valueStride1_);
        info += " valueStride2: " + std::to_string(valueStride2_);
    }
    info += " kHeadSize: " + std::to_string(kHeadSize_);
    if (inOutMode_ == DUAL_IN_OUT) {
        info += " vHeadSize: " + std::to_string(vHeadSize_);
    }
    info += " seqLen: " + std::to_string(seqLen_);
    info += " numBlocks: " + std::to_string(numBlocks_);
    info += " blockSize: " + std::to_string(blockSize_);
    info += " ubSize: " + std::to_string(ubSize_);
    info += " tilingKey: " + std::to_string(tilingKey_);
    info += " kStride: " + std::to_string(kStride_);
    info += " vStride: " + std::to_string(vStride_);
    info += " kOffset: " + std::to_string(kOffset_);
    info += " vOffset: " + std::to_string(vOffset_);
    info += " keyCacheStride0: " + std::to_string(keyCacheStride0_);
    info += " keyCacheStride1: " + std::to_string(keyCacheStride1_);
    info += " keyCacheStride2: " + std::to_string(keyCacheStride2_);
    info += " keyCacheStride3: " + std::to_string(keyCacheStride3_);
    if (inOutMode_ == DUAL_IN_OUT) {
        info += " valueCacheStride0: " + std::to_string(valueCacheStride0_);
        info += " valueCacheStride1: " + std::to_string(valueCacheStride1_);
        info += " valueCacheStride2: " + std::to_string(valueCacheStride2_);
        info += " valueCacheStride3: " + std::to_string(valueCacheStride3_);
    }
    OP_LOGI(context_->GetNodeName(), "%s", info.c_str());
}

ge::graphStatus Tiling4ScatterPaKvCache(gert::TilingContext *context_)
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        ScatterPaKvCacheMembaseTiling tiling(context_);
        return tiling.DoTiling();
    }

    ScatterPaKvCacheTiling tiling(context_, DUAL_IN_OUT);
    return tiling.DoTiling();
}

ge::graphStatus Tiling4ScatterPaCache(gert::TilingContext *context_)
{
    ScatterPaKvCacheTiling tiling(context_, SINGLE_IN_OUT);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepare4ScatterPaKvCache(gert::TilingParseContext *context_)
{
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ScatterPaKvCache op.
IMPL_OP_OPTILING(ScatterPaKvCache)
    .Tiling(Tiling4ScatterPaKvCache)
    .TilingParse<ScatterPaKvCacheCompileInfo>(TilingPrepare4ScatterPaKvCache);
IMPL_OP_OPTILING(ScatterPaCache)
    .Tiling(Tiling4ScatterPaCache)
    .TilingParse<ScatterPaKvCacheCompileInfo>(TilingPrepare4ScatterPaKvCache);

} // namespace optiling
