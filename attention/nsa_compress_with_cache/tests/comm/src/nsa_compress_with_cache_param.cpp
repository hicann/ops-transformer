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
 * \file nsa_compress_with_cache_param.cpp
 * \brief NsaCompressWithCache  参数信息.
 */

#include "nsa_compress_with_cache_param.h"
#include "tests/utils/tests_utils_log.h"

using Tensor = ops::adv::tests::utils::Tensor;

using namespace ops::adv::tests::NsaCompressWithCache;


NsaCompressWithCacheParam::NsaCompressWithCacheParam(int64_t batchSize, int64_t headNum, int64_t headDim,
                                                     ge::DataType dtype, std::vector<int64_t> actSeqLenList,
                                                     LayoutType layoutType, int64_t compressBlockSize,
                                                     int64_t compressStride, int64_t actSeqLenType,
                                                     int64_t pageBlockSize)

    : mBatchSize(batchSize), mHeadNum(headNum), mHeadDim(headDim), mDtype(dtype),
      mActSeqLenList(std::move(actSeqLenList)), mLayoutType(layoutType), mCompressBlockSize(compressBlockSize),
      mCompressStride(compressStride), mActSeqLenType(actSeqLenType), mPageBlockSize(pageBlockSize)
{
    mBlockNumPerbatch = maxSeqLen / mPageBlockSize;
    mTotolBlockNum = mBlockNumPerbatch * mBatchSize;
    mSlotMappingList.resize(batchSize);
    for(int i = 0; i < batchSize; i++) {
        mSlotMappingList[i] = i;
    }
    mBlockTableList.resize(mTotolBlockNum);
    for(int i = 0; i < mTotolBlockNum; i++) {
        mBlockTableList[i] = i;
    }
}


bool NsaCompressWithCacheParam::Init()
{
    switch (mLayoutType) {
        case LayoutType::BSH:
            mLayoutOptional = "BSH";
            break;
        case LayoutType::SBH:
            mLayoutOptional = "SBH";
            break;
        case LayoutType::BNSD:
            mLayoutOptional = "BNSD";
            break;
        case LayoutType::BSND:
            mLayoutOptional = "BSND";
            break;
        case LayoutType::TND:
            mLayoutOptional = "TND";
            break;
        default:
            mLayoutOptional= "UNKOWN";
    }

    mInput = Tensor("input", {mTotolBlockNum, mPageBlockSize, mHeadNum, mHeadDim}, mLayoutOptional.c_str(), mDtype,
                    ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    mWeight = Tensor("weight", {mCompressBlockSize, mHeadNum}, mLayoutOptional.c_str(), mDtype, ge::FORMAT_ND,
                     Tensor::TensorType::REQUIRED_INPUT);
    mOutputCache = Tensor("outputCache", {mBatchSize, mHeadNum, mHeadDim}, mLayoutOptional.c_str(), mDtype,
                          ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    mSlotMapping = Tensor("slotMapping", {mBatchSize}, "B", ge::DataType::DT_INT32, ge::FORMAT_ND,
                          Tensor::TensorType::REQUIRED_INPUT);
    mBlockTableOptional = Tensor("blockTableOptional", {mBatchSize, mBlockNumPerbatch}, "2", ge::DataType::DT_INT32,
                                 ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);

    if (!mActSeqLenList.empty()) {
        mActSeqLenOptional = Tensor("actSeqLenOptional", {mBatchSize}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND,
                                    Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        LOG_ERR("Currently, actSeqLenOptional must not be empty.");
    }

    /**
     * TensorData 初始化
     * 出于性能角度考虑, 此处仅申请 Tiling 阶段必需的 TensorData
     */
    if (!ops::adv::tests::NsaCompressWithCache::NsaCompressWithCacheParam::InitTensor(mActSeqLenOptional,
                                                                                      mActSeqLenList)) {
        return false;
    }
    if (!ops::adv::tests::NsaCompressWithCache::NsaCompressWithCacheParam::InitTensor(mSlotMapping,
                                                                                      mSlotMappingList)) {
        return false;
    }
    if (!ops::adv::tests::NsaCompressWithCache::NsaCompressWithCacheParam::InitTensor(mBlockTableOptional,
                                                                                      mBlockTableList)) {
        return false;
    }
    return true;
}
