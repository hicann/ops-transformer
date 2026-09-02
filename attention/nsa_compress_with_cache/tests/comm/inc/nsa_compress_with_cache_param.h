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
 * \file nsa_compress_with_cache_param.h
 * \brief NsaCompressWithCache  参数信息.
 */

#ifndef UTEST_NSA_COMPRESS_WITH_CACHE_PARAM_H
#define UTEST_NSA_COMPRESS_WITH_CACHE_PARAM_H

#include <cstdint>
#include <vector>
#include "graph/types.h"
#include "tests/utils/tests_utils_log.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::NsaCompressWithCache {

class NsaCompressWithCacheParam {
public:
    using Tensor = ops::adv::tests::utils::Tensor;

    /**
     * Pse Alibi 场景下 S1 取值;
     */
    static constexpr int64_t maxSeqLen = 1024;

public:
    enum class LayoutType {
        BSH,
        SBH,
        BNSD,
        BSND,
        TND,
        UNKOWN
    };

public:
    /* 设置参数 */
    int64_t mBatchSize = 0;
    int64_t mHeadNum = 0;
    int64_t mHeadDim = 0;

    ge::DataType mDtype = ge::DataType::DT_UNDEFINED;
    std::vector<int32_t> mSlotMappingList = {};
    std::vector<int64_t> mActSeqLenList = {};
    std::vector<int32_t> mBlockTableList = {};
    LayoutType mLayoutType = LayoutType::TND;

    int64_t mCompressBlockSize = 0;
    int64_t mCompressStride = 0;
    int64_t mActSeqLenType = 1;
    int64_t mPageBlockSize = 128;

    /* 生成参数 */
    std::string mLayoutOptional;
    int64_t mBlockNumPerbatch;
    int64_t mTotolBlockNum;

    /* 输入输出 */
    Tensor mInput, mWeight, mSlotMapping, mOutputCache, mActSeqLenOptional, mBlockTableOptional;

public:
    NsaCompressWithCacheParam() = default;
    NsaCompressWithCacheParam(int64_t batchSize, int64_t headNum, int64_t headDim, ge::DataType dtype,
                              std::vector<int64_t> actSeqLenList, LayoutType layoutType, int64_t compressBlockSize,
                              int64_t compressStride, int64_t actSeqLenType, int64_t pageBlockSize);

    virtual ~NsaCompressWithCacheParam() = default;

    virtual bool Init();

    template <class T>
    static bool InitTensor(Tensor &tensor, std::vector<T> &hostData)
    {
        if (hostData.empty()) {
            return true;
        }
        int64_t expMinSize = hostData.size() * sizeof(T);
        if (tensor.AllocDevData(0, expMinSize) == nullptr) {
            LOG_ERR("Tensor(%s, %ld) AllocDevData Failed.", tensor.Name().c_str(), expMinSize);
            return false;
        }
        return tensor.CopyHostToDevData(hostData);
    }
};

} // namespace ops::adv::tests::NsaCompressWithCache
#endif // UTEST_NSA_COMPRESS_WITH_CACHE_PARAM_H
