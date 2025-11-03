/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_select_attention_infer_param.h
 * \brief NsaSelectAttentionInfer  参数信息.
 */

#ifndef UTEST_NSA_SELECT_ATTENTION_INFER_PARAM_H
#define UTEST_NSA_SELECT_ATTENTION_INFER_PARAM_H

#include <cstdint>
#include <vector>
#include "graph/types.h"
#include "tests/utils/log.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::NsaSelectAttentionInfer {

class NsaSelectAttentionInferParam {
public:
    using Tensor = ops::adv::tests::utils::Tensor;
public:
    enum class LayoutType {
        BSH,
        BNSD,
        BSND,
        TND,
    };

public:
    /* 设置参数 */
    int64_t batchSize = 1;
    int64_t qSeqSize = 1;
    int64_t headSize = 2;
    int64_t headDim = 192;
    int64_t maxBlockNumPerBatch = 8;
    int64_t blockSize = 128;
    int64_t seqSize = 1024;
    int64_t headDimV = 128;
    int64_t headSizeV = 1;
    int64_t selectedBlockSize = 272;
    int64_t selectedBlockCount = 4;
    int64_t sparseMode = 1;
    float scaleValue = 1.0f;
    std::string inputLayout = "BSND";
    ge::DataType optionalDataType = ge::DT_FLOAT16;
    std::vector<int64_t> actualSeqQLenList = {};
    std::vector<int64_t> actualSeqKVLenList = {};
    int64_t numtokens = 1;

    std::vector<int32_t> topkData = {};
    std::vector<int32_t> blocktableData ={};

    /* 输入输出 */
    Tensor query, key, value, blocktable, actualQSeqLengths, actualKvSeqLengths, topkIndices, attentionOut;

public:
    NsaSelectAttentionInferParam() = default;
    NsaSelectAttentionInferParam(int64_t pBatchSize, int64_t pQSeqSize, int64_t pHeadSize, int64_t pHeadDim, int64_t pMaxBlockNumPerBatch,
                  int64_t pBlockSize, int64_t pSeqSize, int64_t pHeadDimV, int64_t pHeadSizeV, int64_t pSelectedBlockSize, int64_t pSelectedBlockCount, int64_t pSparseMode,
                  float pScaleValue, std::string pInputLayout, ge::DataType pOptionalDataType, std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualSeqKVLenList, int64_t numtokens);

    virtual ~NsaSelectAttentionInferParam() = default;

    virtual bool Init();
    void TNDInitParam();
    void BSNDInitParam();
    void BSHInitParam();
    void TopkInitParam();
    void InitTopkDataTND();
    void InitTopkDataForHead();
    void InitTopkDataForSelectedBlockSize();
    void InitTopkDataOtherLayout();
    void InitTopkDataForS1();
    template <class T> static bool InitTensor(Tensor &tensor, std::vector<T> &hostData)
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

}
#endif // UTEST_NSA_SELECT_ATTENTION_INFER_PARAM_H
