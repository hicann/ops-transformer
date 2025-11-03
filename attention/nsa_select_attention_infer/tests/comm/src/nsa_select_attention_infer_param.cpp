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
 * \file nsa_select_attention_infer_param.cpp
 * \brief NsaSelectAttentionInfer  参数信息.
 */

#include "nsa_select_attention_infer_param.h"
#include "tests/utils/log.h"

using Tensor = ops::adv::tests::utils::Tensor;

using namespace ops::adv::tests::NsaSelectAttentionInfer;


NsaSelectAttentionInferParam::NsaSelectAttentionInferParam(int64_t pBatchSize, int64_t pQSeqSize,
            int64_t pHeadSize, int64_t pHeadDim, int64_t pMaxBlockNumPerBatch,
            int64_t pBlockSize, int64_t pSeqSize, int64_t pHeadDimV, int64_t pHeadSizeV, 
            int64_t pSelectedBlockSize, int64_t pSelectedBlockCount, int64_t pSparseMode,
            float pScaleValue, std::string pInputLayout, ge::DataType pOptionalDataType, 
            std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualSeqKVLenList, int64_t numtokens)
    : batchSize(pBatchSize), qSeqSize(pQSeqSize), headSize(pHeadSize), headDim(pHeadDim),
      maxBlockNumPerBatch(pMaxBlockNumPerBatch), blockSize(pBlockSize), seqSize(pSeqSize),
      headDimV(pHeadDimV), headSizeV(pHeadSizeV), selectedBlockSize(pSelectedBlockSize), selectedBlockCount(pSelectedBlockCount),
      sparseMode(pSparseMode), scaleValue(pScaleValue), inputLayout(pInputLayout), optionalDataType(pOptionalDataType), actualSeqQLenList(std::move(pActualSeqQLenList)),
      actualSeqKVLenList(std::move(pActualSeqKVLenList)), numtokens(numtokens)
{
}

void NsaSelectAttentionInferParam::BSNDInitParam()
{
    query = Tensor("query", {batchSize, qSeqSize, headSize, headDim}, inputLayout.c_str(), optionalDataType,
        ge::FORMAT_ND);
    key = Tensor("key", {maxBlockNumPerBatch, blockSize, headSizeV, headDim}, inputLayout.c_str(), optionalDataType,
        ge::FORMAT_ND);
    value = Tensor("value", {maxBlockNumPerBatch, blockSize, headSizeV, headDimV}, inputLayout.c_str(), optionalDataType,
        ge::FORMAT_ND);
    attentionOut = Tensor("attentionOut", {batchSize, qSeqSize, headSize, headDimV}, inputLayout.c_str(),
        optionalDataType, ge::FORMAT_ND);
}

void NsaSelectAttentionInferParam::BSHInitParam()
{
    query = Tensor("query", {batchSize, qSeqSize, headSize*headDim}, inputLayout.c_str(), optionalDataType,
        ge::FORMAT_ND);
    key = Tensor("key", {maxBlockNumPerBatch, blockSize, headSizeV*headDim}, inputLayout.c_str(), optionalDataType,
        ge::FORMAT_ND);
    value = Tensor("value", {maxBlockNumPerBatch, blockSize, headSizeV*headDimV}, inputLayout.c_str(), optionalDataType,
        ge::FORMAT_ND);
    attentionOut = Tensor("attentionOut", {batchSize, qSeqSize, headSize*headDimV}, inputLayout.c_str(),
        optionalDataType, ge::FORMAT_ND);
}

void NsaSelectAttentionInferParam::TNDInitParam()
{
    query = Tensor("query", {numtokens, headSize, headDim}, inputLayout.c_str(), optionalDataType,
        ge::FORMAT_ND);
    key = Tensor("key", {maxBlockNumPerBatch, blockSize, headSizeV * headDim}, inputLayout.c_str(), optionalDataType,
        ge::FORMAT_ND);
    value = Tensor("value", {maxBlockNumPerBatch, blockSize, headSizeV * headDimV}, inputLayout.c_str(), optionalDataType,
        ge::FORMAT_ND);
    attentionOut = Tensor("attentionOut", {numtokens, headSize , headDimV}, inputLayout.c_str(),
        optionalDataType, ge::FORMAT_ND);
}

void NsaSelectAttentionInferParam::TopkInitParam()
{
    if (inputLayout == "TND") {
        InitTopkDataTND();
    } else {
        InitTopkDataOtherLayout();
    }
}

void NsaSelectAttentionInferParam::InitTopkDataTND()
{
    for (int tokenId = 0; tokenId < numtokens; tokenId++) {
        InitTopkDataForHead();
    }
}

void NsaSelectAttentionInferParam::InitTopkDataForHead()
{
    for (int headSizeId = 0; headSizeId < headSizeV; headSizeId++) {
        InitTopkDataForSelectedBlockSize();
    }
}

void NsaSelectAttentionInferParam::InitTopkDataForSelectedBlockSize()
{
    for (int selectId = 0; selectId < selectedBlockCount; selectId++) {
        topkData.push_back(0);
    }
}

void NsaSelectAttentionInferParam::InitTopkDataOtherLayout()
{
    for (int batchId = 0; batchId < batchSize; batchId++) {
        InitTopkDataForS1();
    }
}

void NsaSelectAttentionInferParam::InitTopkDataForS1()
{
    for (int sId = 0; sId < qSeqSize; sId++) {
        InitTopkDataForHead();
    }
}

bool NsaSelectAttentionInferParam::Init()
{
    if (inputLayout == "BSND") {
        BSNDInitParam();
    } else if (inputLayout == "TND") {
        TNDInitParam();
    } else {
        BSHInitParam();
    }

    if (inputLayout == "TND") {
        topkIndices = Tensor("topkIndices", {numtokens, headSizeV, selectedBlockCount}, inputLayout.c_str(), ge::DT_INT32,
            ge::FORMAT_ND);
    } else {
        topkIndices = Tensor("topkIndices", {batchSize, qSeqSize, headSizeV, selectedBlockCount}, inputLayout.c_str(), ge::DT_INT32,
            ge::FORMAT_ND);
    }
    blocktable = Tensor("blocktable", {batchSize, maxBlockNumPerBatch}, inputLayout.c_str(), ge::DT_INT32, ge::FORMAT_ND);
    
    if (!actualSeqQLenList.empty()) {
        actualQSeqLengths = Tensor("actualQSeqLengths", {batchSize}, inputLayout.c_str(), ge::DataType::DT_INT64, ge::FORMAT_ND,
                                    Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        LOG_ERR("Currently, actualQSeqLengths must not be empty.");
    }

    TopkInitParam();

    if (!actualSeqKVLenList.empty()) {
        actualKvSeqLengths = Tensor("actualKvSeqLengths", {batchSize}, inputLayout.c_str(), ge::DataType::DT_INT64, ge::FORMAT_ND,
                                    Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        LOG_ERR("Currently, actualKvSeqLengths must not be empty.");
    }

    for (int blockId = 0; blockId < batchSize * maxBlockNumPerBatch; blockId++) {
        blocktableData.push_back(0);
    }

    if (!ops::adv::tests::NsaSelectAttentionInfer::NsaSelectAttentionInferParam::InitTensor(actualKvSeqLengths, actualSeqKVLenList)) {
        return false;
    }

    if (!ops::adv::tests::NsaSelectAttentionInfer::NsaSelectAttentionInferParam::InitTensor(actualQSeqLengths, actualSeqQLenList)) {
        return false;
    }

    if (!ops::adv::tests::NsaSelectAttentionInfer::NsaSelectAttentionInferParam::InitTensor(topkIndices, topkData)) {
        return false;
    }
    if (!ops::adv::tests::NsaSelectAttentionInfer::NsaSelectAttentionInferParam::InitTensor(blocktable, blocktableData)) {
        return false;
    }
    return true;
}
