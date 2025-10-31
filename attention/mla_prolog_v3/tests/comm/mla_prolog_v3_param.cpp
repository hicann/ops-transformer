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
 * \file mla_prolog_v3_param.cpp
 * \brief MlaPrologV3 参数信息.
 */

#include "mla_prolog_v3_param.h"
#include "tests/utils/log.h"

using Tensor = ops::adv::tests::utils::Tensor;

using namespace ops::adv::tests::MlaPrologV3;

MlaPrologV3Param::MlaPrologV3Param(int64_t pB, int64_t pS, int64_t pHe, int64_t pHcq, int64_t pHckv, int64_t pN, int64_t pD, int64_t pDr, int64_t pSkv, int64_t pNkv, int64_t pBlockSize, int64_t pBlockNum,
    CacheModeType pCacheModeType, float pRmsnormEpsilonCq, float pRmsnormEpsilonCkv)
    : MlaPrologV3Param(pB, pS, pHe, pHcq, pHckv, pN, pD, pDr, pSkv, pNkv, pBlockSize, pBlockNum, pCacheModeType,
        pRmsnormEpsilonCq, pRmsnormEpsilonCkv, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
{
}

MlaPrologV3Param::MlaPrologV3Param(int64_t pB, int64_t pS, int64_t pHe, int64_t pHcq, int64_t pHckv, int64_t pN,
    int64_t pD, int64_t pDr, int64_t pSkv, int64_t pNkv, int64_t pBlockSize, int64_t pBlockNum,
    CacheModeType pCacheModeType, float pRmsnormEpsilonCq, float pRmsnormEpsilonCkv,
    std::vector<Tensor> pTokenXData, std::vector<Tensor> pWeightDqData,
    std::vector<Tensor> pWeightUqQrData, std::vector<Tensor> pWeightUkData,
    std::vector<Tensor> pWeightDkvKrData, std::vector<Tensor> pRmsnormGammaCqData,
    std::vector<Tensor> pRmsnormGammaCkvData, std::vector<Tensor> pRopeSinData,
    std::vector<Tensor> pRopeCosData, std::vector<Tensor> pCacheIndexData,
    std::vector<Tensor> pKvCacheData, std::vector<Tensor> pKrCacheData,
    std::vector<Tensor> pDequantScaleXData, std::vector<Tensor> pDequantScaleWDqData,
    std::vector<Tensor> pDequantScaleWUqQrData, std::vector<Tensor> pDequantScaleWDkvKrData,
    std::vector<Tensor> pQuantScaleCkvData, std::vector<Tensor> pQuantScaleCkrData,
    std::vector<Tensor> pSmoothScalesCqData, std::vector<Tensor> pActualSeqLenData,
    std::vector<Tensor> pKvCacheOutData,std::vector<Tensor> pKrCacheOutData, std::vector<Tensor> pQueryData,
    std::vector<Tensor> pQueryRopeData, std::vector<Tensor> pDequantScaleQNopeData,
    std::vector<Tensor> pQueryNormOutData, std::vector<Tensor> pDequantScaleQNormOutData)
    : B(pB), S(pS), He(pHe), Hcq(pHcq), Hckv(pHckv), N(pN), D(pD), Dr(pDr), Skv(pSkv),
      Nkv(pNkv), BlockSize(pBlockSize), BlockNum(pBlockNum), cacheModeType(pCacheModeType),
      rmsnormEpsilonCq(pRmsnormEpsilonCq), rmsnormEpsilonCkv(pRmsnormEpsilonCkv),
      tokenXData(std::move(pTokenXData)), weightDqData(std::move(pWeightDqData)),
      weightUqQrData(std::move(pWeightUqQrData)), weightUkData(std::move(pWeightUkData)),
      weightDkvKrData(std::move(pWeightDkvKrData)), rmsnormGammaCqData(std::move(pRmsnormGammaCqData)),
      rmsnormGammaCkvData(std::move(pRmsnormGammaCkvData)), ropeSinData(std::move(pRopeSinData)),
      ropeCosData(std::move(pRopeCosData)), cacheIndexData(std::move(pCacheIndexData)),
      kvCacheData(std::move(pKvCacheData)), krCacheData(std::move(pKrCacheData)),
      dequantScaleXData(std::move(pDequantScaleXData)), dequantScaleWDqData(std::move(pDequantScaleWDqData)),
      dequantScaleWUqQrData(std::move(pDequantScaleWUqQrData)),
      dequantScaleWDkvKrData(std::move(pDequantScaleWDkvKrData)),
      quantScaleCkvData(std::move(pQuantScaleCkvData)),
      quantScaleCkrData(std::move(pQuantScaleCkrData)),
      smoothScalesCqData(std::move(pSmoothScalesCqData)),
      actualSeqLenData(std::move(pActualSeqLenData)),
      kvCacheOutData(std::move(pKvCacheOutData)),
      krCacheOutData(std::move(pKrCacheOutData)),
      queryData(std::move(pQueryData)),
      queryRopeData(std::move(pQueryRopeData)),
      dequantScaleQNopeData(std::move(pDequantScaleQNopeData)),
      queryNormOutData(std::move(pQueryNormOutData)),
      dequantScaleQNormOutData(std::move(pDequantScaleQNormOutData))
{
}

// for TND
MlaPrologV3Param::MlaPrologV3Param(int64_t pT, int64_t pB, int64_t pS, int64_t pHe, int64_t pHcq, int64_t pHckv, int64_t pN, int64_t pD, int64_t pDr, int64_t pSkv, int64_t pNkv, int64_t pBlockSize, int64_t pBlockNum,
    CacheModeType pCacheModeType, float pRmsnormEpsilonCq, float pRmsnormEpsilonCkv)
    : MlaPrologV3Param(pT, pB, pS, pHe, pHcq, pHckv, pN, pD, pDr, pSkv, pNkv, pBlockSize, pBlockNum, pCacheModeType,
        pRmsnormEpsilonCq, pRmsnormEpsilonCkv, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
{
}

MlaPrologV3Param::MlaPrologV3Param(int64_t pT, int64_t pB, int64_t pS, int64_t pHe, int64_t pHcq, int64_t pHckv, int64_t pN,
    int64_t pD, int64_t pDr, int64_t pSkv, int64_t pNkv, int64_t pBlockSize, int64_t pBlockNum,
    CacheModeType pCacheModeType, float pRmsnormEpsilonCq, float pRmsnormEpsilonCkv,
    std::vector<Tensor> pTokenXData, std::vector<Tensor> pWeightDqData,
    std::vector<Tensor> pWeightUqQrData, std::vector<Tensor> pWeightUkData,
    std::vector<Tensor> pWeightDkvKrData, std::vector<Tensor> pRmsnormGammaCqData,
    std::vector<Tensor> pRmsnormGammaCkvData, std::vector<Tensor> pRopeSinData,
    std::vector<Tensor> pRopeCosData, std::vector<Tensor> pCacheIndexData,
    std::vector<Tensor> pKvCacheData, std::vector<Tensor> pKrCacheData,
    std::vector<Tensor> pDequantScaleXData, std::vector<Tensor> pDequantScaleWDqData,
    std::vector<Tensor> pDequantScaleWUqQrData, std::vector<Tensor> pDequantScaleWDkvKrData,
    std::vector<Tensor> pQuantScaleCkvData, std::vector<Tensor> pQuantScaleCkrData,
    std::vector<Tensor> pSmoothScalesCqData, std::vector<Tensor> pActualSeqLenData,
    std::vector<Tensor> pKvCacheOutData,std::vector<Tensor> pKrCacheOutData, std::vector<Tensor> pQueryData,
    std::vector<Tensor> pQueryRopeData, std::vector<Tensor> pDequantScaleQNopeData,
    std::vector<Tensor> pQueryNormOutData, std::vector<Tensor> pDequantScaleQNormOutData)
    : T(pT), B(pB), S(pS), He(pHe), Hcq(pHcq), Hckv(pHckv), N(pN), D(pD), Dr(pDr), Skv(pSkv),
      Nkv(pNkv), BlockSize(pBlockSize), BlockNum(pBlockNum), cacheModeType(pCacheModeType),
      rmsnormEpsilonCq(pRmsnormEpsilonCq), rmsnormEpsilonCkv(pRmsnormEpsilonCkv),
      tokenXData(std::move(pTokenXData)), weightDqData(std::move(pWeightDqData)),
      weightUqQrData(std::move(pWeightUqQrData)), weightUkData(std::move(pWeightUkData)),
      weightDkvKrData(std::move(pWeightDkvKrData)), rmsnormGammaCqData(std::move(pRmsnormGammaCqData)),
      rmsnormGammaCkvData(std::move(pRmsnormGammaCkvData)), ropeSinData(std::move(pRopeSinData)),
      ropeCosData(std::move(pRopeCosData)), cacheIndexData(std::move(pCacheIndexData)),
      kvCacheData(std::move(pKvCacheData)), krCacheData(std::move(pKrCacheData)),
      dequantScaleXData(std::move(pDequantScaleXData)), dequantScaleWDqData(std::move(pDequantScaleWDqData)),
      dequantScaleWUqQrData(std::move(pDequantScaleWUqQrData)),
      dequantScaleWDkvKrData(std::move(pDequantScaleWDkvKrData)),
      quantScaleCkvData(std::move(pQuantScaleCkvData)),
      quantScaleCkrData(std::move(pQuantScaleCkrData)),
      smoothScalesCqData(std::move(pSmoothScalesCqData)),
      actualSeqLenData(std::move(pActualSeqLenData)),
      kvCacheOutData(std::move(pKvCacheOutData)),
      krCacheOutData(std::move(pKrCacheOutData)),
      queryData(std::move(pQueryData)),
      queryRopeData(std::move(pQueryRopeData)),
      dequantScaleQNopeData(std::move(pDequantScaleQNopeData)),
      queryNormOutData(std::move(pQueryNormOutData)),
      dequantScaleQNormOutData(std::move(pDequantScaleQNormOutData))
{
}


bool MlaPrologV3Param::Init()
{
    std::vector<int64_t> shape1;
    std::vector<int64_t> shape2;
    switch (cacheModeType) {
        case CacheModeType::PA_BSND:
            cacheMode = "PA_BSND";
            shape1 = {BlockNum, BlockSize, Nkv, Hckv};
            shape2 = {BlockNum, BlockSize, Nkv, Dr};
            break;
        case CacheModeType::PA_NZ:
            cacheMode = "PA_NZ";
            shape1 = {BlockNum, BlockSize, Nkv, Hckv};
            shape2 = {BlockNum, BlockSize, Nkv, Dr};
            break;
        case CacheModeType::BNSD:
            cacheMode = "BNSD";
            shape1 = {B, Nkv, Skv, Hckv};
            shape2 = {B, Nkv, Skv, Dr};
            break;
        default:
            LOG_ERR("Unknown CacheModeType=%d", static_cast<int32_t>(cacheModeType));
            return false;
    }

    if (T) {
        mTensorList["tokenX"] = Tensor("tokenX", {T, He}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
        mTensorList["ropeSin"] = Tensor("ropeSin", {T, Dr}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
        mTensorList["ropeCos"] = Tensor("ropeCos", {T, Dr}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
        mTensorList["cacheIndex"] = Tensor("cacheIndex", {T}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);

        mTensorList["query"] = Tensor("query", {T, N, Hckv}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
        mTensorList["queryRope"] = Tensor("queryRope", {T, N, Dr}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
        mTensorList["dequantScaleQNopeOut"] = Tensor("dequantScaleQNopeOut", {T, N, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
        mTensorList["queryNormOut"] = Tensor("queryNormOut", {T, N, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
        mTensorList["dequantScaleQNormOut"] = Tensor("dequantScaleQNormOut", {T, N, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);

        mTensorList["dequantScaleX"] = Tensor("dequantScaleX", {T, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        mTensorList["tokenX"] = Tensor("tokenX", {B, S, He}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
        mTensorList["ropeSin"] = Tensor("ropeSin", {B, S, Dr}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
        mTensorList["ropeCos"] = Tensor("ropeCos", {B, S, Dr}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
        mTensorList["cacheIndex"] = Tensor("cacheIndex", {B, S}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);

        mTensorList["query"] = Tensor("query", {B, S, N, Hckv}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
        mTensorList["queryRope"] = Tensor("queryRope", {B, S, N, Dr}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
        mTensorList["dequantScaleQNopeOut"] = Tensor("dequantScaleQNopeOut", {B*S, N, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
        mTensorList["queryNormOut"] = Tensor("queryNormOut", {B*S, N, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
        mTensorList["dequantScaleQNormOut"] = Tensor("dequantScaleQNormOut", {B*S, N, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);

        mTensorList["dequantScaleX"] = Tensor("dequantScaleX", {B*S, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    }
    mTensorList["weightDq"] = Tensor("weightDq", {He, Hcq}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT);
    mTensorList["weightUqQr"] = Tensor("weightUqQr", {Hcq, N*(D+Dr)}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT);
    mTensorList["weightUk"] = Tensor("weightUk", {N, D, Hckv}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    mTensorList["weightDkvKr"] = Tensor("weightDkvKr", {He, (Hckv+Dr)}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT);
    mTensorList["rmsnormGammaCq"] = Tensor("rmsnormGammaCq", {Hcq}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    mTensorList["rmsnormGammaCkv"] = Tensor("rmsnormGammaCkv", {Hckv}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    mTensorList["kvCache"] = Tensor("kvCache", shape1, cacheMode.c_str(), ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    mTensorList["krCache"] = Tensor("krCache", shape2, cacheMode.c_str(), ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);

    mTensorList["dequantScaleWDq"] = Tensor("dequantScaleWDq", {1, Hcq}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    mTensorList["dequantScaleWUqQr"] = Tensor("dequantScaleWUqQr", {1, N * (D + Dr)}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    mTensorList["dequantScaleWDkvKr"] = Tensor("dequantScaleWDkvKr", {1, Hckv + Dr}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    mTensorList["quantScaleCkv"] = Tensor("quantScaleCkv", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    mTensorList["quantScaleCkr"] = Tensor("quantScaleCkr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    mTensorList["smoothScalesCq"] = Tensor("smoothScalesCq", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    mTensorList["actualSeqLen"] = Tensor("actualSeqLen", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);

    mTensorList["kvCacheOut"] = Tensor("kvCacheOut", shape1, cacheMode.c_str(), ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
    mTensorList["krCacheOut"] = Tensor("krCacheOut", shape2, cacheMode.c_str(), ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);

    return InitParam();
}

bool MlaPrologV3Param::InitParam()
{
    std::vector<std::vector<Tensor>> paramListData = {tokenXData, weightDqData, weightUqQrData, weightUkData, weightDkvKrData,
        rmsnormGammaCqData, rmsnormGammaCkvData, ropeSinData, ropeCosData, kvCacheData, krCacheData, cacheIndexData,
        dequantScaleXData, dequantScaleWDqData, dequantScaleWUqQrData, dequantScaleWDkvKrData, quantScaleCkvData,
        quantScaleCkrData, smoothScalesCqData, actualSeqLenData, kvCacheOutData, krCacheOutData, queryData, queryRopeData, dequantScaleQNopeData, queryNormOutData, dequantScaleQNormOutData};

    for (size_t i = 0; i < paramListData.size(); i++) {
        if (!paramListData[i].empty()) {
            mTensorList[paramListData[i][0].Name()] = paramListData[i][0];
        }
    }
    return true;
}
