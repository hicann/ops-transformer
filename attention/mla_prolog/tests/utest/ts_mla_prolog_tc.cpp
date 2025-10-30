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
 * \file ts_mla_prolog_tc.cpp
 * \brief MlaProlog 正反向用例.
 */

#include "ts_mla_prolog.h"

using Tensor = ops::adv::tests::utils::Tensor;

class Ts_MlaProlog_Ascend910B2_tc : public Ts_MlaProlog_WithParam_Ascend910B2 {};

TEST_P(Ts_MlaProlog_Ascend910B2_tc, Tc_Case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_MlaProlog_Case = ::testing::Values(
    // N
    MlaPrologCase("MlaProlog_Tc_000", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {})                                             /* smoothScalesCq */
           ),
    MlaPrologCase("MlaProlog_Tc_001", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 16, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {})                                             /* smoothScalesCq */
           ),
    MlaPrologCase("MlaProlog_Tc_002", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologParam(32, 2, 7168, 1536, 512, 128, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 16, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {})                                             /* smoothScalesCq */
           ),
//cachemode
    MlaPrologCase("MlaProlog_Tc_003", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {})                                             /* smoothScalesCq */
           ),
    MlaPrologCase("MlaProlog_Tc_004", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_NZ,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {})                                             /* smoothScalesCq */
           ),
    MlaPrologCase("MlaProlog_Tc_005", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {})                                             /* smoothScalesCq */
           ),
   // tokenX dimNum
       MlaPrologCase("MlaProlog_Tc_006", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   // tokenX datatype
       MlaPrologCase("MlaProlog_Tc_007", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_INT4, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //weightDqDim=4
       MlaPrologCase("MlaProlog_Tc_008", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {Tensor("weightDq", {1536/16, 7168/16, 16, 16}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //weightDq dimNum
       MlaPrologCase("MlaProlog_Tc_009", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {Tensor("weightDq", {16}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightDq datatype
       MlaPrologCase("MlaProlog_Tc_010", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {Tensor("weightDq", {1536/16, 7168/16, 16, 16}, "1", ge::DataType::DT_INT64, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightDq format
       MlaPrologCase("MlaProlog_Tc_011", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {Tensor("weightDq", {1536/16, 7168/16, 16, 16}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //weightUqQrDim=4
       MlaPrologCase("MlaProlog_Tc_012", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {32*(128+64)/16, 1536/16, 16, 16}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ), 
       // weightUqQr dimNum
       MlaPrologCase("MlaProlog_Tc_013", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {32*(128+64)/16}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightUqQr datatype
       MlaPrologCase("MlaProlog_Tc_014", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {32*(128+64)/16, 1536/16, 16, 16}, "1", ge::DataType::DT_INT32, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightUqQr format
       MlaPrologCase("MlaProlog_Tc_015", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {32*(128+64)/16, 1536/16, 16, 16}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //weightUk dimNum
       MlaPrologCase("MlaProlog_Tc_016", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {Tensor("weightUk", {32}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //weightDkvKr dimNum
       MlaPrologCase("MlaProlog_Tc_017", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {Tensor("weightDkvKr", {(512+64)/16}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightDkvKr datatype
       MlaPrologCase("MlaProlog_Tc_018", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {Tensor("weightDkvKr", {(512+64)/16, 7168/16, 16, 16}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightDkvKr format
       MlaPrologCase("MlaProlog_Tc_019", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {Tensor("weightDkvKr", {(512+64)/16, 7168/16, 16, 16}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //rmsnormGammaCq dimNum
       MlaPrologCase("MlaProlog_Tc_020", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {Tensor("rmsnormGammaCq", {1536, 2}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // rmsnormGammaCq datatype
      MlaPrologCase("MlaProlog_Tc_021", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {Tensor("rmsnormGammaCq", {1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // rmsnormGammaCkv dimNum
       MlaPrologCase("MlaProlog_Tc_022", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {Tensor("rmsnormGammaCkv", {512, 2}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // rmsnormGammaCkv datatype
       MlaPrologCase("MlaProlog_Tc_023", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {Tensor("rmsnormGammaCkv", {512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // CacheModeType=BNSD
       MlaPrologCase("MlaProlog_Tc_024", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::BNSD,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       MlaPrologCase("MlaProlog_Tc_025", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {Tensor("dequantScaleX", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                      {Tensor("dequantScaleWDq", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {Tensor("dequantScaleWDkvKr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
   
   //ropeSin dimNum
       MlaPrologCase("MlaProlog_Tc_026", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {Tensor("ropeSin", {32}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // ropeSin datatype
       MlaPrologCase("MlaProlog_Tc_027", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {Tensor("ropeSin", {32, 2, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //ropeCos dimNum
       MlaPrologCase("MlaProlog_Tc_028", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {Tensor("ropeCos", {32}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // ropeCos datatype
       MlaPrologCase("MlaProlog_Tc_029", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {Tensor("ropeCos", {32, 2, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //cacheIndex dimNum
       MlaPrologCase("MlaProlog_Tc_030", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {Tensor("cacheIndex", {32, 2, 2}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // cacheIndex datatype
       MlaPrologCase("MlaProlog_Tc_031", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {Tensor("cacheIndex", {32, 2}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //kvCache dimNum
       MlaPrologCase("MlaProlog_Tc_032", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {128}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // kvCache datatype
       MlaPrologCase("MlaProlog_Tc_033", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {32, 1, 1024, 512}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
   //krCache dimNum
       MlaPrologCase("MlaProlog_Tc_034", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {Tensor("krCache", {128}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // 
       MlaPrologCase("MlaProlog_Tc_035", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {Tensor("krCache", {32, 1, 1024, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // He
       MlaPrologCase("MlaProlog_Tc_036", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 1536, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // Hckv
       MlaPrologCase("MlaProlog_Tc_037", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 7168, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // Dr
       MlaPrologCase("MlaProlog_Tc_038", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 32, 128, 32,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // B
       MlaPrologCase("MlaProlog_Tc_039", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(103, 16, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       MlaPrologCase("MlaProlog_Tc_040", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(65537, 2, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // S
       MlaPrologCase("MlaProlog_Tc_041", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 17, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // N
       MlaPrologCase("MlaProlog_Tc_042", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(1, 1, 7168, 1536, 512, 8, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       MlaPrologCase("MlaProlog_Tc_043", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 7, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // TND
       MlaPrologCase("MlaProlog_Tc_044", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(8, 8, 32, 7168, 1536, 512, 32, 128, 64,             /* T, B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 64,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {8, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 6144}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {Tensor("weightUk", {32, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                      {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                      {Tensor("rmsnormGammaCq", {1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                      {Tensor("rmsnormGammaCkv", {512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                      {Tensor("ropeSin", {8, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                      {Tensor("ropeCos", {8, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                      {Tensor("cacheIndex", {8}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                      {Tensor("kvCache", {64, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {64, 128, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // TND with T exceeded
       MlaPrologCase("MlaProlog_Tc_045", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(1048577, 8, 32, 7168, 1536, 512, 32, 128, 64,             /* T, B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 64,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {1048577, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 6144}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {Tensor("weightUk", {32, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                      {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                      {Tensor("rmsnormGammaCq", {1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                      {Tensor("rmsnormGammaCkv", {512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                      {Tensor("ropeSin", {1048577, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                      {Tensor("ropeCos", {1048577, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                      {Tensor("cacheIndex", {1048577}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                      {Tensor("kvCache", {64, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {64, 128, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // dequant case
       MlaPrologCase("MlaProlog_Tc_046", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant 
       MlaPrologCase("MlaProlog_Tc_047", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic non-kvCache quant 
       MlaPrologCase("MlaProlog_Tc_048", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // TND
       MlaPrologCase("MlaProlog_Tc_049", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(true,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(8, 8, 32, 7168, 1536, 512, 32, 128, 64,             /* T, B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 64,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {8, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       MlaPrologCase("MlaProlog_Tc_050", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(8, 8, 32, 7168, 1536, 512, 32, 128, 64,             /* T, B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 64,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {8, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {Tensor("ropeSin", {8, 8, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {Tensor("cacheIndex", {8, 8}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - dequantScaleWUqQr datatype
       MlaPrologCase("MlaProlog_Tc_051", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - quantScaleCkv datatype
       MlaPrologCase("MlaProlog_Tc_052", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - quantScaleCkr datatype
       MlaPrologCase("MlaProlog_Tc_053", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - smoothScalesCq datatype
       MlaPrologCase("MlaProlog_Tc_054", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - dequantScaleWUqQr dimNum
       MlaPrologCase("MlaProlog_Tc_055", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - quantScaleCkv dimNum
       MlaPrologCase("MlaProlog_Tc_056", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - quantScaleCkr dimNum
       MlaPrologCase("MlaProlog_Tc_057", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - smoothScalesCq dimNum
       MlaPrologCase("MlaProlog_Tc_058", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - dequantScaleWUqQr shape
       MlaPrologCase("MlaProlog_Tc_059", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 20000}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - quantScaleCkv shape
       MlaPrologCase("MlaProlog_Tc_060", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 513}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - quantScaleCkr shape
       MlaPrologCase("MlaProlog_Tc_061", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {2, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // dynamic kvCache quant - smoothScalesCq shape
       MlaPrologCase("MlaProlog_Tc_062", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                      {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1, 1537}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // weightDq shape
       MlaPrologCase("MlaProlog_Tc_063", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {Tensor("weightDq", {8000, 1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightUqQr shape
       MlaPrologCase("MlaProlog_Tc_064", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 128, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 20000}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightUk shape
       MlaPrologCase("MlaProlog_Tc_065", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {Tensor("weightUk", {64, 128, 256}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightDkvKr shape
       MlaPrologCase("MlaProlog_Tc_066", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {1, 1, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {Tensor("weightDkvKr", {7168, 577}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // rmsnormGammaCq shape
       MlaPrologCase("MlaProlog_Tc_067", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {Tensor("rmsnormGammaCq", {1537}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // rmsnormGammaCkv shape
       MlaPrologCase("MlaProlog_Tc_068", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {Tensor("rmsnormGammaCkv", {513}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // ropeSin shape
       MlaPrologCase("MlaProlog_Tc_069", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {1, 1, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {Tensor("ropeSin", {2, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // ropeCos shape
       MlaPrologCase("MlaProlog_Tc_070", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {1, 1, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {Tensor("ropeSin", {1, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                      {Tensor("ropeCos", {2, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // cacheIndex shape
       MlaPrologCase("MlaProlog_Tc_071", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(11, 8, 32, 7168, 1536, 512, 32, 128, 64,             /* T, B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 64,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {11, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {Tensor("cacheIndex", {12}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // kvCache shape
       MlaPrologCase("MlaProlog_Tc_072", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {1024, 128, 1, 513}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // krCache shape
       MlaPrologCase("MlaProlog_Tc_073", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {1024, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {1024, 16, 1, 513}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // kvCacheType != krCacheType
       MlaPrologCase("MlaProlog_Tc_074", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {1024, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {Tensor("krCache", {1024, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // 强校验拦截
       // dequantScaleX
       MlaPrologCase("MlaProlog_Tc_075", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {Tensor("dequantScaleX", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // dequantScaleWUqQr
       MlaPrologCase("MlaProlog_Tc_076", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {Tensor("dequantScaleWUqQr", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // smoothScalesCq
       MlaPrologCase("MlaProlog_Tc_077", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {Tensor("smoothScalesCq", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)})                                             /* smoothScalesCq */
              ),
       // token = null
       MlaPrologCase("MlaProlog_Tc_078", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {Tensor("tokenX", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightDq = null
       MlaPrologCase("MlaProlog_Tc_079", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {Tensor("weightDq", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightUqQr = null
       MlaPrologCase("MlaProlog_Tc_080", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {Tensor("weightUqQr", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightUk = null
       MlaPrologCase("MlaProlog_Tc_081", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {Tensor("weightUk", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // weightDkvKr = null
       MlaPrologCase("MlaProlog_Tc_082", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {Tensor("weightDkvKr", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // rmsnormGammaCq = null
       MlaPrologCase("MlaProlog_Tc_083", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {Tensor("rmsnormGammaCq", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // rmsnormGammaCkv = null
       MlaPrologCase("MlaProlog_Tc_084", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {Tensor("rmsnormGammaCkv", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // ropeSin = null
       MlaPrologCase("MlaProlog_Tc_085", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {Tensor("ropeSin", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // ropeCos = null
       MlaPrologCase("MlaProlog_Tc_086", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {Tensor("ropeCos", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // cacheIndex = null
       MlaPrologCase("MlaProlog_Tc_087", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {Tensor("cacheIndex", {}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // kvCache = null
       MlaPrologCase("MlaProlog_Tc_088", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {Tensor("kvCache", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                      {},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // krCache = null
       MlaPrologCase("MlaProlog_Tc_089", true,                    /* CaseName, Enable */
              "",                                                     /* DebugInfo */
              OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                     ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              MlaPrologParam(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                      1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                      CacheModeType::PA_BSND,                         /* CacheModeType */
                      0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                      {},                                             /* tokenX */
                      {},                                             /* weightDq */
                      {},                                             /* weightUqQr */
                      {},                                             /* weightUk */
                      {},                                             /* weightDkvKr */
                      {},                                             /* rmsnormGammaCq */
                      {},                                             /* rmsnormGammaCkv */
                      {},                                             /* ropeSin */
                      {},                                             /* ropeCos */
                      {},                                             /* cacheIndex */
                      {},                                             /* kvCache */
                      {Tensor("krCache", {}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                      {},                                             /* dequantScaleX */
                      {},                                             /* dequantScaleWDq */
                      {},                                             /* dequantScaleWUqQr */
                      {},                                             /* dequantScaleWDkvKr */
                      {},                                             /* quantScaleCkv */
                      {},                                             /* quantScaleCkr */
                      {})                                             /* smoothScalesCq */
              ),
       // 非量化kernel
       MlaPrologCase("MlaProlog_Tc_049", true,                    /* CaseName, Enable */
       "",                                                     /* DebugInfo */
       OpInfo(ControlInfo(true, true),                        /* RunTiling, RunKernel */
              ExpectInfo(true,                                 /* ExpectSuccess */
                            17,        /* ExpectTilingKey */
                            24)), /* ExpectTilingBlockDim */
       MlaPrologParam(8, 8, 32, 7168, 1536, 512, 32, 128, 64,             /* T, B, S, He, Hcq, Hckv, N, D, Dr */
                     1024, 1, 128, 64,                                   /* Skv, Nkv, BlockSize, BlockNum */
                     CacheModeType::PA_BSND,                         /* CacheModeType */
                     0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                     {Tensor("tokenX", {8, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                     {},                                             /* weightDq */
                     {},                                             /* weightUqQr */
                     {},                                             /* weightUk */
                     {},                                             /* weightDkvKr */
                     {},                                             /* rmsnormGammaCq */
                     {},                                             /* rmsnormGammaCkv */
                     {},                                             /* ropeSin */
                     {},                                             /* ropeCos */
                     {},                                             /* cacheIndex */
                     {},                                             /* kvCache */
                     {},                                             /* krCache */
                     {},                                             /* dequantScaleX */
                     {},                                             /* dequantScaleWDq */
                     {},                                             /* dequantScaleWUqQr */
                     {},                                             /* dequantScaleWDkvKr */
                     {},                                             /* quantScaleCkv */
                     {},                                             /* quantScaleCkr */
                     {})                                             /* smoothScalesCq */
       )
);
INSTANTIATE_TEST_SUITE_P(MlaProlog, Ts_MlaProlog_Ascend910B2_tc, Tc_MlaProlog_Case);
