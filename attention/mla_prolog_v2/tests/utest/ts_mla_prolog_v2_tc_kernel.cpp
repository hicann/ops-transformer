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
 * \file ts_mla_prolog_v2_tc_kernel.cpp
 * \brief MlaPrologV2 正反向用例.
 */

#include "ts_mla_prolog_v2.h"

using Tensor = ops::adv::tests::utils::Tensor;

class Ts_MlaPrologV2_Ascend910B2_tc_kernel : public Ts_MlaPrologV2_WithParam_Ascend910B2 {};

TEST_P(Ts_MlaPrologV2_Ascend910B2_tc_kernel, Tc_Case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_MlaPrologV2_Case = ::testing::Values(

// 全量化kv量化
    MlaPrologV2Case("MlaPrologV2_Tc_kernel_00001", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, true),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             1249,        /* ExpectTilingKey */
                             24)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {64, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1, 576}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
//     全量化kv不量化
    MlaPrologV2Case("MlaPrologV2_Tc_kernel_00002", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, true),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             1185,        /* ExpectTilingKey */
                             24)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {64, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1, 576}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {32, 2, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
   // 半量化kv量化
    MlaPrologV2Case("MlaPrologV2_Tc_kernel_00003", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, true),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             1121,        /* ExpectTilingKey */
                             24)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {Tensor("weightUk", {64, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {Tensor("rmsnormGammaCq", {1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                   {Tensor("rmsnormGammaCkv", {512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                   {Tensor("ropeSin", {32, 2, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                   {Tensor("ropeCos", {32, 2, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                   {Tensor("cacheIndex", {32, 2}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                   {Tensor("dequantScaleX", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                   {Tensor("quantScaleCkr", {1, 64}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {Tensor("krCacheOut", {16, 128, 1, 64}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* krCacheOut */
                   {Tensor("query", {32, 2, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
//    // 半量化kv不量化
    MlaPrologV2Case("MlaPrologV2_Tc_kernel_00004", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, true),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             1057,        /* ExpectTilingKey */
                             24)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {Tensor("weightUk", {64, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {Tensor("rmsnormGammaCq", {1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                   {Tensor("rmsnormGammaCkv", {512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                   {Tensor("ropeSin", {32, 2, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                   {Tensor("ropeCos", {32, 2, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                   {Tensor("cacheIndex", {32, 2}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                   {Tensor("dequantScaleX", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12288}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                   {Tensor("quantScaleCkr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {Tensor("krCacheOut", {16, 128, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* krCacheOut */
                   {Tensor("query", {32, 2, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
   // 非量化
    MlaPrologV2Case("MlaPrologV2_Tc_kernel_00005", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, true),                        /* RunTiling, RunKernel */
                  ExpectInfo(true,                                 /* ExpectSuccess */
                             17,        /* ExpectTilingKey */
                             24)), /* ExpectTilingBlockDim */
           MlaPrologV2Param(32, 2, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {32, 2, 7168}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12288}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {Tensor("weightUk", {64, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {Tensor("rmsnormGammaCq", {1536}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                   {Tensor("rmsnormGammaCkv", {512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                   {Tensor("ropeSin", {32, 2, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                   {Tensor("ropeCos", {32, 2, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                   {Tensor("cacheIndex", {32, 2}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                   {Tensor("dequantScaleX", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {Tensor("quantScaleCkv", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
                   {Tensor("quantScaleCkr", {}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkr */
                   {},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {Tensor("krCacheOut", {16, 128, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* krCacheOut */
                   {Tensor("query", {32, 2, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           )
);
INSTANTIATE_TEST_SUITE_P(MlaPrologV2, Ts_MlaPrologV2_Ascend910B2_tc_kernel, Tc_MlaPrologV2_Case);