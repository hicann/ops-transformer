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
 * \file ts_mla_prolog_v3_tc_full_quant_int8.cpp
 * \brief MlaPrologV3 全量化KvCache量化用例.
 */

#include "ts_mla_prolog_v3.h"

using Tensor = ops::adv::tests::utils::Tensor;

class Ts_MlaPrologV3_Ascend910B2_tc_full_quant_int8 : public Ts_MlaPrologV3_WithParam_Ascend910B2 {};

TEST_P(Ts_MlaPrologV3_Ascend910B2_tc_full_quant_int8, Tc_Case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_MlaPrologV3_Full_Quant_Int8_Case = ::testing::Values(
	// dataType不符合要求
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00000", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{Tensor("tokenX", {48, 16, 7168}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00001", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00002", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{Tensor("weightUqQr", {1536, 12544}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00003", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{Tensor("weightUk", {64, 128, 512}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00004", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00005", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{Tensor("rmsnormGammaCq", {1536}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00006", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{Tensor("rmsnormGammaCkv", {512}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00007", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("ropeSin", {48, 16, 64}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00008", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("ropeCos", {48, 16, 64}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00009", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("cacheIndex", {48, 16}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00010", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("kvCache", {16, 128, 1, 1024}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00011", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00012", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleX", {768, 1}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00013", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00014", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleWUqQr", {1, 12544}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00015", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleWDkvKr", {1, 576}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00016", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00017", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	//维数不符合要求
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00018", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{Tensor("tokenX", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00019", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{Tensor("weightDq", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00020", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{Tensor("weightUqQr", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00021", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{Tensor("weightUk", {1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00022", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{Tensor("weightDkvKr", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00023", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{Tensor("rmsnormGammaCq", {1, 1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00024", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{},                                             /* weightDkvKr */
				{},                                             /* rmsnormGammaCq */
				{Tensor("rmsnormGammaCkv", {1, 1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
				{},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00025", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("ropeSin", {1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00026", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("ropeCos", {1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00027", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("cacheIndex", {1, 1, 1}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00028", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("kvCache", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00029", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("krCache", {1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00030", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleX", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00031", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleWDq", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00032", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00033", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleWDkvKr", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00034", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00035", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	// 输入shape不符合要求
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00036", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{Tensor("weightDq", {8000, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00037", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 128, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
				{Tensor("weightUqQr", {1536, 20000}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00038", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 128, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{Tensor("weightUk", {64, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00039", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(48, 16, 7168, 1536, 512, 128, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00040", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
				1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
				CacheModeType::PA_BSND,                         /* CacheModeType */
				0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
				{},                                             /* tokenX */
				{},                                             /* weightDq */
				{},                                             /* weightUqQr */
				{},                                             /* weightUk */
				{Tensor("weightDkvKr", {7168, 577}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00041", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00042", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00043", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("ropeSin", {2, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
				{},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00044", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(2, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("ropeCos", {2, 1, 65}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
				{},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00045", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(11, 1, 1, 7168, 1536, 512, 64, 128, 64,             /* T, B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("cacheIndex", {12}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
				{},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00046", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("kvCache", {1024, 128, 1, 513}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
				{},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00047", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("kvCache", {1024, 128, 1, 512}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
				{Tensor("krCache", {1024, 16, 1, 513}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
				{},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00048", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleX", {8, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
				{},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00049", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleWDq", {1, 20000}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
				{},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00050", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleWUqQr", {1, 20000}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
				{},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00051", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("dequantScaleWDkvKr", {1, 20000}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00052", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1, 513}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		),
	MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_Int8_00053", true,                    /* CaseName, Enable */
		"",                                                     /* DebugInfo */
		OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
			   ExpectInfo(false,                                 /* ExpectSuccess */
						  ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
						  ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
		MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
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
				{Tensor("quantScaleCkv", {1, 512}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* quantScaleCkv */
				{},                                             /* quantScaleCkr */
				{Tensor("smoothScalesCq", {1, 1537}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
				{},                                             /* actualSeqLen */
				{},                                             /* kvCacheOut */
				{},                                             /* krCacheOut */
				{},                                             /* queryOut */
				{},                                             /* queryRopeOut */
				{},                                             /* dequantScaleQNopeOut */
				{},                                             /* queryNormOut */
				{})                                             /* dequantScaleQNormOut */
		)
);
INSTANTIATE_TEST_SUITE_P(MlaPrologV3, Ts_MlaPrologV3_Ascend910B2_tc_full_quant_int8, Tc_MlaPrologV3_Full_Quant_Int8_Case);