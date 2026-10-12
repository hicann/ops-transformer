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
 * \file ts_aclnn_nsa_compress_with_cache.cpp
 * \brief NsaCompressWithCache ACLNN 测试用例.
 */

#include "ts_aclnn_nsa_compress_with_cache.h"

namespace {
TEST_P(Ts_Aclnn_NsaCompressWithCache_WithParam_Ascend910B3, Tc_Aclnn_NsaCompressWithCache)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_NsaCompressWithCache_Aclnn_Case = ::testing::Values(

    AclnnNsaCompressWithCacheCase("NsaCompressWithCache_Case_1", true, "",         /* CaseName, Enable, DebugInfo */
                                  OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                         ExpectInfo(true,                          /* ExpectSuccess */
                                                    ExpectInfo::kInvalidTilingKey, /* ExpectTilingKey */
                                                    ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                                  AclnnNsaCompressWithCacheParam(4, 32, 16, /* batchSize, headNum, headDim */
                                                                 ge::DataType::DT_FLOAT16, /* Dtype */
                                                                 {144, 192, 10, 45},       /* ActualSeqLenList */
                                                                 LayoutType::TND,          /* Layout */
                                                                 16,                       /*CompressBlockSize*/
                                                                 16,                       /*CompressStride*/
                                                                 1,                        /*ActSeqLenType*/
                                                                 128                       /*PageBlockSize*/
                                                                 ))

);

INSTANTIATE_TEST_SUITE_P(NsaCompressWithCache, Ts_Aclnn_NsaCompressWithCache_WithParam_Ascend910B3,
                         Tc_NsaCompressWithCache_Aclnn_Case);
} // namespace
