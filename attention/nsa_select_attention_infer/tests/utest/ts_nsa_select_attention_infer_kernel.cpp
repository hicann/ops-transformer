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
 * \file ts_nsa_select_attention_infer_kernel.cpp
 * \brief NsaSelectAttentionInfer kernel用例.
 */

#include "ts_nsa_select_attention_infer.h"
namespace {
    TEST_P(Ts_NsaSelectAttentionInfer_WithParam, Tc_Kernel_NsaSelectAttentionInfer)
    {
        ASSERT_TRUE(case_->Init());
        ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
    }
    const auto Tc_NsaSelectAttentionInfer_Kernel_Case = ::testing::Values(
        NsaSelectAttentionInferCase(
            "NsaSelectAttentionInfer_Case_1", true, "", /* CaseName, Enable, DebugInfo */
            OpInfo(ControlInfo(true, true),          /* RunKernel, RunKernel */
                ExpectInfo(false,                  /* ExpectSuccess */
                            1,                      /* ExpectKernelKey */
                            48)),                   /* ExpectKernelBlockDim */
            NsaSelectAttentionInferParam(
                30, 3, 1, 192, 2, /* batchSize, qSeqSize, headSize, headDim, maxBlockNumPerBatch*/
                128, 256, 128, 1, /* blockSize kvSeqSize, headDimV, headSizeV*/
                128, 2, 0, 1.0f, /* selectedBlockSize selectedBlockCount, sparseMode, scaleValue*/      
                "TND",              /* inputLayout */
                ge::DT_FLOAT16,     /* optionalDataType */
                {3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                3,3,3,3,3,3,3,3,3,3,3,3,3,3,3},     /* actualSeqQLenList */
                {256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,
                256,256,256,256,256,256,256,256,256,256,256,256,256,256,256},      /* actualSeqKVLenList */
                90    /* numtokens */
        )),
        NsaSelectAttentionInferCase(
            "NsaSelectAttentionInfer_Case_2", true, "", /* CaseName, Enable, DebugInfo */
            OpInfo(ControlInfo(true, true),          /* RunKernel, RunKernel */
                ExpectInfo(false,                  /* ExpectSuccess */
                            0,                      /* ExpectKernelKey */
                            48)),                   /* ExpectKernelBlockDim */
            NsaSelectAttentionInferParam(
                30, 1, 1, 192, 2, /* batchSize, qSeqSize, headSize, headDim, maxBlockNumPerBatch*/
                128, 256, 128, 1, /* blockSize kvSeqSize, headDimV, headSizeV*/
                128, 2, 0, 1.0f, /* selectedBlockSize selectedBlockCount, sparseMode, scaleValue*/      
                "BSND",              /* inputLayout */
                ge::DT_FLOAT16,     /* optionalDataType */
                {3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                3,3,3,3,3,3,3,3,3,3,3,3,3,3,3},     /* actualSeqQLenList */
                {256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,
                256,256,256,256,256,256,256,256,256,256,256,256,256,256,256},      /* actualSeqKVLenList */
                30    /* numtokens */
        )),
        NsaSelectAttentionInferCase(
            "NsaSelectAttentionInfer_Case_3", true, "", /* CaseName, Enable, DebugInfo */
            OpInfo(ControlInfo(true, true),          /* RunKernel, RunKernel */
                ExpectInfo(false,                  /* ExpectSuccess */
                            0,                      /* ExpectKernelKey */
                            48)),                   /* ExpectKernelBlockDim */
            NsaSelectAttentionInferParam(
                1, 1, 1, 192, 2, /* batchSize, qSeqSize, headSize, headDim, maxBlockNumPerBatch*/
                128, 256, 128, 1, /* blockSize kvSeqSize, headDimV, headSizeV*/
                128, 2, 0, 1.0f, /* selectedBlockSize selectedBlockCount, sparseMode, scaleValue*/      
                "BSND",              /* inputLayout */
                ge::DT_FLOAT16,     /* optionalDataType */
                {1},     /* actualSeqQLenList */
                {256},      /* actualSeqKVLenList */
                1    /* numtokens */
        )),
        NsaSelectAttentionInferCase(
            "NsaSelectAttentionInfer_Case_4", true, "", /* CaseName, Enable, DebugInfo */
            OpInfo(ControlInfo(true, true),          /* RunKernel, RunKernel */
                ExpectInfo(false,                  /* ExpectSuccess */
                            1,                      /* ExpectKernelKey */
                            48)),                   /* ExpectKernelBlockDim */
            NsaSelectAttentionInferParam(
                1, 1, 1, 192, 2, /* batchSize, qSeqSize, headSize, headDim, maxBlockNumPerBatch*/
                128, 256, 128, 1, /* blockSize kvSeqSize, headDimV, headSizeV*/
                128, 2, 0, 1.0f, /* selectedBlockSize selectedBlockCount, sparseMode, scaleValue*/      
                "BSND",              /* inputLayout */
                ge::DT_FLOAT16,     /* optionalDataType */
                {1},     /* actualSeqQLenList */
                {256},      /* actualSeqKVLenList */
                1    /* numtokens */
        ))
    );

INSTANTIATE_TEST_SUITE_P(NsaSelectAttentionInfer, 
Ts_NsaSelectAttentionInfer_WithParam,Tc_NsaSelectAttentionInfer_Kernel_Case);    
}
