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
 * \file ts_nsa_compress_with_cache_tiling.cpp
 * \brief NsaCompressWithCache tiling用例.
 */

#include "ts_nsa_compress_with_cache.h"

namespace {

TEST_P(Ts_NsaCompressWithCache_WithParam_Ascend910B1, Tc_Tiling_NsaCompressWithCache)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}


const auto Tc_NsaCompressWithCache_Tiling_Case = ::testing::Values(
    // 检查batchSize>2048
    NsaCompressWithCacheCase(
        "NsaCompressWithCache_Case_25", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),          /* RunTiling, RunKernel */
               ExpectInfo(false,                  /* ExpectSuccess */
                          2,                      /* ExpectTilingKey */
                          48)),                   /* ExpectTilingBlockDim */
        NsaCompressWithCacheParam(
            2049, 32, 64,          /* batchSize, headNum, headDim */
            ge::DataType::DT_BF16, /* Dtype */
            {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144}, /* ActualSeqLenList */
            LayoutType::TND,                                                                  /* Layout */
            16,                                                                               /*CompressBlockSize*/
            16,                                                                               /*CompressStride*/
            1,                                                                                /*ActSeqLenType*/
            128                                                                               /*PageBlockSize*/
            )),
    // 拦截过大的输入
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_26", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 256, 512,           /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       64,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 拦截空输入
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_27", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 0, 128,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       64,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 拦截CompressBlockSize除不尽16
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_28", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 2, 128,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       65,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),

    // 拦截CompressStride除不尽16
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_29", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 3, 128,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       64,                    /*CompressBlockSize*/
                                                       15,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),

    // 拦截CompressStride>CompressBlockSize
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_30", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 4, 128,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       16,                    /*CompressBlockSize*/
                                                       32,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),

    // 拦截ActSeqLenType!=1
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_31", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 5, 128,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       16,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       0,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),

    // 拦截HeadDim不对齐32B
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_32", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 6, 127,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       16,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       0,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 拦截HeadDim<HeadNum
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_33", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 32, 16,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       16,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       0,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 拦截未知layout
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_34", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 32, 16,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::UNKOWN,    /* Layout */
                                                       16,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       0,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 拦截不支持的type
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_35", true, "",         /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                  /* RunTiling, RunKernel */
                                    ExpectInfo(false,                          /* ExpectSuccess */
                                               0,                              /* ExpectTilingKey */
                                               48)),                           /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 32, 16,              /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_FLOAT, /* Dtype */
                                                       {144, 192, 144, 144},   /* ActualSeqLenList */
                                                       LayoutType::TND,        /* Layout */
                                                       16,                     /*CompressBlockSize*/
                                                       16,                     /*CompressStride*/
                                                       0,                      /*ActSeqLenType*/
                                                       128                     /*PageBlockSize*/
                                                       )),
    // 拦截compressBlockSize>64
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_36", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 32, 16,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       128,                   /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       0,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 拦截headNum>64
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_36", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 66, 16,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 1, 1},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       32,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       0,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 拦截headNum>50且为奇数
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_36", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
                                    ExpectInfo(false,                         /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 51, 16,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       16,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       0,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       ))

);

INSTANTIATE_TEST_SUITE_P(NsaCompressWithCache, Ts_NsaCompressWithCache_WithParam_Ascend910B1,
                         Tc_NsaCompressWithCache_Tiling_Case);

} // namespace