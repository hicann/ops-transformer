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
 * \file ts_nsa_compress_with_cache_kernel.cpp
 * \brief NsaCompressWithCache kernel用例.
 */

#include "ts_nsa_compress_with_cache.h"

namespace {

TEST_P(Ts_NsaCompressWithCache_WithParam, Tc_Kernel_NsaCompressWithCache)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_NsaCompressWithCache_Kernel_Case = ::testing::Values(
    // 小batch，tilingkey=5
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_0", true, "",            /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                                    ExpectInfo(true,                             /* ExpectSuccess */
                                               2,                                /* ExpectTilingKey */
                                               2)),                              /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 8, 192,                /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_FLOAT16, /* Dtype */
                                                       {144, 192, 10, 45},       /* ActualSeqLenList */
                                                       LayoutType::TND,          /* Layout */
                                                       64,                       /*CompressBlockSize*/
                                                       16,                       /*CompressStride*/
                                                       1,                        /*ActSeqLenType*/
                                                       128                       /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=4
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_1", true, "",         /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                    ExpectInfo(true,                          /* ExpectSuccess */
                                               2,                             /* ExpectTilingKey */
                                               2)),                           /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 8, 192,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 10, 45},    /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       64,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=3
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_2", true, "",            /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                                    ExpectInfo(true,                             /* ExpectSuccess */
                                               1,                                /* ExpectTilingKey */
                                               8)),                              /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 32, 192,               /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_FLOAT16, /* Dtype */
                                                       {144, 192, 10, 45},       /* ActualSeqLenList */
                                                       LayoutType::TND,          /* Layout */
                                                       16,                       /*CompressBlockSize*/
                                                       16,                       /*CompressStride*/
                                                       1,                        /*ActSeqLenType*/
                                                       128                       /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=2
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_3", true, "",         /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                    ExpectInfo(true,                          /* ExpectSuccess */
                                               1,                             /* ExpectTilingKey */
                                               8)),                           /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 32, 192,            /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 10, 45},    /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       16,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=1
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_4", true, "",            /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                                    ExpectInfo(true,                             /* ExpectSuccess */
                                               0,                                /* ExpectTilingKey */
                                               37)),                             /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 37, 256,               /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_FLOAT16, /* Dtype */
                                                       {144, 1, 10, 45},         /* ActualSeqLenList */
                                                       LayoutType::TND,          /* Layout */
                                                       64,                       /*CompressBlockSize*/
                                                       16,                       /*CompressStride*/
                                                       1,                        /*ActSeqLenType*/
                                                       128                       /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=0
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_5", true, "",         /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                    ExpectInfo(true,                          /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               37)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 37, 256,            /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 1, 10, 45},      /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       64,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=15
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_6", true, "",            /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                                    ExpectInfo(true,                             /* ExpectSuccess */
                                               2,                               /* ExpectTilingKey */
                                               2)),                              /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 1, 192,                /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_FLOAT16, /* Dtype */
                                                       {144, 192, 10, 45},       /* ActualSeqLenList */
                                                       LayoutType::TND,          /* Layout */
                                                       48,                       /*CompressBlockSize*/
                                                       16,                       /*CompressStride*/
                                                       1,                        /*ActSeqLenType*/
                                                       128                       /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=14
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_7", true, "",         /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                    ExpectInfo(true,                          /* ExpectSuccess */
                                               2,                            /* ExpectTilingKey */
                                               2)),                           /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 1, 192,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 10, 45},    /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       48,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=11
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_8", true, "",            /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                                    ExpectInfo(true,                             /* ExpectSuccess */
                                               0,                               /* ExpectTilingKey */
                                               26)),                             /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 13, 256,               /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_FLOAT16, /* Dtype */
                                                       {144, 192, 10, 45},       /* ActualSeqLenList */
                                                       LayoutType::TND,          /* Layout */
                                                       48,                       /*CompressBlockSize*/
                                                       16,                       /*CompressStride*/
                                                       1,                        /*ActSeqLenType*/
                                                       128                       /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=10
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_9", true, "",         /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                    ExpectInfo(true,                          /* ExpectSuccess */
                                               0,                            /* ExpectTilingKey */
                                               26)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 13, 256,            /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 10, 45},    /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       48,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=13
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_10", true, "",           /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                                    ExpectInfo(true,                             /* ExpectSuccess */
                                               1,                               /* ExpectTilingKey */
                                               16)),                             /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 64, 96,                /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_FLOAT16, /* Dtype */
                                                       {144, 192, 10, 45},       /* ActualSeqLenList */
                                                       LayoutType::TND,          /* Layout */
                                                       32,                       /*CompressBlockSize*/
                                                       16,                       /*CompressStride*/
                                                       1,                        /*ActSeqLenType*/
                                                       128                       /*PageBlockSize*/
                                                       )),
    // 小batch，tilingkey=12
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_11", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                    ExpectInfo(true,                          /* ExpectSuccess */
                                               1,                            /* ExpectTilingKey */
                                               16)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 64, 96,             /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 10, 45},    /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       32,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),

    NsaCompressWithCacheCase(                     // 大batch，tilingkey=5
        "NsaCompressWithCache_Case_12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),           /* RunTiling, RunKernel */
               ExpectInfo(true,                   /* ExpectSuccess */
                          2,                      /* ExpectTilingKey */
                          48)),                   /* ExpectTilingBlockDim */
        NsaCompressWithCacheParam(
            64, 8, 192,               /* batchSize, headNum, headDim */
            ge::DataType::DT_FLOAT16, /* Dtype */
            {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144}, /* ActualSeqLenList */
            LayoutType::TND,                                                                  /* Layout */
            64,                                                                               /*CompressBlockSize*/
            16,                                                                               /*CompressStride*/
            1,                                                                                /*ActSeqLenType*/
            128                                                                               /*PageBlockSize*/
            )),

    NsaCompressWithCacheCase(                     // 大batch，tilingkey=4
        "NsaCompressWithCache_Case_13", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),           /* RunTiling, RunKernel */
               ExpectInfo(true,                   /* ExpectSuccess */
                          2,                      /* ExpectTilingKey */
                          48)),                   /* ExpectTilingBlockDim */
        NsaCompressWithCacheParam(
            64, 8, 192,            /* batchSize, headNum, headDim */
            ge::DataType::DT_BF16, /* Dtype */
            {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144}, /* ActualSeqLenList */
            LayoutType::TND,                                                                  /* Layout */
            64,                                                                               /*CompressBlockSize*/
            16,                                                                               /*CompressStride*/
            1,                                                                                /*ActSeqLenType*/
            128                                                                               /*PageBlockSize*/
            )),

    NsaCompressWithCacheCase(                     // 大batch，tilingkey=3
        "NsaCompressWithCache_Case_14", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),           /* RunTiling, RunKernel */
               ExpectInfo(true,                   /* ExpectSuccess */
                          1,                      /* ExpectTilingKey */
                          48)),                   /* ExpectTilingBlockDim */
        NsaCompressWithCacheParam(
            32, 32, 192,              /* batchSize, headNum, headDim */
            ge::DataType::DT_FLOAT16, /* Dtype */
            {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144}, /* ActualSeqLenList */
            LayoutType::TND,                                                                  /* Layout */
            16,                                                                               /*CompressBlockSize*/
            16,                                                                               /*CompressStride*/
            1,                                                                                /*ActSeqLenType*/
            128                                                                               /*PageBlockSize*/
            )),

    NsaCompressWithCacheCase(                     // 大batch，tilingkey=15
        "NsaCompressWithCache_Case_15", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),           /* RunTiling, RunKernel */
               ExpectInfo(true,                   /* ExpectSuccess */
                          1,                      /* ExpectTilingKey */
                          48)),                   /* ExpectTilingBlockDim */
        NsaCompressWithCacheParam(
            32, 32, 192,           /* batchSize, headNum, headDim */
            ge::DataType::DT_BF16, /* Dtype */
            {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144}, /* ActualSeqLenList */
            LayoutType::TND,                                                                  /* Layout */
            16,                                                                               /*CompressBlockSize*/
            16,                                                                               /*CompressStride*/
            1,                                                                                /*ActSeqLenType*/
            128                                                                               /*PageBlockSize*/
            )),
    // 大batch，tilingkey=1
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_16", true, "",           /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                                    ExpectInfo(true,                             /* ExpectSuccess */
                                               0,                                /* ExpectTilingKey */
                                               48)),                             /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(2, 37, 256,               /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_FLOAT16, /* Dtype */
                                                       {144, 192},               /* ActualSeqLenList */
                                                       LayoutType::TND,          /* Layout */
                                                       64,                       /*CompressBlockSize*/
                                                       16,                       /*CompressStride*/
                                                       1,                        /*ActSeqLenType*/
                                                       128                       /*PageBlockSize*/
                                                       )),
    // 大batch，tilingkey=0
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_17", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                    ExpectInfo(true,                          /* ExpectSuccess */
                                               0,                             /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(2, 37, 256,            /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192},            /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       64,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 大batch，tilingkey=15
    NsaCompressWithCacheCase(
        "NsaCompressWithCache_Case_18", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),           /* RunTiling, RunKernel */
               ExpectInfo(true,                   /* ExpectSuccess */
                          2,                     /* ExpectTilingKey */
                          48)),                   /* ExpectTilingBlockDim */
        NsaCompressWithCacheParam(
            64, 1, 192,               /* batchSize, headNum, headDim */
            ge::DataType::DT_FLOAT16, /* Dtype */
            {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144}, /* ActualSeqLenList */
            LayoutType::TND,                                                                  /* Layout */
            48,                                                                               /*CompressBlockSize*/
            16,                                                                               /*CompressStride*/
            1,                                                                                /*ActSeqLenType*/
            128                                                                               /*PageBlockSize*/
            )),
    // 大batch，tilingkey=14
    NsaCompressWithCacheCase(
        "NsaCompressWithCache_Case_19", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),           /* RunTiling, RunKernel */
               ExpectInfo(true,                   /* ExpectSuccess */
                          2,                     /* ExpectTilingKey */
                          48)),                   /* ExpectTilingBlockDim */
        NsaCompressWithCacheParam(
            64, 1, 192,            /* batchSize, headNum, headDim */
            ge::DataType::DT_BF16, /* Dtype */
            {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144}, /* ActualSeqLenList */
            LayoutType::TND,                                                                  /* Layout */
            48,                                                                               /*CompressBlockSize*/
            16,                                                                               /*CompressStride*/
            1,                                                                                /*ActSeqLenType*/
            128                                                                               /*PageBlockSize*/
            )),
    // 大batch，tilingkey=11
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_20", true, "",           /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                     /* RunTiling, RunKernel */
                                    ExpectInfo(true,                             /* ExpectSuccess */
                                               0,                               /* ExpectTilingKey */
                                               48)),                             /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 13, 256,               /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_FLOAT16, /* Dtype */
                                                       {144, 192, 144, 144},     /* ActualSeqLenList */
                                                       LayoutType::TND,          /* Layout */
                                                       48,                       /*CompressBlockSize*/
                                                       16,                       /*CompressStride*/
                                                       1,                        /*ActSeqLenType*/
                                                       128                       /*PageBlockSize*/
                                                       )),
    // 大batch，tilingkey=10
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_21", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                    ExpectInfo(true,                          /* ExpectSuccess */
                                               0,                            /* ExpectTilingKey */
                                               48)),                          /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 13, 256,            /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {144, 192, 144, 144},  /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       48,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )),
    // 大batch，tilingkey=13
    NsaCompressWithCacheCase(
        "NsaCompressWithCache_Case_22", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),           /* RunTiling, RunKernel */
               ExpectInfo(true,                   /* ExpectSuccess */
                          1,                     /* ExpectTilingKey */
                          48)),                   /* ExpectTilingBlockDim */
        NsaCompressWithCacheParam(
            32, 64, 96,               /* batchSize, headNum, headDim */
            ge::DataType::DT_FLOAT16, /* Dtype */
            {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144}, /* ActualSeqLenList */
            LayoutType::TND,                                                                  /* Layout */
            32,                                                                               /*CompressBlockSize*/
            16,                                                                               /*CompressStride*/
            1,                                                                                /*ActSeqLenType*/
            128                                                                               /*PageBlockSize*/
            )),
    // 大batch，tilingkey=12
    NsaCompressWithCacheCase(
        "NsaCompressWithCache_Case_23", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),           /* RunTiling, RunKernel */
               ExpectInfo(true,                   /* ExpectSuccess */
                          1,                     /* ExpectTilingKey */
                          48)),                   /* ExpectTilingBlockDim */
        NsaCompressWithCacheParam(
            32, 64, 96,            /* batchSize, headNum, headDim */
            ge::DataType::DT_BF16, /* Dtype */
            {144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144,
             144, 192, 144, 144, 144, 144, 144, 144, 144, 192, 144, 144, 144, 144, 144, 144}, /* ActualSeqLenList */
            LayoutType::TND,                                                                  /* Layout */
            32,                                                                               /*CompressBlockSize*/
            16,                                                                               /*CompressStride*/
            1,                                                                                /*ActSeqLenType*/
            128                                                                               /*PageBlockSize*/
            )),
    // 不应拦截的空输入
    NsaCompressWithCacheCase("NsaCompressWithCache_Case_24", true, "",        /* CaseName, Enable, DebugInfo */
                             OpInfo(ControlInfo(true, true),                  /* RunTiling, RunKernel */
                                    ExpectInfo(true,                          /* ExpectSuccess */
                                               1,                             /* ExpectTilingKey */
                                               1)),                           /* ExpectTilingBlockDim */
                             NsaCompressWithCacheParam(4, 32, 192,            /* batchSize, headNum, headDim */
                                                       ge::DataType::DT_BF16, /* Dtype */
                                                       {1, 2, 3, 4},          /* ActualSeqLenList */
                                                       LayoutType::TND,       /* Layout */
                                                       16,                    /*CompressBlockSize*/
                                                       16,                    /*CompressStride*/
                                                       1,                     /*ActSeqLenType*/
                                                       128                    /*PageBlockSize*/
                                                       )));


INSTANTIATE_TEST_SUITE_P(NsaCompressWithCache, Ts_NsaCompressWithCache_WithParam, Tc_NsaCompressWithCache_Kernel_Case);
} // namespace
