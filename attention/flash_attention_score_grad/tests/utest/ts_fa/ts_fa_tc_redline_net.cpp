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
 * \file ts_fa_tc_redline_net.cpp
 * \brief FlashAttention 自动反向红线用例.
 */

#include "ts_fa.h"

class Ts_Fa_Ascend910B2_Redline_NetCase : public Ts_Fa_WithParam_Ascend910B2 {};
class Ts_Fa_Ascend910B3_Redline_NetCase : public Ts_Fa_WithParam_Ascend910B3 {};

TEST_P(Ts_Fa_Ascend910B2_Redline_NetCase, Tc_Redline)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_TRUE(case_->Run());
}

TEST_P(Ts_Fa_Ascend910B3_Redline_NetCase, Tc_Redline)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_TRUE(case_->Run());
}

const auto Tc_Fa_Redline_Case_NetCase = ::testing::Values(

    FaCase("NetCase_000", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 10, 1, 3840, 512, 64,                        /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_001", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(24, 5, 1, 9216, 9216, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_002", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(24, 5, 1, 9216, 77, 64,                         /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_003", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(24, 10, 1, 2304, 77, 64,                        /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_004", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 6144, 6144, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_005", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 8, 1, 6144, 6144, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_006", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 5, 1, 8192, 8192, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_007", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 10, 1, 2048, 2048, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_008", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 10, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_009", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 32, 1, 8192, 8192, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_010", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 4, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_011", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 1, 8, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_012", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(8, 32, 1, 2048, 2048, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_013", false,                                   /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 1, 8, 8192, 8192, 120,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.09128f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_014", false,                                   /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 1, 8, 8192, 8192, 120,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.09128f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_015", false,                                   /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 4, 2, 8192, 8192, 120,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.09128f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_016", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 8, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_017", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 14, 1, 2048, 2048, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_018", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 14, 1, 2048, 2048, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_019", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 5, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_020", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 4, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_021", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 32, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_022", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 8192, 8192, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_023", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 8192, 8192, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 65536,                   /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_024", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 1, 12, 8192, 8192, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 65536,                   /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_025", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_026", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_027", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 4, 1, 4096, 4096, 256,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_028", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 10, 1, 4096, 4096, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_029", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 10, 1, 4096, 4096, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_030", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 10, 1, 4096, 4096, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_031", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 10, 1, 4096, 77, 64,                         /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_032", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 10, 1, 4096, 512, 64,                        /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_033", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 10, 1, 4096, 512, 64,                        /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_034", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 20, 1, 1024, 1024, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_035", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 20, 1, 1024, 1024, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_036", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 20, 1, 1024, 1024, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_037", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 20, 1, 1024, 77, 64,                         /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_038", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 20, 1, 1024, 512, 64,                        /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_039", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 20, 1, 1024, 512, 64,                        /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_040", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 10, 1, 3840, 3840, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_041", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 10, 1, 3840, 3840, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_042", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 10, 1, 3840, 3840, 64,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_043", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 10, 1, 3840, 77, 64,                         /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("NetCase_044", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 10, 1, 3840, 512, 64,                        /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.125f, 0.9f, 65536, 65536,                     /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           )

);
INSTANTIATE_TEST_SUITE_P(Fa, Ts_Fa_Ascend910B2_Redline_NetCase, Tc_Fa_Redline_Case_NetCase);
INSTANTIATE_TEST_SUITE_P(Fa, Ts_Fa_Ascend910B3_Redline_NetCase, Tc_Fa_Redline_Case_NetCase);
