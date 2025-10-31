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
 * \file ts_fag_tc_us1s2_bbn2gs1s2.cpp
 * \brief FlashAttentionScoreGrad 算子 Us1s2Bbn2gs1s2 模板 UTest 用例.
 */

#include "ts_fag.h"
#include "../../../op_kernel/flash_attention_score_grad.cpp"

class FagCaseUs1s2Bbn2gs1s2 : public FagCase {
public:
    bool mChkTilingData = true;
    uint32_t mIsSparseValue = 1;

    FagCaseUs1s2Bbn2gs1s2() = default;
    FagCaseUs1s2Bbn2gs1s2(const char *name, bool enable, const char *dbgInfo, const OpInfoWithSocversion &reverse,
                          const FaParam &param, bool chkTilingData = true, uint32_t isSparseValue = 1)
        : FagCase(name, enable, dbgInfo, reverse, param, FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2),
          mChkTilingData(chkTilingData), mIsSparseValue(isSparseValue)
    {
    }

    FagCaseUs1s2Bbn2gs1s2(const char *name, bool enable, const char *dbgInfo, 
                          const std::function<void(FAG_INPUT_DTYPE)>& templatekeyKernelFunc,
                          const OpInfoWithSocversion &reverse,
                          const FaParam &param, bool chkTilingData = true, uint32_t isSparseValue = 1)
        : FagCase(name, enable, dbgInfo, templatekeyKernelFunc, reverse, param, 
                  FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2), mChkTilingData(chkTilingData), 
                  mIsSparseValue(isSparseValue)
    {
    }

    bool Run() override
    {
        if (!mEnable) {
            return true;
        }
        if (!mReverse.ProcessTiling(mName, this->socVersion)) {
            return false;
        }
        if (mChkTilingData) {
            auto *td = (const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *)(mReverseCtx.GetTilingData());
            if (td->s1s2BNGS1S2BaseParams.isSparse != mIsSparseValue) {
                LOG_IF(mReverse.mExp.mSuccess,
                       LOG_ERR("Case[%s:%s] TilingData check failed(isSparse), Exp=%u, Act=%u", mName.c_str(),
                               mReverse.mName.c_str(), mIsSparseValue, td->s1s2BNGS1S2BaseParams.isSparse));
                return false;
            }
        }
        if (!mReverse.ProcessKernel(mName)) {
            return false;
        }
        return true;
    }
};

class Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_TilingFailed : public Ts_Ascend910B2<FagCaseUs1s2Bbn2gs1s2> {};

TEST_F(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_TilingFailed, Tc_Us1s2Bbn2gs1s2_IllegalPseShape_001)
{
    FagCaseUs1s2Bbn2gs1s2 cs("Tc_Us1s2Bbn2gs1s2_IllegalPseShape_001", true,          /* CaseName, Enable */
                             "",                                                     /* DebugInfo */
                             OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                    ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                               ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                               ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                             FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                     ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                     0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                     1, 3,                              /* InnerPrecise, SparseMode */
                                     PseShapeType::_1_N1_ALIBI_S1_S2,   /* PseShapeType */
                                     DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                     PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                     AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                     ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                     PrefixShapeType::NONE)             /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mParam.pse = Tensor("pse", {1, 2, 3, 4, 5}, "1_2_3_4_5", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

TEST_F(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_TilingFailed, Tc_Us1s2Bbn2gs1s2_IllegalPseShape_002)
{
    FagCaseUs1s2Bbn2gs1s2 cs("Tc_Us1s2Bbn2gs1s2_IllegalPseShape_002", true,          /* CaseName, Enable */
                             "",                                                     /* DebugInfo */
                             OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                    ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                               ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                               ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                             FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                     ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                     0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                     1, 3,                              /* InnerPrecise, SparseMode */
                                     PseShapeType::_1_N1_ALIBI_S1_S2,   /* PseShapeType */
                                     DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                     PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                     AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                     ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                     PrefixShapeType::NONE)             /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mParam.pse = Tensor("pse", {16, 16, 16, 16}, "16_16_16_16", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

class Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2 : public Ts_WithParam_Ascend910B2<FagCaseUs1s2Bbn2gs1s2> {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2gs1s2_BatchCase = ::testing::Values(
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_004", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234890292UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_005", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234890292UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 2,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::_1_N1_S1_S2,         /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_1_S1_S2,     /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_006", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234890292UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 2, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 3,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_ALIBI_S1_S2,    /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_007", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 5,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::_1_N1_ALIBI_S1_S2,   /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_1_S1_S2,     /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::B,                /* PrefixShapeType */
                                  {1, 2},                            /* PrefixTensorData */
                                  {},                                /* ActualSeqQLenList */
                                  {})                                /* ActualSeqKVLenList */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_008", true,                    /* CaseName, Enable */
                          "compress atten_mask mode not support s1 s2 2048 2048", /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                            10000000000011001434UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 2, 1024, 1024, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 2,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_013", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                            ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, -100, 10,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_014", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 1,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE),            /* PrefixShapeType */
                          true, 0                                    /* CheckTilingData, isSparseValue */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_015", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 128, 128,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_016", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 0,           /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_017", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 2050,        /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE),            /* PrefixShapeType */
                          true, 0                                    /* CheckTilingData, isSparseValue */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_018", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 2050,        /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 2,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_019", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 2050,        /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 3,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_020", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          [](FAG_KERNEL_PARAM_){
                            ::flash_attention_score_grad<4, 3, 4, 0, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
                          OpInfoWithSocversion(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 128, 128, 128,                         /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 2050, 2050,        /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE),            /* PrefixShapeType */
                          true, 0                                    /* CheckTilingData, isSparseValue */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_021", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 100, 2050,         /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_022", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            234955828UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 2, 1, 2048, 2048, 128,                       /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 0.8f, 100, 2050,         /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::B_N1_S1_S2,          /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_023", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            167781428UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(4, 10, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSH,        /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 65504,       /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 6,                               /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                 /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8,  /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,        /* PaddingMaskShapeType */
                                  AttenMaskShapeType::PREFIXCOMPRESS, /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,              /* AttentionMaskDtype */
                                  PrefixShapeType::B,                 /* PrefixShapeType */
                                  {117, 319, 312, 334},               /* PrefixTensorData */
                                  {},                                 /* ActualSeqQLenList */
                                  {})                                 /* ActualSeqKVLenList */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_024", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                            167781428UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSH,        /* Dtype, Layout */
                                  0.08838f, 0.8f, 100, 100,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 4,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::SPARSE,        /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          ),
    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_025", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                            ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(1, 1, 1, 256, 128, 128,                         /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 8,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2, Tc_Fag_Us1s2Bbn2gs1s2_BatchCase);

class Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape : public Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_SoftmaxMax_Invalid)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.softmaxMax = Tensor("softmaxMax", {case_->mParam.b, case_->mParam.n1, case_->mParam.s1, 1},
                                      "B_N1_S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_SoftmaxSum_Invalid)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.softmaxSum = Tensor("softmaxSum", {case_->mParam.b, case_->mParam.n1, case_->mParam.s1, 1},
                                      "B_N1_S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_SoftmaxMax_DimErr)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.softmaxMax = Tensor("softmaxMax", {case_->mParam.b, case_->mParam.n1, case_->mParam.s1},
                                      "B_N1_S1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_SoftmaxSum_DimErr)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.softmaxSum = Tensor("softmaxSum", {case_->mParam.b, case_->mParam.n1, case_->mParam.s1},
                                      "B_N1_S1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_AttenmaskErr)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.attenMask =
        Tensor("attenMask", {case_->mParam.s1, 1}, "S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_AttenmaskErrDim)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.attenMask = Tensor("attenMask", {1, case_->mParam.s1, 1, case_->mParam.s2}, "1_S1_1_S2(Invalid)",
                                     case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape, Tc_AttenmaskErrDimNum)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.attenMask =
        Tensor("attenMask", {1, case_->mParam.s1, 1}, "1_S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2gs1s2_InvalidShape_BatchCase = ::testing::Values(

    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_000", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                            ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(2, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                              /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                                  AttenMaskShapeType::B_N1_S1_S2,    /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)             /* PrefixShapeType */
                          )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidShape,
                         Tc_Fag_Us1s2Bbn2gs1s2_InvalidShape_BatchCase);

class Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape : public Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape, Tc_AttenmaskErrShapeForPrefixCompress)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.attenMask =
        Tensor("attenMask", {2048, 2048}, "not_3072_2048(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape, Tc_PrefixErrShapeForPrefixCompress)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.prefix =
        Tensor("prefix", {110, 110, 110, 110, 110}, "prefixN_gt_B(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape_BatchCase = ::testing::Values(

    FagCaseUs1s2Bbn2gs1s2("Fag_Us1s2Bbn2gs1s2_Case_001", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                            ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(4, 10, 1, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BSH,        /* Dtype, Layout */
                                  0.08838f, 0.8f, 65504, 65504,       /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 6,                               /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                 /* PseShapeType */
                                  DropMaskShapeType::B_N1_S1_S2DIV8,  /* DropMaskShapeType */
                                  PaddingMaskShapeType::S1_S2,        /* PaddingMaskShapeType */
                                  AttenMaskShapeType::PREFIXCOMPRESS, /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,              /* AttentionMaskDtype */
                                  PrefixShapeType::B,                 /* PrefixShapeType */
                                  {117, 319, 312, 334},               /* PrefixTensorData */
                                  {},                                 /* ActualSeqQLenList */
                                  {})                                 /* ActualSeqKVLenList */
                          )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape,
                         Tc_Fag_Us1s2Bbn2gs1s2_InvalidPrefixCompressShape_BatchCase);


class FagCaseUs1s2Bbn2gs1s2s : public FagCase {
public:
    bool mChkTilingData = true;
    uint32_t mIsSparseValue = 0;

    FagCaseUs1s2Bbn2gs1s2s() = default;
    FagCaseUs1s2Bbn2gs1s2s(const char *name, bool enable, const char *dbgInfo, const OpInfoWithSocversion &reverse,
                          const FaParam &param, bool chkTilingData = true, uint32_t isSparseValue = 0)
        : FagCase(name, enable, dbgInfo, reverse, param, FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2),
          mChkTilingData(chkTilingData), mIsSparseValue(isSparseValue)
    {
    }

    bool Run() override
    {
        if (!mEnable) {
            return true;
        }
        if (!mReverse.ProcessTiling(mName, this->socVersion)) {
            return false;
        }
        if (!mReverse.ProcessKernel(mName)) {
            return false;
        }
        return true;
    }
};

class Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2s : public Ts_WithParam_Ascend910B2<FagCaseUs1s2Bbn2gs1s2s> {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2s, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2gs1s2_BatchCases = ::testing::Values(
    FagCaseUs1s2Bbn2gs1s2s("Fag_Us1s2Bbn2gs1s2_Case_026", true,                    /* CaseName, Enable */
                          "",                                                     /* DebugInfo */
                          OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                                 ExpectInfoWithSocversion(true,                                /* ExpectSuccess */
                                            74804UL,               /* ExpectTilingKey */
                                            ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                          FaParam(1, 1, 1, 1024, 1, 64,                           /* B, N2, G, S1, S2, D */
                                  ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                                  0.08838f, 1, 65536, 65536,          /* Scale, KeepProb, PreTokens, NxtTokens */
                                  1, 0,                               /* InnerPrecise, SparseMode */
                                  PseShapeType::NONE,                 /* PseShapeType */
                                  DropMaskShapeType::NONE,            /* DropMaskShapeType */
                                  PaddingMaskShapeType::NONE,         /* PaddingMaskShapeType */
                                  AttenMaskShapeType::NONE,           /* AttentionMaskShapeType */
                                  ge::DataType::DT_BOOL,              /* AttentionMaskDtype */
                                  PrefixShapeType::NONE)              /* PrefixShapeType */
                          )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2gs1s2s, Tc_Fag_Us1s2Bbn2gs1s2_BatchCases);
