/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK2_PIPELINE_RULES_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK2_PIPELINE_RULES_H

#include "matmul_reduce_scatter_v2_smallm_rank2_layout_rules.h"

namespace Tiling_Small_M::Tiling_Rank2_A2 {

// pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, mn_div_k）
const DecisionNode pvalueRule[] = {
    // | 层级0 (索引0-0) |
    Threshold(FeatureType::M_DIV_N, 12.000000f), // 索引0
    // || 层级1 (索引1-2) ||
    Threshold(FeatureType::M_VALUE, 6144.0f), // 索引1
    Threshold(FeatureType::K_VALUE, 6144.0f), // 索引2
    // ||| 层级2 (索引3-6) |||
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),  // 索引3
    Threshold(FeatureType::MN_DIV_K, 98304.000000f), // 索引4
    Threshold(FeatureType::M_MUL_N, 1572864.0f),     // 索引5
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引6
    // |||| 层级3 (索引7-14) ||||
    Threshold(FeatureType::M_VALUE, 3072.0f),    // 索引7
    Threshold(FeatureType::M_MUL_N, 6291456.0f), // 索引8
    Threshold(FeatureType::K_VALUE, 6144.0f),    // 索引9
    Threshold(FeatureType::N_VALUE, 6144.0f),    // 索引10
    Ret(2),                                      // 索引11
    Threshold(FeatureType::N_VALUE, 384.0f),     // 索引12
    Ret(4),                                      // 索引13
    Ret(7),                                      // 索引14
    // ||||| 层级4 (索引15-30) |||||
    Threshold(FeatureType::M_MUL_N, 6291456.0f),                // 索引15
    Threshold(FeatureType::K_VALUE, 3072.0f),                   // 索引16
    Threshold(FeatureType::MN_DIV_K, 6144.000000f),             // 索引17
    Threshold(FeatureType::M_VALUE, 3072.0f),                   // 索引18
    Threshold(FeatureType::M_DIV_N, 1.500000f),                 // 索引19
    Ret(10),                                                    // 索引20
    Ret(20), Ret(20),                                           // 索引21-22
    Placeholder(), Placeholder(),                               // 索引23-24
    Ret(4), Ret(4),                                             // 索引25-26
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引27-30
    // ====================== 层级5 (索引31-62) ======================
    Ret(1),                                                                                   // 索引31
    Ret(2), Ret(2), Ret(2),                                                                   // 索引32-34
    Threshold(FeatureType::M_MUL_N, 786432.0f),                                               // 索引35
    Threshold(FeatureType::M_MUL_N, 3145728.0f),                                              // 索引36
    Threshold(FeatureType::MN_DIV_K, 24576.000000f),                                          // 索引37
    Threshold(FeatureType::MN_DIV_K, 49152.000000f),                                          // 索引38
    Ret(10),                                                                                  // 索引39
    Threshold(FeatureType::MN_DIV_K, 24576.000000f),                                          // 索引40
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引41-46
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引47-52
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引53-58
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引59-62
    // <层级6> (索引63-126)
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引63-68
    Placeholder(), Placeholder(),                                                             // 索引69-70
    Ret(1),                                                                                   // 索引71
    Ret(2),                                                                                   // 索引72
    Ret(4), Ret(4),                                                                           // 索引73-74
    Ret(3),                                                                                   // 索引75
    Ret(8),                                                                                   // 索引76
    Ret(5),                                                                                   // 索引77
    Ret(10),                                                                                  // 索引78
    Placeholder(), Placeholder(),                                                             // 索引79-80
    Ret(4),                                                                                   // 索引81
    Ret(8),                                                                                   // 索引82
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引83-88
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引89-94
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引95-100
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引101-106
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引107-112
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引113-118
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引119-124
    Placeholder(), Placeholder(),                                                             // 索引125-126
};

// ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // (层级0) [索引0-0]
    Threshold(FeatureType::M_MUL_N, 98304.0f), // 索引0
    // (层级1) [索引1-2]
    Threshold(FeatureType::MN_DIV_K, 48.000000f), // 索引1
    Threshold(FeatureType::K_VALUE, 3072.0f),     // 索引2
    // (层级2) [索引3-6]
    Ret(16),                                      // 索引3
    Threshold(FeatureType::MN_DIV_K, 96.000000f), // 索引4
    Threshold(FeatureType::M_MUL_N, 12582912.0f), // 索引5
    Threshold(FeatureType::MN_DIV_K, 96.000000f), // 索引6
    // (层级3) [索引7-14]
    Placeholder(), Placeholder(),                  // 索引7-8
    Threshold(FeatureType::M_VALUE, 192.0f),       // 索引9
    Ret(16),                                       // 索引10
    Threshold(FeatureType::MN_DIV_K, 192.000000f), // 索引11
    Threshold(FeatureType::WORLD_MUL_K, 768.0f),   // 索引12
    Threshold(FeatureType::M_VALUE, 192.0f),       // 索引13
    Threshold(FeatureType::M_VALUE, 6144.0f),      // 索引14
    // 【层级4】(索引15-30)
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引15-18
    Ret(8),                                                     // 索引19
    Ret(16),                                                    // 索引20
    Placeholder(), Placeholder(),                               // 索引21-22
    Threshold(FeatureType::M_DIV_N, 0.187500f),                 // 索引23
    Threshold(FeatureType::M_MUL_N, 196608.0f),                 // 索引24
    Threshold(FeatureType::M_DIV_N, 1.500000f),                 // 索引25
    Threshold(FeatureType::N_VALUE, 6144.0f),                   // 索引26
    Threshold(FeatureType::M_MUL_N, 393216.0f),                 // 索引27
    Threshold(FeatureType::M_MUL_N, 196608.0f),                 // 索引28
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f),              // 索引29
    Threshold(FeatureType::MN_DIV_K, 768.000000f),              // 索引30
    // 【层级5】(索引31-62)
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引31-36
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引37-42
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引43-46
    Threshold(FeatureType::MN_DIV_K, 96.000000f),                                             // 索引47
    Ret(8),                                                                                   // 索引48
    Threshold(FeatureType::M_VALUE, 384.0f),                                                  // 索引49
    Threshold(FeatureType::M_DIV_N, 0.023438f),                                               // 索引50
    Threshold(FeatureType::M_MUL_N, 25165824.0f),                                             // 索引51
    Ret(4),                                                                                   // 索引52
    Threshold(FeatureType::M_MUL_N, 25165824.0f),                                             // 索引53
    Threshold(FeatureType::WORLD_MUL_K, 1536.0f),                                             // 索引54
    Threshold(FeatureType::MN_DIV_K, 24.000000f),                                             // 索引55
    Ret(16),                                                                                  // 索引56
    Ret(8),                                                                                   // 索引57
    Threshold(FeatureType::MN_DIV_K, 48.000000f),                                             // 索引58
    Threshold(FeatureType::M_DIV_N, 0.093750f),                                               // 索引59
    Threshold(FeatureType::M_VALUE, 1536.0f),                                                 // 索引60
    Threshold(FeatureType::K_VALUE, 6144.0f),                                                 // 索引61
    Threshold(FeatureType::N_VALUE, 6144.0f),                                                 // 索引62
    // 【层级6】(索引63-126)
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引63-68
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引69-74
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引75-80
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引81-86
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引87-92
    Placeholder(), Placeholder(),                                                             // 索引93-94
    Ret(4),                                                                                   // 索引95
    Ret(8),                                                                                   // 索引96
    Placeholder(), Placeholder(),                                                             // 索引97-98
    Ret(8),                                                                                   // 索引99
    Ret(4), Ret(4),                                                                           // 索引100-101
    Ret(16),                                                                                  // 索引102
    Ret(8),                                                                                   // 索引103
    Ret(4),                                                                                   // 索引104
    Placeholder(), Placeholder(),                                                             // 索引105-106
    Ret(8), Ret(8), Ret(8), Ret(8), Ret(8),                                                   // 索引107-111
    Ret(4),                                                                                   // 索引112
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引113-116
    Ret(16),                                                                                  // 索引117
    Ret(8),                                                                                   // 索引118
    Ret(4),                                                                                   // 索引119
    Ret(8),                                                                                   // 索引120
    Ret(4), Ret(4),                                                                           // 索引121-122
    Ret(16),                                                                                  // 索引123
    Ret(4),                                                                                   // 索引124
    Ret(8),                                                                                   // 索引125
    Ret(4),                                                                                   // 索引126
};

inline int GetOptimalM0(int m, int k, int n)
{
    return TraverseDecisionTree(m0Rule, m, k, n, RANKSIZE_TWO, DEFAULT_M0);
}

inline int GetOptimalSwizzlCount(int m, int k, int n)
{
    return TraverseDecisionTree(swizzlcountRule, m, k, n, RANKSIZE_TWO, DEFAULT_SWIZZLCOUNT);
}

inline int GetOptimalSwizzlDirect(int m, int k, int n)
{
    return TraverseDecisionTree(swizzldirectRule, m, k, n, RANKSIZE_TWO, DEFAULT_SWIZZLDIRECT);
}

inline int GetOptimalPValue(int m, int k, int n)
{
    return TraverseDecisionTree(pvalueRule, m, k, n, RANKSIZE_TWO, DEFAULT_PVALUE);
}

inline int GetOptimalUbmovenum(int m, int k, int n)
{
    return TraverseDecisionTree(ubmovenumRule, m, k, n, RANKSIZE_TWO, DEFAULT_UBMOVENUM);
}

} // namespace Tiling_Small_M::Tiling_Rank2_A2

#endif
