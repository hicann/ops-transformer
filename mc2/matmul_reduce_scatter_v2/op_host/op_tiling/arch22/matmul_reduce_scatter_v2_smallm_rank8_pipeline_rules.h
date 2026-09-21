/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK8_PIPELINE_RULES_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK8_PIPELINE_RULES_H

#include "matmul_reduce_scatter_v2_smallm_rank8_layout_rules.h"

namespace Tiling_Small_M::Tiling_Rank8_A2 {

// pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode pvalueRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    Threshold(FeatureType::M_MUL_N, 6291456.0f), // 索引0
    // ====================== 层级1 (索引1-2) ======================
    Threshold(FeatureType::M_MUL_N, 1572864.0f), // 索引1
    Threshold(FeatureType::M_VALUE, 6144.0f),    // 索引2
    // ====================== 层级2 (索引3-6) ======================
    Threshold(FeatureType::M_MUL_N, 786432.0f),   // 索引3
    Threshold(FeatureType::K_VALUE, 6144.0f),     // 索引4
    Threshold(FeatureType::M_VALUE, 1536.0f),     // 索引5
    Threshold(FeatureType::M_MUL_N, 12582912.0f), // 索引6
    // ====================== 层级3 (索引7-14) ======================
    Ret(1),                                        // 索引7
    Threshold(FeatureType::N_VALUE, 6144.0f),      // 索引8
    Threshold(FeatureType::M_VALUE, 768.0f),       // 索引9
    Threshold(FeatureType::M_DIV_N, 0.187500f),    // 索引10
    Ret(2),                                        // 索引11
    Threshold(FeatureType::M_MUL_N, 12582912.0f),  // 索引12
    Threshold(FeatureType::WORLD_MUL_K, 24576.0f), // 索引13
    Threshold(FeatureType::WORLD_MUL_K, 49152.0f), // 索引14
    // ====================== 层级4 (索引15-30) ======================
    Placeholder(), Placeholder(),                   // 索引15-16
    Threshold(FeatureType::M_VALUE, 768.0f),        // 索引17
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),   // 索引18
    Threshold(FeatureType::MN_DIV_K, 768.000000f),  // 索引19
    Threshold(FeatureType::M_VALUE, 1536.0f),       // 索引20
    Threshold(FeatureType::M_DIV_N, 0.093750f),     // 索引21
    Threshold(FeatureType::M_DIV_N, 3.000000f),     // 索引22
    Placeholder(), Placeholder(),                   // 索引23-24
    Threshold(FeatureType::MN_DIV_K, 6144.000000f), // 索引25
    Threshold(FeatureType::M_DIV_N, 0.375000f),     // 索引26
    Ret(8),                                         // 索引27
    Ret(10),                                        // 索引28
    Threshold(FeatureType::M_DIV_N, 3.000000f),     // 索引29
    Ret(20),                                        // 索引30
    // ====================== 层级5 (索引31-62) ======================
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引31-34
    Threshold(FeatureType::MN_DIV_K, 192.000000f),              // 索引35
    Threshold(FeatureType::K_VALUE, 1536.0f),                   // 索引36
    Ret(2),                                                     // 索引37
    Ret(1),                                                     // 索引38
    Ret(2),                                                     // 索引39
    Threshold(FeatureType::K_VALUE, 3072.0f),                   // 索引40
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),             // 索引41
    Threshold(FeatureType::MN_DIV_K, 3072.000000f),             // 索引42
    Ret(1),                                                     // 索引43
    Ret(4),                                                     // 索引44
    Threshold(FeatureType::MN_DIV_K, 384.000000f),              // 索引45
    Threshold(FeatureType::M_DIV_N, 6.000000f),                 // 索引46
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引47-50
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),             // 索引51
    Threshold(FeatureType::N_VALUE, 3072.0f),                   // 索引52
    Ret(5),                                                     // 索引53
    Threshold(FeatureType::K_VALUE, 1536.0f),                   // 索引54
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引55-58
    Threshold(FeatureType::MN_DIV_K, 98304.000000f),            // 索引59
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f),              // 索引60
    Placeholder(), Placeholder(),                               // 索引61-62
    // ====================== 层级6 (索引63-126) ======================
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引63-68
    Placeholder(), Placeholder(),                                                             // 索引69-70
    Ret(1),                                                                                   // 索引71
    Ret(2), Ret(2),                                                                           // 索引72-73
    Ret(1),                                                                                   // 索引74
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引75-80
    Ret(2),                                                                                   // 索引81
    Ret(1), Ret(1),                                                                           // 索引82-83
    Ret(4),                                                                                   // 索引84
    Ret(2),                                                                                   // 索引85
    Ret(7),                                                                                   // 索引86
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引87-90
    Ret(3), Ret(3),                                                                           // 索引91-92
    Ret(5),                                                                                   // 索引93
    Ret(7),                                                                                   // 索引94
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引95-100
    Placeholder(), Placeholder(),                                                             // 索引101-102
    Ret(5), Ret(5), Ret(5),                                                                   // 索引103-105
    Ret(8),                                                                                   // 索引106
    Placeholder(), Placeholder(),                                                             // 索引107-108
    Ret(10),                                                                                  // 索引109
    Ret(5),                                                                                   // 索引110
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引111-116
    Placeholder(), Placeholder(),                                                             // 索引117-118
    Ret(20), Ret(20),                                                                         // 索引119-120
    Ret(8),                                                                                   // 索引121
    Ret(10),                                                                                  // 索引122
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引123-126
};

// ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    Threshold(FeatureType::M_VALUE, 768.0f), // 索引0
    // ====================== 层级1 (索引1-2) ======================
    Threshold(FeatureType::M_MUL_N, 3145728.0f), // 索引1
    Threshold(FeatureType::M_DIV_N, 3.000000f),  // 索引2
    // ====================== 层级2 (索引3-6) ======================
    Threshold(FeatureType::M_MUL_N, 786432.0f),      // 索引3
    Threshold(FeatureType::MN_DIV_K, 12288.000000f), // 索引4
    Threshold(FeatureType::M_MUL_N, 6291456.0f),     // 索引5
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引6
    // ====================== 层级3 (索引7-14) ======================
    Threshold(FeatureType::MN_DIV_K, 768.000000f),  // 索引7
    Threshold(FeatureType::M_MUL_N, 1572864.0f),    // 索引8
    Ret(16),                                        // 索引9
    Ret(4),                                         // 索引10
    Threshold(FeatureType::MN_DIV_K, 3072.000000f), // 索引11
    Threshold(FeatureType::WORLD_MUL_K, 49152.0f),  // 索引12
    Threshold(FeatureType::K_VALUE, 1536.0f),       // 索引13
    Threshold(FeatureType::M_MUL_N, 393216.0f),     // 索引14
    // ====================== 层级4 (索引15-30) ======================
    Threshold(FeatureType::M_MUL_N, 98304.0f),                  // 索引15
    Threshold(FeatureType::N_VALUE, 3072.0f),                   // 索引16
    Threshold(FeatureType::M_VALUE, 192.0f),                    // 索引17
    Threshold(FeatureType::MN_DIV_K, 3072.000000f),             // 索引18
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引19-22
    Threshold(FeatureType::M_VALUE, 1536.0f),                   // 索引23
    Threshold(FeatureType::M_DIV_N, 0.750000f),                 // 索引24
    Threshold(FeatureType::M_VALUE, 3072.0f),                   // 索引25
    Threshold(FeatureType::M_DIV_N, 1.500000f),                 // 索引26
    Ret(16),                                                    // 索引27
    Threshold(FeatureType::K_VALUE, 6144.0f),                   // 索引28
    Ret(8), Ret(8),                                             // 索引29-30
    // ====================== 层级5 (索引31-62) ======================
    Threshold(FeatureType::M_VALUE, 192.0f),                                                  // 索引31
    Threshold(FeatureType::N_VALUE, 1536.0f),                                                 // 索引32
    Threshold(FeatureType::N_VALUE, 768.0f),                                                  // 索引33
    Ret(16),                                                                                  // 索引34
    Threshold(FeatureType::MN_DIV_K, 384.000000f),                                            // 索引35
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f),                                            // 索引36
    Threshold(FeatureType::MN_DIV_K, 384.000000f),                                            // 索引37
    Threshold(FeatureType::MN_DIV_K, 6144.000000f),                                           // 索引38
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引39-44
    Placeholder(), Placeholder(),                                                             // 索引45-46
    Threshold(FeatureType::MN_DIV_K, 192.000000f),                                            // 索引47
    Threshold(FeatureType::MN_DIV_K, 768.000000f),                                            // 索引48
    Threshold(FeatureType::M_DIV_N, 0.375000f),                                               // 索引49
    Threshold(FeatureType::MN_DIV_K, 12288.000000f),                                          // 索引50
    Threshold(FeatureType::N_VALUE, 6144.0f),                                                 // 索引51
    Ret(8),                                                                                   // 索引52
    Threshold(FeatureType::M_VALUE, 3072.0f),                                                 // 索引53
    Ret(8),                                                                                   // 索引54
    Placeholder(), Placeholder(),                                                             // 索引55-56
    Ret(8),                                                                                   // 索引57
    Threshold(FeatureType::M_VALUE, 6144.0f),                                                 // 索引58
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引59-62
    // ====================== 层级6 (索引63-126) ======================
    Ret(4),                                                                                   // 索引63
    Ret(8),                                                                                   // 索引64
    Ret(16),                                                                                  // 索引65
    Ret(4),                                                                                   // 索引66
    Ret(8),                                                                                   // 索引67
    Ret(4),                                                                                   // 索引68
    Placeholder(), Placeholder(),                                                             // 索引69-70
    Ret(4), Ret(4), Ret(4),                                                                   // 索引71-73
    Ret(8),                                                                                   // 索引74
    Ret(4), Ret(4),                                                                           // 索引75-76
    Ret(16),                                                                                  // 索引77
    Ret(4),                                                                                   // 索引78
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引79-84
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引85-90
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引91-94
    Ret(4),                                                                                   // 索引95
    Ret(8),                                                                                   // 索引96
    Ret(4),                                                                                   // 索引97
    Ret(8),                                                                                   // 索引98
    Ret(4), Ret(4),                                                                           // 索引99-100
    Ret(8),                                                                                   // 索引101
    Ret(4), Ret(4),                                                                           // 索引102-103
    Ret(8),                                                                                   // 索引104
    Placeholder(), Placeholder(),                                                             // 索引105-106
    Ret(8),                                                                                   // 索引107
    Ret(4),                                                                                   // 索引108
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引109-114
    Placeholder(), Placeholder(),                                                             // 索引115-116
    Ret(8),                                                                                   // 索引117
    Ret(4),                                                                                   // 索引118
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引119-124
    Placeholder(), Placeholder(),                                                             // 索引125-126
};

inline int GetOptimalM0(int m, int k, int n)
{
    return TraverseDecisionTree(m0Rule, m, k, n, RANKSIZE_EIGHT, DEFAULT_M0);
}

inline int GetOptimalSwizzlCount(int m, int k, int n)
{
    return TraverseDecisionTree(swizzlcountRule, m, k, n, RANKSIZE_EIGHT, DEFAULT_SWIZZLCOUNT);
}

inline int GetOptimalSwizzlDirect(int m, int k, int n)
{
    return TraverseDecisionTree(swizzldirectRule, m, k, n, RANKSIZE_EIGHT, DEFAULT_SWIZZLDIRECT);
}

inline int GetOptimalPValue(int m, int k, int n)
{
    return TraverseDecisionTree(pvalueRule, m, k, n, RANKSIZE_EIGHT, DEFAULT_PVALUE);
}

inline int GetOptimalUbmovenum(int m, int k, int n)
{
    return TraverseDecisionTree(ubmovenumRule, m, k, n, RANKSIZE_EIGHT, DEFAULT_UBMOVENUM);
}

} // namespace Tiling_Small_M::Tiling_Rank8_A2

#endif
