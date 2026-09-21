/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK2_LAYOUT_RULES_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK2_LAYOUT_RULES_H

#include "matmul_reduce_scatter_v2_smallm_decision_tree.h"

namespace Tiling_Small_M::Tiling_Rank2_A2 {

const DecisionNode m0Rule[] = {
    // --- 层级0 (索引0-0) ---
    Ret(128), // 索引0
};

// swizzlCount优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzlcountRule[] = {
    // === 层级0 (索引0-0) ===
    Threshold(FeatureType::M_VALUE, 1536.0f), // 索引0
    // ----- 层级1 (索引1-2) -----
    Threshold(FeatureType::M_VALUE, 768.0f),  // 索引1
    Threshold(FeatureType::M_VALUE, 3072.0f), // 索引2
    // ~~~~~ 层级2 (索引3-6) ~~~~~
    Threshold(FeatureType::M_VALUE, 384.0f),         // 索引3
    Threshold(FeatureType::M_DIV_N, 1.500000f),      // 索引4
    Threshold(FeatureType::MN_DIV_K, 24576.000000f), // 索引5
    Threshold(FeatureType::M_VALUE, 6144.0f),        // 索引6
    // >>>>> 层级3 (索引7-14) >>>>>
    Threshold(FeatureType::M_VALUE, 192.0f),         // 索引7
    Threshold(FeatureType::MN_DIV_K, 48.000000f),    // 索引8
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),    // 索引9
    Threshold(FeatureType::MN_DIV_K, 768.000000f),   // 索引10
    Threshold(FeatureType::M_MUL_N, 12582912.0f),    // 索引11
    Threshold(FeatureType::MN_DIV_K, 49152.000000f), // 索引12
    Threshold(FeatureType::N_VALUE, 384.0f),         // 索引13
    Threshold(FeatureType::K_VALUE, 6144.0f),        // 索引14
    // :::::: 层级4 (索引15-30) ::::::
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引15
    Threshold(FeatureType::M_DIV_N, 0.093750f),      // 索引16
    Threshold(FeatureType::MN_DIV_K, 24.000000f),    // 索引17
    Threshold(FeatureType::M_DIV_N, 0.375000f),      // 索引18
    Threshold(FeatureType::K_VALUE, 384.0f),         // 索引19
    Threshold(FeatureType::M_MUL_N, 6291456.0f),     // 索引20
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引21
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),  // 索引22
    Threshold(FeatureType::MN_DIV_K, 6144.000000f),  // 索引23
    Threshold(FeatureType::MN_DIV_K, 3072.000000f),  // 索引24
    Ret(6), Ret(6),                                  // 索引25-26
    Threshold(FeatureType::MN_DIV_K, 192.000000f),   // 索引27
    Threshold(FeatureType::M_DIV_N, 0.750000f),      // 索引28
    Threshold(FeatureType::MN_DIV_K, 49152.000000f), // 索引29
    Threshold(FeatureType::M_DIV_N, 24.000000f),     // 索引30
    // <<<<< 层级5 (索引31-62) <<<<<
    Threshold(FeatureType::MN_DIV_K, 48.000000f),               // 索引31
    Threshold(FeatureType::K_VALUE, 384.0f),                    // 索引32
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),               // 索引33
    Threshold(FeatureType::WORLD_MUL_K, 6144.0f),               // 索引34
    Ret(3),                                                     // 索引35
    Ret(6),                                                     // 索引36
    Threshold(FeatureType::K_VALUE, 768.0f),                    // 索引37
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),             // 索引38
    Ret(8),                                                     // 索引39
    Threshold(FeatureType::MN_DIV_K, 12288.000000f),            // 索引40
    Ret(8),                                                     // 索引41
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),             // 索引42
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),               // 索引43
    Ret(6),                                                     // 索引44
    Ret(8),                                                     // 索引45
    Ret(3),                                                     // 索引46
    Threshold(FeatureType::M_DIV_N, 3.000000f),                 // 索引47
    Threshold(FeatureType::K_VALUE, 384.0f),                    // 索引48
    Ret(3),                                                     // 索引49
    Threshold(FeatureType::K_VALUE, 3072.0f),                   // 索引50
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引51-54
    Ret(3),                                                     // 索引55
    Threshold(FeatureType::MN_DIV_K, 768.000000f),              // 索引56
    Threshold(FeatureType::WORLD_MUL_K, 1536.0f),               // 索引57
    Threshold(FeatureType::MN_DIV_K, 49152.000000f),            // 索引58
    Threshold(FeatureType::M_DIV_N, 1.500000f),                 // 索引59
    Threshold(FeatureType::MN_DIV_K, 98304.000000f),            // 索引60
    Threshold(FeatureType::M_DIV_N, 1.500000f),                 // 索引61
    Ret(64),                                                    // 索引62
    // ++++++ 层级6 (索引63-126) ++++++
    Ret(3),                                                                                   // 索引63
    Ret(1),                                                                                   // 索引64
    Ret(3),                                                                                   // 索引65
    Ret(6),                                                                                   // 索引66
    Ret(2), Ret(2),                                                                           // 索引67-68
    Ret(6),                                                                                   // 索引69
    Ret(2),                                                                                   // 索引70
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引71-74
    Ret(4), Ret(4), Ret(4),                                                                   // 索引75-77
    Ret(6),                                                                                   // 索引78
    Placeholder(), Placeholder(),                                                             // 索引79-80
    Ret(8),                                                                                   // 索引81
    Ret(6),                                                                                   // 索引82
    Placeholder(), Placeholder(),                                                             // 索引83-84
    Ret(3),                                                                                   // 索引85
    Ret(8), Ret(8),                                                                           // 索引86-87
    Ret(3),                                                                                   // 索引88
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引89-94
    Ret(16), Ret(16), Ret(16),                                                                // 索引95-97
    Ret(3),                                                                                   // 索引98
    Placeholder(), Placeholder(),                                                             // 索引99-100
    Ret(3),                                                                                   // 索引101
    Ret(16),                                                                                  // 索引102
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引103-108
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引109-112
    Ret(6), Ret(6), Ret(6),                                                                   // 索引113-115
    Ret(3),                                                                                   // 索引116
    Ret(32),                                                                                  // 索引117
    Ret(6),                                                                                   // 索引118
    Ret(3),                                                                                   // 索引119
    Ret(64),                                                                                  // 索引120
    Ret(3),                                                                                   // 索引121
    Ret(6),                                                                                   // 索引122
    Ret(3), Ret(3),                                                                           // 索引123-124
    Placeholder(), Placeholder(),                                                             // 索引125-126
};

// swizzlDirect优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzldirectRule[] = {
    // -*- 层级0 (索引0-0) -*-
    Threshold(FeatureType::M_MUL_N, 25165824.0f), // 索引0
    // ====== 层级1 (索引1-2) ======
    Threshold(FeatureType::M_MUL_N, 786432.0f), // 索引1
    Threshold(FeatureType::N_VALUE, 6144.0f),   // 索引2
    // #### 层级2 (索引3-6) ####
    Threshold(FeatureType::M_DIV_N, 6.000000f), // 索引3
    Threshold(FeatureType::N_VALUE, 384.0f),    // 索引4
    Threshold(FeatureType::K_VALUE, 6144.0f),   // 索引5
    Ret(1),                                     // 索引6
    // *** 层级3 (索引7-14) ***
    Threshold(FeatureType::MN_DIV_K, 48.000000f), // 索引7
    Threshold(FeatureType::K_VALUE, 1536.0f),     // 索引8
    Threshold(FeatureType::M_VALUE, 6144.0f),     // 索引9
    Threshold(FeatureType::N_VALUE, 6144.0f),     // 索引10
    Threshold(FeatureType::WORLD_MUL_K, 1536.0f), // 索引11
    Ret(1),                                       // 索引12
    Placeholder(), Placeholder(),                 // 索引13-14
    // >>> 层级4 (索引15-30) >>>
    Threshold(FeatureType::M_VALUE, 192.0f),                                                  // 索引15
    Threshold(FeatureType::K_VALUE, 3072.0f),                                                 // 索引16
    Ret(0),                                                                                   // 索引17
    Threshold(FeatureType::K_VALUE, 3072.0f),                                                 // 索引18
    Ret(1),                                                                                   // 索引19
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),                                             // 索引20
    Threshold(FeatureType::K_VALUE, 6144.0f), Threshold(FeatureType::K_VALUE, 6144.0f),       // 索引21-22
    Ret(1),                                                                                   // 索引23
    Ret(0),                                                                                   // 索引24
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引25-30
    // -*- -*- -*- 层级5 (索引31-62) -*- -*- -*-
    Ret(1),                                                                                   // 索引31
    Threshold(FeatureType::M_VALUE, 384.0f),                                                  // 索引32
    Threshold(FeatureType::M_DIV_N, 1.500000f),                                               // 索引33
    Threshold(FeatureType::M_DIV_N, 0.093750f),                                               // 索引34
    Placeholder(), Placeholder(),                                                             // 索引35-36
    Ret(1),                                                                                   // 索引37
    Ret(0),                                                                                   // 索引38
    Placeholder(), Placeholder(),                                                             // 索引39-40
    Ret(1),                                                                                   // 索引41
    Threshold(FeatureType::WORLD_MUL_K, 6144.0f),                                             // 索引42
    Threshold(FeatureType::M_DIV_N, 0.750000f),                                               // 索引43
    Threshold(FeatureType::M_VALUE, 6144.0f),                                                 // 索引44
    Threshold(FeatureType::M_MUL_N, 1572864.0f),                                              // 索引45
    Ret(1),                                                                                   // 索引46
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引47-52
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引53-58
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引59-62
    // [层级6] (索引63-126)
    Placeholder(), Placeholder(),                                                             // 索引63-64
    Ret(1), Ret(1), Ret(1), Ret(1),                                                           // 索引65-68
    Ret(0),                                                                                   // 索引69
    Ret(1),                                                                                   // 索引70
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引71-76
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引77-82
    Placeholder(), Placeholder(),                                                             // 索引83-84
    Ret(0),                                                                                   // 索引85
    Ret(1),                                                                                   // 索引86
    Ret(0), Ret(0), Ret(0),                                                                   // 索引87-89
    Ret(1), Ret(1),                                                                           // 索引90-91
    Ret(0),                                                                                   // 索引92
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引93-98
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引99-104
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引105-110
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引111-116
    Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), Placeholder(), // 索引117-122
    Placeholder(), Placeholder(), Placeholder(), Placeholder(),                               // 索引123-126
};

} // namespace Tiling_Small_M::Tiling_Rank2_A2

#endif
