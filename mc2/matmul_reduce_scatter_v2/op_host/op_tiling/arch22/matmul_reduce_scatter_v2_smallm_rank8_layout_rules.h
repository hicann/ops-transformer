/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK8_LAYOUT_RULES_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK8_LAYOUT_RULES_H

#include "matmul_reduce_scatter_v2_smallm_decision_tree.h"

namespace Tiling_Small_M::Tiling_Rank8_A2 {

// m0优化参数的决策树规则（使用特征：）
const DecisionNode m0Rule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 128}}, // 索引0
};

// swizzlCount优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzlcountRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_VALUE, {.threshold = 1536.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::M_VALUE, {.threshold = 768.0f}},  // 索引1
    {FeatureType::M_VALUE, {.threshold = 3072.0f}}, // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::M_VALUE, {.threshold = 384.0f}},        // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}}, // 索引4
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},       // 索引5
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},       // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::M_VALUE, {.threshold = 192.0f}},        // 索引7
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}}, // 索引8
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},   // 索引9
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}},    // 索引10
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},       // 索引11
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},       // 索引12
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},    // 索引13
    {FeatureType::WORLD_MUL_K, {.threshold = 49152.0f}},  // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::N_VALUE, {.threshold = 384.0f}},         // 索引15
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引17
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},      // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引19
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},      // 索引20
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},      // 索引21
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引22
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引23
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},      // 索引24
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},        // 索引25
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引26
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引27
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引28
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引29
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}},    // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::K_VALUE, {.threshold = 768.0f}},         // 索引31
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引32
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}},  // 索引33
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引34
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}},    // 索引35
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引36
    {FeatureType::RETURN_VALUE, {.return_value = 4}},      // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引38
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引39
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引40
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引41
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引42
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引43
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引44
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引46
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},        // 索引47
    {FeatureType::M_DIV_N, {.threshold = 6.000000f}},      // 索引48
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引49
    {FeatureType::RETURN_VALUE, {.return_value = 16}},     // 索引50
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引51
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引52
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引53
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引54
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引55
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引56
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引57
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引58
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引59
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引60
    {FeatureType::N_VALUE, {.threshold = 384.0f}},         // 索引61
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引62
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引64
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引70
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引74
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引75: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引76: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引78
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引79: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引80: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引81: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引82: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引83: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引84: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引86
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引87: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引88: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引90
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引91: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引92: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引93: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引94: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引96
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引97
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引98
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引99: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引100: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引101: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引102: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引103: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引104: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引105
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引106
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引107: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引108: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引109: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引110: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引111: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引112: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引114
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引115
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引116
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引117: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引118: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引122
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引123
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引124
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引125: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引126: 占位
};

// swizzlDirect优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzldirectRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::N_VALUE, {.threshold = 384.0f}},    // 索引1
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}}, // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}},  // 索引3
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},      // 索引4
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},      // 索引5
    {FeatureType::WORLD_MUL_K, {.threshold = 49152.0f}}, // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},   // 索引7
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},   // 索引8
    {FeatureType::N_VALUE, {.threshold = 768.0f}},        // 索引9
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}},   // 索引10
    {FeatureType::K_VALUE, {.threshold = 384.0f}},        // 索引11
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引12
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}}, // 索引13
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},    // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::M_MUL_N, {.threshold = 196608.0f}},     // 索引15
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引16
    {FeatureType::M_DIV_N, {.threshold = 24.000000f}},    // 索引17
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}}, // 索引18
    {FeatureType::K_VALUE, {.threshold = 768.0f}},        // 索引19
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},     // 索引20
    {FeatureType::M_VALUE, {.threshold = 384.0f}},        // 索引21
    {FeatureType::M_DIV_N, {.threshold = 0.023438f}},     // 索引22
    {FeatureType::RETURN_VALUE, {.return_value = 0}},     // 索引23
    {FeatureType::RETURN_VALUE, {.return_value = 0}},     // 索引24
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引25: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引26: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 0}},     // 索引27
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},     // 索引28
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引29
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::MN_DIV_K, {.threshold = 24.000000f}},   // 索引31
    {FeatureType::RETURN_VALUE, {.return_value = 0}},     // 索引32
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引33: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引34: 占位
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}}, // 索引35
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引36
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},       // 索引37
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引38
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},       // 索引39
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},   // 索引40
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},     // 索引41
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},   // 索引42
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},       // 索引43
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},   // 索引44
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引45
    {FeatureType::M_VALUE, {.threshold = 768.0f}},        // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引47: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引48: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引49: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引50: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引51: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引52: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引53: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引54: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引55: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引56: 占位
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},     // 索引57
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}},  // 索引58
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引59: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引60: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引61: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引62: 占位
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引64
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引65: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引66: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引67: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引68: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引69: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引70: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引72
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引73: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引74: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引76
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引77: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引78: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引79
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引80
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引88
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引90
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引91: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引92: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引93
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引94
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引95: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引96: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引97: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引98: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引99: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引100: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引101: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引102: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引103: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引104: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引105: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引106: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引107: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引108: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引109: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引110: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引111: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引112: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引113: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引114: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引115
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引116
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引118
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引119: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引120: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引121: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引122: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引123: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引124: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引125: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引126: 占位
};

} // namespace Tiling_Small_M::Tiling_Rank8_A2

#endif
