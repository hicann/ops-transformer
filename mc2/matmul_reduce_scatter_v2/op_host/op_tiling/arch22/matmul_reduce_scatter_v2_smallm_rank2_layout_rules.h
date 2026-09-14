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
    {FeatureType::RETURN_VALUE, {.return_value = 128}}, // 索引0
};

// swizzlCount优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzlcountRule[] = {
    // === 层级0 (索引0-0) ===
    {FeatureType::M_VALUE, {.threshold = 1536.0f}}, // 索引0
    // ----- 层级1 (索引1-2) -----
    {FeatureType::M_VALUE, {.threshold = 768.0f}},  // 索引1
    {FeatureType::M_VALUE, {.threshold = 3072.0f}}, // 索引2
    // ~~~~~ 层级2 (索引3-6) ~~~~~
    {FeatureType::M_VALUE, {.threshold = 384.0f}},         // 索引3
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引4
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引5
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},        // 索引6
    // >>>>> 层级3 (索引7-14) >>>>>
    {FeatureType::M_VALUE, {.threshold = 192.0f}},         // 索引7
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},    // 索引8
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引9
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引10
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}},    // 索引11
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引12
    {FeatureType::N_VALUE, {.threshold = 384.0f}},         // 索引13
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引14
    // :::::: 层级4 (索引15-30) ::::::
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引15
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},      // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 24.000000f}},    // 索引17
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引18
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引19
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},     // 索引20
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引21
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引22
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}},  // 索引23
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引24
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引25
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引26
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引27
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},      // 索引28
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引29
    {FeatureType::M_DIV_N, {.threshold = 24.000000f}},     // 索引30
    // <<<<< 层级5 (索引31-62) <<<<<
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},    // 索引31
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引32
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引33
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},    // 索引34
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引35
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引36
    {FeatureType::K_VALUE, {.threshold = 768.0f}},         // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引38
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引39
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引40
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引41
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引42
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引43
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引44
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引46
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},      // 索引47
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引48
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引49
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},        // 索引50
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引51: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引52: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引53: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引54: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引55
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引56
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},    // 索引57
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引58
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引59
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引60
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引61
    {FeatureType::RETURN_VALUE, {.return_value = 64}},     // 索引62
    // ++++++ 层级6 (索引63-126) ++++++
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引64
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引71: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引72: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引73: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引74: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引76
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引78
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引79: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引80: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引82
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引83: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引84: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引88
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引89: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引90: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引91: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引92: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引93: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引94: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引96
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引97
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引98
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引99: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引100: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引101
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引102
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引103: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引104: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引105: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引106: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引107: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引108: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引109: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引110: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引111: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引112: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引114
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引115
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引116
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引118
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引122
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引123
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引124
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引125: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引126: 占位符
};

// swizzlDirect优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzldirectRule[] = {
    // -*- 层级0 (索引0-0) -*-
    {FeatureType::M_MUL_N, {.threshold = 25165824.0f}}, // 索引0
    // ====== 层级1 (索引1-2) ======
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}}, // 索引1
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},   // 索引2
    // #### 层级2 (索引3-6) ####
    {FeatureType::M_DIV_N, {.threshold = 6.000000f}}, // 索引3
    {FeatureType::N_VALUE, {.threshold = 384.0f}},    // 索引4
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},   // 索引5
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引6
    // *** 层级3 (索引7-14) ***
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}}, // 索引7
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},     // 索引8
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},     // 索引9
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},     // 索引10
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}}, // 索引11
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引12
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引13: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引14: 占位符
    // >>> 层级4 (索引15-30) >>>
    {FeatureType::M_VALUE, {.threshold = 192.0f}},      // 索引15
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},     // 索引16
    {FeatureType::RETURN_VALUE, {.return_value = 0}},   // 索引17
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},     // 索引18
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引19
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}}, // 索引20
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},     // 索引21
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},     // 索引22
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引23
    {FeatureType::RETURN_VALUE, {.return_value = 0}},   // 索引24
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引25: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引26: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引27: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引28: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引29: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引30: 占位符
    // -*- -*- -*- 层级5 (索引31-62) -*- -*- -*-
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引31
    {FeatureType::M_VALUE, {.threshold = 384.0f}},      // 索引32
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},   // 索引33
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},   // 索引34
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引35: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引36: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引37
    {FeatureType::RETURN_VALUE, {.return_value = 0}},   // 索引38
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引39: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引40: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引41
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}}, // 索引42
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},   // 索引43
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},     // 索引44
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},  // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引47: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引48: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引49: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引50: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引51: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引52: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引53: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引54: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引55: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引56: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引57: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引58: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引59: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引60: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引61: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引62: 占位符
    // [层级6] (索引63-126)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引63: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引64: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引71: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引72: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引73: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引74: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引75: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引76: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引77: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引78: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引79: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引80: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引81: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引82: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引83: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引84: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引88
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引90
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引91
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引92
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引93: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引94: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引95: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引96: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引97: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引98: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引99: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引100: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引101: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引102: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引103: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引104: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引105: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引106: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引107: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引108: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引109: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引110: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引111: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引112: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引113: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引114: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引115: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引116: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引117: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引118: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引119: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引120: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引121: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引122: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引123: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引124: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引125: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引126: 占位符
};

} // namespace Tiling_Small_M::Tiling_Rank2_A2

#endif
