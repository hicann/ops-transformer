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
    {FeatureType::M_DIV_N, {.threshold = 12.000000f}}, // 索引0
    // || 层级1 (索引1-2) ||
    {FeatureType::M_VALUE, {.threshold = 6144.0f}}, // 索引1
    {FeatureType::K_VALUE, {.threshold = 6144.0f}}, // 索引2
    // ||| 层级2 (索引3-6) |||
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引4
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},     // 索引5
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引6
    // |||| 层级3 (索引7-14) ||||
    {FeatureType::M_VALUE, {.threshold = 3072.0f}},    // 索引7
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}}, // 索引8
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},    // 索引9
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},    // 索引10
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引11
    {FeatureType::N_VALUE, {.threshold = 384.0f}},     // 索引12
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引13
    {FeatureType::RETURN_VALUE, {.return_value = 7}},  // 索引14
    // ||||| 层级4 (索引15-30) |||||
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},    // 索引15
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},       // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引17
    {FeatureType::M_VALUE, {.threshold = 3072.0f}},       // 索引18
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},     // 索引19
    {FeatureType::RETURN_VALUE, {.return_value = 10}},    // 索引20
    {FeatureType::RETURN_VALUE, {.return_value = 20}},    // 索引21
    {FeatureType::RETURN_VALUE, {.return_value = 20}},    // 索引22
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引23: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引24: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引25
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引26
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引27: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引28: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引29: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引30: 占位
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 1}},      // 索引31
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引32
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引33
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引34
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},      // 索引35
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}},     // 索引36
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引38
    {FeatureType::RETURN_VALUE, {.return_value = 10}},     // 索引39
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引40
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引41: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引42: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引43: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引44: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引45: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引46: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引47: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引48: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引49: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引50: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引51: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引52: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引53: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引54: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引55: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引56: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引57: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引58: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引59: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引60: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引61: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引62: 占位符
    // <层级6> (索引63-126)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引63: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引64: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引65: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引66: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引67: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引68: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引69: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引70: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引74
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引76
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 10}}, // 索引78
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引79: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引80: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引82
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引83: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引84: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引85: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引86: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引87: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引88: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引89: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引90: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引91: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引92: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引93: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引94: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引95: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引96: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引97: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引98: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引99: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引100: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引101: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引102: 占位符
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
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引113: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引114: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引115: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引116: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引117: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引118: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引119: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引120: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引121: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引122: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引123: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引124: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引125: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引126: 占位符
};

// ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // (层级0) [索引0-0]
    {FeatureType::M_MUL_N, {.threshold = 98304.0f}}, // 索引0
    // (层级1) [索引1-2]
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}}, // 索引1
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},     // 索引2
    // (层级2) [索引3-6]
    {FeatureType::RETURN_VALUE, {.return_value = 16}},  // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}}, // 索引4
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}}, // 索引5
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}}, // 索引6
    // (层级3) [索引7-14]
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引7: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引8: 占位符
    {FeatureType::M_VALUE, {.threshold = 192.0f}},       // 索引9
    {FeatureType::RETURN_VALUE, {.return_value = 16}},   // 索引10
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}}, // 索引11
    {FeatureType::WORLD_MUL_K, {.threshold = 768.0f}},   // 索引12
    {FeatureType::M_VALUE, {.threshold = 192.0f}},       // 索引13
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},      // 索引14
    // 【层级4】(索引15-30)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引15: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引16: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引17: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引18: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引19
    {FeatureType::RETURN_VALUE, {.return_value = 16}},   // 索引20
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引21: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引22: 未定义
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},    // 索引23
    {FeatureType::M_MUL_N, {.threshold = 196608.0f}},    // 索引24
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},    // 索引25
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},      // 索引26
    {FeatureType::M_MUL_N, {.threshold = 393216.0f}},    // 索引27
    {FeatureType::M_MUL_N, {.threshold = 196608.0f}},    // 索引28
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}}, // 索引29
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}}, // 索引30
    // 【层级5】(索引31-62)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引31: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引32: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引33: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引34: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引35: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引36: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引37: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引38: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引39: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引40: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引41: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引42: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引43: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引44: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引45: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引46: 未定义
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}}, // 索引47
    {FeatureType::RETURN_VALUE, {.return_value = 8}},   // 索引48
    {FeatureType::M_VALUE, {.threshold = 384.0f}},      // 索引49
    {FeatureType::M_DIV_N, {.threshold = 0.023438f}},   // 索引50
    {FeatureType::M_MUL_N, {.threshold = 25165824.0f}}, // 索引51
    {FeatureType::RETURN_VALUE, {.return_value = 4}},   // 索引52
    {FeatureType::M_MUL_N, {.threshold = 25165824.0f}}, // 索引53
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}}, // 索引54
    {FeatureType::MN_DIV_K, {.threshold = 24.000000f}}, // 索引55
    {FeatureType::RETURN_VALUE, {.return_value = 16}},  // 索引56
    {FeatureType::RETURN_VALUE, {.return_value = 8}},   // 索引57
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}}, // 索引58
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},   // 索引59
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},     // 索引60
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},     // 索引61
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},     // 索引62
    // 【层级6】(索引63-126)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引63: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引64: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引65: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引66: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引67: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引68: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引69: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引70: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引71: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引72: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引73: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引74: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引75: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引76: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引77: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引78: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引79: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引80: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引81: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引82: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引83: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引84: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引85: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引86: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引87: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引88: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引89: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引90: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引91: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引92: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引93: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引94: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引96
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引97: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引98: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引99
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引100
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引101
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引102
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引103
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引104
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引105: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引106: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引107
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引108
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引109
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引110
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引111
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引112
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引113: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引114: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引115: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引116: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引118
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引122
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引123
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引124
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引125
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引126
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
