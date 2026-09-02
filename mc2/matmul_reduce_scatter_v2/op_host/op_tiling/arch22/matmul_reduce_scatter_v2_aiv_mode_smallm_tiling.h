/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter_v2_aiv_mode_smallm_tiling.h
 * \brief
 */

#ifndef DECISION_RULES_H
#define DECISION_RULES_H

#include <cstdint>

#include "register/op_def_registry.h"

namespace Tiling_Small_M {
enum class FeatureType {
    M_VALUE = 0,       // m的数值大小
    K_VALUE = 1,       // k的数值大小
    N_VALUE = 2,       // n的数值大小
    M_DIV_N = 3,       // m/n的比值
    MN_DIV_K = 4,      // (m*n)/k的比值
    WORLD_MUL_K = 5,   // world_size*k的乘积
    M_MUL_N = 6,       // m*n的乘积
    RETURN_VALUE = -1, // 叶子节点返回值标记
    PLACEHOLDER = -2   // 空节点占位符
};

constexpr int FEATURE_COUNT = 7;
constexpr int MAX_TREE_DEPTH = 7;
constexpr int32_t RANKSIZE_TWO = 2;
constexpr int32_t RANKSIZE_FOUR = 4;
constexpr int32_t RANKSIZE_EIGHT = 8;

constexpr int32_t DEFAULT_M0 = 128;
constexpr int32_t DEFAULT_SWIZZLCOUNT = 1;
constexpr int32_t DEFAULT_SWIZZLDIRECT = 0;
constexpr int32_t DEFAULT_PVALUE = 2;
constexpr int32_t DEFAULT_UBMOVENUM = 4;
constexpr int32_t DEFAULT_COMMNPUSPLIT = 1;
constexpr int32_t DEFAULT_COMMDATASPLIT = 16;

union NodeValue {
    float threshold;  // 内部节点的阈值
    int return_value; // 叶子节点的返回值
};

struct DecisionNode {
    FeatureType feature; // 特征类型
    NodeValue value;     // 阈值或返回值（通过union区分）
};

inline void PrecomputeFeatures(float features[FEATURE_COUNT], int m, int k, int n, int world_size)
{
    const int64_t mn = static_cast<int64_t>(m) * n;
    const int64_t world_k = static_cast<int64_t>(world_size) * k;
    features[static_cast<int>(FeatureType::M_VALUE)] = static_cast<float>(m);
    features[static_cast<int>(FeatureType::K_VALUE)] = static_cast<float>(k);
    features[static_cast<int>(FeatureType::N_VALUE)] = static_cast<float>(n);
    features[static_cast<int>(FeatureType::M_DIV_N)] = (n != 0) ? static_cast<float>(m) / n : 0.0f;
    features[static_cast<int>(FeatureType::MN_DIV_K)] = (k != 0) ? static_cast<float>(mn) / k : 0.0f;
    features[static_cast<int>(FeatureType::WORLD_MUL_K)] = static_cast<float>(world_k);
    features[static_cast<int>(FeatureType::M_MUL_N)] = static_cast<float>(mn);
}

template <size_t N>
inline int TraverseDecisionTree(const DecisionNode (&decision_rules)[N], int m, int k, int n, int world_size,
                                int default_value)
{
    // 预计算所有特征值
    float features[FEATURE_COUNT];
    PrecomputeFeatures(features, m, k, n, world_size);

    int node_idx = 0; // 从根节点开始
    constexpr int SUB_TREE_SCALE = 2;
    constexpr int LEFT_NODE_OFFSET = 1;
    constexpr int RIGHT_NODE_OFFSET = 2;

    for (int i = 0; i < MAX_TREE_DEPTH; i++) {
        const DecisionNode &node = decision_rules[node_idx];

        // 走到空占位符，理论不应发生，返回默认值
        if (node.feature == FeatureType::PLACEHOLDER) {
            return default_value;
        }

        // 如果是叶子节点，返回值
        if (node.feature == FeatureType::RETURN_VALUE) {
            return node.value.return_value;
        }

        float feature_value = features[static_cast<int>(node.feature)];

        if (feature_value <= node.value.threshold) {
            node_idx = SUB_TREE_SCALE * node_idx + LEFT_NODE_OFFSET; // 左子节点
        } else {
            node_idx = SUB_TREE_SCALE * node_idx + RIGHT_NODE_OFFSET; // 右子节点
        }
    }
    return default_value;
}

namespace Tiling_Rank2_A2 {
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

} // namespace Tiling_Rank2_A2

namespace Tiling_Rank4_A2 {
// m0优化参数的决策树规则（使用特征：）
const DecisionNode m0Rule[] = {
    // >>>> 层级0 (索引0-0) >>>>
    {FeatureType::RETURN_VALUE, {.return_value = 128}}, // 索引0
};

// swizzlCount优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzlcountRule[] = {
    // >> 层级0 (索引0-0) >>>
    {FeatureType::M_VALUE, {.threshold = 3072.0f}}, // 索引0
    // > 层级1 (索引1-2) >
    {FeatureType::M_VALUE, {.threshold = 768.0f}},  // 索引1
    {FeatureType::M_VALUE, {.threshold = 6144.0f}}, // 索引2
    // << 层级2 (索引3-6) <<
    {FeatureType::M_VALUE, {.threshold = 384.0f}},         // 索引3
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},        // 索引4
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引5
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引6
    // <<< 层级3 (索引7-14) <<<
    {FeatureType::M_VALUE, {.threshold = 192.0f}},         // 索引7
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引8
    {FeatureType::M_MUL_N, {.threshold = 393216.0f}},      // 索引9
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引10
    {FeatureType::N_VALUE, {.threshold = 384.0f}},         // 索引11
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引12
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引13
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引14
    // <<<< 层级4 (索引15-30) <<<<
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引15
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},      // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引17
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引19
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引20
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引21
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引22
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},        // 索引23
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引24
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引25
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引26
    {FeatureType::M_DIV_N, {.threshold = 24.000000f}},     // 索引27
    {FeatureType::N_VALUE, {.threshold = 768.0f}},         // 索引28
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},        // 索引29
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引31
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引32
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}},   // 索引33
    {FeatureType::M_MUL_N, {.threshold = 196608.0f}},      // 索引34
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引35
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},        // 索引36
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引38
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},        // 索引39
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引40
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},      // 索引41
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引42
    {FeatureType::N_VALUE, {.threshold = 768.0f}},         // 索引43
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引44
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},     // 索引45
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引46
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引47
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引48
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},        // 索引49
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引50
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引51: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引52: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引53
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引54
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},     // 索引55
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},        // 索引56
    {FeatureType::RETURN_VALUE, {.return_value = 64}},     // 索引57
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引58
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引59
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引60
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引61
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引62
    // <<<<< 层级6 (索引63-126) <<<<<
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引64
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引71: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引72: 占位符
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引74
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引76
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引78
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引79
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引80
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引81: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引82: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引88
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引90
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引91
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引92
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引93
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引94
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引95: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引96: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引97: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引98: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引99
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引100
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引101
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引102
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引103: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引104: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引105: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引106: 未定义
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引107: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引108: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引109: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引110: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引111
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引112
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引114
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引115: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引116: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引117: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引118: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引122
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引123: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引124: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引125: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引126: 未定义
};

// swizzlDirect优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzldirectRule[] = {
    // ## 层级0 (索引0-0) ##
    {FeatureType::M_MUL_N, {.threshold = 25165824.0f}}, // 索引0
    // ## 层级1 (索引1-2) ##
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}}, // 索引1
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},   // 索引2
    // ## 层级2 (索引3-6) ##
    {FeatureType::N_VALUE, {.threshold = 768.0f}},         // 索引3
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},        // 索引4
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引5
    {FeatureType::RETURN_VALUE, {.return_value = 1}},      // 索引6
    // ## 层级3 (索引7-14) ##
    {FeatureType::M_MUL_N, {.threshold = 49152.0f}},    // 索引7
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},     // 索引8
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}}, // 索引9
    {FeatureType::M_VALUE, {.threshold = 384.0f}},      // 索引10
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引11
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}}, // 索引12
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引13: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // 索引14: 占位符
    // ## 层级4 (索引15-30) ##
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引15
    {FeatureType::M_MUL_N, {.threshold = 98304.0f}},     // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}}, // 索引17
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}}, // 索引18
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},    // 索引19
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},      // 索引20
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},  // 索引21
    {FeatureType::K_VALUE, {.threshold = 768.0f}},       // 索引22
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引23: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引24: 未定义
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引25
    {FeatureType::RETURN_VALUE, {.return_value = 0}},    // 索引26
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引27: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引28: 占位符
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引29: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引30: 未定义
    // ## 层级5 (索引31-62) ##
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引31: placeholder
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引32: 占位符
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},  // 索引33
    {FeatureType::K_VALUE, {.threshold = 768.0f}},       // 索引34
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},  // 索引35
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},  // 索引36
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},      // 索引37
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引38
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},   // 索引39
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},      // 索引40
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}}, // 索引41
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},    // 索引42
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},   // 索引43
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}}, // 索引44
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}},  // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引47: 空节点
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引48: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引49: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引50: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引51: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引52: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引53: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引54: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引55: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引56: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引57: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引58: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引59: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引60: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引61: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引62: 占位
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引63: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引64: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引65: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引66: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引70
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引74
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引76
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引77: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引78: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引79
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引80
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引88
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引90
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引91
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引92
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引93: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引94: 占位
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
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引115: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引116: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引117: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引118: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引119: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引120: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引121: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引122: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引123: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引124: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引125: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // 索引126: 占位
};

// pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode pvalueRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_VALUE, {.threshold = 3072.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::M_VALUE, {.threshold = 768.0f}},  // 索引1
    {FeatureType::M_VALUE, {.threshold = 6144.0f}}, // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引4
    {FeatureType::K_VALUE, {.threshold = 768.0f}},         // 索引5
    {FeatureType::M_DIV_N, {.threshold = 12.000000f}},     // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引7
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},      // 索引8
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},     // 索引9
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},        // 索引10
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},        // 索引11
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}},     // 索引12
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引13
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},    // 索引15
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},      // 索引16
    {FeatureType::M_DIV_N, {.threshold = 0.023438f}},      // 索引17
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引19
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},        // 索引20
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引21
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},     // 索引22
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引23
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引24
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引25
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},      // 索引26
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},    // 索引27
    {FeatureType::RETURN_VALUE, {.return_value = 20}},     // 索引28
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引29
    {FeatureType::RETURN_VALUE, {.return_value = 7}},      // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引31
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引32
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引33
    {FeatureType::K_VALUE, {.threshold = 768.0f}},        // 索引34
    {FeatureType::RETURN_VALUE, {.return_value = 2}},     // 索引35
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},    // 索引36
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引37: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引38: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引39
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},     // 索引40
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引41
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},   // 索引42
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引43: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引44: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 5}},     // 索引46
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},    // 索引47
    {FeatureType::RETURN_VALUE, {.return_value = 7}},     // 索引48
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引49
    {FeatureType::RETURN_VALUE, {.return_value = 10}},    // 索引50
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引51: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引52: 占位
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引53
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},       // 索引54
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引55
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},     // 索引56
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引57: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引58: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引59
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引60
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引61: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引62: 占位
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引63: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引64: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引65: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引66: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引67: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引68: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引71: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引72: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引74
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引75: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引76: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引77: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引78: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引79: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引80: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引86
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引87: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引88: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引89: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引90: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引91: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引92: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引93: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引94: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引96
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引97: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引98: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引99: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引100: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引101: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引102: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引103: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引104: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引105: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引106: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引107
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引108
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引109
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引110
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引111: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引112: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 20}}, // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 10}}, // 索引114
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引115: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引116: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引117: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引118: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引119: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引120: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引121: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引122: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引123: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引124: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引125: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引126: 占位
};

// ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_VALUE, {.threshold = 384.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::N_VALUE, {.threshold = 768.0f}},    // 索引1
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}}, // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}}, // 索引3
    {FeatureType::M_DIV_N, {.threshold = 0.046875f}},    // 索引4
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},  // 索引5
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},      // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::K_VALUE, {.threshold = 384.0f}},       // 索引7
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}}, // 索引8
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}}, // 索引9
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},      // 索引10
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}}, // 索引11
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},    // 索引12
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},   // 索引13
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},  // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引15
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},     // 索引16
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引17
    {FeatureType::RETURN_VALUE, {.return_value = 16}},    // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}},   // 索引19
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引20
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},       // 索引21
    {FeatureType::M_VALUE, {.threshold = 192.0f}},        // 索引22
    {FeatureType::RETURN_VALUE, {.return_value = 16}},    // 索引23
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引24
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},   // 索引25
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引26
    {FeatureType::K_VALUE, {.threshold = 384.0f}},        // 索引27
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}},  // 索引28
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},    // 索引29
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},  // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引31: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引32: 占位
    {FeatureType::MN_DIV_K, {.threshold = 12.000000f}},  // 索引33
    {FeatureType::MN_DIV_K, {.threshold = 12.000000f}},  // 索引34
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引35: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引36: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引37: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引38: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引39
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}}, // 索引40
    {FeatureType::M_DIV_N, {.threshold = 0.023438f}},    // 索引41
    {FeatureType::RETURN_VALUE, {.return_value = 16}},   // 索引42
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引43
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},    // 索引44
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引47: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引48: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引49: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引50: 占位
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},    // 索引51
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引52
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引53: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // 索引54: 占位
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},      // 索引55
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}}, // 索引56
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},    // 索引57
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},      // 索引58
    {FeatureType::K_VALUE, {.threshold = 384.0f}},       // 索引59
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},    // 索引60
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},      // 索引61
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},   // 索引62
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引63: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引64: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引65: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引66: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引71: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引72: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引73: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引74: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引75: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引76: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引77: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引78: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引79: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引80: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引84
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引85: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引86: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引87: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引88: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引90
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引91: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引92: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引93: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引94: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引95: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引96: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引97: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引98: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引99: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引100: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引101: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引102: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引103
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引104
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引105: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引106: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引107: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引108: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引109: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引110: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引111
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引112
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引114
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引115
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引116
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引118
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引122
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引123
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引124
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引125
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引126
};

inline int GetOptimalM0(int m, int k, int n)
{
    return TraverseDecisionTree(m0Rule, m, k, n, RANKSIZE_FOUR, DEFAULT_M0);
}

inline int GetOptimalSwizzlCount(int m, int k, int n)
{
    return TraverseDecisionTree(swizzlcountRule, m, k, n, RANKSIZE_FOUR, DEFAULT_SWIZZLCOUNT);
}

inline int GetOptimalSwizzlDirect(int m, int k, int n)
{
    return TraverseDecisionTree(swizzldirectRule, m, k, n, RANKSIZE_FOUR, DEFAULT_SWIZZLDIRECT);
}

inline int GetOptimalPValue(int m, int k, int n)
{
    return TraverseDecisionTree(pvalueRule, m, k, n, RANKSIZE_FOUR, DEFAULT_PVALUE);
}

inline int GetOptimalUbmovenum(int m, int k, int n)
{
    return TraverseDecisionTree(ubmovenumRule, m, k, n, RANKSIZE_FOUR, DEFAULT_UBMOVENUM);
}

} // namespace Tiling_Rank4_A2

namespace Tiling_Rank8_A2 {
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

// pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode pvalueRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}}, // 索引1
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},    // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},   // 索引3
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},     // 索引4
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},     // 索引5
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}}, // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引7
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},      // 索引8
    {FeatureType::M_VALUE, {.threshold = 768.0f}},       // 索引9
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},    // 索引10
    {FeatureType::RETURN_VALUE, {.return_value = 2}},    // 索引11
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}},  // 索引12
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}}, // 索引13
    {FeatureType::WORLD_MUL_K, {.threshold = 49152.0f}}, // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引15: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引16: 占位
    {FeatureType::M_VALUE, {.threshold = 768.0f}},        // 索引17
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},   // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},  // 索引19
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},       // 索引20
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},     // 索引21
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},     // 索引22
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引23: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引24: 占位
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引25
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},     // 索引26
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引27
    {FeatureType::RETURN_VALUE, {.return_value = 10}},    // 索引28
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},     // 索引29
    {FeatureType::RETURN_VALUE, {.return_value = 20}},    // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引31: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引32: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引33: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引34: 占位
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引35
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},        // 索引36
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引37
    {FeatureType::RETURN_VALUE, {.return_value = 1}},      // 索引38
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引39
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},        // 索引40
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引41
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引42
    {FeatureType::RETURN_VALUE, {.return_value = 1}},      // 索引43
    {FeatureType::RETURN_VALUE, {.return_value = 4}},      // 索引44
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引45
    {FeatureType::M_DIV_N, {.threshold = 6.000000f}},      // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引47: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引48: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引49: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引50: 占位
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引51
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引52
    {FeatureType::RETURN_VALUE, {.return_value = 5}},      // 索引53
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},        // 索引54
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引55: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引56: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引57: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引58: 占位
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引59
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引60
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引61: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引62: 占位
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引63: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引64: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引65: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引66: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引67: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引68: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引69: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引70: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引74
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引75: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引76: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引77: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引78: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引79: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引80: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 7}},  // 索引86
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引87: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引88: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引89: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引90: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引91
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引92
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引93
    {FeatureType::RETURN_VALUE, {.return_value = 7}},  // 索引94
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引95: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引96: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引97: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引98: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引99: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引100: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引101: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引102: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引103
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引104
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引105
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引106
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引107: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引108: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 10}}, // 索引109
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引110
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引111: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引112: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引113: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引114: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引115: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引116: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引117: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引118: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 20}}, // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 20}}, // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 10}}, // 索引122
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引123: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引124: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引125: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引126: 占位
};

// ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_VALUE, {.threshold = 768.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}}, // 索引1
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},  // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},      // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引4
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},     // 索引5
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},  // 索引7
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},    // 索引8
    {FeatureType::RETURN_VALUE, {.return_value = 16}},    // 索引9
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引10
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}}, // 索引11
    {FeatureType::WORLD_MUL_K, {.threshold = 49152.0f}},  // 索引12
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},       // 索引13
    {FeatureType::M_MUL_N, {.threshold = 393216.0f}},     // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::M_MUL_N, {.threshold = 98304.0f}},      // 索引15
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},       // 索引16
    {FeatureType::M_VALUE, {.threshold = 192.0f}},        // 索引17
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}}, // 索引18
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引19: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引20: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引21: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // 索引22: 占位
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},       // 索引23
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},     // 索引24
    {FeatureType::M_VALUE, {.threshold = 3072.0f}},       // 索引25
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},     // 索引26
    {FeatureType::RETURN_VALUE, {.return_value = 16}},    // 索引27
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},       // 索引28
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引29
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::M_VALUE, {.threshold = 192.0f}},         // 索引31
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},        // 索引32
    {FeatureType::N_VALUE, {.threshold = 768.0f}},         // 索引33
    {FeatureType::RETURN_VALUE, {.return_value = 16}},     // 索引34
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引35
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引36
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}},  // 索引38
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引39: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引40: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引41: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引42: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引43: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引44: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引45: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引46: 占位
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引47
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引48
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引49
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引50
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},        // 索引51
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引52
    {FeatureType::M_VALUE, {.threshold = 3072.0f}},        // 索引53
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引54
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引55: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引56: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引57
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},        // 索引58
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引59: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引60: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引61: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // 索引62: 占位
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引64
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引68
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引69: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引70: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引74
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引76
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引78
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引79: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引80: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引81: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引82: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引83: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引84: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引85: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引86: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引87: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引88: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引89: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引90: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引91: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引92: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引93: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引94: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引96
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引97
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引98
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引99
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引100
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引101
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引102
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引103
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引104
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引105: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引106: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引107
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引108
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引109: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引110: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引111: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引112: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引113: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引114: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引115: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引116: 占位
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引118
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引119: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引120: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引121: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引122: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引123: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引124: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引125: 占位
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // 索引126: 占位
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

} // namespace Tiling_Rank8_A2

}; // namespace Tiling_Small_M

namespace optiling {
ge::graphStatus MatmulReduceScatterTilingV2AivModeFunc(gert::TilingContext *context);
}

#endif // DECISION_RULES_H
