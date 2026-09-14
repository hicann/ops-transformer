/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_DECISION_TREE_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_DECISION_TREE_H

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

} // namespace Tiling_Small_M

#endif
