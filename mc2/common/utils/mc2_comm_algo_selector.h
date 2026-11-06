/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMM_ALGO_SELECTOR_H
#define MC2_COMM_ALGO_SELECTOR_H

#include <cstdint>
#include <string>
#include "mc2_hcom_topo_info.h"

namespace Mc2Hcom {

constexpr uint32_t MAX_COMM_ALGO_CANDIDATES = 32;

// 组网标志字(uint64)分区布局，由Mc2TopoInfo一次性装配的事实按位拼接，表项按位与匹配：
// 机器当前标志包含表项要求的全部标志位才命中。
// 分区：bit0-3 层数/实例形态 | bit8-15 实例类型存在(按CommTopo枚举值) | bit16-23 覆盖类型(按CommTopo枚举值)
//      | bit32-47 链路协议(按CommProtocol枚举值)
constexpr uint64_t TOPO_FLAG_NONE = 0ULL;
constexpr uint64_t TOPO_FLAG_SINGLE_LEVEL = 1ULL << 0;  ///< 单层网络(topoLevelNums==1)
constexpr uint64_t TOPO_FLAG_MULTI_LEVEL = 1ULL << 1;   ///< 多层网络(topoLevelNums>1)
constexpr uint64_t TOPO_FLAG_L0_MULTI_INST = 1ULL << 2; ///< L0实例数>1(全层视角，来自GetInstSizeListByLayer)
constexpr uint64_t TOPO_FLAG_L0_NONCOVER_INST =
    1ULL << 3; ///< L0存在未覆盖实例(存在rank数小于level0RankSize的实例类型，die内子网实例特征)

// 类型/覆盖/协议区的起始位与掩码(区宽度与Mc2TopoInfo位图宽度一致，新枚举值超出宽度时需同步扩区)
constexpr uint32_t TOPO_FLAG_TOPO_TYPE_SHIFT = 8;
constexpr uint32_t TOPO_FLAG_TOPO_COVER_SHIFT = 16;
constexpr uint32_t TOPO_FLAG_PROTO_SHIFT = 32;
constexpr uint64_t TOPO_TYPE_REGION_MASK = (1ULL << MC2_TOPO_TYPE_BITMAP_WIDTH) - 1;
constexpr uint64_t TOPO_COVER_REGION_MASK = (1ULL << MC2_TOPO_TYPE_BITMAP_WIDTH) - 1;
constexpr uint64_t TOPO_PROTO_REGION_MASK = (1ULL << MC2_PROTO_BITMAP_WIDTH) - 1;

// 按枚举值构造标志位：bit(TOPO_TYPE_SHIFT+CommTopo值)=该类型实例存在；
// bit(TOPO_COVER_SHIFT+CommTopo值)=该类型实例覆盖L0；bit(PROTO_SHIFT+CommProtocol值)=L0存在该协议链路
constexpr uint64_t TopoTypeFlag(const uint32_t commTopo)
{
    return 1ULL << (TOPO_FLAG_TOPO_TYPE_SHIFT + commTopo);
}
constexpr uint64_t TopoCoverFlag(const uint32_t commTopo)
{
    return 1ULL << (TOPO_FLAG_TOPO_COVER_SHIFT + commTopo);
}
constexpr uint64_t LinkProtoFlag(const uint32_t commProtocol)
{
    return 1ULL << (TOPO_FLAG_PROTO_SHIFT + commProtocol);
}

// 常用命名常量
constexpr uint64_t TOPO_FLAG_L0_1DMESH = TopoTypeFlag(COMM_TOPO_1DMESH);
constexpr uint64_t TOPO_FLAG_L0_CLOS = TopoTypeFlag(COMM_TOPO_CLOS);
constexpr uint64_t TOPO_FLAG_L0_1DMESH_COVER = TopoCoverFlag(COMM_TOPO_1DMESH);
constexpr uint64_t TOPO_FLAG_L0_CLOS_COVER = TopoCoverFlag(COMM_TOPO_CLOS);
constexpr uint64_t TOPO_FLAG_L0_PCIE_LINK = LinkProtoFlag(COMM_PROTOCOL_PCIE);
// Mesh+CLOS混联形态(MESH_1D_CLOS)组合常量：1DMESH实例与CLOS实例并存、CLOS实例覆盖L0，
// 且存在未覆盖实例(die内子网实例不覆盖L0全部rank)——混联的定义特征；
// fullmesh等全覆盖形态(1DMESH实例亦覆盖L0)不存在未覆盖实例，不满足该组合
constexpr uint64_t TOPO_FLAG_L0_MESH_1D_CLOS =
    TOPO_FLAG_L0_1DMESH | TOPO_FLAG_L0_CLOS | TOPO_FLAG_L0_CLOS_COVER | TOPO_FLAG_L0_NONCOVER_INST;

struct CommAlgoEntry {
    uint8_t commEngine;
    uint64_t topoFlags; ///< 组网标志位要求(TOPO_FLAG_*按位组合，TOPO_FLAG_NONE表示不要求)
    uint64_t minBytes;
    uint64_t maxBytes;
    uint32_t minRankSize;
    uint32_t maxRankSize;
    int32_t priority;
    const char* algoName;
};

// 计算通信数据量(三因子饱和乘法)：任一步溢出钳位到UINT64_MAX并告警，
// 避免回绕成小值误命中算法表的小数据量分档(溢出钳位命中大数据量分支/兜底默认算法)
uint64_t CalcCommDataBytes(const char* opName, uint64_t factorA, uint64_t factorB, uint64_t factorC);

class Mc2CommAlgoSelector {
public:
    // 表项匹配选择通信算法名（纯匹配，不查组网）：topo标志字由调用方先调GetTopoFlags单次查询获取后传入；
    // 组网未知（相应标志未置）时具体表项不命中，由兜底表项承接回退默认算法
    static std::string SelectAlgoName(const std::string& opName, const char* group, uint8_t commEngine,
                                      uint64_t commDataBytes, uint32_t rankSize, const CommAlgoEntry* entries,
                                      uint32_t entryCount, const std::string& defaultAlgoName, uint64_t topoFlags);

    // 一次性装配组网整体信息并拼接组网标志字（单次查询，与SelectAlgoName表匹配同一事实源，供调用方组网判定复用）；
    // 装配失败返回false，此时topoInfo为默认值、topoFlags相应位不置位（表项不命中，由兜底表项承接回退默认算法）
    static bool GetTopoFlags(const char* group, Mc2TopoInfo& topoInfo, uint64_t& topoFlags);

private:
    // 一次性装配组网整体信息(Mc2TopoInfo)，装配失败返回false，调用方按维度未知处理
    static bool GetTopoInfo(const char* group, Mc2TopoInfo& topoInfo);
    // 由装配事实按位拼接组网标志(层数/实例数/实例类型/覆盖类型/链路协议)，装配失败相应位不置位
    static uint64_t CalcTopoFlags(const Mc2TopoInfo& topoInfo);
};

} // namespace Mc2Hcom

#endif // MC2_COMM_ALGO_SELECTOR_H
